"""Search a one-rule Defender selector with the formal Momentum and gate factor."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from defender.relative_defender_champion import target_schedule as champion_schedule
from defender.relative_defender_rotation import (
    DEFENSIVE_ASSET,
    ROTATION_ASSETS,
    ROTATION_COST_RATES,
    load_rotation_market,
)
from factors.quality_momentum import METADATA as QUALITY_METADATA
from research.defender_curve_momentum import DEFENDER_CANDIDATE
from research.momentum_defender_common_score_trimmed import (
    ExtremeBlockSpec,
    build_extreme_block_mask,
)
from research.momentum_defender_downside_raqm import (
    DownsideRAQMSpec,
    ExactExecutionData,
    build_exact_execution_data,
    run_downside_raqm_spec,
)
from research.momentum_defender_gold_override_overfit import (
    cscv_pbo,
    expanding_walk_forward,
    full_metrics,
    leave_one_year_selection,
    paired_block_bootstrap,
    yearly_reality_check,
)
from research.momentum_defender_occam import HELD_RETURN, performance
from research.momentum_defender_occam_defender import (
    MonthlySelectionSpec,
    build_portfolio_switch_interface,
    monthly_top1_selection,
    score_at_open,
    selected_asset_targets,
)
from research.momentum_volatility import load_ohlc
from research.run_momentum_defender_log_qm_robust import (
    _event_stress,
    _friction,
    _selected_cost_schedule,
    _unique_paths,
)
from research.standard_report import generate_standard_report
from strategy.momentum_defender_downside_raqm import (
    formal_profile,
    run_formal_strategy,
)


DEFAULT_CONFIG = Path(
    "research/configs/momentum_defender_occam_defender_search.yaml"
)
DEFAULT_OUTPUT = Path(
    "experiments/20260825_momentum_defender_occam_defender_search"
)


def _load(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Occam Defender search config must be a mapping")
    return value


def _gate_specs(config: dict) -> list[DownsideRAQMSpec]:
    grid = config["state_grid"]
    gap = float(grid["minimum_hysteresis_gap"])
    result = {}
    for values in product(
        grid["entry_percentiles"],
        grid["recovery_percentiles"],
        grid["entry_confirmation_days"],
        grid["recovery_confirmation_days"],
        grid["momentum_lock_days"],
        grid["defender_lock_days"],
    ):
        entry, recovery, entry_c, recovery_c, momentum_lock, defender_lock = values
        if float(entry) - float(recovery) + 1e-12 < gap:
            continue
        spec = DownsideRAQMSpec(
            profile=formal_profile(),
            history_mode="rolling_504_strict_lag",
            entry_percentile=float(entry),
            exit_percentile=float(recovery),
            momentum_lock_days=int(momentum_lock),
            defender_lock_days=int(defender_lock),
            entry_confirmation_days=int(entry_c),
            recovery_confirmation_days=int(recovery_c),
        )
        result[spec.candidate_id()] = spec
    return list(result.values())


def _occam_targets(
    market: dict[str, pd.DataFrame],
    calendar: pd.DatetimeIndex,
    window: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    spec = MonthlySelectionSpec(window, "return", "lowest")
    scores = score_at_open(market, ROTATION_ASSETS, calendar, spec)
    selection = monthly_top1_selection(
        market, ROTATION_ASSETS, calendar, scores, spec
    )
    core = champion_schedule(market["512890.SH"])
    primary = core["primary_target"].reindex(calendar).ffill().astype(float)
    if primary.isna().any():
        raise ValueError("frozen Defender core does not cover the search calendar")
    targets = selected_asset_targets(
        selection["selected_asset"],
        ROTATION_ASSETS,
        selected_weight=1.0,
        residual_asset=DEFENSIVE_ASSET,
    )
    targets.loc[:, :] = 0.0
    for timestamp in calendar:
        asset = str(selection.at[timestamp, "selected_asset"])
        weight = float(primary.loc[timestamp])
        targets.at[timestamp, asset] = weight
        targets.at[timestamp, DEFENSIVE_ASSET] = 1.0 - weight
    return targets, selection


def _record(window: int, run) -> dict[str, object]:
    spec = run.spec
    return {
        "candidate_id": f"defw{window}__{spec.candidate_id()}",
        "defender_window": int(window),
        "entry_percentile": spec.entry_percentile,
        "recovery_percentile": spec.exit_percentile,
        "entry_confirmation_days": spec.entry_confirmation_days,
        "recovery_confirmation_days": spec.recovery_confirmation_days,
        "momentum_lock_days": spec.momentum_lock_days,
        "defender_lock_days": spec.defender_lock_days,
        "defender_entries": run.defender_entries,
        "defender_days": run.defender_days,
        "sleeve_switches": run.sleeve_switches,
        "candidate_switches": run.candidate_switches,
    }


def _add_metrics(
    metadata: pd.DataFrame,
    returns: pd.DataFrame,
    baseline: pd.Series,
    ordinary: pd.Series,
    config: dict,
) -> pd.DataFrame:
    result = metadata.join(full_metrics(returns, baseline).add_prefix("full_"))
    result = result.join(
        full_metrics(returns.loc[ordinary], baseline.loc[ordinary]).add_prefix(
            "ordinary_"
        )
    )
    for period in ("development", "validation", "recent"):
        start, end = map(pd.Timestamp, config["periods"][period])
        mask = returns.index.to_series().between(start, end).to_numpy()
        metrics = full_metrics(returns.loc[mask], baseline.loc[mask])
        ordinary_metrics = full_metrics(
            returns.loc[mask & ordinary.to_numpy(bool)],
            baseline.loc[mask & ordinary.to_numpy(bool)],
        )
        for field in ("annualized_return_252", "sharpe", "max_drawdown"):
            result[f"{period}_{field}"] = metrics[field]
        for field in ("annualized_return_252", "sharpe"):
            result[f"ordinary_{period}_{field}"] = ordinary_metrics[field]
    result["full_minimum_segment_sharpe"] = result[
        ["development_sharpe", "validation_sharpe", "recent_sharpe"]
    ].min(axis=1)
    result["ordinary_minimum_segment_sharpe"] = result[
        [
            "ordinary_development_sharpe",
            "ordinary_validation_sharpe",
            "ordinary_recent_sharpe",
        ]
    ].min(axis=1)
    return result


def _add_neighborhood(table: pd.DataFrame, config: dict) -> pd.DataFrame:
    grid = config["state_grid"]
    dimensions = {
        "defender_window": list(map(int, config["defender_selector"]["windows"])),
        "entry_percentile": list(map(float, grid["entry_percentiles"])),
        "recovery_percentile": list(map(float, grid["recovery_percentiles"])),
        "entry_confirmation_days": list(map(int, grid["entry_confirmation_days"])),
        "recovery_confirmation_days": list(
            map(int, grid["recovery_confirmation_days"])
        ),
        "momentum_lock_days": list(map(int, grid["momentum_lock_days"])),
        "defender_lock_days": list(map(int, grid["defender_lock_days"])),
    }
    result = table.copy()
    coordinate_columns = []
    for field, values in dimensions.items():
        column = f"_{field}"
        result[column] = result[field].map(
            {value: position for position, value in enumerate(values)}
        )
        coordinate_columns.append(column)
    coords = result[coordinate_columns].to_numpy(int)
    metric_arrays = {
        "full_annualized": result["full_annualized_return_252"].to_numpy(float),
        "full_sharpe": result["full_sharpe"].to_numpy(float),
        "ordinary_annualized": result[
            "ordinary_annualized_return_252"
        ].to_numpy(float),
        "ordinary_sharpe": result["ordinary_sharpe"].to_numpy(float),
    }
    rows = {}
    for position, candidate_id in enumerate(result.index):
        members = np.all(np.abs(coords - coords[position]) <= 1, axis=1)
        rows[str(candidate_id)] = {
            "neighborhood_count": int(members.sum()),
            **{
                f"neighborhood_{name}_q25": float(np.quantile(values[members], 0.25))
                for name, values in metric_arrays.items()
            },
            **{
                f"neighborhood_{name}_median": float(np.median(values[members]))
                for name, values in metric_arrays.items()
            },
        }
    return result.drop(columns=coordinate_columns).join(
        pd.DataFrame.from_dict(rows, orient="index")
    )


def _complexity(row: pd.Series) -> float:
    extra_confirmation = (
        int(row["entry_confirmation_days"])
        + int(row["recovery_confirmation_days"])
        - 2
    )
    lock_asymmetry = abs(
        int(row["momentum_lock_days"]) - int(row["defender_lock_days"])
    ) / 5
    return float(extra_confirmation + lock_asymmetry)


def _select(table: pd.DataFrame, config: dict, key: str, prefix: str):
    rule = config[key]
    eligibility = config["eligibility"]
    eligible = (
        table["defender_entries"].ge(int(eligibility["minimum_defender_entries"]))
        & table["defender_days"].ge(int(eligibility["minimum_defender_days"]))
        & table["full_max_drawdown"].ge(
            float(eligibility["maximum_full_drawdown"])
        )
        & table["full_minimum_segment_sharpe"].ge(
            float(eligibility["minimum_full_segment_sharpe"])
        )
        & table["ordinary_minimum_segment_sharpe"].ge(
            float(eligibility["minimum_ordinary_segment_sharpe"])
        )
    )
    pool = table.loc[eligible].copy()
    if pool.empty:
        raise RuntimeError(f"no eligible candidates for {prefix} selection")
    fields = list(rule["ranking_fields"])
    ranks = pool[fields].rank(pct=True)
    pool["robust_rank_min"] = ranks.min(axis=1)
    pool["robust_rank_mean"] = ranks.mean(axis=1)
    best_min = float(pool["robust_rank_min"].max())
    stable = pool.loc[pool["robust_rank_min"].ge(best_min - 0.03)].copy()
    best_mean = float(stable["robust_rank_mean"].max())
    stable = stable.loc[stable["robust_rank_mean"].ge(best_mean - 0.03)]
    annual_field, sharpe_field = fields[:2]
    near = stable.loc[
        stable[annual_field].ge(
            float(stable[annual_field].max())
            - float(rule["occam_annualized_tolerance"])
        )
        & stable[sharpe_field].ge(
            float(stable[sharpe_field].max())
            - float(rule["occam_sharpe_tolerance"])
        )
    ].copy()
    near["occam_complexity"] = near.apply(_complexity, axis=1)
    selected = near.sort_values(
        ["occam_complexity", annual_field, sharpe_field, "robust_rank_mean"],
        ascending=[True, False, False, False],
    ).iloc[0]
    result = table.copy()
    result[f"eligible_{prefix}"] = eligible
    result.loc[pool.index, f"{prefix}_robust_rank_min"] = pool["robust_rank_min"]
    result.loc[pool.index, f"{prefix}_robust_rank_mean"] = pool[
        "robust_rank_mean"
    ]
    result[f"{prefix}_occam_near"] = result.index.isin(near.index)
    return selected, result


def _pareto(table: pd.DataFrame, prefix: str) -> pd.DataFrame:
    fields = [
        f"{prefix}_annualized_return_252",
        f"{prefix}_sharpe",
        f"{prefix}_max_drawdown",
    ]
    values = table[fields].to_numpy(float)
    keep = np.ones(len(table), dtype=bool)
    for position in range(len(table)):
        dominated = np.all(values >= values[position], axis=1) & np.any(
            values > values[position], axis=1
        )
        dominated[position] = False
        if dominated.any():
            keep[position] = False
    return table.loc[keep].sort_values(fields[:2], ascending=False)


def _selected_detail(
    label: str,
    selected: pd.Series,
    returns: pd.DataFrame,
    runs: dict[str, tuple[ExactExecutionData, object, object]],
    baseline_daily: pd.DataFrame,
    ordinary: pd.Series,
    context,
    output: Path,
    config: dict,
) -> dict:
    candidate_id = str(selected.name)
    data, run, selection = runs[candidate_id]
    candidate_returns = returns[candidate_id].astype(float)
    baseline = baseline_daily["return"].astype(float)
    actual = pd.Series(
        [data.candidates[value] for value in run.actual_target], index=data.calendar
    )
    daily = run.state.copy()
    daily["defender_window"] = int(selected["defender_window"])
    daily["defender_selected_asset"] = selection["selected_asset"].astype(str)
    daily["return"] = candidate_returns
    daily["nav"] = (1.0 + candidate_returns).cumprod()
    daily["requested_candidate"] = [
        data.candidates[value] for value in run.requested_target
    ]
    daily["actual_candidate"] = actual
    daily.to_csv(output / f"selected_{label}_daily.csv")
    daily.to_parquet(output / f"selected_{label}_daily.parquet")

    events, leave_events, deleted, event_summary = _event_stress(
        candidate_returns,
        baseline,
        actual,
        baseline_daily["candidate"].astype(str),
        list(map(int, config["overfit_checks"]["top_positive_event_deletions"])),
    )
    events.to_csv(output / f"selected_{label}_events.csv", index=False)
    leave_events.to_csv(output / f"selected_{label}_leave_event.csv", index=False)
    deleted.to_csv(output / f"selected_{label}_delete_top_events.csv", index=False)
    bootstrap, bootstrap_summary = paired_block_bootstrap(
        candidate_returns,
        baseline,
        block_size=int(config["overfit_checks"]["paired_block_bootstrap_block"]),
        repetitions=int(
            config["overfit_checks"]["paired_block_bootstrap_repetitions"]
        ),
        seed=int(config["overfit_checks"]["random_seed"]),
    )
    bootstrap.to_csv(output / f"selected_{label}_bootstrap.csv", index=False)
    candidate_context = replace(
        context,
        interfaces={
            **context.interfaces,
            DEFENDER_CANDIDATE: data_interface(data, context, runs, candidate_id),
        },
    )
    costs = _selected_cost_schedule(candidate_context, data, run.actual_target)
    friction = _friction(
        candidate_returns,
        costs,
        list(map(float, config["overfit_checks"]["friction_cost_multipliers"])),
    )
    friction.to_csv(output / f"selected_{label}_friction.csv", index=False)
    generate_standard_report(
        candidate_returns,
        baseline,
        "Current Weighted DRAQM Formal",
        output / f"selected_{label}_vs_formal.html",
        {"experiment": config["experiment"], "candidate_id": candidate_id},
    )
    result = {
        "candidate_id": candidate_id,
        "parameters": {
            field: selected[field]
            for field in (
                "defender_window",
                "entry_percentile",
                "recovery_percentile",
                "entry_confirmation_days",
                "recovery_confirmation_days",
                "momentum_lock_days",
                "defender_lock_days",
            )
        },
        "full": performance(candidate_returns),
        "ordinary": performance(candidate_returns.loc[ordinary]),
        "baseline": performance(baseline),
        "bootstrap_vs_formal": bootstrap_summary,
        "event_stress_vs_formal": event_summary,
        "three_x_cost": friction.loc[
            friction["cost_multiplier"].eq(3.0)
        ].iloc[0].to_dict(),
        "daily_return_sha256_float64_le": hashlib.sha256(
            candidate_returns.to_numpy(dtype="<f8").tobytes()
        ).hexdigest(),
    }
    plain_result = json.loads(
        json.dumps(
            result,
            default=lambda value: (
                value.item() if isinstance(value, np.generic) else str(value)
            ),
        )
    )
    (output / f"selected_{label}_config.yaml").write_text(
        yaml.safe_dump(plain_result, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return plain_result


def data_interface(data, context, runs, candidate_id):
    """Return the alternative interface stored alongside a selected run."""
    del data, context
    return runs[candidate_id][2].attrs["switch_interface"]


def run_experiment(root: Path, config_path: Path, output: Path) -> dict:
    config = _load(config_path)
    if QUALITY_METADATA["version"] != config["frozen_layers"][
        "momentum_factor_version"
    ]:
        raise AssertionError("Momentum factor version mismatch")
    output.mkdir(parents=True, exist_ok=True)
    end = pd.Timestamp(config["periods"]["full"][1]).date()
    formal = run_formal_strategy(root, end=end)
    context = formal.context
    calendar = context.calendar
    market = load_rotation_market(end=end)
    specs = _gate_specs(config)
    rows: list[dict[str, object]] = []
    columns: dict[str, np.ndarray] = {}
    runs: dict[str, tuple[ExactExecutionData, object, pd.DataFrame]] = {}

    for window in map(int, config["defender_selector"]["windows"]):
        targets, selection = _occam_targets(market, calendar, window)
        interface = build_portfolio_switch_interface(
            market, targets, ROTATION_COST_RATES
        )
        selection.attrs["switch_interface"] = interface
        candidate_context = replace(
            context,
            interfaces={**context.interfaces, DEFENDER_CANDIDATE: interface},
        )
        data = build_exact_execution_data(candidate_context)
        for spec in specs:
            run = run_downside_raqm_spec(data, formal.features, spec)
            record = _record(window, run)
            candidate_id = str(record["candidate_id"])
            rows.append(record)
            columns[candidate_id] = run.returns
            runs[candidate_id] = (data, run, selection)
        print(f"Occam Defender window {window}: {len(specs)} gates", flush=True)

    returns = pd.DataFrame(columns, index=calendar, dtype=np.float32)
    metadata = pd.DataFrame(rows).set_index("candidate_id")
    baseline_daily = formal.daily[["return", "candidate"]].copy()
    baseline = baseline_daily["return"].astype(float)
    trim = config["extreme_block_trim"]
    extreme = build_extreme_block_mask(
        {
            asset: load_ohlc(asset, end)["close"]
            for asset in trim["shock_assets"]
        },
        calendar,
        ExtremeBlockSpec(
            shock_return_window=int(trim["shock_return_window"]),
            block_length_sessions=int(trim["block_length_sessions"]),
            excluded_block_fraction=float(trim["excluded_block_fraction"]),
            normalization_mode=str(trim["normalization_mode"]),
        ),
    )
    extreme.blocks.to_csv(output / "shock_blocks.csv")
    ordinary = extreme.selection_mask.astype(bool)
    table = _add_neighborhood(
        _add_metrics(metadata, returns, baseline, ordinary, config), config
    )
    selected_full, table = _select(
        table, config, "selection_full_sample", "full"
    )
    selected_ordinary, table = _select(
        table, config, "selection_excluding_extremes", "ordinary"
    )
    table.to_csv(output / "candidate_grid.csv")
    _pareto(table, "full").to_csv(output / "pareto_full.csv")
    _pareto(table, "ordinary").to_csv(output / "pareto_ordinary.csv")
    unique = _unique_paths(returns)
    unique.to_parquet(output / "unique_candidate_returns.parquet")
    cscv_full, cscv_full_summary = cscv_pbo(unique, baseline, block_count=12)
    cscv_ordinary, cscv_ordinary_summary = cscv_pbo(
        unique.loc[ordinary], baseline.loc[ordinary], block_count=12
    )
    cscv_full.to_csv(output / "cscv_full.csv", index=False)
    cscv_ordinary.to_csv(output / "cscv_ordinary.csv", index=False)
    reality = {
        "full": yearly_reality_check(
            unique,
            baseline,
            repetitions=int(
                config["overfit_checks"]["yearly_reality_check_repetitions"]
            ),
            seed=int(config["overfit_checks"]["random_seed"]),
        ),
        "ordinary": yearly_reality_check(
            unique.loc[ordinary],
            baseline.loc[ordinary],
            repetitions=int(
                config["overfit_checks"]["yearly_reality_check_repetitions"]
            ),
            seed=int(config["overfit_checks"]["random_seed"]),
        ),
    }
    walk = expanding_walk_forward(unique, baseline)
    leave = leave_one_year_selection(unique, baseline)
    walk.to_csv(output / "walk_forward.csv", index=False)
    leave.to_csv(output / "leave_one_year.csv", index=False)
    details = {
        "full_sample": _selected_detail(
            "full_sample",
            selected_full,
            returns,
            runs,
            baseline_daily,
            ordinary,
            context,
            output,
            config,
        ),
        "excluding_extremes": _selected_detail(
            "excluding_extremes",
            selected_ordinary,
            returns,
            runs,
            baseline_daily,
            ordinary,
            context,
            output,
            config,
        ),
    }
    audit = {
        "experiment": config["experiment"],
        "calendar": {
            "start": calendar.min().date().isoformat(),
            "end": calendar.max().date().isoformat(),
            "observations": len(calendar),
            "ordinary_observations": int(ordinary.sum()),
        },
        "search": {
            "candidate_ids": len(table),
            "unique_paths": int(unique.shape[1]),
            "selected_full_sample": str(selected_full.name),
            "selected_excluding_extremes": str(selected_ordinary.name),
        },
        "formal_baseline": performance(baseline),
        "selected": details,
        "cscv": {"full": cscv_full_summary, "ordinary": cscv_ordinary_summary},
        "reality_check": reality,
        "walk_forward": {
            "return_win_rate": float(walk["test_return_delta"].gt(0.0).mean()),
            "sharpe_win_rate": float(walk["test_sharpe_delta"].gt(0.0).mean()),
        },
        "leave_one_year": {
            "return_win_rate": float(leave["test_return_delta"].gt(0.0).mean()),
            "sharpe_win_rate": float(leave["test_sharpe_delta"].gt(0.0).mean()),
        },
        "decision": config["decision"],
    }
    (output / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    audit = run_experiment(root, args.config.resolve(), args.output.resolve())
    print(json.dumps(audit["search"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
