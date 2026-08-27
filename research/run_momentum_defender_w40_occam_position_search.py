"""Search simple dividend/bond position rules under the frozen formal W40 gate."""

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

from defender.relative_defender_rotation import (
    BASE_PRIMARY_ASSET,
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
from research.momentum_defender_downside_raqm import build_exact_execution_data
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
)
from research.momentum_defender_occam_position import (
    DRAWDOWN_SCALE,
    FIXED_WEIGHT,
    FROZEN_CHAMPION,
    RANGE_LOCATION,
    RELATIVE_VOLATILITY_CAP,
    TREND_BINARY,
    VOLATILITY_TARGET,
    PositionSpec,
    build_position_targets,
)
from research.momentum_defender_w40_loss_gate import run_w40_loss_gate
from research.momentum_volatility import load_ohlc
from research.run_momentum_defender_log_qm_robust import (
    _event_stress,
    _friction,
    _selected_cost_schedule,
    _unique_paths,
)
from research.standard_report import generate_standard_report
from strategy.momentum_defender_w40_loss import (
    FORMAL_STRATEGY_ID,
    formal_spec,
    run_formal_strategy,
)


DEFAULT_CONFIG = Path(
    "research/configs/momentum_defender_w40_occam_position_search.yaml"
)
DEFAULT_OUTPUT = Path(
    "experiments/20260825_momentum_defender_w40_occam_position_search"
)


def _load(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("W40 Occam position config must be a mapping")
    return value


def _specs(config: dict) -> list[PositionSpec]:
    families = config["position_families"]
    result = [PositionSpec(FROZEN_CHAMPION)]
    result.extend(
        PositionSpec(FIXED_WEIGHT, level=float(weight))
        for weight in families["fixed_weight"]["weights"]
    )
    for family in (TREND_BINARY, RANGE_LOCATION):
        values = families[family]
        result.extend(
            PositionSpec(family, str(source), int(window))
            for source, window in product(
                values["signal_sources"], values["windows"]
            )
        )
    values = families[VOLATILITY_TARGET]
    result.extend(
        PositionSpec(
            VOLATILITY_TARGET, str(source), int(window), float(target)
        )
        for source, window, target in product(
            values["signal_sources"],
            values["windows"],
            values["annual_targets"],
        )
    )
    values = families[RELATIVE_VOLATILITY_CAP]
    result.extend(
        PositionSpec(
            RELATIVE_VOLATILITY_CAP,
            str(source),
            int(window),
            float(quantile),
        )
        for source, window, quantile in product(
            values["signal_sources"], values["windows"], values["quantiles"]
        )
    )
    values = families[DRAWDOWN_SCALE]
    result.extend(
        PositionSpec(DRAWDOWN_SCALE, str(source), int(window), float(scale))
        for source, window, scale in product(
            values["signal_sources"],
            values["windows"],
            values["drawdown_scales"],
        )
    )
    unique = {spec.candidate_id: spec for spec in result}
    return list(unique.values())


def _selection(
    market: dict[str, pd.DataFrame], calendar: pd.DatetimeIndex
) -> pd.DataFrame:
    spec = MonthlySelectionSpec(40, "return", "lowest")
    return monthly_top1_selection(
        market,
        ROTATION_ASSETS,
        calendar,
        score_at_open(market, ROTATION_ASSETS, calendar, spec),
        spec,
    )


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
        ordinary_mask = mask & ordinary.to_numpy(bool)
        metrics = full_metrics(returns.loc[mask], baseline.loc[mask])
        ordinary_metrics = full_metrics(
            returns.loc[ordinary_mask], baseline.loc[ordinary_mask]
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


def _add_neighborhood(table: pd.DataFrame) -> pd.DataFrame:
    result = table.copy()
    summaries = {}
    for (family, source), group in result.groupby(
        ["family", "signal_source"], dropna=False, sort=False
    ):
        coordinate_fields = []
        coords = []
        for field in ("window", "level", "secondary_level"):
            finite = sorted(group[field].dropna().unique())
            if len(finite) > 1:
                coordinate_fields.append(field)
                lookup = {value: position for position, value in enumerate(finite)}
                coords.append(group[field].map(lookup).to_numpy(int))
        if coords:
            coordinate = np.column_stack(coords)
        else:
            coordinate = np.zeros((len(group), 1), dtype=int)
        arrays = {
            "full_annualized": group["full_annualized_return_252"].to_numpy(float),
            "full_sharpe": group["full_sharpe"].to_numpy(float),
            "ordinary_annualized": group[
                "ordinary_annualized_return_252"
            ].to_numpy(float),
            "ordinary_sharpe": group["ordinary_sharpe"].to_numpy(float),
        }
        for position, candidate_id in enumerate(group.index):
            members = np.all(np.abs(coordinate - coordinate[position]) <= 1, axis=1)
            summaries[str(candidate_id)] = {
                "neighborhood_count": int(members.sum()),
                **{
                    f"neighborhood_{name}_q25": float(
                        np.quantile(values[members], 0.25)
                    )
                    for name, values in arrays.items()
                },
                **{
                    f"neighborhood_{name}_median": float(
                        np.median(values[members])
                    )
                    for name, values in arrays.items()
                },
            }
    return result.join(pd.DataFrame.from_dict(summaries, orient="index"))


def _eligibility(table: pd.DataFrame, config: dict) -> pd.Series:
    values = config["eligibility"]
    return (
        table["full_max_drawdown"].ge(float(values["maximum_full_drawdown"]))
        & table["full_minimum_segment_sharpe"].ge(
            float(values["minimum_full_segment_sharpe"])
        )
        & table["ordinary_minimum_segment_sharpe"].ge(
            float(values["minimum_ordinary_segment_sharpe"])
        )
    )


def _performance_leader(table: pd.DataFrame, eligible: pd.Series) -> pd.Series:
    fields = [
        "full_annualized_return_252",
        "full_sharpe",
        "ordinary_annualized_return_252",
        "ordinary_sharpe",
    ]
    pool = table.loc[eligible].copy()
    ranks = pool[fields].rank(pct=True)
    pool["joint_rank_min"] = ranks.min(axis=1)
    pool["joint_rank_mean"] = ranks.mean(axis=1)
    return pool.sort_values(
        ["joint_rank_min", "joint_rank_mean"], ascending=False
    ).iloc[0]


def _occam_selection(
    table: pd.DataFrame,
    eligible: pd.Series,
    leader: pd.Series,
    config: dict,
) -> tuple[pd.Series, pd.Series]:
    rule = config["joint_occam_selection"]
    near = eligible.copy()
    for field, tolerance in (
        ("full_annualized_return_252", rule["full_annualized_tolerance"]),
        ("full_sharpe", rule["full_sharpe_tolerance"]),
        ("ordinary_annualized_return_252", rule["ordinary_annualized_tolerance"]),
        ("ordinary_sharpe", rule["ordinary_sharpe_tolerance"]),
    ):
        near &= table[field].ge(float(leader[field]) - float(tolerance))
    pool = table.loc[near].copy()
    if pool.empty:
        raise RuntimeError("joint Occam near-performance set is empty")
    pool["robust_sharpe_floor"] = pool[
        [
            "neighborhood_full_sharpe_q25",
            "neighborhood_ordinary_sharpe_q25",
            "full_minimum_segment_sharpe",
            "ordinary_minimum_segment_sharpe",
        ]
    ].min(axis=1)
    selected = pool.sort_values(
        [
            "fitted_parameter_count",
            "robust_sharpe_floor",
            "full_annualized_return_252",
            "full_sharpe",
        ],
        ascending=[True, False, False, False],
    ).iloc[0]
    return selected, near


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
        keep[position] = not dominated.any()
    return table.loc[keep].sort_values(fields[:2], ascending=False)


def _selected_detail(
    label: str,
    row: pd.Series,
    artifacts: dict[str, dict[str, object]],
    returns: pd.DataFrame,
    baseline_daily: pd.DataFrame,
    ordinary: pd.Series,
    formal_context,
    frozen_champion_returns: pd.Series,
    output: Path,
    config: dict,
) -> dict:
    candidate_id = str(row.name)
    artifact = artifacts[candidate_id]
    data = artifact["data"]
    run = artifact["run"]
    interface = artifact["interface"]
    diagnostics = artifact["diagnostics"]
    candidate_context = replace(
        formal_context,
        interfaces={**formal_context.interfaces, DEFENDER_CANDIDATE: interface},
    )
    candidate_returns = returns[candidate_id].astype(float)
    baseline = baseline_daily["return"].astype(float)
    actual = pd.Series(
        [data.candidates[value] for value in run.actual_target], index=data.calendar
    )
    daily = run.state.copy()
    daily = daily.join(diagnostics)
    daily["return"] = candidate_returns
    daily["nav"] = (1.0 + candidate_returns).cumprod()
    daily["requested_candidate"] = [
        data.candidates[value] for value in run.requested_target
    ]
    daily["actual_candidate"] = actual
    daily.to_csv(output / f"selected_{label}_daily.csv")
    daily.to_parquet(output / f"selected_{label}_daily.parquet")
    risk_on = run.state["risk_on"].astype(bool)
    candidate_event_target = pd.Series(
        np.where(risk_on, "MOMENTUM", "OCCAM_DEFENDER"), index=data.calendar
    )
    baseline_event_target = pd.Series(
        np.where(risk_on, "MOMENTUM", "FORMAL_DEFENDER"), index=data.calendar
    )
    events, leave_events, deleted, event_summary = _event_stress(
        candidate_returns,
        baseline,
        candidate_event_target,
        baseline_event_target,
        list(map(int, config["overfit_checks"]["top_positive_event_deletions"])),
    )
    events.to_csv(output / f"selected_{label}_events_vs_formal.csv", index=False)
    leave_events.to_csv(output / f"selected_{label}_leave_event.csv", index=False)
    deleted.to_csv(output / f"selected_{label}_delete_top_events.csv", index=False)
    bootstrap_formal, bootstrap_formal_summary = paired_block_bootstrap(
        candidate_returns,
        baseline,
        block_size=int(config["overfit_checks"]["paired_block_bootstrap_block"]),
        repetitions=int(
            config["overfit_checks"]["paired_block_bootstrap_repetitions"]
        ),
        seed=int(config["overfit_checks"]["random_seed"]),
    )
    bootstrap_champion, bootstrap_champion_summary = paired_block_bootstrap(
        candidate_returns,
        frozen_champion_returns,
        block_size=int(config["overfit_checks"]["paired_block_bootstrap_block"]),
        repetitions=int(
            config["overfit_checks"]["paired_block_bootstrap_repetitions"]
        ),
        seed=int(config["overfit_checks"]["random_seed"]),
    )
    bootstrap_formal.to_csv(
        output / f"selected_{label}_bootstrap_vs_formal.csv", index=False
    )
    bootstrap_champion.to_csv(
        output / f"selected_{label}_bootstrap_vs_frozen_champion.csv", index=False
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
        "Current Formal W40",
        output / f"selected_{label}_vs_formal.html",
        {"experiment": config["experiment"], "candidate_id": candidate_id},
    )
    result = {
        "candidate_id": candidate_id,
        "position_spec": {
            field: row[field]
            for field in (
                "family",
                "signal_source",
                "window",
                "level",
                "secondary_level",
                "fitted_parameter_count",
            )
        },
        "full": performance(candidate_returns),
        "ordinary": performance(candidate_returns.loc[ordinary]),
        "formal_baseline": performance(baseline),
        "same_selector_frozen_champion": performance(frozen_champion_returns),
        "bootstrap_vs_formal": bootstrap_formal_summary,
        "bootstrap_vs_same_selector_frozen_champion": bootstrap_champion_summary,
        "events_vs_formal": event_summary,
        "three_x_cost": friction.loc[
            friction["cost_multiplier"].eq(3.0)
        ].iloc[0].to_dict(),
        "daily_return_sha256_float64_le": hashlib.sha256(
            candidate_returns.to_numpy(dtype="<f8").tobytes()
        ).hexdigest(),
    }
    plain = json.loads(
        json.dumps(
            result,
            default=lambda value: (
                value.item() if isinstance(value, np.generic) else str(value)
            ),
        )
    )
    (output / f"selected_{label}_config.yaml").write_text(
        yaml.safe_dump(plain, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    return plain


def run_experiment(
    root: Path,
    config_path: Path,
    output: Path,
    *,
    specs_override: list[PositionSpec] | None = None,
) -> dict:
    config = _load(config_path)
    if QUALITY_METADATA["version"] != config["frozen_layers"][
        "momentum_factor_version"
    ]:
        raise AssertionError("Momentum factor version mismatch")
    if config["frozen_layers"]["top_gate_strategy_id"] != FORMAL_STRATEGY_ID:
        raise AssertionError("search is not pinned to the current formal W40 gate")
    output.mkdir(parents=True, exist_ok=True)
    end = pd.Timestamp(config["periods"]["full"][1]).date()
    formal = run_formal_strategy(root, end=end)
    context = formal.context
    calendar = context.calendar
    market = load_rotation_market(end=end)
    selection = _selection(market, calendar)
    specs = _specs(config) if specs_override is None else specs_override
    columns = {}
    records = []
    artifacts: dict[str, dict[str, object]] = {}
    for position, spec in enumerate(specs, start=1):
        targets, diagnostics = build_position_targets(
            market,
            ROTATION_ASSETS,
            DEFENSIVE_ASSET,
            selection["selected_asset"].astype(str),
            calendar,
            BASE_PRIMARY_ASSET,
            spec,
        )
        interface = build_portfolio_switch_interface(
            market, targets, ROTATION_COST_RATES
        )
        candidate_context = replace(
            context,
            interfaces={**context.interfaces, DEFENDER_CANDIDATE: interface},
        )
        data = build_exact_execution_data(candidate_context)
        run = run_w40_loss_gate(data, formal.score_at_open, formal_spec())
        if not run.state.equals(formal.state):
            raise AssertionError("a position candidate changed the frozen W40 state")
        candidate_id = spec.candidate_id
        columns[candidate_id] = run.returns
        defender_metrics = performance(interface[HELD_RETURN].astype(float))
        records.append(
            {
                "candidate_id": candidate_id,
                "family": spec.family,
                "signal_source": spec.signal_source,
                "window": spec.window,
                "level": spec.level,
                "secondary_level": spec.secondary_level,
                "fitted_parameter_count": spec.fitted_parameter_count,
                "equity_weight_mean": float(diagnostics["equity_weight"].mean()),
                "equity_weight_std": float(diagnostics["equity_weight"].std(ddof=1)),
                "equity_weight_min": float(diagnostics["equity_weight"].min()),
                "equity_weight_max": float(diagnostics["equity_weight"].max()),
                "equity_weight_zero_days": int(
                    diagnostics["equity_weight"].le(1e-14).sum()
                ),
                "equity_weight_full_days": int(
                    diagnostics["equity_weight"].ge(1.0 - 1e-14).sum()
                ),
                "internal_rebalances": int(interface["internal_rebalanced"].sum()),
                "defender_annualized_return_252": defender_metrics[
                    "annualized_return_252"
                ],
                "defender_sharpe": defender_metrics["sharpe"],
                "defender_max_drawdown": defender_metrics["max_drawdown"],
            }
        )
        artifacts[candidate_id] = {
            "data": data,
            "run": run,
            "interface": interface,
            "diagnostics": diagnostics,
        }
        print(f"W40 position: {position}/{len(specs)} {candidate_id}", flush=True)

    returns = pd.DataFrame(columns, index=calendar, dtype=np.float64)
    metadata = pd.DataFrame(records).set_index("candidate_id")
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
        _add_metrics(metadata, returns, baseline, ordinary, config)
    )
    eligible = _eligibility(table, config)
    leader = _performance_leader(table, eligible)
    occam, near = _occam_selection(table, eligible, leader, config)
    table["eligible"] = eligible
    table["joint_occam_near"] = near
    table["selected_performance_leader"] = table.index == str(leader.name)
    table["selected_joint_occam"] = table.index == str(occam.name)
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
    frozen_returns = returns[FROZEN_CHAMPION].astype(float)
    selected_rows = {
        "performance_leader": leader,
        "joint_occam": occam,
    }
    details = {}
    completed_ids = set()
    for label, row in selected_rows.items():
        if str(row.name) in completed_ids:
            details[label] = details[next(iter(details))]
            continue
        details[label] = _selected_detail(
            label,
            row,
            artifacts,
            returns,
            baseline_daily,
            ordinary,
            context,
            frozen_returns,
            output,
            config,
        )
        completed_ids.add(str(row.name))
    audit = {
        "experiment": config["experiment"],
        "frozen_gate": {
            "strategy_id": FORMAL_STRATEGY_ID,
            "candidate_state_equals_formal": True,
            "performance": performance(baseline),
        },
        "calendar": {
            "start": calendar.min().date().isoformat(),
            "end": calendar.max().date().isoformat(),
            "observations": len(calendar),
            "ordinary_observations": int(ordinary.sum()),
        },
        "search": {
            "candidate_ids": len(specs),
            "unique_paths": int(unique.shape[1]),
            "performance_leader": str(leader.name),
            "joint_occam": str(occam.name),
            "joint_occam_near_count": int(near.sum()),
        },
        "same_selector_frozen_champion": performance(frozen_returns),
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
