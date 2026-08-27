"""Search X/Y for the formal W40 Top1 quality-momentum escape overlay."""

from __future__ import annotations

import argparse
import hashlib
import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from factors.quality_momentum import METADATA as QUALITY_METADATA
from research.momentum_defender_common_score_trimmed import (
    ExtremeBlockSpec,
    build_extreme_block_mask,
)
from research.momentum_defender_downside_raqm import build_exact_execution_data
from research.momentum_defender_gold_override_overfit import (
    cscv_pbo,
    expanding_walk_forward,
    leave_one_year_selection,
    paired_block_bootstrap,
    yearly_reality_check,
)
from research.momentum_defender_occam import performance
from research.momentum_defender_w40_top1_escape import (
    DEFENDER_ELIGIBILITY_DAYS,
    QUALITY_WINDOW,
    TOP1_HARD_HOLD_DAYS,
    W40Top1EscapeSpec,
    quality_metrics_at_open,
    run_w40_top1_escape,
)
from research.momentum_volatility import load_ohlc
from research.run_momentum_defender_common_score_trimmed import _add_metrics
from research.run_momentum_defender_log_qm_robust import (
    _event_stress,
    _friction,
    _selected_cost_schedule,
    _unique_paths,
)
from research.standard_report import generate_standard_report
from strategy.momentum_defender_w40_full_equity import (
    FORMAL_STRATEGY_ID,
    run_formal_strategy,
)


DEFAULT_CONFIG = Path(
    "research/configs/momentum_defender_w40_top1_escape_search.yaml"
)
DEFAULT_OUTPUT = Path(
    "experiments/20260825_momentum_defender_w40_top1_escape_search"
)
PRIOR_PATHS = Path(
    "experiments/20260825_momentum_defender_w40_occam_position_focused/"
    "global_unique_candidate_returns.parquet"
)


def _load(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("W40 Top1 escape config must be a mapping")
    return value


def _specs(config: dict) -> list[W40Top1EscapeSpec]:
    grid = config["threshold_grid"]
    result = {}
    for x, y in product(grid["entry_x"], grid["exit_y"]):
        if float(y) > float(x):
            continue
        spec = W40Top1EscapeSpec(float(x), float(y))
        result[spec.candidate_id] = spec
    return list(result.values())


def _add_neighborhood(table: pd.DataFrame, config: dict) -> pd.DataFrame:
    x_values = list(map(float, config["threshold_grid"]["entry_x"]))
    y_values = list(map(float, config["threshold_grid"]["exit_y"]))
    x_lookup = {value: position for position, value in enumerate(x_values)}
    y_lookup = {value: position for position, value in enumerate(y_values)}
    result = table.copy()
    coords = np.column_stack(
        [
            result["entry_x"].map(x_lookup).to_numpy(int),
            result["exit_y"].map(y_lookup).to_numpy(int),
        ]
    )
    arrays = {
        "full_annualized": result["full_annualized_return_252"].to_numpy(float),
        "full_sharpe": result["full_sharpe"].to_numpy(float),
        "ordinary_annualized": result[
            "ordinary_annualized_return_252"
        ].to_numpy(float),
        "ordinary_sharpe": result["ordinary_sharpe"].to_numpy(float),
        "full_delta_annualized": result[
            "full_delta_annualized_return_252"
        ].to_numpy(float),
        "full_delta_sharpe": result["full_delta_sharpe"].to_numpy(float),
    }
    rows = {}
    for position, candidate_id in enumerate(result.index):
        members = np.all(np.abs(coords - coords[position]) <= 1, axis=1)
        rows[str(candidate_id)] = {
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
            "neighborhood_full_dual_pass_rate": float(
                np.mean(
                    (arrays["full_delta_annualized"][members] > 0.0)
                    & (arrays["full_delta_sharpe"][members] > 0.0)
                )
            ),
        }
    return result.join(pd.DataFrame.from_dict(rows, orient="index"))


def _eligible(table: pd.DataFrame, config: dict) -> pd.Series:
    values = config["eligibility"]
    return (
        table["escape_entries"].ge(int(values["minimum_escape_entries"]))
        & table["lock_break_entries"].ge(
            int(values["minimum_lock_break_entries"])
        )
        & table["escape_days"].ge(int(values["minimum_escape_days"]))
        & table["full_max_drawdown"].ge(float(values["maximum_full_drawdown"]))
        & table["full_minimum_segment_sharpe"].ge(
            float(values["minimum_full_segment_sharpe"])
        )
        & table["ordinary_minimum_segment_sharpe"].ge(
            float(values["minimum_ordinary_segment_sharpe"])
        )
    )


def _select(
    table: pd.DataFrame,
    pool_mask: pd.Series,
    fields: list[str],
) -> pd.Series:
    pool = table.loc[pool_mask].copy()
    if pool.empty:
        raise RuntimeError("escape selection pool is empty")
    ranks = pool[fields].rank(pct=True)
    pool["robust_rank_min"] = ranks.min(axis=1)
    pool["robust_rank_mean"] = ranks.mean(axis=1)
    return pool.sort_values(
        ["robust_rank_min", "robust_rank_mean", fields[0], fields[1]],
        ascending=False,
    ).iloc[0]


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


def _plain(value):
    return json.loads(
        json.dumps(
            value,
            default=lambda item: (
                item.item() if isinstance(item, np.generic) else str(item)
            ),
        )
    )


def run_experiment(root: Path, config_path: Path, output: Path) -> dict:
    config = _load(config_path)
    frozen = config["frozen_layers"]
    if frozen["formal_strategy_id"] != FORMAL_STRATEGY_ID:
        raise AssertionError("escape search is not pinned to current formal strategy")
    if QUALITY_METADATA["version"] != frozen["momentum_factor_version"]:
        raise AssertionError("Momentum factor version mismatch")
    if not (
        int(frozen["metric_window"]) == QUALITY_WINDOW
        and int(frozen["defender_eligibility_days"]) == DEFENDER_ELIGIBILITY_DAYS
        and int(frozen["top1_hard_hold_days"]) == TOP1_HARD_HOLD_DAYS
    ):
        raise AssertionError("fixed escape mechanism mismatch")
    output.mkdir(parents=True, exist_ok=True)
    cutoff = pd.Timestamp(config["periods"]["full"][1]).date()
    formal = run_formal_strategy(root, end=cutoff)
    context = formal.context
    baseline = formal.daily["return"].astype(float)
    metrics_at_open = quality_metrics_at_open(context)
    specs = _specs(config)
    columns = {}
    records = []
    runs = {}
    for position, spec in enumerate(specs, start=1):
        run = run_w40_top1_escape(
            context, formal.state, spec, metrics=metrics_at_open
        )
        candidate_id = spec.candidate_id
        columns[candidate_id] = run.daily["return"].to_numpy(float)
        records.append(
            {
                "candidate_id": candidate_id,
                "entry_x": spec.entry_difference,
                "exit_y": spec.exit_difference,
                "hysteresis_gap": spec.entry_difference - spec.exit_difference,
                "escape_entries": run.audit["escape_entries"],
                "escape_returns_to_defender": run.audit[
                    "escape_returns_to_defender"
                ],
                "escape_days": run.audit["escape_days"],
                "lock_break_entries": run.audit["lock_break_entries"],
                "escape_normal_rotations": run.audit["escape_normal_rotations"],
                **{
                    f"escape_days_{asset}": run.audit["escape_asset_days"][asset]
                    for asset in run.audit["escape_asset_days"]
                },
            }
        )
        runs[candidate_id] = run
        if position % 25 == 0 or position == len(specs):
            print(f"W40 Top1 escape: {position}/{len(specs)}", flush=True)
    returns = pd.DataFrame(columns, index=context.calendar, dtype=np.float64)
    metadata = pd.DataFrame(records).set_index("candidate_id")

    disabled = run_w40_top1_escape(
        context,
        formal.state,
        W40Top1EscapeSpec(999.0, -999.0),
        metrics=metrics_at_open,
    )
    disabled_parity = float(
        (disabled.daily["return"].astype(float) - baseline).abs().max()
    )
    if disabled_parity > 1e-14:
        raise AssertionError("disabled escape does not reproduce formal baseline")

    trim = config["extreme_block_trim"]
    extreme = build_extreme_block_mask(
        {
            asset: load_ohlc(asset, cutoff)["close"]
            for asset in trim["shock_assets"]
        },
        context.calendar,
        ExtremeBlockSpec(
            shock_return_window=int(trim["shock_return_window"]),
            block_length_sessions=int(trim["block_length_sessions"]),
            excluded_block_fraction=float(trim["excluded_block_fraction"]),
            normalization_mode=str(trim["normalization_mode"]),
        ),
    )
    ordinary = extreme.selection_mask.astype(bool)
    extreme.blocks.to_csv(output / "shock_blocks.csv")
    evaluated = _add_metrics(metadata, returns, baseline, ordinary, config)
    evaluated = evaluated.rename(
        columns={
            column: column.replace("trimmed_", "ordinary_", 1)
            for column in evaluated.columns
            if column.startswith("trimmed_")
        }
    )
    evaluated["full_minimum_segment_sharpe"] = evaluated[
        ["development_sharpe", "validation_sharpe", "recent_sharpe"]
    ].min(axis=1)
    table = _add_neighborhood(evaluated, config)
    eligible = _eligible(table, config)
    strict_dual = (
        eligible
        & table["full_delta_annualized_return_252"].gt(0.0)
        & table["full_delta_sharpe"].gt(0.0)
    )
    selection = config["selection"]
    selection_pool = strict_dual if strict_dual.any() else eligible
    selected_full = _select(
        table, selection_pool, list(selection["ranking_fields_full"])
    )
    selected_ordinary = _select(
        table, selection_pool, list(selection["ranking_fields_ordinary"])
    )
    selected_joint = _select(
        table, selection_pool, list(selection["ranking_fields_joint"])
    )
    table["eligible"] = eligible
    table["strict_full_dual_improvement"] = strict_dual
    table["selected_full"] = table.index == str(selected_full.name)
    table["selected_ordinary"] = table.index == str(selected_ordinary.name)
    table["selected_joint"] = table.index == str(selected_joint.name)
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
    family_reality = {
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
    prior = pd.read_parquet(root / PRIOR_PATHS)
    prior.columns = [f"position::{column}" for column in prior]
    current = unique.copy()
    current.columns = [f"escape::{column}" for column in current]
    global_paths = _unique_paths(pd.concat([prior, current], axis=1))
    global_paths.to_parquet(output / "global_unique_candidate_returns.parquet")
    global_reality = {
        "full": yearly_reality_check(
            global_paths,
            baseline,
            repetitions=int(
                config["overfit_checks"]["yearly_reality_check_repetitions"]
            ),
            seed=int(config["overfit_checks"]["random_seed"]),
        ),
        "ordinary": yearly_reality_check(
            global_paths.loc[ordinary],
            baseline.loc[ordinary],
            repetitions=int(
                config["overfit_checks"]["yearly_reality_check_repetitions"]
            ),
            seed=int(config["overfit_checks"]["random_seed"]),
        ),
    }
    global_cscv, global_cscv_summary = cscv_pbo(
        global_paths, baseline, block_count=12
    )
    global_cscv.to_csv(output / "global_cscv_full.csv", index=False)
    walk = expanding_walk_forward(unique, baseline)
    leave = leave_one_year_selection(unique, baseline)
    walk.to_csv(output / "walk_forward.csv", index=False)
    leave.to_csv(output / "leave_one_year.csv", index=False)

    selected_id = str(selected_joint.name)
    selected_run = runs[selected_id]
    selected_returns = selected_run.daily["return"].astype(float)
    bootstrap, bootstrap_summary = paired_block_bootstrap(
        selected_returns,
        baseline,
        block_size=int(config["overfit_checks"]["paired_block_bootstrap_block"]),
        repetitions=int(
            config["overfit_checks"]["paired_block_bootstrap_repetitions"]
        ),
        seed=int(config["overfit_checks"]["random_seed"]),
    )
    bootstrap.to_csv(output / "selected_bootstrap.csv", index=False)
    events, leave_events, deleted, event_summary = _event_stress(
        selected_returns,
        baseline,
        selected_run.state["target_candidate"].astype(str),
        formal.daily["candidate"].astype(str),
        list(map(int, config["overfit_checks"]["top_positive_event_deletions"])),
    )
    events.to_csv(output / "selected_events.csv", index=False)
    leave_events.to_csv(output / "selected_leave_event.csv", index=False)
    deleted.to_csv(output / "selected_delete_top_events.csv", index=False)
    data = build_exact_execution_data(context)
    actual_target = selected_run.daily["candidate"].map(
        data.candidate_index
    ).to_numpy(int)
    costs = _selected_cost_schedule(context, data, actual_target)
    friction = _friction(
        selected_returns,
        costs,
        list(map(float, config["overfit_checks"]["friction_cost_multipliers"])),
    )
    friction.to_csv(output / "selected_friction.csv", index=False)
    selected_run.state.join(
        selected_run.daily, rsuffix="_execution"
    ).to_csv(output / "selected_daily.csv")
    selected_run.state.join(
        selected_run.daily, rsuffix="_execution"
    ).to_parquet(output / "selected_daily.parquet")
    selected_run.state.loc[
        selected_run.state["escape_entry"].astype(bool)
    ].to_csv(output / "selected_entries.csv")
    generate_standard_report(
        selected_returns,
        baseline,
        "Current Formal W40 Full Equity",
        output / "selected_vs_formal.html",
        {"experiment": config["experiment"], "selected_id": selected_id},
    )

    selected_config = {
        "strategy_id": "momentum_defender_w40_top1_qm20_escape_candidate_v1",
        "status": (
            "research_candidate_not_production"
            if strict_dual.any()
            else "rejected_research_candidate"
        ),
        "base_strategy_id": FORMAL_STRATEGY_ID,
        "entry_x": float(selected_joint["entry_x"]),
        "exit_y": float(selected_joint["exit_y"]),
        "defender_eligibility_days": DEFENDER_ELIGIBILITY_DAYS,
        "top1_hard_hold_days": TOP1_HARD_HOLD_DAYS,
        "metric": "quality_momentum_log_log",
        "metric_window": QUALITY_WINDOW,
        "checkpoint": {
            **performance(selected_returns),
            "ordinary": performance(selected_returns.loc[ordinary]),
            "escape_entries": selected_run.audit["escape_entries"],
            "lock_break_entries": selected_run.audit["lock_break_entries"],
            "escape_days": selected_run.audit["escape_days"],
            "daily_return_sha256_float64_le": hashlib.sha256(
                selected_returns.to_numpy(dtype="<f8").tobytes()
            ).hexdigest(),
        },
        "bootstrap_vs_formal": bootstrap_summary,
        "events_vs_formal": event_summary,
        "three_x_cost": friction.loc[
            friction["cost_multiplier"].eq(3.0)
        ].iloc[0].to_dict(),
        "evidence_status": config["experiment"]["evidence_status"],
        "strict_full_dual_improvement_passed": bool(strict_dual.any()),
        "production_replacement": False,
    }
    (output / "selected_config.yaml").write_text(
        yaml.safe_dump(_plain(selected_config), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    audit = {
        "experiment": config["experiment"],
        "calendar": {
            "start": context.calendar.min().date().isoformat(),
            "end": context.calendar.max().date().isoformat(),
            "observations": len(context.calendar),
            "ordinary_observations": int(ordinary.sum()),
        },
        "baseline": performance(baseline),
        "disabled_parity_max_abs_error": disabled_parity,
        "search": {
            "candidate_ids": len(specs),
            "unique_paths": int(unique.shape[1]),
            "eligible": int(eligible.sum()),
            "strict_full_dual_improvement": int(strict_dual.sum()),
            "selected_full": str(selected_full.name),
            "selected_ordinary": str(selected_ordinary.name),
            "selected_joint": selected_id,
        },
        "selected": _plain(selected_joint.to_dict()),
        "selected_checkpoint": _plain(selected_config),
        "family_cscv": {
            "full": cscv_full_summary,
            "ordinary": cscv_ordinary_summary,
        },
        "family_reality_check": family_reality,
        "global_paths": {
            "input_prior": int(prior.shape[1]),
            "input_escape": int(unique.shape[1]),
            "unique": int(global_paths.shape[1]),
            "cscv_full": global_cscv_summary,
            "reality_check": global_reality,
        },
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
