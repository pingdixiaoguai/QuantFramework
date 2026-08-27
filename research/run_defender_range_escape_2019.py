"""Run the 2019-start Occam audit of Defender high-range partial escape."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from datetime import date
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from research.audit_current_strategy_occam_robustness import (
    _metric_row,
    _periods,
)
from research.audit_defender_selector_2019 import (
    _difference_events,
    _leave_one_event,
)
from research.audit_momentum_hold_2019_followup import (
    _calendar_year_comparison,
    _fixed_leave_one_year,
    _rolling_comparison,
)
from research.momentum_defender_defender_range_escape import (
    DefenderRangeEscapeParams,
    build_range_escape_context,
    execute_formal_targets_at_cost,
    range_locations_at_open,
    run_range_escape,
)
from research.momentum_defender_gold_override_overfit import (
    cscv_pbo,
    expanding_walk_forward,
    leave_one_year_selection,
    paired_block_bootstrap,
    yearly_reality_check,
)
from research.momentum_defender_occam import performance
from research.standard_report import generate_standard_report


DEFAULT_CONFIG = Path("research/configs/defender_range_escape_2019.yaml")
DEFAULT_OUTPUT = Path(
    "experiments/20260827_momentum_defender_defender_range_escape_2019"
)


def _return_hash(returns: pd.Series) -> str:
    return hashlib.sha256(returns.to_numpy(dtype="<f8").tobytes()).hexdigest()


def _json_default(value: object) -> object:
    if isinstance(value, (pd.Timestamp, date)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"cannot JSON encode {type(value).__name__}")


def _params_from_config(config: dict[str, object]) -> list[DefenderRangeEscapeParams]:
    settings = config["occam_sensitivity"]
    result = []
    for anchor, policy, window, upper, step in product(
        settings["anchor_modes"],
        settings["policies"],
        settings["range_windows"],
        settings["upper_thresholds"],
        settings["position_steps"],
    ):
        result.append(
            DefenderRangeEscapeParams(
                anchor_mode=str(anchor),
                policy=str(policy),
                range_window=int(window),
                lower_threshold=float(settings["lower_threshold"]),
                upper_threshold=float(upper),
                position_step=float(step),
            )
        )
    return result


def _apply_deltas(
    surface: pd.DataFrame,
    baseline: pd.Series,
    periods: dict[str, tuple[str, str]],
) -> pd.DataFrame:
    result = surface.copy()
    baseline_metrics = performance(baseline)
    for field in ("annualized_return_252", "sharpe", "max_drawdown"):
        result[f"delta_{field}"] = result[field] - baseline_metrics[field]
    for name, bounds in periods.items():
        measured = performance(baseline.loc[bounds[0] : bounds[1]])
        for field in ("annualized_return_252", "sharpe", "max_drawdown"):
            source = f"{name}_{field}"
            result[f"{name}_delta_{field}"] = result[source] - measured[field]
    result["worst_core_annualized_delta"] = result[
        [
            "delta_annualized_return_252",
            "development_delta_annualized_return_252",
            "validation_delta_annualized_return_252",
        ]
    ].min(axis=1)
    result["worst_core_sharpe_delta"] = result[
        ["delta_sharpe", "development_delta_sharpe", "validation_delta_sharpe"]
    ].min(axis=1)
    result["full_dual_improvement"] = (
        result["delta_annualized_return_252"].gt(0.0)
        & result["delta_sharpe"].gt(0.0)
    )
    result["development_dual_improvement"] = (
        result["development_delta_annualized_return_252"].gt(0.0)
        & result["development_delta_sharpe"].gt(0.0)
    )
    result["validation_dual_improvement"] = (
        result["validation_delta_annualized_return_252"].gt(0.0)
        & result["validation_delta_sharpe"].gt(0.0)
    )
    rank_fields = [
        "annualized_return_252",
        "sharpe",
        "development_annualized_return_252",
        "development_sharpe",
        "validation_annualized_return_252",
        "validation_sharpe",
    ]
    result["diagnostic_mean_percentile_rank"] = result[rank_fields].rank(
        pct=True, method="average"
    ).mean(axis=1)
    return result


def _family_summary(surface: pd.DataFrame) -> pd.DataFrame:
    return (
        surface.groupby(["anchor_mode", "policy"], as_index=False)
        .agg(
            candidates=("candidate_id", "count"),
            full_dual_candidates=("full_dual_improvement", "sum"),
            robust_candidates=("robust_eligible", "sum"),
            best_annualized_delta=("delta_annualized_return_252", "max"),
            best_sharpe_delta=("delta_sharpe", "max"),
            median_annualized_delta=("delta_annualized_return_252", "median"),
            median_sharpe_delta=("delta_sharpe", "median"),
            best_worst_core_annualized_delta=("worst_core_annualized_delta", "max"),
            best_worst_core_sharpe_delta=("worst_core_sharpe_delta", "max"),
        )
        .sort_values(["anchor_mode", "policy"])
    )


def run_research(root: Path, config_path: Path, output: Path) -> dict[str, object]:
    applied_config = config_path if config_path.is_absolute() else root / config_path
    config = yaml.safe_load(applied_config.read_text(encoding="utf-8"))
    experiment = config["experiment"]
    start = date.fromisoformat(str(experiment["evaluation_start"]))
    end = date.fromisoformat(str(experiment["evidence_cutoff"]))
    periods = _periods(config)
    checks = config["overfit_checks"]

    context = build_range_escape_context(root, start=start, end=end)
    formal_returns = context.formal.daily["return"].astype(float)
    direct_returns = context.direct_baseline_daily["return"].astype(float)
    feature_cache = {
        window: range_locations_at_open(context, window)
        for window in sorted(
            {int(value) for value in config["occam_sensitivity"]["range_windows"]}
        )
    }

    rows: list[dict[str, object]] = []
    returns: dict[str, pd.Series] = {}
    runs = {}
    for params in _params_from_config(config):
        candidate_id = params.candidate_id()
        run = run_range_escape(
            context,
            params,
            locations_at_open=feature_cache[params.range_window],
        )
        candidate_returns = run.daily["return"].astype(float)
        returns[candidate_id] = candidate_returns
        runs[candidate_id] = run
        rows.append(
            {
                **_metric_row(
                    candidate_id,
                    "defender_range_escape",
                    candidate_returns,
                    periods,
                ),
                **asdict(params),
                "overlay_days": run.audit["overlay_days"],
                "formal_defender_days": run.audit["formal_defender_days"],
                "average_defender_weight": run.audit[
                    "average_defender_weight_on_formal_defender_days"
                ],
                "zero_defender_days": run.audit["zero_defender_days"],
                "high_reduce_observations": run.audit[
                    "high_reduce_observations"
                ],
                "low_add_observations": run.audit["low_add_observations"],
            }
        )

    surface = _apply_deltas(pd.DataFrame(rows), formal_returns, periods)
    selection = config["selection"]
    surface["robust_eligible"] = (
        surface["full_dual_improvement"]
        & surface["development_dual_improvement"]
        & surface["validation_dual_improvement"]
        & surface["delta_max_drawdown"].ge(
            -float(selection["maximum_mdd_worsening"])
        )
    )
    eligible = surface.loc[surface["robust_eligible"]].copy()
    diagnostic = surface.sort_values(
        [
            "diagnostic_mean_percentile_rank",
            "worst_core_sharpe_delta",
            "worst_core_annualized_delta",
            "overlay_days",
            "candidate_id",
        ],
        ascending=[False, False, False, True, True],
    ).iloc[0]
    diagnostic_id = str(diagnostic["candidate_id"])
    supplied_id = str(config["supplied_mechanism"]["candidate_id"])
    if supplied_id not in returns:
        raise AssertionError(f"supplied mechanism missing from surface: {supplied_id}")

    # Deduplicate mechanically identical daily paths before multiplicity tests.
    unique_returns: dict[str, pd.Series] = {}
    seen_hashes: set[str] = set()
    for candidate_id, candidate_returns in returns.items():
        digest = _return_hash(candidate_returns)
        if digest in seen_hashes:
            continue
        seen_hashes.add(digest)
        unique_returns[candidate_id] = candidate_returns
    panel = pd.DataFrame(unique_returns, index=context.calendar)

    cscv_frame, cscv = cscv_pbo(
        panel,
        formal_returns,
        block_count=int(checks["cscv_blocks"]),
    )
    reality = yearly_reality_check(
        panel,
        formal_returns,
        repetitions=int(checks["yearly_reality_check_repetitions"]),
        seed=int(checks["random_seed"]),
    )
    walk_forward = expanding_walk_forward(panel, formal_returns)
    leave_year_selection = leave_one_year_selection(panel, formal_returns)
    diagnostic_returns = returns[diagnostic_id]
    bootstrap_frame, bootstrap = paired_block_bootstrap(
        diagnostic_returns,
        formal_returns,
        block_size=int(checks["paired_block_bootstrap_block"]),
        repetitions=int(checks["paired_block_bootstrap_repetitions"]),
        seed=int(checks["random_seed"]),
    )
    fixed_leave_year = _fixed_leave_one_year(
        diagnostic_returns, formal_returns
    )
    annual = _calendar_year_comparison(diagnostic_returns, formal_returns)
    rolling = _rolling_comparison(
        diagnostic_returns,
        formal_returns,
        [int(value) for value in checks["rolling_windows"]],
    )
    # Event windows use the weight-level baseline from the same execution
    # engine.  Comparing to the candidate-level formal replay would turn its
    # harmless ~1e-8 cost-ordering difference into many false micro-events.
    events = _difference_events(diagnostic_returns, direct_returns)
    leave_event = _leave_one_event(diagnostic_returns, direct_returns, events)

    cost_rows: list[dict[str, object]] = []
    for multiplier_value in checks["transaction_cost_multipliers"]:
        multiplier = float(multiplier_value)
        cost_baseline = execute_formal_targets_at_cost(context, multiplier)
        for candidate_id in dict.fromkeys([supplied_id, diagnostic_id]):
            params = runs[candidate_id].params
            cost_run = run_range_escape(
                context,
                params,
                locations_at_open=feature_cache[params.range_window],
                cost_multiplier=multiplier,
            )
            measured = performance(cost_run.daily["return"].astype(float))
            baseline_measured = performance(cost_baseline)
            cost_rows.append(
                {
                    "cost_multiplier": multiplier,
                    "candidate_id": candidate_id,
                    "annualized_return_252": measured["annualized_return_252"],
                    "sharpe": measured["sharpe"],
                    "max_drawdown": measured["max_drawdown"],
                    "delta_annualized_return_252": measured[
                        "annualized_return_252"
                    ]
                    - baseline_measured["annualized_return_252"],
                    "delta_sharpe": measured["sharpe"]
                    - baseline_measured["sharpe"],
                    "delta_max_drawdown": measured["max_drawdown"]
                    - baseline_measured["max_drawdown"],
                }
            )
    cost_stress = pd.DataFrame(cost_rows)

    family = _family_summary(surface)
    rolling_summary = {
        str(window): {
            "observations": int(len(group)),
            "annualized_return_win_rate": float(
                group["annualized_return_delta"].gt(0.0).mean()
            ),
            "sharpe_win_rate": float(group["sharpe_delta"].gt(0.0).mean()),
            "dual_win_rate": float(
                (
                    group["annualized_return_delta"].gt(0.0)
                    & group["sharpe_delta"].gt(0.0)
                ).mean()
            ),
        }
        for window, group in rolling.groupby("window")
    }

    supplied_row = surface.set_index("candidate_id").loc[supplied_id]
    diagnostic_row = surface.set_index("candidate_id").loc[diagnostic_id]
    decision = (
        "reject_no_robust_dual_improvement"
        if eligible.empty
        else "research_candidate_only_requires_independent_forward_evidence"
    )
    audit: dict[str, object] = {
        "research_id": experiment["id"],
        "status": "completed_research_only",
        "evidence_status": experiment["evidence_status"],
        "evaluation_start": start.isoformat(),
        "evidence_cutoff": end.isoformat(),
        "baseline_strategy_id": experiment["baseline_strategy_id"],
        "formal_return_hash": _return_hash(formal_returns),
        "direct_baseline_return_hash": _return_hash(direct_returns),
        "baseline_parity_max_abs_error": context.baseline_parity_max_abs_error,
        "baseline_metrics": performance(formal_returns),
        "candidate_ids": int(len(surface)),
        "unique_paths": int(len(panel.columns)),
        "full_dual_improvement_candidates": int(
            surface["full_dual_improvement"].sum()
        ),
        "robust_eligible_candidates": int(len(eligible)),
        "supplied_candidate": supplied_id,
        "supplied_metrics": {
            key: supplied_row[key]
            for key in (
                "annualized_return_252",
                "sharpe",
                "max_drawdown",
                "delta_annualized_return_252",
                "delta_sharpe",
                "delta_max_drawdown",
                "overlay_days",
                "average_defender_weight",
                "zero_defender_days",
            )
        },
        "diagnostic_leader": diagnostic_id,
        "diagnostic_leader_metrics": {
            key: diagnostic_row[key]
            for key in (
                "annualized_return_252",
                "sharpe",
                "max_drawdown",
                "delta_annualized_return_252",
                "delta_sharpe",
                "delta_max_drawdown",
                "worst_core_annualized_delta",
                "worst_core_sharpe_delta",
                "overlay_days",
                "average_defender_weight",
            )
        },
        "paired_block_bootstrap_diagnostic_leader": bootstrap,
        "cscv": cscv,
        "reality_check": reality,
        "walk_forward_dual_win_rate": float(
            (
                walk_forward["test_return_delta"].gt(0.0)
                & walk_forward["test_sharpe_delta"].gt(0.0)
            ).mean()
        ),
        "leave_one_year_selection_dual_win_rate": float(
            (
                leave_year_selection["test_return_delta"].gt(0.0)
                & leave_year_selection["test_sharpe_delta"].gt(0.0)
            ).mean()
        ),
        "fixed_diagnostic_delete_year_dual_pass_rate": float(
            (
                fixed_leave_year["annualized_return_delta"].gt(0.0)
                & fixed_leave_year["sharpe_delta"].gt(0.0)
            ).mean()
        ),
        "calendar_year_dual_win_rate": float(
            (
                annual["total_return_delta"].gt(0.0)
                & annual["sharpe_delta"].gt(0.0)
            ).mean()
        ),
        "rolling": rolling_summary,
        "difference_events": {
            "events": int(len(events)),
            "positive": int(events["log_excess"].gt(0.0).sum()),
            "negative": int(events["log_excess"].lt(0.0).sum()),
            "leave_one_min_annualized_return_252": float(
                leave_event["annualized_return_252"].min()
            ),
            "leave_one_min_sharpe": float(leave_event["sharpe"].min()),
        },
        "decision": decision,
        "production_changed": False,
    }

    output.mkdir(parents=True, exist_ok=True)
    surface.sort_values(
        ["diagnostic_mean_percentile_rank", "candidate_id"],
        ascending=[False, True],
    ).to_csv(output / "candidate_surface.csv", index=False)
    eligible.to_csv(output / "robust_eligible_candidates.csv", index=False)
    family.to_csv(output / "family_summary.csv", index=False)
    pd.DataFrame(returns, index=context.calendar).to_parquet(
        output / "daily_returns.parquet"
    )
    context.direct_baseline_daily.to_parquet(output / "direct_baseline_daily.parquet")
    runs[supplied_id].state.to_parquet(output / "supplied_state.parquet")
    runs[supplied_id].targets.to_parquet(output / "supplied_targets.parquet")
    runs[diagnostic_id].state.to_parquet(output / "diagnostic_state.parquet")
    runs[diagnostic_id].targets.to_parquet(output / "diagnostic_targets.parquet")
    bootstrap_frame.to_csv(output / "paired_bootstrap.csv", index=False)
    cscv_frame.to_csv(output / "cscv.csv", index=False)
    walk_forward.to_csv(output / "walk_forward.csv", index=False)
    leave_year_selection.to_csv(
        output / "leave_one_year_selection.csv", index=False
    )
    fixed_leave_year.to_csv(
        output / "fixed_diagnostic_leave_one_year.csv", index=False
    )
    annual.to_csv(output / "calendar_year_comparison.csv", index=False)
    rolling.to_csv(output / "rolling_comparison.csv", index=False)
    events.to_csv(output / "difference_events.csv", index=False)
    leave_event.to_csv(output / "leave_one_event.csv", index=False)
    cost_stress.to_csv(output / "cost_stress.csv", index=False)
    (output / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    (output / "applied_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    generate_standard_report(
        diagnostic_returns,
        formal_returns,
        str(experiment["baseline_strategy_id"]),
        output / "diagnostic_vs_formal.html",
        {
            "strategy_name": diagnostic_id,
            "research_status": "research_only_not_production",
            "start": start,
            "end": end,
        },
    )

    print(json.dumps(audit, ensure_ascii=False, indent=2, default=_json_default))
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run_research(args.root.resolve(), args.config, args.output)


if __name__ == "__main__":
    main()
