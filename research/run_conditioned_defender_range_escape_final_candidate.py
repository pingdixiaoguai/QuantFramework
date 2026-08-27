"""Final audit of the Occam multi-horizon conditioned range-escape candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, replace
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from research.audit_current_strategy_occam_robustness import _periods
from research.audit_defender_selector_2019 import _difference_events, _leave_one_event
from research.audit_momentum_hold_2019_followup import (
    _calendar_year_comparison,
    _fixed_leave_one_year,
    _rolling_comparison,
)
from research.momentum_defender_conditioned_range_escape import (
    ConditionedRangeEscapeParams,
    MomentumQualityGate,
    aggregate_quality_metrics,
    build_weight_execution_cache,
    execute_targets_cached,
    run_conditioned_range_escape,
)
from research.momentum_defender_defender_range_escape import (
    build_range_escape_context,
    range_locations_at_open,
)
from research.momentum_defender_gold_override import QUALITY_METRIC
from research.momentum_defender_gold_override_overfit import (
    cscv_pbo,
    expanding_walk_forward,
    leave_one_year_selection,
    paired_block_bootstrap,
    yearly_reality_check,
)
from research.momentum_defender_occam import performance
from research.momentum_top1_defender_escape import all_metrics_at_open
from research.run_conditioned_defender_range_escape_2019 import (
    _add_deltas,
    _json_default,
    _metric_record,
)
from research.standard_report import generate_standard_report


DEFAULT_CONFIG = Path(
    "research/configs/conditioned_defender_range_escape_final_candidate.yaml"
)
DEFAULT_OUTPUT = Path(
    "experiments/20260827_conditioned_defender_range_escape_final_candidate"
)


def _return_hash(returns: pd.Series) -> str:
    return hashlib.sha256(returns.to_numpy(dtype="<f8").tobytes()).hexdigest()


def _optional_float(value: object) -> float | None:
    return None if value is None or pd.isna(value) else float(value)


def _candidate(config: dict[str, object]) -> ConditionedRangeEscapeParams:
    values = config["candidate"]
    windows = tuple(int(value) for value in values["quality_windows"])
    return ConditionedRangeEscapeParams(
        anchor_mode=str(values["anchor_mode"]),
        range_window=int(values["range_window"]),
        upper_threshold=float(values["upper_threshold"]),
        momentum_weight=float(values["momentum_weight"]),
        quality_window=40,
        gate=MomentumQualityGate(
            absolute_floor=_optional_float(values["absolute_floor"]),
            relative_floor=_optional_float(values["relative_floor"]),
        ),
        hold_policy=str(values["hold_policy"]),
        hold_days=int(values["hold_days"]),
        quality_windows=windows,
        quality_aggregation=str(values["quality_aggregation"]),
    )


def _replace_axis(
    center: ConditionedRangeEscapeParams,
    field: str,
    value: object,
) -> ConditionedRangeEscapeParams:
    if field == "absolute_floor":
        return replace(
            center,
            gate=MomentumQualityGate(
                absolute_floor=_optional_float(value),
                relative_floor=center.gate.relative_floor,
            ),
        )
    if field == "relative_floor":
        return replace(
            center,
            gate=MomentumQualityGate(
                absolute_floor=center.gate.absolute_floor,
                relative_floor=float(value),
            ),
        )
    cast = int if field in {"range_window", "hold_days"} else float
    return replace(center, **{field: cast(value)})


def _strict_dual_flags(
    surface: pd.DataFrame,
    epsilon: float,
) -> pd.DataFrame:
    result = surface.copy()
    result["full_dual"] = (
        result["delta_annualized_return_252"].gt(epsilon)
        & result["delta_sharpe"].gt(epsilon)
    )
    for name in ("development", "validation", "recent", "complete_pool"):
        result[f"{name}_dual"] = (
            result[f"{name}_delta_annualized_return_252"].gt(epsilon)
            & result[f"{name}_delta_sharpe"].gt(epsilon)
        )
    return result


def _axis_summary(
    surface: pd.DataFrame,
    config: dict[str, object],
) -> pd.DataFrame:
    rows = []
    rules = config["robustness"]
    for axis in config["one_factor_axes"]:
        source_tokens = surface["axis"].str.split("|")
        sample = surface.loc[
            source_tokens.map(lambda values: axis in values or "center" in values)
        ].drop_duplicates("candidate_id")
        row = {
            "axis": axis,
            "candidate_count": int(len(sample)),
            "full_dual_rate": float(sample["full_dual"].mean()),
            "development_dual_rate": float(sample["development_dual"].mean()),
            "validation_dual_rate": float(sample["validation_dual"].mean()),
            "recent_dual_rate": float(sample["recent_dual"].mean()),
            "annualized_delta_q25": float(
                sample["delta_annualized_return_252"].quantile(0.25)
            ),
            "sharpe_delta_q25": float(sample["delta_sharpe"].quantile(0.25)),
            "development_annualized_delta_q25": float(
                sample["development_delta_annualized_return_252"].quantile(0.25)
            ),
            "development_sharpe_delta_q25": float(
                sample["development_delta_sharpe"].quantile(0.25)
            ),
            "validation_annualized_delta_q25": float(
                sample["validation_delta_annualized_return_252"].quantile(0.25)
            ),
            "validation_sharpe_delta_q25": float(
                sample["validation_delta_sharpe"].quantile(0.25)
            ),
            "recent_annualized_delta_q25": float(
                sample["recent_delta_annualized_return_252"].quantile(0.25)
            ),
            "recent_sharpe_delta_q25": float(
                sample["recent_delta_sharpe"].quantile(0.25)
            ),
        }
        row["axis_pass"] = bool(
            row["full_dual_rate"] >= float(rules["minimum_full_dual_rate"])
            and row["development_dual_rate"]
            >= float(rules["minimum_development_dual_rate"])
            and row["validation_dual_rate"]
            >= float(rules["minimum_validation_dual_rate"])
            and row["recent_dual_rate"] >= float(rules["minimum_recent_dual_rate"])
            and row["annualized_delta_q25"] > 0.0
            and row["sharpe_delta_q25"] > 0.0
            and row["development_annualized_delta_q25"] > 0.0
            and row["development_sharpe_delta_q25"] > 0.0
            and row["validation_annualized_delta_q25"] > 0.0
            and row["validation_sharpe_delta_q25"] > 0.0
            and row["recent_annualized_delta_q25"] >= -float(rules["delta_epsilon"])
            and row["recent_sharpe_delta_q25"] >= -float(rules["delta_epsilon"])
        )
        rows.append(row)
    return pd.DataFrame(rows)


def run_research(root: Path, config_path: Path, output: Path) -> dict[str, object]:
    applied = config_path if config_path.is_absolute() else root / config_path
    config = yaml.safe_load(applied.read_text(encoding="utf-8"))
    experiment = config["experiment"]
    start = date.fromisoformat(str(experiment["evaluation_start"]))
    end = date.fromisoformat(str(experiment["evidence_cutoff"]))
    periods = _periods(config)
    checks = config["overfit_checks"]
    center = _candidate(config)

    context = build_range_escape_context(root, start=start, end=end)
    cache = build_weight_execution_cache(context)
    baseline = context.formal.daily["return"].astype(float)
    direct_baseline = execute_targets_cached(
        context, context.baseline_targets, cache
    )["return"].astype(float)
    quality_panels = {
        window: all_metrics_at_open(
            context.formal.context.curves, QUALITY_METRIC, window
        )
        for window in center.quality_windows
    }
    ensemble_metrics = aggregate_quality_metrics(
        quality_panels, center.quality_windows, center.quality_aggregation
    )

    catalog = {center.candidate_id: center}
    sources: dict[str, set[str]] = {center.candidate_id: {"center"}}
    for field, values in config["one_factor_axes"].items():
        for value in values:
            params = _replace_axis(center, str(field), value)
            catalog[params.candidate_id] = params
            sources.setdefault(params.candidate_id, set()).add(str(field))
    for field, values in config.get("stress_axes", {}).items():
        for value in values:
            params = _replace_axis(center, str(field), value)
            catalog[params.candidate_id] = params
            sources.setdefault(params.candidate_id, set()).add(
                f"stress:{field}"
            )

    location_cache: dict[int, pd.DataFrame] = {}
    records = []
    returns = {}
    runs = {}
    for params in catalog.values():
        if params.range_window not in location_cache:
            location_cache[params.range_window] = range_locations_at_open(
                context, params.range_window
            )
        run = run_conditioned_range_escape(
            context,
            params,
            locations_at_open=location_cache[params.range_window],
            quality_metrics_at_open=ensemble_metrics,
            execution_cache=cache,
        )
        runs[params.candidate_id] = run
        returns[params.candidate_id] = run.daily["return"].astype(float)
        record = _metric_record(run, periods, stage="final_candidate")
        record["axis"] = "|".join(sorted(sources[params.candidate_id]))
        records.append(record)
    surface = _strict_dual_flags(
        _add_deltas(pd.DataFrame(records), baseline, periods),
        float(config["robustness"]["delta_epsilon"]),
    )
    axes = _axis_summary(surface, config)

    group_path = Path(str(experiment["ensemble_group_summary"]))
    if not group_path.is_absolute():
        group_path = root / group_path
    groups = pd.read_csv(group_path)
    selected_group = groups.loc[
        groups["profile_name"].eq(config["robustness"]["selected_ensemble_profile"])
        & groups["aggregation"].eq(
            config["robustness"]["selected_ensemble_aggregation"]
        )
    ].iloc[0]
    profile_pass = bool(
        selected_group["full_dual_rate"]
        >= float(config["robustness"]["minimum_ensemble_threshold_full_dual_rate"])
        and selected_group["validation_dual_rate"]
        >= float(
            config["robustness"]["minimum_ensemble_threshold_validation_dual_rate"]
        )
        and selected_group["annualized_delta_q25"] > 0.0
        and selected_group["sharpe_delta_q25"] > 0.0
    )
    axis_pass = bool(axes["axis_pass"].all())

    combined = {}
    for value in experiment["prior_return_panels"]:
        path = Path(str(value))
        if not path.is_absolute():
            path = root / path
        panel = pd.read_parquet(path)
        combined.update({str(column): panel[column] for column in panel.columns})
    combined.update(returns)
    unique = {}
    seen = set()
    for candidate_id, candidate_returns in combined.items():
        candidate_returns = candidate_returns.astype(float)
        digest = _return_hash(candidate_returns)
        if digest not in seen:
            seen.add(digest)
            unique[candidate_id] = candidate_returns
    panel = pd.DataFrame(unique, index=context.calendar)

    cscv_frame, cscv = cscv_pbo(panel, baseline, block_count=int(checks["cscv_blocks"]))
    reality = yearly_reality_check(
        panel,
        baseline,
        repetitions=int(checks["yearly_reality_check_repetitions"]),
        seed=int(checks["random_seed"]),
    )
    walk_forward = expanding_walk_forward(panel, baseline)
    leave_year_selection = leave_one_year_selection(panel, baseline)
    center_returns = returns[center.candidate_id]
    bootstrap_frame, bootstrap = paired_block_bootstrap(
        center_returns,
        baseline,
        block_size=int(checks["paired_block_bootstrap_block"]),
        repetitions=int(checks["paired_block_bootstrap_repetitions"]),
        seed=int(checks["random_seed"]),
    )
    fixed_leave_year = _fixed_leave_one_year(center_returns, direct_baseline)
    annual = _calendar_year_comparison(center_returns, direct_baseline)
    rolling = _rolling_comparison(
        center_returns,
        direct_baseline,
        [int(value) for value in checks["rolling_windows"]],
    )
    events = _difference_events(center_returns, direct_baseline)
    leave_event = _leave_one_event(center_returns, direct_baseline, events)

    cost_rows = []
    for multiplier_value in checks["transaction_cost_multipliers"]:
        multiplier = float(multiplier_value)
        cost_baseline = execute_targets_cached(
            context,
            context.baseline_targets,
            cache,
            cost_multiplier=multiplier,
        )["return"].astype(float)
        cost_run = run_conditioned_range_escape(
            context,
            center,
            locations_at_open=location_cache[center.range_window],
            quality_metrics_at_open=ensemble_metrics,
            execution_cache=cache,
            cost_multiplier=multiplier,
        )
        measured = performance(cost_run.daily["return"].astype(float))
        base_measured = performance(cost_baseline)
        cost_rows.append(
            {
                "cost_multiplier": multiplier,
                "annualized_return_252": measured["annualized_return_252"],
                "sharpe": measured["sharpe"],
                "max_drawdown": measured["max_drawdown"],
                "delta_annualized_return_252": measured["annualized_return_252"]
                - base_measured["annualized_return_252"],
                "delta_sharpe": measured["sharpe"] - base_measured["sharpe"],
                "delta_max_drawdown": measured["max_drawdown"]
                - base_measured["max_drawdown"],
            }
        )
    cost_stress = pd.DataFrame(cost_rows)

    center_row = surface.loc[surface["candidate_id"].eq(center.candidate_id)].iloc[0]
    research_robust = bool(axis_pass and profile_pass)
    rolling_summary = {
        str(window): {
            "observations": int(len(sample)),
            "dual_win_rate": float(
                (
                    sample["annualized_return_delta"].gt(
                        float(config["robustness"]["delta_epsilon"])
                    )
                    & sample["sharpe_delta"].gt(
                        float(config["robustness"]["delta_epsilon"])
                    )
                ).mean()
            ),
        }
        for window, sample in rolling.groupby("window")
    }
    audit = {
        "research_id": experiment["id"],
        "status": "completed_research_only",
        "evidence_status": experiment["evidence_status"],
        "baseline_strategy_id": experiment["baseline_strategy_id"],
        "baseline_metrics": performance(baseline),
        "candidate_id": center.candidate_id,
        "candidate_params": {
            **asdict(center),
            "gate_family": center.gate.family,
            "gate_id": center.gate.gate_id,
        },
        "candidate_metrics": {
            key: center_row[key]
            for key in (
                "annualized_return_252",
                "sharpe",
                "max_drawdown",
                "delta_annualized_return_252",
                "delta_sharpe",
                "development_delta_annualized_return_252",
                "development_delta_sharpe",
                "validation_delta_annualized_return_252",
                "validation_delta_sharpe",
                "recent_delta_annualized_return_252",
                "recent_delta_sharpe",
                "escape_entries",
                "escape_days",
            )
        },
        "axis_candidate_ids": int(len(surface)),
        "axes_passed": int(axes["axis_pass"].sum()),
        "axes_total": int(len(axes)),
        "axis_robust": axis_pass,
        "ensemble_threshold_profile": selected_group.to_dict(),
        "ensemble_profile_robust": profile_pass,
        "research_robust": research_robust,
        "combined_unique_paths": int(len(panel.columns)),
        "paired_block_bootstrap": bootstrap,
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
        "fixed_candidate_delete_year_dual_pass_rate": float(
            (
                fixed_leave_year["annualized_return_delta"].gt(0.0)
                & fixed_leave_year["sharpe_delta"].gt(0.0)
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
        "decision": (
            "robust_research_candidate_not_statistically_proven"
            if research_robust
            else "reject_robustness_failure"
        ),
        "production_changed": False,
    }

    output.mkdir(parents=True, exist_ok=True)
    surface.to_csv(output / "axis_surface.csv", index=False)
    axes.to_csv(output / "axis_summary.csv", index=False)
    pd.DataFrame(returns, index=context.calendar).to_parquet(
        output / "candidate_family_returns.parquet"
    )
    runs[center.candidate_id].state.to_parquet(output / "candidate_state.parquet")
    runs[center.candidate_id].targets.to_parquet(output / "candidate_targets.parquet")
    runs[center.candidate_id].daily.to_parquet(output / "candidate_daily.parquet")
    bootstrap_frame.to_csv(output / "paired_bootstrap.csv", index=False)
    cscv_frame.to_csv(output / "cscv.csv", index=False)
    walk_forward.to_csv(output / "walk_forward.csv", index=False)
    leave_year_selection.to_csv(output / "leave_one_year_selection.csv", index=False)
    fixed_leave_year.to_csv(output / "fixed_candidate_leave_one_year.csv", index=False)
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
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    generate_standard_report(
        center_returns,
        baseline,
        str(experiment["baseline_strategy_id"]),
        output / "candidate_vs_formal.html",
        {
            "strategy_name": center.candidate_id,
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
