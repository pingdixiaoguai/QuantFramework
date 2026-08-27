"""Final per-axis audit of the strongest conditioned range-escape candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, replace
from datetime import date
from itertools import product
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
    "research/configs/conditioned_defender_range_escape_final_axis.yaml"
)
DEFAULT_OUTPUT = Path(
    "experiments/20260827_conditioned_defender_range_escape_final_axis"
)


def _return_hash(returns: pd.Series) -> str:
    return hashlib.sha256(returns.to_numpy(dtype="<f8").tobytes()).hexdigest()


def _optional_float(value: object) -> float | None:
    return None if value is None or pd.isna(value) else float(value)


def _center(config: dict[str, object]) -> ConditionedRangeEscapeParams:
    values = config["center"]
    return ConditionedRangeEscapeParams(
        anchor_mode=str(values["anchor_mode"]),
        range_window=int(values["range_window"]),
        upper_threshold=float(values["upper_threshold"]),
        momentum_weight=float(values["momentum_weight"]),
        quality_window=int(values["quality_window"]),
        gate=MomentumQualityGate(
            absolute_floor=_optional_float(values["absolute_floor"]),
            relative_floor=_optional_float(values["relative_floor"]),
        ),
        hold_policy=str(values["hold_policy"]),
        hold_days=int(values["hold_days"]),
    )


def _replace_field(
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
                relative_floor=_optional_float(value),
            ),
        )
    cast = int if field in {"range_window", "quality_window", "hold_days"} else float
    return replace(center, **{field: cast(value)})


def _candidate_catalog(
    center: ConditionedRangeEscapeParams,
    config: dict[str, object],
) -> tuple[dict[str, ConditionedRangeEscapeParams], dict[str, set[str]]]:
    params_by_id = {center.candidate_id: center}
    sources: dict[str, set[str]] = {center.candidate_id: {"center"}}

    def add(params: ConditionedRangeEscapeParams, source: str) -> None:
        params_by_id[params.candidate_id] = params
        sources.setdefault(params.candidate_id, set()).add(source)

    for field, values in config["one_factor_axes"].items():
        for value in values:
            add(_replace_field(center, str(field), value), f"axis:{field}")
    rq = config["interaction_surfaces"]["range_quality"]
    for range_window, quality_window in product(
        rq["range_window"], rq["quality_window"]
    ):
        add(
            replace(
                center,
                range_window=int(range_window),
                quality_window=int(quality_window),
            ),
            "interaction:range_quality",
        )
    tw = config["interaction_surfaces"]["threshold_weight"]
    for relative_floor, momentum_weight in product(
        tw["relative_floor"], tw["momentum_weight"]
    ):
        add(
            replace(
                center,
                momentum_weight=float(momentum_weight),
                gate=MomentumQualityGate(
                    absolute_floor=center.gate.absolute_floor,
                    relative_floor=float(relative_floor),
                ),
            ),
            "interaction:threshold_weight",
        )
    hh = config["interaction_surfaces"]["high_hold"]
    for upper_threshold, hold_days in product(
        hh["upper_threshold"], hh["hold_days"]
    ):
        add(
            replace(
                center,
                upper_threshold=float(upper_threshold),
                hold_days=int(hold_days),
            ),
            "interaction:high_hold",
        )
    return params_by_id, sources


def _axis_summary(
    surface: pd.DataFrame,
    config: dict[str, object],
) -> pd.DataFrame:
    rows = []
    for field in config["one_factor_axes"]:
        sample = surface.loc[
            surface["sources"].str.contains(f"axis:{field}", regex=False)
            | surface["sources"].str.contains("center", regex=False)
        ].drop_duplicates("candidate_id")
        rows.append(
            {
                "axis": field,
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
        )
    result = pd.DataFrame(rows)
    rules = config["axis_robustness"]
    result["axis_pass"] = (
        result["full_dual_rate"].ge(float(rules["minimum_full_dual_rate"]))
        & result["development_dual_rate"].ge(
            float(rules["minimum_development_dual_rate"])
        )
        & result["validation_dual_rate"].ge(
            float(rules["minimum_validation_dual_rate"])
        )
        & result["recent_dual_rate"].ge(float(rules["minimum_recent_dual_rate"]))
        & result["annualized_delta_q25"].gt(0.0)
        & result["sharpe_delta_q25"].gt(0.0)
        & result["validation_annualized_delta_q25"].gt(0.0)
        & result["validation_sharpe_delta_q25"].gt(0.0)
    )
    return result


def _interaction_summary(surface: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name in ("range_quality", "threshold_weight", "high_hold"):
        sample = surface.loc[
            surface["sources"].str.contains(
                f"interaction:{name}", regex=False
            )
        ].drop_duplicates("candidate_id")
        rows.append(
            {
                "interaction": name,
                "candidate_count": int(len(sample)),
                "full_dual_rate": float(sample["full_dual"].mean()),
                "development_dual_rate": float(sample["development_dual"].mean()),
                "validation_dual_rate": float(sample["validation_dual"].mean()),
                "recent_dual_rate": float(sample["recent_dual"].mean()),
                "annualized_delta_q25": float(
                    sample["delta_annualized_return_252"].quantile(0.25)
                ),
                "sharpe_delta_q25": float(sample["delta_sharpe"].quantile(0.25)),
                "validation_annualized_delta_q25": float(
                    sample["validation_delta_annualized_return_252"].quantile(0.25)
                ),
                "validation_sharpe_delta_q25": float(
                    sample["validation_delta_sharpe"].quantile(0.25)
                ),
            }
        )
    return pd.DataFrame(rows)


def run_research(root: Path, config_path: Path, output: Path) -> dict[str, object]:
    applied = config_path if config_path.is_absolute() else root / config_path
    config = yaml.safe_load(applied.read_text(encoding="utf-8"))
    experiment = config["experiment"]
    start = date.fromisoformat(str(experiment["evaluation_start"]))
    end = date.fromisoformat(str(experiment["evidence_cutoff"]))
    periods = _periods(config)
    checks = config["overfit_checks"]
    center = _center(config)
    catalog, sources = _candidate_catalog(center, config)

    context = build_range_escape_context(root, start=start, end=end)
    cache = build_weight_execution_cache(context)
    baseline = context.formal.daily["return"].astype(float)
    direct_baseline = execute_targets_cached(
        context, context.baseline_targets, cache
    )["return"].astype(float)
    location_cache: dict[int, pd.DataFrame] = {}
    quality_cache: dict[int, pd.DataFrame] = {}
    records = []
    returns: dict[str, pd.Series] = {}
    runs = {}
    for params in catalog.values():
        if params.range_window not in location_cache:
            location_cache[params.range_window] = range_locations_at_open(
                context, params.range_window
            )
        if params.quality_window not in quality_cache:
            quality_cache[params.quality_window] = all_metrics_at_open(
                context.formal.context.curves,
                QUALITY_METRIC,
                params.quality_window,
            )
        run = run_conditioned_range_escape(
            context,
            params,
            locations_at_open=location_cache[params.range_window],
            quality_metrics_at_open=quality_cache[params.quality_window],
            execution_cache=cache,
        )
        runs[params.candidate_id] = run
        returns[params.candidate_id] = run.daily["return"].astype(float)
        record = _metric_record(run, periods, stage="final_axis")
        record["sources"] = "|".join(sorted(sources[params.candidate_id]))
        records.append(record)
    surface = _add_deltas(pd.DataFrame(records), baseline, periods)
    axes = _axis_summary(surface, config)
    interactions = _interaction_summary(surface)
    center_row = surface.loc[surface["candidate_id"].eq(center.candidate_id)].iloc[0]
    all_axes_pass = bool(axes["axis_pass"].all())

    broad_path = Path(str(experiment["broad_output"]))
    focused_path = Path(str(experiment["focused_output"]))
    if not broad_path.is_absolute():
        broad_path = root / broad_path
    if not focused_path.is_absolute():
        focused_path = root / focused_path
    combined = {
        str(column): values
        for path, filename in (
            (broad_path, "daily_returns.parquet"),
            (focused_path, "focused_daily_returns.parquet"),
        )
        for column, values in pd.read_parquet(path / filename).items()
    }
    combined.update(returns)
    unique: dict[str, pd.Series] = {}
    seen: set[str] = set()
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
    fixed_leave_year = _fixed_leave_one_year(center_returns, baseline)
    annual = _calendar_year_comparison(center_returns, baseline)
    rolling = _rolling_comparison(
        center_returns,
        baseline,
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
            quality_metrics_at_open=quality_cache[center.quality_window],
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

    audit = {
        "research_id": experiment["id"],
        "status": "completed_research_only",
        "evidence_status": experiment["evidence_status"],
        "baseline_metrics": performance(baseline),
        "axis_candidate_ids": int(len(surface)),
        "combined_unique_paths": int(len(panel.columns)),
        "center_candidate": center.candidate_id,
        "center_params": {
            **asdict(center),
            "gate_family": center.gate.family,
            "gate_id": center.gate.gate_id,
        },
        "center_metrics": {
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
        "axes_passed": int(axes["axis_pass"].sum()),
        "axes_total": int(len(axes)),
        "all_axes_pass": all_axes_pass,
        "failed_axes": axes.loc[~axes["axis_pass"], "axis"].tolist(),
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
        "fixed_center_delete_year_dual_pass_rate": float(
            (
                fixed_leave_year["annualized_return_delta"].gt(0.0)
                & fixed_leave_year["sharpe_delta"].gt(0.0)
            ).mean()
        ),
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
            "axis_robust_research_candidate_requires_explicit_promotion"
            if all_axes_pass
            else "reject_axis_instability"
        ),
        "production_changed": False,
    }

    output.mkdir(parents=True, exist_ok=True)
    surface.to_csv(output / "axis_surface.csv", index=False)
    axes.to_csv(output / "axis_summary.csv", index=False)
    interactions.to_csv(output / "interaction_summary.csv", index=False)
    pd.DataFrame(returns, index=context.calendar).to_parquet(
        output / "axis_daily_returns.parquet"
    )
    runs[center.candidate_id].state.to_parquet(output / "center_state.parquet")
    runs[center.candidate_id].targets.to_parquet(output / "center_targets.parquet")
    runs[center.candidate_id].daily.to_parquet(output / "center_daily.parquet")
    bootstrap_frame.to_csv(output / "paired_bootstrap.csv", index=False)
    cscv_frame.to_csv(output / "cscv.csv", index=False)
    walk_forward.to_csv(output / "walk_forward.csv", index=False)
    leave_year_selection.to_csv(output / "leave_one_year_selection.csv", index=False)
    fixed_leave_year.to_csv(output / "fixed_center_leave_one_year.csv", index=False)
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
        output / "center_vs_formal.html",
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
