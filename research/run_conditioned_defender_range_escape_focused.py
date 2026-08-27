"""Focused stability audit after the broad conditioned range-escape screen."""

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
    "research/configs/conditioned_defender_range_escape_focused.yaml"
)
DEFAULT_OUTPUT = Path(
    "experiments/20260827_conditioned_defender_range_escape_focused"
)


def _return_hash(returns: pd.Series) -> str:
    return hashlib.sha256(returns.to_numpy(dtype="<f8").tobytes()).hexdigest()


def _optional_float(value: object) -> float | None:
    return None if value is None or pd.isna(value) else float(value)


def _params(family: dict[str, object], overrides: dict[str, object] | None = None):
    values = {**family["center"], **(overrides or {})}
    return ConditionedRangeEscapeParams(
        anchor_mode=str(values["anchor_mode"]),
        range_window=int(values["range_window"]),
        upper_threshold=float(values["upper_threshold"]),
        momentum_weight=float(values["momentum_weight"]),
        quality_window=int(values["quality_window"]),
        gate=MomentumQualityGate(
            absolute_floor=_optional_float(values.get("absolute_floor")),
            relative_floor=_optional_float(values.get("relative_floor")),
        ),
        hold_policy=str(values["hold_policy"]),
        hold_days=int(values["hold_days"]),
    )


def _family_params(
    family_name: str,
    family: dict[str, object],
) -> dict[str, tuple[ConditionedRangeEscapeParams, set[str]]]:
    result: dict[str, tuple[ConditionedRangeEscapeParams, set[str]]] = {}

    def add(params: ConditionedRangeEscapeParams, source: str) -> None:
        if params.candidate_id not in result:
            result[params.candidate_id] = (params, set())
        result[params.candidate_id][1].add(f"{family_name}:{source}")

    add(_params(family), "center")
    gate = family["gate_surface"]
    for quality_window, absolute, relative in product(
        gate["quality_windows"],
        gate["absolute_floors"],
        gate["relative_floors"],
    ):
        add(
            _params(
                family,
                {
                    "quality_window": quality_window,
                    "absolute_floor": absolute,
                    "relative_floor": relative,
                },
            ),
            "gate_surface",
        )
    cube = family["local_cube"]
    for quality_window, absolute, relative, hold_days, weight in product(
        cube["quality_windows"],
        cube["absolute_floors"],
        cube["relative_floors"],
        cube["hold_days"],
        cube["momentum_weights"],
    ):
        add(
            _params(
                family,
                {
                    "quality_window": quality_window,
                    "absolute_floor": absolute,
                    "relative_floor": relative,
                    "hold_days": hold_days,
                    "momentum_weight": weight,
                },
            ),
            "local_cube",
        )
    ranges = family["range_surface"]
    for range_window, upper in product(
        ranges["range_windows"], ranges["upper_thresholds"]
    ):
        add(
            _params(
                family,
                {"range_window": range_window, "upper_threshold": upper},
            ),
            "range_surface",
        )
    return result


def _one_factor_summary(
    row: pd.Series,
    family_surface: pd.DataFrame,
) -> dict[str, float | int]:
    fields = [
        "range_window",
        "upper_threshold",
        "momentum_weight",
        "quality_window",
        "absolute_floor",
        "relative_floor",
        "hold_days",
    ]
    differences = pd.Series(0, index=family_surface.index, dtype=int)
    changed: dict[str, pd.Series] = {}
    for field in fields:
        selected = row[field]
        values = family_surface[field]
        if pd.isna(selected):
            changed[field] = values.notna()
        else:
            changed[field] = ~values.fillna(np.inf).eq(selected)
        differences += changed[field].astype(int)
    neighbors = family_surface.loc[differences.le(1)].drop_duplicates("candidate_id")
    covered_axes = sum(
        bool((changed[field] & differences.eq(1)).any()) for field in fields
    )
    return {
        "neighbor_count": int(len(neighbors)),
        "neighbor_covered_axes": int(covered_axes),
        "neighbor_full_dual_rate": float(neighbors["full_dual"].mean()),
        "neighbor_validation_dual_rate": float(neighbors["validation_dual"].mean()),
        "neighbor_recent_dual_rate": float(neighbors["recent_dual"].mean()),
        "neighbor_annualized_delta_q25": float(
            neighbors["delta_annualized_return_252"].quantile(0.25)
        ),
        "neighbor_sharpe_delta_q25": float(
            neighbors["delta_sharpe"].quantile(0.25)
        ),
        "neighbor_validation_annualized_delta_q25": float(
            neighbors["validation_delta_annualized_return_252"].quantile(0.25)
        ),
        "neighbor_validation_sharpe_delta_q25": float(
            neighbors["validation_delta_sharpe"].quantile(0.25)
        ),
    }


def run_research(root: Path, config_path: Path, output: Path) -> dict[str, object]:
    applied = config_path if config_path.is_absolute() else root / config_path
    config = yaml.safe_load(applied.read_text(encoding="utf-8"))
    experiment = config["experiment"]
    start = date.fromisoformat(str(experiment["evaluation_start"]))
    end = date.fromisoformat(str(experiment["evidence_cutoff"]))
    periods = _periods(config)
    checks = config["overfit_checks"]

    context = build_range_escape_context(root, start=start, end=end)
    cache = build_weight_execution_cache(context)
    baseline = context.formal.daily["return"].astype(float)
    direct_baseline = execute_targets_cached(
        context, context.baseline_targets, cache
    )["return"].astype(float)
    location_cache: dict[int, pd.DataFrame] = {}
    quality_cache: dict[int, pd.DataFrame] = {}

    params_by_id: dict[str, ConditionedRangeEscapeParams] = {}
    sources: dict[str, set[str]] = {}
    family_by_id: dict[str, set[str]] = {}
    for family_name, family in config["families"].items():
        for candidate_id, (params, candidate_sources) in _family_params(
            str(family_name), family
        ).items():
            params_by_id[candidate_id] = params
            sources.setdefault(candidate_id, set()).update(candidate_sources)
            family_by_id.setdefault(candidate_id, set()).add(str(family_name))

    records = []
    returns: dict[str, pd.Series] = {}
    runs = {}
    candidates = list(params_by_id.values())
    for index, params in enumerate(candidates, start=1):
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
        record = _metric_record(
            run,
            periods,
            stage="focused",
            seed_source="|".join(sorted(sources[params.candidate_id])),
        )
        record["focused_families"] = "|".join(sorted(family_by_id[params.candidate_id]))
        records.append(record)
        if index % 100 == 0:
            print(f"focused {index}/{len(candidates)}", flush=True)

    surface = _add_deltas(pd.DataFrame(records), baseline, periods)
    selection = config["robust_selection"]
    preliminary = surface.loc[
        surface["development_dual"]
        & surface["validation_dual"]
        & surface["recent_dual"]
        & surface["full_dual"]
        & surface["delta_max_drawdown"].ge(-float(selection["maximum_mdd_worsening"]))
        & surface["escape_entries"].ge(int(selection["minimum_escape_entries"]))
    ].copy()
    neighbor_rows = []
    for _, row in preliminary.iterrows():
        family_name = str(row["focused_families"]).split("|")[0]
        family_surface = surface.loc[
            surface["focused_families"].str.contains(family_name, regex=False)
        ]
        neighbor_rows.append(
            {
                "candidate_id": row["candidate_id"],
                **_one_factor_summary(row, family_surface),
            }
        )
    neighbors = pd.DataFrame(neighbor_rows)
    if neighbors.empty:
        neighbors = pd.DataFrame(
            columns=[
                "candidate_id",
                "neighbor_count",
                "neighbor_covered_axes",
                "neighbor_full_dual_rate",
                "neighbor_validation_dual_rate",
                "neighbor_recent_dual_rate",
                "neighbor_annualized_delta_q25",
                "neighbor_sharpe_delta_q25",
                "neighbor_validation_annualized_delta_q25",
                "neighbor_validation_sharpe_delta_q25",
            ]
        )
    evaluated = preliminary.merge(neighbors, on="candidate_id", how="left")
    robust = evaluated.loc[
        evaluated["neighbor_covered_axes"].ge(
            int(selection["minimum_covered_neighbor_axes"])
        )
        & evaluated["neighbor_full_dual_rate"].ge(
            float(selection["minimum_one_factor_full_dual_rate"])
        )
        & evaluated["neighbor_validation_dual_rate"].ge(
            float(selection["minimum_one_factor_validation_dual_rate"])
        )
        & evaluated["neighbor_recent_dual_rate"].ge(
            float(selection["minimum_one_factor_recent_dual_rate"])
        )
        & evaluated["neighbor_annualized_delta_q25"].gt(0.0)
        & evaluated["neighbor_sharpe_delta_q25"].gt(0.0)
        & evaluated["neighbor_validation_annualized_delta_q25"].gt(0.0)
        & evaluated["neighbor_validation_sharpe_delta_q25"].gt(0.0)
    ].copy()

    rank_source = robust if not robust.empty else (evaluated if not evaluated.empty else surface)
    rank_source = rank_source.copy()
    rank_fields = [
        "annualized_return_252",
        "sharpe",
        "development_sharpe",
        "validation_sharpe",
        "recent_sharpe",
    ]
    if "neighbor_annualized_delta_q25" in rank_source:
        rank_fields.extend(["neighbor_annualized_delta_q25", "neighbor_sharpe_delta_q25"])
    rank_source["rank_score"] = rank_source[rank_fields].rank(
        pct=True, method="average"
    ).mean(axis=1)
    selected_row = rank_source.sort_values(
        ["rank_score", "candidate_id"], ascending=[False, True]
    ).iloc[0]
    selected_id = str(selected_row["candidate_id"])
    selected_returns = returns[selected_id]
    selected_params = runs[selected_id].params

    broad_path = Path(str(experiment["broad_output"]))
    if not broad_path.is_absolute():
        broad_path = root / broad_path
    broad_returns = pd.read_parquet(broad_path / "daily_returns.parquet")
    combined = {str(column): broad_returns[column] for column in broad_returns.columns}
    combined.update(returns)
    unique: dict[str, pd.Series] = {}
    seen: set[str] = set()
    for candidate_id, candidate_returns in combined.items():
        digest = _return_hash(candidate_returns.astype(float))
        if digest not in seen:
            seen.add(digest)
            unique[candidate_id] = candidate_returns.astype(float)
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
    bootstrap_frame, bootstrap = paired_block_bootstrap(
        selected_returns,
        baseline,
        block_size=int(checks["paired_block_bootstrap_block"]),
        repetitions=int(checks["paired_block_bootstrap_repetitions"]),
        seed=int(checks["random_seed"]),
    )
    fixed_leave_year = _fixed_leave_one_year(selected_returns, baseline)
    annual = _calendar_year_comparison(selected_returns, baseline)
    rolling = _rolling_comparison(
        selected_returns,
        baseline,
        [int(value) for value in checks["rolling_windows"]],
    )
    events = _difference_events(selected_returns, direct_baseline)
    leave_event = _leave_one_event(selected_returns, direct_baseline, events)

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
            selected_params,
            locations_at_open=location_cache[selected_params.range_window],
            quality_metrics_at_open=quality_cache[selected_params.quality_window],
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

    selected_is_robust = selected_id in set(robust["candidate_id"])
    audit = {
        "research_id": experiment["id"],
        "status": "completed_research_only",
        "evidence_status": experiment["evidence_status"],
        "baseline_strategy_id": experiment["baseline_strategy_id"],
        "baseline_metrics": performance(baseline),
        "focused_candidate_ids": int(len(surface)),
        "combined_unique_paths": int(len(panel.columns)),
        "preliminary_candidates": int(len(preliminary)),
        "robust_candidates": int(len(robust)),
        "selected_candidate": selected_id,
        "selected_is_robust": selected_is_robust,
        "selected_metrics": {
            key: selected_row[key]
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
        "selected_params": {
            **asdict(selected_params),
            "gate_family": selected_params.gate.family,
            "gate_id": selected_params.gate.gate_id,
        },
        "selected_neighborhood": {
            key: selected_row[key]
            for key in selected_row.index
            if str(key).startswith("neighbor_")
        },
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
        "fixed_selected_delete_year_dual_pass_rate": float(
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
            "robust_research_candidate_requires_explicit_promotion"
            if selected_is_robust
            else "reject_no_robust_candidate"
        ),
        "production_changed": False,
    }

    output.mkdir(parents=True, exist_ok=True)
    surface.to_csv(output / "focused_surface.csv", index=False)
    preliminary.to_csv(output / "preliminary_candidates.csv", index=False)
    neighbors.to_csv(output / "neighbor_summary.csv", index=False)
    robust.to_csv(output / "robust_candidates.csv", index=False)
    pd.DataFrame(returns, index=context.calendar).to_parquet(
        output / "focused_daily_returns.parquet"
    )
    runs[selected_id].state.to_parquet(output / "selected_state.parquet")
    runs[selected_id].targets.to_parquet(output / "selected_targets.parquet")
    runs[selected_id].daily.to_parquet(output / "selected_daily.parquet")
    bootstrap_frame.to_csv(output / "paired_bootstrap.csv", index=False)
    cscv_frame.to_csv(output / "cscv.csv", index=False)
    walk_forward.to_csv(output / "walk_forward.csv", index=False)
    leave_year_selection.to_csv(output / "leave_one_year_selection.csv", index=False)
    fixed_leave_year.to_csv(output / "fixed_selected_leave_one_year.csv", index=False)
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
        selected_returns,
        baseline,
        str(experiment["baseline_strategy_id"]),
        output / "selected_vs_formal.html",
        {
            "strategy_name": selected_id,
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
