"""Two-stage robust search for conditioned Defender-to-Momentum escape."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, replace
from datetime import date
from itertools import product
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yaml

from research.audit_current_strategy_occam_robustness import _metric_row, _periods
from research.audit_defender_selector_2019 import _difference_events, _leave_one_event
from research.audit_momentum_hold_2019_followup import (
    _calendar_year_comparison,
    _fixed_leave_one_year,
    _rolling_comparison,
)
from research.momentum_defender_conditioned_range_escape import (
    HOLD_FIXED_PULSE_REARM,
    HOLD_MINIMUM_UNTIL_FAIL,
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
from research.standard_report import generate_standard_report


DEFAULT_CONFIG = Path(
    "research/configs/conditioned_defender_range_escape_2019.yaml"
)
DEFAULT_OUTPUT = Path(
    "experiments/20260827_conditioned_defender_range_escape_2019"
)


def _return_hash(returns: pd.Series) -> str:
    return hashlib.sha256(returns.to_numpy(dtype="<f8").tobytes()).hexdigest()


def _json_default(value: object) -> object:
    if isinstance(value, (pd.Timestamp, date)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"cannot encode {type(value).__name__}")


def _gates(settings: dict[str, object]) -> list[MomentumQualityGate]:
    result: list[MomentumQualityGate] = []
    if bool(settings["include_no_gate"]):
        result.append(MomentumQualityGate())
    result.extend(
        MomentumQualityGate(absolute_floor=float(value))
        for value in settings["absolute_floors"]
    )
    result.extend(
        MomentumQualityGate(relative_floor=float(value))
        for value in settings["relative_floors"]
    )
    result.extend(
        MomentumQualityGate(
            absolute_floor=float(absolute), relative_floor=float(relative)
        )
        for absolute, relative in product(
            settings["joint_absolute_floors"],
            settings["joint_relative_floors"],
        )
    )
    return result


def _stage1_params(config: dict[str, object]) -> list[ConditionedRangeEscapeParams]:
    settings = config["stage1_structure"]
    gates = _gates(settings)
    result = []
    for anchor, weight, policy, hold_days, gate in product(
        settings["anchor_modes"],
        settings["momentum_weights"],
        settings["hold_policies"],
        settings["hold_days"],
        gates,
    ):
        result.append(
            ConditionedRangeEscapeParams(
                anchor_mode=str(anchor),
                range_window=int(settings["range_window"]),
                upper_threshold=float(settings["upper_threshold"]),
                momentum_weight=float(weight),
                quality_window=int(settings["quality_window"]),
                gate=gate,
                hold_policy=str(policy),
                hold_days=int(hold_days),
            )
        )
    return result


def _metric_record(
    run,
    periods: dict[str, tuple[str, str]],
    *,
    stage: str,
    seed_source: str = "",
) -> dict[str, object]:
    params = run.params
    return {
        **_metric_row(
            params.candidate_id,
            "conditioned_defender_range_escape",
            run.daily["return"].astype(float),
            periods,
        ),
        **{
            key: value
            for key, value in asdict(params).items()
            if key != "gate"
        },
        "absolute_floor": params.gate.absolute_floor,
        "relative_floor": params.gate.relative_floor,
        "gate_family": params.gate.family,
        "gate_id": params.gate.gate_id,
        "stage": stage,
        "seed_source": seed_source,
        "escape_entries": run.audit["escape_entries"],
        "escape_days": run.audit["escape_days"],
        "escape_asset_rotations": run.audit["escape_asset_rotations"],
        "average_momentum_weight": run.audit[
            "average_momentum_weight_on_formal_defender_days"
        ],
    }


def _add_deltas(
    surface: pd.DataFrame,
    baseline: pd.Series,
    periods: dict[str, tuple[str, str]],
) -> pd.DataFrame:
    result = surface.copy()
    full = performance(baseline)
    for field in ("annualized_return_252", "sharpe", "max_drawdown"):
        result[f"delta_{field}"] = result[field] - full[field]
    for name, bounds in periods.items():
        base = performance(baseline.loc[bounds[0] : bounds[1]])
        for field in ("annualized_return_252", "sharpe", "max_drawdown"):
            result[f"{name}_delta_{field}"] = (
                result[f"{name}_{field}"] - base[field]
            )
    for name in ("development", "validation", "recent", "complete_pool"):
        result[f"{name}_dual"] = (
            result[f"{name}_delta_annualized_return_252"].gt(0.0)
            & result[f"{name}_delta_sharpe"].gt(0.0)
        )
    result["full_dual"] = (
        result["delta_annualized_return_252"].gt(0.0)
        & result["delta_sharpe"].gt(0.0)
    )
    return result


def _rank_development(
    stage1: pd.DataFrame,
    config: dict[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selection = config["development_selection"]
    eligible = stage1.loc[
        stage1["escape_entries"].ge(int(selection["minimum_escape_entries"]))
        & stage1["development_delta_annualized_return_252"].gt(0.0)
        & stage1["development_delta_sharpe"].gt(0.0)
        & stage1["development_delta_max_drawdown"].ge(
            -float(selection["maximum_development_mdd_worsening"])
        )
    ].copy()
    source = stage1.copy()
    fields = [str(value) for value in selection["ranking_fields"]]
    source["development_rank_score"] = source[fields].rank(
        pct=True, method="average"
    ).mean(axis=1)
    eligible_ids = set(eligible["candidate_id"])
    source["development_eligible"] = source["candidate_id"].isin(eligible_ids)
    group_fields = [str(value) for value in selection["seed_group_fields"]]
    seeds = (
        source.sort_values(
            ["development_eligible", "development_rank_score", "candidate_id"],
            ascending=[False, False, True],
        )
        .groupby(group_fields, as_index=False, group_keys=False)
        .head(int(selection["seeds_per_group"]))
        .copy()
    )
    return source, seeds


def _neighbor_values(values: Iterable[float], selected: float) -> list[float]:
    ordered = sorted({float(value) for value in values})
    position = min(range(len(ordered)), key=lambda index: abs(ordered[index] - selected))
    return ordered[max(0, position - 1) : min(len(ordered), position + 2)]


def _gate_neighbors(
    seed: ConditionedRangeEscapeParams,
    stage1_settings: dict[str, object],
) -> list[MomentumQualityGate]:
    absolute = seed.gate.absolute_floor
    relative = seed.gate.relative_floor
    absolute_values = (
        [None]
        if absolute is None
        else _neighbor_values(stage1_settings["absolute_floors"], absolute)
    )
    relative_values = (
        [None]
        if relative is None
        else _neighbor_values(stage1_settings["relative_floors"], relative)
    )
    return [
        MomentumQualityGate(
            absolute_floor=None if a is None else float(a),
            relative_floor=None if r is None else float(r),
        )
        for a, r in product(absolute_values, relative_values)
    ]


def _params_from_row(row: pd.Series) -> ConditionedRangeEscapeParams:
    absolute = row["absolute_floor"]
    relative = row["relative_floor"]
    return ConditionedRangeEscapeParams(
        anchor_mode=str(row["anchor_mode"]),
        range_window=int(row["range_window"]),
        upper_threshold=float(row["upper_threshold"]),
        momentum_weight=float(row["momentum_weight"]),
        quality_window=int(row["quality_window"]),
        gate=MomentumQualityGate(
            absolute_floor=None if pd.isna(absolute) else float(absolute),
            relative_floor=None if pd.isna(relative) else float(relative),
        ),
        hold_policy=str(row["hold_policy"]),
        hold_days=int(row["hold_days"]),
    )


def _expand_seed(
    seed: ConditionedRangeEscapeParams,
    config: dict[str, object],
) -> set[ConditionedRangeEscapeParams]:
    settings = config["stage2_neighborhood"]
    result = {seed}
    ranges = [int(value) for value in settings["range_windows"]]
    highs = [float(value) for value in settings["upper_thresholds"]]
    qualities = [int(value) for value in settings["quality_windows"]]
    weights = [float(value) for value in settings["momentum_weights"]]
    holds = [int(value) for value in settings["hold_days"]]
    for value in ranges:
        result.add(replace(seed, range_window=value))
    for value in highs:
        result.add(replace(seed, upper_threshold=value))
    for value in qualities:
        result.add(replace(seed, quality_window=value))
    for value in weights:
        result.add(replace(seed, momentum_weight=value))
    for value in holds:
        result.add(replace(seed, hold_days=value))
    for range_window, upper_threshold in product(ranges, highs):
        result.add(
            replace(
                seed,
                range_window=range_window,
                upper_threshold=upper_threshold,
            )
        )
    for momentum_weight, hold_days in product(weights, holds):
        result.add(
            replace(
                seed,
                momentum_weight=momentum_weight,
                hold_days=hold_days,
            )
        )
    for gate in _gate_neighbors(seed, config["stage1_structure"]):
        result.add(replace(seed, gate=gate))
    return result


def _one_factor_neighbor_summary(
    candidate: pd.Series,
    surface: pd.DataFrame,
) -> dict[str, float | int]:
    parameter_fields = [
        "range_window",
        "upper_threshold",
        "momentum_weight",
        "quality_window",
        "absolute_floor",
        "relative_floor",
        "hold_days",
    ]
    fixed_fields = ["anchor_mode", "hold_policy", "gate_family"]
    pool = surface.copy()
    for field in fixed_fields:
        pool = pool.loc[pool[field].eq(candidate[field])]
    differences = pd.Series(0, index=pool.index, dtype=int)
    for field in parameter_fields:
        left = pool[field]
        right = candidate[field]
        if pd.isna(right):
            differences += left.notna().astype(int)
        else:
            differences += (~left.fillna(np.inf).eq(right)).astype(int)
    neighbors = pool.loc[differences.le(1)].drop_duplicates("candidate_id")
    return {
        "neighbor_count": int(len(neighbors)),
        "neighbor_full_dual_rate": float(neighbors["full_dual"].mean()),
        "neighbor_validation_dual_rate": float(neighbors["validation_dual"].mean()),
        "neighbor_recent_dual_rate": float(neighbors["recent_dual"].mean()),
        "neighbor_annualized_delta_q25": float(
            neighbors["delta_annualized_return_252"].quantile(0.25)
        ),
        "neighbor_sharpe_delta_q25": float(
            neighbors["delta_sharpe"].quantile(0.25)
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
    if float((direct_baseline - context.direct_baseline_daily["return"]).abs().max()) > 1e-14:
        raise AssertionError("cached baseline does not match exact interface baseline")
    location_cache: dict[int, pd.DataFrame] = {}
    quality_cache: dict[int, pd.DataFrame] = {}

    def evaluate(params: ConditionedRangeEscapeParams):
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
        return run_conditioned_range_escape(
            context,
            params,
            locations_at_open=location_cache[params.range_window],
            quality_metrics_at_open=quality_cache[params.quality_window],
            execution_cache=cache,
        )

    runs = {}
    returns: dict[str, pd.Series] = {}
    records: list[dict[str, object]] = []
    stage1_params = _stage1_params(config)
    for index, params in enumerate(stage1_params, start=1):
        run = evaluate(params)
        runs[params.candidate_id] = run
        returns[params.candidate_id] = run.daily["return"].astype(float)
        records.append(_metric_record(run, periods, stage="stage1"))
        if index % 100 == 0:
            print(f"stage1 {index}/{len(stage1_params)}", flush=True)

    stage1_surface = _add_deltas(pd.DataFrame(records), baseline, periods)
    ranked_stage1, seeds = _rank_development(stage1_surface, config)
    seed_params = [_params_from_row(row) for _, row in seeds.iterrows()]
    stage2_sources: dict[str, set[str]] = {}
    stage2_params: dict[str, ConditionedRangeEscapeParams] = {}
    for seed in seed_params:
        for params in _expand_seed(seed, config):
            stage2_params[params.candidate_id] = params
            stage2_sources.setdefault(params.candidate_id, set()).add(seed.candidate_id)

    stage2_new = [
        params
        for candidate_id, params in stage2_params.items()
        if candidate_id not in runs
    ]
    for index, params in enumerate(stage2_new, start=1):
        run = evaluate(params)
        runs[params.candidate_id] = run
        returns[params.candidate_id] = run.daily["return"].astype(float)
        records.append(
            _metric_record(
                run,
                periods,
                stage="stage2",
                seed_source="|".join(sorted(stage2_sources[params.candidate_id])),
            )
        )
        if index % 100 == 0:
            print(f"stage2 {index}/{len(stage2_new)}", flush=True)

    surface = _add_deltas(pd.DataFrame(records), baseline, periods)
    stage1_ids = set(ranked_stage1["candidate_id"])
    surface.loc[surface["candidate_id"].isin(stage1_ids), "development_eligible"] = (
        surface.loc[surface["candidate_id"].isin(stage1_ids), "candidate_id"].map(
            ranked_stage1.set_index("candidate_id")["development_eligible"]
        )
    )
    for _, row in seeds.iterrows():
        surface.loc[
            surface["candidate_id"].eq(row["candidate_id"]), "development_seed"
        ] = True
    surface["development_seed"] = surface["development_seed"].fillna(False)

    final = config["final_selection"]
    preliminary = surface.loc[
        surface["development_dual"]
        & surface["validation_dual"]
        & surface["full_dual"]
        & surface["recent_dual"]
        & surface["delta_max_drawdown"].ge(-float(final["maximum_mdd_worsening"]))
        & surface["escape_entries"].ge(int(final["minimum_escape_entries"]))
    ].copy()
    neighbor_rows = []
    for _, row in preliminary.iterrows():
        neighbor_rows.append(
            {"candidate_id": row["candidate_id"], **_one_factor_neighbor_summary(row, surface)}
        )
    neighbor_summary = pd.DataFrame(neighbor_rows)
    if neighbor_summary.empty:
        neighbor_summary = pd.DataFrame(
            columns=[
                "candidate_id",
                "neighbor_count",
                "neighbor_full_dual_rate",
                "neighbor_validation_dual_rate",
                "neighbor_recent_dual_rate",
                "neighbor_annualized_delta_q25",
                "neighbor_sharpe_delta_q25",
            ]
        )
    final_pool = preliminary.merge(neighbor_summary, on="candidate_id", how="left")
    robust = final_pool.loc[
        final_pool["neighbor_full_dual_rate"].ge(
            float(final["minimum_one_factor_neighbor_full_dual_rate"])
        )
        & final_pool["neighbor_validation_dual_rate"].ge(
            float(final["minimum_one_factor_neighbor_validation_dual_rate"])
        )
        & final_pool["neighbor_annualized_delta_q25"].gt(0.0)
        & final_pool["neighbor_sharpe_delta_q25"].gt(0.0)
    ].copy()

    ranking_source = robust if not robust.empty else (preliminary if not preliminary.empty else surface)
    rank_fields = [
        "annualized_return_252",
        "sharpe",
        "development_sharpe",
        "validation_sharpe",
        "recent_sharpe",
    ]
    ranking_source = ranking_source.copy()
    ranking_source["final_rank_score"] = ranking_source[rank_fields].rank(
        pct=True, method="average"
    ).mean(axis=1)
    selected_row = ranking_source.sort_values(
        ["final_rank_score", "candidate_id"], ascending=[False, True]
    ).iloc[0]
    selected_id = str(selected_row["candidate_id"])
    selected_returns = returns[selected_id]

    unique_returns: dict[str, pd.Series] = {}
    seen: set[str] = set()
    for candidate_id, candidate_returns in returns.items():
        digest = _return_hash(candidate_returns)
        if digest not in seen:
            seen.add(digest)
            unique_returns[candidate_id] = candidate_returns
    panel = pd.DataFrame(unique_returns, index=context.calendar)
    cscv_frame, cscv = cscv_pbo(
        panel, baseline, block_count=int(checks["cscv_blocks"])
    )
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

    selected_params = runs[selected_id].params
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

    decision = (
        "robust_research_candidate_requires_explicit_promotion"
        if not robust.empty
        else "reject_no_robust_candidate"
    )
    audit = {
        "research_id": experiment["id"],
        "status": "completed_research_only",
        "evidence_status": experiment["evidence_status"],
        "evaluation_start": start.isoformat(),
        "evidence_cutoff": end.isoformat(),
        "baseline_strategy_id": experiment["baseline_strategy_id"],
        "baseline_return_hash": _return_hash(baseline),
        "cached_baseline_parity_max_abs_error": float(
            (direct_baseline - context.direct_baseline_daily["return"]).abs().max()
        ),
        "baseline_metrics": performance(baseline),
        "stage1_candidates": int(len(stage1_params)),
        "development_eligible_candidates": int(ranked_stage1["development_eligible"].sum()),
        "development_seeds": int(len(seeds)),
        "stage2_new_candidates": int(len(stage2_new)),
        "total_candidate_ids": int(len(surface)),
        "unique_paths": int(len(panel.columns)),
        "preliminary_full_segment_candidates": int(len(preliminary)),
        "robust_candidates": int(len(robust)),
        "selected_candidate": selected_id,
        "selected_is_robust": bool(selected_id in set(robust["candidate_id"])),
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
        "decision": decision,
        "production_changed": False,
    }

    output.mkdir(parents=True, exist_ok=True)
    ranked_stage1.to_csv(output / "stage1_surface.csv", index=False)
    seeds.to_csv(output / "development_seeds.csv", index=False)
    surface.to_csv(output / "all_candidate_surface.csv", index=False)
    preliminary.to_csv(output / "preliminary_candidates.csv", index=False)
    neighbor_summary.to_csv(output / "neighbor_summary.csv", index=False)
    robust.to_csv(output / "robust_candidates.csv", index=False)
    pd.DataFrame(returns, index=context.calendar).to_parquet(
        output / "daily_returns.parquet"
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
