"""Test equal-weight multi-horizon QM gates for the fixed five-day pulse."""

from __future__ import annotations

import argparse
import json
from datetime import date
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from research.audit_current_strategy_occam_robustness import _periods
from research.momentum_defender_conditioned_range_escape import (
    ConditionedRangeEscapeParams,
    MomentumQualityGate,
    aggregate_quality_metrics,
    build_weight_execution_cache,
    run_conditioned_range_escape,
)
from research.momentum_defender_defender_range_escape import (
    build_range_escape_context,
    range_locations_at_open,
)
from research.momentum_defender_gold_override import QUALITY_METRIC
from research.momentum_defender_occam import performance
from research.momentum_top1_defender_escape import all_metrics_at_open
from research.run_conditioned_defender_range_escape_2019 import (
    _add_deltas,
    _json_default,
    _metric_record,
)
from research.standard_report import generate_standard_report


DEFAULT_CONFIG = Path(
    "research/configs/conditioned_defender_range_escape_ensemble.yaml"
)
DEFAULT_OUTPUT = Path(
    "experiments/20260827_conditioned_defender_range_escape_ensemble"
)


def run_research(root: Path, config_path: Path, output: Path) -> dict[str, object]:
    applied = config_path if config_path.is_absolute() else root / config_path
    config = yaml.safe_load(applied.read_text(encoding="utf-8"))
    experiment = config["experiment"]
    start = date.fromisoformat(str(experiment["evaluation_start"]))
    end = date.fromisoformat(str(experiment["evidence_cutoff"]))
    periods = _periods(config)
    structure = config["frozen_structure"]

    context = build_range_escape_context(root, start=start, end=end)
    cache = build_weight_execution_cache(context)
    baseline = context.formal.daily["return"].astype(float)
    locations = range_locations_at_open(context, int(structure["range_window"]))
    all_windows = sorted(
        {int(window) for values in config["profiles"].values() for window in values}
    )
    panels = {
        window: all_metrics_at_open(
            context.formal.context.curves, QUALITY_METRIC, window
        )
        for window in all_windows
    }

    records = []
    returns = {}
    runs = {}
    profile_surfaces = {}
    candidates = []
    for profile_name, windows_value in config["profiles"].items():
        windows = tuple(int(value) for value in windows_value)
        for aggregation, relative_floor in product(
            config["aggregations"], config["relative_floors"]
        ):
            params = ConditionedRangeEscapeParams(
                anchor_mode=str(structure["anchor_mode"]),
                range_window=int(structure["range_window"]),
                upper_threshold=float(structure["upper_threshold"]),
                momentum_weight=float(structure["momentum_weight"]),
                quality_window=40,
                gate=MomentumQualityGate(
                    absolute_floor=float(structure["absolute_floor"]),
                    relative_floor=float(relative_floor),
                ),
                hold_policy=str(structure["hold_policy"]),
                hold_days=int(structure["hold_days"]),
                quality_windows=windows,
                quality_aggregation=str(aggregation),
            )
            candidates.append((str(profile_name), params))
            profile_surfaces[(str(profile_name), str(aggregation))] = (
                aggregate_quality_metrics(panels, windows, str(aggregation))
            )

    for profile_name, params in candidates:
        metrics = profile_surfaces[(profile_name, params.quality_aggregation)]
        run = run_conditioned_range_escape(
            context,
            params,
            locations_at_open=locations,
            quality_metrics_at_open=metrics,
            execution_cache=cache,
        )
        runs[params.candidate_id] = run
        returns[params.candidate_id] = run.daily["return"].astype(float)
        record = _metric_record(run, periods, stage="qm_ensemble")
        record["profile_name"] = profile_name
        record["profile_size"] = len(params.quality_windows)
        records.append(record)

    surface = _add_deltas(pd.DataFrame(records), baseline, periods)
    selection = config["selection"]
    eligible = surface.loc[
        surface["development_dual"]
        & surface["validation_dual"]
        & surface["recent_dual"]
        & surface["full_dual"]
        & surface["escape_entries"].ge(int(selection["minimum_escape_entries"]))
    ].copy()
    source = eligible if not eligible.empty else surface.copy()
    aggregation_complexity = {"mean": 0, "median": 1, "minimum": 2}
    source["aggregation_complexity"] = source["quality_aggregation"].map(
        aggregation_complexity
    )
    source["minimum_segment_sharpe_delta"] = source[
        [
            "development_delta_sharpe",
            "validation_delta_sharpe",
            "recent_delta_sharpe",
        ]
    ].min(axis=1)
    selected_row = source.sort_values(
        [
            "minimum_segment_sharpe_delta",
            "profile_size",
            "aggregation_complexity",
            "candidate_id",
        ],
        ascending=[False, True, True, True],
    ).iloc[0]
    selected_id = str(selected_row["candidate_id"])

    group_rows = []
    for (profile_name, aggregation), sample in surface.groupby(
        ["profile_name", "quality_aggregation"]
    ):
        group_rows.append(
            {
                "profile_name": profile_name,
                "aggregation": aggregation,
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
    groups = pd.DataFrame(group_rows)
    audit = {
        "research_id": experiment["id"],
        "status": "completed_research_only",
        "evidence_status": experiment["evidence_status"],
        "baseline_metrics": performance(baseline),
        "candidate_ids": int(len(surface)),
        "eligible_candidates": int(len(eligible)),
        "selected_candidate": selected_id,
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
        "selected_profile": {
            "profile_name": selected_row["profile_name"],
            "quality_windows": selected_row["quality_windows"],
            "aggregation": selected_row["quality_aggregation"],
            "relative_floor": selected_row["relative_floor"],
        },
        "production_changed": False,
    }

    output.mkdir(parents=True, exist_ok=True)
    surface.to_csv(output / "ensemble_surface.csv", index=False)
    groups.to_csv(output / "ensemble_group_summary.csv", index=False)
    eligible.to_csv(output / "eligible_candidates.csv", index=False)
    pd.DataFrame(returns, index=context.calendar).to_parquet(
        output / "ensemble_daily_returns.parquet"
    )
    runs[selected_id].state.to_parquet(output / "selected_state.parquet")
    runs[selected_id].targets.to_parquet(output / "selected_targets.parquet")
    runs[selected_id].daily.to_parquet(output / "selected_daily.parquet")
    (output / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    (output / "applied_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    generate_standard_report(
        returns[selected_id],
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
