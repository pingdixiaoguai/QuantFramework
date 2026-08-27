"""Decompose Gold-exception candidates into base-gate and overlay effects."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.momentum_defender_downside_raqm import (
    build_exact_execution_data,
    exact_candidate_schedule,
)
from research.momentum_defender_gold_override import build_gold_override_context
from research.momentum_defender_occam import performance


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "experiments/20260825_momentum_defender_gold_exception_search"
UNIVERSAL = ROOT / "experiments/20260824_momentum_defender_downside_raqm_final_selection"
OUTPUT = ROOT / "experiments/20260825_gold_exception_underperformance"


def _compound(values):
    return float(np.expm1(np.log1p(pd.Series(values).astype(float)).sum()))


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    context = build_gold_override_context(ROOT, end=pd.Timestamp("2026-08-21").date())
    data = build_exact_execution_data(context)
    universal = pd.read_parquet(UNIVERSAL / "selected_daily.parquet")
    path_rows = []
    yearly_rows = []
    event_frames = []
    for label in ("including_extremes", "excluding_extremes"):
        selected = pd.read_parquet(SOURCE / f"selected_{label}_daily.parquet")
        requested = np.where(
            selected["base_risk_on"].to_numpy(bool),
            data.momentum_target,
            data.candidate_index["DEFENDER"],
        ).astype(int)
        base_values, base_actual, base_switches = exact_candidate_schedule(data, requested)
        base = pd.Series(base_values, index=selected.index)
        base_target = pd.Series(
            [data.candidates[value] for value in base_actual], index=selected.index
        )
        paths = {
            "universal": universal["return"],
            "candidate_base_only": base,
            "candidate_with_gold": selected["return"],
        }
        for name, returns in paths.items():
            path_rows.append(
                {
                    "selection": label,
                    "path": name,
                    **performance(returns),
                    "candidate_switches": (
                        base_switches
                        if name == "candidate_base_only"
                        else int(
                            (
                                (universal if name == "universal" else selected)[
                                    "actual_candidate"
                                ].ne(
                                    (universal if name == "universal" else selected)[
                                        "actual_candidate"
                                    ].shift()
                                )
                            ).sum()
                            - 1
                        )
                    ),
                }
            )
        gold_component = np.log1p(selected["return"]) - np.log1p(base)
        base_component = np.log1p(base) - np.log1p(universal["return"])
        for year in sorted(selected.index.year.unique()):
            mask = selected.index.year == year
            yearly_rows.append(
                {
                    "selection": label,
                    "year": int(year),
                    "gold_overlay_log_excess_vs_own_base": float(
                        gold_component.loc[mask].sum()
                    ),
                    "base_log_excess_vs_universal": float(
                        base_component.loc[mask].sum()
                    ),
                    "total_log_excess_vs_universal": float(
                        (gold_component.loc[mask] + base_component.loc[mask]).sum()
                    ),
                }
            )
        different = selected["actual_candidate"].astype(str).ne(base_target)
        groups = different.ne(different.shift()).cumsum()
        rows = []
        for event, (_, sample) in enumerate(
            selected.loc[different].groupby(groups.loc[different]), start=1
        ):
            start = selected.index.get_loc(sample.index.min())
            end = min(selected.index.get_loc(sample.index.max()) + 1, len(selected) - 1)
            index = selected.index[start : end + 1]
            candidate_return = _compound(selected.loc[index, "return"])
            base_return = _compound(base.loc[index])
            rows.append(
                {
                    "selection": label,
                    "event": event,
                    "start": index.min().date().isoformat(),
                    "end_including_exit": index.max().date().isoformat(),
                    "observations": len(index),
                    "candidate_return": candidate_return,
                    "base_return": base_return,
                    "log_excess": float(
                        np.log1p(candidate_return) - np.log1p(base_return)
                    ),
                    "candidate_targets": ";".join(
                        f"{key}:{value}"
                        for key, value in selected.loc[
                            index, "actual_candidate"
                        ].value_counts().items()
                    ),
                    "base_targets": ";".join(
                        f"{key}:{value}"
                        for key, value in base_target.loc[index].value_counts().items()
                    ),
                    "gold_score_min": float(
                        selected.loc[index, "gold_score_at_open"].min()
                    ),
                    "gold_score_max": float(
                        selected.loc[index, "gold_score_at_open"].max()
                    ),
                }
            )
        event_frames.append(pd.DataFrame(rows))
    paths = pd.DataFrame(path_rows)
    yearly = pd.DataFrame(yearly_rows)
    events = pd.concat(event_frames).sort_values(["selection", "log_excess"])
    paths.to_csv(OUTPUT / "path_metrics.csv", index=False)
    yearly.to_csv(OUTPUT / "yearly_components.csv", index=False)
    events.to_csv(OUTPUT / "gold_overlay_events.csv", index=False)
    audit = {
        "paths": paths.to_dict(orient="records"),
        "yearly_components": yearly.to_dict(orient="records"),
        "worst_overlay_events": {
            label: events.loc[events["selection"].eq(label)].head(5).to_dict(
                orient="records"
            )
            for label in events["selection"].unique()
        },
        "conclusion": {
            "including_extremes_primary_loss": "Gold overlay loses versus its own base",
            "excluding_extremes_primary_loss": "reoptimized anchor base loses versus universal",
            "gold_health_not_relative_strength": True,
            "reset_and_transition_path_cost": True,
        },
    }
    (OUTPUT / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    sources = [
        Path(__file__),
        SOURCE / "selected_including_extremes_daily.parquet",
        SOURCE / "selected_excluding_extremes_daily.parquet",
        UNIVERSAL / "selected_daily.parquet",
    ]
    manifest = {
        "sources": {
            str(path.relative_to(ROOT)): _sha(path) for path in sources
        },
        "artifacts": {
            path.name: _sha(path)
            for path in OUTPUT.iterdir()
            if path.is_file() and path.name != "experiment_manifest.json"
        },
    }
    (OUTPUT / "experiment_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
