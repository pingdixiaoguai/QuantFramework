"""Global multiple-testing audit across prior DRAQM and Gold-exception rounds."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from research.momentum_defender_common_score_trimmed import (
    ExtremeBlockSpec,
    build_extreme_block_mask,
)
from research.momentum_defender_downside_raqm import (
    build_exact_execution_data,
    exact_candidate_schedule,
)
from research.momentum_defender_gold_override import build_gold_override_context
from research.momentum_defender_gold_override_overfit import (
    cscv_pbo,
    expanding_walk_forward,
    yearly_reality_check,
)
from research.momentum_volatility import load_ohlc


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "experiments/20260825_momentum_defender_gold_exception_search"
EXPERIMENTS = (
    "20260824_momentum_defender_selected_asset_draqm",
    "20260824_momentum_defender_selected_asset_draqm_focused",
    "20260824_momentum_defender_selected_asset_draqm_final_neighborhood",
    "20260824_momentum_defender_common_score_trimmed",
    "20260824_momentum_defender_common_score_raw_trim",
    "20260824_momentum_defender_common_score_raw_trim_focused",
    "20260825_momentum_defender_dual_regime_search",
    "20260825_momentum_defender_gold_exception_search",
)


def main() -> None:
    seen = set()
    frames = []
    counts = []
    for name in EXPERIMENTS:
        frame = pd.read_parquet(ROOT / "experiments" / name / "unique_candidate_returns.parquet")
        keep = []
        for column in frame:
            digest = hashlib.sha1(frame[column].to_numpy(float).tobytes()).hexdigest()
            if digest not in seen:
                seen.add(digest)
                keep.append(str(column))
        selected = frame.loc[:, keep].copy()
        selected.columns = [f"{name}::{column}" for column in keep]
        frames.append(selected)
        counts.append(
            {
                "experiment": name,
                "input_unique_paths": int(frame.shape[1]),
                "new_global_unique_paths": len(keep),
            }
        )
    returns = pd.concat(frames, axis=1)
    context = build_gold_override_context(ROOT, end=pd.Timestamp("2026-08-21").date())
    data = build_exact_execution_data(context)
    momentum_values, _, _ = exact_candidate_schedule(data, data.momentum_target)
    momentum = pd.Series(momentum_values, index=data.calendar)
    closes = {
        asset: load_ohlc(asset, pd.Timestamp("2026-08-21").date())["close"]
        for asset in ("510300.SH", "518880.SH")
    }
    ordinary = build_extreme_block_mask(
        closes,
        data.calendar,
        ExtremeBlockSpec(normalization_mode="raw_absolute_log_return"),
    ).selection_mask
    pbo_full, pbo_full_summary = cscv_pbo(returns, momentum, block_count=12)
    pbo_ordinary, pbo_ordinary_summary = cscv_pbo(
        returns.loc[ordinary], momentum.loc[ordinary], block_count=12
    )
    reality_full = yearly_reality_check(
        returns, momentum, repetitions=5000, seed=20260825
    )
    reality_ordinary = yearly_reality_check(
        returns.loc[ordinary], momentum.loc[ordinary], repetitions=5000, seed=20260825
    )
    walk = expanding_walk_forward(returns, momentum)
    pbo_full.to_csv(OUTPUT / "global_prior_cscv_full.csv", index=False)
    pbo_ordinary.to_csv(OUTPUT / "global_prior_cscv_ordinary.csv", index=False)
    walk.to_csv(OUTPUT / "global_prior_walk_forward.csv", index=False)
    audit = {
        "input_candidate_ids": 55355 + 72 + 8400 + 4851,
        "global_unique_paths": int(returns.shape[1]),
        "family_path_counts": counts,
        "cscv_full": pbo_full_summary,
        "cscv_ordinary": pbo_ordinary_summary,
        "reality_full": reality_full,
        "reality_ordinary": reality_ordinary,
        "walk_forward_return_win_rate": float(walk["test_return_delta"].gt(0.0).mean()),
        "walk_forward_sharpe_win_rate": float(walk["test_sharpe_delta"].gt(0.0).mean()),
    }
    (OUTPUT / "global_prior_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
