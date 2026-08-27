"""Global audit of all prior asset-gate, Gold, and relative-Gold paths."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from research.momentum_defender_common_score_trimmed import (
    ExtremeBlockSpec,
    build_extreme_block_mask,
)
from research.momentum_defender_gold_override import build_gold_override_context
from research.momentum_defender_gold_override_overfit import (
    cscv_pbo,
    expanding_walk_forward,
    leave_one_year_selection,
    yearly_reality_check,
)
from research.momentum_volatility import load_ohlc


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "experiments/20260825_momentum_defender_relative_gold_overlay_search"
EXPERIMENTS = (
    "20260824_momentum_defender_selected_asset_draqm",
    "20260824_momentum_defender_selected_asset_draqm_focused",
    "20260824_momentum_defender_selected_asset_draqm_final_neighborhood",
    "20260824_momentum_defender_common_score_trimmed",
    "20260824_momentum_defender_common_score_raw_trim",
    "20260824_momentum_defender_common_score_raw_trim_focused",
    "20260825_momentum_defender_dual_regime_search",
    "20260825_momentum_defender_gold_exception_search",
    "20260825_momentum_defender_relative_gold_overlay_search",
)


def main() -> None:
    seen: set[str] = set()
    frames = []
    counts = []
    for name in EXPERIMENTS:
        path = ROOT / "experiments" / name / "unique_candidate_returns.parquet"
        frame = pd.read_parquet(path)
        keep = []
        for column in frame:
            digest = hashlib.sha1(
                frame[column].to_numpy(float).tobytes()
            ).hexdigest()
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
    context = build_gold_override_context(
        ROOT, end=pd.Timestamp("2026-08-21").date()
    )
    universal_daily = pd.read_parquet(
        ROOT
        / "experiments/20260824_momentum_defender_downside_raqm_final_selection/selected_daily.parquet"
    ).reindex(context.calendar)
    universal = universal_daily["return"].astype(float)
    closes = {
        asset: load_ohlc(asset, pd.Timestamp("2026-08-21").date())["close"]
        for asset in ("510300.SH", "518880.SH")
    }
    ordinary = build_extreme_block_mask(
        closes,
        context.calendar,
        ExtremeBlockSpec(normalization_mode="raw_absolute_log_return"),
    ).selection_mask.astype(bool)
    pbo_full, pbo_full_summary = cscv_pbo(returns, universal, block_count=12)
    pbo_ordinary, pbo_ordinary_summary = cscv_pbo(
        returns.loc[ordinary], universal.loc[ordinary], block_count=12
    )
    reality_full = yearly_reality_check(
        returns, universal, repetitions=5000, seed=20260825
    )
    reality_ordinary = yearly_reality_check(
        returns.loc[ordinary],
        universal.loc[ordinary],
        repetitions=5000,
        seed=20260825,
    )
    walk = expanding_walk_forward(returns, universal)
    leave = leave_one_year_selection(returns, universal)
    pbo_full.to_csv(OUTPUT / "global_all_cscv_full.csv", index=False)
    pbo_ordinary.to_csv(OUTPUT / "global_all_cscv_ordinary.csv", index=False)
    walk.to_csv(OUTPUT / "global_all_walk_forward.csv", index=False)
    leave.to_csv(OUTPUT / "global_all_leave_one_year.csv", index=False)
    audit = {
        "input_candidate_ids": 68678 + 7140,
        "global_unique_paths": int(returns.shape[1]),
        "baseline": "frozen_universal_510300_gate",
        "family_path_counts": counts,
        "cscv_full": pbo_full_summary,
        "cscv_ordinary": pbo_ordinary_summary,
        "reality_full": reality_full,
        "reality_ordinary": reality_ordinary,
        "walk_forward_return_win_rate": float(
            walk["test_return_delta"].gt(0.0).mean()
        ),
        "walk_forward_sharpe_win_rate": float(
            walk["test_sharpe_delta"].gt(0.0).mean()
        ),
        "leave_one_year_return_win_rate": float(
            leave["test_return_delta"].gt(0.0).mean()
        ),
        "leave_one_year_sharpe_win_rate": float(
            leave["test_sharpe_delta"].gt(0.0).mean()
        ),
    }
    (OUTPUT / "global_all_prior_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
