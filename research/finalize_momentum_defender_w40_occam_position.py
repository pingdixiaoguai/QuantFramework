"""Combine both W40 position-search stages and finalize global evidence."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd

from research.momentum_defender_common_score_trimmed import (
    ExtremeBlockSpec,
    build_extreme_block_mask,
)
from research.momentum_defender_gold_override_overfit import (
    cscv_pbo,
    yearly_reality_check,
)
from research.momentum_defender_occam import performance
from research.momentum_volatility import load_ohlc
from research.run_momentum_defender_log_qm_robust import _unique_paths
from strategy.momentum_defender_w40_loss import run_formal_strategy


STAGE1 = Path(
    "experiments/20260825_momentum_defender_w40_occam_position_search"
)
STAGE2 = Path(
    "experiments/20260825_momentum_defender_w40_occam_position_focused"
)
SELECTED_ID = "fixed_w1.00"
END = date(2026, 8, 21)


def _leave_one_year(
    candidate: pd.Series, baseline: pd.Series
) -> pd.DataFrame:
    rows = []
    for year in sorted(candidate.index.year.unique()):
        keep = candidate.index.year != year
        rows.append(
            {
                "removed_year": int(year),
                **{
                    f"candidate_{key}": value
                    for key, value in performance(candidate.loc[keep]).items()
                },
                **{
                    f"formal_{key}": value
                    for key, value in performance(baseline.loc[keep]).items()
                },
            }
        )
    return pd.DataFrame(rows)


def _calendar_years(
    candidate: pd.Series, baseline: pd.Series
) -> pd.DataFrame:
    rows = []
    for year in sorted(candidate.index.year.unique()):
        mask = candidate.index.year == year
        candidate_metrics = performance(candidate.loc[mask])
        baseline_metrics = performance(baseline.loc[mask])
        rows.append(
            {
                "year": int(year),
                "candidate_return": float((1.0 + candidate.loc[mask]).prod() - 1.0),
                "formal_return": float((1.0 + baseline.loc[mask]).prod() - 1.0),
                "candidate_sharpe": candidate_metrics["sharpe"],
                "formal_sharpe": baseline_metrics["sharpe"],
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    stage1 = root / STAGE1
    stage2 = root / STAGE2
    first = pd.read_parquet(stage1 / "unique_candidate_returns.parquet")
    second = pd.read_parquet(stage2 / "unique_candidate_returns.parquet")
    first.columns = [f"stage1::{column}" for column in first]
    second.columns = [f"stage2::{column}" for column in second]
    combined = _unique_paths(pd.concat([first, second], axis=1))
    combined.to_parquet(stage2 / "global_unique_candidate_returns.parquet")

    formal = run_formal_strategy(root, end=END)
    baseline = formal.daily["return"].astype(float)
    candidate = pd.read_parquet(
        stage2 / "selected_joint_occam_daily.parquet"
    )["return"].astype(float)
    trim = build_extreme_block_mask(
        {
            asset: load_ohlc(asset, END)["close"]
            for asset in ("510300.SH", "518880.SH")
        },
        formal.context.calendar,
        ExtremeBlockSpec(
            shock_return_window=5,
            block_length_sessions=20,
            excluded_block_fraction=0.10,
            normalization_mode="raw_absolute_log_return",
        ),
    )
    ordinary = trim.selection_mask.astype(bool)
    cscv_full, cscv_full_summary = cscv_pbo(
        combined, baseline, block_count=12
    )
    cscv_ordinary, cscv_ordinary_summary = cscv_pbo(
        combined.loc[ordinary], baseline.loc[ordinary], block_count=12
    )
    cscv_full.to_csv(stage2 / "global_cscv_full.csv", index=False)
    cscv_ordinary.to_csv(stage2 / "global_cscv_ordinary.csv", index=False)
    reality = {
        "full": yearly_reality_check(
            combined, baseline, repetitions=5000, seed=20260825
        ),
        "ordinary": yearly_reality_check(
            combined.loc[ordinary],
            baseline.loc[ordinary],
            repetitions=5000,
            seed=20260825,
        ),
    }
    leave = _leave_one_year(candidate, baseline)
    yearly = _calendar_years(candidate, baseline)
    leave.to_csv(stage2 / "selected_joint_occam_leave_one_year.csv", index=False)
    yearly.to_csv(stage2 / "selected_joint_occam_calendar_years.csv", index=False)
    audit = {
        "input_candidate_ids": int(first.shape[1] + second.shape[1]),
        "global_unique_paths": int(combined.shape[1]),
        "selected_candidate": SELECTED_ID,
        "selected_full": performance(candidate),
        "selected_ordinary": performance(candidate.loc[ordinary]),
        "formal_full": performance(baseline),
        "formal_ordinary": performance(baseline.loc[ordinary]),
        "global_cscv": {
            "full": cscv_full_summary,
            "ordinary": cscv_ordinary_summary,
        },
        "global_reality_check": reality,
        "leave_one_year": {
            "candidate_min_annualized_return_252": float(
                leave["candidate_annualized_return_252"].min()
            ),
            "candidate_min_sharpe": float(leave["candidate_sharpe"].min()),
            "candidate_return_beats_formal_rate": float(
                leave["candidate_annualized_return_252"]
                .gt(leave["formal_annualized_return_252"])
                .mean()
            ),
            "candidate_sharpe_beats_formal_rate": float(
                leave["candidate_sharpe"].gt(leave["formal_sharpe"]).mean()
            ),
        },
        "calendar_years": {
            "return_win_rate": float(
                yearly["candidate_return"].gt(yearly["formal_return"]).mean()
            ),
            "sharpe_win_rate": float(
                yearly["candidate_sharpe"].gt(yearly["formal_sharpe"]).mean()
            ),
        },
        "evidence_status": "retrospective_not_independent_oos",
    }
    (stage2 / "global_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
