"""Finalize direct stability comparisons for dynamic versus fixed Gold escape."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from research.momentum_defender_occam import performance


QUANTILE_OUTPUT = Path(
    "experiments/20260825_momentum_defender_w40_quantile_escape_search"
)
FIXED_OUTPUT = Path(
    "experiments/20260825_momentum_defender_w40_asset_specific_escape_search"
)
FORMAL_OUTPUT = Path(
    "experiments/20260825_momentum_defender_w40_reversal_full_equity_v1_formal"
)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output = root / QUANTILE_OUTPUT
    quantile = pd.read_parquet(output / "selected_daily.parquet")["return"].astype(float)
    fixed = pd.read_parquet(root / FIXED_OUTPUT / "selected_daily.parquet")[
        "return"
    ].astype(float)
    formal = pd.read_parquet(root / FORMAL_OUTPUT / "daily_backtest.parquet")[
        "return"
    ].astype(float)
    leave_rows = []
    yearly_rows = []
    for year in sorted(quantile.index.year.unique()):
        keep = quantile.index.year != year
        q_leave = performance(quantile.loc[keep])
        f_leave = performance(fixed.loc[keep])
        b_leave = performance(formal.loc[keep])
        leave_rows.append(
            {
                "removed_year": int(year),
                "quantile_annualized_return_252": q_leave[
                    "annualized_return_252"
                ],
                "fixed_annualized_return_252": f_leave["annualized_return_252"],
                "formal_annualized_return_252": b_leave["annualized_return_252"],
                "quantile_sharpe": q_leave["sharpe"],
                "fixed_sharpe": f_leave["sharpe"],
                "formal_sharpe": b_leave["sharpe"],
            }
        )
        mask = quantile.index.year == year
        q_year = quantile.loc[mask]
        f_year = fixed.loc[mask]
        b_year = formal.loc[mask]
        yearly_rows.append(
            {
                "year": int(year),
                "quantile_return": float((1.0 + q_year).prod() - 1.0),
                "fixed_return": float((1.0 + f_year).prod() - 1.0),
                "formal_return": float((1.0 + b_year).prod() - 1.0),
                "quantile_sharpe": performance(q_year)["sharpe"],
                "fixed_sharpe": performance(f_year)["sharpe"],
                "formal_sharpe": performance(b_year)["sharpe"],
            }
        )
    leave = pd.DataFrame(leave_rows)
    yearly = pd.DataFrame(yearly_rows)
    leave.to_csv(output / "selected_leave_one_year_vs_fixed.csv", index=False)
    yearly.to_csv(output / "selected_calendar_years_vs_fixed.csv", index=False)
    q_metrics = performance(quantile)
    f_metrics = performance(fixed)
    audit = {
        "quantile_selected": q_metrics,
        "fixed_gold_candidate": f_metrics,
        "point_comparison": {
            "annualized_delta": q_metrics["annualized_return_252"]
            - f_metrics["annualized_return_252"],
            "sharpe_delta": q_metrics["sharpe"] - f_metrics["sharpe"],
            "matched_or_better_both": bool(
                q_metrics["annualized_return_252"]
                >= f_metrics["annualized_return_252"]
                and q_metrics["sharpe"] >= f_metrics["sharpe"]
            ),
        },
        "leave_one_year": {
            "quantile_beats_fixed_annualized_rate": float(
                leave["quantile_annualized_return_252"]
                .ge(leave["fixed_annualized_return_252"])
                .mean()
            ),
            "quantile_beats_fixed_sharpe_rate": float(
                leave["quantile_sharpe"].ge(leave["fixed_sharpe"]).mean()
            ),
            "quantile_beats_formal_both_rate": float(
                (
                    leave["quantile_annualized_return_252"]
                    .gt(leave["formal_annualized_return_252"])
                    & leave["quantile_sharpe"].gt(leave["formal_sharpe"])
                ).mean()
            ),
        },
        "calendar_years": {
            "quantile_return_not_below_fixed_rate": float(
                yearly["quantile_return"].ge(yearly["fixed_return"]).mean()
            ),
            "quantile_sharpe_not_below_fixed_rate": float(
                yearly["quantile_sharpe"].ge(yearly["fixed_sharpe"]).mean()
            ),
        },
    }
    (output / "comparison_vs_fixed_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
