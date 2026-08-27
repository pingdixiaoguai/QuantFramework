"""Mitigate daily-target noise in the drift-triggered reversal strategy.

The 10% drift trigger underperforms the fixed monthly check because the
daily-updated reversal target is itself high-frequency noise (daily weight
changes of the equity sleeve have ~9.5pp std and -0.27 lag-1 autocorrelation),
so the band ends up chasing target extremes. This experiment compares three
mitigation families against both references:

1. Smooth the anchor: 5/10-day moving average of the daily target vectors.
2. Slow the anchor: measure drift against the latest month-start target
   (a step series), optionally with a guaranteed month-start rebalance.
3. Slow the signal: 60-day reversal window, or halve the bet (25% tilt).

All variants keep the 20,000 monthly deposit, 100-share lots, 0.05% one-sided
cost, and the 10,000 minimum rebalance notional.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .factor_allocation import FactorSpec, MechanismSpec
from .factor_research import _extended_metrics
from .rebalance_timing import daily_reversal_targets, month_start_anchored_targets
from .reduced_pool_research import REDUCED_TARGET
from .strategy import CASH_ASSET, load_confirmed_market
from .threshold_rebalance import TRIGGER_SPECS, TriggerSpec, simulate_threshold_rebalance

ROOT = Path(__file__).parent
OUTPUT = ROOT / "noise_mitigation_experiments"

REVERSAL_60 = FactorSpec("reversal_60", "reversal", 60, "负的60日简单收益率")
GLOBAL_TILT_025 = MechanismSpec("global_tilt_025", "global_tilt", 0.25, "基线权重乘以全池25%排名倾斜后归一化")

REFERENCES = ("calendar_monthly_reference", "portfolio_drift_10_monthly_cap1")


def _smoothed(
    daily_targets: dict[pd.Timestamp, dict[str, float]],
    dates: pd.DatetimeIndex,
    window: int,
) -> dict[pd.Timestamp, dict[str, float]]:
    frame = pd.DataFrame(daily_targets).T.reindex(dates).astype(float)
    smoothed = frame.rolling(window, min_periods=1).mean()
    smoothed = smoothed.div(smoothed.sum(axis=1), axis=0)
    return {timestamp: dict(smoothed.loc[timestamp]) for timestamp in dates}


def run() -> pd.DataFrame:
    _, market = load_confirmed_market()
    dates = pd.DatetimeIndex(market.dates)
    standard = daily_reversal_targets(market, REDUCED_TARGET)
    anchored = month_start_anchored_targets(standard, dates)

    specs = {spec.name: spec for spec in TRIGGER_SPECS}
    drift10_cap1 = specs["portfolio_drift_10_monthly_cap1"]
    anchor_drift20_cap1 = TriggerSpec(
        "portfolio_drift_20_monthly_cap1",
        "portfolio_drift",
        0.20,
        description="组合单边偏离达到20%，每月最多再平衡一次",
        max_rebalances_per_month=1,
    )

    variants = {
        "ref_calendar_monthly": (standard, specs["calendar_monthly_reference"]),
        "ref_drift10_cap1": (standard, drift10_cap1),
        "smooth05_drift10_cap1": (_smoothed(standard, dates, 5), drift10_cap1),
        "smooth10_drift10_cap1": (_smoothed(standard, dates, 10), drift10_cap1),
        "anchor_drift10_cap1": (anchored, drift10_cap1),
        "anchor_drift20_cap1": (anchored, anchor_drift20_cap1),
        "calendar_or_anchor_drift10": (anchored, specs["calendar_or_drift_10"]),
        "reversal60_drift10_cap1": (
            daily_reversal_targets(market, REDUCED_TARGET, factor=REVERSAL_60),
            drift10_cap1,
        ),
        "tilt025_drift10_cap1": (
            daily_reversal_targets(market, REDUCED_TARGET, mechanism=GLOBAL_TILT_025),
            drift10_cap1,
        ),
    }

    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = []
    results = {}
    for name, (targets, spec) in variants.items():
        result = simulate_threshold_rebalance(
            market,
            targets,
            spec,
            initial_target=REDUCED_TARGET,
            cash_asset=CASH_ASSET,
            min_rebalance_notional=10_000.0,
        )
        results[name] = result
        rebalance_trades = result.trades.loc[result.trades["reason"] == "threshold_rebalance"]
        rows.append({
            "variant": name,
            **_extended_metrics(result),
            "trigger_signals": len(result.signals),
            "rebalance_dates": pd.to_datetime(rebalance_trades["date"]).nunique()
            if not rebalance_trades.empty else 0,
            "deposit_trade_count": int((result.trades["reason"] == "deposit_invest").sum()),
            "rebalance_trade_count": int((result.trades["reason"] == "threshold_rebalance").sum()),
            "average_cash_weight": float(result.daily["cash_weight"].mean()),
        })

    metrics = pd.DataFrame(rows)
    metrics = metrics.sort_values(
        ["meets_5pct_return_floor", "sharpe", "annualized_return"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    metrics.insert(0, "rank", np.arange(1, len(metrics) + 1))
    metrics.to_csv(OUTPUT / "noise_mitigation_metrics.csv", index=False)
    (OUTPUT / "SUMMARY.md").write_text(_summary(metrics), encoding="utf-8")

    conditional = metrics.loc[~metrics["variant"].isin({"ref_calendar_monthly", "ref_drift10_cap1"})]
    save_names = {str(conditional.iloc[0]["variant"]), "calendar_or_anchor_drift10"}
    for name in save_names:
        result = results[name]
        result.daily.to_csv(OUTPUT / f"{name}_daily.csv")
        result.trades.to_csv(OUTPUT / f"{name}_trades.csv", index=False)
        result.signals.to_csv(OUTPUT / f"{name}_trigger_signals.csv", index=False)
    return metrics


def _summary(metrics: pd.DataFrame) -> str:
    display = metrics[[
        "variant", "annualized_return", "volatility", "sharpe", "max_drawdown",
        "sortino", "rebalance_dates", "trades", "estimated_transaction_cost",
        "average_cash_weight",
    ]].copy()
    for column in ("annualized_return", "volatility", "max_drawdown", "average_cash_weight"):
        display[column] = display[column].map(lambda value: f"{value:.2%}")
    for column in ("sharpe", "sortino"):
        display[column] = display[column].map(lambda value: f"{value:.2f}")
    display["estimated_transaction_cost"] = display["estimated_transaction_cost"].map(
        lambda value: f"{value:,.0f}"
    )
    return f"""# 日频目标噪声缓解实验

所有变体共享：每月首个交易日20,000元入金按前收盘目标只买配置、100股整数手、单边0.05%成本、再平衡单笔10,000元门槛。除 calendar_or_anchor_drift10 外均限制每月最多再平衡一次。ref_ 前缀为两个参照策略。排名规则：年化收益先达到5%，再按Sharpe从高到低。本轮是全样本探索，不是样本外结论。

{display.to_markdown(index=False)}
"""


if __name__ == "__main__":
    print(run().to_string(index=False))
