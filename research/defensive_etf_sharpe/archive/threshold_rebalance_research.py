"""Search rebalance standards for the reduced four-ETF reversal strategy."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .factor_research import _extended_metrics
from .rebalance_timing import daily_reversal_targets
from .reduced_pool_research import REDUCED_TARGET
from .strategy import CASH_ASSET, load_confirmed_market
from .threshold_rebalance import TRIGGER_SPECS, TriggerSpec, simulate_threshold_rebalance


ROOT = Path(__file__).parent
OUTPUT = ROOT / "threshold_rebalance_experiments"
NEVER = TriggerSpec("deposit_only_no_rebalance", "target_change", 2.0, description="仅按月末目标投入新增资金，不主动再平衡")


def _summary(metrics: pd.DataFrame) -> str:
    display = metrics[[
        "standard", "annualized_return", "volatility", "sharpe", "max_drawdown",
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
    return f"""# 4只ETF反转策略：条件再平衡实验

每月首个交易日开盘，新增20,000元按上月最后一个交易日收盘后计算的目标权重做只买不卖的配置；每个交易日收盘重新计算目标，只有触发指定标准才在下一交易日开盘完整再平衡。再平衡计划单笔与实际单笔均须达到10,000元。

排名规则为：年化收益先达到5%，再按Sharpe从高到低。本轮是全样本探索，不是样本外结论。

{display.to_markdown(index=False)}
"""


def run() -> pd.DataFrame:
    _, market = load_confirmed_market()
    daily_targets = daily_reversal_targets(market, REDUCED_TARGET)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = []
    results = {}

    for spec in (NEVER, *TRIGGER_SPECS):
        result = simulate_threshold_rebalance(
            market,
            daily_targets,
            spec,
            initial_target=REDUCED_TARGET,
            cash_asset=CASH_ASSET,
            min_rebalance_notional=10_000.0,
        )
        results[spec.name] = result
        metrics = _extended_metrics(result)
        rebalance_trades = result.trades.loc[result.trades["reason"] == "threshold_rebalance"]
        rebalance_dates = pd.to_datetime(rebalance_trades["date"]).nunique() if not rebalance_trades.empty else 0
        rows.append({
            "standard": spec.name,
            "description": spec.description,
            "trigger_kind": spec.kind,
            "threshold": spec.threshold,
            "max_days": spec.max_days,
            "min_days": spec.min_days,
            "confirmation_days": spec.confirmation_days,
            "max_rebalances_per_month": spec.max_rebalances_per_month,
            **metrics,
            "trigger_signals": len(result.signals),
            "rebalance_dates": rebalance_dates,
            "deposit_trade_count": int((result.trades["reason"] == "deposit_invest").sum()),
            "rebalance_trade_count": int((result.trades["reason"] == "threshold_rebalance").sum()),
            "average_cash_weight": float(result.daily["cash_weight"].mean()),
        })

    metrics = pd.DataFrame(rows).sort_values(
        ["meets_5pct_return_floor", "sharpe", "annualized_return"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    metrics.insert(0, "rank", np.arange(1, len(metrics) + 1))
    metrics.to_csv(OUTPUT / "threshold_rebalance_metrics.csv", index=False)
    pd.DataFrame([
        {
            "standard": spec.name,
            "kind": spec.kind,
            "threshold": spec.threshold,
            "max_days": spec.max_days,
            "min_days": spec.min_days,
            "confirmation_days": spec.confirmation_days,
            "max_rebalances_per_month": spec.max_rebalances_per_month,
            "description": spec.description,
        }
        for spec in (NEVER, *TRIGGER_SPECS)
    ]).to_csv(OUTPUT / "standards_manifest.csv", index=False)
    (OUTPUT / "SUMMARY.md").write_text(_summary(metrics), encoding="utf-8")

    conditional = metrics.loc[
        ~metrics["standard"].isin({"calendar_monthly_reference", "deposit_only_no_rebalance"})
    ]
    best_name = str(conditional.iloc[0]["standard"])
    best = results[best_name]
    best.daily.to_csv(OUTPUT / "best_conditional_strategy_daily.csv")
    best.trades.to_csv(OUTPUT / "best_conditional_strategy_trades.csv", index=False)
    best.signals.to_csv(OUTPUT / "best_conditional_strategy_trigger_signals.csv", index=False)
    monthly_cap = results["portfolio_drift_15_monthly_cap1"]
    monthly_cap.daily.to_csv(OUTPUT / "monthly_cap_strategy_daily.csv")
    monthly_cap.trades.to_csv(OUTPUT / "monthly_cap_strategy_trades.csv", index=False)
    monthly_cap.signals.to_csv(OUTPUT / "monthly_cap_strategy_trigger_signals.csv", index=False)
    for threshold_name in (
        "portfolio_drift_125_monthly_cap1",
        "portfolio_drift_10_monthly_cap1",
    ):
        threshold_result = results[threshold_name]
        threshold_result.daily.to_csv(OUTPUT / f"{threshold_name}_daily.csv")
        threshold_result.trades.to_csv(OUTPUT / f"{threshold_name}_trades.csv", index=False)
        threshold_result.signals.to_csv(OUTPUT / f"{threshold_name}_trigger_signals.csv", index=False)
    return metrics


if __name__ == "__main__":
    print(run().to_string(index=False))
