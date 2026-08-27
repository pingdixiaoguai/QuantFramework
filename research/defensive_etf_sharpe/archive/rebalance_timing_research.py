"""Compare rebalance timing for reversal-20/global-tilt-50 allocation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .factor_research import _extended_metrics
from .rebalance_timing import (
    _first_trading_days,
    daily_reversal_targets,
    simulate_timed_allocation,
    timing_schedules,
)
from .strategy import STATIC_BENCHMARK_TARGET, load_confirmed_market


ROOT = Path(__file__).parent
OUTPUT = ROOT / "rebalance_timing_experiments"


DESCRIPTIONS = {
    "fixed_baseline_buy_only": "固定35/40/15/10；月初入金当日开盘只买不卖，不主动再平衡",
    "fixed_baseline_monthly_rebalanced": "固定35/40/15/10；入金当日买入，月初收盘后完整再平衡",
    "monthly_first_close": "月初收盘计算目标，次日开盘再平衡；入金当日按旧目标买入",
    "monthly_last_close": "月末收盘计算目标，下一交易日开盘再平衡，通常与次月入金同步",
    "monthly_day_05_close": "每月第5个交易日收盘计算，下一交易日开盘再平衡",
    "monthly_day_10_close": "每月第10个交易日收盘计算，下一交易日开盘再平衡",
    "monthly_day_15_close": "每月第15个交易日收盘计算，下一交易日开盘再平衡",
    "monthly_day_20_close": "每月第20个交易日收盘计算，下一交易日开盘再平衡；不足20日则该月不调",
    "weekly_last_close": "每周最后交易日收盘计算，下一交易日开盘再平衡",
    "every_10_trading_days": "每10个交易日收盘计算，下一交易日开盘再平衡",
    "quarterly_last_close": "每季最后交易日收盘计算，下一交易日开盘再平衡",
    "daily_close": "每日收盘计算，下一交易日开盘再平衡",
    "target_change_05pct": "每日观察；目标组合单边变化达到5%才再平衡",
    "target_change_10pct": "每日观察；目标组合单边变化达到10%才再平衡",
    "target_change_20pct": "每日观察；目标组合单边变化达到20%才再平衡",
    "monthly_plus_target_change_10pct": "月初必调；月内目标组合单边变化达到10%追加再平衡",
}


def _summary(metrics: pd.DataFrame) -> str:
    display = metrics[[
        "strategy", "annualized_return", "volatility", "sharpe", "max_drawdown",
        "sortino", "trades", "estimated_transaction_cost",
    ]].copy()
    for column in ("annualized_return", "volatility", "max_drawdown"):
        display[column] = display[column].map(lambda value: f"{value:.2%}")
    for column in ("sharpe", "sortino"):
        display[column] = display[column].map(lambda value: f"{value:.2f}")
    display["estimated_transaction_cost"] = display["estimated_transaction_cost"].map(
        lambda value: f"{value:,.0f}"
    )
    return f"""# 20日反转＋全池50%倾斜：入金与再平衡时点实验

所有动态策略使用相同的20日反转因子和全池50%倾斜公式。每月首个交易日入金20,000元；若当日没有上一交易日生成的待执行信号，则在当日开盘按最近已知目标权重只买入低配标的、不因入金卖出。所有新目标均在收盘后计算、下一交易日开盘执行。

本轮使用全样本作探索性比较，未生成单独HTML报告。

{display.to_markdown(index=False)}
"""


def run() -> pd.DataFrame:
    _, market = load_confirmed_market()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    daily_targets = daily_reversal_targets(market)
    schedules = timing_schedules(market, daily_targets)

    rows = []
    baseline = simulate_timed_allocation(market, {})
    rows.append({
        "strategy": "fixed_baseline_buy_only",
        "description": DESCRIPTIONS["fixed_baseline_buy_only"],
        "signal_count": 0,
        **_extended_metrics(baseline),
    })
    monthly_dates = set(_first_trading_days(market).values())
    fixed_monthly_schedule = {
        timestamp: dict(STATIC_BENCHMARK_TARGET) for timestamp in monthly_dates
    }
    fixed_monthly = simulate_timed_allocation(market, fixed_monthly_schedule)
    rows.append({
        "strategy": "fixed_baseline_monthly_rebalanced",
        "description": DESCRIPTIONS["fixed_baseline_monthly_rebalanced"],
        "signal_count": len(fixed_monthly_schedule),
        **_extended_metrics(fixed_monthly),
    })
    results = {}
    for name, schedule in schedules.items():
        result = simulate_timed_allocation(market, schedule)
        results[name] = result
        rows.append({
            "strategy": name,
            "description": DESCRIPTIONS[name],
            "signal_count": len(schedule),
            **_extended_metrics(result),
        })

    metrics = pd.DataFrame(rows)
    baseline_row = metrics.loc[metrics["strategy"] == "fixed_baseline_monthly_rebalanced"].iloc[0]
    for column in ("annualized_return", "volatility", "sharpe", "max_drawdown", "sortino", "calmar"):
        metrics[f"delta_{column}_vs_fixed_baseline"] = metrics[column] - float(baseline_row[column])
    metrics = metrics.sort_values(
        ["meets_5pct_return_floor", "sharpe", "annualized_return"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    metrics.insert(0, "rank", np.arange(1, len(metrics) + 1))
    metrics.to_csv(OUTPUT / "rebalance_timing_metrics.csv", index=False)
    pd.DataFrame([
        {"strategy": name, "description": description}
        for name, description in DESCRIPTIONS.items()
    ]).to_csv(OUTPUT / "strategy_manifest.csv", index=False)
    (OUTPUT / "SUMMARY.md").write_text(_summary(metrics), encoding="utf-8")

    best_name = str(metrics.loc[~metrics["strategy"].str.startswith("fixed_baseline")].iloc[0]["strategy"])
    results[best_name].daily.to_csv(OUTPUT / "best_strategy_daily.csv")
    results[best_name].trades.to_csv(OUTPUT / "best_strategy_trades.csv", index=False)
    return metrics


if __name__ == "__main__":
    print(run().to_string(index=False))
