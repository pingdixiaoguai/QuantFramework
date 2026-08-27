"""Four-ETF reversal strategy with deposit netting and order filtering."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .factor_research import _extended_metrics
from .rebalance_timing import (
    _first_trading_days,
    daily_reversal_targets,
    simulate_timed_allocation,
)
from .strategy import load_confirmed_market


ROOT = Path(__file__).parent
OUTPUT = ROOT / "reduced_pool_experiments"

REDUCED_TARGET = {
    "512890.SH": 0.35,
    "511260.SH": 0.40,
    "511360.SH": 0.15,
    "511880.SH": 0.10,
}


EXPERIMENTS = {
    "fixed_monthly_deferred_10k": {
        "factor": False,
        "immediate": False,
        "threshold": 10_000.0,
        "description": "4只ETF固定35/40/15/10；入金等到次日完整再平衡；单笔不足1万元不交易",
    },
    "reversal20_immediate_no_threshold": {
        "factor": True,
        "immediate": True,
        "threshold": 0.0,
        "description": "因子策略；入金当日按旧权重买入；次日完整再平衡；无单笔门槛",
    },
    "reversal20_deferred_no_threshold": {
        "factor": True,
        "immediate": False,
        "threshold": 0.0,
        "description": "因子策略；入金留待次日与再平衡净额合并；无单笔门槛",
    },
    "reversal20_deferred_05k": {
        "factor": True,
        "immediate": False,
        "threshold": 5_000.0,
        "description": "因子策略；入金留待次日净额再平衡；单笔不足5000元不交易",
    },
    "reversal20_deferred_10k": {
        "factor": True,
        "immediate": False,
        "threshold": 10_000.0,
        "description": "目标策略：入金留待次日净额再平衡；单笔不足1万元不交易",
    },
    "reversal20_deferred_20k": {
        "factor": True,
        "immediate": False,
        "threshold": 20_000.0,
        "description": "因子策略；入金留待次日净额再平衡；单笔不足2万元不交易",
    },
}


def _summary(metrics: pd.DataFrame) -> str:
    display = metrics[[
        "strategy", "annualized_return", "volatility", "sharpe", "max_drawdown",
        "sortino", "final_nav", "trades", "estimated_transaction_cost", "average_cash_weight",
    ]].copy()
    for column in ("annualized_return", "volatility", "max_drawdown", "average_cash_weight"):
        display[column] = display[column].map(lambda value: f"{value:.2%}")
    for column in ("sharpe", "sortino"):
        display[column] = display[column].map(lambda value: f"{value:.2f}")
    display["final_nav"] = display["final_nav"].map(lambda value: f"{value:,.0f}")
    display["estimated_transaction_cost"] = display["estimated_transaction_cost"].map(
        lambda value: f"{value:,.0f}"
    )
    return f"""# 4只ETF反转策略：入金净额合并与最小成交门槛

候选池缩减为512890.SH（35%基线）、511260.SH（40%）、511360.SH（15%）和511880.SH（10%）。动态版本仍使用20日反转和全池50%排名倾斜。

目标策略在每月首个交易日入金但不交易；当日收盘计算新权重，第二个交易日开盘以包含入金后的总资产计算净目标差额。计划单笔成交不足10,000元的订单先过滤，合格卖单先执行，买单最多使用现有现金与合格卖单所得；部分买入后的实际成交仍须达到10,000元。

{display.to_markdown(index=False)}
"""


def run() -> pd.DataFrame:
    _, market = load_confirmed_market()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    daily_targets = daily_reversal_targets(market, REDUCED_TARGET)
    first_dates = set(_first_trading_days(market).values())
    factor_schedule = {
        timestamp: daily_targets[timestamp] for timestamp in first_dates
    }
    fixed_schedule = {
        timestamp: dict(REDUCED_TARGET) for timestamp in first_dates
    }

    rows = []
    results = {}
    for name, config in EXPERIMENTS.items():
        schedule = factor_schedule if config["factor"] else fixed_schedule
        result = simulate_timed_allocation(
            market,
            schedule,
            initial_target=REDUCED_TARGET,
            invest_deposits_immediately=bool(config["immediate"]),
            min_trade_notional=float(config["threshold"]),
        )
        results[name] = result
        metrics = _extended_metrics(result)
        rows.append({
            "strategy": name,
            "description": config["description"],
            "min_trade_notional": config["threshold"],
            "invest_deposits_immediately": config["immediate"],
            **metrics,
            "average_cash_weight": float(result.daily["cash_weight"].mean()),
            "max_cash_weight": float(result.daily["cash_weight"].max()),
        })

    metrics = pd.DataFrame(rows).sort_values(
        ["meets_5pct_return_floor", "sharpe", "annualized_return"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    metrics.insert(0, "rank", np.arange(1, len(metrics) + 1))
    metrics.to_csv(OUTPUT / "reduced_pool_metrics.csv", index=False)
    pd.DataFrame([
        {"strategy": name, **config} for name, config in EXPERIMENTS.items()
    ]).to_csv(OUTPUT / "strategy_manifest.csv", index=False)
    (OUTPUT / "SUMMARY.md").write_text(_summary(metrics), encoding="utf-8")

    requested = results["reversal20_deferred_10k"]
    requested.trades.to_csv(OUTPUT / "requested_strategy_trades.csv", index=False)
    requested.daily.to_csv(OUTPUT / "requested_strategy_daily.csv")
    weight_rows = [
        {"date": timestamp.date().isoformat(), "asset": asset, "target_weight": weight}
        for timestamp, weights in sorted(factor_schedule.items())
        for asset, weight in weights.items()
    ]
    pd.DataFrame(weight_rows).to_csv(OUTPUT / "requested_monthly_targets.csv", index=False)
    return metrics


if __name__ == "__main__":
    print(run().to_string(index=False))
