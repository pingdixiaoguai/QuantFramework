"""Threshold-triggered rebalancing with prior-close monthly deposit weights."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping

import numpy as np
import pandas as pd

from .engine import BacktestResult, MarketData, StrategyParams, _exact, _mark_prices
from .rebalance_timing import (
    _available_target,
    _buy_cash_toward_target,
    _execute_target_with_min_notional,
    _first_trading_days,
)


@dataclass(frozen=True)
class TriggerSpec:
    name: str
    kind: str
    threshold: float
    max_days: int | None = None
    description: str = ""
    min_days: int = 0
    confirmation_days: int = 1
    max_rebalances_per_month: int | None = None


TRIGGER_SPECS = (
    TriggerSpec("calendar_monthly_reference", "calendar_monthly", 0.5, description="对照：每月首个交易日收盘固定触发"),
    TriggerSpec("portfolio_drift_01", "portfolio_drift", 0.01, description="组合单边偏离达到1%"),
    TriggerSpec("portfolio_drift_02", "portfolio_drift", 0.02, description="组合单边偏离达到2%"),
    TriggerSpec("portfolio_drift_03", "portfolio_drift", 0.03, description="组合单边偏离达到3%"),
    TriggerSpec("portfolio_drift_05", "portfolio_drift", 0.05, description="组合单边偏离达到5%"),
    TriggerSpec("portfolio_drift_075", "portfolio_drift", 0.075, description="组合单边偏离达到7.5%"),
    TriggerSpec("portfolio_drift_10", "portfolio_drift", 0.10, description="组合单边偏离达到10%"),
    TriggerSpec("portfolio_drift_15", "portfolio_drift", 0.15, description="组合单边偏离达到15%"),
    TriggerSpec(
        "portfolio_drift_15_monthly_cap1",
        "portfolio_drift",
        0.15,
        description="组合单边偏离达到15%，每月最多再平衡一次",
        max_rebalances_per_month=1,
    ),
    TriggerSpec(
        "portfolio_drift_125_monthly_cap1",
        "portfolio_drift",
        0.125,
        description="组合单边偏离达到12.5%，每月最多再平衡一次",
        max_rebalances_per_month=1,
    ),
    TriggerSpec(
        "portfolio_drift_10_monthly_cap1",
        "portfolio_drift",
        0.10,
        description="组合单边偏离达到10%，每月最多再平衡一次",
        max_rebalances_per_month=1,
    ),
    TriggerSpec(
        "portfolio_drift_10_confirm05d_monthly_cap1",
        "portfolio_drift",
        0.10,
        description="组合单边偏离连续5日达到10%，每月最多再平衡一次",
        confirmation_days=5,
        max_rebalances_per_month=1,
    ),
    TriggerSpec(
        "calendar_or_drift_10",
        "calendar_or_portfolio_drift",
        0.10,
        description="每月首个交易日固定触发，或组合单边偏离达到10%",
    ),
    TriggerSpec("portfolio_drift_20", "portfolio_drift", 0.20, description="组合单边偏离达到20%"),
    TriggerSpec("portfolio_drift_25", "portfolio_drift", 0.25, description="组合单边偏离达到25%"),
    TriggerSpec("portfolio_drift_30", "portfolio_drift", 0.30, description="组合单边偏离达到30%"),
    TriggerSpec("portfolio_drift_40", "portfolio_drift", 0.40, description="组合单边偏离达到40%"),
    TriggerSpec("max_asset_drift_03", "max_asset_drift", 0.03, description="任一ETF绝对偏离达到3个百分点"),
    TriggerSpec("max_asset_drift_01", "max_asset_drift", 0.01, description="任一ETF绝对偏离达到1个百分点"),
    TriggerSpec("max_asset_drift_02", "max_asset_drift", 0.02, description="任一ETF绝对偏离达到2个百分点"),
    TriggerSpec("max_asset_drift_05", "max_asset_drift", 0.05, description="任一ETF绝对偏离达到5个百分点"),
    TriggerSpec("max_asset_drift_075", "max_asset_drift", 0.075, description="任一ETF绝对偏离达到7.5个百分点"),
    TriggerSpec("max_asset_drift_10", "max_asset_drift", 0.10, description="任一ETF绝对偏离达到10个百分点"),
    TriggerSpec("max_asset_drift_15", "max_asset_drift", 0.15, description="任一ETF绝对偏离达到15个百分点"),
    TriggerSpec("max_asset_drift_20", "max_asset_drift", 0.20, description="任一ETF绝对偏离达到20个百分点"),
    TriggerSpec("max_asset_drift_25", "max_asset_drift", 0.25, description="任一ETF绝对偏离达到25个百分点"),
    TriggerSpec("max_asset_drift_30", "max_asset_drift", 0.30, description="任一ETF绝对偏离达到30个百分点"),
    TriggerSpec("target_change_05", "target_change", 0.05, description="目标组合相对上次再平衡单边变化达到5%"),
    TriggerSpec("target_change_025", "target_change", 0.025, description="目标组合相对上次再平衡单边变化达到2.5%"),
    TriggerSpec("target_change_075", "target_change", 0.075, description="目标组合相对上次再平衡单边变化达到7.5%"),
    TriggerSpec("target_change_10", "target_change", 0.10, description="目标组合相对上次再平衡单边变化达到10%"),
    TriggerSpec("target_change_15", "target_change", 0.15, description="目标组合相对上次再平衡单边变化达到15%"),
    TriggerSpec("target_change_20", "target_change", 0.20, description="目标组合相对上次再平衡单边变化达到20%"),
    TriggerSpec("target_change_25", "target_change", 0.25, description="目标组合相对上次再平衡单边变化达到25%"),
    TriggerSpec("target_change_30", "target_change", 0.30, description="目标组合相对上次再平衡单边变化达到30%"),
    TriggerSpec("target_change_40", "target_change", 0.40, description="目标组合相对上次再平衡单边变化达到40%"),
    TriggerSpec("portfolio_drift_05_or_20d", "portfolio_drift", 0.05, 20, "组合偏离5%或距上次成交20个交易日"),
    TriggerSpec("portfolio_drift_075_or_20d", "portfolio_drift", 0.075, 20, "组合偏离7.5%或距上次成交20个交易日"),
    TriggerSpec("portfolio_drift_10_or_20d", "portfolio_drift", 0.10, 20, "组合偏离10%或距上次成交20个交易日"),
    TriggerSpec("portfolio_drift_15_min05d", "portfolio_drift", 0.15, description="组合偏离15%且距上次成交至少5日", min_days=5),
    TriggerSpec("portfolio_drift_15_min10d", "portfolio_drift", 0.15, description="组合偏离15%且距上次成交至少10日", min_days=10),
    TriggerSpec("portfolio_drift_15_min20d", "portfolio_drift", 0.15, description="组合偏离15%且距上次成交至少20日", min_days=20),
    TriggerSpec("portfolio_drift_20_min10d", "portfolio_drift", 0.20, description="组合偏离20%且距上次成交至少10日", min_days=10),
    TriggerSpec("target_change_15_min05d", "target_change", 0.15, description="目标变化15%且距上次成交至少5日", min_days=5),
    TriggerSpec("target_change_15_min10d", "target_change", 0.15, description="目标变化15%且距上次成交至少10日", min_days=10),
    TriggerSpec("target_change_15_min20d", "target_change", 0.15, description="目标变化15%且距上次成交至少20日", min_days=20),
    TriggerSpec("target_change_15_confirm02d", "target_change", 0.15, description="目标变化15%连续满足2日", confirmation_days=2),
    TriggerSpec("target_change_15_confirm03d", "target_change", 0.15, description="目标变化15%连续满足3日", confirmation_days=3),
    TriggerSpec("target_change_15_confirm05d", "target_change", 0.15, description="目标变化15%连续满足5日", confirmation_days=5),
    TriggerSpec("portfolio_drift_15_confirm02d", "portfolio_drift", 0.15, description="组合偏离15%连续满足2日", confirmation_days=2),
    TriggerSpec("portfolio_drift_15_confirm03d", "portfolio_drift", 0.15, description="组合偏离15%连续满足3日", confirmation_days=3),
    TriggerSpec("portfolio_drift_15_confirm05d", "portfolio_drift", 0.15, description="组合偏离15%连续满足5日", confirmation_days=5),
    TriggerSpec("target_change_20_confirm03d", "target_change", 0.20, description="目标变化20%连续满足3日", confirmation_days=3),
    TriggerSpec("portfolio_drift_20_confirm03d", "portfolio_drift", 0.20, description="组合偏离20%连续满足3日", confirmation_days=3),
)


def _portfolio_weights(
    holdings: Mapping[str, int],
    cash: float,
    prices: Mapping[str, float],
) -> tuple[dict[str, float], float]:
    nav = cash + sum(shares * prices.get(asset, 0.0) for asset, shares in holdings.items())
    if nav <= 0:
        return {}, 1.0
    return (
        {asset: shares * prices.get(asset, 0.0) / nav for asset, shares in holdings.items()},
        cash / nav,
    )


def _trigger_value(
    spec: TriggerSpec,
    actual_weights: Mapping[str, float],
    actual_cash_weight: float,
    current_target: Mapping[str, float],
    last_rebalance_target: Mapping[str, float],
) -> float:
    assets = set(current_target) | set(actual_weights)
    if spec.kind in ("portfolio_drift", "calendar_or_portfolio_drift"):
        return 0.5 * (
            sum(abs(actual_weights.get(asset, 0.0) - current_target.get(asset, 0.0)) for asset in assets)
            + actual_cash_weight
        )
    if spec.kind == "max_asset_drift":
        return max(
            [abs(actual_weights.get(asset, 0.0) - current_target.get(asset, 0.0)) for asset in assets]
            + [actual_cash_weight]
        )
    if spec.kind == "target_change":
        return 0.5 * sum(
            abs(current_target.get(asset, 0.0) - last_rebalance_target.get(asset, 0.0))
            for asset in set(current_target) | set(last_rebalance_target)
        )
    raise ValueError(f"unsupported trigger kind: {spec.kind}")


def simulate_threshold_rebalance(
    data: MarketData,
    daily_targets: Mapping[pd.Timestamp, Mapping[str, float]],
    trigger: TriggerSpec,
    *,
    initial_target: Mapping[str, float],
    cash_asset: str,
    monthly_deposit: float = 20_000.0,
    cost_rate: float = 0.0005,
    lot_size: int = 100,
    min_rebalance_notional: float = 10_000.0,
) -> BacktestResult:
    """Invest monthly cash using the prior close target and rebalance on trigger."""
    deposit_dates = _first_trading_days(data)
    date_positions = {timestamp: index for index, timestamp in enumerate(data.dates)}
    cash = 0.0
    holdings: dict[str, int] = {}
    pending_target: dict[str, float] | None = None
    last_rebalance_target = dict(initial_target)
    last_rebalance_trade_index = 0
    last_nav = 0.0
    total_deposits = 0.0
    last_close_prices: dict[str, float] = {}
    rows: list[dict[str, object]] = []
    trade_rows: list[dict[str, object]] = []
    signal_rows: list[dict[str, object]] = []
    consecutive_hits = 0
    rebalance_counts: dict[tuple[int, int], int] = {}

    for day_index, timestamp in enumerate(data.dates):
        month = (timestamp.year, timestamp.month)
        deposit = monthly_deposit if deposit_dates.get(month) == timestamp else 0.0
        cash += deposit
        total_deposits += deposit

        open_prices = {
            asset: price
            for asset, series in data.opens.items()
            if (price := _exact(series, timestamp)) is not None and price > 0
        }
        mark_open = _mark_prices(data.closes, timestamp, set(holdings))
        execution_marks = {**last_close_prices, **mark_open}
        if pending_target is not None:
            execution_month = (timestamp.year, timestamp.month)
            monthly_capacity = (
                trigger.max_rebalances_per_month is None
                or rebalance_counts.get(execution_month, 0) < trigger.max_rebalances_per_month
            )
            if monthly_capacity:
                cash, holdings, executed = _execute_target_with_min_notional(
                    cash,
                    holdings,
                    pending_target,
                    open_prices,
                    execution_marks,
                    cost_rate,
                    lot_size,
                    min_rebalance_notional,
                )
            else:
                executed = []
            reason = "threshold_rebalance"
            if executed:
                last_rebalance_target = dict(pending_target)
                last_rebalance_trade_index = day_index
                rebalance_counts[execution_month] = rebalance_counts.get(execution_month, 0) + 1
        elif deposit > 0:
            previous_index = day_index - 1
            deposit_target = (
                dict(daily_targets[data.dates[previous_index]])
                if previous_index >= 0
                else dict(initial_target)
            )
            available = _available_target(data, timestamp, deposit_target, cash_asset)
            cash, holdings, executed = _buy_cash_toward_target(
                cash,
                holdings,
                available,
                open_prices,
                execution_marks,
                cost_rate,
                lot_size,
            )
            reason = "deposit_invest"
        else:
            executed = []
            reason = ""
        for trade in executed:
            trade_rows.append({"date": timestamp, "reason": reason, **trade})
        pending_target = None

        close_prices = _mark_prices(data.closes, timestamp, set(holdings))
        nav = cash + sum(
            shares * close_prices.get(asset, last_close_prices.get(asset, 0.0))
            for asset, shares in holdings.items()
        )
        daily_return = (nav - deposit) / last_nav - 1.0 if last_nav > 0 else np.nan
        actual_weights, raw_cash_weight = _portfolio_weights(holdings, cash, close_prices)
        rows.append({
            "date": timestamp,
            "nav": nav,
            "cash": cash,
            "deposit": deposit,
            "return": daily_return,
            "cash_weight": raw_cash_weight,
            "positions": actual_weights,
        })
        last_close_prices.update(close_prices)
        last_nav = nav

        if day_index >= len(data.dates) - 1:
            continue
        target = dict(daily_targets[timestamp])
        available_target = _available_target(data, timestamp, target, cash_asset)
        target_cash = max(0.0, 1.0 - sum(available_target.values()))
        target_for_trigger = dict(available_target)
        if trigger.kind == "calendar_monthly":
            value = 1.0 if deposit_dates.get(month) == timestamp else 0.0
        else:
            value = _trigger_value(
                trigger,
                actual_weights,
                max(0.0, raw_cash_weight - target_cash),
                target_for_trigger,
                last_rebalance_target,
            )
            if trigger.kind == "calendar_or_portfolio_drift" and deposit_dates.get(month) == timestamp:
                value = 1.0
        days_since_trade = day_index - last_rebalance_trade_index
        time_trigger = trigger.max_days is not None and days_since_trade >= trigger.max_days
        if value >= trigger.threshold and days_since_trade >= trigger.min_days:
            consecutive_hits += 1
        else:
            consecutive_hits = 0
        threshold_trigger = consecutive_hits >= trigger.confirmation_days
        next_timestamp = data.dates[day_index + 1]
        next_month = (next_timestamp.year, next_timestamp.month)
        monthly_capacity = (
            trigger.max_rebalances_per_month is None
            or rebalance_counts.get(next_month, 0) < trigger.max_rebalances_per_month
        )
        if (threshold_trigger or time_trigger) and monthly_capacity:
            pending_target = available_target
            consecutive_hits = 0
            signal_rows.append({
                "date": timestamp,
                "trigger_value": value,
                "threshold": trigger.threshold,
                "time_trigger": time_trigger,
                "days_since_rebalance_trade": days_since_trade,
                "confirmation_days": trigger.confirmation_days,
                "execution_month_rebalance_count_before": rebalance_counts.get(next_month, 0),
            })

    daily = pd.DataFrame(rows).set_index("date")
    trades = pd.DataFrame(trade_rows)
    if trades.empty:
        trades = pd.DataFrame(columns=["date", "reason", "asset", "side", "shares", "notional"])
    return BacktestResult(
        daily=daily,
        trades=trades,
        params=StrategyParams(rebalance_frequency="days"),
        total_deposits=total_deposits,
        final_nav=float(last_nav),
        signals=pd.DataFrame(signal_rows),
    )
