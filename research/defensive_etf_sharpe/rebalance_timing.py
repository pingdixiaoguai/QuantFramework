"""Dynamic allocation simulator with immediate monthly-deposit investment."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from .engine import BacktestResult, MarketData, StrategyParams, _exact, _execute_target, _mark_prices, _monthly_deposit_dates
from .factor_allocation import FactorSpec, MechanismSpec, adjusted_weights, factor_ranks
from .strategy import CASH_ASSET, STATIC_BENCHMARK_TARGET


REVERSAL_20 = FactorSpec("reversal_20", "reversal", 20, "负的20日简单收益率")
GLOBAL_TILT_050 = MechanismSpec(
    "global_tilt_050", "global_tilt", 0.50, "基线权重乘以全池50%排名倾斜后归一化"
)


def _first_trading_days(data: MarketData) -> dict[tuple[int, int], pd.Timestamp]:
    result: dict[tuple[int, int], pd.Timestamp] = {}
    for timestamp in data.dates:
        result.setdefault((timestamp.year, timestamp.month), timestamp)
    return result


def _available_target(
    data: MarketData,
    timestamp: pd.Timestamp,
    target: Mapping[str, float],
    cash_asset: str,
) -> dict[str, float]:
    available = {
        asset: weight
        for asset, weight in target.items()
        if asset != cash_asset
        and _exact(data.closes.get(asset, pd.Series(dtype=float)), timestamp) is not None
    }
    unavailable_weight = sum(target.values()) - sum(available.values())
    if _exact(data.closes.get(cash_asset, pd.Series(dtype=float)), timestamp) is not None:
        available[cash_asset] = unavailable_weight
    return available


def _buy_cash_toward_target(
    cash: float,
    holdings: dict[str, int],
    target: Mapping[str, float],
    open_prices: Mapping[str, float],
    mark_prices: Mapping[str, float],
    cost_rate: float,
    lot_size: int,
) -> tuple[float, dict[str, int], list[dict[str, float | str]]]:
    """Invest available cash toward target deficits without selling holdings."""
    nav_open = cash + sum(
        shares * mark_prices.get(asset, open_prices.get(asset, 0.0))
        for asset, shares in holdings.items()
    )
    deficits: dict[str, float] = {}
    for asset, weight in target.items():
        price = open_prices.get(asset)
        if price is None or price <= 0:
            continue
        current_value = holdings.get(asset, 0) * price
        deficits[asset] = max(0.0, nav_open * weight - current_value)

    trades: list[dict[str, float | str]] = []
    for asset in sorted(deficits, key=deficits.get, reverse=True):
        price = float(open_prices[asset])
        desired = int(deficits[asset] / price / lot_size) * lot_size
        affordable = int(cash / (price * (1.0 + cost_rate)) / lot_size) * lot_size
        shares = min(desired, affordable)
        if shares <= 0:
            continue
        notional = shares * price
        cash -= notional * (1.0 + cost_rate)
        holdings[asset] = holdings.get(asset, 0) + shares
        trades.append({"asset": asset, "side": "buy", "shares": shares, "notional": notional})
    return cash, holdings, trades


def _execute_target_with_min_notional(
    cash: float,
    holdings: dict[str, int],
    target: Mapping[str, float],
    open_prices: Mapping[str, float],
    mark_prices: Mapping[str, float],
    cost_rate: float,
    lot_size: int,
    min_trade_notional: float,
) -> tuple[float, dict[str, int], list[dict[str, float | str]]]:
    """Rebalance with per-order filtering and cash-constrained buys.

    Planned orders below ``min_trade_notional`` are removed first. Eligible
    sells execute before buys. Eligible buys may be partially filled using the
    resulting cash, but the executed buy must still meet the same threshold.
    """
    nav_open = cash + sum(
        shares * mark_prices.get(asset, open_prices.get(asset, 0.0))
        for asset, shares in holdings.items()
    )
    if nav_open <= 0:
        return cash, holdings, []

    desired_shares: dict[str, int] = {}
    for asset, weight in target.items():
        price = open_prices.get(asset)
        if price is not None and price > 0:
            desired_shares[asset] = int(nav_open * weight / price / lot_size) * lot_size

    eligible_sells: dict[str, int] = {}
    eligible_buys: dict[str, int] = {}
    for asset in set(holdings) | set(desired_shares):
        price = open_prices.get(asset)
        if price is None or price <= 0:
            continue
        delta = desired_shares.get(asset, 0) - holdings.get(asset, 0)
        notional = abs(delta) * price
        if notional < min_trade_notional:
            continue
        if delta < 0:
            eligible_sells[asset] = -delta
        elif delta > 0:
            eligible_buys[asset] = delta

    trades: list[dict[str, float | str]] = []
    for asset in sorted(eligible_sells):
        shares = eligible_sells[asset]
        price = float(open_prices[asset])
        notional = shares * price
        cash += notional * (1.0 - cost_rate)
        holdings[asset] = holdings.get(asset, 0) - shares
        trades.append({"asset": asset, "side": "sell", "shares": shares, "notional": notional})

    for asset in sorted(eligible_buys, key=lambda item: eligible_buys[item] * open_prices[item], reverse=True):
        price = float(open_prices[asset])
        desired = eligible_buys[asset]
        affordable = int(cash / (price * (1.0 + cost_rate)) / lot_size) * lot_size
        shares = min(desired, affordable)
        notional = shares * price
        if shares <= 0 or notional < min_trade_notional:
            continue
        cash -= notional * (1.0 + cost_rate)
        holdings[asset] = holdings.get(asset, 0) + shares
        trades.append({"asset": asset, "side": "buy", "shares": shares, "notional": notional})

    holdings = {asset: shares for asset, shares in holdings.items() if shares > 0}
    return cash, holdings, trades


def simulate_timed_allocation(
    data: MarketData,
    target_schedule: Mapping[pd.Timestamp, Mapping[str, float]],
    *,
    initial_target: Mapping[str, float] = STATIC_BENCHMARK_TARGET,
    cash_asset: str = CASH_ASSET,
    monthly_deposit: float = 20_000.0,
    cost_rate: float = 0.0005,
    lot_size: int = 100,
    invest_deposits_immediately: bool = True,
    min_trade_notional: float = 0.0,
) -> BacktestResult:
    """Execute close-generated targets next open and invest deposits immediately.

    On a deposit date with no pending rebalance, cash is invested at that day's
    open toward deficits under the last known target. Existing positions are
    not sold merely because a deposit arrived.
    """
    if not np.isclose(sum(initial_target.values()), 1.0):
        raise ValueError("initial_target must sum to one")

    schedule = {pd.Timestamp(date): dict(weights) for date, weights in target_schedule.items()}
    deposit_dates = _first_trading_days(data)
    cash = 0.0
    holdings: dict[str, int] = {}
    current_target = dict(initial_target)
    pending_target: dict[str, float] | None = None
    last_nav = 0.0
    total_deposits = 0.0
    last_close_prices: dict[str, float] = {}
    rows: list[dict[str, object]] = []
    trade_rows: list[dict[str, object]] = []

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
        if pending_target is not None:
            executor = _execute_target_with_min_notional if min_trade_notional > 0 else _execute_target
            if min_trade_notional > 0:
                cash, holdings, executed = executor(
                    cash,
                    holdings,
                    pending_target,
                    open_prices,
                    {**last_close_prices, **mark_open},
                    cost_rate,
                    lot_size,
                    min_trade_notional,
                )
            else:
                cash, holdings, executed = executor(
                    cash,
                    holdings,
                    pending_target,
                    open_prices,
                    {**last_close_prices, **mark_open},
                    cost_rate,
                    lot_size,
                )
            current_target = pending_target
            trade_reason = "rebalance"
        elif deposit > 0 and invest_deposits_immediately:
            invest_target = _available_target(data, timestamp, current_target, cash_asset)
            cash, holdings, executed = _buy_cash_toward_target(
                cash,
                holdings,
                invest_target,
                open_prices,
                {**last_close_prices, **mark_open},
                cost_rate,
                lot_size,
            )
            trade_reason = "deposit_invest"
        else:
            executed = []
            trade_reason = ""
        for trade in executed:
            trade_rows.append({"date": timestamp, "reason": trade_reason, **trade})
        pending_target = None

        close_prices = _mark_prices(data.closes, timestamp, set(holdings))
        nav = cash + sum(
            shares * close_prices.get(asset, last_close_prices.get(asset, 0.0))
            for asset, shares in holdings.items()
        )
        daily_return = (nav - deposit) / last_nav - 1.0 if last_nav > 0 else np.nan
        position_weights = {
            asset: shares * close_prices.get(asset, last_close_prices.get(asset, 0.0)) / nav
            for asset, shares in holdings.items()
            if nav > 0
        }
        rows.append({
            "date": timestamp,
            "nav": nav,
            "cash": cash,
            "deposit": deposit,
            "return": daily_return,
            "cash_weight": cash / nav if nav > 0 else np.nan,
            "positions": position_weights,
        })
        last_close_prices.update(close_prices)
        last_nav = nav

        if timestamp in schedule and day_index < len(data.dates) - 1:
            weights = schedule[timestamp]
            if not np.isclose(sum(weights.values()), 1.0):
                raise ValueError(f"target on {timestamp.date()} does not sum to one")
            pending_target = _available_target(data, timestamp, weights, cash_asset)

    daily = pd.DataFrame(rows).set_index("date")
    trades = pd.DataFrame(trade_rows)
    if trades.empty:
        trades = pd.DataFrame(columns=["date", "reason", "asset", "side", "shares", "notional"])
    return BacktestResult(
        daily=daily,
        trades=trades,
        params=StrategyParams(rebalance_frequency="monthly"),
        total_deposits=total_deposits,
        final_nav=float(last_nav),
        signals=pd.DataFrame(columns=["date", "asset", "target_weight"]),
    )


def daily_reversal_targets(
    data: MarketData,
    baseline: Mapping[str, float] = STATIC_BENCHMARK_TARGET,
    factor: FactorSpec = REVERSAL_20,
    mechanism: MechanismSpec = GLOBAL_TILT_050,
) -> dict[pd.Timestamp, dict[str, float]]:
    assets = tuple(baseline)
    return {
        timestamp: adjusted_weights(
            factor_ranks(data, timestamp, factor, assets),
            mechanism,
            dict(baseline),
        )
        for timestamp in data.dates
    }


def month_start_anchored_targets(
    daily_targets: Mapping[pd.Timestamp, Mapping[str, float]],
    dates: pd.DatetimeIndex | list[pd.Timestamp],
) -> dict[pd.Timestamp, dict[str, float]]:
    """Replace each day's target with the target sampled at its month start."""
    first_dates: dict[tuple[int, int], pd.Timestamp] = {}
    for timestamp in dates:
        first_dates.setdefault((timestamp.year, timestamp.month), timestamp)
    return {
        timestamp: dict(daily_targets[first_dates[(timestamp.year, timestamp.month)]])
        for timestamp in dates
    }


def _last_dates_by_period(dates: list[pd.Timestamp], frequency: str) -> set[pd.Timestamp]:
    index = pd.DatetimeIndex(dates)
    periods = index.to_period(frequency)
    return set(pd.Series(index, index=periods).groupby(level=0).last().tolist())


def timing_schedules(
    data: MarketData,
    daily_targets: Mapping[pd.Timestamp, Mapping[str, float]],
) -> dict[str, dict[pd.Timestamp, dict[str, float]]]:
    dates = data.dates
    first_month = set(_first_trading_days(data).values())
    last_month = _last_dates_by_period(dates, "M")
    last_week = _last_dates_by_period(dates, "W-FRI")
    last_quarter = _last_dates_by_period(dates, "Q")

    date_sets = {
        "monthly_first_close": first_month,
        "monthly_last_close": last_month,
        "weekly_last_close": last_week,
        "every_10_trading_days": set(dates[20::10]),
        "quarterly_last_close": last_quarter,
        "daily_close": set(dates),
    }
    monthly_groups: dict[tuple[int, int], list[pd.Timestamp]] = {}
    for timestamp in dates:
        monthly_groups.setdefault((timestamp.year, timestamp.month), []).append(timestamp)
    for trading_day_number in (5, 10, 15, 20):
        date_sets[f"monthly_day_{trading_day_number:02d}_close"] = {
            month_dates[trading_day_number - 1]
            for month_dates in monthly_groups.values()
            if len(month_dates) >= trading_day_number
        }
    schedules = {
        name: {date: dict(daily_targets[date]) for date in dates if date in selected}
        for name, selected in date_sets.items()
    }

    for threshold in (0.05, 0.10, 0.20):
        selected: dict[pd.Timestamp, dict[str, float]] = {}
        last_target = dict(STATIC_BENCHMARK_TARGET)
        for timestamp in dates:
            target = dict(daily_targets[timestamp])
            turnover_distance = 0.5 * sum(
                abs(target[asset] - last_target[asset]) for asset in target
            )
            if turnover_distance >= threshold:
                selected[timestamp] = target
                last_target = target
        schedules[f"target_change_{int(threshold * 100):02d}pct"] = selected

    selected = {}
    last_target = dict(STATIC_BENCHMARK_TARGET)
    for timestamp in dates:
        target = dict(daily_targets[timestamp])
        turnover_distance = 0.5 * sum(
            abs(target[asset] - last_target[asset]) for asset in target
        )
        if timestamp in first_month or turnover_distance >= 0.10:
            selected[timestamp] = target
            last_target = target
    schedules["monthly_plus_target_change_10pct"] = selected
    return schedules


def simulate_fixed_buyandhold(
    data: MarketData,
    target_weights: Mapping[str, float],
    *,
    cash_asset: str = CASH_ASSET,
    monthly_deposit: float = 20_000.0,
    cost_rate: float = 0.0005,
    lot_size: int = 100,
    deposit_reference_asset: str = "511880.SH",
) -> BacktestResult:
    """Fixed-weight buy-and-hold baseline: deposit only, never rebalance.

    Each monthly deposit is invested across all assets according to the fixed
    target_weights. Existing holdings are never sold, so weights drift freely
    with market movements. This is the "unbalanced fixed proportion" base.
    """
    from .engine import _asof

    deposit_dates = _monthly_deposit_dates(data, deposit_reference_asset)
    cash = 0.0
    holdings: dict[str, int] = {}
    last_nav = 0.0
    total_deposits = 0.0
    last_close_prices: dict[str, float] = {}
    rows: list[dict[str, object]] = []
    trade_rows: list[dict[str, object]] = []

    for timestamp in data.dates:
        month = (timestamp.year, timestamp.month)
        deposit = monthly_deposit if deposit_dates.get(month) == timestamp else 0.0
        cash += deposit
        total_deposits += deposit

        open_prices = {
            asset: price
            for asset, series in data.opens.items()
            if (price := _exact(series, timestamp)) is not None and price > 0
        }

        if deposit > 0:
            available_assets = {
                asset for asset in target_weights
                if open_prices.get(asset) is not None and open_prices[asset] > 0
            }
            unavailable_weight = sum(
                target_weights[asset] for asset in target_weights if asset not in available_assets
            )
            adjusted = dict(target_weights)
            if cash_asset in available_assets:
                adjusted[cash_asset] = adjusted.get(cash_asset, 0.0) + unavailable_weight
            adjusted = {
                asset: weight
                for asset, weight in adjusted.items()
                if asset in available_assets
            }
            total_weight = sum(adjusted.values())
            if total_weight > 0:
                adjusted = {asset: weight / total_weight for asset, weight in adjusted.items()}

            for asset in sorted(adjusted, key=lambda a: -adjusted[a]):
                if asset == cash_asset:
                    continue
                price = open_prices[asset]
                desired_notional = deposit * adjusted[asset]
                shares = int(desired_notional / price / lot_size) * lot_size
                if shares <= 0:
                    continue
                notional = shares * price
                cost = notional * cost_rate
                if cash < notional + cost:
                    continue
                cash -= notional + cost
                holdings[asset] = holdings.get(asset, 0) + shares
                trade_rows.append({
                    "date": timestamp,
                    "reason": "deposit_invest",
                    "asset": asset,
                    "side": "buy",
                    "shares": shares,
                    "notional": notional,
                })

            if cash_asset in available_assets:
                price = open_prices[cash_asset]
                if price > 0 and cash > 0:
                    shares = int(cash / (price * (1.0 + cost_rate)) / lot_size) * lot_size
                    if shares > 0:
                        notional = shares * price
                        cash -= notional * (1.0 + cost_rate)
                        holdings[cash_asset] = holdings.get(cash_asset, 0) + shares
                        trade_rows.append({
                            "date": timestamp,
                            "reason": "deposit_invest",
                            "asset": cash_asset,
                            "side": "buy",
                            "shares": shares,
                            "notional": notional,
                        })

        close_prices = _mark_prices(data.closes, timestamp, set(holdings))
        nav = cash + sum(
            holdings[asset] * close_prices.get(asset, last_close_prices.get(asset, 0.0))
            for asset in holdings
        )
        daily_return = (nav - deposit) / last_nav - 1.0 if last_nav > 0 else np.nan
        position_weights = {
            asset: holdings[asset]
            * close_prices.get(asset, last_close_prices.get(asset, 0.0))
            / nav
            for asset in holdings
            if nav > 0
        }
        rows.append({
            "date": timestamp,
            "nav": nav,
            "cash": cash,
            "deposit": deposit,
            "return": daily_return,
            "cash_weight": cash / nav if nav > 0 else np.nan,
            "positions": position_weights,
        })
        last_close_prices.update(close_prices)
        last_nav = nav

    daily = pd.DataFrame(rows).set_index("date")
    trades = pd.DataFrame(trade_rows)
    if trades.empty:
        trades = pd.DataFrame(columns=["date", "reason", "asset", "side", "shares", "notional"])
    return BacktestResult(
        daily=daily,
        trades=trades,
        params=StrategyParams(rebalance_frequency="monthly"),
        total_deposits=total_deposits,
        final_nav=float(last_nav),
    )
