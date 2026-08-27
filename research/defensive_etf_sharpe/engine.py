"""Capital-aware backtest engine for the defensive ETF research branch.

Unlike the production backtest runner, this module models external monthly
cash flows, integer ETF lots, cash, and open execution explicitly. Signals are
calculated after a close and executed at the next available open.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from data.store import query


@dataclass(frozen=True)
class StrategyParams:
    momentum_window: int = 120
    trend_window: int = 200
    volatility_window: int = 60
    rebalance_days: int = 20
    top_n: int = 2
    min_momentum: float = 0.0
    volatility_floor: float = 0.08
    weight_mode: str = "inverse_volatility"
    risk_off_regime: bool = False
    regime_window: int = 120
    regime_asset: str = ""
    defensive_assets: tuple[str, ...] = ()
    risk_assets: tuple[str, ...] = ()
    cash_asset: str | None = None
    max_risk_asset_weight: float = 1.0
    rebalance_frequency: str = "days"
    score_mode: str = "momentum_over_volatility"
    min_score: float | None = None

    def __post_init__(self) -> None:
        if self.momentum_window < 1 or self.trend_window < 1:
            raise ValueError("lookback windows must be positive")
        if self.volatility_window < 2 or self.rebalance_days < 1 or self.top_n < 1:
            raise ValueError("volatility_window, rebalance_days, and top_n must be positive")
        if self.regime_window < 1:
            raise ValueError("regime_window must be positive")
        if self.weight_mode not in {"equal", "inverse_volatility"}:
            raise ValueError("weight_mode must be 'equal' or 'inverse_volatility'")
        if not 0 < self.max_risk_asset_weight <= 1:
            raise ValueError("max_risk_asset_weight must be in (0, 1]")
        if self.rebalance_frequency not in {"days", "monthly", "monthly_then_daily_until_positive"}:
            raise ValueError(
                "rebalance_frequency must be 'days', 'monthly', or "
                "'monthly_then_daily_until_positive'"
            )
        if self.score_mode not in {"momentum_over_volatility", "momentum_times_er"}:
            raise ValueError("unsupported score_mode")


@dataclass
class BacktestResult:
    daily: pd.DataFrame
    trades: pd.DataFrame
    params: StrategyParams
    total_deposits: float
    final_nav: float
    signals: pd.DataFrame | None = None

    def metrics(self) -> dict[str, float]:
        returns = self.daily["return"].dropna()
        if returns.empty:
            return {
                "annualized_return": 0.0,
                "sharpe": 0.0,
                "max_drawdown": 0.0,
                "final_nav": self.final_nav,
                "total_deposits": self.total_deposits,
            }
        curve = (1.0 + returns).cumprod()
        annualized_return = float(curve.iloc[-1] ** (252.0 / len(returns)) - 1.0)
        volatility = float(returns.std(ddof=1))
        sharpe = float(returns.mean() / volatility * np.sqrt(252.0)) if volatility > 0 else 0.0
        drawdown = curve / curve.cummax() - 1.0
        return {
            "annualized_return": annualized_return,
            "sharpe": sharpe,
            "max_drawdown": float(drawdown.min()),
            "final_nav": self.final_nav,
            "total_deposits": self.total_deposits,
        }

    def summary(self) -> dict[str, float]:
        return {**asdict(self.params), **self.metrics()}


@dataclass
class MarketData:
    opens: dict[str, pd.Series]
    closes: dict[str, pd.Series]
    dates: list[pd.Timestamp]


def load_market_data(
    asset_pool: list[str],
    start: date,
    end: date,
) -> MarketData:
    opens: dict[str, pd.Series] = {}
    closes: dict[str, pd.Series] = {}
    all_dates: set[pd.Timestamp] = set()
    for asset in asset_pool:
        frame = query(asset, start, end)
        if frame.empty:
            continue
        frame = frame.sort_values("date").drop_duplicates("date")
        index = pd.DatetimeIndex(frame["date"])
        opens[asset] = pd.Series(frame["open"].astype(float).to_numpy(), index=index)
        closes[asset] = pd.Series(frame["close"].astype(float).to_numpy(), index=index)
        all_dates.update(index.tolist())
    if not all_dates:
        raise RuntimeError("no local market data available for the requested asset pool")
    return MarketData(opens=opens, closes=closes, dates=sorted(all_dates))


def _asof(series: pd.Series, timestamp: pd.Timestamp) -> float | None:
    position = int(series.index.searchsorted(timestamp, side="right") - 1)
    if position < 0:
        return None
    value = series.iloc[position]
    return float(value) if pd.notna(value) else None


def _exact(series: pd.Series, timestamp: pd.Timestamp) -> float | None:
    position = int(series.index.searchsorted(timestamp))
    if position >= len(series.index) or series.index[position] != timestamp:
        return None
    value = series.iloc[position]
    return float(value) if pd.notna(value) else None


def _mark_prices(
    series_by_asset: dict[str, pd.Series],
    timestamp: pd.Timestamp,
    assets: set[str],
) -> dict[str, float]:
    prices: dict[str, float] = {}
    for asset in assets:
        price = _asof(series_by_asset[asset], timestamp)
        if price is not None and price > 0:
            prices[asset] = price
    return prices


def _monthly_deposit_dates(
    data: MarketData,
    reference_asset: str,
) -> dict[tuple[int, int], pd.Timestamp]:
    """Use the reference ETF calendar for the first trading day of each month."""
    reference_index = data.closes.get(reference_asset, pd.Series(dtype=float)).index
    dates = list(reference_index) if len(reference_index) else data.dates
    result: dict[tuple[int, int], pd.Timestamp] = {}
    for timestamp in dates:
        result.setdefault((timestamp.year, timestamp.month), timestamp)
    return result


def _minimum_history(params: StrategyParams) -> int:
    if params.score_mode == "momentum_times_er":
        return params.momentum_window + 1
    return max(
        params.momentum_window, params.trend_window, params.volatility_window
    ) + 1


def _signal(
    data: MarketData,
    timestamp: pd.Timestamp,
    params: StrategyParams,
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    scores: dict[str, float] = {}
    vols: dict[str, float] = {}
    diagnostics: dict[str, dict[str, float]] = {}
    min_history = _minimum_history(params)

    eligible_assets = set(params.risk_assets) if params.risk_assets else set(data.closes)
    for asset, close in data.closes.items():
        if asset not in eligible_assets:
            continue
        end_position = int(close.index.searchsorted(timestamp, side="right"))
        history = close.iloc[:end_position].dropna()
        if len(history) < min_history or _exact(close, timestamp) is None:
            continue
        latest = float(history.iloc[-1])
        momentum = latest / float(history.iloc[-1 - params.momentum_window]) - 1.0
        if params.score_mode == "momentum_times_er":
            path_length = history.diff().abs().tail(params.momentum_window).sum()
            er = abs(latest - float(history.iloc[-1 - params.momentum_window])) / path_length if path_length > 0 else np.nan
            score = momentum * er
            diagnostics[asset] = {
                "momentum": momentum,
                "er": er,
                "score": score,
            }
            if not np.isfinite(score) or (
                params.min_score is not None and score <= params.min_score
            ):
                continue
            scores[asset] = score
            vols[asset] = 1.0
            continue

        trend = latest / float(history.iloc[-1 - params.trend_window]) - 1.0
        log_returns = np.log(history).diff().dropna().tail(params.volatility_window)
        volatility = float(log_returns.std(ddof=1) * np.sqrt(252.0))
        if not np.isfinite(volatility):
            continue
        diagnostics[asset] = {
            "momentum": momentum,
            "trend": trend,
            "volatility": volatility,
        }
        if momentum < params.min_momentum or trend <= 0:
            continue
        adjusted_vol = max(volatility, params.volatility_floor)
        scores[asset] = momentum / adjusted_vol
        vols[asset] = adjusted_vol

    if params.risk_off_regime and params.regime_asset in data.closes:
        regime_close = data.closes[params.regime_asset]
        regime_end = int(regime_close.index.searchsorted(timestamp, side="right"))
        regime_history = regime_close.iloc[:regime_end].dropna()
        if len(regime_history) >= params.regime_window + 1:
            regime_return = (
                float(regime_history.iloc[-1])
                / float(regime_history.iloc[-1 - params.regime_window])
                - 1.0
            )
            if regime_return <= 0:
                scores = {
                    asset: score
                    for asset, score in scores.items()
                    if asset in params.defensive_assets
                }

    selected = sorted(scores, key=scores.get, reverse=True)[: params.top_n]
    if not selected:
        if params.cash_asset and params.cash_asset in data.closes:
            return {params.cash_asset: 1.0}, diagnostics
        return {}, diagnostics
    if params.weight_mode == "equal":
        raw = {asset: 1.0 for asset in selected}
    else:
        raw = {asset: 1.0 / vols[asset] for asset in selected}

    # A single qualifying risk asset remains deliberately capped. Any unused
    # risk budget is held in the configured money-market ETF (or cash if none).
    risk_budget = min(1.0, len(selected) * params.max_risk_asset_weight)
    remaining = set(selected)
    remaining_budget = risk_budget
    weights: dict[str, float] = {}
    while remaining:
        raw_total = sum(raw[asset] for asset in remaining)
        tentative = {
            asset: remaining_budget * raw[asset] / raw_total for asset in remaining
        }
        capped = [
            asset
            for asset, value in tentative.items()
            if value >= params.max_risk_asset_weight
        ]
        if not capped:
            weights.update(tentative)
            break
        for asset in capped:
            weights[asset] = params.max_risk_asset_weight
            remaining_budget -= params.max_risk_asset_weight
            remaining.remove(asset)

    residual = max(0.0, 1.0 - sum(weights.values()))
    if params.cash_asset and residual > 1e-12 and params.cash_asset in data.closes:
        weights[params.cash_asset] = residual
    return weights, diagnostics


def _execute_target(
    cash: float,
    holdings: dict[str, int],
    target: dict[str, float],
    open_prices: dict[str, float],
    last_close_prices: dict[str, float],
    cost_rate: float,
    lot_size: int,
) -> tuple[float, dict[str, int], list[dict[str, float | str]]]:
    assets = set(holdings) | set(target)
    mark = {**last_close_prices, **open_prices}
    nav_open = cash + sum(holdings.get(asset, 0) * mark.get(asset, 0.0) for asset in holdings)
    if nav_open <= 0:
        return cash, holdings, []

    trades: list[dict[str, float | str]] = []
    # Sell first, including reductions in existing target positions.
    desired_shares: dict[str, int] = {}
    for asset, weight in target.items():
        price = open_prices.get(asset)
        if price is not None and price > 0:
            desired_shares[asset] = int(nav_open * weight / price / lot_size) * lot_size

    for asset in sorted(assets):
        current = holdings.get(asset, 0)
        desired = desired_shares.get(asset, 0)
        if current <= desired or asset not in open_prices:
            continue
        shares = current - desired
        notional = shares * open_prices[asset]
        cash += notional * (1.0 - cost_rate)
        holdings[asset] = desired
        trades.append({"asset": asset, "side": "sell", "shares": shares, "notional": notional})

    # Buy deficits in target-score order, using remaining cash after costs.
    for asset in sorted(desired_shares, key=lambda item: target[item], reverse=True):
        price = open_prices[asset]
        current = holdings.get(asset, 0)
        deficit = desired_shares[asset] - current
        if deficit <= 0:
            continue
        affordable = int(cash / (price * (1.0 + cost_rate)) / lot_size) * lot_size
        shares = min(deficit, affordable)
        if shares <= 0:
            continue
        notional = shares * price
        cash -= notional * (1.0 + cost_rate)
        holdings[asset] = current + shares
        trades.append({"asset": asset, "side": "buy", "shares": shares, "notional": notional})

    holdings = {asset: shares for asset, shares in holdings.items() if shares > 0}
    return cash, holdings, trades


def simulate(
    data: MarketData,
    params: StrategyParams,
    *,
    monthly_deposit: float = 20_000.0,
    cost_rate: float = 0.0005,
    lot_size: int = 100,
    parameter_schedule: dict[pd.Timestamp, StrategyParams] | None = None,
    deposit_reference_asset: str = "511880.SH",
) -> BacktestResult:
    if monthly_deposit < 0 or cost_rate < 0 or lot_size < 1:
        raise ValueError("monthly_deposit, cost_rate, and lot_size must be non-negative/positive")

    cash = 0.0
    holdings: dict[str, int] = {}
    last_nav = 0.0
    total_deposits = 0.0
    pending_target: dict[str, float] | None = None
    last_close_prices: dict[str, float] = {}
    rows: list[dict[str, object]] = []
    trade_rows: list[dict[str, object]] = []
    signal_rows: list[dict[str, object]] = []
    last_month: tuple[int, int] | None = None
    schedule = sorted((pd.Timestamp(key), value) for key, value in (parameter_schedule or {}).items())
    schedule_position = 0
    active_params = params
    active_since_index = 0
    last_signal_month: tuple[int, int] | None = None
    waiting_for_positive_score = False
    deposit_dates = _monthly_deposit_dates(data, deposit_reference_asset)

    for day_index, timestamp in enumerate(data.dates):
        switched = False
        while schedule_position < len(schedule) and timestamp >= schedule[schedule_position][0]:
            active_params = schedule[schedule_position][1]
            schedule_position += 1
            switched = True
        if switched:
            pending_target = None
            active_since_index = day_index

        month = (timestamp.year, timestamp.month)
        deposit = monthly_deposit if deposit_dates.get(month) == timestamp else 0.0
        last_month = month
        cash += deposit
        total_deposits += deposit

        tradable_assets = set(holdings)
        if pending_target is not None:
            tradable_assets.update(pending_target)
        open_prices = {
            asset: price
            for asset, series in data.opens.items()
            if (price := _exact(series, timestamp)) is not None and price > 0
        }
        mark_open = _mark_prices(data.closes, timestamp, tradable_assets)
        if pending_target is not None:
            cash, holdings, executed = _execute_target(
                cash,
                holdings,
                pending_target,
                open_prices,
                {**last_close_prices, **mark_open},
                cost_rate,
                lot_size,
            )
        else:
            executed = []
        for trade in executed:
            trade_rows.append({"date": timestamp, **trade})
        pending_target = None

        close_assets = set(holdings)
        close_prices = _mark_prices(data.closes, timestamp, close_assets)
        nav = cash + sum(holdings[asset] * close_prices.get(asset, last_close_prices.get(asset, 0.0)) for asset in holdings)
        daily_return = (nav - deposit) / last_nav - 1.0 if last_nav > 0 else np.nan
        weights = {
            asset: holdings[asset] * close_prices.get(asset, last_close_prices.get(asset, 0.0)) / nav
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
            "positions": weights,
        })
        last_close_prices.update(close_prices)
        last_nav = nav

        min_history = _minimum_history(active_params)
        if day_index + 1 >= min_history and day_index < len(data.dates) - 1:
            signal_trigger = ""
            if active_params.rebalance_frequency == "monthly":
                should_rebalance = last_signal_month != month
                signal_trigger = "monthly" if should_rebalance else ""
            elif active_params.rebalance_frequency == "monthly_then_daily_until_positive":
                if last_signal_month != month:
                    should_rebalance = True
                    signal_trigger = "monthly"
                else:
                    should_rebalance = waiting_for_positive_score
                    signal_trigger = "wait_for_positive" if should_rebalance else ""
            else:
                should_rebalance = (day_index - active_since_index) % active_params.rebalance_days == 0
                signal_trigger = "periodic" if should_rebalance else ""
            if should_rebalance:
                pending_target, diagnostics = _signal(data, timestamp, active_params)
                positive_score_exists = any(
                    values.get("score", float("-inf")) > 0
                    for values in diagnostics.values()
                )
                for asset, values in diagnostics.items():
                    signal_rows.append(
                        {
                            "date": timestamp,
                            "asset": asset,
                            **values,
                            "trigger": signal_trigger,
                            "selected": asset in pending_target,
                            "target_weight": pending_target.get(asset, 0.0),
                        }
                    )
                last_signal_month = month
                if active_params.rebalance_frequency == "monthly_then_daily_until_positive":
                    waiting_for_positive_score = not positive_score_exists

    daily = pd.DataFrame(rows).set_index("date")
    trades = pd.DataFrame(trade_rows)
    if trades.empty:
        trades = pd.DataFrame(columns=["date", "asset", "side", "shares", "notional"])
    signals = pd.DataFrame(signal_rows)
    if signals.empty:
        signals = pd.DataFrame(columns=["date", "asset", "selected", "target_weight"])
    return BacktestResult(
        daily=daily,
        trades=trades,
        params=params,
        total_deposits=total_deposits,
        final_nav=float(last_nav),
        signals=signals,
    )


def simulate_static_allocation(
    data: MarketData,
    target_weights: dict[str, float],
    *,
    cash_asset: str,
    monthly_deposit: float = 20_000.0,
    cost_rate: float = 0.0005,
    lot_size: int = 100,
    deposit_reference_asset: str = "511880.SH",
    target_schedule: dict[pd.Timestamp, dict[str, float]] | None = None,
) -> BacktestResult:
    """Simulate a monthly rebalanced fixed-weight defensive benchmark.

    Before an ETF lists, its prescribed weight is moved to the cash sleeve;
    this avoids using a pre-listing proxy or retrospectively changing weights.
    """
    if not np.isclose(sum(target_weights.values()), 1.0):
        raise ValueError("target_weights must sum to 1")
    if cash_asset not in target_weights:
        raise ValueError("target_weights must include the cash_asset")
    for scheduled_weights in (target_schedule or {}).values():
        if not np.isclose(sum(scheduled_weights.values()), 1.0):
            raise ValueError("every scheduled target_weights mapping must sum to 1")
        if cash_asset not in scheduled_weights:
            raise ValueError("every scheduled target must include the cash_asset")

    cash = 0.0
    holdings: dict[str, int] = {}
    last_nav = 0.0
    total_deposits = 0.0
    pending_target: dict[str, float] | None = None
    last_close_prices: dict[str, float] = {}
    rows: list[dict[str, object]] = []
    trade_rows: list[dict[str, object]] = []
    deposit_dates = _monthly_deposit_dates(data, deposit_reference_asset)
    last_target_month: tuple[int, int] | None = None
    schedule = sorted(
        (pd.Timestamp(timestamp), weights)
        for timestamp, weights in (target_schedule or {}).items()
    )
    schedule_position = 0
    active_target_weights = target_weights

    for timestamp in data.dates:
        while (
            schedule_position < len(schedule)
            and timestamp >= schedule[schedule_position][0]
        ):
            active_target_weights = schedule[schedule_position][1]
            schedule_position += 1
        month = (timestamp.year, timestamp.month)
        deposit = monthly_deposit if deposit_dates.get(month) == timestamp else 0.0
        cash += deposit
        total_deposits += deposit

        tradable_assets = set(holdings)
        if pending_target is not None:
            tradable_assets.update(pending_target)
        open_prices = {
            asset: price
            for asset, series in data.opens.items()
            if (price := _exact(series, timestamp)) is not None and price > 0
        }
        mark_open = _mark_prices(data.closes, timestamp, tradable_assets)
        if pending_target is not None:
            cash, holdings, executed = _execute_target(
                cash,
                holdings,
                pending_target,
                open_prices,
                {**last_close_prices, **mark_open},
                cost_rate,
                lot_size,
            )
        else:
            executed = []
        for trade in executed:
            trade_rows.append({"date": timestamp, **trade})
        pending_target = None

        close_assets = set(holdings)
        close_prices = _mark_prices(data.closes, timestamp, close_assets)
        nav = cash + sum(
            holdings[asset] * close_prices.get(asset, last_close_prices.get(asset, 0.0))
            for asset in holdings
        )
        daily_return = (nav - deposit) / last_nav - 1.0 if last_nav > 0 else np.nan
        weights = {
            asset: holdings[asset]
            * close_prices.get(asset, last_close_prices.get(asset, 0.0))
            / nav
            for asset in holdings
            if nav > 0
        }
        rows.append(
            {
                "date": timestamp,
                "nav": nav,
                "cash": cash,
                "deposit": deposit,
                "return": daily_return,
                "cash_weight": cash / nav if nav > 0 else np.nan,
                "positions": weights,
            }
        )
        last_close_prices.update(close_prices)
        last_nav = nav

        if last_target_month != month and timestamp != data.dates[-1]:
            available = {
                asset: weight
                for asset, weight in active_target_weights.items()
                if asset != cash_asset and _exact(data.closes.get(asset, pd.Series(dtype=float)), timestamp)
                is not None
            }
            unavailable_weight = sum(active_target_weights.values()) - sum(available.values())
            if _exact(data.closes.get(cash_asset, pd.Series(dtype=float)), timestamp) is not None:
                available[cash_asset] = unavailable_weight
            pending_target = available
            last_target_month = month

    daily = pd.DataFrame(rows).set_index("date")
    trades = pd.DataFrame(trade_rows)
    if trades.empty:
        trades = pd.DataFrame(columns=["date", "asset", "side", "shares", "notional"])
    return BacktestResult(
        daily=daily,
        trades=trades,
        params=StrategyParams(rebalance_frequency="monthly"),
        total_deposits=total_deposits,
        final_nav=float(last_nav),
        signals=pd.DataFrame(columns=["date", "asset", "selected", "target_weight"]),
    )


def simulate_buy_and_hold(
    data: MarketData,
    asset: str,
    *,
    monthly_deposit: float = 20_000.0,
    cost_rate: float = 0.0005,
    lot_size: int = 100,
    deposit_reference_asset: str = "511880.SH",
) -> pd.DataFrame:
    """Simulate a cashflow-matched buy-and-hold baseline for one ETF."""
    if asset not in data.opens or asset not in data.closes:
        raise ValueError(f"baseline asset {asset!r} has no local data")
    cash = 0.0
    shares = 0
    last_nav = 0.0
    last_close: float | None = None
    last_month: tuple[int, int] | None = None
    rows: list[dict[str, object]] = []
    deposit_dates = _monthly_deposit_dates(data, deposit_reference_asset)

    for timestamp in data.dates:
        month = (timestamp.year, timestamp.month)
        deposit = monthly_deposit if deposit_dates.get(month) == timestamp else 0.0
        last_month = month
        cash += deposit
        open_price = _exact(data.opens[asset], timestamp)
        if open_price is not None and open_price > 0 and cash > 0:
            buy_shares = int(cash / (open_price * (1.0 + cost_rate)) / lot_size) * lot_size
            if buy_shares > 0:
                notional = buy_shares * open_price
                cash -= notional * (1.0 + cost_rate)
                shares += buy_shares
        close_price = _asof(data.closes[asset], timestamp)
        if close_price is not None and close_price > 0:
            last_close = close_price
        nav = cash + shares * (last_close or 0.0)
        daily_return = (nav - deposit) / last_nav - 1.0 if last_nav > 0 else np.nan
        rows.append({"date": timestamp, "nav": nav, "cash": cash, "deposit": deposit, "return": daily_return})
        last_nav = nav

    return pd.DataFrame(rows).set_index("date")
