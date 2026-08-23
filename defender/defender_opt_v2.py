"""Defender v2 with a stable-uptrend full-position re-entry overlay.

The original grid strategy only adds at the bottom of its 55-day range or
after a fast five-day breakout.  This module adds a third, causal route for a
slow and orderly uptrend.  Signals are calculated at the close and executed
at the next trading day's open, exactly like the original strategy.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import date
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from .grid_reproduction import (
    ASSET,
    INITIAL_CAPITAL,
    TRADING_DAYS,
    GridParams,
    _realized_volatility,
    _target_from_close,
    _volatility_cap,
)
from .market_risk_overlay import (
    MarketRiskOverlayParams,
    calculate_market_risk_overlay,
)
from data.store import read_local


PRIMARY_ASSET = ASSET
DEFENSIVE_ASSETS = ("511260.SH", "511360.SH", "511880.SH")
TRANSACTION_COST_RATES: dict[str, float] = {
    PRIMARY_ASSET: 0.0001,
    "511260.SH": 0.00001,
    "511360.SH": 0.0,
    "511880.SH": 0.0,
}
CostRateSpec = float | Mapping[str, float]


def _asset_cost_rate(asset: str, cost_rates: CostRateSpec) -> float:
    """Resolve and validate the one-way commission for one asset."""
    rate = (
        float(cost_rates[asset])
        if isinstance(cost_rates, Mapping)
        else float(cost_rates)
    )
    if rate < 0:
        raise ValueError(f"transaction cost rate must be non-negative for {asset}")
    return rate


@dataclass(frozen=True)
class ScoreParams:
    """Parameters for the stable-uptrend score and its hysteresis."""

    momentum_method: str = "efficiency_adjusted"
    momentum_window: int = 10
    score_volatility_method: str = "rogers_satchell"
    score_volatility_window: int = 5
    volatility_transform: str = "low_volatility"
    low_volatility_ceiling: float = 0.11
    regime_window: int = 120
    regime_threshold: float = 0.0
    entry_threshold: float = 0.0029
    exit_threshold: float = 0.0017


@dataclass(frozen=True)
class DefensiveAllocationParams:
    """Monthly allocation of the capital not assigned to 512890.

    The factor is the main branch's volatility-adjusted reversal definition:
    negative trailing return divided by volatility over the same window.  The
    highest-scoring available ETF receives the whole defensive sleeve.
    """

    assets: tuple[str, ...] = DEFENSIVE_ASSETS
    factor_window: int = 20
    rebalance_frequency: str = "monthly"
    allocation_mode: str = "top1"
    fallback_asset: str = "511880.SH"
    volatility_floor: float = 0.0
    min_score_advantage: float = 1.0
    min_holding_months: int = 1
    top1_blend: float = 1.0
    midmonth_fallback: bool = True
    fixed_asset: str | None = None

    def __post_init__(self) -> None:
        if self.factor_window < 2:
            raise ValueError("factor_window must be at least 2")
        if self.rebalance_frequency != "monthly":
            raise ValueError("only monthly defensive rebalancing is supported")
        if self.allocation_mode not in {"top1", "rank", "top1_rank_blend"}:
            raise ValueError("unsupported defensive allocation_mode")
        if self.fallback_asset not in self.assets:
            raise ValueError("fallback_asset must be included in assets")
        if self.volatility_floor < 0:
            raise ValueError("volatility_floor must be non-negative")
        if self.min_score_advantage < 0:
            raise ValueError("min_score_advantage must be non-negative")
        if self.min_holding_months < 1:
            raise ValueError("min_holding_months must be at least 1")
        if not 0 <= self.top1_blend <= 1:
            raise ValueError("top1_blend must be in [0, 1]")
        if self.fixed_asset is not None and self.fixed_asset not in self.assets:
            raise ValueError("fixed_asset must be included in assets")


@dataclass(frozen=True)
class DefenderOptV2Params:
    grid: GridParams = GridParams()
    score: ScoreParams = ScoreParams()
    defensive: DefensiveAllocationParams = DefensiveAllocationParams()
    market_risk_overlay: MarketRiskOverlayParams = MarketRiskOverlayParams()


def robust_reselected_params() -> DefenderOptV2Params:
    """Return the low-complexity profile selected by rolling research.

    The core engine stays at its existing anchor.  The unconfirmed broad
    overlay is disabled and the residual sleeve is fixed to 511260.
    """
    base = DefenderOptV2Params()
    return replace(
        base,
        defensive=replace(base.defensive, fixed_asset="511260.SH"),
        market_risk_overlay=replace(base.market_risk_overlay, enabled=False),
    )


def load_market_prices(
    start: date = date(2019, 1, 18),
    end: date | None = None,
    assets: tuple[str, ...] = (PRIMARY_ASSET, *DEFENSIVE_ASSETS),
) -> dict[str, pd.DataFrame]:
    """Load HFQ OHLC through the latest primary-asset date by default."""
    result: dict[str, pd.DataFrame] = {}
    for asset in assets:
        frame = read_local(asset)
        if frame is None or frame.empty:
            if asset == PRIMARY_ASSET:
                raise RuntimeError(f"missing local data for {asset}")
            continue
        frame = frame.copy()
        frame["date"] = pd.to_datetime(frame["date"])
        frame = frame.loc[frame["date"] >= pd.Timestamp(start)]
        if end is not None:
            frame = frame.loc[frame["date"] <= pd.Timestamp(end)]
        frame = frame.sort_values("date").drop_duplicates("date")
        required = ["date", "open", "high", "low", "close"]
        if frame[required].isna().any().any():
            raise ValueError(f"price data for {asset} contains missing OHLC values")
        result[asset] = frame.reset_index(drop=True)
    return result


def _regression_r2_momentum(close: pd.Series, window: int) -> pd.Series:
    """Return log-price regression momentum weighted by regression R-squared."""
    x = np.arange(window, dtype=float)
    centered_x = x - x.mean()
    denominator = float(np.square(centered_x).sum())

    def calculate(values: np.ndarray) -> float:
        centered_y = values - values.mean()
        slope = float(np.dot(centered_x, centered_y) / denominator)
        fitted = values.mean() + slope * centered_x
        total = float(np.square(centered_y).sum())
        r_squared = 1.0 - float(np.square(values - fitted).sum()) / total if total > 0 else 0.0
        return (np.exp(slope * window) - 1.0) * max(r_squared, 0.0)

    return np.log(close).rolling(window).apply(calculate, raw=True)


def _momentum_component(close: pd.Series, params: ScoreParams) -> pd.Series:
    """Calculate one of the momentum definitions explored for v2."""
    window = params.momentum_window
    if window < 2:
        raise ValueError("momentum_window must be at least 2")

    trailing_return = close.pct_change(window)
    if params.momentum_method == "trailing_return":
        return trailing_return
    if params.momentum_method == "ema_spread":
        average = close.ewm(span=window, adjust=False, min_periods=window).mean()
        return close / average - 1.0
    if params.momentum_method == "regression_r2":
        return _regression_r2_momentum(close, window)
    if params.momentum_method == "efficiency_adjusted":
        change = close.diff()
        path_length = change.abs().rolling(window).sum()
        efficiency = change.rolling(window).sum() / path_length.replace(0.0, np.nan)
        return trailing_return.clip(lower=0.0) * efficiency.clip(lower=0.0)

    supported = "trailing_return, ema_spread, regression_r2, efficiency_adjusted"
    raise ValueError(
        f"unsupported momentum_method {params.momentum_method!r}; use one of: {supported}"
    )


def calculate_score(prices: pd.DataFrame, params: ScoreParams = ScoreParams()) -> pd.DataFrame:
    """Calculate causal score components used by the v2 state machine."""
    frame = prices.sort_values("date").reset_index(drop=True).copy()
    close = frame["close"].astype(float)
    momentum = _momentum_component(close, params)
    volatility = _realized_volatility(
        frame,
        GridParams(
            volatility_method=params.score_volatility_method,
            volatility_window=params.score_volatility_window,
        ),
    )
    volatility_series = pd.Series(volatility, index=frame.index)

    if params.volatility_transform == "raw":
        volatility_score = volatility_series
    elif params.volatility_transform == "inverse":
        volatility_score = 1.0 / volatility_series.replace(0.0, np.nan)
    elif params.volatility_transform == "low_volatility":
        if params.low_volatility_ceiling <= 0:
            raise ValueError("low_volatility_ceiling must be positive")
        volatility_score = (
            1.0 - volatility_series / params.low_volatility_ceiling
        ).clip(lower=0.0, upper=1.0)
    else:
        raise ValueError(
            "unsupported volatility_transform; use raw, inverse, or low_volatility"
        )

    regime_momentum = close.pct_change(params.regime_window)
    return pd.DataFrame({
        "momentum_component": momentum,
        "score_volatility": volatility_series,
        "volatility_score": volatility_score,
        "momentum_score": momentum.clip(lower=0.0) * volatility_score,
        "regime_momentum": regime_momentum,
    })


def _normalize_market_input(
    prices: pd.DataFrame | Mapping[str, pd.DataFrame] | None,
    params: DefenderOptV2Params,
) -> dict[str, pd.DataFrame]:
    if prices is None:
        market = load_market_prices(assets=(PRIMARY_ASSET, *params.defensive.assets))
    elif isinstance(prices, pd.DataFrame):
        primary = prices.copy()
        primary["date"] = pd.to_datetime(primary["date"])
        market = load_market_prices(
            start=primary["date"].min().date(),
            end=primary["date"].max().date(),
            assets=(PRIMARY_ASSET, *params.defensive.assets),
        )
        market[PRIMARY_ASSET] = primary
    else:
        market = {asset: frame.copy() for asset, frame in prices.items()}

    if PRIMARY_ASSET not in market:
        raise RuntimeError(f"missing primary asset {PRIMARY_ASSET}")
    normalized: dict[str, pd.DataFrame] = {}
    for asset, frame in market.items():
        if frame.empty:
            continue
        frame = frame.copy()
        frame["date"] = pd.to_datetime(frame["date"])
        normalized[asset] = frame.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    return normalized


def _indexed_market(market: Mapping[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    return {asset: frame.set_index("date").sort_index() for asset, frame in market.items()}


def _asof_price(frame: pd.DataFrame, timestamp: pd.Timestamp, column: str) -> float | None:
    history = frame.loc[:timestamp, column].dropna()
    if history.empty:
        return None
    value = float(history.iloc[-1])
    return value if value > 0 else None


def defensive_reversal_scores(
    market: Mapping[str, pd.DataFrame],
    timestamp: pd.Timestamp,
    params: DefensiveAllocationParams = DefensiveAllocationParams(),
) -> dict[str, float]:
    """Calculate the main-branch 20-day volatility-adjusted reversal score."""
    indexed = _indexed_market(market)
    scores: dict[str, float] = {}
    for asset in params.assets:
        if asset not in indexed:
            continue
        history = indexed[asset].loc[:timestamp, "close"].dropna().astype(float)
        if len(history) < params.factor_window + 1:
            continue
        window = params.factor_window
        trailing_return = float(history.iloc[-1] / history.iloc[-1 - window] - 1.0)
        returns = history.pct_change().dropna().tail(window)
        window_volatility = float(returns.std(ddof=1) * np.sqrt(window))
        denominator = max(window_volatility, params.volatility_floor)
        if np.isfinite(denominator) and denominator > 0:
            scores[asset] = -trailing_return / denominator
    return scores


def _rank_weights(scores: Mapping[str, float]) -> dict[str, float]:
    """Convert scores to ordinal weights without depending on score magnitude."""
    if not scores:
        return {}
    ranks = pd.Series(scores, dtype=float).rank(method="average")
    total = float(ranks.sum())
    return {asset: float(rank / total) for asset, rank in ranks.items()}


def _defensive_weights(
    scores: Mapping[str, float],
    selected_asset: str,
    params: DefensiveAllocationParams,
) -> dict[str, float]:
    if params.allocation_mode == "top1":
        return {selected_asset: 1.0}
    rank_weights = _rank_weights(scores)
    if not rank_weights:
        return {selected_asset: 1.0}
    if params.allocation_mode == "rank":
        return rank_weights
    blend = params.top1_blend
    weights = {
        asset: (1.0 - blend) * weight for asset, weight in rank_weights.items()
    }
    weights[selected_asset] = weights.get(selected_asset, 0.0) + blend
    return weights


def _select_defensive_allocation(
    market: Mapping[str, pd.DataFrame],
    timestamp: pd.Timestamp | None,
    params: DefensiveAllocationParams,
    available_at_open: set[str],
    current_asset: str,
    months_held: int,
) -> tuple[str, dict[str, float], dict[str, float]]:
    scores = (
        defensive_reversal_scores(market, timestamp, params)
        if timestamp is not None
        else {}
    )
    if params.fixed_asset is not None:
        if params.fixed_asset in available_at_open:
            return params.fixed_asset, {params.fixed_asset: 1.0}, scores
        if params.fallback_asset in available_at_open:
            return params.fallback_asset, {params.fallback_asset: 1.0}, scores
        eligible = [asset for asset in params.assets if asset in available_at_open]
        if not eligible:
            raise RuntimeError("no defensive ETF is available for the residual sleeve")
        return eligible[0], {eligible[0]: 1.0}, scores
    eligible_scores = {
        asset: score for asset, score in scores.items() if asset in available_at_open
    }
    if eligible_scores:
        best_asset = max(eligible_scores, key=eligible_scores.get)
        current_is_eligible = current_asset in eligible_scores
        can_switch = (
            not current_is_eligible
            or (
                months_held >= params.min_holding_months
                and eligible_scores[best_asset] - eligible_scores[current_asset]
                >= params.min_score_advantage
            )
        )
        selected_asset = best_asset if can_switch else current_asset
        return (
            selected_asset,
            _defensive_weights(eligible_scores, selected_asset, params),
            scores,
        )
    if params.fallback_asset in available_at_open:
        return params.fallback_asset, {params.fallback_asset: 1.0}, scores
    eligible = [asset for asset in params.assets if asset in available_at_open]
    if not eligible:
        raise RuntimeError("no defensive ETF is available for the residual sleeve")
    return eligible[0], {eligible[0]: 1.0}, scores


def _redistribute_unavailable_defensive_weights(
    weights: Mapping[str, float],
    scores: Mapping[str, float],
    available_at_open: set[str],
    params: DefensiveAllocationParams,
) -> tuple[dict[str, float], bool]:
    """Move unavailable defensive mass to the best available monthly candidate."""
    unavailable_mass = sum(
        weight for asset, weight in weights.items() if asset not in available_at_open
    )
    if unavailable_mass <= 1e-14:
        return dict(weights), False
    available_scores = {
        asset: score for asset, score in scores.items() if asset in available_at_open
    }
    if available_scores:
        replacement = max(available_scores, key=available_scores.get)
    elif params.fallback_asset in available_at_open:
        replacement = params.fallback_asset
    else:
        candidates = [asset for asset in params.assets if asset in available_at_open]
        if not candidates:
            return dict(weights), False
        replacement = candidates[0]
    result = {
        asset: weight for asset, weight in weights.items() if asset in available_at_open
    }
    result[replacement] = result.get(replacement, 0.0) + unavailable_mass
    return result, True


def _execute_portfolio_target(
    cash: float,
    shares: dict[str, float],
    target: Mapping[str, float],
    open_prices: Mapping[str, float],
    mark_prices: Mapping[str, float],
    cost_rates: CostRateSpec,
) -> tuple[float, dict[str, float], list[dict[str, float | str]]]:
    """Rebalance fractional ETF holdings, selling before buying."""
    nav_open = cash + sum(
        quantity * mark_prices.get(asset, 0.0) for asset, quantity in shares.items()
    )
    if nav_open <= 0:
        return cash, shares, []
    desired = {asset: nav_open * weight for asset, weight in target.items()}
    executions: list[dict[str, float | str]] = []

    for asset in sorted(set(shares) | set(target)):
        price = open_prices.get(asset)
        if price is None:
            continue
        current_value = shares.get(asset, 0.0) * price
        sell_value = max(0.0, current_value - desired.get(asset, 0.0))
        if sell_value <= 1e-14:
            continue
        asset_cost_rate = _asset_cost_rate(asset, cost_rates)
        shares[asset] = shares.get(asset, 0.0) - sell_value / price
        cash += sell_value * (1.0 - asset_cost_rate)
        executions.append({
            "asset": asset,
            "side": "sell",
            "execution_price": price,
            "notional": sell_value,
            "turnover": sell_value / nav_open,
            "cost_rate": asset_cost_rate,
            "cost": sell_value * asset_cost_rate,
        })

    needs: dict[str, float] = {}
    for asset, desired_value in desired.items():
        price = open_prices.get(asset)
        if price is None:
            continue
        needs[asset] = max(0.0, desired_value - shares.get(asset, 0.0) * price)
    total_cash_need = sum(
        value * (1.0 + _asset_cost_rate(asset, cost_rates))
        for asset, value in needs.items()
    )
    scale = min(1.0, cash / total_cash_need) if total_cash_need else 0.0
    for asset in sorted(needs, key=lambda item: target[item], reverse=True):
        buy_value = needs[asset] * scale
        if buy_value <= 1e-14:
            continue
        price = open_prices[asset]
        asset_cost_rate = _asset_cost_rate(asset, cost_rates)
        shares[asset] = shares.get(asset, 0.0) + buy_value / price
        cash -= buy_value * (1.0 + asset_cost_rate)
        executions.append({
            "asset": asset,
            "side": "buy",
            "execution_price": price,
            "notional": buy_value,
            "turnover": buy_value / nav_open,
            "cost_rate": asset_cost_rate,
            "cost": buy_value * asset_cost_rate,
        })
    return cash, {asset: qty for asset, qty in shares.items() if qty > 1e-14}, executions


def run_backtest(
    prices: pd.DataFrame | Mapping[str, pd.DataFrame] | None = None,
    params: DefenderOptV2Params = DefenderOptV2Params(),
    cost_rate: CostRateSpec | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float | int | str]]:
    """Run the causal, fully invested v2 strategy."""
    market = _normalize_market_input(prices, params)
    applied_cost_rates = TRANSACTION_COST_RATES if cost_rate is None else cost_rate
    indexed = _indexed_market(market)
    frame = market[PRIMARY_ASSET].copy()
    closes = frame["close"].to_numpy(dtype=float)
    realized_volatility = _realized_volatility(frame, params.grid)
    score_frame = calculate_score(frame, params.score)
    overlay_frame = calculate_market_risk_overlay(frame, params.market_risk_overlay)

    grid_target = params.grid.max_exposure
    primary_target = params.grid.max_exposure
    pending_primary_target: float | None = primary_target
    pending_reason = "initial_buy"
    score_active = False
    selected_defensive = params.defensive.fallback_asset
    defensive_weights: dict[str, float] = {selected_defensive: 1.0}
    defensive_months_held = 0
    defensive_switch_count = 0
    midmonth_fallback_count = 0
    current_defensive_scores: dict[str, float] = {}
    cash = INITIAL_CAPITAL
    shares: dict[str, float] = {}
    previous_closes: dict[str, float] = {}
    last_nav = 0.0
    current_target: dict[str, float] = {}
    overlay_was_active = False
    rows: list[dict[str, object]] = []
    trades: list[dict[str, object]] = []

    for index, row in frame.iterrows():
        timestamp = pd.Timestamp(row["date"])
        open_prices = {
            asset: float(asset_frame.at[timestamp, "open"])
            for asset, asset_frame in indexed.items()
            if timestamp in asset_frame.index and float(asset_frame.at[timestamp, "open"]) > 0
        }
        mark_open = {
            asset: (_asof_price(asset_frame, timestamp, "close") or 0.0)
            for asset, asset_frame in indexed.items()
        }
        mark_open.update(open_prices)
        day_gross_pnl: dict[str, float] = {}
        day_cost: dict[str, float] = {}
        for asset, quantity in shares.items():
            if asset in previous_closes:
                pnl = quantity * (
                    mark_open.get(asset, previous_closes[asset]) - previous_closes[asset]
                )
                day_gross_pnl[asset] = day_gross_pnl.get(asset, 0.0) + pnl

        previous_timestamp = (
            pd.Timestamp(frame.iloc[index - 1]["date"]) if index > 0 else None
        )
        month_changed = (
            index == 0
            or timestamp.to_period("M") != previous_timestamp.to_period("M")
        )
        defensive_changed = False
        if month_changed:
            previous_defensive = selected_defensive
            previous_weights = defensive_weights
            (
                selected_defensive,
                defensive_weights,
                current_defensive_scores,
            ) = _select_defensive_allocation(
                market,
                previous_timestamp,
                params.defensive,
                set(open_prices),
                selected_defensive,
                defensive_months_held,
            )
            if selected_defensive != previous_defensive:
                defensive_months_held = 1
                defensive_switch_count += 1
            else:
                defensive_months_held += 1
            defensive_changed = defensive_weights != previous_weights

        rebalance_reason: str | None = None
        if pending_primary_target is not None:
            primary_target = pending_primary_target
            pending_primary_target = None
            rebalance_reason = pending_reason
        elif defensive_changed and primary_target < params.grid.max_exposure:
            rebalance_reason = "defensive_rotation"

        overlay_active = bool(overlay_frame.at[timestamp, "overlay_active"])
        effective_primary_target = (
            min(primary_target, params.market_risk_overlay.primary_cap)
            if overlay_active
            else primary_target
        )
        overlay_changes_target = (
            overlay_active != overlay_was_active
            and effective_primary_target
            != current_target.get(PRIMARY_ASSET, effective_primary_target)
        )
        if overlay_changes_target and rebalance_reason is None:
            rebalance_reason = (
                "market_risk_overlay_on" if overlay_active else "market_risk_overlay_off"
            )

        if (
            params.defensive.midmonth_fallback
            and not month_changed
            and rebalance_reason is not None
        ):
            defensive_weights, used_fallback = _redistribute_unavailable_defensive_weights(
                defensive_weights,
                current_defensive_scores,
                set(open_prices),
                params.defensive,
            )
            if used_fallback:
                selected_defensive = max(defensive_weights, key=defensive_weights.get)
                defensive_months_held = 1
                defensive_switch_count += 1
                midmonth_fallback_count += 1

        defensive_budget = params.grid.max_exposure - effective_primary_target
        next_target = {PRIMARY_ASSET: effective_primary_target}
        next_target.update({
            asset: defensive_budget * weight
            for asset, weight in defensive_weights.items()
        })
        next_target = {asset: weight for asset, weight in next_target.items() if weight > 1e-14}
        if rebalance_reason is not None or next_target != current_target:
            cash, shares, executions = _execute_portfolio_target(
                cash,
                shares,
                next_target,
                open_prices,
                mark_open,
                applied_cost_rates,
            )
            for execution in executions:
                asset = str(execution["asset"])
                day_cost[asset] = day_cost.get(asset, 0.0) + float(execution["cost"])
                trades.append({
                    "date": timestamp,
                    "reason": rebalance_reason or "target_change",
                    "old_target": current_target.get(asset, 0.0),
                    "new_target": next_target.get(asset, 0.0),
                    **execution,
                })
            current_target = next_target
        overlay_was_active = overlay_active

        close_prices = {
            asset: (_asof_price(asset_frame, timestamp, "close") or 0.0)
            for asset, asset_frame in indexed.items()
        }
        for asset, quantity in shares.items():
            if asset in open_prices:
                pnl = quantity * (
                    close_prices.get(asset, open_prices[asset]) - open_prices[asset]
                )
                day_gross_pnl[asset] = day_gross_pnl.get(asset, 0.0) + pnl
        nav = cash + sum(
            quantity * close_prices.get(asset, 0.0) for asset, quantity in shares.items()
        )
        deposit = INITIAL_CAPITAL if index == 0 else 0.0
        daily_return = (nav - deposit) / last_nav - 1.0 if last_nav > 0 else np.nan
        actual_weights = {
            asset: quantity * close_prices.get(asset, 0.0) / nav
            for asset, quantity in shares.items()
        } if nav > 0 else {}
        score_row = score_frame.iloc[index]
        daily_row: dict[str, object] = {
            "date": timestamp,
            "nav": nav,
            "return": daily_return,
            "cash": cash,
            "etf_weight": actual_weights.get(PRIMARY_ASSET, 0.0),
            "primary_weight": actual_weights.get(PRIMARY_ASSET, 0.0),
            "defensive_weight": sum(actual_weights.get(asset, 0.0) for asset in params.defensive.assets),
            "target_weight": effective_primary_target,
            "base_target_weight": primary_target,
            "selected_defensive_asset": selected_defensive,
            "defensive_months_held": defensive_months_held,
            "grid_target_weight": grid_target,
            "realized_volatility": realized_volatility[index],
            "volatility_cap": _volatility_cap(realized_volatility[index], params.grid),
            "momentum_component": score_row["momentum_component"],
            "score_volatility": score_row["score_volatility"],
            "volatility_score": score_row["volatility_score"],
            "momentum_score": score_row["momentum_score"],
            "regime_momentum": score_row["regime_momentum"],
            "score_active": score_active,
            "market_risk_overlay_active": overlay_active,
            "market_risk_relative_stretched": bool(
                overlay_frame.at[timestamp, "relative_stretched"]
            ),
            "market_risk_nasdaq_leadership": bool(
                overlay_frame.at[timestamp, "nasdaq_leadership"]
            ),
            "market_risk_relative_price_percentile": overlay_frame.at[
                timestamp, "relative_price_percentile"
            ],
            "market_risk_nasdaq_sp500_relative_return": overlay_frame.at[
                timestamp, "nasdaq_sp500_relative_return"
            ],
        }
        for asset in (PRIMARY_ASSET, *params.defensive.assets):
            daily_row[f"weight_{asset}"] = actual_weights.get(asset, 0.0)
            daily_row[f"target_{asset}"] = current_target.get(asset, 0.0)
            daily_row[f"defensive_score_{asset}"] = current_defensive_scores.get(asset, np.nan)
            gross = day_gross_pnl.get(asset, 0.0)
            cost = day_cost.get(asset, 0.0)
            daily_row[f"gross_pnl_{asset}"] = gross
            daily_row[f"transaction_cost_{asset}"] = cost
            daily_row[f"net_pnl_{asset}"] = gross - cost
        rows.append(daily_row)
        last_nav = nav
        previous_closes = close_prices

        if index < len(frame) - 1:
            next_grid_target, grid_reason, location = _target_from_close(
                closes, index, grid_target, params.grid
            )
            score = float(score_row["momentum_score"])
            regime = float(score_row["regime_momentum"])
            threshold = params.score.exit_threshold if score_active else params.score.entry_threshold
            next_score_active = (
                np.isfinite(score)
                and np.isfinite(regime)
                and regime > params.score.regime_threshold
                and score > threshold
            )
            if next_score_active:
                next_grid_target = params.grid.max_exposure
                grid_reason = "stable_momentum_full"
            cap = _volatility_cap(realized_volatility[index], params.grid)
            next_primary_target = (
                params.grid.max_exposure if next_score_active else min(next_grid_target, cap)
            )
            if next_primary_target != primary_target:
                pending_primary_target = next_primary_target
                if next_score_active:
                    pending_reason = "stable_momentum_full"
                elif next_grid_target != grid_target:
                    pending_reason = grid_reason
                elif next_primary_target < primary_target:
                    pending_reason = "volatility_cap"
                else:
                    pending_reason = "volatility_release"
                trades.append({
                    "date": timestamp,
                    "asset": PRIMARY_ASSET,
                    "side": "signal",
                    "reason": pending_reason,
                    "old_target": primary_target,
                    "new_target": next_primary_target,
                    "old_grid_target": grid_target,
                    "new_grid_target": next_grid_target,
                    "signal_close": float(row["close"]),
                    "range_location": location,
                    "realized_volatility": realized_volatility[index],
                    "volatility_cap": cap,
                    "momentum_score": score,
                    "regime_momentum": regime,
                })
            grid_target = next_grid_target
            score_active = next_score_active

    daily = pd.DataFrame(rows).set_index("date")
    trade_frame = pd.DataFrame(trades)
    returns = daily["return"].dropna().astype(float)
    curve = daily["nav"] / INITIAL_CAPITAL
    drawdown = curve / curve.cummax() - 1.0
    stdev = float(returns.std(ddof=1)) if len(returns) > 1 else 0.0
    years = len(returns) / TRADING_DAYS
    execution_mask = trade_frame["side"] != "signal" if not trade_frame.empty else pd.Series(dtype=bool)
    metrics: dict[str, float | int | str] = {
        "asset": PRIMARY_ASSET,
        "defensive_assets": ",".join(params.defensive.assets),
        "start": str(daily.index.min().date()),
        "end": str(daily.index.max().date()),
        "observations": int(len(returns)),
        "final_nav": float(daily["nav"].iloc[-1]),
        "total_return": float(daily["nav"].iloc[-1] / INITIAL_CAPITAL - 1.0),
        "annualized_return": float((daily["nav"].iloc[-1] / INITIAL_CAPITAL) ** (1.0 / years) - 1.0),
        "max_drawdown": float(drawdown.min()),
        "sharpe": float(returns.mean() / stdev * np.sqrt(TRADING_DAYS)) if stdev > 0 else 0.0,
        "average_exposure": float(daily["primary_weight"].mean()),
        "average_defensive_exposure": float(daily["defensive_weight"].mean()),
        "defensive_switch_count": defensive_switch_count,
        "midmonth_fallback_count": midmonth_fallback_count,
        "score_active_days": int(daily["score_active"].sum()),
        "market_risk_overlay_days": int(daily["market_risk_overlay_active"].sum()),
        "signal_count": int((trade_frame["side"] == "signal").sum()) if not trade_frame.empty else 0,
        "execution_count": int(execution_mask.sum()) if not trade_frame.empty else 0,
        "total_turnover": float(trade_frame.loc[execution_mask, "turnover"].sum()) if not trade_frame.empty else 0.0,
        "primary_turnover": float(
            trade_frame.loc[
                execution_mask & (trade_frame["asset"] == PRIMARY_ASSET), "turnover"
            ].sum()
        ) if not trade_frame.empty else 0.0,
        "defensive_turnover": float(
            trade_frame.loc[
                execution_mask & trade_frame["asset"].isin(params.defensive.assets),
                "turnover",
            ].sum()
        ) if not trade_frame.empty else 0.0,
        "total_cost": float(trade_frame.loc[execution_mask, "cost"].sum()) if not trade_frame.empty else 0.0,
        "transaction_cost_rate_512890": _asset_cost_rate(
            PRIMARY_ASSET, applied_cost_rates
        ),
        "transaction_cost_rate_511260": _asset_cost_rate(
            "511260.SH", applied_cost_rates
        ),
        "transaction_cost_rate_511360": _asset_cost_rate(
            "511360.SH", applied_cost_rates
        ),
        "transaction_cost_rate_511880": _asset_cost_rate(
            "511880.SH", applied_cost_rates
        ),
    }
    return daily, trade_frame, metrics


def main() -> None:
    daily, trades, metrics = run_backtest()
    output = Path(__file__).parent / "deliverable"
    output.mkdir(parents=True, exist_ok=True)
    daily.to_csv(output / "defender_opt_v2_daily.csv")
    trades.to_csv(output / "defender_opt_v2_trades.csv", index=False)
    pd.Series(metrics).to_json(
        output / "defender_opt_v2_metrics.json",
        force_ascii=False,
        indent=2,
    )
    print("params", asdict(DefenderOptV2Params()))
    for key, value in metrics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
