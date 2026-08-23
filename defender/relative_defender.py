"""Portable percentile-grid Defender with a fixed bond sleeve.

The strategy deliberately removes the asset-fitted breakout, stable-score,
fixed trailing-stop, and golden-pit mechanisms.  Its primary sleeve is driven
only by the primary ETF's position in its own rolling price range, while a
common volatility budget caps risk.  Signals use the close and execute at the
next open.  The unallocated sleeve is always invested in 511260.SH.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import date
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from data.store import read_local

from .defender_opt_v2 import (
    CostRateSpec,
    _asof_price,
    _asset_cost_rate,
    _execute_portfolio_target,
    _indexed_market,
)
from .grid_reproduction import (
    INITIAL_CAPITAL,
    TRADING_DAYS,
    GridParams,
    _realized_volatility,
)


PRIMARY_ASSET = "512890.SH"
DEFENSIVE_ASSET = "511260.SH"
RELATIVE_DEFENDER_COST_RATES: dict[str, float] = {
    PRIMARY_ASSET: 0.0001,
    "159545.SZ": 0.0001,
    "513530.SH": 0.0001,
    "515080.SH": 0.0001,
    "510880.SH": 0.0001,
    "563020.SH": 0.0001,
    DEFENSIVE_ASSET: 0.00001,
}


@dataclass(frozen=True)
class RelativeDefenderParams:
    """Low-dimensional rule selected across six equally weighted ETFs."""

    range_window: int = 40
    lower_percentile: float = 0.10
    upper_percentile: float = 0.95
    exposure_step: float = 0.25
    volatility_method: str = "rogers_satchell"
    volatility_window: int = 20
    target_volatility: float = 0.18
    min_exposure: float = 0.0
    max_exposure: float = 1.0
    primary_asset: str = PRIMARY_ASSET
    defensive_asset: str = DEFENSIVE_ASSET

    def __post_init__(self) -> None:
        if self.range_window < 2:
            raise ValueError("range_window must be at least 2")
        if not 0.0 <= self.lower_percentile < self.upper_percentile <= 1.0:
            raise ValueError("range percentiles must satisfy 0 <= lower < upper <= 1")
        if not 0.0 < self.exposure_step <= 1.0:
            raise ValueError("exposure_step must lie in (0, 1]")
        if not 0.0 <= self.min_exposure < self.max_exposure <= 1.0:
            raise ValueError("exposure limits must satisfy 0 <= min < max <= 1")
        steps = (self.max_exposure - self.min_exposure) / self.exposure_step
        if not np.isclose(steps, round(steps), atol=1e-12):
            raise ValueError("exposure range must be divisible by exposure_step")
        if self.volatility_window < 2:
            raise ValueError("volatility_window must be at least 2")
        if self.target_volatility <= 0.0:
            raise ValueError("target_volatility must be positive")
        if self.primary_asset == self.defensive_asset:
            raise ValueError("primary and defensive assets must differ")


@dataclass(frozen=True)
class PortfolioSimulation:
    daily: pd.DataFrame
    trades: pd.DataFrame
    metrics: dict[str, float | int]


def relative_defender_params(
    primary_asset: str = PRIMARY_ASSET,
) -> RelativeDefenderParams:
    """Return the same fixed rule for a requested portability-test asset."""
    return replace(RelativeDefenderParams(), primary_asset=primary_asset)


def load_relative_defender_market(
    params: RelativeDefenderParams = RelativeDefenderParams(),
    start: date = date(2019, 1, 18),
    end: date | None = None,
) -> dict[str, pd.DataFrame]:
    """Load primary and 511260 history without synthetic pre-listing rows."""
    result: dict[str, pd.DataFrame] = {}
    for asset in (params.primary_asset, params.defensive_asset):
        frame = read_local(asset)
        if frame is None or frame.empty:
            raise RuntimeError(f"missing local data for {asset}")
        frame = frame.copy()
        frame["date"] = pd.to_datetime(frame["date"])
        frame = frame.loc[frame["date"] >= pd.Timestamp(start)]
        if end is not None:
            frame = frame.loc[frame["date"] <= pd.Timestamp(end)]
        frame = frame.sort_values("date").drop_duplicates("date")
        required = ["date", "open", "high", "low", "close"]
        if frame.empty or frame[required].isna().any().any():
            raise ValueError(f"price data for {asset} is empty or incomplete")
        if (frame[["open", "high", "low", "close"]] <= 0.0).any().any():
            raise ValueError(f"price data for {asset} contains non-positive OHLC")
        result[asset] = frame.reset_index(drop=True)
    return result


def _volatility_cap(
    realized_volatility: float,
    params: RelativeDefenderParams,
) -> float:
    if not np.isfinite(realized_volatility) or realized_volatility <= 0.0:
        return params.max_exposure
    raw = min(
        params.max_exposure,
        params.target_volatility / realized_volatility,
    )
    stepped = (
        np.floor(
            (raw - params.min_exposure) / params.exposure_step + 1e-12
        )
        * params.exposure_step
        + params.min_exposure
    )
    return float(
        np.clip(stepped, params.min_exposure, params.max_exposure)
    )


def target_schedule(
    prices: pd.DataFrame,
    params: RelativeDefenderParams = RelativeDefenderParams(),
) -> pd.DataFrame:
    """Build the causal close-signal/next-open primary target schedule."""
    frame = prices.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    close = frame["close"].astype(float)
    rolling_low = close.rolling(params.range_window).min()
    rolling_high = close.rolling(params.range_window).max()
    width = rolling_high - rolling_low
    location = (close - rolling_low) / width.replace(0.0, np.nan)
    location = location.where(width != 0.0, 0.5)
    volatility = _realized_volatility(
        frame,
        GridParams(
            volatility_method=params.volatility_method,
            volatility_window=params.volatility_window,
        ),
    )

    grid_target = params.max_exposure
    signal_rows: list[dict[str, object]] = []
    for position, timestamp in enumerate(pd.DatetimeIndex(frame["date"])):
        location_value = float(location.iloc[position])
        reason = "warmup" if not np.isfinite(location_value) else "hold"
        if np.isfinite(location_value):
            if location_value <= params.lower_percentile:
                next_grid = min(
                    params.max_exposure,
                    grid_target + params.exposure_step,
                )
                reason = "relative_low_add"
            elif location_value >= params.upper_percentile:
                next_grid = max(
                    params.min_exposure,
                    grid_target - params.exposure_step,
                )
                reason = "relative_high_reduce"
            else:
                next_grid = grid_target
        else:
            next_grid = grid_target

        cap = _volatility_cap(float(volatility[position]), params)
        next_target = min(next_grid, cap)
        if next_target < next_grid:
            reason = "volatility_cap"
        elif (
            signal_rows
            and next_grid == grid_target
            and next_target > float(signal_rows[-1]["signal_primary_target"])
        ):
            if next_grid == grid_target:
                reason = "volatility_release"

        signal_rows.append({
            "date": pd.Timestamp(timestamp),
            "range_location": location_value,
            "realized_volatility": float(volatility[position]),
            "volatility_cap": cap,
            "signal_grid_target": next_grid,
            "signal_primary_target": next_target,
            "signal_reason": reason,
        })
        grid_target = next_grid

    signals = pd.DataFrame(signal_rows).set_index("date")
    schedule = signals.copy()
    schedule["primary_target"] = signals["signal_primary_target"].shift(1)
    schedule["execution_reason"] = signals["signal_reason"].shift(1)
    schedule.iloc[0, schedule.columns.get_loc("primary_target")] = params.max_exposure
    schedule.iloc[0, schedule.columns.get_loc("execution_reason")] = "initial_buy"
    schedule["primary_target"] = schedule["primary_target"].astype(float)
    schedule["defensive_target"] = 1.0 - schedule["primary_target"]
    return schedule


def _performance_metrics(daily: pd.DataFrame) -> dict[str, float | int]:
    returns = daily["return"].dropna().astype(float)
    curve = daily["nav"].astype(float) / INITIAL_CAPITAL
    drawdown = curve / curve.cummax() - 1.0
    stdev = float(returns.std(ddof=1)) if len(returns) > 1 else 0.0
    years = len(returns) / TRADING_DAYS
    return {
        "observations": int(len(returns)),
        "final_nav": float(daily["nav"].iloc[-1]),
        "total_return": float(curve.iloc[-1] - 1.0),
        "annualized_return": float(curve.iloc[-1] ** (1.0 / years) - 1.0),
        "annualized_volatility": float(stdev * np.sqrt(TRADING_DAYS)),
        "sharpe": (
            float(returns.mean() / stdev * np.sqrt(TRADING_DAYS))
            if stdev
            else 0.0
        ),
        "max_drawdown": float(drawdown.min()),
    }


def _simulate(
    market: Mapping[str, pd.DataFrame],
    schedule: pd.DataFrame,
    params: RelativeDefenderParams,
    cost_rates: CostRateSpec,
) -> PortfolioSimulation:
    indexed = _indexed_market(market)
    calendar = pd.DatetimeIndex(schedule.index)
    cash = INITIAL_CAPITAL
    shares: dict[str, float] = {}
    previous_target: dict[str, float] = {}
    previous_closes: dict[str, float] = {}
    last_nav = 0.0
    rows: list[dict[str, object]] = []
    trades: list[dict[str, object]] = []
    total_cost = 0.0
    gross_pnl = {params.primary_asset: 0.0, params.defensive_asset: 0.0}

    for position, timestamp in enumerate(calendar):
        timestamp = pd.Timestamp(timestamp)
        primary_weight = float(schedule.at[timestamp, "primary_target"])
        open_prices = {
            asset: float(frame.at[timestamp, "open"])
            for asset, frame in indexed.items()
            if timestamp in frame.index
            and pd.notna(frame.at[timestamp, "open"])
            and float(frame.at[timestamp, "open"]) > 0.0
        }
        mark_open = {
            asset: (_asof_price(frame, timestamp, "close") or 0.0)
            for asset, frame in indexed.items()
        }
        mark_open.update(open_prices)
        day_gross: dict[str, float] = {}
        day_cost: dict[str, float] = {}
        for asset, quantity in shares.items():
            if asset in previous_closes:
                pnl = quantity * (
                    mark_open.get(asset, previous_closes[asset])
                    - previous_closes[asset]
                )
                gross_pnl[asset] = gross_pnl.get(asset, 0.0) + pnl
                day_gross[asset] = day_gross.get(asset, 0.0) + pnl

        target = {
            params.primary_asset: primary_weight,
            params.defensive_asset: 1.0 - primary_weight,
        }
        target = {
            asset: weight for asset, weight in target.items() if weight > 1e-14
        }
        if target != previous_target:
            cash, shares, executions = _execute_portfolio_target(
                cash,
                shares,
                target,
                open_prices,
                mark_open,
                cost_rates,
            )
            for execution in executions:
                asset = str(execution["asset"])
                cost = float(execution["cost"])
                total_cost += cost
                day_cost[asset] = day_cost.get(asset, 0.0) + cost
                trades.append({
                    "date": timestamp,
                    "reason": schedule.at[timestamp, "execution_reason"],
                    "primary_target": primary_weight,
                    **execution,
                })
            previous_target = target

        close_prices = {
            asset: (_asof_price(frame, timestamp, "close") or 0.0)
            for asset, frame in indexed.items()
        }
        for asset, quantity in shares.items():
            if asset in open_prices:
                pnl = quantity * (
                    close_prices.get(asset, open_prices[asset])
                    - open_prices[asset]
                )
                gross_pnl[asset] = gross_pnl.get(asset, 0.0) + pnl
                day_gross[asset] = day_gross.get(asset, 0.0) + pnl

        nav = cash + sum(
            quantity * close_prices.get(asset, 0.0)
            for asset, quantity in shares.items()
        )
        daily_return = nav / last_nav - 1.0 if position > 0 else np.nan
        actual_weights = (
            {
                asset: quantity * close_prices.get(asset, 0.0) / nav
                for asset, quantity in shares.items()
            }
            if nav > 0.0
            else {}
        )
        row: dict[str, object] = {
            "date": timestamp,
            "nav": nav,
            "return": daily_return,
            "cash": cash,
            **schedule.loc[timestamp].to_dict(),
        }
        for asset in (params.primary_asset, params.defensive_asset):
            row[f"weight_{asset}"] = actual_weights.get(asset, 0.0)
            row[f"target_{asset}"] = target.get(asset, 0.0)
            gross = day_gross.get(asset, 0.0)
            cost = day_cost.get(asset, 0.0)
            row[f"gross_pnl_{asset}"] = gross
            row[f"transaction_cost_{asset}"] = cost
            row[f"net_pnl_{asset}"] = gross - cost
        rows.append(row)
        last_nav = nav
        previous_closes = close_prices

    daily = pd.DataFrame(rows).set_index("date")
    trade_frame = pd.DataFrame(trades)
    metrics = _performance_metrics(daily)
    metrics.update({
        "execution_count": int(len(trade_frame)),
        "total_turnover": (
            float(trade_frame["turnover"].sum())
            if not trade_frame.empty
            else 0.0
        ),
        "total_cost": total_cost,
        "average_primary_target": float(schedule["primary_target"].mean()),
    })
    for asset in (params.primary_asset, params.defensive_asset):
        code = asset.split(".", maxsplit=1)[0]
        metrics[f"gross_pnl_{code}"] = gross_pnl.get(asset, 0.0)
    return PortfolioSimulation(daily, trade_frame, metrics)


def run_backtest(
    market: Mapping[str, pd.DataFrame] | None = None,
    params: RelativeDefenderParams | None = None,
    cost_rate: CostRateSpec | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float | int | str], pd.DataFrame]:
    """Run the portable rule with one primary ETF and fixed 511260 filling."""
    selected = relative_defender_params() if params is None else params
    prices = (
        load_relative_defender_market(selected)
        if market is None
        else {asset: frame.copy() for asset, frame in market.items()}
    )
    missing = {
        selected.primary_asset,
        selected.defensive_asset,
    } - set(prices)
    if missing:
        raise RuntimeError(f"missing local data for: {', '.join(sorted(missing))}")
    applied_costs = (
        RELATIVE_DEFENDER_COST_RATES if cost_rate is None else cost_rate
    )
    schedule = target_schedule(prices[selected.primary_asset], selected)
    simulation = _simulate(prices, schedule, selected, applied_costs)
    daily = simulation.daily
    metrics: dict[str, float | int | str] = {
        **simulation.metrics,
        "strategy": "relative_defender",
        "start": str(daily.index.min().date()),
        "end": str(daily.index.max().date()),
        **asdict(selected),
        "transaction_cost_rate_primary": _asset_cost_rate(
            selected.primary_asset, applied_costs
        ),
        "transaction_cost_rate_defensive": _asset_cost_rate(
            selected.defensive_asset, applied_costs
        ),
    }
    return daily, simulation.trades, metrics, schedule


def main() -> None:
    daily, trades, metrics, schedule = run_backtest()
    output = Path(__file__).parent / "deliverable"
    output.mkdir(parents=True, exist_ok=True)
    daily.to_csv(output / "relative_defender_daily.csv")
    trades.to_csv(output / "relative_defender_trades.csv", index=False)
    schedule.to_csv(output / "relative_defender_signals.csv")
    print("params", asdict(relative_defender_params()))
    for key, value in metrics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
