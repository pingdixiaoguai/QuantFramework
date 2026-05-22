"""Backtest engine — time-series traversal with future-info truncation."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from data.store import query
from factors.registry import load_registered_factors
from factors.validator import validate
from strategy.loader import load_strategy


@dataclass
class BacktestResult:
    daily_returns: pd.Series       # strategy daily returns (full period)
    benchmark_returns: pd.Series   # equal-weight benchmark daily returns
    positions: pd.DataFrame        # daily position weights (date × asset)
    train_end: date                # train set end date
    config: dict                   # original config snapshot
    baseline_strategy_name: str | None = None  # populated by callers when comparing against a baseline run


def _default_config() -> dict:
    return {
        "strategy_name": "momentum_rotation",
        "asset_pool": ["510300.SH", "159915.SZ", "513100.SH", "518880.SH"],
        "start": date(2016, 1, 1),
        "end": date.today(),
        "factors": [
            {"name": "momentum", "weight": 0.7, "params": {"window": 20}},
            {"name": "volatility", "weight": 0.3, "direction_flip": True, "params": {"window": 20}},
        ],
        "train_ratio": 0.7,
        "rebalance_rule": "daily",
    }


def _weighted_return_between(
    weights: dict[str, float],
    start_prices: dict[str, pd.Series],
    end_prices: dict[str, pd.Series],
    start_t: pd.Timestamp,
    end_t: pd.Timestamp,
) -> float | None:
    """Weighted simple return from start_t prices to end_t prices."""
    if not weights:
        return None

    total = 0.0
    has_price = False
    for asset, weight in weights.items():
        start = start_prices.get(asset)
        end = end_prices.get(asset)
        if start is None or end is None:
            continue
        if start_t not in start.index or end_t not in end.index:
            continue
        total += weight * (end[end_t] / start[start_t] - 1)
        has_price = True

    return total if has_price else None


def _equal_weight_return_between(
    asset_pool: list[str],
    start_prices: dict[str, pd.Series],
    end_prices: dict[str, pd.Series],
    start_t: pd.Timestamp,
    end_t: pd.Timestamp,
) -> float | None:
    returns = []
    for asset in asset_pool:
        start = start_prices.get(asset)
        end = end_prices.get(asset)
        if start is None or end is None:
            continue
        if start_t not in start.index or end_t not in end.index:
            continue
        returns.append(end[end_t] / start[start_t] - 1)

    return float(np.mean(returns)) if returns else None


def _chain_returns(*returns: float | None) -> float | None:
    product = 1.0
    has_return = False
    for ret in returns:
        if ret is None:
            continue
        product *= 1 + ret
        has_return = True
    return product - 1 if has_return else None


def _should_hold_position(
    current_weights: dict[str, float],
    holding_days: int | None,
    rebalance_days: int,
) -> bool:
    if rebalance_days <= 1:
        return False
    if not current_weights:
        return False
    if holding_days is None:
        return True
    return holding_days < rebalance_days


def run(config: dict | None = None) -> BacktestResult:
    """Run a backtest with the given configuration."""
    if config is None:
        config = _default_config()

    asset_pool = config["asset_pool"]
    start = config["start"]
    end = config["end"]
    factor_configs = config["factors"]
    train_ratio = config.get("train_ratio", 0.7)
    rebalance_days = int(config.get("rebalance_days", 1))
    if rebalance_days < 1:
        raise ValueError(f"rebalance_days must be >= 1, got {rebalance_days}")

    # Load strategy and factor modules
    strategy = load_strategy(config)
    all_factors = load_registered_factors()

    # Load data for all assets
    asset_data: dict[str, pd.DataFrame] = {}
    for asset in asset_pool:
        df = query(asset, start, end)
        if len(df) > 0:
            asset_data[asset] = df

    if not asset_data:
        raise RuntimeError("no data available for any asset in the pool")

    # Build union of all trading days
    all_dates: set[pd.Timestamp] = set()
    for df in asset_data.values():
        all_dates.update(df["date"].tolist())
    trading_days = sorted(all_dates)

    # Need at least some history before we can compute factors
    max_min_history = max(
        all_factors[fc["name"]]["METADATA"]["min_history"]
        for fc in factor_configs
    )

    # Train/test split
    split_idx = int(len(trading_days) * train_ratio)
    train_end_date = trading_days[split_idx].date() if split_idx < len(trading_days) else trading_days[-1].date()

    # Pre-compute open/close prices for return calculation
    open_prices: dict[str, pd.Series] = {}
    close_prices: dict[str, pd.Series] = {}
    for asset, df in asset_data.items():
        open_prices[asset] = pd.Series(
            df["open"].values, index=df["date"]
        )
        close_prices[asset] = pd.Series(
            df["close"].values, index=df["date"]
        )

    # Run day-by-day
    positions_records: list[dict] = []
    strategy_returns: list[tuple[pd.Timestamp, float]] = []
    benchmark_returns: list[tuple[pd.Timestamp, float]] = []

    current_weights: dict[str, float] = {}
    current_entry_idx: int | None = None
    pending_weights: dict[str, float] | None = None
    pending_entry_idx: int | None = None

    for day_idx, t in enumerate(trading_days):
        # Open-execution timing:
        # 1. yesterday's position earns the overnight close[t-1] -> open[t];
        # 2. any signal generated at yesterday's close is executed at open[t];
        # 3. the post-open position earns open[t] -> close[t].
        # On non-rebalance days this collapses to the usual close-to-close
        # return for the carried position.
        if day_idx > 0:
            prev_t = trading_days[day_idx - 1]
            old_weights = current_weights
            opened_today = (
                pending_entry_idx == day_idx and pending_weights is not None
            )

            if opened_today:
                overnight_ret = _weighted_return_between(
                    old_weights, close_prices, open_prices, prev_t, t
                )
                current_weights = pending_weights or {}
                current_entry_idx = day_idx
                positions_records.append({"date": t, **current_weights})
                pending_weights = None
                pending_entry_idx = None
                intraday_ret = _weighted_return_between(
                    current_weights, open_prices, close_prices, t, t
                )
                strat_ret = _chain_returns(overnight_ret, intraday_ret)
            elif current_weights:
                strat_ret = _weighted_return_between(
                    current_weights, close_prices, close_prices, prev_t, t
                )
            else:
                strat_ret = None

            if strat_ret is not None:
                strategy_returns.append((t, strat_ret))

                if opened_today and not old_weights:
                    bench_ret = _equal_weight_return_between(
                        asset_pool, open_prices, close_prices, t, t
                    )
                else:
                    bench_ret = _equal_weight_return_between(
                        asset_pool, close_prices, close_prices, prev_t, t
                    )
                if bench_ret is not None:
                    benchmark_returns.append((t, bench_ret))

        holding_days = (
            day_idx - current_entry_idx + 1
            if current_entry_idx is not None and current_weights
            else None
        )
        should_signal = (
            pending_weights is None
            and not _should_hold_position(current_weights, holding_days, rebalance_days)
        )

        if should_signal:
            # Compute factor values for each asset at time t
            asset_factor_values: dict[str, dict[str, float]] = {}

            for asset, df in asset_data.items():
                # Future info truncation: only data up to t
                mask = df["date"] <= t
                truncated = df.loc[mask]

                if len(truncated) < max_min_history:
                    continue

                factor_vals: dict[str, float] = {}
                for fc in factor_configs:
                    fname = fc["name"]
                    fmod = all_factors[fname]
                    params = fc.get("params")
                    try:
                        series = fmod["compute"](truncated.copy(), params)
                        validate(series, truncated, fmod["METADATA"])
                        # Take the latest value (at time t)
                        last_val = series.iloc[-1]
                        if pd.notna(last_val):
                            factor_vals[fname] = float(last_val)
                    except (ValueError, Exception) as exc:
                        warnings.warn(
                            f"factor '{fname}' failed for {asset} on {t}: {exc}",
                            stacklevel=2,
                        )

                if len(factor_vals) == len(factor_configs):
                    asset_factor_values[asset] = factor_vals

            # Delegate to strategy for target weights. These are not effective
            # until the next trading day's open.
            new_weights = strategy.generate_weights(
                asset_factor_values,
                current_weights=current_weights,
            )

            if new_weights and new_weights != current_weights:
                next_idx = day_idx + 1
                if next_idx < len(trading_days):
                    pending_weights = new_weights
                    pending_entry_idx = next_idx
            # else: no executed rebalance happened; keep the current position
            # and re-evaluate on the next day once the hold window has elapsed.

    # Build result series
    if strategy_returns:
        ret_dates, ret_vals = zip(*strategy_returns)
        daily_ret = pd.Series(ret_vals, index=pd.DatetimeIndex(ret_dates), dtype=float)
    else:
        daily_ret = pd.Series(dtype=float)

    if benchmark_returns:
        bench_dates, bench_vals = zip(*benchmark_returns)
        bench_ret = pd.Series(bench_vals, index=pd.DatetimeIndex(bench_dates), dtype=float)
    else:
        bench_ret = pd.Series(dtype=float)

    positions_df = pd.DataFrame(positions_records)
    if len(positions_df) > 0:
        positions_df = positions_df.set_index("date")

    result = BacktestResult(
        daily_returns=daily_ret,
        benchmark_returns=bench_ret,
        positions=positions_df,
        train_end=train_end_date,
        config=config,
    )

    # Overfit warning
    _check_overfit(result)

    return result


def _check_overfit(result: BacktestResult) -> None:
    """Print warning if train Sharpe >> test Sharpe."""
    if len(result.daily_returns) == 0:
        return

    train_end_ts = pd.Timestamp(result.train_end)
    train_ret = result.daily_returns[result.daily_returns.index <= train_end_ts]
    test_ret = result.daily_returns[result.daily_returns.index > train_end_ts]

    if len(train_ret) < 20 or len(test_ret) < 20:
        return

    train_sharpe = _sharpe(train_ret)
    test_sharpe = _sharpe(test_ret)

    if test_sharpe > 0 and train_sharpe > 2 * test_sharpe:
        warnings.warn(
            f"Potential overfitting: train Sharpe ({train_sharpe:.2f}) > "
            f"2x test Sharpe ({test_sharpe:.2f})",
            stacklevel=2,
        )


def _sharpe(returns: pd.Series, risk_free: float = 0.0, periods: int = 252) -> float:
    """Annualized Sharpe ratio."""
    excess = returns - risk_free / periods
    if excess.std() == 0:
        return 0.0
    return float(excess.mean() / excess.std() * np.sqrt(periods))
