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


@dataclass
class BacktestResult:
    daily_returns: pd.Series       # strategy daily returns (full period)
    benchmark_returns: pd.Series   # equal-weight benchmark daily returns
    positions: pd.DataFrame        # daily position weights (date × asset)
    train_end: date                # train set end date
    config: dict                   # original config snapshot


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


def run(config: dict | None = None) -> BacktestResult:
    """Run a backtest with the given configuration."""
    if config is None:
        config = _default_config()

    asset_pool = config["asset_pool"]
    start = config["start"]
    end = config["end"]
    factor_configs = config["factors"]
    train_ratio = config.get("train_ratio", 0.7)

    # Load all factor modules
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

    # Pre-compute close prices for return calculation
    close_prices: dict[str, pd.Series] = {}
    for asset, df in asset_data.items():
        close_prices[asset] = pd.Series(
            df["close"].values, index=df["date"]
        )

    # Run day-by-day
    positions_records: list[dict] = []
    strategy_returns: list[tuple[pd.Timestamp, float]] = []
    benchmark_returns: list[tuple[pd.Timestamp, float]] = []

    prev_weights: dict[str, float] | None = None

    for day_idx, t in enumerate(trading_days):
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

        # Compute weighted rank -> target weights
        weights = _compute_weights(asset_factor_values, factor_configs)

        if weights:
            positions_records.append({"date": t, **weights})

        # Compute returns (weights determined at t, return realized at t+1)
        if prev_weights and day_idx > 0:
            strat_ret = 0.0
            bench_assets = []

            for asset in asset_pool:
                if asset not in close_prices:
                    continue
                cp = close_prices[asset]
                if t in cp.index and trading_days[day_idx - 1] in cp.index:
                    asset_ret = cp[t] / cp[trading_days[day_idx - 1]] - 1
                    bench_assets.append(asset_ret)
                    if asset in prev_weights:
                        strat_ret += prev_weights[asset] * asset_ret

            if bench_assets:
                strategy_returns.append((t, strat_ret))
                benchmark_returns.append((t, float(np.mean(bench_assets))))

        prev_weights = weights if weights else prev_weights

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


def _compute_weights(
    asset_factor_values: dict[str, dict[str, float]],
    factor_configs: list[dict],
) -> dict[str, float]:
    """Compute portfolio weights from factor values via weighted ranking."""
    if not asset_factor_values:
        return {}

    assets = list(asset_factor_values.keys())
    if len(assets) == 1:
        return {assets[0]: 1.0}

    # For each factor, rank assets
    total_scores: dict[str, float] = {a: 0.0 for a in assets}

    for fc in factor_configs:
        fname = fc["name"]
        weight = fc["weight"]
        flip = fc.get("direction_flip", False)

        values = [(a, asset_factor_values[a][fname]) for a in assets]
        # Sort by factor value ascending -> rank 1 = lowest
        values.sort(key=lambda x: x[1])
        n = len(values)

        for rank_idx, (asset, _) in enumerate(values):
            rank = rank_idx + 1  # 1-based
            if flip:
                rank = n - rank + 1  # reverse: highest original = lowest rank
            total_scores[asset] += weight * rank

    # Normalize to weights (higher score = higher weight)
    total = sum(total_scores.values())
    if total == 0:
        return {a: 1.0 / len(assets) for a in assets}

    return {a: score / total for a, score in total_scores.items()}


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
