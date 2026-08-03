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
from strategy.rebalance import normalize_rebalance_mode, should_hold_position


@dataclass
class BacktestResult:
    daily_returns: pd.Series       # strategy daily returns (full period)
    benchmark_returns: pd.Series   # equal-weight benchmark daily returns
    positions: pd.DataFrame        # daily position weights (date × asset)
    train_end: date                # train set end date
    config: dict                   # original config snapshot
    baseline_strategy_name: str | None = None  # populated by callers when comparing against a baseline run
    gross_daily_returns: pd.Series | None = None  # pre-cost strategy returns
    turnover: pd.Series | None = None  # executed Σ|Δw| by rebalance date
    costs: pd.Series | None = None  # daily cost deductions aligned to daily_returns


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


def _turnover_between(
    target: dict[str, float],
    current: dict[str, float],
) -> float:
    """Executed turnover as Σ_assets |w_target - w_current|."""
    assets = set(target) | set(current)
    return float(sum(abs(target.get(asset, 0.0) - current.get(asset, 0.0)) for asset in assets))


def _should_hold_position(
    current_weights: dict[str, float],
    holding_days: int | None,
    rebalance_days: int,
    rebalance_mode: str | None = "min_hold",
) -> bool:
    return should_hold_position(
        current_weights,
        holding_days,
        rebalance_days,
        rebalance_mode,
    )


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
    rebalance_mode = normalize_rebalance_mode(config.get("rebalance_mode"))
    min_hold_drawdown_threshold = config.get("min_hold_drawdown_threshold")
    if min_hold_drawdown_threshold is not None:
        min_hold_drawdown_threshold = float(min_hold_drawdown_threshold)
        if not 0 < min_hold_drawdown_threshold <= 1:
            raise ValueError(
                "min_hold_drawdown_threshold must be in (0, 1], got "
                f"{min_hold_drawdown_threshold}"
            )
    transaction_cost_rate = float(
        config.get(
            "transaction_cost_rate",
            config.get("cost_rate", config.get("fee", 0.0)),
        )
        or 0.0
    )
    if transaction_cost_rate < 0:
        raise ValueError(
            f"transaction_cost_rate must be >= 0, got {transaction_cost_rate}"
        )

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
    gross_returns: list[tuple[pd.Timestamp, float]] = []
    benchmark_returns: list[tuple[pd.Timestamp, float]] = []
    turnover_records: list[tuple[pd.Timestamp, float]] = []
    cost_records: list[tuple[pd.Timestamp, float]] = []

    current_weights: dict[str, float] = {}
    current_entry_idx: int | None = None
    current_position_wealth: float | None = None
    current_position_peak: float | None = None
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
                executed_turnover = _turnover_between(current_weights, old_weights)
                turnover_records.append((t, executed_turnover))
                pending_weights = None
                pending_entry_idx = None
                intraday_ret = _weighted_return_between(
                    current_weights, open_prices, close_prices, t, t
                )
                current_position_wealth = (
                    1.0 + intraday_ret if intraday_ret is not None else 1.0
                )
                current_position_peak = max(1.0, current_position_wealth)
                strat_ret = _chain_returns(overnight_ret, intraday_ret)
            elif current_weights:
                strat_ret = _weighted_return_between(
                    current_weights, close_prices, close_prices, prev_t, t
                )
                if strat_ret is not None:
                    current_position_wealth = (
                        (current_position_wealth or 1.0) * (1.0 + strat_ret)
                    )
                    current_position_peak = max(
                        current_position_peak or 1.0,
                        current_position_wealth,
                    )
            else:
                strat_ret = None

            if strat_ret is not None:
                gross_returns.append((t, strat_ret))
                cost = executed_turnover * transaction_cost_rate if opened_today else 0.0
                if cost:
                    cost_records.append((t, cost))
                strategy_returns.append((t, strat_ret - cost))

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
        position_drawdown = (
            current_position_wealth / current_position_peak - 1.0
            if current_position_wealth is not None
            and current_position_peak is not None
            and current_position_peak > 0
            else 0.0
        )
        factor_unlock = False
        factor_unlock_config = config.get("min_hold_factor_unlock")
        if (
            factor_unlock_config
            and rebalance_mode == "min_hold"
            and current_weights
            and pending_weights is None
        ):
            primary_asset = max(current_weights, key=current_weights.get)
            unlock_factor_name = factor_unlock_config["factor"]
            unlock_threshold = factor_unlock_config.get(
                "thresholds",
                {},
            ).get(primary_asset, factor_unlock_config.get("threshold"))
            unlock_factor_config = next(
                (
                    fc
                    for fc in factor_configs
                    if fc["name"] == unlock_factor_name
                ),
                None,
            )
            if unlock_threshold is not None and unlock_factor_config is not None:
                unlock_df = asset_data[primary_asset]
                unlock_truncated = unlock_df.loc[unlock_df["date"] <= t]
                unlock_module = all_factors[unlock_factor_name]
                unlock_params = unlock_factor_config.get("params")
                unlock_asset_params = unlock_factor_config.get(
                    "params_by_asset",
                    {},
                ).get(primary_asset)
                if unlock_asset_params:
                    unlock_params = {
                        **(unlock_params or {}),
                        **unlock_asset_params,
                    }
                unlock_series = unlock_module["compute"](
                    unlock_truncated.copy(),
                    unlock_params,
                )
                if len(unlock_series) > 0 and pd.notna(unlock_series.iloc[-1]):
                    factor_unlock = (
                        float(unlock_series.iloc[-1])
                        < float(unlock_threshold)
                    )
        drawdown_unlock = (
            min_hold_drawdown_threshold is not None
            and rebalance_mode == "min_hold"
            and holding_days is not None
            and holding_days < rebalance_days
            and position_drawdown <= -min_hold_drawdown_threshold
        )
        should_signal = (
            pending_weights is None
            and (
                drawdown_unlock
                or factor_unlock
                or not _should_hold_position(
                    current_weights,
                    holding_days,
                    rebalance_days,
                    rebalance_mode,
                )
            )
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
                    asset_params = fc.get("params_by_asset", {}).get(asset)
                    if asset_params:
                        params = {**(params or {}), **asset_params}
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
            new_weights = strategy.generate_weights(asset_factor_values)

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

    if gross_returns:
        gross_dates, gross_vals = zip(*gross_returns)
        gross_daily_ret = pd.Series(
            gross_vals,
            index=pd.DatetimeIndex(gross_dates),
            dtype=float,
        )
    else:
        gross_daily_ret = pd.Series(dtype=float)

    if benchmark_returns:
        bench_dates, bench_vals = zip(*benchmark_returns)
        bench_ret = pd.Series(bench_vals, index=pd.DatetimeIndex(bench_dates), dtype=float)
    else:
        bench_ret = pd.Series(dtype=float)

    positions_df = pd.DataFrame(positions_records)
    if len(positions_df) > 0:
        positions_df = positions_df.set_index("date")

    if turnover_records:
        turnover = pd.Series(
            dict(turnover_records),
            index=pd.DatetimeIndex([dt for dt, _ in turnover_records]),
            dtype=float,
        )
    else:
        turnover = pd.Series(dtype=float)

    if len(daily_ret) > 0:
        costs = pd.Series(0.0, index=daily_ret.index, dtype=float)
        for dt, cost in cost_records:
            if dt in costs.index:
                costs.loc[dt] = cost
    else:
        costs = pd.Series(dtype=float)

    result = BacktestResult(
        daily_returns=daily_ret,
        benchmark_returns=bench_ret,
        positions=positions_df,
        train_end=train_end_date,
        config=config,
        gross_daily_returns=gross_daily_ret,
        turnover=turnover,
        costs=costs,
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
