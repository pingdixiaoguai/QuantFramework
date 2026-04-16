"""Live daily run entry point.

Usage:
    uv run python run_daily.py --config strategy/configs/momentum_rotation.yaml

Requires env vars:
    TUSHARE_TOKEN       — data sync
    DINGTALK_WEBHOOK    — notification
    DINGTALK_SECRET     — (optional) webhook signing
"""

from __future__ import annotations

import argparse
import warnings
from datetime import date
from pathlib import Path

import pandas as pd
import yaml

from data.store import query, read_local
from data.sync import sync_all
from execution.interfaces import diff
from execution.position import (
    PositionPeriod,
    PositionState,
    read_position,
    save_position,
    write_position,
)
from factors.registry import load_registered_factors
from factors.validator import validate
from notification.dingtalk import DingTalkNotifier
from notification.formatter import ASSET_NAMES, NotificationContext, format_notification
from strategy.loader import load_strategy


def _load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    for key in ("start", "end"):
        if key in raw and isinstance(raw[key], str):
            raw[key] = date.fromisoformat(raw[key])
    return raw


_STALE_DAYS = 5  # Max calendar days before data is considered stale


def _sync_and_check(asset_pool: list[str], today: date) -> None:
    """Sync all assets and verify data freshness."""
    print("=== Syncing data ===")
    sync_all(asset_pool)
    print()

    stale = []
    for asset in asset_pool:
        df = read_local(asset)
        if df is None or len(df) == 0:
            stale.append((asset, "no local data"))
            continue
        latest = df["date"].max().date()
        gap = (today - latest).days
        if gap > _STALE_DAYS:
            stale.append((asset, f"latest={latest}, {gap} days behind"))

    if stale:
        msg = "Data freshness check FAILED:\n"
        for asset, reason in stale:
            msg += f"  {asset}: {reason}\n"
        raise RuntimeError(msg)

    print("Data freshness check passed.\n")


def _backfill_open_prices(
    state: PositionState,
    today: date,
) -> PositionState:
    """If entry_prices is null, read open prices for entry_date and write back.

    Also backfills exit_prices on the last ytd_history entry using the same
    open prices (same entry_date = previous position's exit date).
    Returns the updated state (may be unchanged if no backfill needed).
    """
    if state.entry_prices is not None or state.entry_date is None:
        return state

    entry_dt = date.fromisoformat(state.entry_date)

    # Read open prices for all assets currently held
    entry_prices: dict[str, float] = {}
    for asset in state.weights:
        df = query(asset, entry_dt, entry_dt)
        if len(df) > 0:
            entry_prices[asset] = float(df.iloc[0]["open"])

    if not entry_prices:
        return state  # data not yet available

    # Backfill exit_prices on last ytd_history entry (same date, their assets)
    new_history = list(state.ytd_history)
    if new_history and new_history[-1].exit_prices is None:
        last = new_history[-1]
        exit_prices: dict[str, float] = {}
        for asset in last.weights:
            df = query(asset, entry_dt, entry_dt)
            if len(df) > 0:
                exit_prices[asset] = float(df.iloc[0]["open"])
        new_history[-1] = PositionPeriod(
            weights=last.weights,
            entry_date=last.entry_date,
            exit_date=last.exit_date,
            entry_prices=last.entry_prices,
            exit_prices=exit_prices if exit_prices else None,
        )

    updated = PositionState(
        weights=state.weights,
        entry_date=state.entry_date,
        entry_prices=entry_prices,
        ytd_history=new_history,
    )
    write_position(updated)
    print(f"Backfilled entry prices for {state.entry_date}: {entry_prices}")
    return updated


def _count_holding_days(entry_date_str: str | None, today: date, asset_pool: list[str]) -> int | None:
    """Count trading days in [entry_date, today] using any pool asset."""
    if entry_date_str is None:
        return None
    entry_dt = date.fromisoformat(entry_date_str)
    for asset in asset_pool:
        df = query(asset, entry_dt, today)
        if len(df) > 0:
            return len(df)
    return None


def _compute_position_return(
    weights: dict[str, float],
    entry_prices: dict[str, float] | None,
    today: date,
) -> float | None:
    """Weighted sum of (close_latest / open_entry - 1) for each held asset.

    Uses the latest available close price up to *today* (handles the case
    where today's data is not yet available).
    """
    if entry_prices is None or not weights:
        return None
    total = 0.0
    for asset, weight in weights.items():
        open_entry = entry_prices.get(asset)
        if open_entry is None:
            return None
        # Try today first; fall back to latest available close
        df = query(asset, today, today)
        if len(df) == 0:
            df = read_local(asset)
            if df is None or len(df) == 0:
                return None
            df = df[df["date"] <= pd.Timestamp(today)]
            if len(df) == 0:
                return None
            df = df.tail(1)
        close_today = float(df.iloc[-1]["close"])
        total += weight * (close_today / open_entry - 1)
    return total


def _compute_benchmark_returns(
    asset_pool: list[str],
    entry_date_str: str | None,
    today: date,
) -> dict[str, float]:
    """Compute same-period return for every asset in the pool."""
    if entry_date_str is None:
        return {}
    entry_dt = date.fromisoformat(entry_date_str)
    returns: dict[str, float] = {}
    for asset in asset_pool:
        df_entry = query(asset, entry_dt, entry_dt)
        if len(df_entry) == 0:
            continue
        open_entry = float(df_entry.iloc[0]["open"])
        df_today = query(asset, today, today)
        if len(df_today) == 0:
            # Fall back to latest available close
            df_local = read_local(asset)
            if df_local is None or len(df_local) == 0:
                continue
            df_today = df_local[df_local["date"] <= pd.Timestamp(today)].tail(1)
            if len(df_today) == 0:
                continue
        close_today = float(df_today.iloc[-1]["close"])
        returns[asset] = close_today / open_entry - 1
    return returns


def _compute_ytd_return(
    ytd_history: list[PositionPeriod],
    current_return: float | None,
) -> float | None:
    """Chain-multiply closed periods plus current period return."""
    product = 1.0
    has_data = False

    for period in ytd_history:
        if period.entry_prices is None or period.exit_prices is None:
            continue
        period_ret = sum(
            w * (period.exit_prices[a] / period.entry_prices[a] - 1)
            for a, w in period.weights.items()
            if a in period.entry_prices and a in period.exit_prices
        )
        product *= 1 + period_ret
        has_data = True

    if current_return is not None:
        product *= 1 + current_return
        has_data = True

    return product - 1 if has_data else None


def _next_entry_date(today: date, asset_pool: list[str]) -> date:
    """Return the next trading date after today available in the data store."""
    for asset in asset_pool:
        df = read_local(asset)
        if df is None:
            continue
        future = df[df["date"] > pd.Timestamp(today)]
        if len(future) > 0:
            return future.iloc[0]["date"].date()
    # Fallback: no future data yet (happens in live run before next sync)
    from datetime import timedelta
    return today + timedelta(days=1)


def run(config: dict) -> None:
    today = date.today()
    asset_pool = config["asset_pool"]
    factor_configs = config["factors"]
    strategy_name = config.get("strategy_name", "Strategy")

    # 1. Sync data and verify freshness
    _sync_and_check(asset_pool, today)

    # 2. Read state (auto-migrates old format)
    current_state = read_position()

    # 3. Backfill open prices if entry_prices is null
    current_state = _backfill_open_prices(current_state, today)

    strategy = load_strategy(config)
    all_factors = load_registered_factors()

    # 4. Compute today's factor values for each asset
    asset_factor_values: dict[str, dict[str, float]] = {}

    for asset in asset_pool:
        df = query(asset, config.get("start", date(2016, 1, 1)), today)
        if len(df) == 0:
            continue

        factor_vals: dict[str, float] = {}
        for fc in factor_configs:
            fname = fc["name"]
            fmod = all_factors[fname]
            params = fc.get("params")
            try:
                series = fmod["compute"](df.copy(), params)
                validate(series, df, fmod["METADATA"])
                last_val = series.iloc[-1]
                if pd.notna(last_val):
                    factor_vals[fname] = float(last_val)
            except Exception as exc:
                warnings.warn(
                    f"factor '{fname}' failed for {asset}: {exc}",
                    stacklevel=2,
                )

        if len(factor_vals) == len(factor_configs):
            asset_factor_values[asset] = factor_vals

    # 5. Strategy → target weights
    target_weights = strategy.generate_weights(asset_factor_values)

    # 6. Execution → diff against current position
    current_weights = current_state.weights
    orders = diff(target_weights, current_weights)

    # 7. Compute metrics for notification
    holding_days = _count_holding_days(current_state.entry_date, today, asset_pool)
    position_return = _compute_position_return(
        current_weights, current_state.entry_prices, today
    )
    benchmark_returns = _compute_benchmark_returns(
        asset_pool, current_state.entry_date, today
    )
    ytd_return = _compute_ytd_return(current_state.ytd_history, position_return)

    entry_date_obj = (
        date.fromisoformat(current_state.entry_date)
        if current_state.entry_date
        else None
    )

    # 8. Assemble NotificationContext and send
    ctx = NotificationContext(
        strategy_name=strategy_name,
        signal_date=today,
        orders=orders,
        target_weights=target_weights,
        current_weights=current_weights,
        entry_date=entry_date_obj,
        holding_days=holding_days,
        position_return=position_return,
        benchmark_returns=benchmark_returns,
        ytd_return=ytd_return,
        asset_names=ASSET_NAMES,
        asset_factor_values=asset_factor_values,
    )

    message = format_notification(ctx)
    print(message)

    try:
        notifier = DingTalkNotifier()
        notifier.send(message)
        print("\nDingTalk notification sent.")
    except ValueError as exc:
        print(f"\nDingTalk skipped: {exc}")

    # 9. Persist new position (exit_prices backfilled on next run)
    next_entry_date = _next_entry_date(today, asset_pool)
    save_position(target_weights, next_entry_date, entry_prices=None)
    print(f"Position saved: {target_weights}, next_entry_date={next_entry_date}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Daily strategy run")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("strategy/configs/momentum_rotation.yaml"),
        help="Path to strategy config YAML",
    )
    args = parser.parse_args()

    config = _load_config(args.config)
    run(config)


if __name__ == "__main__":
    main()
