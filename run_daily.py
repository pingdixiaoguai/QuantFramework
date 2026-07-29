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
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import tushare as ts
import yaml

from data.config import get_tushare_token
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
from strategy.rebalance import normalize_rebalance_mode, should_hold_position


def _load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    for key in ("start", "end"):
        if key in raw and isinstance(raw[key], str):
            if raw[key].lower() == "today":
                raw[key] = date.today()
            else:
                raw[key] = date.fromisoformat(raw[key])
    return raw


# Max trading days a local file may lag behind today before being considered
# stale. 1 tolerates "ran before today's bar was published" while still catching
# real pipeline breakage (>=2 missed trading days).
_STALE_TRADING_DAYS = 1


def _is_sse_trading_day(day: date) -> bool:
    """Return whether *day* is an open SSE trading day.

    A missing or malformed calendar response is treated as an operational
    failure rather than a market closure.  This prevents an upstream outage
    from being reported as a successful no-op by the unattended daily job.
    """
    day_str = day.strftime("%Y%m%d")
    pro = ts.pro_api(get_tushare_token())
    calendar = pro.trade_cal(
        exchange="SSE",
        start_date=day_str,
        end_date=day_str,
    )
    required = {"cal_date", "is_open"}
    if calendar is None or calendar.empty or not required.issubset(calendar.columns):
        raise RuntimeError(
            f"Cannot determine SSE trading status for {day}; "
            "trade_cal returned no usable row."
        )

    matching = calendar[calendar["cal_date"].astype(str) == day_str]
    if matching.empty:
        raise RuntimeError(
            f"Cannot determine SSE trading status for {day}; "
            "trade_cal did not return the requested date."
        )
    return str(matching.iloc[0]["is_open"]) == "1"


def _count_trading_days_behind(latest: date, today: date) -> int:
    """Number of SSE trading days strictly after `latest` and up to `today`."""
    if latest >= today:
        return 0
    pro = ts.pro_api(get_tushare_token())
    df = pro.trade_cal(
        exchange="SSE",
        start_date=(latest + timedelta(days=1)).strftime("%Y%m%d"),
        end_date=today.strftime("%Y%m%d"),
        is_open="1",
    )
    if df is None or df.empty:
        return 0
    return len(df)


def _sync_and_check(asset_pool: list[str], today: date) -> None:
    """Sync all assets and verify data freshness in trading-day terms."""
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
        gap = _count_trading_days_behind(latest, today)
        if gap > _STALE_TRADING_DAYS:
            stale.append((asset, f"latest={latest}, {gap} trading days behind"))

    if stale:
        msg = "Data freshness check FAILED:\n"
        for asset, reason in stale:
            msg += f"  {asset}: {reason}\n"
        raise RuntimeError(msg)

    print("Data freshness check passed.\n")


def _latest_common_data_date(asset_pool: list[str], today: date) -> date:
    """Return the latest date for which every pool asset has local data."""
    latest_dates: list[date] = []
    missing: list[str] = []

    for asset in asset_pool:
        df = read_local(asset)
        if df is None or len(df) == 0:
            missing.append(asset)
            continue
        priced = df[df["date"] <= pd.Timestamp(today)]
        if len(priced) == 0:
            missing.append(asset)
            continue
        latest_dates.append(priced["date"].max().date())

    if missing:
        raise RuntimeError(
            "No local data available through today for: " + ", ".join(missing)
        )

    return min(latest_dates)


def _priced_state_as_of(state: PositionState, priced_date: date) -> PositionState:
    """Return the position that is actually priced as of `priced_date`.

    `save_position()` records the next open's target immediately after a signal.
    If a morning run happens before that entry date is priced, the current state
    is still a pending target from the data's point of view. In that case, value
    the last archived holding through the latest priced close instead of
    dropping the open-to-pending segment from YTD.
    """
    if state.entry_date is None:
        return state

    entry_date = date.fromisoformat(state.entry_date)
    if entry_date <= priced_date:
        return state

    if not state.ytd_history:
        return PositionState()

    active = state.ytd_history[-1]
    return PositionState(
        weights=active.weights,
        entry_date=active.entry_date or None,
        entry_prices=active.entry_prices,
        ytd_history=list(state.ytd_history[:-1]),
    )


def _backfill_open_prices(
    state: PositionState,
    today: date,
    strategy_name: str,
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
        ytd_entry_date=state.ytd_entry_date,
        ytd_entry_prices=state.ytd_entry_prices,
        ytd_history=new_history,
    )
    write_position(updated, strategy_name)
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
    """Chain the live open-execution ledger for DingTalk YTD.

    Closed periods are entry-open -> exit-open returns. The current period is
    entry-open -> latest-close, so a rebalance day preserves the outgoing
    holding's overnight PnL and the incoming holding's intraday PnL.
    """
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


def _should_hold(
    current_weights: dict[str, float],
    holding_days: int | None,
    rebalance_days: int,
    rebalance_mode: str | None = "min_hold",
) -> bool:
    """Decide whether to override today's signal and keep the current position.

    Rules:
    - rebalance_days <= 1 → never hold (daily rebalancing).
    - No current position → never hold (must allow first entry).
    - holding_days is None with a position → just bought, today's bar not yet
      reflected; hold to avoid same-day churn.
    - holding_days < rebalance_days → inside the hold window; hold.
    - holding_days >= rebalance_days → window elapsed; allow rebalance.
    - rebalance_mode="fixed_cycle" → only allow rebalance on exact multiples
      of rebalance_days.
    """
    return should_hold_position(
        current_weights,
        holding_days,
        rebalance_days,
        rebalance_mode,
    )


def _next_entry_date(signal_date: date) -> date:
    """Return the next SSE trading day after a signal date.

    Local daily bars normally end at the signal date, so they cannot determine
    whether the following calendar day is a market holiday.  Persisting a
    target against a holiday makes it look executed on the next run and can
    suppress the outstanding rebalance instruction.
    """
    pro = ts.pro_api(get_tushare_token())
    calendar = pro.trade_cal(
        exchange="SSE",
        start_date=(signal_date + timedelta(days=1)).strftime("%Y%m%d"),
        # Covers the longest domestic market closure plus the following week.
        end_date=(signal_date + timedelta(days=14)).strftime("%Y%m%d"),
        is_open="1",
    )
    if calendar is None or calendar.empty or "cal_date" not in calendar:
        raise RuntimeError(
            "Cannot determine the next SSE trading day; refusing to save "
            "an unpriced rebalance target."
        )
    open_dates = pd.to_datetime(calendar["cal_date"], format="%Y%m%d")
    return open_dates.min().date()


def _is_unpriced_pending_target(state: PositionState, signal_date: date) -> bool:
    """Whether *state* records a target whose scheduled entry is still ahead.

    The position file intentionally stores tomorrow's target after a signal.
    While the corresponding open has not appeared in price data, that target
    is an instruction, not the portfolio's current holding.
    """
    if state.entry_date is None or state.entry_prices is not None:
        return False
    return date.fromisoformat(state.entry_date) > signal_date


def _save_or_update_rebalance_target(
    current_state: PositionState,
    target_weights: dict[str, float],
    entry_date: date,
    signal_date: date,
    strategy_name: str,
) -> bool:
    """Persist a rebalance target without duplicating an unfilled order.

    Returns ``True`` when an existing pending target was updated.  This keeps
    the archived outgoing holding intact when the same signal is sent again
    before the next trading open (for example across a market holiday).
    """
    if not _is_unpriced_pending_target(current_state, signal_date):
        save_position(target_weights, entry_date, strategy_name, entry_prices=None)
        return False

    updated_history = list(current_state.ytd_history)
    if updated_history and updated_history[-1].exit_prices is None:
        outgoing = updated_history[-1]
        updated_history[-1] = PositionPeriod(
            weights=outgoing.weights,
            entry_date=outgoing.entry_date,
            exit_date=entry_date.isoformat(),
            entry_prices=outgoing.entry_prices,
            exit_prices=None,
        )

    write_position(
        PositionState(
            weights=target_weights,
            entry_date=entry_date.isoformat(),
            entry_prices=None,
            ytd_entry_date=current_state.ytd_entry_date,
            ytd_entry_prices=current_state.ytd_entry_prices,
            ytd_history=updated_history,
        ),
        strategy_name,
    )
    return True


def run(config: dict) -> None:
    today = date.today()
    asset_pool = config["asset_pool"]
    factor_configs = config["factors"]
    strategy_name = config["strategy_name"]
    enable_dingtalk = config.get("enable_dingtalk", True)
    notifier = DingTalkNotifier() if enable_dingtalk else None
    rebalance_days = int(config.get("rebalance_days", 1))
    if rebalance_days < 1:
        raise ValueError(f"rebalance_days must be >= 1, got {rebalance_days}")
    rebalance_mode = normalize_rebalance_mode(config.get("rebalance_mode"))

    if not _is_sse_trading_day(today):
        print(f"{today} is not an SSE trading day; daily signal skipped.")
        return

    # 1. Sync data and verify freshness
    _sync_and_check(asset_pool, today)
    signal_date = _latest_common_data_date(asset_pool, today)
    if signal_date != today:
        print(f"Using latest common priced date as signal date: {signal_date}\n")

    # 2. Read state (auto-migrates old format)
    current_state = read_position(strategy_name)

    # 3. Backfill open prices if entry_prices is null
    current_state = _backfill_open_prices(current_state, today, strategy_name)

    strategy = load_strategy(config)
    all_factors = load_registered_factors()

    # 4. Compute today's factor values for each asset
    asset_factor_values: dict[str, dict[str, float]] = {}

    for asset in asset_pool:
        df = query(asset, config.get("start", date(2016, 1, 1)), signal_date)
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

    # 5. Strategy → today's signal weights (raw output before hold filter)
    signal_weights = strategy.generate_weights(asset_factor_values)
    # A target saved for a future market open is not a filled position yet.
    # Use the actually priced holding for the hold rule, execution diff and
    # notification so a holiday cannot turn an outstanding rebalance into a
    # misleading "当前持仓" message.
    priced_state = _priced_state_as_of(current_state, signal_date)
    current_weights = priced_state.weights

    # 5b. Hold-window filter: if we're inside a rebalance_days window,
    # override the signal with the current position so the diff produces
    # no orders. This is what reduces friction cost.
    holding_days = _count_holding_days(priced_state.entry_date, signal_date, asset_pool)
    if _should_hold(current_weights, holding_days, rebalance_days, rebalance_mode):
        target_weights = current_weights
        print(
            f"Hold window active: mode={rebalance_mode} "
            f"holding_days={holding_days}/{rebalance_days} "
            f"— signal {signal_weights} suppressed."
        )
    else:
        target_weights = signal_weights

    # 6. Execution → diff against current position
    orders = diff(target_weights, current_weights)

    # 7. Compute metrics for notification
    priced_holding_days = _count_holding_days(
        priced_state.entry_date,
        signal_date,
        asset_pool,
    )
    position_return = _compute_position_return(
        priced_state.weights,
        priced_state.entry_prices,
        signal_date,
    )
    current_ytd_entry_prices = (
        priced_state.ytd_entry_prices
        if priced_state.ytd_entry_date is not None
        else priced_state.entry_prices
    )
    current_ytd_return = _compute_position_return(
        priced_state.weights,
        current_ytd_entry_prices,
        signal_date,
    )
    benchmark_returns = _compute_benchmark_returns(
        asset_pool,
        priced_state.entry_date,
        signal_date,
    )
    ytd_return = _compute_ytd_return(priced_state.ytd_history, current_ytd_return)

    entry_date_obj = (
        date.fromisoformat(priced_state.entry_date)
        if priced_state.entry_date
        else None
    )

    # 8. Assemble NotificationContext and send
    ctx = NotificationContext(
        strategy_name=strategy_name,
        signal_date=signal_date,
        orders=orders,
        target_weights=target_weights,
        current_weights=priced_state.weights,
        entry_date=entry_date_obj,
        holding_days=priced_holding_days,
        position_return=position_return,
        benchmark_returns=benchmark_returns,
        ytd_return=ytd_return,
        asset_names=ASSET_NAMES,
        asset_factor_values=asset_factor_values,
    )

    message = format_notification(ctx)
    print(message)

    if notifier is not None:
        notifier.send(message)
        print("\nDingTalk notification sent.")
    else:
        print("\nDingTalk disabled by config (enable_dingtalk: false).")

    # 9. Persist new position only on rebalance (exit_prices backfilled on next run)
    is_rebalance = any(o.action in ("buy", "sell") for o in orders)
    if is_rebalance:
        next_entry_date = _next_entry_date(signal_date)
        updated_pending = _save_or_update_rebalance_target(
            current_state,
            target_weights,
            next_entry_date,
            signal_date,
            strategy_name,
        )
        action = "Pending position updated" if updated_pending else "Position saved"
        print(f"{action}: {target_weights}, next_entry_date={next_entry_date}")
    else:
        print(f"Hold signal — position unchanged: {current_weights}")


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
