"""Backfill YTD position state by replaying strategy signals.

Usage:
    uv run python backfill_ytd.py --config strategy/configs/quality_momentum_top1.yaml
    uv run python backfill_ytd.py --config strategy/configs/quality_momentum_top1.yaml --from-date 2026-04-13
    uv run python backfill_ytd.py --config strategy/configs/quality_momentum_top1.yaml --from-date 2025-12-15

This is a maintenance tool for the case where run_daily.py was not executed
every trading day. It assumes the strategy/config was unchanged over the
replayed period, then reconstructs state/{strategy_name}_position.json using
the same live semantics as run_daily.py:

1. signal on day T uses data through T close;
2. the trade is booked at the next trading day's open;
3. the outgoing holding keeps close[T] -> open[T+1] overnight PnL;
4. the incoming holding earns open[T+1] -> close[T+1] intraday PnL;
5. config["rebalance_mode"] and config["rebalance_days"] are honored.

When --from-date is before Jan 1, the replay may use prior-year signals to
discover the position carried into the year. Periods that span the year
boundary are rebased to use the **close of the last prior-year trading day**
as their YTD basis (entry_prices). This way the live YTD chain includes the
close[prev]->open[ytd_start] overnight return and matches the backtest's
close-to-close daily chain across the year boundary.
"""

from __future__ import annotations

import argparse
import shutil
import warnings
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Callable

import pandas as pd
import yaml

from data.store import query
from execution import position as position_store
from execution.position import (
    PositionPeriod,
    PositionState,
    read_position,
    write_position,
)
from factors.registry import load_registered_factors
from factors.validator import validate
from notification.formatter import ASSET_NAMES
from run_daily import _should_hold
from strategy.loader import load_strategy
from strategy.rebalance import normalize_rebalance_mode


PriceLookup = Callable[[list[str], date], dict[str, float]]


@dataclass(frozen=True)
class ReplayResult:
    state: PositionState
    closed_returns: list[tuple[PositionPeriod, float | None]]
    latest_trading_day: date
    unpriced_target_weights: dict[str, float] | None = None


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


def _get_open_prices(assets: list[str], d: date) -> dict[str, float]:
    prices = {}
    for asset in assets:
        df = query(asset, d, d)
        if len(df) > 0:
            prices[asset] = float(df.iloc[0]["open"])
    return prices


def _get_close_prices(assets: list[str], d: date) -> dict[str, float]:
    prices = {}
    for asset in assets:
        df = query(asset, d, d)
        if len(df) > 0:
            prices[asset] = float(df.iloc[0]["close"])
    return prices


def _trading_days(asset_pool: list[str], start: date, end: date) -> list[date]:
    all_dates: set[pd.Timestamp] = set()
    for asset in asset_pool:
        df = query(asset, start, end)
        if len(df) > 0:
            all_dates.update(df["date"].tolist())
    return [ts.date() for ts in sorted(all_dates)]


def _next_trading_day(d: date, trading_days: list[date]) -> date | None:
    for td in trading_days:
        if td > d:
            return td
    return None


def _count_holding_days(
    entry_date: date | None,
    today: date,
    trading_days: list[date],
) -> int | None:
    if entry_date is None:
        return None
    return sum(1 for td in trading_days if entry_date <= td <= today)


def _weighted_return(
    weights: dict[str, float],
    entry_prices: dict[str, float] | None,
    exit_prices: dict[str, float] | None,
) -> float | None:
    if not weights or entry_prices is None or exit_prices is None:
        return None

    total = 0.0
    has_price = False
    for asset, weight in weights.items():
        entry = entry_prices.get(asset)
        exit_ = exit_prices.get(asset)
        if entry is None or exit_ is None:
            continue
        total += weight * (exit_ / entry - 1)
        has_price = True

    return total if has_price else None


def _period_return(period: PositionPeriod) -> float | None:
    return _weighted_return(period.weights, period.entry_prices, period.exit_prices)


def _clip_result_to_ytd(
    result: ReplayResult,
    ytd_start: date,
    price_lookup: PriceLookup,
    *,
    carry_in_basis_date: date | None = None,
    carry_in_basis_lookup: PriceLookup | None = None,
) -> ReplayResult:
    """Keep only the current year's return ledger after a wider replay.

    A replay may begin before Jan 1 to discover the position carried into the
    year. The persisted ledger still needs to represent YTD returns only, so
    periods that span the first trading day are rebased.

    When `carry_in_basis_date` and `carry_in_basis_lookup` are supplied (e.g.
    the previous trading day and its close-price lookup), the rebased period
    uses those prices as its `entry_prices`. This makes the live YTD chain
    include the close[last prior-year day] → open[first current-year day]
    overnight return, matching the backtest's close-to-close daily chain
    across the year boundary. Without them, the basis defaults to `ytd_start`
    open prices (the older, "first-trading-day open" convention).
    """
    basis_date = carry_in_basis_date if carry_in_basis_date is not None else ytd_start
    basis_lookup = carry_in_basis_lookup or price_lookup

    clipped_history: list[PositionPeriod] = []

    for period in result.state.ytd_history:
        entry_date = (
            date.fromisoformat(period.entry_date)
            if period.entry_date
            else None
        )
        exit_date = date.fromisoformat(period.exit_date)

        if exit_date <= ytd_start:
            continue

        if entry_date is not None and entry_date >= ytd_start:
            clipped = period
        else:
            ytd_entry_prices = basis_lookup(list(period.weights), basis_date)
            clipped = PositionPeriod(
                weights=period.weights,
                entry_date=ytd_start.isoformat(),
                exit_date=period.exit_date,
                entry_prices=ytd_entry_prices if ytd_entry_prices else None,
                exit_prices=period.exit_prices,
            )
        clipped_history.append(clipped)

    state = result.state
    ytd_entry_date = state.ytd_entry_date
    ytd_entry_prices = state.ytd_entry_prices
    if state.weights and state.entry_date is not None:
        current_entry_date = date.fromisoformat(state.entry_date)
        if current_entry_date < ytd_start:
            prices = basis_lookup(list(state.weights), basis_date)
            ytd_entry_date = ytd_start.isoformat()
            ytd_entry_prices = prices if prices else None

    clipped_state = PositionState(
        weights=state.weights,
        entry_date=state.entry_date,
        entry_prices=state.entry_prices,
        ytd_entry_date=ytd_entry_date,
        ytd_entry_prices=ytd_entry_prices,
        ytd_history=clipped_history,
    )

    return ReplayResult(
        state=clipped_state,
        closed_returns=[(period, _period_return(period)) for period in clipped_history],
        latest_trading_day=result.latest_trading_day,
        unpriced_target_weights=result.unpriced_target_weights,
    )


def _compute_ytd_return(result: ReplayResult) -> float | None:
    product = 1.0
    has_data = False

    for _, period_return in result.closed_returns:
        if period_return is None:
            continue
        product *= 1 + period_return
        has_data = True

    current = result.state
    if current.weights:
        entry_prices = (
            current.ytd_entry_prices
            if current.ytd_entry_date is not None
            else current.entry_prices
        )
        close_prices = _get_close_prices(
            list(current.weights),
            result.latest_trading_day,
        )
        current_return = _weighted_return(
            current.weights,
            entry_prices,
            close_prices if close_prices else None,
        )
        if current_return is not None:
            product *= 1 + current_return
            has_data = True

    return product - 1 if has_data else None


def _asset_label(weights: dict[str, float]) -> str:
    if not weights:
        return "-"
    asset = max(weights, key=weights.get)
    return ASSET_NAMES.get(asset, asset)


def _state_as_of(state: PositionState, as_of: date) -> PositionState:
    """Slice a state file down to the account state after `as_of` open.

    Closed periods whose exit open is on/before `as_of` are preserved. If a
    later closed period spans `as_of`, it becomes the current open position so
    replay can replace everything after that date.
    """
    preserved: list[PositionPeriod] = []
    active: PositionPeriod | None = None

    for period in state.ytd_history:
        entry_date = date.fromisoformat(period.entry_date)
        exit_date = date.fromisoformat(period.exit_date)

        if exit_date <= as_of:
            preserved.append(period)
        elif entry_date <= as_of < exit_date:
            if active is None or entry_date >= date.fromisoformat(active.entry_date):
                active = period

    if active is not None:
        return PositionState(
            weights=active.weights,
            entry_date=active.entry_date,
            entry_prices=active.entry_prices,
            ytd_history=preserved,
        )

    if state.entry_date is not None:
        entry_date = date.fromisoformat(state.entry_date)
        if entry_date <= as_of:
            return PositionState(
                weights=state.weights,
                entry_date=state.entry_date,
                entry_prices=state.entry_prices,
                ytd_entry_date=state.ytd_entry_date,
                ytd_entry_prices=state.ytd_entry_prices,
                ytd_history=preserved,
            )

    return PositionState(ytd_history=preserved)


def _compute_signal_weights(
    config: dict,
    signal_date: date,
    strategy,
    all_factors: dict,
) -> dict[str, float]:
    asset_factor_values: dict[str, dict[str, float]] = {}
    asset_pool = config["asset_pool"]
    factor_configs = config["factors"]

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
                    f"factor '{fname}' failed for {asset} on {signal_date}: {exc}",
                    stacklevel=2,
                )

        if len(factor_vals) == len(factor_configs):
            asset_factor_values[asset] = factor_vals

    return strategy.generate_weights(asset_factor_values)


def _replay_signals_to_state(
    signal_weights_by_date: list[tuple[date, dict[str, float]]],
    trading_days: list[date],
    price_lookup: PriceLookup,
    rebalance_days: int,
    initial_state: PositionState | None = None,
    holding_calendar: list[date] | None = None,
    rebalance_mode: str | None = "min_hold",
) -> ReplayResult:
    """Replay precomputed daily signals into a PositionState ledger."""
    if not trading_days:
        raise RuntimeError("no trading days found for this year")
    rebalance_mode = normalize_rebalance_mode(rebalance_mode)

    initial_state = initial_state or PositionState()
    current_weights: dict[str, float] = dict(initial_state.weights)
    current_entry_date = (
        date.fromisoformat(initial_state.entry_date)
        if initial_state.entry_date
        else None
    )
    current_entry_prices = initial_state.entry_prices
    current_ytd_entry_date = (
        date.fromisoformat(initial_state.ytd_entry_date)
        if initial_state.ytd_entry_date
        else None
    )
    current_ytd_entry_prices = initial_state.ytd_entry_prices
    ytd_history = list(initial_state.ytd_history)
    closed_returns = [(period, _period_return(period)) for period in ytd_history]
    pending_weights: dict[str, float] | None = None
    pending_entry_date: date | None = None
    unpriced_target_weights: dict[str, float] | None = None
    holding_calendar = holding_calendar or trading_days

    for signal_date, signal_weights in signal_weights_by_date:
        if pending_entry_date == signal_date and pending_weights is not None:
            if current_weights:
                exit_prices = price_lookup(list(current_weights), signal_date)
                period_entry_date = current_ytd_entry_date or current_entry_date
                period_entry_prices = (
                    current_ytd_entry_prices
                    if current_ytd_entry_date is not None
                    else current_entry_prices
                )
                period = PositionPeriod(
                    weights=current_weights,
                    entry_date=period_entry_date.isoformat()
                    if period_entry_date
                    else "",
                    exit_date=signal_date.isoformat(),
                    entry_prices=period_entry_prices,
                    exit_prices=exit_prices if exit_prices else None,
                )
                ytd_history.append(period)
                closed_returns.append(
                    (
                        period,
                        _weighted_return(
                            current_weights,
                            period_entry_prices,
                            exit_prices if exit_prices else None,
                        ),
                    )
                )

            current_weights = pending_weights
            current_entry_date = signal_date
            entry_prices = price_lookup(list(current_weights), signal_date)
            current_entry_prices = entry_prices if entry_prices else None
            current_ytd_entry_date = None
            current_ytd_entry_prices = None
            pending_weights = None
            pending_entry_date = None

        holding_days = _count_holding_days(
            current_entry_date,
            signal_date,
            holding_calendar,
        )
        target_weights = (
            current_weights
            if _should_hold(
                current_weights,
                holding_days,
                rebalance_days,
                rebalance_mode,
            )
            else signal_weights
        )

        if target_weights and target_weights != current_weights:
            next_entry_date = _next_trading_day(signal_date, trading_days)
            if next_entry_date is None:
                unpriced_target_weights = target_weights
            else:
                pending_weights = target_weights
                pending_entry_date = next_entry_date

    state = PositionState(
        weights=current_weights,
        entry_date=current_entry_date.isoformat() if current_entry_date else None,
        entry_prices=current_entry_prices,
        ytd_entry_date=current_ytd_entry_date.isoformat()
        if current_ytd_entry_date
        else None,
        ytd_entry_prices=current_ytd_entry_prices,
        ytd_history=ytd_history,
    )

    return ReplayResult(
        state=state,
        closed_returns=closed_returns,
        latest_trading_day=trading_days[-1],
        unpriced_target_weights=unpriced_target_weights,
    )


def _replay_strategy(
    config: dict,
    as_of: date,
    from_date: date | None = None,
    initial_state: PositionState | None = None,
) -> ReplayResult:
    asset_pool = config["asset_pool"]
    year_start = date(as_of.year, 1, 1)
    replay_start = from_date or year_start
    if replay_start > as_of:
        raise ValueError(f"from_date must be <= as_of, got {replay_start} > {as_of}")

    calendar_start = min(year_start, replay_start)
    if initial_state and initial_state.entry_date:
        calendar_start = min(
            calendar_start,
            date.fromisoformat(initial_state.entry_date),
        )

    holding_calendar = _trading_days(asset_pool, calendar_start, as_of)
    trading_days = [td for td in holding_calendar if replay_start <= td <= as_of]
    if not trading_days:
        raise RuntimeError("no trading days found for the requested backfill range")
    ytd_days = [td for td in holding_calendar if year_start <= td <= as_of]
    if not ytd_days:
        raise RuntimeError("no trading days found for this year")

    strategy = load_strategy(config)
    all_factors = load_registered_factors()
    rebalance_days = int(config.get("rebalance_days", 1))
    if rebalance_days < 1:
        raise ValueError(f"rebalance_days must be >= 1, got {rebalance_days}")
    rebalance_mode = normalize_rebalance_mode(config.get("rebalance_mode"))

    signal_weights_by_date = [
        (
            signal_date,
            _compute_signal_weights(config, signal_date, strategy, all_factors),
        )
        for signal_date in trading_days
    ]
    result = _replay_signals_to_state(
        signal_weights_by_date,
        trading_days,
        _get_open_prices,
        rebalance_days,
        initial_state=initial_state,
        holding_calendar=holding_calendar,
        rebalance_mode=rebalance_mode,
    )
    # If the wider calendar reaches before YTD, rebase any carried period to
    # close[last prior-year trading day] so the chained YTD return matches the
    # backtest's close-to-close traversal across the year boundary.
    pre_ytd_days = [td for td in holding_calendar if td < ytd_days[0]]
    carry_in_basis_date = pre_ytd_days[-1] if pre_ytd_days else None
    return _clip_result_to_ytd(
        result,
        ytd_days[0],
        _get_open_prices,
        carry_in_basis_date=carry_in_basis_date,
        carry_in_basis_lookup=_get_close_prices if carry_in_basis_date else None,
    )


def _backup_state_file(strategy_name: str) -> Path | None:
    path = position_store._state_file(strategy_name)
    if not path.exists():
        return None

    suffix = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup_path = path.with_name(f"{path.name}.bak-{suffix}")
    shutil.copy2(path, backup_path)
    return backup_path


def _print_result(result: ReplayResult) -> None:
    product = 1.0
    print("Reconstructed periods:")
    for period, period_return in result.closed_returns:
        if period_return is not None:
            product *= 1 + period_return
            ret_str = f"{period_return:+.2%}"
            cum_str = f"cum={product - 1:+.2%}"
        else:
            ret_str = "price missing"
            cum_str = ""
        print(
            f"  {period.entry_date} -> {period.exit_date}  "
            f"{_asset_label(period.weights)}  {ret_str} {cum_str}".rstrip()
        )

    state = result.state
    print("\nBackfill preview:")
    print(f"  Closed periods: {len(result.closed_returns)}")
    print(f"  Current position: {state.weights}")
    print(f"  Entry date: {state.entry_date}")
    print(f"  Entry prices: {state.entry_prices or 'pending'}")
    if state.ytd_entry_date or state.ytd_entry_prices:
        print(f"  YTD basis date: {state.ytd_entry_date}")
        print(f"  YTD basis prices: {state.ytd_entry_prices or 'pending'}")
    print(f"  Latest priced trading day: {result.latest_trading_day}")
    ytd_return = _compute_ytd_return(result)
    if ytd_return is None:
        print("  YTD return: data insufficient")
    else:
        print(f"  YTD return: {ytd_return:+.2%}")

    if result.unpriced_target_weights:
        print(
            "  Pending unpriced target after latest trading day: "
            f"{result.unpriced_target_weights}"
        )


def backfill(
    config: dict,
    as_of: date | None = None,
    from_date: date | None = None,
    dry_run: bool = False,
    backup: bool = True,
) -> ReplayResult:
    as_of = as_of or date.today()
    strategy_name = config["strategy_name"]
    initial_state = None
    if from_date is not None:
        initial_state = _state_as_of(read_position(strategy_name), from_date)

    result = _replay_strategy(
        config,
        as_of,
        from_date=from_date,
        initial_state=initial_state,
    )
    _print_result(result)

    if dry_run:
        print("\nDry run only; state file not written.")
        return result

    backup_path = _backup_state_file(strategy_name) if backup else None
    if backup_path is not None:
        print(f"\nBacked up existing state to: {backup_path}")

    write_position(result.state, strategy_name)
    print(f"State written for strategy: {strategy_name}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill YTD position history")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("strategy/configs/quality_momentum_top1.yaml"),
    )
    parser.add_argument(
        "--from-date",
        type=date.fromisoformat,
        default=None,
        help=(
            "Replay from this date (YYYY-MM-DD), preserving existing state "
            "before that date. Dates before Jan 1 are allowed to recover "
            "a carried-in position; only current-year YTD history is written."
        ),
    )
    parser.add_argument(
        "--as-of",
        type=date.fromisoformat,
        default=None,
        help="Backfill through this date (YYYY-MM-DD); defaults to today.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print reconstructed state without writing state/*.json.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not create a timestamped backup before overwriting state.",
    )
    args = parser.parse_args()

    config = _load_config(args.config)
    backfill(
        config,
        as_of=args.as_of,
        from_date=args.from_date,
        dry_run=args.dry_run,
        backup=not args.no_backup,
    )


if __name__ == "__main__":
    main()
