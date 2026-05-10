"""Backfill YTD position state by replaying the strategy from year start.

Usage:
    uv run python backfill_ytd.py --config strategy/configs/quality_momentum_top1.yaml

This is a maintenance tool for the case where run_daily.py was not executed
every trading day. It assumes the strategy/config was unchanged over the
backfilled period, then reconstructs state/{strategy_name}_position.json using
the same live semantics as run_daily.py:

1. signal on day T uses data through T close;
2. the trade is booked at the next trading day's open;
3. config["rebalance_days"] is honored.
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
from execution.position import PositionPeriod, PositionState, write_position
from factors.registry import load_registered_factors
from factors.validator import validate
from notification.formatter import ASSET_NAMES
from run_daily import _should_hold
from strategy.loader import load_strategy


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
        close_prices = _get_close_prices(list(current.weights), result.latest_trading_day)
        current_return = _weighted_return(
            current.weights,
            current.entry_prices,
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
) -> ReplayResult:
    """Replay precomputed daily signals into a PositionState ledger."""
    if not trading_days:
        raise RuntimeError("no trading days found for this year")

    current_weights: dict[str, float] = {}
    current_entry_date: date | None = None
    current_entry_prices: dict[str, float] | None = None
    ytd_history: list[PositionPeriod] = []
    closed_returns: list[tuple[PositionPeriod, float | None]] = []
    pending_weights: dict[str, float] | None = None
    pending_entry_date: date | None = None
    unpriced_target_weights: dict[str, float] | None = None

    for signal_date, signal_weights in signal_weights_by_date:
        if pending_entry_date == signal_date and pending_weights is not None:
            if current_weights:
                exit_prices = price_lookup(list(current_weights), signal_date)
                period = PositionPeriod(
                    weights=current_weights,
                    entry_date=current_entry_date.isoformat()
                    if current_entry_date
                    else "",
                    exit_date=signal_date.isoformat(),
                    entry_prices=current_entry_prices,
                    exit_prices=exit_prices if exit_prices else None,
                )
                ytd_history.append(period)
                closed_returns.append(
                    (
                        period,
                        _weighted_return(
                            current_weights,
                            current_entry_prices,
                            exit_prices if exit_prices else None,
                        ),
                    )
                )

            current_weights = pending_weights
            current_entry_date = signal_date
            entry_prices = price_lookup(list(current_weights), signal_date)
            current_entry_prices = entry_prices if entry_prices else None
            pending_weights = None
            pending_entry_date = None

        holding_days = _count_holding_days(
            current_entry_date,
            signal_date,
            trading_days,
        )
        target_weights = (
            current_weights
            if _should_hold(current_weights, holding_days, rebalance_days)
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
        ytd_history=ytd_history,
    )

    return ReplayResult(
        state=state,
        closed_returns=closed_returns,
        latest_trading_day=trading_days[-1],
        unpriced_target_weights=unpriced_target_weights,
    )


def _replay_strategy(config: dict, as_of: date) -> ReplayResult:
    asset_pool = config["asset_pool"]
    year_start = date(as_of.year, 1, 1)
    trading_days = _trading_days(asset_pool, year_start, as_of)
    if not trading_days:
        raise RuntimeError("no trading days found for this year")

    strategy = load_strategy(config)
    all_factors = load_registered_factors()
    rebalance_days = int(config.get("rebalance_days", 1))
    if rebalance_days < 1:
        raise ValueError(f"rebalance_days must be >= 1, got {rebalance_days}")

    signal_weights_by_date = [
        (
            signal_date,
            _compute_signal_weights(config, signal_date, strategy, all_factors),
        )
        for signal_date in trading_days
    ]
    return _replay_signals_to_state(
        signal_weights_by_date,
        trading_days,
        _get_open_prices,
        rebalance_days,
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
    dry_run: bool = False,
    backup: bool = True,
) -> ReplayResult:
    as_of = as_of or date.today()
    strategy_name = config["strategy_name"]

    result = _replay_strategy(config, as_of)
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
        dry_run=args.dry_run,
        backup=not args.no_backup,
    )


if __name__ == "__main__":
    main()
