"""Backfill YTD position history by replaying the strategy from year start.

Usage:
    uv run python backfill_ytd.py --config strategy/configs/quality_momentum_top1.yaml

Replays the strategy day-by-day from Jan 1 of the current year to today,
detects position transitions, and writes a correct current_position.json
with proper entry/exit prices (using open prices on transition days).
"""

from __future__ import annotations

import argparse
import warnings
from datetime import date
from pathlib import Path

import pandas as pd
import yaml

from data.store import query
from execution.position import PositionPeriod, PositionState, write_position
from factors.registry import load_registered_factors
from factors.validator import validate
from strategy.loader import load_strategy


def _load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    for key in ("start", "end"):
        if key in raw and isinstance(raw[key], str):
            raw[key] = date.fromisoformat(raw[key])
    return raw


def _get_open_prices(assets: list[str], d: date) -> dict[str, float]:
    """Get open prices for assets on a given date."""
    prices = {}
    for asset in assets:
        df = query(asset, d, d)
        if len(df) > 0:
            prices[asset] = float(df.iloc[0]["open"])
    return prices


def backfill(config: dict) -> None:
    today = date.today()
    year_start = date(today.year, 1, 1)
    asset_pool = config["asset_pool"]
    factor_configs = config["factors"]

    strategy = load_strategy(config)
    all_factors = load_registered_factors()

    # Build trading day calendar from any pool asset
    all_dates: set[pd.Timestamp] = set()
    for asset in asset_pool:
        df = query(asset, year_start, today)
        if len(df) > 0:
            all_dates.update(df["date"].tolist())
    trading_days = sorted(all_dates)

    if not trading_days:
        print("No trading days found for this year.")
        return

    print(f"Replaying strategy from {year_start} to {today}")
    print(f"Trading days: {len(trading_days)}")

    # Replay strategy day by day, collecting (signal_date, weights)
    daily_weights: list[tuple[date, dict[str, float]]] = []

    for t in trading_days:
        t_date = t.date()
        asset_factor_values: dict[str, dict[str, float]] = {}

        for asset in asset_pool:
            df = query(asset, config.get("start", date(2016, 1, 1)), t_date)
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
                    warnings.warn(f"factor '{fname}' failed for {asset} on {t_date}: {exc}")

            if len(factor_vals) == len(factor_configs):
                asset_factor_values[asset] = factor_vals

        weights = strategy.generate_weights(asset_factor_values)
        if weights:
            daily_weights.append((t_date, weights))

    if not daily_weights:
        print("No signals generated.")
        return

    # Detect position transitions:
    # Signal on day T → entry on day T+1 at open
    # We group consecutive days with the same weights into periods.
    periods: list[tuple[dict[str, float], date, date]] = []  # (weights, signal_start, signal_end)
    current_w = daily_weights[0][1]
    period_start = daily_weights[0][0]
    prev_date = period_start

    for signal_date, weights in daily_weights[1:]:
        if weights != current_w:
            periods.append((current_w, period_start, prev_date))
            current_w = weights
            period_start = signal_date
        prev_date = signal_date

    # Last period is still open (current position)
    last_signal_date = daily_weights[-1][0]

    # Build trading days list as dates for lookup
    td_dates = [t.date() for t in trading_days]

    def next_trading_day(d: date) -> date | None:
        """Find the next trading day after d."""
        for td in td_dates:
            if td > d:
                return td
        return None

    # Build ytd_history from closed periods
    ytd_history: list[PositionPeriod] = []
    for weights, sig_start, sig_end in periods:
        entry_d = next_trading_day(sig_start)
        if entry_d is None:
            # First period might start on first trading day
            entry_d = sig_start
        # Exit date = next trading day after the last signal day of this period
        exit_d = next_trading_day(sig_end)
        if exit_d is None:
            exit_d = sig_end

        entry_prices = _get_open_prices(list(weights.keys()), entry_d)
        exit_prices = _get_open_prices(list(weights.keys()), exit_d)

        ytd_history.append(PositionPeriod(
            weights=weights,
            entry_date=entry_d.isoformat(),
            exit_date=exit_d.isoformat(),
            entry_prices=entry_prices if entry_prices else None,
            exit_prices=exit_prices if exit_prices else None,
        ))

        held_asset = max(weights, key=weights.get)
        entry_p = entry_prices.get(held_asset, 0)
        exit_p = exit_prices.get(held_asset, 0)
        ret = (exit_p / entry_p - 1) if entry_p else 0
        from notification.formatter import ASSET_NAMES
        name = ASSET_NAMES.get(held_asset, held_asset)
        print(f"  {entry_d} → {exit_d}  {name}  {ret:+.2%}")

    # Current position: the last open period
    # Entry is the trading day after the signal that started this position
    current_entry_d = next_trading_day(period_start)
    if current_entry_d is None or current_entry_d > today:
        # If no future trading day yet, entry is tomorrow
        from datetime import timedelta
        current_entry_d = today + timedelta(days=1)

    # Check if entry_date has actual data (for backfill)
    current_entry_prices = _get_open_prices(list(current_w.keys()), current_entry_d)

    state = PositionState(
        weights=current_w,
        entry_date=current_entry_d.isoformat(),
        entry_prices=current_entry_prices if current_entry_prices else None,
        ytd_history=ytd_history,
    )

    write_position(state)

    print(f"\nBackfill complete!")
    print(f"  Closed periods: {len(ytd_history)}")
    print(f"  Current position: {current_w}")
    print(f"  Entry date: {current_entry_d}")
    print(f"  Entry prices: {current_entry_prices or 'pending'}")

    # Compute and show YTD return
    product = 1.0
    for p in ytd_history:
        if p.entry_prices and p.exit_prices:
            period_ret = sum(
                w * (p.exit_prices[a] / p.entry_prices[a] - 1)
                for a, w in p.weights.items()
                if a in p.entry_prices and a in p.exit_prices
            )
            product *= (1 + period_ret)

    # Add current period if we have prices
    if current_entry_prices and current_w:
        for asset, w in current_w.items():
            open_p = current_entry_prices.get(asset)
            if open_p:
                df_today = query(asset, today, today)
                if len(df_today) > 0:
                    close_p = float(df_today.iloc[0]["close"])
                    product *= (1 + w * (close_p / open_p - 1))

    print(f"  YTD return: {product - 1:+.2%}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill YTD position history")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("strategy/configs/quality_momentum_top1.yaml"),
    )
    args = parser.parse_args()
    config = _load_config(args.config)
    backfill(config)


if __name__ == "__main__":
    main()
