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

from data.store import query
from execution.interfaces import diff
from execution.position import read_position, save_position
from factors.registry import load_registered_factors
from factors.validator import validate
from notification.dingtalk import DingTalkNotifier
from notification.formatter import format_orders
from strategy.loader import load_strategy


def _load_config(path: Path) -> dict:
    with open(path) as f:
        raw = yaml.safe_load(f)
    for key in ("start", "end"):
        if key in raw and isinstance(raw[key], str):
            raw[key] = date.fromisoformat(raw[key])
    return raw


def run(config: dict) -> None:
    today = date.today()
    asset_pool = config["asset_pool"]
    factor_configs = config["factors"]

    strategy = load_strategy(config)
    all_factors = load_registered_factors()

    # Compute today's factor values for each asset
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

    # Strategy → target weights
    target_weights = strategy.generate_weights(asset_factor_values)

    # Execution → diff against current position
    current_weights = read_position()
    orders = diff(target_weights, current_weights)

    # Format and send notification
    message = format_orders(orders, target_weights, run_date=today)
    print(message)

    try:
        notifier = DingTalkNotifier()
        notifier.send(message)
        print("\nDingTalk notification sent.")
    except ValueError as exc:
        print(f"\nDingTalk skipped: {exc}")

    # Persist new position
    save_position(target_weights)
    print(f"Position saved: {target_weights}")


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
