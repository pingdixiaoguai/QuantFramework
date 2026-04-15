"""CLI entry point: python -m data.sync <asset_code> | --config <yaml>."""

import argparse
from pathlib import Path

import yaml

from data.sync import sync, sync_all


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync ETF daily bars from Tushare")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("asset_code", nargs="?", help="Single asset code, e.g. 510300.SH")
    group.add_argument(
        "--config",
        type=Path,
        help="Strategy config YAML (reads asset_pool for batch sync)",
    )
    args = parser.parse_args()

    if args.config:
        with open(args.config, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        asset_pool = cfg.get("asset_pool")
        if not asset_pool:
            parser.error(f"no asset_pool found in {args.config}")
        sync_all(asset_pool)
    else:
        n = sync(args.asset_code)
        print(f"synced {n} rows for {args.asset_code}")


if __name__ == "__main__":
    main()
