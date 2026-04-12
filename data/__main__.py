"""CLI entry point: python -m data.sync <asset_code>."""

import sys

from data.sync import sync


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python -m data.sync <asset_code>")
        sys.exit(1)

    asset_code = sys.argv[1]
    n = sync(asset_code)
    print(f"synced {n} rows for {asset_code}")


if __name__ == "__main__":
    main()
