"""Tushare incremental sync logic."""

import time

import pandas as pd
import tushare as ts

from data.config import get_tushare_token
from data.store import merge_and_save, read_local

# Full history start date for first-time sync
_HISTORY_START = "20160101"


def _is_rate_limit_error(exc: Exception) -> bool:
    """Check if exception is a Tushare rate limit error."""
    msg = str(exc).lower()
    return "rate" in msg or "40203" in msg or "freq" in msg or "exceed" in msg


def sync(asset_code: str) -> int:
    """Sync daily bars for an ETF from Tushare.

    Returns the number of new rows added.
    """
    token = get_tushare_token()
    pro = ts.pro_api(token)

    # Determine start date
    existing = read_local(asset_code)
    if existing is not None and len(existing) > 0:
        max_local = existing["date"].max()
        start_date = (max_local + pd.Timedelta(days=1)).strftime("%Y%m%d")
    else:
        start_date = _HISTORY_START

    today = pd.Timestamp.now().strftime("%Y%m%d")

    if start_date > today:
        return 0

    # Fetch with retry on rate limit
    df = None
    for attempt in range(3):
        try:
            df = ts.pro_bar(
                ts_code=asset_code,
                api=pro,
                asset="FD",
                start_date=start_date,
                end_date=today,
                adj="qfq",
            )
            break
        except Exception as exc:
            if _is_rate_limit_error(exc) and attempt < 2:
                time.sleep(60)
                continue
            raise

    if df is None or df.empty:
        return 0

    row_count = len(df)
    merge_and_save(asset_code, df)
    return row_count


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python -m data.sync <asset_code>")
        sys.exit(1)

    asset_code = sys.argv[1]
    n = sync(asset_code)
    print(f"synced {n} rows for {asset_code}")
