"""Tushare incremental sync logic."""

import time

import pandas as pd
import tushare as ts

from data.config import get_tushare_token
from data.store import fill_adjustment_factors, merge_and_save, read_local

# Full history start date for first-time sync
_HISTORY_START = "20130101"


def _is_rate_limit_error(exc: Exception) -> bool:
    """Check if exception is a Tushare rate limit error."""
    msg = str(exc).lower()
    return "rate" in msg or "40203" in msg or "freq" in msg or "exceed" in msg


def _date_chunks(start_date: str, end_date: str) -> list[tuple[str, str]]:
    """Split API requests by calendar year to avoid endpoint row limits."""
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    chunks: list[tuple[str, str]] = []
    current = pd.Timestamp(year=start.year, month=1, day=1)

    while current <= end:
        chunk_start = max(start, current)
        chunk_end = min(end, pd.Timestamp(year=current.year, month=12, day=31))
        if chunk_start <= chunk_end:
            chunks.append(
                (
                    chunk_start.strftime("%Y%m%d"),
                    chunk_end.strftime("%Y%m%d"),
                )
            )
        current = pd.Timestamp(year=current.year + 1, month=1, day=1)

    return chunks


def _fetch_with_retry(fetcher):
    for attempt in range(3):
        try:
            return fetcher()
        except Exception as exc:
            if _is_rate_limit_error(exc) and attempt < 2:
                time.sleep(60)
                continue
            raise
    return None


def _fetch_raw_bars(asset_code: str, pro, start_date: str, end_date: str) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for chunk_start, chunk_end in _date_chunks(start_date, end_date):
        df = _fetch_with_retry(
            lambda: ts.pro_bar(
                ts_code=asset_code,
                api=pro,
                asset="FD",
                start_date=chunk_start,
                end_date=chunk_end,
                adj=None,
            )
        )
        if df is not None and not df.empty:
            parts.append(df)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _fetch_adj_factors(asset_code: str, pro, start_date: str, end_date: str) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for chunk_start, chunk_end in _date_chunks(start_date, end_date):
        df = _fetch_with_retry(
            lambda: pro.fund_adj(
                ts_code=asset_code,
                start_date=chunk_start,
                end_date=chunk_end,
            )
        )
        if df is not None and not df.empty:
            parts.append(df)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _merge_raw_with_adj(raw_df: pd.DataFrame, adj_df: pd.DataFrame) -> pd.DataFrame:
    if raw_df.empty:
        return raw_df
    if adj_df.empty:
        raise RuntimeError("fund_adj returned no adjustment factors")

    raw = raw_df.copy()
    adj = adj_df.copy()
    raw["trade_date"] = pd.to_datetime(raw["trade_date"], format="%Y%m%d")
    adj["trade_date"] = pd.to_datetime(adj["trade_date"], format="%Y%m%d")

    merged = raw.merge(
        adj[["trade_date", "adj_factor"]],
        on="trade_date",
        how="left",
    )
    merged = fill_adjustment_factors(merged.rename(columns={"trade_date": "date"}))
    merged["trade_date"] = merged["date"].dt.strftime("%Y%m%d")
    return merged.drop(columns=["date"])


def sync(asset_code: str) -> int:
    """Sync raw daily bars and fund adjustment factors for an ETF.

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

    raw_df = _fetch_raw_bars(asset_code, pro, start_date, today)
    if raw_df.empty:
        return 0

    adj_df = _fetch_adj_factors(asset_code, pro, start_date, today)
    df = _merge_raw_with_adj(raw_df, adj_df)

    row_count = len(df)
    merge_and_save(asset_code, df)
    return row_count


def sync_all(asset_codes: list[str]) -> dict[str, int]:
    """Sync multiple assets. Returns {asset_code: rows_added}."""
    results = {}
    for code in asset_codes:
        print(f"syncing {code} ... ", end="", flush=True)
        n = sync(code)
        print(f"{n} new rows")
        results[code] = n
    return results
