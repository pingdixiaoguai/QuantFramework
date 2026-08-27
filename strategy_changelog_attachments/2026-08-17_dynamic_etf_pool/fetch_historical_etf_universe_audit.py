"""Fetch month-end point-in-time ETF size snapshots for survivorship audit.

Generated caches are written under data/db (gitignored). Re-running resumes
from the existing cache and only requests missing month ends.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
import tushare as ts

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from data import store  # noqa: E402
from data.config import get_tushare_token  # noqa: E402

SNAPSHOT_CACHE = store.DB_DIR / "historical_etf_month_end_size.parquet"
BASIC_CACHE = store.DB_DIR / "historical_etf_basic.parquet"


def request_with_retry(fn):
    for attempt in range(5):
        try:
            return fn()
        except Exception as exc:
            text = str(exc).lower()
            rate_limited = any(token in text for token in ("rate", "freq", "exceed", "40203", "每分钟"))
            if attempt < 4:
                time.sleep(60 if rate_limited else min(5 * (attempt + 1), 20))
                continue
            raise
    raise RuntimeError("unreachable")


def month_ends() -> list[pd.Timestamp]:
    df = store.read_local("510300.SH")
    if df is None or df.empty:
        raise RuntimeError("510300.SH local calendar is required")
    dates = pd.DatetimeIndex(df.loc[(df.date >= "2014-01-01") & (df.date <= "2026-08-14"), "date"])
    return list(pd.Series(dates, index=dates).groupby(dates.to_period("M")).max())


def fetch_basic(pro) -> pd.DataFrame:
    fields = "ts_code,csname,extname,cname,index_code,index_name,setup_date,list_date,list_status,exchange,mgr_name,etf_type"
    parts = []
    for status in ("L", "D"):
        df = request_with_retry(lambda status=status: pro.etf_basic(list_status=status, fields=fields))
        if df is not None and not df.empty:
            parts.append(df)
    basic = pd.concat(parts, ignore_index=True).drop_duplicates("ts_code", keep="first")
    BASIC_CACHE.parent.mkdir(parents=True, exist_ok=True)
    basic.to_parquet(BASIC_CACHE, index=False)
    return basic


def fetch_one(pro, date: pd.Timestamp) -> pd.DataFrame:
    stamp = date.strftime("%Y%m%d")
    shares = []
    for market in ("SH", "SZ"):
        df = request_with_retry(
            lambda market=market: pro.fund_share(trade_date=stamp, market=market)
        )
        if df is not None and not df.empty:
            shares.append(df)
    if not shares:
        return pd.DataFrame()
    share = pd.concat(shares, ignore_index=True)
    daily = request_with_retry(lambda: pro.fund_daily(trade_date=stamp))
    if daily is None or daily.empty:
        return pd.DataFrame()
    out = share.merge(daily[["ts_code", "close", "amount"]], on="ts_code", how="inner")
    out = out[out["fund_type"].eq("ETF")].copy()
    out["month_end"] = date
    out["estimated_size_yi"] = pd.to_numeric(out["fd_share"], errors="coerce") * pd.to_numeric(out["close"], errors="coerce") / 10000.0
    out["amount_yi"] = pd.to_numeric(out["amount"], errors="coerce") / 10000.0
    # Keep a buffer below the eventual 50yi screen for boundary auditing.
    return out.loc[out["estimated_size_yi"] >= 40.0, [
        "month_end", "ts_code", "estimated_size_yi", "amount_yi", "market"
    ]]


def main() -> None:
    pro = ts.pro_api(get_tushare_token())
    basic = fetch_basic(pro)
    existing = pd.read_parquet(SNAPSHOT_CACHE) if SNAPSHOT_CACHE.exists() else pd.DataFrame()
    completed = set(pd.to_datetime(existing.get("month_end", pd.Series(dtype="datetime64[ns]"))))
    pending = [date for date in month_ends() if date not in completed]
    print(f"ETF basic: {len(basic)} rows; month ends pending: {len(pending)}", flush=True)
    parts = [existing] if not existing.empty else []
    for number, date in enumerate(pending, 1):
        frame = fetch_one(pro, date)
        if frame.empty:
            raise RuntimeError(f"no ETF snapshot for {date.date()}")
        parts.append(frame)
        if number % 12 == 0 or number == len(pending):
            combined = pd.concat(parts, ignore_index=True)
            combined["month_end"] = pd.to_datetime(combined["month_end"])
            combined = combined.drop_duplicates(["month_end", "ts_code"], keep="last")
            combined = combined.sort_values(["month_end", "estimated_size_yi"], ascending=[True, False])
            combined.to_parquet(SNAPSHOT_CACHE, index=False)
            parts = [combined]
            print(
                f"fetched {number}/{len(pending)} through {date.date()} "
                f"({len(combined)} buffered rows)",
                flush=True,
            )
    if not pending:
        print(f"cache already complete: {SNAPSHOT_CACHE}", flush=True)


if __name__ == "__main__":
    main()
