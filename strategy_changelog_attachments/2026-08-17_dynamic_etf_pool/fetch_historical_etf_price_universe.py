"""Fetch raw ETF bars and adjustment factors for the Phase-3 audit universe."""

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
from data.sync import _merge_raw_with_adj  # noqa: E402

UNION_PATH = Path(__file__).resolve().parent / "2026-08-17_dynamic_etf_pool_phase3_union.csv"
END_DATE = pd.Timestamp("2026-08-14")


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


def chunks(start: pd.Timestamp, end: pd.Timestamp) -> list[tuple[str, str]]:
    out = []
    current = start
    while current <= end:
        chunk_end = min(pd.Timestamp(current.year + 3, 12, 31), end)
        out.append((current.strftime("%Y%m%d"), chunk_end.strftime("%Y%m%d")))
        current = chunk_end + pd.Timedelta(days=1)
    return out


def fetch_code(pro, code: str, list_date: pd.Timestamp) -> int:
    existing = store.read_storage(code)
    if existing is not None and not existing.empty:
        start = pd.to_datetime(existing["date"]).max() + pd.Timedelta(days=1)
    else:
        start = max(pd.Timestamp("2013-01-01"), list_date)
    if start > END_DATE:
        return 0

    raw_parts, adj_parts = [], []
    for chunk_start, chunk_end in chunks(start, END_DATE):
        raw = request_with_retry(
            lambda chunk_start=chunk_start, chunk_end=chunk_end: ts.pro_bar(
                ts_code=code,
                api=pro,
                asset="FD",
                start_date=chunk_start,
                end_date=chunk_end,
                adj=None,
            )
        )
        adj = request_with_retry(
            lambda chunk_start=chunk_start, chunk_end=chunk_end: pro.fund_adj(
                ts_code=code,
                start_date=chunk_start,
                end_date=chunk_end,
            )
        )
        if raw is not None and not raw.empty:
            raw_parts.append(raw)
        if adj is not None and not adj.empty:
            adj_parts.append(adj)
    if not raw_parts:
        return 0
    if not adj_parts:
        raise RuntimeError(f"fund_adj returned no rows for {code}")
    raw = pd.concat(raw_parts, ignore_index=True).drop_duplicates("trade_date")
    adj = pd.concat(adj_parts, ignore_index=True).drop_duplicates("trade_date")
    merged = _merge_raw_with_adj(raw, adj)
    store.merge_and_save(code, merged)
    return len(merged)


def main() -> None:
    if not UNION_PATH.exists():
        raise RuntimeError("run build_historical_etf_universe_audit.py first")
    universe = pd.read_csv(UNION_PATH)
    universe["list_date"] = pd.to_datetime(universe["list_date"])
    pro = ts.pro_api(get_tushare_token())
    total = len(universe)
    for number, row in enumerate(universe.itertuples(index=False), 1):
        added = fetch_code(pro, row.ts_code, row.list_date)
        print(f"{number:03d}/{total} {row.ts_code}: {added} rows", flush=True)


if __name__ == "__main__":
    main()
