"""Fetch point-in-time Tushare fields for the fixed four-ETF tail-factor study."""

from __future__ import annotations

import time
from pathlib import Path
import sys

import pandas as pd
import tushare as ts

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from data.config import get_tushare_token

HERE = Path(__file__).resolve().parent
CACHE = ROOT / "data/db/four_etf_tushare_fields.parquet"
COVERAGE = HERE / "2026-08-17_four_etf_tushare_fields_coverage.csv"
CORES = ["510300.SH", "159915.SZ", "513100.SH", "518880.SH"]
START = "20130101"
END = "20260814"


def request_with_retry(callable_, attempts: int = 5) -> pd.DataFrame:
    for attempt in range(attempts):
        try:
            return callable_()
        except Exception:
            if attempt == attempts - 1:
                raise
            time.sleep(2 ** attempt)
    raise AssertionError("unreachable")


def fetch_share_history(pro, code: str) -> pd.DataFrame:
    chunks = []
    for year in range(2013, 2027):
        chunk = request_with_retry(
            lambda code=code, year=year: pro.fund_share(
                ts_code=code,
                start_date=f"{year}0101",
                end_date=f"{year}1231" if year < 2026 else END,
            )
        )
        chunks.append(chunk)
    return pd.concat(chunks, ignore_index=True)


def main() -> None:
    pro = ts.pro_api(get_tushare_token())
    frames = []
    coverage_rows = []
    for code in CORES:
        daily = request_with_retry(
            lambda code=code: pro.fund_daily(ts_code=code, start_date=START, end_date=END)
        )
        share = fetch_share_history(pro, code)
        nav = request_with_retry(
            lambda code=code: pro.fund_nav(ts_code=code, start_date=START, end_date=END)
        )
        daily = daily.rename(columns={"trade_date": "date"})
        share = share.rename(columns={"trade_date": "date"})
        nav = nav.rename(columns={"ann_date": "date"})
        for frame in (daily, share, nav):
            frame["date"] = pd.to_datetime(frame["date"], format="%Y%m%d")
        daily = daily.sort_values("date").drop_duplicates("date", keep="last")
        share = share[["date", "fd_share"]].sort_values("date").drop_duplicates("date", keep="last")
        nav_columns = ["date", "nav_date", "unit_nav", "accum_nav", "adj_nav", "update_flag"]
        nav = nav[nav_columns].sort_values("date").drop_duplicates("date", keep="last")
        merged = daily.merge(share, on="date", how="left").merge(nav, on="date", how="left")
        merged["ts_code"] = code
        frames.append(merged)
        coverage_rows.append(
            {
                "ts_code": code,
                "fund_daily_rows": len(daily),
                "fund_daily_start": daily["date"].min(),
                "fund_daily_end": daily["date"].max(),
                "amount_nonnull": int(daily["amount"].notna().sum()),
                "fund_share_rows": len(share),
                "fund_share_start": share["date"].min(),
                "fund_share_end": share["date"].max(),
                "fund_nav_rows": len(nav),
                "fund_nav_announcement_start": nav["date"].min(),
                "fund_nav_announcement_end": nav["date"].max(),
                "unit_nav_nonnull": int(nav["unit_nav"].notna().sum()),
            }
        )
        print(f"fetched {code}: daily={len(daily)} share={len(share)} nav={len(nav)}", flush=True)

    result = pd.concat(frames, ignore_index=True).sort_values(["ts_code", "date"])
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(CACHE, index=False)
    coverage = pd.DataFrame(coverage_rows)
    coverage.to_csv(COVERAGE, index=False)
    print(f"saved {len(result)} rows to {CACHE}")
    print(coverage.to_string(index=False))


if __name__ == "__main__":
    main()
