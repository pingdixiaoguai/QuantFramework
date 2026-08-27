"""Point-in-time ETF fund-share observations from Tushare."""

from __future__ import annotations

from datetime import date

import pandas as pd
import tushare as ts

from data.config import get_tushare_token


def fetch_fund_share(
    asset_code: str,
    start_date: date,
    end_date: date,
    *,
    pro=None,
) -> pd.Series:
    """Return dated ETF shares without moving observations backward in time."""

    if end_date < start_date:
        raise ValueError("fund-share end date cannot precede start date")
    client = ts.pro_api(get_tushare_token()) if pro is None else pro
    frame = client.fund_share(
        ts_code=asset_code,
        start_date=start_date.strftime("%Y%m%d"),
        end_date=end_date.strftime("%Y%m%d"),
    )
    if frame is None or frame.empty:
        return pd.Series(dtype=float, name=asset_code)
    required = {"trade_date", "fd_share"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"fund_share response missing columns: {sorted(missing)}")
    working = frame[["trade_date", "fd_share"]].copy()
    working["date"] = pd.to_datetime(
        working["trade_date"], format="%Y%m%d", errors="raise"
    )
    working["fd_share"] = pd.to_numeric(
        working["fd_share"], errors="coerce"
    )
    working = (
        working.dropna(subset=["fd_share"])
        .sort_values("date")
        .drop_duplicates("date", keep="last")
    )
    result = working.set_index("date")["fd_share"].astype(float)
    result.name = asset_code
    return result
