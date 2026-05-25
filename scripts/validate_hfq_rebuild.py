"""Validate raw+fund_adj HFQ reconstruction without writing local Parquet."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import pandas as pd
import tushare as ts

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.config import get_tushare_token
from data.store import fill_adjustment_factors


ASSET_STARTS = {
    "510300.SH": "20130104",
    "159915.SZ": "20130104",
    "513100.SH": "20130515",
    "518880.SH": "20130729",
}
END_DATE = "20260519"
ABNORMAL_THRESHOLD = 0.30

EX_DIVIDEND_CHECKS = {
    "510300.SH": [
        "2014-01-21",
        "2015-01-20",
        "2016-01-20",
        "2019-12-11",
        "2021-01-18",
        "2022-01-19",
        "2024-01-18",
        "2025-06-18",
        "2026-01-19",
    ]
}

SENSITIVE_DATES = {
    "159915.SZ": ["2021-02-08"],
    "513100.SH": ["2020-09-18", "2022-01-13", "2022-01-14"],
    "518880.SH": ["2020-09-18"],
}


@dataclass(frozen=True)
class RebuiltAsset:
    code: str
    raw: pd.DataFrame
    adj: pd.DataFrame
    frame: pd.DataFrame
    raw_only_dates: list[pd.Timestamp]
    adj_only_dates: list[pd.Timestamp]


def _ymd(value: object) -> str:
    return pd.Timestamp(value).strftime("%Y%m%d")


def _date_chunks(start_date: str, end_date: str) -> list[tuple[str, str]]:
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    chunks: list[tuple[str, str]] = []
    current = pd.Timestamp(year=start.year, month=1, day=1)
    while current <= end:
        chunk_start = max(start, current)
        chunk_end = min(end, pd.Timestamp(year=current.year, month=12, day=31))
        if chunk_start <= chunk_end:
            chunks.append((_ymd(chunk_start), _ymd(chunk_end)))
        current = pd.Timestamp(year=current.year + 1, month=1, day=1)
    return chunks


def _fetch_raw(pro, code: str) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for start, end in _date_chunks(ASSET_STARTS[code], END_DATE):
        df = ts.pro_bar(
            ts_code=code,
            api=pro,
            asset="FD",
            start_date=start,
            end_date=end,
            adj=None,
        )
        if df is not None and not df.empty:
            parts.append(df)
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True).drop_duplicates("trade_date")
    out["date"] = pd.to_datetime(out["trade_date"], format="%Y%m%d")
    return out.sort_values("date").reset_index(drop=True)


def _fetch_adj(pro, code: str) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for start, end in _date_chunks(ASSET_STARTS[code], END_DATE):
        df = pro.fund_adj(ts_code=code, start_date=start, end_date=end)
        if df is not None and not df.empty:
            parts.append(df)
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True).drop_duplicates("trade_date")
    out["date"] = pd.to_datetime(out["trade_date"], format="%Y%m%d")
    return out.sort_values("date").reset_index(drop=True)


def _rebuild_asset(pro, code: str) -> RebuiltAsset:
    raw = _fetch_raw(pro, code)
    adj = _fetch_adj(pro, code)
    raw_dates = set(raw["date"])
    adj_dates = set(adj["date"])
    raw_only = sorted(raw_dates - adj_dates)
    adj_only = sorted(adj_dates - raw_dates)

    merged = raw.merge(adj[["date", "adj_factor"]], on="date", how="left")
    merged = fill_adjustment_factors(merged)
    baseline = float(merged["adj_factor"].iloc[0])
    for column in ["open", "high", "low", "close"]:
        merged[f"hfq_{column}"] = merged[column] * merged["adj_factor"] / baseline
    merged["raw_close_ret"] = merged["close"].pct_change()
    merged["hfq_close_ret"] = merged["hfq_close"].pct_change()
    merged["raw_overnight"] = merged["open"] / merged["close"].shift(1) - 1
    merged["hfq_overnight"] = merged["hfq_open"] / merged["hfq_close"].shift(1) - 1
    return RebuiltAsset(code, raw, adj, merged, raw_only, adj_only)


def _format_window(frame: pd.DataFrame) -> str:
    columns = [
        "date",
        "pre_close",
        "open",
        "close",
        "adj_factor",
        "raw_close_ret",
        "hfq_close_ret",
        "raw_overnight",
        "hfq_overnight",
        "hfq_open",
        "hfq_close",
    ]
    view = frame.loc[:, [column for column in columns if column in frame.columns]].copy()
    return view.to_string(
        index=False,
        formatters={
            "date": lambda value: pd.Timestamp(value).strftime("%Y-%m-%d"),
            "pre_close": lambda value: f"{value:.4f}",
            "open": lambda value: f"{value:.4f}",
            "close": lambda value: f"{value:.4f}",
            "adj_factor": lambda value: f"{value:.6f}",
            "raw_close_ret": lambda value: "" if pd.isna(value) else f"{value:.4%}",
            "hfq_close_ret": lambda value: "" if pd.isna(value) else f"{value:.4%}",
            "raw_overnight": lambda value: "" if pd.isna(value) else f"{value:.4%}",
            "hfq_overnight": lambda value: "" if pd.isna(value) else f"{value:.4%}",
            "hfq_open": lambda value: f"{value:.4f}",
            "hfq_close": lambda value: f"{value:.4f}",
        },
    )


def _window_around(frame: pd.DataFrame, date_text: str, radius: int = 1) -> pd.DataFrame:
    date = pd.Timestamp(date_text)
    matches = frame.index[frame["date"].eq(date)].tolist()
    if not matches:
        return pd.DataFrame(columns=frame.columns)
    pos = matches[0]
    return frame.iloc[max(0, pos - radius) : min(len(frame), pos + radius + 1)]


def _nearest_window(frame: pd.DataFrame, date_text: str, radius: int = 1) -> pd.DataFrame:
    date = pd.Timestamp(date_text)
    pos = int(frame["date"].searchsorted(date))
    start = max(0, pos - radius)
    end = min(len(frame), pos + radius + 1)
    return frame.iloc[start:end]


def main() -> None:
    pro = ts.pro_api(get_tushare_token())
    assets = [_rebuild_asset(pro, code) for code in ASSET_STARTS]

    print("HFQ_REBUILD_VALIDATION")
    print(f"abnormal_threshold={ABNORMAL_THRESHOLD:.0%}")
    print("")

    for asset in assets:
        frame = asset.frame
        max_idx = frame["hfq_close_ret"].abs().idxmax()
        abnormal = frame.loc[frame["hfq_close_ret"].abs().gt(ABNORMAL_THRESHOLD)]
        print(
            f"{asset.code}: raw_rows={len(asset.raw)} adj_rows={len(asset.adj)} "
            f"range={frame['date'].min().date()}~{frame['date'].max().date()} "
            f"raw_only={len(asset.raw_only_dates)} adj_only={len(asset.adj_only_dates)} "
            f"hfq_abs_ret_gt_{ABNORMAL_THRESHOLD:.0%}={len(abnormal)} "
            f"max_hfq_abs_ret={abs(frame.at[max_idx, 'hfq_close_ret']):.4%} "
            f"max_hfq_abs_ret_date={frame.at[max_idx, 'date'].date()}"
        )
        if asset.raw_only_dates:
            print(
                "  raw_only_dates="
                + ", ".join(date.strftime("%Y-%m-%d") for date in asset.raw_only_dates)
            )
        if asset.adj_only_dates:
            print(
                "  adj_only_dates="
                + ", ".join(date.strftime("%Y-%m-%d") for date in asset.adj_only_dates)
            )

    print("")
    print("CHECK_A_513100_SPLIT")
    asset_513100 = next(asset for asset in assets if asset.code == "513100.SH")
    print(_format_window(_window_around(asset_513100.frame, "2022-01-14", radius=2)))

    print("")
    print("CHECK_B_510300_EX_DIVIDENDS")
    asset_510300 = next(asset for asset in assets if asset.code == "510300.SH")
    for date_text in EX_DIVIDEND_CHECKS["510300.SH"]:
        window = _window_around(asset_510300.frame, date_text, radius=1)
        if window.empty:
            print(f"510300.SH {date_text}: missing raw trading day")
        else:
            event_row = window.loc[window["date"].eq(pd.Timestamp(date_text))].iloc[0]
            print(
                f"510300.SH {date_text}: raw_overnight={event_row['raw_overnight']:.4%} "
                f"hfq_overnight={event_row['hfq_overnight']:.4%} "
                f"raw_close_ret={event_row['raw_close_ret']:.4%} "
                f"hfq_close_ret={event_row['hfq_close_ret']:.4%}"
            )

    print("")
    print("CHECK_C_CALENDAR_MISMATCH_POINTS")
    for asset in assets:
        for date_text in SENSITIVE_DATES.get(asset.code, []):
            exists = asset.frame["date"].eq(pd.Timestamp(date_text)).any()
            print(f"{asset.code} {date_text} raw_trading_day={exists}")
            print(_format_window(_nearest_window(asset.frame, date_text, radius=1)))

    print("")
    print("CHECK_D_ABNORMAL_ROWS")
    for asset in assets:
        abnormal = asset.frame.loc[
            asset.frame["hfq_close_ret"].abs().gt(ABNORMAL_THRESHOLD),
            ["date", "close", "adj_factor", "hfq_close", "hfq_close_ret"],
        ]
        print(f"{asset.code}: abnormal_rows={len(abnormal)}")
        if not abnormal.empty:
            print(_format_window(abnormal))


if __name__ == "__main__":
    main()
