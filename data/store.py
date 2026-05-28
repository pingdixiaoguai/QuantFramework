"""Parquet local storage: raw bars + local HFQ projection."""

from datetime import date
from pathlib import Path

import pandas as pd

DB_DIR = Path(__file__).parent / "db"

# Tushare raw bar columns -> storage columns.
_COLUMN_MAP = {
    "trade_date": "date",
    "open": "raw_open",
    "high": "raw_high",
    "low": "raw_low",
    "close": "raw_close",
    "vol": "volume",
}

_STORAGE_COLUMNS = [
    "date",
    "raw_open",
    "raw_high",
    "raw_low",
    "raw_close",
    "volume",
    "adj_factor",
]

_QUERY_COLUMNS = ["date", "open", "high", "low", "close", "volume"]


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize raw Tushare bars + fund_adj factors to storage schema."""
    df = df.rename(columns=_COLUMN_MAP)
    missing = [column for column in _STORAGE_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"raw price frame missing required columns: {missing}")
    df = df[_STORAGE_COLUMNS].copy()
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    for column in ["raw_open", "raw_high", "raw_low", "raw_close", "volume", "adj_factor"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = fill_adjustment_factors(df)
    return df


def _parquet_path(asset_code: str) -> Path:
    return DB_DIR / f"{asset_code}.parquet"


def fill_adjustment_factors(df: pd.DataFrame) -> pd.DataFrame:
    """Fill sparse adjustment factors without looking ahead except at the head.

    The factor is a monotone step series. Missing raw trading days inherit the
    previous known factor. Only a leading missing prefix may use the first
    future factor because no earlier factor exists.
    """
    out = df.sort_values("date").reset_index(drop=True).copy()
    out["adj_factor"] = out["adj_factor"].ffill()
    if out["adj_factor"].isna().any():
        out["adj_factor"] = out["adj_factor"].bfill()
    if out["adj_factor"].isna().any():
        raise ValueError("adj_factor cannot be filled; no known factor in frame")
    return out


def _is_raw_storage_schema(df: pd.DataFrame) -> bool:
    return all(column in df.columns for column in _STORAGE_COLUMNS)


def _project_hfq(df: pd.DataFrame) -> pd.DataFrame:
    """Project stored raw bars to fixed-baseline HFQ prices for callers."""
    if df.empty:
        return pd.DataFrame(columns=_QUERY_COLUMNS)

    if not _is_raw_storage_schema(df):
        # Legacy qfq Parquet support until the full phase-2 rebuild replaces
        # the local files. New writes always use _STORAGE_COLUMNS.
        return df.loc[:, _QUERY_COLUMNS].copy()

    ordered = fill_adjustment_factors(df)
    baseline = float(ordered["adj_factor"].iloc[0])
    if baseline == 0:
        raise ValueError("first adj_factor must be non-zero")

    projected = pd.DataFrame(
        {
            "date": ordered["date"],
            "open": ordered["raw_open"] * ordered["adj_factor"] / baseline,
            "high": ordered["raw_high"] * ordered["adj_factor"] / baseline,
            "low": ordered["raw_low"] * ordered["adj_factor"] / baseline,
            "close": ordered["raw_close"] * ordered["adj_factor"] / baseline,
            "volume": ordered["volume"],
        }
    )
    return projected


def read_storage(asset_code: str) -> pd.DataFrame | None:
    """Read the stored parquet rows. Returns None if not exists."""
    path = _parquet_path(asset_code)
    if not path.exists():
        return None
    return pd.read_parquet(path)


def read_local(asset_code: str) -> pd.DataFrame | None:
    """Read local data as HFQ open/high/low/close rows for callers."""
    stored = read_storage(asset_code)
    if stored is None:
        return None
    return _project_hfq(stored)


def merge_and_save(asset_code: str, new_df: pd.DataFrame) -> None:
    """Append raw bars and factors to local storage, preserving history."""
    new_df = _normalize_columns(new_df)
    existing = read_storage(asset_code)

    if existing is not None:
        if not _is_raw_storage_schema(existing):
            raise RuntimeError(
                f"{asset_code} uses legacy adjusted-price schema; run the "
                "full raw+adj_factor rebuild before incremental sync"
            )
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df

    # Dedup by date, keep existing rows. Incremental sync must not rewrite
    # history because the local HFQ baseline is fixed to the earliest row.
    combined = combined.drop_duplicates(subset=["date"], keep="first")
    combined = combined.sort_values("date").reset_index(drop=True)

    DB_DIR.mkdir(parents=True, exist_ok=True)
    combined.loc[:, _STORAGE_COLUMNS].to_parquet(_parquet_path(asset_code), index=False)


def query(asset_code: str, start: date, end: date) -> pd.DataFrame:
    """Query local data for an asset within [start, end].

    Returns DataFrame with columns [date, open, high, low, close, volume],
    sorted by date ascending.
    """
    df = read_local(asset_code)
    if df is None:
        return pd.DataFrame(columns=_QUERY_COLUMNS)

    projected = _project_hfq(df)
    mask = (projected["date"] >= pd.Timestamp(start)) & (projected["date"] <= pd.Timestamp(end))
    return projected.loc[mask].reset_index(drop=True)
