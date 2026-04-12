"""Parquet local storage: read, write, merge, and query."""

from datetime import date
from pathlib import Path

import pandas as pd

DB_DIR = Path(__file__).parent / "db"

# Tushare column names -> contract column names
_COLUMN_MAP = {
    "trade_date": "date",
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "vol": "volume",
}


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename Tushare columns to contract names and parse date."""
    df = df.rename(columns=_COLUMN_MAP)
    df = df[["date", "open", "high", "low", "close", "volume"]]
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    return df


def _parquet_path(asset_code: str) -> Path:
    return DB_DIR / f"{asset_code}.parquet"


def read_local(asset_code: str) -> pd.DataFrame | None:
    """Read local parquet file. Returns None if not exists."""
    path = _parquet_path(asset_code)
    if not path.exists():
        return None
    return pd.read_parquet(path)


def merge_and_save(asset_code: str, new_df: pd.DataFrame) -> None:
    """Merge new data with existing local data, dedup, sort, and save."""
    new_df = _normalize_columns(new_df)
    existing = read_local(asset_code)

    if existing is not None:
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df

    # Dedup by date, keep last (newest data wins)
    combined = combined.drop_duplicates(subset=["date"], keep="last")
    combined = combined.sort_values("date").reset_index(drop=True)

    DB_DIR.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(_parquet_path(asset_code), index=False)


def query(asset_code: str, start: date, end: date) -> pd.DataFrame:
    """Query local data for an asset within [start, end].

    Returns DataFrame with columns [date, open, high, low, close, volume],
    sorted by date ascending.
    """
    df = read_local(asset_code)
    if df is None:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])

    mask = (df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))
    return df.loc[mask].reset_index(drop=True)
