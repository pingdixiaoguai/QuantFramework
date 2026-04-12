"""Tests for merge/dedup logic in data.store."""

import pandas as pd
import pytest

from data.store import DB_DIR, merge_and_save, read_local


@pytest.fixture(autouse=True)
def clean_db(tmp_path, monkeypatch):
    """Redirect DB_DIR to a temp directory for each test."""
    monkeypatch.setattr("data.store.DB_DIR", tmp_path)
    yield


def _make_df(rows: list[dict]) -> pd.DataFrame:
    """Create a normalized DataFrame (as if already stored)."""
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df


def _make_raw_df(rows: list[dict]) -> pd.DataFrame:
    """Create a raw Tushare-style DataFrame for merge_and_save."""
    return pd.DataFrame(rows)


class TestDedupKeepsLatest:
    def test_overlapping_dates_keep_new(self, tmp_path):
        asset = "TEST.SH"
        # Write initial data directly as parquet
        old = _make_df([
            {"date": "2024-01-01", "open": 1.0, "high": 1.5, "low": 0.9, "close": 1.2, "volume": 100},
            {"date": "2024-01-02", "open": 1.1, "high": 1.6, "low": 1.0, "close": 1.3, "volume": 200},
        ])
        path = tmp_path / f"{asset}.parquet"
        old.to_parquet(path, index=False)

        # Merge new data with overlapping date (2024-01-02) and new date
        new_raw = _make_raw_df([
            {"trade_date": "20240102", "open": 9.0, "high": 9.5, "low": 8.9, "close": 9.2, "vol": 900},
            {"trade_date": "20240103", "open": 1.2, "high": 1.7, "low": 1.1, "close": 1.4, "vol": 300},
        ])
        merge_and_save(asset, new_raw)

        result = read_local(asset)
        assert len(result) == 3  # 3 unique dates
        # The overlapping date should have the NEW values
        row_jan2 = result[result["date"] == pd.Timestamp("2024-01-02")].iloc[0]
        assert row_jan2["close"] == 9.2


class TestSortAscending:
    def test_merged_data_sorted_by_date(self, tmp_path):
        asset = "TEST2.SH"
        # New data arrives in reverse order (Tushare returns newest first)
        new_raw = _make_raw_df([
            {"trade_date": "20240105", "open": 1.0, "high": 1.5, "low": 0.9, "close": 1.2, "vol": 100},
            {"trade_date": "20240103", "open": 1.1, "high": 1.6, "low": 1.0, "close": 1.3, "vol": 200},
            {"trade_date": "20240101", "open": 1.2, "high": 1.7, "low": 1.1, "close": 1.4, "vol": 300},
        ])
        merge_and_save(asset, new_raw)

        result = read_local(asset)
        dates = result["date"].tolist()
        assert dates == sorted(dates)
