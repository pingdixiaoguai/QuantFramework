"""Tests for merge/dedup logic in data.store."""

import pandas as pd
import pytest

from data.store import fill_adjustment_factors, merge_and_save, query, read_local, read_storage


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


class TestDedupPreservesHistory:
    def test_overlapping_dates_keep_existing(self, tmp_path):
        asset = "TEST.SH"
        # Write initial data directly as parquet
        old = _make_df([
            {"date": "2024-01-01", "raw_open": 1.0, "raw_high": 1.5, "raw_low": 0.9, "raw_close": 1.2, "volume": 100, "adj_factor": 1.0},
            {"date": "2024-01-02", "raw_open": 1.1, "raw_high": 1.6, "raw_low": 1.0, "raw_close": 1.3, "volume": 200, "adj_factor": 1.0},
        ])
        path = tmp_path / f"{asset}.parquet"
        old.to_parquet(path, index=False)

        # Merge new data with overlapping date (2024-01-02) and new date.
        new_raw = _make_raw_df([
            {"trade_date": "20240102", "open": 9.0, "high": 9.5, "low": 8.9, "close": 9.2, "vol": 900, "adj_factor": 2.0},
            {"trade_date": "20240103", "open": 1.2, "high": 1.7, "low": 1.1, "close": 1.4, "vol": 300, "adj_factor": 2.0},
        ])
        merge_and_save(asset, new_raw)

        result = read_storage(asset)
        assert len(result) == 3  # 3 unique dates
        # The overlapping date should keep existing history unchanged.
        row_jan2 = result[result["date"] == pd.Timestamp("2024-01-02")].iloc[0]
        assert row_jan2["raw_close"] == 1.3
        assert row_jan2["adj_factor"] == 1.0


class TestSortAscending:
    def test_merged_data_sorted_by_date(self, tmp_path):
        asset = "TEST2.SH"
        # New data arrives in reverse order (Tushare returns newest first)
        new_raw = _make_raw_df([
            {"trade_date": "20240105", "open": 1.0, "high": 1.5, "low": 0.9, "close": 1.2, "vol": 100, "adj_factor": 1.0},
            {"trade_date": "20240103", "open": 1.1, "high": 1.6, "low": 1.0, "close": 1.3, "vol": 200, "adj_factor": 1.0},
            {"trade_date": "20240101", "open": 1.2, "high": 1.7, "low": 1.1, "close": 1.4, "vol": 300, "adj_factor": 1.0},
        ])
        merge_and_save(asset, new_raw)

        result = read_storage(asset)
        dates = result["date"].tolist()
        assert dates == sorted(dates)


class TestHFQProjection:
    def test_read_local_projects_raw_prices_with_fixed_first_factor(self):
        asset = "TEST3.SH"
        raw = _make_raw_df([
            {"trade_date": "20240101", "open": 10.0, "high": 10.0, "low": 10.0, "close": 10.0, "vol": 100, "adj_factor": 2.0},
            {"trade_date": "20240102", "open": 10.0, "high": 10.0, "low": 10.0, "close": 10.0, "vol": 100, "adj_factor": 4.0},
        ])
        merge_and_save(asset, raw)

        result = read_local(asset)

        assert list(result.columns) == ["date", "open", "high", "low", "close", "volume"]
        assert result.loc[0, "close"] == 10.0
        assert result.loc[1, "close"] == 20.0

    def test_query_filters_projected_hfq_rows(self):
        asset = "TEST4.SH"
        raw = _make_raw_df([
            {"trade_date": "20240101", "open": 10.0, "high": 10.0, "low": 10.0, "close": 10.0, "vol": 100, "adj_factor": 2.0},
            {"trade_date": "20240102", "open": 10.0, "high": 10.0, "low": 10.0, "close": 10.0, "vol": 100, "adj_factor": 4.0},
        ])
        merge_and_save(asset, raw)

        result = query(asset, pd.Timestamp("2024-01-02").date(), pd.Timestamp("2024-01-02").date())

        assert len(result) == 1
        assert result.iloc[0]["close"] == 20.0


class TestAdjustmentFactorFill:
    def test_missing_middle_factor_uses_previous_factor_only(self):
        frame = _make_df([
            {"date": "2024-01-01", "adj_factor": 1.0},
            {"date": "2024-01-02", "adj_factor": None},
            {"date": "2024-01-03", "adj_factor": 5.0},
        ])

        result = fill_adjustment_factors(frame)

        assert result.loc[1, "adj_factor"] == 1.0

    def test_leading_missing_factor_uses_first_future_factor(self):
        frame = _make_df([
            {"date": "2024-01-01", "adj_factor": None},
            {"date": "2024-01-02", "adj_factor": 3.0},
        ])

        result = fill_adjustment_factors(frame)

        assert result.loc[0, "adj_factor"] == 3.0
