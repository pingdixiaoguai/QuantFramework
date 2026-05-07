"""Tests for factors.risk_adjusted_quality_momentum."""

import numpy as np
import pandas as pd

from factors.risk_adjusted_quality_momentum import METADATA, compute


def _make_df(prices: list[float]) -> pd.DataFrame:
    n = len(prices)
    dates = pd.bdate_range("2024-01-01", periods=n)
    return pd.DataFrame({
        "date": dates,
        "open": prices,
        "high": [p * 1.01 for p in prices],
        "low": [p * 0.99 for p in prices],
        "close": prices,
        "volume": [1000.0] * n,
    })


class TestOutputShape:
    def test_length_matches_input(self):
        df = _make_df([100.0 + i for i in range(80)])
        result = compute(df)
        assert len(result) == len(df)

    def test_index_is_date(self):
        df = _make_df([100.0 + i for i in range(80)])
        result = compute(df)
        assert (result.index == df["date"]).all()

    def test_dtype_is_float(self):
        df = _make_df([100.0 + i for i in range(80)])
        result = compute(df)
        assert result.dtype == float

    def test_first_60_rows_are_nan(self):
        df = _make_df([100.0 + i for i in range(80)])
        result = compute(df)
        # min_history - 1 = 60
        assert result.iloc[:60].isna().all()

    def test_rows_60_and_after_are_finite(self):
        df = _make_df([100.0 + i for i in range(80)])
        result = compute(df)
        tail = result.iloc[60:]
        assert tail.notna().all()
        assert np.isfinite(tail).all()

    def test_input_df_not_mutated(self):
        df = _make_df([100.0 + i for i in range(80)])
        before = df.copy(deep=True)
        compute(df)
        pd.testing.assert_frame_equal(df, before)
