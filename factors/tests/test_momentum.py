"""Tests for factors.momentum."""

import numpy as np
import pandas as pd

from factors.momentum import METADATA, compute


class TestComputeShapeAndDtype:
    def test_output_shape_dtype_nans(self):
        n = 50
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=n),
            "open": np.random.default_rng(42).uniform(1, 10, n),
            "high": np.random.default_rng(42).uniform(1, 10, n),
            "low": np.random.default_rng(42).uniform(1, 10, n),
            "close": np.random.default_rng(42).uniform(1, 10, n),
            "volume": np.random.default_rng(42).uniform(100, 1000, n),
        })

        result = compute(df)

        assert len(result) == n
        assert result.dtype == np.float64
        # First 20 rows (indices 0..19) should be NaN (pct_change(20) needs 21 values)
        assert result.iloc[: METADATA["min_history"] - 1].isna().all()
        # From row 20 onward, no NaN
        assert not result.iloc[METADATA["min_history"] - 1 :].isna().any()
