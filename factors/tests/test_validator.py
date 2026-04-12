"""Tests for factors.validator."""

import pandas as pd
import pytest

from factors.validator import validate

_METADATA = {"name": "test_factor", "min_history": 5}


def _make_df(n: int) -> pd.DataFrame:
    return pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=n),
        "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 100.0,
    })


class TestLengthMismatch:
    def test_raises_on_wrong_length(self):
        df = _make_df(10)
        series = pd.Series([1.0] * 8, index=df["date"][:8])
        with pytest.raises(ValueError, match="length mismatch"):
            validate(series, df, _METADATA)


class TestTailNanRejected:
    def test_raises_on_nan_after_min_history(self):
        df = _make_df(10)
        values = [float("nan")] * 4 + [1.0] * 5 + [float("nan")]
        series = pd.Series(values, index=df["date"])
        with pytest.raises(ValueError, match="NaN"):
            validate(series, df, _METADATA)


class TestPrefixNanAllowed:
    def test_passes_with_nan_in_prefix_only(self):
        df = _make_df(10)
        values = [float("nan")] * 4 + [1.0] * 6
        series = pd.Series(values, index=df["date"])
        validate(series, df, _METADATA)  # should not raise
