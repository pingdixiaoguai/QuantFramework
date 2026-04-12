"""Tests for standardization.methods."""

import numpy as np
import pandas as pd
import pytest

from standardization.methods import standardize


class TestZscoreBasic:
    def test_linear_series_produces_finite_values(self):
        dates = pd.date_range("2024-01-01", periods=120)
        series = pd.Series(np.arange(120, dtype=float), index=dates)
        raw = {"linear": series}

        result = standardize(raw, method="z_score", window=60)

        valid = result["linear"].dropna()
        assert len(valid) > 0
        # All non-NaN values should be finite
        assert np.isfinite(valid).all()
        # A linear series should have constant z-score (all values equal)
        assert valid.std() < 1e-10


class TestPercentileRange:
    def test_values_in_zero_one(self):
        dates = pd.date_range("2024-01-01", periods=120)
        series = pd.Series(np.random.default_rng(42).uniform(0, 100, 120), index=dates)
        raw = {"rand": series}

        result = standardize(raw, method="percentile", window=60)

        valid = result["rand"].dropna()
        assert (valid >= 0).all()
        assert (valid <= 1).all()


class TestCrossSectionalRankRaises:
    def test_raises_not_implemented(self):
        raw = {"dummy": pd.Series([1.0, 2.0, 3.0])}
        with pytest.raises(NotImplementedError, match="Phase 4"):
            standardize(raw, method="cross_sectional_rank")
