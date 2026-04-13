"""Tests for factors.quality_momentum."""

import numpy as np
import pandas as pd

from factors.quality_momentum import METADATA, compute


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
        df = _make_df([100.0 + i for i in range(30)])
        result = compute(df)
        assert len(result) == len(df)

    def test_first_rows_are_nan(self):
        df = _make_df([100.0 + i for i in range(30)])
        result = compute(df)
        # First min_history-1 = 20 rows should be NaN
        assert result.iloc[:20].isna().all()

    def test_later_rows_are_finite(self):
        df = _make_df([100.0 + i for i in range(30)])
        result = compute(df)
        tail = result.iloc[20:]
        assert tail.notna().all()
        assert np.isfinite(tail).all()


class TestEfficiencyRatio:
    def test_straight_line_has_high_er(self):
        """A straight-line move should produce ER close to 1."""
        # Monotonically increasing: 100, 101, 102, ..., 130
        prices = [100.0 + i for i in range(31)]
        df = _make_df(prices)
        result = compute(df)
        # QMom = momentum * ER; for a straight line ER ≈ 1
        # so QMom ≈ momentum
        last_qmom = result.iloc[-1]
        last_momentum = (prices[-1] / prices[-21]) - 1
        # ER should be very close to 1, so QMom ≈ momentum
        assert abs(last_qmom - last_momentum) < 0.01

    def test_choppy_path_has_lower_qmom(self):
        """A zigzag path with same net move should have lower QMom."""
        n = 31
        # Straight line
        straight = [100.0 + i for i in range(n)]
        # Zigzag: same start and end, but oscillates
        zigzag = [100.0] * n
        for i in range(1, n):
            if i % 2 == 1:
                zigzag[i] = zigzag[i - 1] + 5.0
            else:
                zigzag[i] = zigzag[i - 1] - 4.0
        # Adjust endpoint to match straight line endpoint
        zigzag[-1] = straight[-1]

        qmom_straight = compute(_make_df(straight)).iloc[-1]
        qmom_zigzag = compute(_make_df(zigzag)).iloc[-1]

        assert qmom_straight > qmom_zigzag


class TestMetadata:
    def test_required_fields(self):
        for field in ("name", "params", "min_history", "direction"):
            assert field in METADATA

    def test_direction(self):
        assert METADATA["direction"] == "higher_better"
