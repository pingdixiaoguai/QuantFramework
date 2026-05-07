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


class TestEfficiencyRatio:
    def test_straight_line_er_equals_one(self):
        """Strictly monotonic prices => ER = 1.0 exactly => score == R/vol."""
        prices = [100.0 + i for i in range(80)]
        df = _make_df(prices)
        result = compute(df)

        # Manual R/vol for the last row
        log_close = np.log(np.array(prices))
        log_ret = np.diff(log_close)
        n = METADATA["params"]["window"]
        R = log_close[-1] - log_close[-1 - n]
        vol = np.std(log_ret[-n:], ddof=1) * np.sqrt(n)
        expected_score = R / vol  # ER = 1, no floor / no winsor at this scale

        assert abs(result.iloc[-1] - expected_score) < 1e-9

    def test_smooth_path_beats_choppy_same_endpoints(self):
        """Two paths, same start and end after >N steps; smoother wins."""
        n_pts = 80
        smooth = [100.0 + i * 0.5 for i in range(n_pts)]

        # Zigzag oscillating but ending at the same point as smooth[-1]
        zigzag = [smooth[0]]
        for i in range(1, n_pts - 1):
            step = 3.0 if i % 2 == 1 else -2.5
            zigzag.append(zigzag[-1] + step)
        zigzag.append(smooth[-1])  # force matching endpoint

        score_smooth = compute(_make_df(smooth)).iloc[-1]
        score_zigzag = compute(_make_df(zigzag)).iloc[-1]
        assert score_smooth > score_zigzag

    def test_score_equals_ram_times_er_on_zigzag(self):
        """Analytic verification: score == (R/vol) * ER for a non-monotonic fixture.

        Regression guard against silently dropping the ER multiplier.
        On the zigzag fixture (ER << 1), score must NOT equal the bare R/vol.
        """
        n_pts = 80
        zigzag = [100.0]
        for i in range(1, n_pts - 1):
            zigzag.append(zigzag[-1] + (3.0 if i % 2 == 1 else -2.5))
        zigzag.append(100.0 + (n_pts - 1) * 0.5)

        df = _make_df(zigzag)
        result = compute(df).iloc[-1]

        # Analytic recomputation
        n = METADATA["params"]["window"]
        log_close = np.log(np.array(zigzag))
        log_ret = np.diff(log_close)
        R = log_close[-1] - log_close[-1 - n]
        path = float(np.sum(np.abs(log_ret[-n:])))
        vol = float(np.std(log_ret[-n:], ddof=1) * np.sqrt(n))
        er = abs(R) / path

        # Confirm we're actually in the choppy regime (otherwise the regression guard is empty)
        assert er < 0.5, f"Test fixture not choppy enough (ER={er:.3f}); revise"

        bare_ram = R / vol
        expected = bare_ram * er
        assert abs(result - expected) < 1e-9
        # Confirm score is NOT equal to the bare R/vol (i.e., ER actually changed it)
        assert abs(result - bare_ram) > 1e-3
