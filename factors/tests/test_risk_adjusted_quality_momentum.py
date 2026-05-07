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
        """Strictly monotonic prices => ER = 1.0 exactly => score == R/adj_vol."""
        prices = [100.0 + i for i in range(80)]
        df = _make_df(prices)
        result = compute(df)

        # Manual R/adj_vol for the last row (floor may bind on this low-vol fixture)
        log_close = np.log(np.array(prices))
        log_ret = np.diff(log_close)
        n = METADATA["params"]["window"]
        vol_floor_annual = METADATA["params"]["vol_floor_annual"]
        R = log_close[-1] - log_close[-1 - n]
        vol = np.std(log_ret[-n:], ddof=1) * np.sqrt(n)
        floor_n = vol_floor_annual * np.sqrt(n / 252.0)
        adj_vol = max(vol, floor_n)
        expected_score = R / adj_vol  # ER = 1, no winsor at this scale

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


class TestVolFloor:
    def test_floor_binds_when_vol_is_zero(self):
        """Constant-rate log growth => vol=0; floor must rescue the divisor."""
        # Tiny constant daily log return; std == 0 exactly
        daily_log = 0.0001
        prices = [100.0 * np.exp(daily_log * i) for i in range(80)]
        df = _make_df(prices)
        result = compute(df)
        last = result.iloc[-1]

        # Manual expected value
        n = METADATA["params"]["window"]                      # 60
        vol_floor_annual = METADATA["params"]["vol_floor_annual"]  # 0.08
        floor_n = vol_floor_annual * np.sqrt(n / 252.0)
        R = daily_log * n
        # ER == 1 (constant positive log return: |R| == path)
        expected = (R / floor_n) * 1.0

        assert np.isfinite(last)
        assert abs(last - expected) < 1e-9

    def test_floor_does_not_bind_for_normal_vol(self):
        """A noisy ~1% daily-vol series: vol_60 ≈ 7.7% > floor ≈ 3.9%, floor irrelevant."""
        rng = np.random.default_rng(seed=42)
        log_rets = rng.normal(loc=0.0005, scale=0.01, size=79)
        log_close = np.concatenate([[np.log(100.0)], np.log(100.0) + np.cumsum(log_rets)])
        prices = list(np.exp(log_close))
        df = _make_df(prices)
        result = compute(df).iloc[-1]

        # Recompute the no-floor score and assert they match exactly
        n = METADATA["params"]["window"]
        log_ret = np.diff(log_close)
        R = log_close[-1] - log_close[-1 - n]
        path = np.sum(np.abs(log_ret[-n:]))
        vol = np.std(log_ret[-n:], ddof=1) * np.sqrt(n)
        er = abs(R) / path
        expected = (R / vol) * er  # vol > floor so adj_vol == vol

        assert abs(result - expected) < 1e-9

    def test_vol_floor_annual_param_override(self):
        """Caller-supplied vol_floor_annual must override METADATA default."""
        daily_log = 0.0001
        prices = [100.0 * np.exp(daily_log * i) for i in range(80)]
        df = _make_df(prices)

        n = METADATA["params"]["window"]
        custom_floor_annual = 0.20  # much larger; floor binds harder
        result = compute(df, params={"vol_floor_annual": custom_floor_annual}).iloc[-1]

        floor_n = custom_floor_annual * np.sqrt(n / 252.0)
        R = daily_log * n
        expected = (R / floor_n) * 1.0
        assert abs(result - expected) < 1e-9
