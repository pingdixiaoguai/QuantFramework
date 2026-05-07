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
        raw_ram = R / adj_vol
        # Clip may engage on this low-vol linear fixture (raw ~10.46 > 3.0)
        expected_score = min(raw_ram, 3.0)  # ER = 1, so score = clipped ram * 1

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


class TestWinsorize:
    def test_positive_extreme_is_clipped_to_3(self):
        """High constant log return + bound floor => raw R/adj_vol > 3 => clipped."""
        # 0.5% per day compounding => R_60 = 0.30; vol = 0; floor binds at ~0.039
        daily_log = 0.005
        prices = [100.0 * np.exp(daily_log * i) for i in range(80)]
        df = _make_df(prices)
        result = compute(df).iloc[-1]

        n = METADATA["params"]["window"]
        floor_n = METADATA["params"]["vol_floor_annual"] * np.sqrt(n / 252.0)
        R = daily_log * n
        # Sanity-check the clip engages
        assert R / floor_n > 3.0
        # ER == 1 on a strictly monotonic positive series; expected score = 3 * 1
        assert abs(result - 3.0) < 1e-9

    def test_negative_extreme_is_clipped_to_minus_3(self):
        """Symmetric: -0.5% per day => raw R/adj_vol < -3 => clipped."""
        daily_log = -0.005
        prices = [100.0 * np.exp(daily_log * i) for i in range(80)]
        df = _make_df(prices)
        result = compute(df).iloc[-1]

        n = METADATA["params"]["window"]
        floor_n = METADATA["params"]["vol_floor_annual"] * np.sqrt(n / 252.0)
        R = daily_log * n
        assert R / floor_n < -3.0
        # ER on a strictly monotonic negative series is also 1 (|R| == path)
        assert abs(result - (-3.0)) < 1e-9

    def test_in_range_is_not_clipped(self):
        """Within ±3: score must equal (R/adj_vol)*ER exactly, no clipping."""
        rng = np.random.default_rng(seed=7)
        log_rets = rng.normal(loc=0.0005, scale=0.01, size=79)
        log_close = np.concatenate([[np.log(100.0)], np.log(100.0) + np.cumsum(log_rets)])
        prices = list(np.exp(log_close))
        df = _make_df(prices)
        result = compute(df).iloc[-1]

        n = METADATA["params"]["window"]
        log_ret = np.diff(log_close)
        R = log_close[-1] - log_close[-1 - n]
        path = np.sum(np.abs(log_ret[-n:]))
        vol = np.std(log_ret[-n:], ddof=1) * np.sqrt(n)
        floor_n = METADATA["params"]["vol_floor_annual"] * np.sqrt(n / 252.0)
        adj_vol = max(vol, floor_n)
        raw = R / adj_vol
        # Confirm we're inside the winsor band so this test is actually testing "no clip"
        assert -3.0 < raw < 3.0
        expected = raw * (abs(R) / path)
        assert abs(result - expected) < 1e-9


class TestCrossAssetComparability:
    """The point of the factor: low-vol clean trend should beat high-vol bigger-but-noisier trend."""

    def _series_with(self, *, mu: float, sigma: float, n_days: int, seed: int) -> list[float]:
        """Geometric Brownian path with given daily log-return mean/std."""
        rng = np.random.default_rng(seed=seed)
        log_rets = rng.normal(loc=mu, scale=sigma, size=n_days - 1)
        log_close = np.concatenate([[np.log(100.0)],
                                    np.log(100.0) + np.cumsum(log_rets)])
        return list(np.exp(log_close))

    def test_low_vol_clean_trend_beats_high_vol_big_move(self):
        n_days = 61
        # High-vol asset: ~+12% over 60d, daily sigma ~1.57% (annualized ~25%)
        prices_high = self._series_with(mu=0.00189, sigma=0.0157, n_days=n_days, seed=3)
        # Low-vol asset: ~+5% over 60d, daily sigma ~0.5% (annualized ~8%)
        prices_low = self._series_with(mu=0.000813, sigma=0.005, n_days=n_days, seed=2)

        score_high = compute(_make_df(prices_high)).iloc[-1]
        score_low = compute(_make_df(prices_low)).iloc[-1]

        # The low-vol clean trend must score higher
        assert score_low > score_high, (
            f"Expected low-vol clean trend to win; got "
            f"score_low={score_low:.4f} vs score_high={score_high:.4f}"
        )

    def test_raw_momentum_would_have_ranked_them_oppositely(self):
        """Sanity check: under the OLD raw-return logic, high-vol asset wins.
        Demonstrates the new factor genuinely fixes the bias."""
        n_days = 61
        prices_high = self._series_with(mu=0.00189, sigma=0.0157, n_days=n_days, seed=3)
        prices_low = self._series_with(mu=0.000813, sigma=0.005, n_days=n_days, seed=2)

        n = METADATA["params"]["window"]
        raw_mom_high = prices_high[-1] / prices_high[-1 - n] - 1
        raw_mom_low = prices_low[-1] / prices_low[-1 - n] - 1
        assert raw_mom_high > raw_mom_low, (
            "Test setup invalid: raw momentum should favor the high-vol asset "
            "for this comparison to be meaningful."
        )


class TestMetadata:
    def test_required_fields(self):
        for field in ("name", "author", "version", "params", "min_history", "direction", "description"):
            assert field in METADATA

    def test_direction(self):
        assert METADATA["direction"] == "higher_better"

    def test_min_history(self):
        # min_history must be window + 1: pct_change/diff(window) leaves window NaN rows,
        # so we need window+1 prices to produce the first valid output.
        assert METADATA["min_history"] == METADATA["params"]["window"] + 1


class TestParamOverride:
    def test_window_override_changes_min_history_for_output(self):
        """params={'window': N} must shrink the NaN prefix to N rows."""
        n = 30
        # Need at least n+1 prices for any valid output
        prices = [100.0 + i for i in range(n + 20)]
        df = _make_df(prices)
        result = compute(df, params={"window": n})

        # First n rows are NaN (need n+1 prices to compute n-period log return)
        assert result.iloc[:n].isna().all()
        # From row n onward, values must be finite
        tail = result.iloc[n:]
        assert tail.notna().all()
        assert np.isfinite(tail).all()
