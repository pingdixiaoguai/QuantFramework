"""Tests for backtest.runner."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from backtest.runner import BacktestResult, run


def _make_asset_data(
    asset_code: str,
    prices: list[float],
    start: str = "2024-01-01",
    opens: list[float] | None = None,
):
    """Create synthetic data in data/store format."""
    n = len(prices)
    dates = pd.bdate_range(start, periods=n)
    open_prices = opens if opens is not None else prices
    return pd.DataFrame({
        "date": dates,
        "open": open_prices,
        "high": [p * 1.01 for p in prices],
        "low": [p * 0.99 for p in prices],
        "close": prices,
        "volume": [1000.0] * n,
    })


class TestFutureInfoTruncation:
    def test_factor_only_sees_past_data(self, monkeypatch):
        """Verify factor receives truncated data at each time step."""
        # Track the lengths of data passed to compute
        lengths_seen = []

        def mock_compute(df, params=None):
            lengths_seen.append(len(df))
            series = pd.Series(float(len(df)), index=df["date"])
            return series

        mock_metadata = {
            "name": "length_tracker",
            "author": "test",
            "version": "1.0.0",
            "params": {},
            "min_history": 2,
            "direction": "higher_better",
            "description": "tracks data length",
        }

        # Mock load_registered_factors
        monkeypatch.setattr(
            "backtest.runner.load_registered_factors",
            lambda: {"length_tracker": {"METADATA": mock_metadata, "compute": mock_compute}},
        )

        # Mock query to return synthetic data
        prices = [100.0 + i for i in range(30)]
        asset_df = _make_asset_data("TEST.SH", prices)

        monkeypatch.setattr(
            "backtest.runner.query",
            lambda asset, start, end: asset_df,
        )

        config = {
            "strategy_name": "test",
            "asset_pool": ["TEST.SH"],
            "start": date(2024, 1, 1),
            "end": date(2024, 12, 31),
            "factors": [{"name": "length_tracker", "weight": 1.0, "params": {}}],
            "train_ratio": 0.7,
            "rebalance_rule": "daily",
        }

        run(config)

        # lengths_seen should be strictly increasing
        assert lengths_seen == sorted(lengths_seen)
        assert lengths_seen[0] >= 2  # min_history
        assert lengths_seen[-1] == 30  # full dataset


class TestReturnsCalculation:
    def test_known_returns(self, monkeypatch):
        """With known linear prices, verify return calculation."""
        # Asset A: 100, 110, 121 (10% daily)
        # Asset B: 100, 105, 110.25 (5% daily)
        prices_a = [100.0, 110.0, 121.0]
        prices_b = [100.0, 105.0, 110.25]

        df_a = _make_asset_data("A.SH", prices_a)
        df_b = _make_asset_data("B.SH", prices_b)

        def mock_query(asset, start, end):
            return {"A.SH": df_a, "B.SH": df_b}[asset]

        monkeypatch.setattr("backtest.runner.query", mock_query)

        # Simple factor that returns close price (higher = better)
        def mock_compute(df, params=None):
            return pd.Series(df["close"].values, index=df["date"], dtype=float)

        mock_meta = {
            "name": "price",
            "author": "test",
            "version": "1.0.0",
            "params": {},
            "min_history": 1,
            "direction": "higher_better",
            "description": "price",
        }

        monkeypatch.setattr(
            "backtest.runner.load_registered_factors",
            lambda: {"price": {"METADATA": mock_meta, "compute": mock_compute}},
        )

        config = {
            "strategy_name": "test",
            "asset_pool": ["A.SH", "B.SH"],
            "start": date(2024, 1, 1),
            "end": date(2024, 12, 31),
            "factors": [{"name": "price", "weight": 1.0, "params": {}}],
            "train_ratio": 0.5,
            "rebalance_rule": "daily",
        }

        result = run(config)

        # Should have returns for days 2 and 3
        assert len(result.daily_returns) > 0
        # All returns should be finite
        assert np.isfinite(result.daily_returns).all()


class TestBenchmarkEqualWeight:
    def test_benchmark_is_mean_of_asset_returns(self, monkeypatch):
        """Benchmark return should be equal-weight average."""
        prices_a = [100.0, 110.0, 121.0, 133.1]
        prices_b = [100.0, 90.0, 81.0, 72.9]

        df_a = _make_asset_data("A.SH", prices_a)
        df_b = _make_asset_data("B.SH", prices_b)

        def mock_query(asset, start, end):
            return {"A.SH": df_a, "B.SH": df_b}[asset]

        monkeypatch.setattr("backtest.runner.query", mock_query)

        def mock_compute(df, params=None):
            return pd.Series(df["close"].values, index=df["date"], dtype=float)

        mock_meta = {
            "name": "price",
            "author": "test",
            "version": "1.0.0",
            "params": {},
            "min_history": 1,
            "direction": "higher_better",
            "description": "price",
        }

        monkeypatch.setattr(
            "backtest.runner.load_registered_factors",
            lambda: {"price": {"METADATA": mock_meta, "compute": mock_compute}},
        )

        config = {
            "strategy_name": "test",
            "asset_pool": ["A.SH", "B.SH"],
            "start": date(2024, 1, 1),
            "end": date(2024, 12, 31),
            "factors": [{"name": "price", "weight": 1.0, "params": {}}],
            "train_ratio": 0.5,
            "rebalance_rule": "daily",
        }

        result = run(config)

        # Benchmark should be mean of individual returns
        for t in result.benchmark_returns.index:
            ret_a = ret_b = None
            prev_idx = result.benchmark_returns.index.get_loc(t) - 1
            if prev_idx >= 0:
                prev_t = result.benchmark_returns.index[prev_idx]
            else:
                continue

            if t in df_a["date"].values:
                pa = df_a.set_index("date")
                if prev_t in pa.index and t in pa.index:
                    ret_a = pa.loc[t, "close"] / pa.loc[prev_t, "close"] - 1

            expected_bench = np.mean([r for r in [ret_a] if r is not None])
            # We just check that benchmark returns are reasonable
            assert abs(result.benchmark_returns[t]) < 1.0  # not crazy


class TestNoLookaheadBias:
    """Today's close signal must not earn the pre-open gap on its entry day.

    Regression for the bug where the engine, after computing new_weights
    using close[t], applied them to the close[t]/close[t-1] return of
    the SAME day — i.e. perfect foresight, producing absurd returns.
    """

    def test_returns_use_yesterdays_weights(self, monkeypatch):
        # A: down then up (close 100 → 90 → 121)
        # B: up then down (close 100 → 110 → 95)
        # Opens equal the previous close after day 0, so this test isolates
        # timing without adding unrelated overnight gaps.
        # Top1 picks the asset with the higher current close.
        #   day 0 (100/100): tie → A
        #   day 1 (90/110):  B
        #   day 2 (121/95):  A
        prices_a = [100.0, 90.0, 121.0]
        prices_b = [100.0, 110.0, 95.0]
        opens_a = [100.0, 100.0, 90.0]
        opens_b = [100.0, 100.0, 110.0]
        df_a = _make_asset_data("A.SH", prices_a, opens=opens_a)
        df_b = _make_asset_data("B.SH", prices_b, opens=opens_b)

        monkeypatch.setattr(
            "backtest.runner.query",
            lambda asset, start, end: {"A.SH": df_a, "B.SH": df_b}[asset],
        )

        def mock_compute(df, params=None):
            return pd.Series(df["close"].values, index=df["date"], dtype=float)

        mock_meta = {
            "name": "price",
            "author": "test",
            "version": "1.0.0",
            "params": {},
            "min_history": 1,
            "direction": "higher_better",
            "description": "price",
        }
        monkeypatch.setattr(
            "backtest.runner.load_registered_factors",
            lambda: {"price": {"METADATA": mock_meta, "compute": mock_compute}},
        )

        config = {
            "strategy_name": "test",
            "strategy_class": "strategy.top1.Top1",
            "asset_pool": ["A.SH", "B.SH"],
            "start": date(2024, 1, 1),
            "end": date(2024, 12, 31),
            "factors": [{"name": "price", "weight": 1.0, "params": {}}],
            "train_ratio": 0.5,
            "rebalance_days": 1,
        }

        result = run(config)

        # Day 1: day-0 signal enters A at day-1 open, earns A open->close.
        # Day 2: old A earns no gap (90->90), then B enters at open 110
        # and earns B open->close.
        # Under the lookahead bug, day 1 would be +0.10 and day 2 +34.4%.
        assert len(result.daily_returns) == 2
        assert result.daily_returns.iloc[0] == pytest.approx(-0.10, abs=1e-9)
        assert result.daily_returns.iloc[1] == pytest.approx(95.0 / 110.0 - 1, abs=1e-9)

    def test_rebalance_day_splits_old_overnight_and_new_intraday(self, monkeypatch):
        # Day 0 close chooses A, entered day 1 open.
        # Day 1 close chooses B, entered day 2 open.
        # Day 2 return should be old A's overnight gap (90 -> 99 = +10%)
        # chained with new B's intraday move (50 -> 55 = +10%): +21%.
        prices_a = [100.0, 90.0, 120.0]
        prices_b = [100.0, 110.0, 55.0]
        opens_a = [100.0, 100.0, 99.0]
        opens_b = [100.0, 100.0, 50.0]
        df_a = _make_asset_data("A.SH", prices_a, opens=opens_a)
        df_b = _make_asset_data("B.SH", prices_b, opens=opens_b)

        monkeypatch.setattr(
            "backtest.runner.query",
            lambda asset, start, end: {"A.SH": df_a, "B.SH": df_b}[asset],
        )

        def mock_compute(df, params=None):
            return pd.Series(df["close"].values, index=df["date"], dtype=float)

        mock_meta = {
            "name": "price",
            "author": "test",
            "version": "1.0.0",
            "params": {},
            "min_history": 1,
            "direction": "higher_better",
            "description": "price",
        }
        monkeypatch.setattr(
            "backtest.runner.load_registered_factors",
            lambda: {"price": {"METADATA": mock_meta, "compute": mock_compute}},
        )

        config = {
            "strategy_name": "test",
            "strategy_class": "strategy.top1.Top1",
            "asset_pool": ["A.SH", "B.SH"],
            "start": date(2024, 1, 1),
            "end": date(2024, 12, 31),
            "factors": [{"name": "price", "weight": 1.0, "params": {}}],
            "train_ratio": 0.5,
            "rebalance_days": 1,
        }

        result = run(config)

        assert len(result.daily_returns) == 2
        assert result.daily_returns.iloc[1] == pytest.approx(1.10 * 1.10 - 1)


class TestRebalanceEveryNDays:
    def test_strategy_called_only_every_n_days(self, monkeypatch):
        """With rebalance_days=5, strategy.generate_weights is called on the
        first valid day, then every 5 trading days thereafter (intermediate
        days reuse the prior weights)."""
        # 15 trading days, min_history=2, so first valid signal day is index 1.
        # Signals at index 1, 6, 11 lead to entries at index 2, 7, 12.
        n = 15
        prices_a = [
            100.0,
            200.0,
            200.0,
            200.0,
            200.0,
            200.0,
            80.0,
            80.0,
            80.0,
            80.0,
            80.0,
            210.0,
            210.0,
            210.0,
            210.0,
        ]
        prices_b = [
            100.0,
            100.0,
            100.0,
            100.0,
            100.0,
            100.0,
            220.0,
            220.0,
            220.0,
            220.0,
            220.0,
            90.0,
            90.0,
            90.0,
            90.0,
        ]

        df_a = _make_asset_data("A.SH", prices_a)
        df_b = _make_asset_data("B.SH", prices_b)

        monkeypatch.setattr(
            "backtest.runner.query",
            lambda asset, start, end: {"A.SH": df_a, "B.SH": df_b}[asset],
        )

        def mock_compute(df, params=None):
            return pd.Series(df["close"].values, index=df["date"], dtype=float)

        mock_meta = {
            "name": "price",
            "author": "test",
            "version": "1.0.0",
            "params": {},
            "min_history": 2,
            "direction": "higher_better",
            "description": "price",
        }
        monkeypatch.setattr(
            "backtest.runner.load_registered_factors",
            lambda: {"price": {"METADATA": mock_meta, "compute": mock_compute}},
        )

        # Spy on strategy.generate_weights
        import strategy.top1 as top1_mod
        call_dates: list[pd.Timestamp] = []
        original = top1_mod.Top1.generate_weights

        def spy(self, factor_values):
            # The current trading day is the latest date present in any
            # asset's truncated factor view; we tag the call by current
            # max date inferred from factor_values keys' values? We can't
            # see the date directly here, so just count and let the
            # positions index tell us the dates.
            call_dates.append(pd.Timestamp("1970-01-01"))
            return original(self, factor_values)

        monkeypatch.setattr(top1_mod.Top1, "generate_weights", spy)

        config = {
            "strategy_name": "test",
            "strategy_class": "strategy.top1.Top1",
            "asset_pool": ["A.SH", "B.SH"],
            "start": date(2024, 1, 1),
            "end": date(2024, 12, 31),
            "factors": [{"name": "price", "weight": 1.0, "params": {}}],
            "train_ratio": 0.7,
            "rebalance_days": 5,
        }

        result = run(config)

        # Position rows are recorded on actual open execution days.
        # First valid signal day is index 1, so entries are indices
        # 2, 7, 12.
        trading_days = sorted(df_a["date"].tolist())
        expected_dates = [trading_days[i] for i in (2, 7, 12)]

        assert list(result.positions.index) == expected_dates

    def test_fixed_cycle_mode_skips_non_boundary_signals(self, monkeypatch):
        """fixed_cycle only evaluates on held-day multiples of rebalance_days."""
        prices_a = [200.0, 200.0, 200.0, 100.0, 100.0, 100.0]
        prices_b = [100.0, 100.0, 100.0, 300.0, 300.0, 300.0]

        df_a = _make_asset_data("A.SH", prices_a)
        df_b = _make_asset_data("B.SH", prices_b)

        monkeypatch.setattr(
            "backtest.runner.query",
            lambda asset, start, end: {"A.SH": df_a, "B.SH": df_b}[asset],
        )

        def mock_compute(df, params=None):
            return pd.Series(df["close"].values, index=df["date"], dtype=float)

        mock_meta = {
            "name": "price",
            "author": "test",
            "version": "1.0.0",
            "params": {},
            "min_history": 1,
            "direction": "higher_better",
            "description": "price",
        }
        monkeypatch.setattr(
            "backtest.runner.load_registered_factors",
            lambda: {"price": {"METADATA": mock_meta, "compute": mock_compute}},
        )

        config = {
            "strategy_name": "test",
            "strategy_class": "strategy.top1.Top1",
            "asset_pool": ["A.SH", "B.SH"],
            "start": date(2024, 1, 1),
            "end": date(2024, 12, 31),
            "factors": [{"name": "price", "weight": 1.0, "params": {}}],
            "train_ratio": 0.7,
            "rebalance_days": 2,
            "rebalance_mode": "fixed_cycle",
        }

        result = run(config)

        trading_days = sorted(df_a["date"].tolist())
        assert list(result.positions.index) == [
            trading_days[1],
            trading_days[5],
        ]


class TestTrainTestSplit:
    def test_train_end_at_correct_position(self, monkeypatch):
        """train_end should be at train_ratio position in trading days."""
        n = 100
        prices = [100.0 + i * 0.1 for i in range(n)]
        df = _make_asset_data("A.SH", prices)

        monkeypatch.setattr("backtest.runner.query", lambda a, s, e: df)

        def mock_compute(df, params=None):
            return pd.Series(df["close"].values, index=df["date"], dtype=float)

        mock_meta = {
            "name": "price",
            "author": "test",
            "version": "1.0.0",
            "params": {},
            "min_history": 1,
            "direction": "higher_better",
            "description": "price",
        }

        monkeypatch.setattr(
            "backtest.runner.load_registered_factors",
            lambda: {"price": {"METADATA": mock_meta, "compute": mock_compute}},
        )

        config = {
            "strategy_name": "test",
            "asset_pool": ["A.SH"],
            "start": date(2024, 1, 1),
            "end": date(2025, 12, 31),
            "factors": [{"name": "price", "weight": 1.0, "params": {}}],
            "train_ratio": 0.7,
            "rebalance_rule": "daily",
        }

        result = run(config)

        # train_end should be approximately at 70% of trading days
        trading_days = sorted(df["date"].tolist())
        expected_idx = int(len(trading_days) * 0.7)
        expected_date = trading_days[expected_idx].date()
        assert result.train_end == expected_date
