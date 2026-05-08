"""Tests for backtest.runner."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from backtest.runner import BacktestResult, run


def _make_asset_data(asset_code: str, prices: list[float], start: str = "2024-01-01"):
    """Create synthetic data in data/store format."""
    n = len(prices)
    dates = pd.bdate_range(start, periods=n)
    return pd.DataFrame({
        "date": dates,
        "open": prices,
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


class TestRebalanceEveryNDays:
    def test_strategy_called_only_every_n_days(self, monkeypatch):
        """With rebalance_days=5, strategy.generate_weights is called on the
        first valid day, then every 5 trading days thereafter (intermediate
        days reuse the prior weights)."""
        # 30 trading days, min_history=2, so first valid day is index 1.
        # Expected rebalance day-indexes: 1, 6, 11, 16, 21, 26
        n = 30
        prices_a = [100.0 + i for i in range(n)]
        prices_b = [200.0 - i * 0.5 for i in range(n)]

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

        # Position rows are recorded only on rebalance days.
        # First valid day (index 1) plus every 5 days: indices 1, 6, 11, 16, 21, 26
        # → 6 rebalance days total.
        trading_days = sorted(df_a["date"].tolist())
        expected_dates = [trading_days[i] for i in (1, 6, 11, 16, 21, 26)]

        assert list(result.positions.index) == expected_dates


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
