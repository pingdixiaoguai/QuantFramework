"""Tests for run_backtest.py orchestration helpers."""

from datetime import date

import pandas as pd
import pytest

from backtest.runner import BacktestResult
from run_backtest import _apply_baseline, _load_config_from_yaml


def _make_result(start: str, periods: int) -> BacktestResult:
    dates = pd.bdate_range(start, periods=periods)
    return BacktestResult(
        daily_returns=pd.Series([0.001] * periods, index=dates),
        benchmark_returns=pd.Series([0.0005] * periods, index=dates),
        positions=pd.DataFrame({"A.SH": [0.5] * periods}, index=dates),
        train_end=dates[periods // 2].date(),
        config={"strategy_name": "foo"},
    )


class TestApplyBaseline:
    def test_replaces_benchmark_with_aligned_baseline_returns(self):
        # foo: 2024-01-01 + 100 bdays; bar: 2024-01-15 + 80 bdays — overlap is the
        # later 86-ish days of foo. We just check the returned benchmark equals
        # bar restricted to foo's index, not foo's own benchmark.
        foo = _make_result("2024-01-01", 100)
        bar = _make_result("2024-01-15", 80)
        # Make bar's daily_returns identifiable
        bar.daily_returns.iloc[:] = 0.002

        out = _apply_baseline(foo, bar, baseline_strategy_name="bar")

        # daily_returns untouched (foo's full series)
        assert len(out.daily_returns) == 100
        assert (out.daily_returns == 0.001).all()

        # benchmark_returns now reflects bar, restricted to foo's index
        assert (out.benchmark_returns == 0.002).all()
        # length = size of intersection between foo and bar indices
        expected_len = len(foo.daily_returns.index.intersection(bar.daily_returns.index))
        assert len(out.benchmark_returns) == expected_len

        # baseline_strategy_name set
        assert out.baseline_strategy_name == "bar"

    def test_raises_on_empty_overlap(self):
        foo = _make_result("2024-01-01", 30)
        bar = _make_result("2025-06-01", 30)  # no overlap with foo

        with pytest.raises(RuntimeError, match="No overlapping trading days"):
            _apply_baseline(foo, bar, baseline_strategy_name="bar")

    def test_does_not_mutate_original_foo_result(self):
        foo = _make_result("2024-01-01", 50)
        bar = _make_result("2024-01-01", 50)
        original_bench = foo.benchmark_returns.copy()

        _apply_baseline(foo, bar, baseline_strategy_name="bar")

        # foo's benchmark_returns must be unchanged (we used dataclasses.replace, not in-place)
        pd.testing.assert_series_equal(foo.benchmark_returns, original_bench)
        assert foo.baseline_strategy_name is None


class TestLoadConfigFromYaml:
    def test_dynamic_end_today_uses_current_date(self, tmp_path, monkeypatch):
        class FixedDate(date):
            @classmethod
            def today(cls):
                return cls(2026, 5, 11)

        monkeypatch.setattr("run_backtest.date", FixedDate)
        path = tmp_path / "cfg.yaml"
        path.write_text(
            "strategy_name: test\n"
            "asset_pool: []\n"
            "start: '2016-01-01'\n"
            "end: 'today'\n"
            "factors: []\n",
            encoding="utf-8",
        )

        config = _load_config_from_yaml(path)

        assert config["start"] == date(2016, 1, 1)
        assert config["end"] == date(2026, 5, 11)

    def test_missing_end_defaults_to_current_date(self, tmp_path, monkeypatch):
        class FixedDate(date):
            @classmethod
            def today(cls):
                return cls(2026, 5, 11)

        monkeypatch.setattr("run_backtest.date", FixedDate)
        path = tmp_path / "cfg.yaml"
        path.write_text(
            "strategy_name: test\n"
            "asset_pool: []\n"
            "start: '2016-01-01'\n"
            "factors: []\n",
            encoding="utf-8",
        )

        config = _load_config_from_yaml(path)

        assert config["end"] == date(2026, 5, 11)

    def test_explicit_end_stays_reproducible(self, tmp_path, monkeypatch):
        class FixedDate(date):
            @classmethod
            def today(cls):
                return cls(2026, 5, 11)

        monkeypatch.setattr("run_backtest.date", FixedDate)
        path = tmp_path / "cfg.yaml"
        path.write_text(
            "strategy_name: test\n"
            "asset_pool: []\n"
            "start: '2016-01-01'\n"
            "end: '2026-04-13'\n"
            "factors: []\n",
            encoding="utf-8",
        )

        config = _load_config_from_yaml(path)

        assert config["end"] == date(2026, 4, 13)
