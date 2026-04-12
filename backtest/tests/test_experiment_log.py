"""Tests for backtest.experiment_log."""

from datetime import date

import pandas as pd
import yaml

from backtest.experiment_log import save
from backtest.runner import BacktestResult


def _make_result() -> BacktestResult:
    dates = pd.bdate_range("2024-01-01", periods=100)
    returns = pd.Series(0.001, index=dates)
    benchmark = pd.Series(0.0005, index=dates)
    positions = pd.DataFrame({"A.SH": 0.5, "B.SH": 0.5}, index=dates)
    return BacktestResult(
        daily_returns=returns,
        benchmark_returns=benchmark,
        positions=positions,
        train_end=date(2024, 4, 1),
        config={
            "strategy_name": "test",
            "asset_pool": ["A.SH", "B.SH"],
            "start": date(2024, 1, 1),
            "end": date(2024, 12, 31),
            "factors": [{"name": "momentum", "weight": 1.0, "params": {"window": 20}}],
            "train_ratio": 0.7,
        },
    )


class TestSaveAndLoad:
    def test_saves_valid_yaml(self, tmp_path):
        result = _make_result()
        path = save(result, output_dir=tmp_path)

        assert path.exists()
        with open(path) as f:
            log = yaml.safe_load(f)

        exp = log["experiment"]
        assert "id" in exp
        assert "timestamp" in exp
        assert "strategy" in exp
        assert "results" in exp
        assert "train" in exp["results"]
        assert "test" in exp["results"]
        assert "total_return" in exp["results"]["train"]
        assert "sharpe" in exp["results"]["train"]
        assert "max_drawdown" in exp["results"]["train"]


class TestIdAutoIncrement:
    def test_sequential_ids(self, tmp_path):
        result = _make_result()
        path1 = save(result, output_dir=tmp_path)
        path2 = save(result, output_dir=tmp_path)

        # Extract sequence numbers
        seq1 = int(path1.stem.split("-")[-1])
        seq2 = int(path2.stem.split("-")[-1])
        assert seq2 == seq1 + 1
