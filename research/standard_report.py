"""Project-standard QuantStats report adapter for research strategies."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from backtest.report import generate as generate_backtest_report
from backtest.runner import BacktestResult


def generate_standard_report(
    returns: pd.Series,
    benchmark: pd.Series,
    benchmark_name: str,
    output_path: Path,
    config: dict[str, object],
) -> Path:
    aligned = pd.concat(
        [returns.rename("strategy"), benchmark.rename("benchmark")], axis=1
    ).dropna()
    if len(aligned) != len(returns):
        raise ValueError(
            f"standard report lost observations after aligning {benchmark_name}: "
            f"{len(aligned)} != {len(returns)}"
        )
    result = BacktestResult(
        daily_returns=aligned["strategy"],
        benchmark_returns=aligned["benchmark"],
        positions=pd.DataFrame(index=aligned.index),
        train_end=date(2024, 12, 31),
        config=config,
        baseline_strategy_name=benchmark_name,
    )
    return generate_backtest_report(
        result,
        output_path,
        benchmark_title=benchmark_name,
    )
