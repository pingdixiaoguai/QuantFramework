"""Backtest report generation via quantstats."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from backtest.runner import BacktestResult


def generate(result: BacktestResult, output_path: Path) -> Path:
    """Generate an HTML report using quantstats.

    Returns the output path on success.
    """
    import quantstats as qs

    output_path.parent.mkdir(parents=True, exist_ok=True)
    qs.reports.html(
        result.daily_returns,
        benchmark=result.benchmark_returns,
        output=str(output_path),
    )
    return output_path
