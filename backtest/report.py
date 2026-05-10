"""Backtest report generation via quantstats."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from backtest.runner import BacktestResult


def generate(
    result: BacktestResult,
    output_path: Path,
    benchmark_title: str | None = None,
) -> Path:
    """Generate an HTML report using quantstats.

    When `benchmark_title` is set, it labels the benchmark series in the rendered
    HTML; when None, quantstats falls back to its default ("Benchmark" or
    benchmark.name).

    Returns the output path on success.
    """
    import quantstats as qs

    output_path.parent.mkdir(parents=True, exist_ok=True)

    extra_kwargs = {}
    if benchmark_title is not None:
        extra_kwargs["benchmark_title"] = benchmark_title

    qs.reports.html(
        result.daily_returns,
        benchmark=result.benchmark_returns,
        output=str(output_path),
        **extra_kwargs,
    )
    return output_path
