"""Backtest report generation via quantstats."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

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
    import quantstats._plotting.core as qs_plotting_core

    output_path.parent.mkdir(parents=True, exist_ok=True)

    extra_kwargs = {}
    if benchmark_title is not None:
        extra_kwargs["benchmark_title"] = benchmark_title

    strategy_returns = result.daily_returns.copy()
    strategy_returns.name = "Strategy"
    benchmark_returns = result.benchmark_returns.copy()
    benchmark_returns.name = benchmark_title or "Benchmark"

    original_plot_returns_bars = qs_plotting_core.plot_returns_bars

    def _plot_returns_bars_aligned(returns, benchmark=None, *args, **kwargs):
        """Keep pandas 2.x annual bars on the same 0..N axis as year ticks."""
        if kwargs.get("resample") != "YE":
            return original_plot_returns_bars(
                returns,
                benchmark,
                *args,
                **kwargs,
            )

        def _annualize(value):
            annual = qs_plotting_core.safe_resample(
                value,
                "YE",
                qs_plotting_core._get_stats().comp,
            )
            annual = qs_plotting_core.safe_resample(annual, "YE", "last")
            annual.index = pd.Index(annual.index.year.astype(str))
            return annual

        annual_returns = _annualize(returns)
        annual_benchmark = (
            _annualize(benchmark) if benchmark is not None else None
        )
        kwargs["resample"] = None
        kwargs["subtitle"] = False
        return original_plot_returns_bars(
            annual_returns,
            annual_benchmark,
            *args,
            **kwargs,
        )

    qs_plotting_core.plot_returns_bars = _plot_returns_bars_aligned
    try:
        qs.reports.html(
            strategy_returns,
            benchmark=benchmark_returns,
            output=str(output_path),
            **extra_kwargs,
        )
    finally:
        qs_plotting_core.plot_returns_bars = original_plot_returns_bars
    return output_path
