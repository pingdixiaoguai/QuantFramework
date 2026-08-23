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
    # QuantStats 0.0.81 emits ``<body onload="save()">`` without defining a
    # global ``save`` function.  The report renders, but every open produces a
    # browser console error.  Remove only that dead handler and leave the
    # generated tables and inline SVGs byte-for-byte otherwise unchanged.
    document = output_path.read_text(encoding="utf-8")
    document = document.replace('<body onload="save()">', "<body>")
    output_path.write_text(document, encoding="utf-8")
    return output_path
