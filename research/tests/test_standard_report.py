from pathlib import Path

import pandas as pd
import pytest

from research.standard_report import _annual_returns, _repair_eoy_chart


def test_annual_returns_compounds_each_calendar_year() -> None:
    index = pd.to_datetime(["2025-12-30", "2025-12-31", "2026-01-02"])
    strategy = pd.Series([0.10, -0.05, 0.20], index=index)
    benchmark = pd.Series([0.0, 0.10, -0.10], index=index)

    annual = _annual_returns(strategy, benchmark)

    assert annual.at[2025, "strategy"] == (1.10 * 0.95) - 1.0
    assert annual.at[2026, "strategy"] == pytest.approx(0.20)
    assert annual.at[2025, "benchmark"] == pytest.approx(0.10)


def test_eoy_repair_changes_only_target_svg(tmp_path: Path) -> None:
    index = pd.to_datetime(["2025-12-31", "2026-01-02"])
    strategy = pd.Series([0.10, 0.20], index=index)
    benchmark = pd.Series([0.05, -0.10], index=index)
    report = tmp_path / "report.html"
    report.write_text(
        "<html><body><svg><text>EOY Returns vs Benchmark</text></svg>"
        "<svg id='other'><text>keep me</text></svg></body></html>",
        encoding="utf-8",
    )

    _repair_eoy_chart(report, strategy, benchmark)

    document = report.read_text(encoding="utf-8")
    assert "keep me" in document
    assert "2025" in document and "2026" in document
    assert document.count("EOY Returns") == 1
