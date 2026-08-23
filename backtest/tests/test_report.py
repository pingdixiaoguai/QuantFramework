"""Tests for backtest.report."""

from datetime import date
from pathlib import Path

import pandas as pd

from backtest.report import generate
from backtest.runner import BacktestResult


def _make_result() -> BacktestResult:
    dates = pd.bdate_range("2024-01-01", periods=30)
    return BacktestResult(
        daily_returns=pd.Series(0.001, index=dates),
        benchmark_returns=pd.Series(0.0005, index=dates),
        positions=pd.DataFrame({"A.SH": 0.5, "B.SH": 0.5}, index=dates),
        train_end=date(2024, 1, 20),
        config={"strategy_name": "test"},
    )


class TestBenchmarkTitle:
    def test_passes_benchmark_title_to_quantstats(self, tmp_path, monkeypatch):
        captured = {}

        def fake_html(returns, benchmark=None, output=None, **kwargs):
            captured["kwargs"] = kwargs
            Path(output).write_text("<html></html>")

        monkeypatch.setattr("quantstats.reports.html", fake_html)

        result = _make_result()
        out = tmp_path / "report.html"
        generate(result, out, benchmark_title="my_baseline")

        assert captured["kwargs"].get("benchmark_title") == "my_baseline"

    def test_omits_benchmark_title_when_none(self, tmp_path, monkeypatch):
        captured = {}

        def fake_html(returns, benchmark=None, output=None, **kwargs):
            captured["kwargs"] = kwargs
            Path(output).write_text("<html></html>")

        monkeypatch.setattr("quantstats.reports.html", fake_html)

        result = _make_result()
        out = tmp_path / "report.html"
        generate(result, out)  # no benchmark_title

        assert "benchmark_title" not in captured["kwargs"]

    def test_removes_undefined_quantstats_onload_handler(self, tmp_path, monkeypatch):
        def fake_html(returns, benchmark=None, output=None, **kwargs):
            Path(output).write_text(
                '<html><body onload="save()"><p>report</p></body></html>',
                encoding="utf-8",
            )

        monkeypatch.setattr("quantstats.reports.html", fake_html)
        out = tmp_path / "report.html"

        generate(_make_result(), out)

        document = out.read_text(encoding="utf-8")
        assert '<body onload="save()">' not in document
        assert "<body><p>report</p>" in document
