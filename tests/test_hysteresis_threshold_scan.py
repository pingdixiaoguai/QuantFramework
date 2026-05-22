"""Tests for the hysteresis threshold research helpers."""

from datetime import date

import pandas as pd
import pytest

from backtest.runner import BacktestResult
from scripts.hysteresis_threshold_scan import (
    apply_transaction_costs,
    extract_position_periods,
    forward_filled_positions,
    summarize_metrics,
    write_csv,
)


def test_forward_filled_positions_expands_execution_rows():
    dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    result = BacktestResult(
        daily_returns=pd.Series([0.01, 0.02, -0.01], index=dates),
        benchmark_returns=pd.Series(dtype=float),
        positions=pd.DataFrame(
            [{"date": dates[0], "A.SH": 1.0}, {"date": dates[2], "B.SH": 1.0}]
        ).set_index("date"),
        train_end=date(2024, 1, 3),
        config={},
    )

    positions = forward_filled_positions(result)

    assert list(positions.index) == list(dates)
    assert positions.loc[dates[1], "A.SH"] == 1.0
    assert positions.loc[dates[1], "B.SH"] == 0.0
    assert positions.loc[dates[2], "A.SH"] == 0.0
    assert positions.loc[dates[2], "B.SH"] == 1.0


def test_cost_adjusted_returns_charge_entry_and_full_switch():
    raw = pd.Series(
        [0.01, 0.02],
        index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
    )
    positions = pd.DataFrame(
        [
            {"date": pd.Timestamp("2024-01-02"), "A.SH": 1.0},
            {"date": pd.Timestamp("2024-01-03"), "B.SH": 1.0},
        ]
    ).set_index("date")

    adjusted, trades = apply_transaction_costs(raw, positions, cost_rate=0.0001)

    assert trades["traded_weight"].tolist() == [1.0, 2.0]
    assert adjusted.iloc[0] == pytest.approx(0.01 - 0.0001)
    assert adjusted.iloc[1] == pytest.approx(0.02 - 0.0002)


def test_extract_position_periods_compounds_period_pnl():
    dates = pd.to_datetime(
        ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
    )
    returns = pd.Series([0.01, 0.02, -0.01, 0.03], index=dates)
    positions = pd.DataFrame(
        [{"date": dates[0], "A.SH": 1.0}, {"date": dates[2], "B.SH": 1.0}]
    ).set_index("date")

    periods = extract_position_periods(positions, returns)

    assert periods["asset"].tolist() == ["A.SH", "B.SH"]
    assert periods["holding_days"].tolist() == [2, 2]
    assert periods.loc[0, "pnl"] == pytest.approx(1.01 * 1.02 - 1)
    assert periods.loc[1, "pnl"] == pytest.approx(0.99 * 1.03 - 1)


def test_summarize_metrics_reports_turnover_and_switch_count():
    dates = pd.bdate_range("2024-01-02", periods=252)
    returns = pd.Series([0.001] * len(dates), index=dates)
    trades = pd.DataFrame({"traded_weight": [1.0, 2.0]})
    periods = pd.DataFrame({"holding_days": [126, 126]})

    metrics = summarize_metrics(returns, trades, periods)

    assert metrics["annualized_turnover"] == pytest.approx(3.0)
    assert metrics["average_holding_days"] == pytest.approx(126.0)
    assert metrics["switch_count"] == 1
    assert metrics["annualized_return"] > 0
    assert metrics["max_drawdown"] == pytest.approx(0.0)


def test_write_csv_creates_parent_directory(tmp_path):
    path = tmp_path / "nested" / "rows.csv"

    write_csv(pd.DataFrame({"tau": [0.0]}), path)

    assert path.exists()
    assert "tau" in path.read_text(encoding="utf-8")
