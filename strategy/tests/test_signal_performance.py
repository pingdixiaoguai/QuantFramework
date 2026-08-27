from __future__ import annotations

import pandas as pd
import pytest

from strategy.signal_performance import (
    _entered_sleeve_return,
    _last_target_change,
    _period_performance,
)


def test_last_target_change_finds_start_of_current_allocation() -> None:
    index = pd.date_range("2026-08-03", periods=5, freq="B")
    targets = pd.DataFrame(
        {
            "A": [1.0, 1.0, 0.0, 0.0, 0.0],
            "B": [0.0, 0.0, 1.0, 1.0, 1.0],
            "target_cash_weight": 0.0,
        },
        index=index,
    )

    assert _last_target_change(targets) == index[2]


def test_entered_sleeve_return_uses_entry_leg_only_on_first_day() -> None:
    index = pd.date_range("2026-08-03", periods=3, freq="B")
    interface = pd.DataFrame(
        {
            "enter_open_to_close_net_return": [0.10, 9.0, 9.0],
            "daily_net_return_if_held": [0.50, 0.02, -0.01],
        },
        index=index,
    )

    result = _entered_sleeve_return(interface, index[0], index[-1])

    assert result == pytest.approx(1.10 * 1.02 * 0.99 - 1.0)


def test_period_performance_uses_calendar_month_quarter_and_year() -> None:
    index = pd.to_datetime(
        ["2026-01-02", "2026-07-01", "2026-08-03", "2026-08-25"]
    )
    returns = pd.Series([0.10, 0.20, 0.03, 0.04], index=index)

    measured = _period_performance(returns, pd.Timestamp("2026-08-25"))

    assert measured.month == pytest.approx(1.03 * 1.04 - 1.0)
    assert measured.quarter == pytest.approx(1.20 * 1.03 * 1.04 - 1.0)
    assert measured.year == pytest.approx(1.10 * 1.20 * 1.03 * 1.04 - 1.0)
