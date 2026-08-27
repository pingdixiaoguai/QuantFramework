from __future__ import annotations

import pandas as pd
import pytest

from research.momentum_defender_switching import (
    SwitchParams,
    _execute_target,
    _state_schedule,
)


def test_close_signal_only_changes_state_at_next_open() -> None:
    dates = pd.date_range("2026-01-01", periods=4, freq="D")
    close = pd.Series([100.0, 100.0, 110.0, 110.0], index=dates)

    state = _state_schedule(
        close,
        SwitchParams(lookback=1, risk_on_threshold=0.05, min_hold_days=1),
    )

    # The +10% close move occurs on day 3.  It cannot turn risk-on until the
    # following day's open.
    assert state.tolist() == [True, True, False, True]


def test_full_sleeve_switch_sells_then_buys_at_open() -> None:
    cash, shares, initial = _execute_target(
        1.0,
        {},
        {"510300.SH": 1.0},
        {"510300.SH": 1.0, "512890.SH": 1.0, "511260.SH": 1.0},
    )
    cash, shares, switched = _execute_target(
        cash,
        shares,
        {"512890.SH": 0.6, "511260.SH": 0.4},
        {"510300.SH": 1.0, "512890.SH": 1.0, "511260.SH": 1.0},
    )

    assert [item["side"] for item in switched] == ["sell", "buy", "buy"]
    assert shares.get("510300.SH", 0.0) == pytest.approx(0.0)
    assert shares["512890.SH"] > 0
    assert shares["511260.SH"] > 0
    assert sum(float(item["turnover"]) for item in switched) == pytest.approx(
        1.9998, abs=5e-4
    )
    assert sum(float(item["cost"]) for item in initial + switched) > 0
