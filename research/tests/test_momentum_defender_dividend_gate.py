from __future__ import annotations

import pandas as pd
import pytest

from research.momentum_defender_dividend_gate import (
    CONJUNCTION,
    ENTRY_ONLY,
    DividendGateParams,
    apply_dividend_gate_schedule,
    defender_primary_target_at_open,
    trailing_return_at_open,
)


def test_trailing_return_is_effective_only_at_the_following_open() -> None:
    dates = pd.date_range("2026-01-01", periods=4, freq="B")
    close = pd.Series([100.0, 110.0, 121.0, 108.9], index=dates)

    result = trailing_return_at_open(close, dates, lookback=1)

    assert pd.isna(result.iloc[0])
    assert pd.isna(result.iloc[1])
    assert result.iloc[2] == pytest.approx(0.10)
    assert result.iloc[3] == pytest.approx(0.10)


def test_defender_primary_target_sums_all_dividend_etfs() -> None:
    index = pd.DatetimeIndex(["2026-08-21"])
    frame = pd.DataFrame(index=index)
    for code in ("512890", "159545", "513530", "515080", "510880", "563020"):
        frame[f"target_weight_{code}"] = 0.0
    frame["target_weight_510880"] = 0.6
    frame["target_weight_515080"] = 0.2

    primary = defender_primary_target_at_open(frame)

    assert primary.iloc[0] == pytest.approx(0.8)


def test_entry_only_gate_does_not_exit_when_primary_target_falls() -> None:
    calendar = pd.date_range("2026-01-01", periods=5, freq="B")
    slow = pd.Series([-0.05] * 5, index=calendar)
    primary = pd.Series([0.8, 0.8, 0.2, 0.2, 0.2], index=calendar)
    params = DividendGateParams(min_hold_days=1, exit_mode=ENTRY_ONLY)

    state, condition = apply_dividend_gate_schedule(
        slow, primary, calendar, params
    )

    assert condition.tolist() == [True, True, False, False, False]
    assert state["risk_on"].tolist() == [False, False, False, False, False]
    entries = state["state_changed"] & ~state["risk_on"]
    assert condition.loc[entries].all()


def test_conjunction_gate_exits_when_primary_target_falls() -> None:
    calendar = pd.date_range("2026-01-01", periods=5, freq="B")
    slow = pd.Series([-0.05] * 5, index=calendar)
    primary = pd.Series([0.8, 0.8, 0.2, 0.2, 0.2], index=calendar)
    params = DividendGateParams(min_hold_days=1, exit_mode=CONJUNCTION)

    state, _ = apply_dividend_gate_schedule(slow, primary, calendar, params)

    assert state["risk_on"].tolist() == [False, False, True, True, True]
    assert state.iloc[2]["state_reason"] == "joint_gate_exit_defender"
