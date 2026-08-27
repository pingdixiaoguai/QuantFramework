"""Tests for the promoted absolute-stability production state machine."""

import numpy as np
import pandas as pd

from strategy.momentum_defender_absolute_stability import (
    RAPID_REVERSAL_ENTRY_DIFFERENCE,
    RAPID_REVERSAL_EXIT_DIFFERENCE,
    RISK_OFF_CONFIRMATION_DAYS,
    RISK_ON_CONFIRMATION_DAYS,
    base_state_schedule,
    rapid_reversal_state_schedule,
    raw_raqm_score,
)


def test_raw_raqm_has_no_floor_or_winsor():
    daily = np.array([0.0010, 0.0011, 0.0009, 0.0012, 0.0008])
    curve = pd.Series(np.exp(np.r_[0.0, daily.cumsum()]))
    total = daily.sum()
    expected = (
        total / (daily.std(ddof=1) * np.sqrt(5.0))
        * abs(total) / np.abs(daily).sum()
    )
    assert np.isclose(raw_raqm_score(curve).iloc[-1], expected)


def test_downside_emergency_bypasses_risk_off_confirmation():
    index = pd.bdate_range("2024-01-01", periods=3)
    indicators = pd.DataFrame(
        {
            "wanted_risk_on": [True, True, True],
            "emergency_alert": [False, True, False],
        },
        index=index,
    )
    state = base_state_schedule(indicators)
    assert state["risk_on"].tolist() == [True, False, False]
    assert state.iloc[1]["state_reason"] == "downside_emergency_exit"


def test_dual_trend_uses_confirmation_instead_of_holding_locks():
    wanted = (
        [False] * RISK_OFF_CONFIRMATION_DAYS
        + [False]
        + [True] * RISK_ON_CONFIRMATION_DAYS
    )
    index = pd.bdate_range("2024-01-01", periods=len(wanted))
    indicators = pd.DataFrame(
        {"wanted_risk_on": wanted, "emergency_alert": False},
        index=index,
    )

    state = base_state_schedule(indicators)

    assert state["risk_on"].iloc[RISK_OFF_CONFIRMATION_DAYS - 2]
    assert not state["risk_on"].iloc[RISK_OFF_CONFIRMATION_DAYS - 1]
    assert not state["risk_on"].iloc[-2]
    assert state["risk_on"].iloc[-1]
    assert state.iloc[RISK_OFF_CONFIRMATION_DAYS - 1]["state_reason"] == (
        "dual_trend_confirmed_to_defender"
    )
    assert state.iloc[-1]["state_reason"] == "dual_trend_confirmed_to_momentum"


def test_opposite_evidence_resets_confirmation_progress():
    wanted = (
        [False] * (RISK_OFF_CONFIRMATION_DAYS - 1)
        + [True]
        + [False] * (RISK_OFF_CONFIRMATION_DAYS - 1)
    )
    index = pd.bdate_range("2024-01-01", periods=len(wanted))
    indicators = pd.DataFrame(
        {"wanted_risk_on": wanted, "emergency_alert": False},
        index=index,
    )

    state = base_state_schedule(indicators)

    assert state["risk_on"].all()
    assert state.iloc[-1]["risk_off_streak"] == RISK_OFF_CONFIRMATION_DAYS - 1


def test_rapid_reversal_bridge_has_no_minimum_holding_period():
    index = pd.bdate_range("2024-01-01", periods=4)
    metrics = pd.DataFrame(
        {
            "rapid_reversal_difference_at_open": [
                RAPID_REVERSAL_ENTRY_DIFFERENCE + 0.01,
                RAPID_REVERSAL_EXIT_DIFFERENCE + 0.05,
                RAPID_REVERSAL_EXIT_DIFFERENCE,
                RAPID_REVERSAL_ENTRY_DIFFERENCE + 0.01,
            ]
        },
        index=index,
    )

    state = rapid_reversal_state_schedule(
        pd.Series(False, index=index),
        pd.Series(False, index=index),
        metrics,
    )

    assert state["rapid_reversal_active"].tolist() == [True, True, False, True]
    assert state["state_reason"].tolist() == [
        "rapid_reversal_entry",
        "hold",
        "rapid_reversal_exit",
        "rapid_reversal_entry",
    ]
