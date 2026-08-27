from __future__ import annotations

import pandas as pd

from research.c2_monthly_underperformance import (
    apply_asymmetric_state_schedule,
    classify_daily_excess,
)


def test_daily_cause_classification_separates_slow_gate_lock_and_emergency() -> None:
    index = pd.date_range("2026-01-01", periods=4, freq="B")
    daily = pd.DataFrame(
        {
            "risk_on": [False, False, False, True],
            "slow_signal_asof_previous_close": [False, True, True, True],
            "emergency_asof_previous_close": [False, False, True, False],
            "held_days_at_open": [10, 10, 40, 5],
            "transition": ["defender_hold", "defender_hold", "defender_hold", "momentum_hold"],
            "return": [0.0] * 4,
            "momentum_exact_return": [0.01] * 4,
        },
        index=index,
    )

    result = classify_daily_excess(daily)

    assert result["cause_category"].tolist() == [
        "slow_gate_defender",
        "defender_exit_lock_delay",
        "emergency_cap_hold",
        "momentum_hold",
    ]


def test_shorter_defender_lock_exits_earlier_without_changing_entry_rule() -> None:
    index = pd.date_range("2026-01-01", periods=8, freq="B")
    slow = pd.Series([False, False, True, True, True, True, True, True], index=index)
    emergency = pd.Series(False, index=index)

    locked = apply_asymmetric_state_schedule(
        slow,
        emergency,
        index,
        momentum_min_hold_days=30,
        defender_min_hold_days=30,
    )
    faster = apply_asymmetric_state_schedule(
        slow,
        emergency,
        index,
        momentum_min_hold_days=30,
        defender_min_hold_days=1,
    )

    assert not bool(locked.iloc[-1]["risk_on"])
    assert bool(faster.iloc[2]["risk_on"])
    assert locked.iloc[0]["state_reason"] == faster.iloc[0]["state_reason"]
