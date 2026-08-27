import pandas as pd

from research.audit_defender_exit_mechanism_2019 import (
    ExitPolicy,
    exit_state_schedule,
)
from research.audit_defender_signed_recovery_2019 import (
    signed_recovery_state_schedule,
)
from research.momentum_defender_downside_raqm import (
    downside_raqm_state_schedule,
)
from strategy.momentum_defender_w40_loss import formal_spec


def test_fixed_30_day_policy_matches_formal_state_machine() -> None:
    index = pd.bdate_range("2024-01-02", periods=90)
    score = pd.Series(
        [0.70] * 3 + [0.20] * 35 + [0.75] * 35 + [0.10] * 17,
        index=index,
        dtype=float,
    )
    expected = downside_raqm_state_schedule(
        score,
        formal_spec().state_spec(),
    )
    actual = exit_state_schedule(
        score,
        ExitPolicy("fixed_lock", 30, 1),
    )
    pd.testing.assert_series_equal(
        actual["risk_on"], expected["risk_on"], check_names=False
    )


def test_signed_recovery_can_exit_after_minimum_and_confirmation() -> None:
    index = pd.bdate_range("2024-01-02", periods=20)
    score = pd.Series([0.70] + [0.60] * 19, index=index, dtype=float)
    evidence = pd.Series([False] + [True] * 19, index=index)
    signal = pd.Series([-0.01] + [0.01] * 19, index=index, dtype=float)

    state = signed_recovery_state_schedule(
        score,
        evidence,
        signal,
        confirmation_days=3,
        minimum_lock_days=5,
        fallback_day=30,
    )

    exits = state.index[state["state_reason"].eq("to_momentum_signed_early")]
    assert len(exits) == 1
    assert state.at[exits[0], "held_days_at_open"] == 0
    assert exits[0] == index[5]
