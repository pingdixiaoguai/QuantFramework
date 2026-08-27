from __future__ import annotations

import pandas as pd

from research.momentum_breadth_hysteresis import (
    HysteresisParams,
    MOMENTUM_ASSETS,
    state_schedule,
)


def _frame(rows: list[list[float]]) -> pd.DataFrame:
    return pd.DataFrame(
        rows,
        index=pd.date_range("2026-01-01", periods=len(rows), freq="D"),
        columns=list(MOMENTUM_ASSETS),
    )


def test_entry_requires_all_four_and_exit_requires_any_recovery_asset() -> None:
    returns = _frame(
        [
            [-0.10, -0.10, -0.10, -0.10],  # enter after this close
            [0.10, -0.10, -0.10, -0.10],  # CSI 300 exits after this close
            [-0.10, -0.10, -0.10, 0.00],  # gold prevents a new entry
            [-0.10, -0.10, -0.10, -0.10],  # enter again
            [-0.10, 0.20, -0.10, -0.10],  # ChiNext cannot trigger exit
            [-0.10, -0.10, 0.10, -0.10],  # Nasdaq exits
            [0.00, 0.00, 0.00, 0.00],
        ]
    )

    state, entry, exit_ = state_schedule(
        returns, HysteresisParams(40, -0.05, 0.05)
    )

    assert entry.tolist() == [True, False, False, True, False, False, False]
    assert exit_.tolist() == [False, True, False, False, False, True, False]
    assert state.tolist() == [True, False, True, True, False, False, True]


def test_threshold_comparisons_are_strict() -> None:
    returns = _frame(
        [
            [-0.05, -0.05, -0.05, -0.05],
            [-0.051, -0.051, -0.051, -0.051],
            [0.05, -0.10, -0.10, -0.10],
            [0.051, -0.10, -0.10, -0.10],
            [0.00, 0.00, 0.00, 0.00],
        ]
    )

    state, entry, exit_ = state_schedule(
        returns, HysteresisParams(40, -0.05, 0.05)
    )

    assert entry.tolist() == [False, True, False, False, False]
    assert exit_.tolist() == [False, False, False, True, False]
    assert state.tolist() == [True, True, False, False, True]


def test_entry_threshold_may_exceed_exit_threshold() -> None:
    params = HysteresisParams(40, 0.10, 0.025)
    assert params.entry_threshold == 0.10
    assert params.exit_threshold == 0.025
