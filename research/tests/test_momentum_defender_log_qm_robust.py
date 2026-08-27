"""Tests for broad robust regime mechanisms."""

import numpy as np
import pandas as pd

from research.momentum_defender_log_qm_robust import (
    ANCHOR_OR_HELD,
    COMBINED_VOTE,
    DRAWDOWN,
    EmergencySpec,
    FeatureBundle,
    GateSpec,
    EnsembleGateSpec,
    StatePolicy,
    asymmetric_state_schedule,
    emergency_signal,
    ensemble_gate_signal,
    gate_signal,
)


def _features() -> FeatureBundle:
    index = pd.bdate_range("2024-01-01", periods=4)
    assets = ("510300.SH", "159915.SZ", "513100.SH", "518880.SH")
    returns = pd.DataFrame(
        [
            [-0.1, 0.1, -0.1, -0.1],
            [-0.1, -0.1, -0.1, -0.1],
            [0.1, -0.1, -0.1, -0.1],
            [0.1, 0.1, 0.1, 0.1],
        ],
        index=index,
        columns=assets,
    )
    drawdown = returns.copy()
    empty_alert = pd.DataFrame(False, index=index, columns=assets)
    return FeatureBundle(
        calendar=index,
        previous_asset=pd.Series(["159915.SZ"] * 4, index=index),
        log_returns={1: returns, 20: returns},
        drawdowns={20: drawdown},
        downside_alerts={(5, 0.9, "expanding_strict_lag"): empty_alert},
        rs_alerts={(5, 0.9, "expanding_strict_lag"): empty_alert},
        relative_returns={
            20: pd.Series([0.1, -0.1, 0.1, -0.1], index=index)
        },
    )


def test_anchor_or_held_gate_uses_either_market():
    policy = StatePolicy("p", 1, 1, 1, 1)
    spec = GateSpec(ANCHOR_OR_HELD, 20, 0.0, 2, policy)
    assert gate_signal(_features(), spec).tolist() == [True, False, True, True]


def test_drawdown_emergency_uses_only_held_asset():
    spec = EmergencySpec(DRAWDOWN, window=20, threshold=-0.05)
    assert emergency_signal(
        _features(), spec, negative_trend_window=1
    ).tolist() == [False, True, True, False]


def test_asymmetric_confirmation_and_holds():
    index = pd.bdate_range("2024-01-01", periods=8)
    wanted = pd.Series([True, False, False, False, True, True, True, True], index=index)
    emergency = pd.Series(False, index=index)
    policy = StatePolicy("p", 2, 3, 3, 2)
    risk_on, entries, switches = asymmetric_state_schedule(wanted, emergency, policy)
    assert risk_on.tolist() == [True, True, True, False, False, False, True, True]
    assert entries == 1
    assert switches == 2


def test_combined_ensemble_counts_trend_and_relative_votes():
    policy = StatePolicy("p", 1, 1, 1, 1)
    spec = EnsembleGateSpec(COMBINED_VOTE, (20,), 0.5, 0.0, policy)
    assert ensemble_gate_signal(_features(), spec).tolist() == [True, False, True, True]
