"""Tests for the fixed log-QM switching research primitives."""

import numpy as np
import pandas as pd

from research.momentum_defender_log_qm_switch import (
    EXPANDING_HISTORY,
    LOG_RETURN,
    SIMPLE_RETURN,
    FastSwitchData,
    fast_candidate_schedule,
    fast_gold_targets,
    fast_state_schedule,
    pareto_frontier,
    slow_regime_at_open,
    strict_lag_volatility_cap,
)


def test_slow_regime_is_effective_only_at_next_open():
    index = pd.bdate_range("2024-01-01", periods=5)
    close = pd.Series([100.0, 100.0, 110.0, 90.0, 95.0], index=index)
    simple = slow_regime_at_open(
        close, index, mode=SIMPLE_RETURN, lookback=1, threshold=0.05
    )
    log = slow_regime_at_open(
        close, index, mode=LOG_RETURN, lookback=1, threshold=np.log(1.05)
    )
    assert pd.isna(simple.iloc[0])
    assert not bool(simple.iloc[2])
    assert bool(simple.iloc[3])
    assert simple.equals(log)


def test_expanding_cap_excludes_current_volatility():
    index = pd.bdate_range("2024-01-01", periods=5)
    volatility = pd.Series([1.0, 2.0, 3.0, 100.0, 4.0], index=index)
    cap = strict_lag_volatility_cap(
        volatility,
        0.5,
        history=EXPANDING_HISTORY,
        minimum_history=2,
        step=0.20,
    )
    assert np.isclose(cap.iloc[3]["threshold"], 2.0)
    assert np.isclose(cap.iloc[3]["cap"], 0.0)
    assert np.isclose(cap.iloc[4]["threshold"], 2.5)


def test_pareto_frontier_marks_only_non_dominated_rows():
    frame = pd.DataFrame(
        {
            "annual": [1.0, 2.0, 1.5],
            "sharpe": [1.0, 2.0, 3.0],
            "mdd": [-0.20, -0.10, -0.15],
        },
        index=["dominated", "balanced", "high_sharpe"],
    )
    result = pareto_frontier(frame, ["annual", "sharpe", "mdd"])
    assert not bool(result["dominated"])
    assert bool(result["balanced"])
    assert bool(result["high_sharpe"])


def test_fast_state_schedule_emergency_overrides_hold_lock():
    slow = np.array([True, True, True, True], dtype=object)
    emergency = np.array([False, True, False, False])
    risk_on, entries, switches = fast_state_schedule(slow, emergency, 30)
    assert risk_on.tolist() == [True, False, False, False]
    assert entries == 1
    assert switches == 1


def test_fast_gold_and_candidate_schedule_hard_hold():
    calendar = pd.bdate_range("2024-01-01", periods=7)
    candidates = ("MOM", "518880.SH", "DEFENDER")
    held = np.zeros((3, 7))
    enter = np.zeros((3, 7))
    exit_ = np.zeros((3, 7))
    data = FastSwitchData(
        calendar=calendar,
        candidates=candidates,
        candidate_index={name: index for index, name in enumerate(candidates)},
        momentum_target=np.zeros(7, dtype=int),
        held_returns=held,
        enter_returns=enter,
        exit_returns=exit_,
        initial_candidate=0,
        gold_difference=np.array([3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    )
    risk_on = np.array([False, True, True, True, True, True, True])
    target, entries, days = fast_gold_targets(data, risk_on)
    assert target.tolist() == [1, 1, 1, 1, 1, 0, 0]
    assert entries == 1
    assert days == 5
    returns, actual, switches = fast_candidate_schedule(data, target)
    assert np.allclose(returns, 0.0)
    assert actual.tolist() == target.tolist()
    assert switches == 2
