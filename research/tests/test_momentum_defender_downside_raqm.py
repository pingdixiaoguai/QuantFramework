from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.momentum_defender_downside_raqm import (
    EXPANDING_STRICT_LAG,
    DownsideRAQMSpec,
    ExactExecutionData,
    FactorProfile,
    build_downside_raqm_features,
    downside_raqm_state_schedule,
    downside_regularized_raqm,
    exact_candidate_schedule,
    strict_lag_percentile,
)


def _spec(**overrides) -> DownsideRAQMSpec:
    values = {
        "profile": FactorProfile("w20", (20,), (1.0,)),
        "history_mode": EXPANDING_STRICT_LAG,
        "entry_percentile": 0.8,
        "exit_percentile": 0.3,
        "momentum_lock_days": 20,
        "defender_lock_days": 20,
        "entry_confirmation_days": 1,
        "recovery_confirmation_days": 1,
    }
    values.update(overrides)
    return DownsideRAQMSpec(**values)


def test_profile_rejects_any_horizon_below_twenty() -> None:
    with pytest.raises(ValueError, match="at least 20"):
        FactorProfile("invalid", (10, 20), (0.5, 0.5))


@pytest.mark.parametrize("field,value", [("momentum_lock_days", 19), ("defender_lock_days", 31)])
def test_spec_enforces_both_lock_bounds(field: str, value: int) -> None:
    with pytest.raises(ValueError, match="lock"):
        _spec(**{field: value})


def test_downside_raqm_is_positive_only_for_negative_trend() -> None:
    falling = pd.Series(np.exp(np.linspace(0.0, -0.2, 61)))
    rising = pd.Series(np.exp(np.linspace(0.0, 0.2, 61)))
    falling_score = downside_regularized_raqm(falling, 20)
    rising_score = downside_regularized_raqm(rising, 20)
    assert falling_score.iloc[-1] > 0.0
    assert rising_score.iloc[-1] == 0.0


def test_downside_raqm_matches_registered_regularized_formula() -> None:
    close = pd.Series(np.exp(np.linspace(0.0, -0.15, 61)))
    log_close = np.log(close)
    daily = log_close.diff()
    total = log_close.diff(20)
    path = daily.abs().rolling(20).sum()
    efficiency = total.abs() / path
    volatility = daily.rolling(20).std(ddof=1) * np.sqrt(20)
    floor = 0.08 * np.sqrt(20 / 252.0)
    expected = -(
        (total / np.maximum(volatility, floor)).clip(-3.0, 3.0) * efficiency
    )
    actual = downside_regularized_raqm(close, 20)
    assert np.isclose(actual.iloc[-1], expected.iloc[-1])


def test_percentile_reference_excludes_current_observation() -> None:
    values = pd.Series([1.0, 2.0, 3.0, 100.0])
    result = strict_lag_percentile(values, history_window=None, min_history=3)
    assert result.iloc[:3].isna().all()
    assert result.iloc[3] == 1.0


def test_zero_downside_score_maps_to_zero_percentile() -> None:
    values = pd.Series([0.0, 1.0, 2.0, 0.0])
    result = strict_lag_percentile(values, history_window=None, min_history=3)
    assert result.iloc[-1] == 0.0


def test_state_never_bypasses_either_twenty_day_lock() -> None:
    index = pd.date_range("2026-01-01", periods=50, freq="B")
    score = pd.Series([0.9] * 5 + [0.0] * 20 + [0.9] * 25, index=index)
    state = downside_raqm_state_schedule(score, _spec())
    switches = state.loc[state["state_changed"]]
    assert switches.index.tolist() == [index[0], index[20], index[40]]
    assert switches["risk_on"].tolist() == [False, True, False]
    assert switches["held_days_at_open"].tolist() == [0, 0, 0]


def test_confirmation_must_be_consecutive() -> None:
    index = pd.date_range("2026-01-01", periods=8, freq="B")
    score = pd.Series([0.9, 0.7, 0.9, 0.9, 0.7, 0.9, 0.9, 0.9], index=index)
    state = downside_raqm_state_schedule(
        score,
        _spec(entry_confirmation_days=3),
    )
    assert state["state_changed"].sum() == 1
    assert state.index[state["state_changed"]][0] == index[-1]


def test_feature_is_known_only_at_the_next_open_and_weights_are_exact() -> None:
    index = pd.date_range("2024-01-01", periods=80, freq="B")
    close = pd.Series(np.exp(np.linspace(0.0, -0.3, len(index))), index=index)
    calendar = index[50:]
    profiles = {
        "weighted": FactorProfile("weighted", (20, 40), (0.25, 0.75))
    }
    features = build_downside_raqm_features(
        close,
        calendar,
        profiles,
        {EXPANDING_STRICT_LAG: None},
        min_history=5,
        volatility_floor_annual=0.08,
        winsor_limit=3.0,
    )
    execution = calendar[-1]
    prior_close = index[index.get_loc(execution) - 1]
    raw_20 = downside_regularized_raqm(close, 20)
    assert np.isclose(features.raw_at_open[20].loc[execution], raw_20.loc[prior_close])
    p20 = features.percentile_at_open[20, EXPANDING_STRICT_LAG].loc[execution]
    p40 = features.percentile_at_open[40, EXPANDING_STRICT_LAG].loc[execution]
    composite = features.composite_at_open[
        "weighted", EXPANDING_STRICT_LAG
    ].loc[execution]
    assert np.isclose(composite, 0.25 * p20 + 0.75 * p40)


def test_exact_executor_compounds_exit_and_entry_legs() -> None:
    calendar = pd.date_range("2026-01-01", periods=2, freq="B")
    data = ExactExecutionData(
        calendar=calendar,
        candidates=("A", "B"),
        candidate_index={"A": 0, "B": 1},
        momentum_target=np.array([0, 1]),
        held_returns=np.array([[0.01, 0.02], [0.03, 0.04]]),
        enter_returns=np.array([[0.001, 0.002], [0.003, 0.004]]),
        exit_returns=np.array([[0.005, 0.006], [0.007, 0.008]]),
        initial_candidate=0,
    )
    returns, actual, switches = exact_candidate_schedule(data, np.array([0, 1]))
    assert returns[0] == 0.01
    assert np.isclose(returns[1], (1.0 + 0.006) * (1.0 + 0.004) - 1.0)
    assert actual.tolist() == [0, 1]
    assert switches == 1
