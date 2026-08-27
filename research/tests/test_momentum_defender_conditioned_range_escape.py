from __future__ import annotations

from dataclasses import replace

import pandas as pd

from research.defender_curve_momentum import DEFENDER_CANDIDATE
from research.momentum_defender_conditioned_range_escape import (
    HOLD_FIXED_PULSE_REARM,
    HOLD_MINIMUM_UNTIL_FAIL,
    ConditionedRangeEscapeParams,
    MomentumQualityGate,
    aggregate_quality_metrics,
    conditioned_range_escape_state,
)
from research.momentum_defender_defender_range_escape import (
    ANCHOR_FIXED_512890,
)


def _params(policy: str) -> ConditionedRangeEscapeParams:
    return ConditionedRangeEscapeParams(
        anchor_mode=ANCHOR_FIXED_512890,
        range_window=40,
        upper_threshold=0.95,
        momentum_weight=0.20,
        quality_window=20,
        gate=MomentumQualityGate(absolute_floor=0.0),
        hold_policy=policy,
        hold_days=5,
    )


def _inputs(periods: int = 9):
    calendar = pd.date_range("2026-01-05", periods=periods, freq="B")
    formal = pd.Series(DEFENDER_CANDIDATE, index=calendar)
    momentum = pd.Series("510300.SH", index=calendar)
    selected = pd.Series("512890.SH", index=calendar)
    locations = pd.DataFrame({"512890.SH": 0.99}, index=calendar)
    metrics = pd.DataFrame(
        {"510300.SH": 0.01, DEFENDER_CANDIDATE: 0.0},
        index=calendar,
    )
    return calendar, formal, momentum, selected, locations, metrics


def test_minimum_hold_ignores_failed_condition_for_five_days() -> None:
    calendar, formal, momentum, selected, locations, metrics = _inputs()
    metrics.loc[calendar[1]:, "510300.SH"] = -0.01
    state = conditioned_range_escape_state(
        formal,
        momentum,
        selected,
        locations,
        metrics,
        _params(HOLD_MINIMUM_UNTIL_FAIL),
    )
    assert state["escape_active"].tolist()[:6] == [True] * 5 + [False]
    assert state.loc[calendar[:5], "escape_asset"].eq("510300.SH").all()


def test_fixed_pulse_requires_condition_to_clear_before_rearming() -> None:
    calendar, formal, momentum, selected, locations, metrics = _inputs()
    state = conditioned_range_escape_state(
        formal,
        momentum,
        selected,
        locations,
        metrics,
        _params(HOLD_FIXED_PULSE_REARM),
    )
    assert state["escape_active"].tolist() == [True] * 5 + [False] * 4

    locations.loc[calendar[6], "512890.SH"] = 0.50
    state = conditioned_range_escape_state(
        formal,
        momentum,
        selected,
        locations,
        metrics,
        _params(HOLD_FIXED_PULSE_REARM),
    )
    assert bool(state.at[calendar[6], "pulse_armed"])
    assert bool(state.at[calendar[7], "escape_active"])


def test_joint_gate_requires_absolute_and_relative_strength() -> None:
    calendar, formal, momentum, selected, locations, metrics = _inputs(3)
    joint = replace(
        _params(HOLD_MINIMUM_UNTIL_FAIL),
        gate=MomentumQualityGate(absolute_floor=0.005, relative_floor=0.015),
    )
    state = conditioned_range_escape_state(
        formal,
        momentum,
        selected,
        locations,
        metrics,
        joint,
    )
    assert not state["escape_active"].any()
    metrics[DEFENDER_CANDIDATE] = -0.01
    state = conditioned_range_escape_state(
        formal,
        momentum,
        selected,
        locations,
        metrics,
        joint,
    )
    assert state["escape_active"].all()


def test_quality_metric_ensemble_uses_equal_weight_aggregation() -> None:
    index = pd.date_range("2026-01-05", periods=2, freq="B")
    panels = {
        20: pd.DataFrame({"asset": [1.0, 4.0]}, index=index),
        40: pd.DataFrame({"asset": [3.0, 2.0]}, index=index),
    }
    mean = aggregate_quality_metrics(panels, (20, 40), "mean")
    minimum = aggregate_quality_metrics(panels, (20, 40), "minimum")
    assert mean["asset"].tolist() == [2.0, 3.0]
    assert minimum["asset"].tolist() == [1.0, 2.0]
