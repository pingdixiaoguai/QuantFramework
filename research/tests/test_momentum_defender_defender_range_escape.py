from __future__ import annotations

import pandas as pd

from research.defender_curve_momentum import DEFENDER_CANDIDATE
from research.momentum_defender_defender_range_escape import (
    ANCHOR_FIXED_512890,
    POLICY_BINARY_PARTIAL,
    POLICY_CONTINUOUS_GRID,
    POLICY_DEFENDER_EPISODE_GRID,
    DefenderRangeEscapeParams,
    _target_schedule,
    range_escape_state,
)


def _inputs():
    calendar = pd.date_range("2026-01-05", periods=5, freq="B")
    formal = pd.Series(
        [DEFENDER_CANDIDATE, DEFENDER_CANDIDATE, "510300.SH", DEFENDER_CANDIDATE, DEFENDER_CANDIDATE],
        index=calendar,
    )
    momentum = pd.Series("510300.SH", index=calendar)
    selected = pd.Series("512890.SH", index=calendar)
    locations = pd.DataFrame(
        {"512890.SH": [0.99, 0.99, 0.50, 0.99, 0.01]},
        index=calendar,
    )
    return calendar, formal, momentum, selected, locations


def test_continuous_grid_restarts_full_then_keeps_memory() -> None:
    _, formal, momentum, selected, locations = _inputs()
    params = DefenderRangeEscapeParams(
        anchor_mode=ANCHOR_FIXED_512890,
        policy=POLICY_CONTINUOUS_GRID,
        position_step=0.20,
    )
    state = range_escape_state(formal, momentum, selected, locations, params)
    assert state["raw_defender_weight"].tolist() == [1.0, 0.8, 0.8, 0.6, 0.8]
    assert state["overlay_active"].tolist() == [False, True, False, True, True]


def test_episode_grid_resets_outside_formal_defender() -> None:
    _, formal, momentum, selected, locations = _inputs()
    params = DefenderRangeEscapeParams(
        policy=POLICY_DEFENDER_EPISODE_GRID,
        position_step=0.20,
    )
    state = range_escape_state(formal, momentum, selected, locations, params)
    assert state["raw_defender_weight"].tolist() == [1.0, 0.8, 1.0, 0.8, 1.0]


def test_binary_policy_never_accumulates_reductions() -> None:
    _, formal, momentum, selected, locations = _inputs()
    params = DefenderRangeEscapeParams(
        policy=POLICY_BINARY_PARTIAL,
        position_step=0.20,
    )
    state = range_escape_state(formal, momentum, selected, locations, params)
    assert state["raw_defender_weight"].tolist() == [1.0, 0.8, 1.0, 0.8, 1.0]


def test_target_schedule_blends_only_formal_defender_days() -> None:
    calendar = pd.date_range("2026-01-05", periods=2, freq="B")
    formal = pd.Series([DEFENDER_CANDIDATE, "518880.SH"], index=calendar)
    momentum = pd.Series(["510300.SH", "510300.SH"], index=calendar)
    defender = pd.DataFrame(
        {"512890.SH": [1.0, 1.0], "510880.SH": [0.0, 0.0]},
        index=calendar,
    )
    weights = pd.Series([0.8, 0.2], index=calendar)
    targets = _target_schedule(
        formal,
        momentum,
        defender,
        weights,
        ("510300.SH", "518880.SH", "512890.SH", "510880.SH"),
    )
    assert targets.loc[calendar[0]].to_dict() == {
        "510300.SH": 0.19999999999999996,
        "518880.SH": 0.0,
        "512890.SH": 0.8,
        "510880.SH": 0.0,
    }
    assert targets.loc[calendar[1], "518880.SH"] == 1.0
    assert targets.loc[calendar[1], "512890.SH"] == 0.0
