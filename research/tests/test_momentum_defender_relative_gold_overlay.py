from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.momentum_defender_downside_raqm import FactorProfile
from research.momentum_defender_gold_exception_gate import (
    GOLD_BIDIRECTIONAL,
    GOLD_EXEMPTION_ONLY,
)
from research.momentum_defender_relative_gold_overlay import (
    RelativeGoldOverlaySpec,
    fast_relative_gold_state,
    relative_gold_overlay_state,
    signed_raqm_profiles_at_open,
)


def _spec(**overrides) -> RelativeGoldOverlaySpec:
    values = {
        "profile": FactorProfile("w20", (20,), (1.0,)),
        "entry_difference": 0.2,
        "exit_difference": 0.0,
        "entry_confirmation_days": 1,
        "exit_confirmation_days": 1,
        "minimum_gold_hold_days": 0,
        "override_mode": GOLD_EXEMPTION_ONLY,
    }
    values.update(overrides)
    return RelativeGoldOverlaySpec(**values)


def _state(spec: RelativeGoldOverlaySpec, targets, base, gold, defender):
    calendar = pd.date_range("2024-01-02", periods=len(targets), freq="B")
    metric = pd.DataFrame(
        {
            "518880.SH": gold,
            "DEFENDER": defender,
            "difference": np.asarray(gold) - np.asarray(defender),
        },
        index=calendar,
    )
    return relative_gold_overlay_state(
        calendar,
        pd.Series(targets, index=calendar),
        pd.Series(base, index=calendar),
        metric,
        spec,
    )


def test_signed_raqm_is_shifted_to_next_open() -> None:
    dates = pd.date_range("2024-01-01", periods=25, freq="B")
    curve = pd.Series(np.exp(np.arange(25) * 0.01), index=dates)
    curves = pd.DataFrame({"518880.SH": curve, "DEFENDER": curve * 1.01})
    profile = FactorProfile("w20", (20,), (1.0,))
    metric = signed_raqm_profiles_at_open(curves, {"w20": profile})["w20"]
    assert metric.iloc[20].isna().all()
    assert metric.iloc[21].notna().all()
    assert metric.iloc[21]["difference"] == pytest.approx(0.0, abs=1e-12)


def test_exemption_requires_healthy_gold_and_relative_advantage() -> None:
    state = _state(
        _spec(),
        ["518880.SH"] * 4,
        [False] * 4,
        [-0.1, 0.1, 0.4, 0.3],
        [-0.5, 0.0, 0.1, 0.4],
    )
    assert state["gold_overlay_active"].tolist() == [False, False, True, False]
    assert state["effective_risk_on"].tolist() == [False, False, True, False]


def test_overlay_resets_when_gold_leaves_top1() -> None:
    state = _state(
        _spec(),
        ["518880.SH", "510300.SH", "518880.SH"],
        [False, True, False],
        [0.4, 0.4, 0.1],
        [0.0, 0.0, 0.0],
    )
    assert state["gold_overlay_active"].tolist() == [True, False, False]
    assert state["effective_risk_on"].tolist() == [True, True, False]


def test_exemption_mode_never_turns_a_base_risk_on_day_off() -> None:
    state = _state(
        _spec(),
        ["518880.SH"] * 2,
        [True, True],
        [0.1, -0.1],
        [0.2, 0.2],
    )
    assert state["effective_risk_on"].tolist() == [True, True]


def test_bidirectional_mode_can_add_defender_on_weak_gold() -> None:
    state = _state(
        _spec(override_mode=GOLD_BIDIRECTIONAL),
        ["518880.SH"] * 2,
        [True, True],
        [0.1, -0.1],
        [0.2, 0.2],
    )
    assert state["effective_risk_on"].tolist() == [False, False]


def test_weak_gold_exit_bypasses_relative_hold() -> None:
    state = _state(
        _spec(minimum_gold_hold_days=30),
        ["518880.SH"] * 2,
        [False, False],
        [0.4, -0.1],
        [0.0, -0.2],
    )
    assert state["gold_overlay_active"].tolist() == [True, False]
    assert state.iloc[1]["gold_overlay_reason"] == "weak_gold_hard_exit"


def test_hold_must_be_a_five_day_multiple() -> None:
    with pytest.raises(ValueError, match="multiple of five"):
        _spec(minimum_gold_hold_days=7)


@pytest.mark.parametrize("mode", [GOLD_EXEMPTION_ONLY, GOLD_BIDIRECTIONAL])
def test_fast_search_state_matches_detailed_ledger(mode: str) -> None:
    targets = [
        "510300.SH",
        "518880.SH",
        "518880.SH",
        "518880.SH",
        "518880.SH",
        "510300.SH",
        "518880.SH",
        "518880.SH",
    ]
    base = [True, False, False, False, True, False, False, False]
    gold = [0.2, 0.1, 0.4, 0.3, -0.1, 0.5, 0.4, 0.2]
    defender = [0.0, 0.0, 0.1, 0.4, -0.2, 0.0, 0.1, 0.3]
    spec = _spec(
        override_mode=mode,
        entry_confirmation_days=2,
        exit_confirmation_days=1,
        minimum_gold_hold_days=5,
    )
    detailed = _state(spec, targets, base, gold, defender)
    fast = fast_relative_gold_state(
        np.asarray(targets) == "518880.SH",
        np.asarray(base),
        np.asarray(gold),
        np.asarray(defender),
        np.asarray(gold) - np.asarray(defender),
        spec,
    )
    np.testing.assert_array_equal(
        fast.effective_risk_on, detailed["effective_risk_on"].to_numpy(bool)
    )
    np.testing.assert_array_equal(
        fast.gold_overlay_active,
        detailed["gold_overlay_active"].to_numpy(bool),
    )
    np.testing.assert_array_equal(
        fast.gold_overlay_changed,
        detailed["gold_overlay_changed"].to_numpy(bool),
    )
    np.testing.assert_array_equal(
        fast.gold_overrides_base,
        detailed["gold_overrides_base"].to_numpy(bool),
    )
