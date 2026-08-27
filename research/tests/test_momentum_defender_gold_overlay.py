from __future__ import annotations

import pandas as pd

from research.momentum_defender_downside_raqm import DownsideRAQMFeatures, FactorProfile
from research.momentum_defender_gold_exception_gate import (
    GOLD_BIDIRECTIONAL,
    GOLD_EXEMPTION_ONLY,
)
from research.momentum_defender_gold_overlay import (
    GoldOverlaySpec,
    gold_overlay_state_schedule,
)
from research.momentum_defender_selected_asset_draqm import AssetDRAQMPolicy


def _features(index, anchor, gold):
    return {
        "510300.SH": DownsideRAQMFeatures(
            index,
            {},
            {},
            {("anchor", "rolling_504_strict_lag"): pd.Series(anchor, index=index)},
        ),
        "518880.SH": DownsideRAQMFeatures(
            index,
            {},
            {},
            {("gold", "rolling_504_strict_lag"): pd.Series(gold, index=index)},
        ),
    }


def _spec(mode):
    return GoldOverlaySpec(
        AssetDRAQMPolicy(
            "510300.SH", FactorProfile("anchor", (20,), (1.0,)), 0.6, 0.2, 1, 1
        ),
        AssetDRAQMPolicy(
            "518880.SH", FactorProfile("gold", (20,), (1.0,)), 0.7, 0.1, 1, 1
        ),
        0,
        0,
        mode,
    )


def test_gold_override_never_mutates_independent_base_state() -> None:
    index = pd.date_range("2026-01-01", periods=4, freq="B")
    features = _features(index, [0.9] * 4, [0.0] * 4)
    gold_target = pd.Series("518880.SH", index=index)
    other_target = pd.Series("513100.SH", index=index)
    gold = gold_overlay_state_schedule(
        index, gold_target, features, _spec(GOLD_EXEMPTION_ONLY)
    )
    other = gold_overlay_state_schedule(
        index, other_target, features, _spec(GOLD_EXEMPTION_ONLY)
    )
    pd.testing.assert_series_equal(
        gold["base_risk_on"], other["base_risk_on"], check_names=False
    )
    assert not gold["base_risk_on"].any()
    assert gold["effective_risk_on"].all()


def test_leaving_gold_immediately_returns_to_base_state() -> None:
    index = pd.date_range("2026-01-01", periods=3, freq="B")
    target = pd.Series(["518880.SH", "518880.SH", "513100.SH"], index=index)
    state = gold_overlay_state_schedule(
        index,
        target,
        _features(index, [0.9] * 3, [0.0] * 3),
        _spec(GOLD_EXEMPTION_ONLY),
    )
    assert bool(state.iloc[1]["effective_risk_on"])
    assert not bool(state.iloc[2]["effective_risk_on"])


def test_bidirectional_gold_can_add_defender_without_changing_base() -> None:
    index = pd.date_range("2026-01-01", periods=2, freq="B")
    target = pd.Series("518880.SH", index=index)
    state = gold_overlay_state_schedule(
        index,
        target,
        _features(index, [0.0, 0.0], [0.9, 0.9]),
        _spec(GOLD_BIDIRECTIONAL),
    )
    assert state["base_risk_on"].all()
    assert not bool(state.iloc[0]["effective_risk_on"])


def test_gold_streak_resets_after_top1_gap() -> None:
    index = pd.date_range("2026-01-01", periods=4, freq="B")
    gold_policy = AssetDRAQMPolicy(
        "518880.SH", FactorProfile("gold", (20,), (1.0,)), 0.7, 0.1, 2, 1
    )
    spec = GoldOverlaySpec(
        _spec(GOLD_BIDIRECTIONAL).anchor_policy,
        gold_policy,
        0,
        0,
        GOLD_BIDIRECTIONAL,
    )
    target = pd.Series(
        ["518880.SH", "513100.SH", "518880.SH", "518880.SH"], index=index
    )
    state = gold_overlay_state_schedule(
        index, target, _features(index, [0.0] * 4, [0.9] * 4), spec
    )
    assert bool(state.iloc[2]["effective_risk_on"])
    assert not bool(state.iloc[3]["effective_risk_on"])
