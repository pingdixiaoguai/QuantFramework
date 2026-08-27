from __future__ import annotations

import pandas as pd

from research.momentum_defender_downside_raqm import DownsideRAQMFeatures, FactorProfile
from research.momentum_defender_gold_exception_gate import (
    GOLD_BIDIRECTIONAL,
    GOLD_EXEMPTION_ONLY,
    GoldExceptionSpec,
    gold_exception_state_schedule,
)
from research.momentum_defender_selected_asset_draqm import AssetDRAQMPolicy


def _policy(asset: str, profile: str, entry: float, recovery: float):
    return AssetDRAQMPolicy(
        asset, FactorProfile(profile, (20,), (1.0,)), entry, recovery, 1, 1
    )


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
    return GoldExceptionSpec(
        _policy("510300.SH", "anchor", 0.6, 0.2),
        _policy("518880.SH", "gold", 0.7, 0.1),
        0,
        0,
        mode,
    )


def test_non_gold_top1_uses_anchor_gate() -> None:
    index = pd.date_range("2026-01-01", periods=2, freq="B")
    target = pd.Series("513100.SH", index=index)
    state = gold_exception_state_schedule(
        index, target, _features(index, [0.9, 0.9], [0.0, 0.0]), _spec(GOLD_EXEMPTION_ONLY)
    )
    assert not bool(state.iloc[0]["risk_on"])
    assert state.iloc[0]["evidence_source"] == "anchor"


def test_healthy_gold_blocks_anchor_defender_entry() -> None:
    index = pd.date_range("2026-01-01", periods=2, freq="B")
    target = pd.Series("518880.SH", index=index)
    state = gold_exception_state_schedule(
        index, target, _features(index, [0.9, 0.9], [0.0, 0.0]), _spec(GOLD_EXEMPTION_ONLY)
    )
    assert state["risk_on"].all()
    assert state.iloc[0]["state_reason"] == "gold_exception_blocks_anchor_defender"


def test_bidirectional_gold_can_enter_defender_while_anchor_is_healthy() -> None:
    index = pd.date_range("2026-01-01", periods=2, freq="B")
    target = pd.Series("518880.SH", index=index)
    features = _features(index, [0.0, 0.0], [0.9, 0.9])
    exemption = gold_exception_state_schedule(
        index, target, features, _spec(GOLD_EXEMPTION_ONLY)
    )
    bidirectional = gold_exception_state_schedule(
        index, target, features, _spec(GOLD_BIDIRECTIONAL)
    )
    assert exemption["risk_on"].all()
    assert not bool(bidirectional.iloc[0]["risk_on"])


def test_gold_recovery_overrides_anchor_while_gold_is_top1() -> None:
    index = pd.date_range("2026-01-01", periods=3, freq="B")
    target = pd.Series("518880.SH", index=index)
    state = gold_exception_state_schedule(
        index,
        target,
        _features(index, [0.9, 0.9, 0.9], [0.9, 0.0, 0.0]),
        _spec(GOLD_BIDIRECTIONAL),
    )
    assert not bool(state.iloc[0]["risk_on"])
    assert bool(state.iloc[1]["risk_on"])


def test_confirmation_resets_when_source_changes() -> None:
    index = pd.date_range("2026-01-01", periods=4, freq="B")
    anchor = AssetDRAQMPolicy(
        "510300.SH", FactorProfile("anchor", (20,), (1.0,)), 0.6, 0.2, 2, 1
    )
    gold = AssetDRAQMPolicy(
        "518880.SH", FactorProfile("gold", (20,), (1.0,)), 0.7, 0.1, 2, 1
    )
    spec = GoldExceptionSpec(anchor, gold, 0, 0, GOLD_BIDIRECTIONAL)
    target = pd.Series(
        ["513100.SH", "518880.SH", "513100.SH", "513100.SH"], index=index
    )
    state = gold_exception_state_schedule(
        index, target, _features(index, [0.9] * 4, [0.9] * 4), spec
    )
    assert bool(state.iloc[2]["risk_on"])
    assert not bool(state.iloc[3]["risk_on"])
