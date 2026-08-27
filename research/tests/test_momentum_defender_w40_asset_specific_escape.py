from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from research.defender_curve_momentum import DEFENDER_CANDIDATE
from research.momentum_defender_occam import MOMENTUM_ASSETS
from research.momentum_defender_w40_asset_specific_escape import (
    AssetXYPolicy,
    asset_specific_escape_schedule,
)


def _policies() -> dict[str, AssetXYPolicy | None]:
    result = {asset: None for asset in MOMENTUM_ASSETS}
    result["518880.SH"] = AssetXYPolicy(0.01, -0.02)
    result["513100.SH"] = AssetXYPolicy(0.08, 0.02)
    return result


def test_each_entry_uses_current_top1_own_x() -> None:
    index = pd.bdate_range("2026-01-01", periods=8)
    context = SimpleNamespace(
        calendar=index,
        initial_previous_candidate="510300.SH",
        momentum_target=pd.Series(
            ["518880.SH"] * 6 + ["513100.SH"] * 2, index=index
        ),
    )
    formal = pd.DataFrame(
        {"risk_on": False, "held_days_at_open": range(8)}, index=index
    )
    metrics = pd.DataFrame(
        0.0, index=index, columns=[*MOMENTUM_ASSETS, DEFENDER_CANDIDATE]
    )
    metrics["518880.SH"] = 0.02
    metrics["513100.SH"] = 0.05
    state = asset_specific_escape_schedule(
        context, formal, metrics, _policies()
    )
    assert state.iloc[5]["target_candidate"] == "518880.SH"
    assert state.iloc[5]["current_top1_entry_x"] == 0.01


def test_after_hard_hold_exit_uses_new_current_top1_y() -> None:
    index = pd.bdate_range("2026-01-01", periods=12)
    context = SimpleNamespace(
        calendar=index,
        initial_previous_candidate="510300.SH",
        momentum_target=pd.Series(
            ["518880.SH"] * 7 + ["513100.SH"] * 5, index=index
        ),
    )
    formal = pd.DataFrame(
        {"risk_on": False, "held_days_at_open": range(12)}, index=index
    )
    metrics = pd.DataFrame(
        0.0, index=index, columns=[*MOMENTUM_ASSETS, DEFENDER_CANDIDATE]
    )
    metrics["518880.SH"] = 0.03
    metrics["513100.SH"] = 0.01
    state = asset_specific_escape_schedule(
        context, formal, metrics, _policies()
    )
    assert state.iloc[5:10]["target_candidate"].eq("518880.SH").all()
    assert state.iloc[10]["momentum_top1"] == "513100.SH"
    assert state.iloc[10]["current_top1_exit_y"] == 0.02
    assert state.iloc[10]["target_candidate"] == DEFENDER_CANDIDATE
    assert state.iloc[10]["state_reason"] == "asset_escape_return_below_y"


def test_immediate_gold_veto_changes_execution_not_base_defender_state() -> None:
    index = pd.bdate_range("2026-01-01", periods=7)
    context = SimpleNamespace(
        calendar=index,
        initial_previous_candidate="510300.SH",
        momentum_target=pd.Series(["518880.SH"] * len(index), index=index),
    )
    formal = pd.DataFrame(
        {
            "risk_on": False,
            "state_changed": [True] + [False] * (len(index) - 1),
            "held_days_at_open": range(len(index)),
        },
        index=index,
    )
    metrics = pd.DataFrame(
        0.0, index=index, columns=[*MOMENTUM_ASSETS, DEFENDER_CANDIDATE]
    )
    metrics["518880.SH"] = 0.02

    original = asset_specific_escape_schedule(
        context, formal, metrics, _policies()
    )
    candidate = asset_specific_escape_schedule(
        context,
        formal,
        metrics,
        _policies(),
        immediate_entry_veto=True,
    )

    assert original.iloc[0]["target_candidate"] == DEFENDER_CANDIDATE
    assert candidate.iloc[0]["base_w40_risk_on"] == False
    assert candidate.iloc[0]["base_w40_defender_entry"] == True
    assert candidate.iloc[0]["actual_defender_held_days_at_open"] == 0
    assert candidate.iloc[0]["target_candidate"] == "518880.SH"
    assert candidate.iloc[0]["state_reason"] == "asset_escape_veto_defender_entry"
    assert candidate.iloc[:5]["target_candidate"].eq("518880.SH").all()
