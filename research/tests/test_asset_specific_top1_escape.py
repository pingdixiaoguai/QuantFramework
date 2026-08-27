from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from research.asset_specific_top1_escape import (
    AssetEscapePolicy,
    asset_specific_schedule,
    build_policy_grid,
)
from research.defender_curve_momentum import DEFENDER_CANDIDATE
from research.momentum_defender_occam import MOMENTUM_ASSETS


def test_each_top1_uses_its_own_threshold() -> None:
    index = pd.date_range("2026-01-01", periods=4, freq="B")
    base = pd.DataFrame({"risk_on": False}, index=index)
    top1 = pd.Series(
        ["518880.SH", "518880.SH", "513100.SH", "513100.SH"], index=index
    )
    context = SimpleNamespace(
        calendar=index,
        integrated=SimpleNamespace(result=SimpleNamespace(state=base)),
        momentum_target=top1,
    )
    gold_policy = AssetEscapePolicy("return", 10, 0.5, -0.2, min_hold_days=1)
    nasdaq_policy = AssetEscapePolicy("return", 10, 0.8, -0.2, min_hold_days=1)
    policies = {asset: None for asset in MOMENTUM_ASSETS}
    policies["518880.SH"] = gold_policy
    policies["513100.SH"] = nasdaq_policy
    frame = pd.DataFrame(0.0, index=index, columns=[*MOMENTUM_ASSETS, DEFENDER_CANDIDATE])
    frame.loc[index[:2], "518880.SH"] = 0.6
    frame.loc[index[2:], "513100.SH"] = 0.6

    state = asset_specific_schedule(context, policies, {("return", 10): frame})

    assert state.iloc[0]["target_candidate"] == "518880.SH"
    # Nasdaq uses the stricter 0.8 threshold, so the escape returns Defender.
    assert state.iloc[2]["target_candidate"] == DEFENDER_CANDIDATE


def test_policy_grid_is_shared_but_creates_distinct_policies() -> None:
    policies = build_policy_grid(
        {
            "return": {
                "windows": [10],
                "entry_differences": [0.1, 0.2],
                "exit_differences": [0.0],
                "min_hold_days": [1, 5],
            }
        }
    )

    assert len(policies) == 4
    assert len({policy.policy_id() for policy in policies}) == 4
