from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from research.defender_curve_momentum import DEFENDER_CANDIDATE
from research.momentum_top1_defender_escape import (
    Top1EscapeParams,
    all_metrics_at_open,
    top1_escape_schedule,
)


def test_unified_metric_uses_previous_close_for_every_candidate() -> None:
    index = pd.date_range("2026-01-01", periods=4, freq="B")
    curves = pd.DataFrame(
        {
            "510300.SH": [100.0, 110.0, 121.0, 133.1],
            "159915.SZ": [100.0, 105.0, 110.25, 115.7625],
            "513100.SH": [100.0, 102.0, 104.04, 106.1208],
            "518880.SH": [100.0, 101.0, 102.01, 103.0301],
            DEFENDER_CANDIDATE: [100.0, 100.5, 101.0025, 101.5075],
        },
        index=index,
    )

    metrics = all_metrics_at_open(curves, "return", window=1)

    assert pd.isna(metrics.iloc[0]).all()
    assert pd.isna(metrics.iloc[1]).all()
    assert metrics.iloc[2]["510300.SH"] > metrics.iloc[2]["159915.SZ"]


def test_escape_uses_current_top1_and_one_shared_rule() -> None:
    index = pd.date_range("2026-01-01", periods=5, freq="B")
    base_state = pd.DataFrame({"risk_on": [False, False, False, False, True]}, index=index)
    top1 = pd.Series(
        ["518880.SH", "518880.SH", "513100.SH", "513100.SH", "159915.SZ"],
        index=index,
    )
    context = SimpleNamespace(
        calendar=index,
        integrated=SimpleNamespace(result=SimpleNamespace(state=base_state)),
        momentum_target=top1,
    )
    metrics = pd.DataFrame(0.0, index=index, columns=[
        "510300.SH", "159915.SZ", "513100.SH", "518880.SH", DEFENDER_CANDIDATE
    ])
    metrics.loc[index[:2], "518880.SH"] = 0.2
    metrics.loc[index[2:4], "513100.SH"] = 0.3
    params = Top1EscapeParams(
        metric="return",
        window=20,
        entry_difference=0.1,
        exit_difference=0.0,
        min_escape_hold_days=1,
    )

    state = top1_escape_schedule(context, metrics, params)

    assert state.iloc[0]["target_candidate"] == "518880.SH"
    assert state.iloc[2]["target_candidate"] == "513100.SH"
    assert state.iloc[2]["state_reason"] == "top1_escape_rotation"
    # Base Momentum always takes precedence over the escape state.
    assert state.iloc[4]["target_candidate"] == "159915.SZ"
