from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from research.defender_curve_momentum import DEFENDER_CANDIDATE
from research.momentum_defender_w40_top1_escape import (
    W40Top1EscapeSpec,
    top1_escape_schedule,
)


def _context(index: pd.DatetimeIndex) -> SimpleNamespace:
    return SimpleNamespace(
        calendar=index,
        initial_previous_candidate="510300.SH",
        momentum_target=pd.Series(
            ["510300.SH"] * 7 + ["518880.SH"] * (len(index) - 7), index=index
        ),
    )


def test_escape_waits_five_defender_days_and_hard_holds_entry_top1() -> None:
    index = pd.bdate_range("2026-01-01", periods=13)
    formal_state = pd.DataFrame(
        {"risk_on": False, "held_days_at_open": range(13)}, index=index
    )
    metrics = pd.DataFrame(
        0.0,
        index=index,
        columns=[
            "510300.SH",
            "159915.SZ",
            "513100.SH",
            "518880.SH",
            DEFENDER_CANDIDATE,
        ],
    )
    metrics["510300.SH"] = 0.02
    metrics["518880.SH"] = 0.03
    metrics.loc[index[10]:, ["510300.SH", "518880.SH"]] = -0.02
    state = top1_escape_schedule(
        _context(index),
        formal_state,
        metrics,
        W40Top1EscapeSpec(0.01, -0.01),
    )

    assert state.iloc[:5]["target_candidate"].eq(DEFENDER_CANDIDATE).all()
    assert state.iloc[5]["state_reason"] == "top1_escape_break_defender_lock"
    assert state.iloc[5:10]["target_candidate"].eq("510300.SH").all()
    assert state.iloc[10]["state_reason"] == "top1_escape_return_to_defender"


def test_after_hard_hold_escape_delegates_to_normal_momentum_top1() -> None:
    index = pd.bdate_range("2026-01-01", periods=12)
    formal_state = pd.DataFrame(
        {"risk_on": False, "held_days_at_open": range(12)}, index=index
    )
    metrics = pd.DataFrame(
        0.0,
        index=index,
        columns=[
            "510300.SH",
            "159915.SZ",
            "513100.SH",
            "518880.SH",
            DEFENDER_CANDIDATE,
        ],
    )
    metrics["510300.SH"] = 0.03
    metrics["518880.SH"] = 0.04
    state = top1_escape_schedule(
        _context(index),
        formal_state,
        metrics,
        W40Top1EscapeSpec(0.01, -0.01),
    )

    assert state.iloc[5:10]["target_candidate"].eq("510300.SH").all()
    assert state.iloc[10]["target_candidate"] == "518880.SH"
    assert state.iloc[10]["state_reason"] == "top1_escape_normal_rotation"
