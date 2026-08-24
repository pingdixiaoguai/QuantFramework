from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from factors.risk_adjusted_quality_momentum import compute
from research.defender_curve_momentum import DEFENDER_CANDIDATE
from research.momentum_defender_occam import MOMENTUM_ASSETS
from research.top1_raqm_w5_bridge import (
    Top1RAQMW5BridgeParams,
    registered_raqm_w5_at_open,
    top1_raqm_w5_bridge_schedule,
)


def _context(
    index: pd.DatetimeIndex,
    top1: list[str],
    slow: list[bool],
    emergency: list[bool],
):
    state = pd.DataFrame(
        {
            "slow_signal_asof_previous_close": slow,
            "emergency_asof_previous_close": emergency,
        },
        index=index,
    )
    return SimpleNamespace(
        calendar=index,
        momentum_target=pd.Series(top1, index=index),
        integrated=SimpleNamespace(result=SimpleNamespace(state=state)),
    )


def test_registered_raqm_w5_uses_exact_factor_and_next_open_shift() -> None:
    index = pd.date_range("2026-01-01", periods=12, freq="B", name="date")
    curves = pd.DataFrame(index=index)
    for position, candidate in enumerate((*MOMENTUM_ASSETS, DEFENDER_CANDIDATE)):
        curves[candidate] = 100.0 * np.exp(
            np.arange(len(index)) * (0.002 + position * 0.001)
        )

    result = registered_raqm_w5_at_open(curves)
    expected = compute(
        pd.DataFrame(
            {"date": index, "close": curves["510300.SH"].to_numpy(float)}
        ),
        {"window": 5, "vol_floor_annual": 0.08},
    ).reindex(index).shift(1)

    pd.testing.assert_series_equal(
        result["510300.SH"], expected, check_names=False, check_freq=False
    )


def test_bridge_requires_confirmation_and_never_overrides_emergency_or_formal() -> None:
    index = pd.date_range("2026-01-01", periods=7, freq="B")
    context = _context(
        index,
        [
            "510300.SH",
            "510300.SH",
            "159915.SZ",
            "159915.SZ",
            "159915.SZ",
            "513100.SH",
            "513100.SH",
        ],
        [True] * 7,
        [False, False, False, True, False, False, False],
    )
    formal = pd.Series(
        [
            DEFENDER_CANDIDATE,
            DEFENDER_CANDIDATE,
            DEFENDER_CANDIDATE,
            DEFENDER_CANDIDATE,
            "518880.SH",
            DEFENDER_CANDIDATE,
            DEFENDER_CANDIDATE,
        ],
        index=index,
    )
    metrics = pd.DataFrame(
        0.0,
        index=index,
        columns=[*MOMENTUM_ASSETS, DEFENDER_CANDIDATE],
    )
    for timestamp, candidate in zip(index, context.momentum_target, strict=True):
        metrics.at[timestamp, candidate] = 2.5
    params = Top1RAQMW5BridgeParams(
        entry_minimum=2.2,
        confirmation_days=2,
    )

    state = top1_raqm_w5_bridge_schedule(context, formal, metrics, params)

    assert not bool(state.iloc[0]["top1_bridge_active"])
    assert state.iloc[1]["state_reason"] == "top1_bridge_entry"
    # Once active, the bridge follows the live Momentum Top-1 instead of
    # freezing the entry ETF.
    assert state.iloc[2]["target_candidate"] == "159915.SZ"
    assert state.iloc[3]["state_reason"] == "bridge_safety_exit"
    assert state.iloc[3]["target_candidate"] == DEFENDER_CANDIDATE
    # The formal Gold target has priority even though Top-1 remains strong.
    assert state.iloc[4]["target_candidate"] == "518880.SH"
    assert not bool(state.iloc[4]["top1_bridge_active"])
    # Confirmation does not carry through a formal non-Defender day.
    assert not bool(state.iloc[5]["top1_bridge_active"])
    assert state.iloc[6]["state_reason"] == "top1_bridge_entry"


def test_threshold_above_observed_factor_preserves_formal_schedule() -> None:
    index = pd.date_range("2026-01-01", periods=4, freq="B")
    context = _context(
        index,
        ["510300.SH"] * 4,
        [True] * 4,
        [False] * 4,
    )
    formal = pd.Series(DEFENDER_CANDIDATE, index=index)
    metrics = pd.DataFrame(
        3.0,
        index=index,
        columns=[*MOMENTUM_ASSETS, DEFENDER_CANDIDATE],
    )
    params = Top1RAQMW5BridgeParams(
        entry_minimum=3.0,
        confirmation_days=1,
    )

    state = top1_raqm_w5_bridge_schedule(context, formal, metrics, params)

    pd.testing.assert_series_equal(
        state["target_candidate"], formal, check_names=False, check_freq=False
    )
