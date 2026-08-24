from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from factors.risk_adjusted_quality_momentum import compute
from research.all_asset_raqm_override import (
    AssetRAQMThresholds,
    CommonRAQMSpec,
    all_asset_raqm_schedule,
    common_raqm_at_open,
)
from research.defender_curve_momentum import DEFENDER_CANDIDATE
from research.momentum_defender_occam import MOMENTUM_ASSETS


def _context(index: pd.DatetimeIndex, risk_on: list[bool], top1: list[str]):
    return SimpleNamespace(
        calendar=index,
        momentum_target=pd.Series(top1, index=index),
        integrated=SimpleNamespace(
            result=SimpleNamespace(
                state=pd.DataFrame({"risk_on": risk_on}, index=index)
            )
        ),
    )


def test_common_w5_er1_matches_registered_factor_with_next_open_shift() -> None:
    index = pd.date_range("2026-01-01", periods=15, freq="B", name="date")
    curves = pd.DataFrame(index=index)
    for position, candidate in enumerate((*MOMENTUM_ASSETS, DEFENDER_CANDIDATE)):
        curves[candidate] = 100.0 * np.exp(
            np.arange(len(index)) * (0.001 + position * 0.0005)
        )
    spec = CommonRAQMSpec(window=5, efficiency_power=1.0)

    actual = common_raqm_at_open(curves, spec)["510300.SH"]
    expected = compute(
        pd.DataFrame(
            {"date": index, "close": curves["510300.SH"].to_numpy(float)}
        ),
        {"window": 5, "vol_floor_annual": 0.08},
    ).reindex(index).shift(1)

    pd.testing.assert_series_equal(
        actual, expected, check_names=False, check_freq=False
    )


def test_asset_specific_thresholds_choose_largest_margin() -> None:
    index = pd.date_range("2026-01-01", periods=2, freq="B")
    context = _context(index, [False, False], ["510300.SH", "510300.SH"])
    metrics = pd.DataFrame(
        0.0,
        index=index,
        columns=[*MOMENTUM_ASSETS, DEFENDER_CANDIDATE],
    )
    metrics.loc[:, "510300.SH"] = 2.0
    metrics.loc[:, "159915.SZ"] = 2.5
    policies = {
        "510300.SH": AssetRAQMThresholds(1.0, 0.0),
        "159915.SZ": AssetRAQMThresholds(2.0, 0.0),
        "513100.SH": None,
        "518880.SH": None,
    }

    state = all_asset_raqm_schedule(context, metrics, policies)

    # CSI300 clears by 1.0 while ChiNext clears by only 0.5.
    assert state.iloc[0]["target_candidate"] == "510300.SH"


def test_hard_five_day_hold_precedes_base_momentum_handoff() -> None:
    index = pd.date_range("2026-01-01", periods=7, freq="B")
    context = _context(
        index,
        [False, True, True, True, True, True, True],
        ["159915.SZ"] * len(index),
    )
    metrics = pd.DataFrame(
        0.0,
        index=index,
        columns=[*MOMENTUM_ASSETS, DEFENDER_CANDIDATE],
    )
    metrics.loc[:, "518880.SH"] = 3.0
    policies = {
        asset: AssetRAQMThresholds(2.0, 0.5)
        if asset == "518880.SH"
        else None
        for asset in MOMENTUM_ASSETS
    }

    state = all_asset_raqm_schedule(context, metrics, policies)

    assert state.iloc[0]["state_reason"] == "raqm_entry"
    assert state.iloc[4]["target_candidate"] == "518880.SH"
    assert state.iloc[5]["state_reason"] == "raqm_to_base_momentum_after_min_hold"
    assert state.iloc[5]["target_candidate"] == "159915.SZ"
