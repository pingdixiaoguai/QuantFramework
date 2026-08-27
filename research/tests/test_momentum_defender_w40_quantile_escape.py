from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from research.defender_curve_momentum import DEFENDER_CANDIDATE
from research.momentum_defender_occam import MOMENTUM_ASSETS
from research.momentum_defender_w40_quantile_escape import (
    QuantileXYPolicy,
    quantile_escape_schedule,
    rolling_quantiles_at_open,
)


def test_rolling_quantile_is_strictly_lagged() -> None:
    index = pd.bdate_range("2026-01-01", periods=5)
    metrics = pd.DataFrame(
        0.0, index=index, columns=[*MOMENTUM_ASSETS, DEFENDER_CANDIDATE]
    )
    metrics["518880.SH"] = [0.0, 0.1, 0.2, 0.3, 10.0]
    frames = rolling_quantiles_at_open(
        metrics, [0.5], history_window=4, min_history=2
    )
    assert frames["518880.SH", 0.5].iloc[-1] == pytest.approx(0.15)


def test_quantile_entry_line_uses_top1_a_minus_common_defender_c() -> None:
    index = pd.bdate_range("2026-01-01", periods=7)
    context = SimpleNamespace(
        calendar=index,
        initial_previous_candidate="510300.SH",
        momentum_target=pd.Series("518880.SH", index=index),
    )
    formal = pd.DataFrame(
        {"risk_on": False, "held_days_at_open": range(7)}, index=index
    )
    metrics = pd.DataFrame(
        0.0, index=index, columns=[*MOMENTUM_ASSETS, DEFENDER_CANDIDATE]
    )
    metrics["518880.SH"] = 0.05
    frames = {
        (asset, q): pd.Series(0.0, index=index)
        for asset in (*MOMENTUM_ASSETS, DEFENDER_CANDIDATE)
        for q in (0.2, 0.5, 0.7)
    }
    frames["518880.SH", 0.7] = pd.Series(0.03, index=index)
    frames["518880.SH", 0.2] = pd.Series(-0.01, index=index)
    frames[DEFENDER_CANDIDATE, 0.5] = pd.Series(0.01, index=index)
    policies = {asset: None for asset in MOMENTUM_ASSETS}
    policies["518880.SH"] = QuantileXYPolicy(0.7, 0.2)
    state = quantile_escape_schedule(
        context, formal, metrics, frames, 0.5, policies
    )
    assert state.iloc[5]["dynamic_entry_line_at_open"] == pytest.approx(0.02)
    assert state.iloc[5]["target_candidate"] == "518880.SH"
