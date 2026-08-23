from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from research.defender_curve_momentum import DEFENDER_CANDIDATE
from research.momentum_defender_gold_override import (
    GOLD_ASSET,
    GoldOverrideParams,
    gold_override_schedule,
    metric_at_open,
    simulate_candidate_schedule,
)
from research.momentum_defender_occam import (
    ENTER_RETURN,
    ENTRY_COST,
    EXIT_COST,
    EXIT_RETURN,
    HELD_RETURN,
    INTERNAL_COST,
)


def test_return_metric_becomes_available_only_at_following_open() -> None:
    index = pd.date_range("2026-01-01", periods=4, freq="B")
    curves = pd.DataFrame(
        {
            GOLD_ASSET: [100.0, 110.0, 121.0, 133.1],
            DEFENDER_CANDIDATE: [100.0, 101.0, 102.01, 103.0301],
        },
        index=index,
    )

    result = metric_at_open(curves, "return", window=1)

    assert pd.isna(result.iloc[0]["difference"])
    assert pd.isna(result.iloc[1]["difference"])
    assert result.iloc[2]["difference"] == pytest.approx(0.09)


def test_gold_override_enters_inside_defender_and_base_momentum_forces_exit() -> None:
    index = pd.date_range("2026-01-01", periods=5, freq="B")
    base_state = pd.DataFrame(
        {"risk_on": [False, False, False, True, False]}, index=index
    )
    context = SimpleNamespace(
        calendar=index,
        integrated=SimpleNamespace(result=SimpleNamespace(state=base_state)),
        momentum_target=pd.Series("510300.SH", index=index),
    )
    metrics = pd.DataFrame(
        {
            GOLD_ASSET: [1.0] * 5,
            DEFENDER_CANDIDATE: [0.0] * 5,
            "difference": [0.2, 0.2, -0.2, 0.2, 0.2],
        },
        index=index,
    )
    params = GoldOverrideParams(
        metric="return",
        window=20,
        entry_threshold=0.1,
        exit_threshold=-0.1,
        min_gold_hold_days=5,
    )

    state = gold_override_schedule(context, metrics, params)

    assert state.iloc[0]["target_candidate"] == GOLD_ASSET
    # Negative difference cannot end the override before its own five-day hold.
    assert state.iloc[2]["target_candidate"] == GOLD_ASSET
    # Base C2 Momentum always takes precedence and breaks the Gold hold.
    assert state.iloc[3]["target_candidate"] == "510300.SH"
    assert state.iloc[4]["target_candidate"] == GOLD_ASSET


def _interfaces(index: pd.DatetimeIndex):
    return {
        candidate: pd.DataFrame(
            {
                HELD_RETURN: 0.001,
                ENTER_RETURN: 0.002,
                EXIT_RETURN: 0.003,
                INTERNAL_COST: 0.0,
                ENTRY_COST: 0.0001,
                EXIT_COST: 0.0001,
            },
            index=index,
        )
        for candidate in ("510300.SH", GOLD_ASSET, DEFENDER_CANDIDATE)
    }


def test_candidate_schedule_chains_exit_and_entry_legs() -> None:
    index = pd.date_range("2026-01-01", periods=3, freq="B")
    target = pd.Series(
        ["510300.SH", GOLD_ASSET, DEFENDER_CANDIDATE], index=index
    )

    daily = simulate_candidate_schedule(
        target, _interfaces(index), initial_previous_candidate="510300.SH"
    )

    assert daily.iloc[0]["return"] == pytest.approx(0.001)
    expected = (1.0 + 0.003) * (1.0 + 0.002) - 1.0
    assert daily.iloc[1]["return"] == pytest.approx(expected)
    assert daily.iloc[2]["return"] == pytest.approx(expected)
