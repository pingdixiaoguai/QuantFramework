from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from research.defender_curve_momentum import DEFENDER_CANDIDATE
from research.gold_min5_risk_adjusted_escape import (
    GoldMin5Params,
    gold_min5_schedule,
)
from research.momentum_defender_gold_override import GOLD_ASSET


def _context(index: pd.DatetimeIndex, risk_on: list[bool]):
    return SimpleNamespace(
        calendar=index,
        integrated=SimpleNamespace(
            result=SimpleNamespace(
                state=pd.DataFrame({"risk_on": risk_on}, index=index)
            )
        ),
        momentum_target=pd.Series("513100.SH", index=index),
    )


def _metrics(index: pd.DatetimeIndex, differences: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            GOLD_ASSET: differences,
            DEFENDER_CANDIDATE: 0.0,
            "difference": differences,
        },
        index=index,
    )


def test_base_momentum_cannot_break_first_five_gold_sessions() -> None:
    index = pd.date_range("2026-01-01", periods=7, freq="B")
    context = _context(index, [False, True, True, True, True, True, True])
    params = GoldMin5Params(entry_difference=0.5, exit_difference=-0.2)

    state = gold_min5_schedule(
        context, _metrics(index, [0.6] * 7), params
    )

    assert state.iloc[:5]["target_candidate"].eq(GOLD_ASSET).all()
    assert state.iloc[5]["target_candidate"] == "513100.SH"
    assert state.iloc[5]["state_reason"] == "gold_to_momentum_after_min_hold"


def test_after_five_days_difference_controls_return_to_defender() -> None:
    index = pd.date_range("2026-01-01", periods=7, freq="B")
    context = _context(index, [False] * 7)
    params = GoldMin5Params(entry_difference=0.5, exit_difference=-0.2)
    differences = [0.6, -0.3, -0.3, -0.3, -0.3, -0.3, -0.3]

    state = gold_min5_schedule(context, _metrics(index, differences), params)

    assert state.iloc[:5]["target_candidate"].eq(GOLD_ASSET).all()
    assert state.iloc[5]["target_candidate"] == DEFENDER_CANDIDATE
    assert state.iloc[5]["state_reason"] == "gold_to_defender_after_min_hold"
