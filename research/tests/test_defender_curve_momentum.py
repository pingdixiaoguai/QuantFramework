from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from factors.quality_momentum import compute as quality_momentum
from research.defender_curve_momentum import (
    ALL_CANDIDATES,
    DEFENDER_CANDIDATE,
    CurveMomentumParams,
    _simulate,
    score_close_curves,
)
from research.momentum_defender_occam import (
    ENTER_RETURN,
    ENTRY_COST,
    EXIT_COST,
    EXIT_RETURN,
    HELD_RETURN,
    INTERNAL_COST,
)


def test_defender_whole_nav_uses_exact_same_quality_momentum_function() -> None:
    index = pd.date_range("2026-01-01", periods=30, freq="B", name="date")
    curves = pd.DataFrame(index=index)
    for position, candidate in enumerate(ALL_CANDIDATES):
        curves[candidate] = 100.0 + np.arange(len(index)) * (position + 1)
    scores = score_close_curves(curves, window=20)
    expected = quality_momentum(
        pd.DataFrame(
            {
                "date": index,
                "close": curves[DEFENDER_CANDIDATE].to_numpy(float),
            }
        ),
        {"window": 20},
    )

    pd.testing.assert_series_equal(
        scores[DEFENDER_CANDIDATE], expected, check_names=False, check_freq=False
    )


def _interfaces(index: pd.DatetimeIndex) -> dict[str, pd.DataFrame]:
    result: dict[str, pd.DataFrame] = {}
    for candidate in ALL_CANDIDATES:
        result[candidate] = pd.DataFrame(
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
    return result


def test_direct_top1_uses_prior_close_scores_at_next_open() -> None:
    index = pd.date_range("2026-01-01", periods=5, freq="B")
    scores = pd.DataFrame(0.0, index=index, columns=ALL_CANDIDATES)
    scores.loc[index[0], "510300.SH"] = 1.0
    scores.loc[index[1], DEFENDER_CANDIDATE] = 2.0
    scores.loc[index[2], "518880.SH"] = 3.0
    scores.loc[index[3], "513100.SH"] = 4.0

    desired, daily = _simulate(
        scores,
        _interfaces(index),
        CurveMomentumParams(window=20, rebalance_days=1, start=date(2026, 1, 2)),
    )

    assert desired.tolist() == ["510300.SH", DEFENDER_CANDIDATE, "518880.SH", "513100.SH"]
    assert daily["candidate"].tolist() == desired.tolist()
    assert daily.iloc[0]["return"] == pytest.approx(0.002)
    assert daily.iloc[1]["return"] == pytest.approx(
        (1.0 + 0.003) * (1.0 + 0.002) - 1.0
    )


def test_five_day_constraint_keeps_actual_holding_but_not_raw_desired() -> None:
    index = pd.date_range("2026-01-01", periods=8, freq="B")
    scores = pd.DataFrame(0.0, index=index, columns=ALL_CANDIDATES)
    scores["510300.SH"] = 1.0
    scores.loc[index[1]:, DEFENDER_CANDIDATE] = 2.0

    desired, daily = _simulate(
        scores,
        _interfaces(index),
        CurveMomentumParams(window=20, rebalance_days=5, start=date(2026, 1, 2)),
    )

    assert desired.iloc[1] == DEFENDER_CANDIDATE
    assert daily.iloc[1]["candidate"] == "510300.SH"
    assert daily.iloc[5]["candidate"] == DEFENDER_CANDIDATE
