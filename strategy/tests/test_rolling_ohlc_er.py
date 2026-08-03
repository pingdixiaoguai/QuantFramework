"""Tests for the research-only quarterly rolling OHLC ER search."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from strategy.rolling_ohlc_er import (
    _candidate_weights,
    _softmax_confidence,
    build_signal_comparison,
)


def _frame(scale: float, daily_return: float) -> pd.DataFrame:
    dates = pd.bdate_range("2026-06-01", periods=41)
    close = scale * np.power(1 + daily_return, np.arange(len(dates)))
    open_price = close * 0.998
    return pd.DataFrame(
        {
            "date": dates,
            "open": open_price,
            "high": np.maximum(open_price, close) * 1.005,
            "low": np.minimum(open_price, close) * 0.995,
            "close": close,
            "volume": 1_000.0,
        }
    )


def _config() -> dict:
    return {
        "window": 20,
        "history_days": 1008,
        "search_radius": 0.05,
        "search_step": 0.01,
        "top_k": 10,
        "min_hold_days": 5,
        "transaction_cost_rate": 0.0001,
        "seed": {
            "effective_date": "2026-07-01",
            "training_start": "2022-05-05",
            "training_end": "2026-06-30",
            "weights": {
                "close": 0.853,
                "gap": 0.337,
                "body": 0.029,
                "range": 0.281,
            },
        },
    }


def test_candidate_grid_enforces_er_bounds():
    candidates = _candidate_weights(
        np.asarray([0.8, 0.2, 0.03, 0.19]),
        0.05,
        0.01,
    )
    assert len(candidates) > 0
    assert (candidates >= 0).all()
    assert (candidates[:, 0] + candidates[:, 1] >= 1).all()
    assert (
        candidates[:, 0] + candidates[:, 2] + candidates[:, 3] >= 1
    ).all()


def test_softmax_confidence_sums_to_one_and_preserves_rank():
    result = _softmax_confidence({"A": -0.02, "B": 0.01, "C": 0.03, "D": 0.00})
    assert sum(result.values()) == pytest.approx(1.0)
    assert max(result, key=result.get) == "C"
    assert min(result, key=result.get) == "A"


def test_equal_scores_have_equal_confidence():
    result = _softmax_confidence({"A": 0.1, "B": 0.1, "C": 0.1, "D": 0.1})
    assert set(result.values()) == {0.25}


def test_build_comparison_reports_old_and_new_without_rolling_seed(tmp_path):
    assets = ["A", "B", "C", "D"]
    frames = {
        "A": _frame(100, 0.001),
        "B": _frame(110, 0.002),
        "C": _frame(120, -0.001),
        "D": _frame(130, 0.0005),
    }
    result = build_signal_comparison(
        frames,
        assets,
        date(2026, 7, 27),
        "test",
        _config(),
        tmp_path,
    )
    assert result.weights.values == (0.853, 0.337, 0.029, 0.281)
    assert set(result.assets) == set(assets)
    for values in result.assets.values():
        assert {
            "momentum",
            "old_er",
            "old_score",
            "old_confidence",
            "new_er",
            "new_score",
            "new_confidence",
        } == set(values)
    assert sum(v["old_confidence"] for v in result.assets.values()) == pytest.approx(1)
    assert sum(v["new_confidence"] for v in result.assets.values()) == pytest.approx(1)
