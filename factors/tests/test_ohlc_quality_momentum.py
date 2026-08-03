"""Tests for the registered post-adjusted OHLC ER factor."""

from __future__ import annotations

import numpy as np
import pandas as pd

from factors.ohlc_quality_momentum import compute


def test_weighted_ohlc_er_matches_formula():
    dates = pd.bdate_range("2026-01-01", periods=25)
    close = pd.Series(np.arange(100.0, 125.0))
    frame = pd.DataFrame(
        {
            "date": dates,
            "open": close - 0.5,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": 1_000.0,
        }
    )
    params = {
        "window": 20,
        "weights": {"close": 1.0, "gap": 0.0, "body": 0.0, "range": 0.0},
    }

    result = compute(frame, params)

    # With close-only weights this is exactly the existing close-only ER
    # multiplied by 20-day momentum.
    momentum = frame["close"].pct_change(20)
    er = (frame["close"] - frame["close"].shift(20)).abs() / frame[
        "close"
    ].diff().abs().rolling(20).sum()
    expected = momentum * er
    assert result.iloc[-1] == expected.iloc[-1]
    assert result.index.equals(pd.Index(frame["date"]))
