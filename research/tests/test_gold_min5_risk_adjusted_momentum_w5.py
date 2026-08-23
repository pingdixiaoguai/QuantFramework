from __future__ import annotations

import numpy as np
import pandas as pd

from factors.risk_adjusted_quality_momentum import compute
from research.defender_curve_momentum import DEFENDER_CANDIDATE
from research.gold_min5_risk_adjusted_momentum import (
    risk_adjusted_momentum_at_open,
)
from research.gold_min5_risk_adjusted_momentum_w5 import GoldRAQMW5Params
from research.momentum_defender_gold_override import GOLD_ASSET


def test_registered_five_day_factor_is_used_with_next_open_shift() -> None:
    index = pd.date_range("2026-01-01", periods=20, freq="B", name="date")
    curves = pd.DataFrame(
        {
            GOLD_ASSET: 100.0 * np.exp(np.arange(len(index)) * 0.01),
            DEFENDER_CANDIDATE: 100.0 * np.exp(np.arange(len(index)) * 0.002),
        },
        index=index,
    )
    result = risk_adjusted_momentum_at_open(curves, window=5)
    expected = compute(
        pd.DataFrame({"date": index, "close": curves[GOLD_ASSET].to_numpy()}),
        {"window": 5, "vol_floor_annual": 0.08},
    ).reindex(index).shift(1)

    pd.testing.assert_series_equal(
        result[GOLD_ASSET], expected, check_names=False, check_freq=False
    )


def test_w5_candidate_id_and_threshold_validation() -> None:
    params = GoldRAQMW5Params(1.0, -0.5)
    assert "_w5_" in params.candidate_id()
