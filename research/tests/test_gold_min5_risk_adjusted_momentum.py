from __future__ import annotations

import numpy as np
import pandas as pd

from factors.risk_adjusted_quality_momentum import compute
from research.defender_curve_momentum import DEFENDER_CANDIDATE
from research.gold_min5_risk_adjusted_momentum import (
    GoldRAQMParams,
    risk_adjusted_momentum_at_open,
)
from research.momentum_defender_gold_override import GOLD_ASSET


def test_registered_risk_adjusted_momentum_is_used_and_shifted_one_open() -> None:
    index = pd.date_range("2026-01-01", periods=30, freq="B", name="date")
    curves = pd.DataFrame(
        {
            GOLD_ASSET: 100.0 + np.arange(len(index)) * 2.0,
            DEFENDER_CANDIDATE: 100.0 + np.arange(len(index)) * 0.5,
        },
        index=index,
    )
    result = risk_adjusted_momentum_at_open(curves)
    expected = compute(
        pd.DataFrame({"date": index, "close": curves[GOLD_ASSET].to_numpy()}),
        {"window": 20, "vol_floor_annual": 0.08},
    ).reindex(index).shift(1)

    pd.testing.assert_series_equal(
        result[GOLD_ASSET], expected, check_names=False, check_freq=False
    )
    assert np.isfinite(result["difference"].iloc[-1])


def test_threshold_validation_preserves_hysteresis() -> None:
    params = GoldRAQMParams(entry_difference=0.5, exit_difference=-0.2)
    assert params.candidate_id().startswith("risk_adjusted_quality_momentum_w20")
