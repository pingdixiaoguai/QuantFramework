import numpy as np
import pandas as pd
import pytest

from research.run_momentum_held_asset_c2_rolling_500_quantile import (
    rolling_volatility_cap,
)


def test_rolling_threshold_uses_only_prior_500_observations() -> None:
    index = pd.date_range("2020-01-01", periods=520)
    volatility = pd.Series(np.arange(1.0, 521.0), index=index)

    result = rolling_volatility_cap(
        volatility,
        0.70,
        max_history=500,
        min_history=20,
    )

    expected = volatility.iloc[19:519].quantile(0.70)
    assert result.at[index[519], "threshold"] == pytest.approx(expected)


def test_current_observation_does_not_enter_its_own_threshold() -> None:
    index = pd.date_range("2020-01-01", periods=30)
    normal = pd.Series(np.ones(30), index=index)
    shocked = normal.copy()
    shocked.iloc[-1] = 1000.0

    normal_result = rolling_volatility_cap(normal, 0.90, min_history=20)
    shocked_result = rolling_volatility_cap(shocked, 0.90, min_history=20)

    assert shocked_result.iloc[-1]["threshold"] == pytest.approx(
        normal_result.iloc[-1]["threshold"]
    )
    assert shocked_result.iloc[-1]["cap"] < normal_result.iloc[-1]["cap"]
