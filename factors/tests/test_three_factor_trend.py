"""Tests for factors.three_factor_trend."""

import numpy as np
import pandas as pd
import pytest

from factors.three_factor_trend import METADATA, compute
from factors.validator import validate


def _make_df(prices: list[float]) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=len(prices))
    return pd.DataFrame(
        {
            "date": dates,
            "open": prices,
            "high": [price * 1.01 for price in prices],
            "low": [price * 0.99 for price in prices],
            "close": prices,
            "volume": [1000.0] * len(prices),
        }
    )


def _prices(size: int = 140) -> list[float]:
    returns = [
        0.001 + 0.005 * np.sin(i / 4.0) + 0.002 * np.cos(i / 11.0)
        for i in range(size - 1)
    ]
    prices = [100.0]
    for daily_return in returns:
        prices.append(prices[-1] * (1.0 + daily_return))
    return prices


def _manual_last_score(prices: list[float]) -> float:
    close = pd.Series(prices, dtype=float)
    momentum = close.iloc[-1] / close.iloc[-21] - 1.0
    displacement = abs(close.iloc[-1] - close.iloc[-21])
    efficiency = displacement / close.diff().abs().iloc[-20:].sum()

    y = np.log(close.iloc[-10:].to_numpy())
    x = np.arange(10, dtype=float)
    linearity = float(np.corrcoef(x, y)[0, 1] ** 2)

    volatility = close.pct_change(fill_method=None).rolling(20).std(ddof=1)
    threshold = volatility.shift(1).iloc[-252:].dropna().quantile(0.80)
    multiplier = np.clip(
        np.exp(1.0 - volatility.iloc[-1] / threshold),
        0.25,
        2.0,
    )
    return momentum * efficiency**0.75 * (0.5 + linearity) * multiplier**0.50


def test_matches_selected_plateau_formula() -> None:
    prices = _prices()
    result = compute(_make_df(prices))

    assert result.iloc[-1] == pytest.approx(_manual_last_score(prices))


def test_default_output_obeys_factor_contract() -> None:
    frame = _make_df(_prices())
    result = compute(frame)

    validate(result, frame, METADATA)
    assert result.iloc[:20].isna().all()
    assert result.iloc[20:].notna().all()


def test_future_price_does_not_change_existing_scores() -> None:
    prices = _prices()
    original = compute(_make_df(prices))
    extended = compute(_make_df([*prices, 1000.0])).iloc[:-1]

    pd.testing.assert_series_equal(original, extended)


def test_signed_momentum_is_preserved() -> None:
    rising = compute(_make_df([100.0 + i for i in range(100)])).iloc[-1]
    falling = compute(_make_df([150.0 - 0.5 * i for i in range(100)])).iloc[-1]

    assert rising > 0.0
    assert falling < 0.0


def test_flat_series_is_valid_and_scores_zero() -> None:
    frame = _make_df([100.0] * 100)
    result = compute(frame)

    validate(result, frame, METADATA)
    assert result.iloc[-1] == 0.0


def test_momentum_window_is_fixed_at_twenty() -> None:
    with pytest.raises(ValueError, match="fixed at 20"):
        compute(_make_df(_prices()), {"momentum_window": 15})


def test_hyperparameter_override_is_supported() -> None:
    default = compute(_make_df(_prices()))
    adjusted = compute(
        _make_df(_prices()),
        {
            "er_power": 1.0,
            "low_vol_shape": "cap",
            "low_vol_power": 1.0,
        },
    )

    assert default.iloc[-1] != pytest.approx(adjusted.iloc[-1])
