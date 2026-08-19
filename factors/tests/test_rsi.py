"""Tests for factors.rsi."""

import numpy as np
import pandas as pd
import pytest

from factors.rsi import METADATA, compute


def _make_df(prices: list[float]) -> pd.DataFrame:
    return pd.DataFrame({
        "date": pd.bdate_range("2024-01-01", periods=len(prices)),
        "open": prices,
        "high": prices,
        "low": prices,
        "close": prices,
        "volume": [1000.0] * len(prices),
    })


def test_metadata_describes_rsi14() -> None:
    assert METADATA["name"] == "rsi"
    assert METADATA["params"] == {"window": 14}
    assert METADATA["min_history"] == 15
    assert METADATA["direction"] == "higher_better"


def test_initial_rsi_uses_fourteen_price_changes() -> None:
    # Seven +2 moves and seven -1 moves: avg_gain=1, avg_loss=0.5.
    moves = [2.0, -1.0] * 7
    prices = [100.0]
    for move in moves:
        prices.append(prices[-1] + move)

    result = compute(_make_df(prices))

    assert result.iloc[:14].isna().all()
    assert result.iloc[14] == pytest.approx(100.0 - 100.0 / 3.0)


def test_wilder_recursive_update() -> None:
    moves = [2.0, -1.0] * 7 + [2.0]
    prices = [100.0]
    for move in moves:
        prices.append(prices[-1] + move)

    result = compute(_make_df(prices))
    next_gain = (1.0 * 13.0 + 2.0) / 14.0
    next_loss = (0.5 * 13.0) / 14.0
    expected = 100.0 - 100.0 / (1.0 + next_gain / next_loss)

    assert result.iloc[15] == pytest.approx(expected)


@pytest.mark.parametrize(
    ("prices", "expected"),
    [
        ([100.0 + i for i in range(20)], 100.0),
        ([100.0 - i for i in range(20)], 0.0),
        ([100.0] * 20, 50.0),
    ],
)
def test_directional_and_flat_boundaries(
    prices: list[float], expected: float
) -> None:
    result = compute(_make_df(prices))
    assert result.iloc[14:].notna().all()
    assert np.isfinite(result.iloc[14:]).all()
    assert result.iloc[-1] == pytest.approx(expected)


def test_window_must_be_positive() -> None:
    with pytest.raises(ValueError, match="window must be >= 1"):
        compute(_make_df([100.0, 101.0]), {"window": 0})
