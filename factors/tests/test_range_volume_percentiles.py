"""Tests for historical range-position and volume percentile factors."""

import pandas as pd
import pytest

from factors.drawdown_percentile import compute as compute_drawdown
from factors.rebound_percentile import compute as compute_rebound
from factors.volume_percentile import compute as compute_volume


def _make_df(prices: list[float], volumes: list[float] | None = None) -> pd.DataFrame:
    volume = volumes if volumes is not None else [1000.0] * len(prices)
    return pd.DataFrame({
        "date": pd.bdate_range("2024-01-01", periods=len(prices)),
        "open": prices,
        "high": prices,
        "low": prices,
        "close": prices,
        "volume": volume,
    })


@pytest.mark.parametrize(
    "compute",
    [compute_drawdown, compute_rebound, compute_volume],
)
def test_first_value_requires_window_plus_history_minus_one_rows(compute) -> None:
    df = _make_df(
        [100.0 + index for index in range(12)],
        [1000.0 + index * 10.0 for index in range(12)],
    )
    result = compute(df, {"window": 3, "history": 4})

    assert result.iloc[:5].isna().all()
    assert result.iloc[5:].notna().all()
    assert result.dropna().between(0.0, 1.0).all()
    assert result.index.equals(pd.Index(df["date"]))


def test_drawdown_percentile_uses_only_trailing_values() -> None:
    df = _make_df([10.0, 12.0, 11.0, 9.0, 10.0, 8.0])
    result = compute_drawdown(df, {"window": 2, "history": 3})

    # Last three drawdowns are 2/11, 0, and 2/10; the latest is the largest.
    assert result.iloc[-1] == pytest.approx(1.0)


def test_rebound_percentile_uses_only_trailing_values() -> None:
    df = _make_df([10.0, 8.0, 9.0, 12.0, 11.0, 13.0])
    result = compute_rebound(df, {"window": 2, "history": 3})

    # Last three rebounds are 1/3, 0, and 2/11; the latest ranks in the middle.
    assert result.iloc[-1] == pytest.approx(2.0 / 3.0)


def test_volume_percentile_ranks_relative_volume_not_absolute_volume() -> None:
    df = _make_df(
        [10.0] * 6,
        [100.0, 100.0, 200.0, 100.0, 100.0, 300.0],
    )
    result = compute_volume(df, {"window": 2, "history": 3})

    assert result.iloc[-1] == pytest.approx(1.0)


@pytest.mark.parametrize(
    "compute",
    [compute_drawdown, compute_rebound, compute_volume],
)
def test_future_rows_do_not_change_existing_values(compute) -> None:
    original = _make_df(
        [10.0, 12.0, 11.0, 9.0, 10.0, 8.0, 13.0],
        [100.0, 110.0, 120.0, 90.0, 130.0, 80.0, 200.0],
    )
    extended = _make_df(
        [10.0, 12.0, 11.0, 9.0, 10.0, 8.0, 13.0, 1_000.0],
        [100.0, 110.0, 120.0, 90.0, 130.0, 80.0, 200.0, 1_000_000.0],
    )

    before_future_row = compute(original, {"window": 2, "history": 3})
    after_future_row = compute(extended, {"window": 2, "history": 3}).iloc[:-1]

    pd.testing.assert_series_equal(before_future_row, after_future_row)


@pytest.mark.parametrize(
    "compute",
    [compute_drawdown, compute_rebound, compute_volume],
)
def test_window_and_history_must_be_positive(compute) -> None:
    with pytest.raises(ValueError, match="window and history must be >= 1"):
        compute(_make_df([10.0, 11.0]), {"window": 0, "history": 2})
