from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.momentum_defender_w40_loss_gate import (
    W40LossGateSpec,
    downside_log_loss,
    w40_loss_percentile_at_open,
)


def test_downside_log_loss_is_zero_for_gains_and_positive_for_losses() -> None:
    gain = pd.Series(np.exp(np.arange(6) * 0.01))
    loss = pd.Series(np.exp(-np.arange(6) * 0.01))
    assert downside_log_loss(gain, 5).iloc[-1] == 0.0
    assert downside_log_loss(loss, 5).iloc[-1] == pytest.approx(0.05)


def test_percentile_uses_previous_close_and_prior_history_only() -> None:
    dates = pd.bdate_range("2024-01-01", periods=46)
    close = pd.Series(np.exp(-np.arange(46) * 0.001), index=dates)
    calendar = pd.DatetimeIndex([dates[44], dates[45]])
    raw, percentile = w40_loss_percentile_at_open(
        close, calendar, history_window=3, min_history=1
    )
    expected = downside_log_loss(close, 40)
    assert raw.iloc[0] == pytest.approx(expected.loc[dates[43]])
    assert raw.iloc[1] == pytest.approx(expected.loc[dates[44]])
    assert percentile.notna().all()


def test_zero_loss_maps_to_zero_percentile() -> None:
    dates = pd.bdate_range("2024-01-01", periods=46)
    close = pd.Series(np.exp(np.arange(46) * 0.001), index=dates)
    _, percentile = w40_loss_percentile_at_open(
        close,
        pd.DatetimeIndex([dates[-1]]),
        history_window=3,
        min_history=1,
    )
    assert percentile.iloc[0] == 0.0


def test_locks_are_five_day_multiples_between_20_and_30() -> None:
    with pytest.raises(ValueError, match="five-day multiple"):
        W40LossGateSpec(0.55, 0.20, 3, 1, 22, 30)
