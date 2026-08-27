from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from notification.peak_warning import evaluate_peak_warning


def _price_frame(signal_date: date) -> pd.DataFrame:
    index = pd.bdate_range(end=pd.Timestamp(signal_date), periods=221)
    close = np.full(len(index), 100.0)
    close[-1] = 120.0
    volume = np.full(len(index), 100.0)
    volume[-1] = 200.0
    return pd.DataFrame({"date": index, "close": close, "volume": volume})


def test_non_chinext_warning_uses_price_and_volume_only(monkeypatch) -> None:
    signal_date = date(2026, 8, 21)
    monkeypatch.setattr(
        "notification.peak_warning.query",
        lambda *_: _price_frame(signal_date),
    )

    warning = evaluate_peak_warning(
        "513100.SH",
        signal_date,
        share_loader=lambda *_: (_ for _ in ()).throw(
            AssertionError("share loader must not be called")
        ),
    )

    assert warning.triggered
    assert warning.price_breakout == pytest.approx(0.20)
    assert warning.price_return20 == pytest.approx(0.20)
    assert warning.volume_ratio20 == 2.0
    assert not warning.share_filter_required


def test_chinext_requires_positive_twenty_session_share_change(monkeypatch) -> None:
    signal_date = date(2026, 8, 21)
    price = _price_frame(signal_date)
    monkeypatch.setattr("notification.peak_warning.query", lambda *_: price)
    dates = pd.DatetimeIndex(price["date"])

    def shares(last_value: float) -> pd.Series:
        values = np.full(len(dates), 100.0)
        values[-1] = last_value
        return pd.Series(values, index=dates)

    increasing = evaluate_peak_warning(
        "159915.SZ",
        signal_date,
        share_loader=lambda *_: shares(110.0),
    )
    decreasing = evaluate_peak_warning(
        "159915.SZ",
        signal_date,
        share_loader=lambda *_: shares(90.0),
    )

    assert increasing.triggered
    assert increasing.share_flow20 == pytest.approx(0.10)
    assert not decreasing.triggered
    assert decreasing.share_flow20 == pytest.approx(-0.10)
    assert "持平或下降" in decreasing.reason


def test_chinext_missing_signal_date_share_suppresses_warning(monkeypatch) -> None:
    signal_date = date(2026, 8, 21)
    price = _price_frame(signal_date)
    monkeypatch.setattr("notification.peak_warning.query", lambda *_: price)
    shares = pd.Series(
        100.0,
        index=pd.DatetimeIndex(price["date"]).delete(-1),
    )

    warning = evaluate_peak_warning(
        "159915.SZ",
        signal_date,
        share_loader=lambda *_: shares,
    )

    assert not warning.triggered
    assert not warning.share_data_available
    assert "数据不可用" in warning.reason


def test_chinext_share_progress_is_loaded_before_price_conditions_pass(
    monkeypatch,
) -> None:
    signal_date = date(2026, 8, 21)
    price = _price_frame(signal_date)
    price.loc[price.index[-1], "close"] = 105.0
    price.loc[price.index[-1], "volume"] = 100.0
    monkeypatch.setattr("notification.peak_warning.query", lambda *_: price)
    dates = pd.DatetimeIndex(price["date"])
    values = np.full(len(dates), 100.0)
    values[-1] = 110.0
    called = []

    warning = evaluate_peak_warning(
        "159915.SZ",
        signal_date,
        share_loader=lambda *_: called.append(True)
        or pd.Series(values, index=dates),
    )

    assert called == [True]
    assert warning.share_data_available
    assert warning.share_flow20 == pytest.approx(0.10)
    assert not warning.triggered
