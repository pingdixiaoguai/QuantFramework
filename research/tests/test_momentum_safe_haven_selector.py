from __future__ import annotations

import pandas as pd

from research.momentum_safe_haven_selector import (
    SelectorParams,
    risk_indicator,
    risk_on_at_open,
    safe_haven_at_open,
)


def test_risk_signal_has_one_day_lag_and_no_holding_lock() -> None:
    dates = pd.date_range("2026-01-01", periods=5, freq="D")
    close = pd.Series([100.0, 90.0, 110.0, 90.0, 110.0], index=dates)

    states = risk_on_at_open(
        close,
        SelectorParams((1,), 0.0, "quality_momentum", 20),
    )

    # Signals on days 2–4 are off/on/off and are acted on at the following
    # opens.  Consecutive reversals prove that no minimum-holding lock exists.
    assert states.tolist() == [True, True, False, True, False]


def test_multi_window_indicator_requires_all_horizons() -> None:
    dates = pd.date_range("2026-01-01", periods=5, freq="D")
    close = pd.Series([100.0, 101.0, 102.0, 103.0, 104.0], index=dates)

    indicator = risk_indicator(close, (1, 3))

    assert indicator.iloc[:3].isna().all()
    expected = ((103.0 / 102.0 - 1.0) + (103.0 / 100.0 - 1.0)) / 2.0
    assert indicator.iloc[3] == expected


def test_safe_haven_choice_uses_only_previous_close_scores() -> None:
    dates = pd.date_range("2026-01-01", periods=3, freq="D")
    scores = pd.DataFrame(
        {
            "defender": [0.1, 0.0, 0.0],
            "gold": [0.3, 0.1, 0.0],
            "nasdaq": [0.2, 0.5, 0.0],
        },
        index=dates,
    )

    choice = safe_haven_at_open(scores)

    assert choice.tolist() == ["defender", "gold", "nasdaq"]
