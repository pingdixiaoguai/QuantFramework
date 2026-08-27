from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.run_momentum_volatility_signal_abcd import (
    AlertSpec,
    asof_previous_close,
    choose_by_asset,
    expanding_volatility_cap,
    rogers_satchell_volatility,
    select_candidate,
)


def test_rs_volatility_uses_only_current_and_prior_window_observations() -> None:
    dates = pd.date_range("2026-01-01", periods=4, freq="D")
    prices = pd.DataFrame(
        {
            "open": [100.0, 100.0, 100.0, 100.0],
            "high": [101.0, 102.0, 103.0, 104.0],
            "low": [99.0, 98.0, 97.0, 96.0],
            "close": [100.0, 100.0, 100.0, 100.0],
        },
        index=dates,
    )
    measured = rogers_satchell_volatility(prices, 2)
    daily_variance = (
        np.log(prices["high"] / prices["close"])
        * np.log(prices["high"] / prices["open"])
        + np.log(prices["low"] / prices["close"])
        * np.log(prices["low"] / prices["open"])
    ).clip(lower=0.0)
    expected = np.sqrt(252.0 * daily_variance.iloc[:2].mean())
    assert np.isnan(measured.iloc[0])
    assert measured.iloc[1] == pytest.approx(expected)


def test_cap_threshold_is_strictly_lagged_and_quantized_down() -> None:
    dates = pd.date_range("2026-01-01", periods=4, freq="D")
    volatility = pd.Series([0.10, 0.20, 0.40, 0.16], index=dates)
    result = expanding_volatility_cap(
        volatility,
        0.50,
        min_history=2,
        step=0.20,
    )
    assert np.isnan(result.loc[dates[1], "threshold"])
    assert result.loc[dates[2], "threshold"] == pytest.approx(0.15)
    assert result.loc[dates[2], "raw_cap"] == pytest.approx(0.375)
    assert result.loc[dates[2], "cap"] == pytest.approx(0.20)


def test_previous_close_mapping_excludes_same_day_close_and_carries_suspension() -> None:
    source = pd.Series(
        [0.8, 0.6],
        index=pd.to_datetime(["2026-01-02", "2026-01-06"]),
    )
    calendar = pd.to_datetime(["2026-01-05", "2026-01-06", "2026-01-07"])
    mapped = asof_previous_close(source, calendar)
    assert mapped.tolist() == pytest.approx([0.8, 0.8, 0.6])


def test_choose_by_asset_uses_previous_close_asset_signal() -> None:
    dates = pd.date_range("2026-01-05", periods=3, freq="B")
    chosen = choose_by_asset(
        {
            "A": pd.Series([0.8, 0.6, 0.4], index=dates),
            "B": pd.Series([0.2, 0.4, 0.6], index=dates),
        },
        pd.Series(["A", "B", "A"], index=dates),
    )
    assert chosen.tolist() == pytest.approx([0.8, 0.4, 0.4])


def test_variant_id_records_all_tuned_dimensions() -> None:
    spec = AlertSpec("D", "confirmed", 20, 0.80, 0.60, 10, -0.02)
    assert spec.variant_id() == "D_vw20_q0.80_cap0.6_dw10_dtm0.02"


def test_selection_rejects_dead_signal_even_when_its_sharpe_is_higher() -> None:
    grid = pd.DataFrame(
        {
            "scheme": ["B", "B"],
            "variant_id": ["dead", "active"],
            "development_2019_2022_annualized_delta": [0.20, 0.10],
            "development_2019_2022_sharpe_delta": [2.0, 1.0],
            "development_2019_2022_max_drawdown_improvement": [0.10, 0.05],
            "development_2019_2022_emergency_entries": [0, 1],
            "development_2019_2022_switches": [10, 11],
            "development_2019_2022_alert_days": [0, 3],
            "switches": [10, 11],
            "alert_days": [0, 3],
        }
    )
    selected = select_candidate(grid, "B", "development_2019_2022")
    assert selected["variant_id"] == "active"
    assert selected["selection_pool"] == "metric_and_activity"
