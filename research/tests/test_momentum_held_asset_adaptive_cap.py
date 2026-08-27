from __future__ import annotations

import pandas as pd

from research.run_momentum_held_asset_adaptive_cap import (
    AdaptiveCSpec,
    held_asset_cap_alert,
    select_candidate,
)


def test_asset_specific_cap_thresholds_follow_previous_close_asset() -> None:
    dates = pd.date_range("2026-01-05", periods=4, freq="B")
    caps = {
        "510300.SH": pd.Series([0.8, 0.8, 0.8, 0.8], index=dates),
        "159915.SZ": pd.Series([0.6, 0.6, 0.6, 0.6], index=dates),
        "513100.SH": pd.Series([0.4, 0.4, 0.4, 0.4], index=dates),
        "518880.SH": pd.Series([0.8, 0.8, 0.8, 0.8], index=dates),
    }
    previous = pd.Series(
        ["510300.SH", "159915.SZ", "513100.SH", "518880.SH"], index=dates
    )
    thresholds = {
        "510300.SH": 0.8,
        "159915.SZ": 0.4,
        "513100.SH": 0.4,
        "518880.SH": 0.6,
    }
    assert held_asset_cap_alert(caps, previous, thresholds).tolist() == [
        True,
        False,
        True,
        False,
    ]


def test_variant_id_records_every_asset_parameter() -> None:
    spec = AdaptiveCSpec(
        "C1",
        "asset caps",
        volatility_window=20,
        expanding_quantile=0.8,
        cap_510300=0.8,
        cap_159915=0.6,
        cap_513100=0.4,
        cap_518880=0.6,
    )
    assert spec.variant_id() == "C1_vw20_q0.80_c3000.80_cyb0.60_ndx0.40_au0.60"


def test_selection_requires_improvement_over_no_cap_and_real_activity() -> None:
    frame = pd.DataFrame(
        {
            "scheme": ["C1", "C1", "C1"],
            "variant_id": ["sacrifices_return", "dead", "balanced"],
            "development_2019_2022_annualized_delta_vs_no_cap": [-0.10, 0.02, 0.01],
            "development_2019_2022_sharpe_delta_vs_no_cap": [0.50, 0.30, 0.10],
            "development_2019_2022_max_drawdown_improvement_vs_no_cap": [0.10, 0.05, 0.02],
            "development_2019_2022_emergency_entries": [2, 0, 1],
            "development_2019_2022_switches": [5, 3, 4],
            "switches": [5, 3, 4],
        }
    )
    selected = select_candidate(frame, "C1", "development_2019_2022")
    assert selected["variant_id"] == "balanced"
    assert selected["selection_pool"] == "beats_no_cap_and_active"
