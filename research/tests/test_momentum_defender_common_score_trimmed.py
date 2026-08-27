from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.momentum_defender_common_score_trimmed import (
    ExtremeBlockSpec,
    build_extreme_block_mask,
    validate_common_score_policies,
    volatility_adjusted_absolute_return,
)
from research.momentum_defender_downside_raqm import FactorProfile
from research.momentum_defender_selected_asset_draqm import AssetDRAQMPolicy


def _policy(asset: str, profile: FactorProfile) -> AssetDRAQMPolicy:
    return AssetDRAQMPolicy(asset, profile, 0.6, 0.2, 1, 1)


def test_common_score_rejects_different_horizons_or_weights() -> None:
    with pytest.raises(ValueError, match="identical score"):
        validate_common_score_policies(
            {
                "510300.SH": _policy(
                    "510300.SH", FactorProfile("csi", (30, 40), (0.25, 0.75))
                ),
                "518880.SH": _policy(
                    "518880.SH", FactorProfile("gold", (20, 40), (0.25, 0.75))
                ),
            }
        )


def test_common_score_accepts_same_formula_with_different_thresholds() -> None:
    profile_csi = FactorProfile("csi", (30, 40), (0.25, 0.75))
    profile_gold = FactorProfile("gold", (30, 40), (0.25, 0.75))
    policies = {
        "510300.SH": AssetDRAQMPolicy(
            "510300.SH", profile_csi, 0.5, 0.2, 1, 1
        ),
        "518880.SH": AssetDRAQMPolicy(
            "518880.SH", profile_gold, 0.8, 0.1, 5, 3
        ),
    }
    validate_common_score_policies(policies)


def test_shock_score_is_symmetric_for_rise_and_fall() -> None:
    up = pd.Series(np.exp(np.linspace(0.0, 0.4, 61)))
    down = pd.Series(np.exp(np.linspace(0.0, -0.4, 61)))
    up_score = volatility_adjusted_absolute_return(
        up, return_window=5, volatility_window=20, volatility_floor_annual=0.08
    )
    down_score = volatility_adjusted_absolute_return(
        down, return_window=5, volatility_window=20, volatility_floor_annual=0.08
    )
    assert np.isclose(up_score.iloc[-1], down_score.iloc[-1])


def test_exactly_top_ten_percent_fixed_blocks_are_excluded() -> None:
    index = pd.date_range("2024-01-01", periods=200, freq="B")
    base = np.exp(np.linspace(0.0, 0.2, len(index)))
    shocked = base.copy()
    shocked[100:105] *= np.linspace(1.0, 1.5, 5)
    closes = {
        "510300.SH": pd.Series(shocked, index=index),
        "518880.SH": pd.Series(base, index=index),
    }
    result = build_extreme_block_mask(
        closes,
        index[30:],
        ExtremeBlockSpec(block_length_sessions=20, excluded_block_fraction=0.10),
    )
    assert len(result.blocks) == 9
    assert int(result.blocks["excluded_from_selection"].sum()) == 1
    assert int((~result.selection_mask).sum()) == 20
    excluded = result.blocks.index[result.blocks["excluded_from_selection"]][0]
    assert result.blocks.loc[excluded, "start"] <= index[104]
    assert result.blocks.loc[excluded, "end"] >= index[100]


def test_trim_mask_depends_only_on_prices_not_candidate_returns() -> None:
    index = pd.date_range("2024-01-01", periods=100, freq="B")
    close = pd.Series(np.exp(np.sin(np.arange(100) / 10.0) * 0.1), index=index)
    first = build_extreme_block_mask(
        {"510300.SH": close, "518880.SH": close * 2.0},
        index[25:],
        ExtremeBlockSpec(),
    )
    second = build_extreme_block_mask(
        {"510300.SH": close, "518880.SH": close * 2.0},
        index[25:],
        ExtremeBlockSpec(),
    )
    pd.testing.assert_series_equal(first.selection_mask, second.selection_mask)


def test_raw_absolute_mode_does_not_downweight_a_high_volatility_crash() -> None:
    index = pd.date_range("2024-01-01", periods=120, freq="B")
    calm = np.ones(120)
    crash = calm.copy()
    crash[60:66] = np.linspace(1.0, 0.70, 6)
    crash[66:] = 0.70
    result = build_extreme_block_mask(
        {
            "510300.SH": pd.Series(crash, index=index),
            "518880.SH": pd.Series(calm, index=index),
        },
        index[20:],
        ExtremeBlockSpec(
            block_length_sessions=20,
            excluded_block_fraction=0.10,
            normalization_mode="raw_absolute_log_return",
        ),
    )
    excluded = result.blocks.loc[result.blocks["excluded_from_selection"]].iloc[0]
    assert excluded["start"] <= index[65]
    assert excluded["end"] >= index[60]
