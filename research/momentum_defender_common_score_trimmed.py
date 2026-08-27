"""Candidate-independent extreme-block trimming for common-score research."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from research.momentum_defender_selected_asset_draqm import AssetDRAQMPolicy
from research.momentum_volatility import asof_previous_close


@dataclass(frozen=True)
class ExtremeBlockSpec:
    shock_return_window: int = 5
    shock_volatility_window: int = 20
    volatility_floor_annual: float = 0.08
    block_length_sessions: int = 20
    excluded_block_fraction: float = 0.10
    normalization_mode: str = "volatility_adjusted"

    def __post_init__(self) -> None:
        if min(
            self.shock_return_window,
            self.shock_volatility_window,
            self.block_length_sessions,
        ) < 1:
            raise ValueError("shock and block windows must be positive")
        if self.volatility_floor_annual < 0.0:
            raise ValueError("volatility floor cannot be negative")
        if not 0.0 < self.excluded_block_fraction < 0.5:
            raise ValueError("excluded block fraction must lie in (0, 0.5)")
        if self.normalization_mode not in {
            "volatility_adjusted",
            "raw_absolute_log_return",
        }:
            raise ValueError("unsupported shock normalization mode")


@dataclass(frozen=True)
class ExtremeBlockMask:
    selection_mask: pd.Series
    shock_score_at_open: pd.Series
    asset_shock_at_open: pd.DataFrame
    blocks: pd.DataFrame


def volatility_adjusted_absolute_return(
    close: pd.Series,
    *,
    return_window: int,
    volatility_window: int,
    volatility_floor_annual: float,
) -> pd.Series:
    """Magnitude-only short-horizon shock score on a close calendar."""
    values = close.astype(float)
    if (values <= 0.0).any():
        raise ValueError("shock score requires positive prices")
    daily = np.log(values).diff()
    total = np.log(values).diff(return_window).abs()
    scaled_volatility = daily.rolling(volatility_window).std(ddof=1) * np.sqrt(
        return_window
    )
    floor = volatility_floor_annual * np.sqrt(return_window / 252.0)
    adjusted = np.maximum(scaled_volatility, floor)
    result = total / adjusted
    result.name = "volatility_adjusted_absolute_return"
    return result.astype(float)


def build_extreme_block_mask(
    closes: Mapping[str, pd.Series],
    calendar: pd.DatetimeIndex,
    spec: ExtremeBlockSpec,
) -> ExtremeBlockMask:
    """Exclude the highest-scoring fixed blocks from parameter selection only."""
    if not closes:
        raise ValueError("at least one shock asset is required")
    asset_scores = {}
    for asset, close in closes.items():
        if spec.normalization_mode == "volatility_adjusted":
            close_score = volatility_adjusted_absolute_return(
                close,
                return_window=spec.shock_return_window,
                volatility_window=spec.shock_volatility_window,
                volatility_floor_annual=spec.volatility_floor_annual,
            )
        else:
            values = close.astype(float)
            if (values <= 0.0).any():
                raise ValueError("shock score requires positive prices")
            close_score = np.log(values).diff(spec.shock_return_window).abs()
            close_score.name = "raw_absolute_log_return"
        asset_scores[asset] = asof_previous_close(close_score, calendar)
    panel = pd.DataFrame(asset_scores, index=calendar)
    combined = panel.max(axis=1, skipna=True).rename("shock_score_at_open")
    block_ids = np.arange(len(calendar), dtype=int) // spec.block_length_sessions
    rows = []
    for block_id, positions in pd.Series(
        np.arange(len(calendar)), index=calendar
    ).groupby(block_ids):
        block_index = pd.DatetimeIndex(positions.index)
        values = combined.loc[block_index]
        rows.append(
            {
                "block_id": int(block_id),
                "start": block_index.min(),
                "end": block_index.max(),
                "observations": len(block_index),
                "shock_score": float(values.max()) if values.notna().any() else np.nan,
            }
        )
    blocks = pd.DataFrame(rows).set_index("block_id")
    finite = blocks.loc[blocks["shock_score"].notna()].sort_values(
        ["shock_score"], ascending=False, kind="mergesort"
    )
    excluded_count = max(
        1, int(np.ceil(len(blocks) * spec.excluded_block_fraction - 1e-12))
    )
    excluded_ids = set(finite.head(excluded_count).index.astype(int))
    blocks["excluded_from_selection"] = blocks.index.to_series().isin(excluded_ids)
    blocks["shock_rank_descending"] = blocks["shock_score"].rank(
        method="first", ascending=False
    )
    selection = pd.Series(
        [int(block_id) not in excluded_ids for block_id in block_ids],
        index=calendar,
        name="ordinary_regime_selection_mask",
        dtype=bool,
    )
    return ExtremeBlockMask(selection, combined, panel, blocks)


def validate_common_score_policies(
    policies: Mapping[str, AssetDRAQMPolicy],
) -> None:
    """Require both gated assets to use identical horizons and weights."""
    if set(policies) != {"510300.SH", "518880.SH"}:
        raise ValueError("common score requires both 510300.SH and 518880.SH")
    csi = policies["510300.SH"].profile
    gold = policies["518880.SH"].profile
    if csi.horizons != gold.horizons or not np.allclose(
        csi.weights, gold.weights, atol=1e-12
    ):
        raise ValueError("both ETFs must use identical score horizons and weights")
