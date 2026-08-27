"""Occam position rules for the selected dividend ETF versus 511260."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from defender.relative_defender_champion import target_schedule as champion_schedule
from research.momentum_volatility import asof_previous_close


FROZEN_CHAMPION = "frozen_champion"
FIXED_WEIGHT = "fixed_weight"
TREND_BINARY = "trend_binary"
RANGE_LOCATION = "range_location"
VOLATILITY_TARGET = "volatility_target"
RELATIVE_VOLATILITY_CAP = "relative_volatility_cap"
DRAWDOWN_SCALE = "drawdown_scale"
RANGE_HIGH_CUT = "range_high_cut"
VOLATILITY_HIGH_CUT = "volatility_high_cut"
CONTRARIAN_TREND = "contrarian_trend"
POSITION_FAMILIES = {
    FROZEN_CHAMPION,
    FIXED_WEIGHT,
    TREND_BINARY,
    RANGE_LOCATION,
    VOLATILITY_TARGET,
    RELATIVE_VOLATILITY_CAP,
    DRAWDOWN_SCALE,
    RANGE_HIGH_CUT,
    VOLATILITY_HIGH_CUT,
    CONTRARIAN_TREND,
}
ANCHOR_SOURCE = "anchor"
SELECTED_SOURCE = "selected"
SIGNAL_SOURCES = {ANCHOR_SOURCE, SELECTED_SOURCE}


@dataclass(frozen=True)
class PositionSpec:
    """A low-dimensional equity/bond exposure rule."""

    family: str
    signal_source: str | None = None
    window: int | None = None
    level: float | None = None
    secondary_level: float | None = None

    def __post_init__(self) -> None:
        if self.family not in POSITION_FAMILIES:
            raise ValueError(f"unsupported position family: {self.family}")
        if self.family in {FROZEN_CHAMPION, FIXED_WEIGHT}:
            if self.signal_source is not None or self.window is not None:
                raise ValueError(f"{self.family} does not accept a signal source/window")
        elif self.signal_source not in SIGNAL_SOURCES or self.window is None:
            raise ValueError(f"{self.family} requires a signal source and window")
        if self.window is not None and self.window < 2:
            raise ValueError("position window must be at least two")
        if self.family == FROZEN_CHAMPION and (
            self.level is not None or self.secondary_level is not None
        ):
            raise ValueError("frozen champion does not accept a level")
        if self.family == FIXED_WEIGHT and (
            self.level is None or not 0.0 <= self.level <= 1.0
        ):
            raise ValueError("fixed weight requires a level in [0, 1]")
        if self.family in {VOLATILITY_TARGET, DRAWDOWN_SCALE} and (
            self.level is None or self.level <= 0.0
        ):
            raise ValueError(f"{self.family} requires a positive level")
        if self.family == RELATIVE_VOLATILITY_CAP and (
            self.level is None or not 0.0 < self.level < 1.0
        ):
            raise ValueError("relative volatility cap requires a quantile in (0, 1)")
        if self.family in {RANGE_HIGH_CUT, VOLATILITY_HIGH_CUT} and (
            self.level is None or not 0.0 < self.level < 1.0
        ):
            raise ValueError(f"{self.family} requires a threshold in (0, 1)")
        if self.family in {
            RANGE_HIGH_CUT,
            VOLATILITY_HIGH_CUT,
            CONTRARIAN_TREND,
        } and (
            self.secondary_level is None
            or not 0.0 <= self.secondary_level <= 1.0
        ):
            raise ValueError(f"{self.family} requires a fallback weight in [0, 1]")
        if self.family in {TREND_BINARY, RANGE_LOCATION} and self.level is not None:
            raise ValueError(f"{self.family} does not accept a level")
        if self.family not in {
            RANGE_HIGH_CUT,
            VOLATILITY_HIGH_CUT,
            CONTRARIAN_TREND,
        } and self.secondary_level is not None:
            raise ValueError(f"{self.family} does not accept a secondary level")

    @property
    def candidate_id(self) -> str:
        if self.family == FROZEN_CHAMPION:
            return FROZEN_CHAMPION
        if self.family == FIXED_WEIGHT:
            return f"fixed_w{self.level:.2f}"
        base = f"{self.family}_{self.signal_source}_w{self.window}"
        if self.level is None and self.secondary_level is None:
            return base
        level = "" if self.level is None else f"_l{self.level:.2f}"
        secondary = (
            "" if self.secondary_level is None else f"_f{self.secondary_level:.2f}"
        )
        return f"{base}{level}{secondary}"

    @property
    def fitted_parameter_count(self) -> int:
        if self.family == FROZEN_CHAMPION:
            return 18
        if self.family in {FIXED_WEIGHT, TREND_BINARY, RANGE_LOCATION}:
            return 1
        if self.family == CONTRARIAN_TREND:
            return 2
        if self.family in {RANGE_HIGH_CUT, VOLATILITY_HIGH_CUT}:
            return 3
        return 2


def _close(frame: pd.DataFrame) -> pd.Series:
    prices = frame.copy()
    prices["date"] = pd.to_datetime(prices["date"])
    prices = prices.sort_values("date").drop_duplicates("date").set_index("date")
    close = prices["close"].astype(float)
    if close.empty or close.isna().any() or close.le(0.0).any():
        raise ValueError("position signals require finite positive closes")
    return close


def _signal_on_close(close: pd.Series, spec: PositionSpec) -> pd.Series:
    log_close = np.log(close)
    window = int(spec.window or 0)
    if spec.family in {TREND_BINARY, CONTRARIAN_TREND}:
        return log_close.diff(window)
    if spec.family in {RANGE_LOCATION, RANGE_HIGH_CUT}:
        low = close.rolling(window).min()
        high = close.rolling(window).max()
        width = high - low
        return ((close - low) / width.replace(0.0, np.nan)).where(
            width.ne(0.0), 0.5
        )
    if spec.family in {
        VOLATILITY_TARGET,
        RELATIVE_VOLATILITY_CAP,
        VOLATILITY_HIGH_CUT,
    }:
        volatility = log_close.diff().rolling(window).std(ddof=1) * np.sqrt(252.0)
        if spec.family == VOLATILITY_TARGET:
            return volatility
        threshold = volatility.shift(1).rolling(504, min_periods=252).quantile(
            float(spec.level)
        )
        return threshold / volatility.replace(0.0, np.nan)
    if spec.family == DRAWDOWN_SCALE:
        peak = close.rolling(window).max()
        return close / peak - 1.0
    raise ValueError(f"family has no close signal: {spec.family}")


def _source_signal_at_open(
    market: Mapping[str, pd.DataFrame],
    assets: tuple[str, ...],
    selection: pd.Series,
    calendar: pd.DatetimeIndex,
    anchor_asset: str,
    spec: PositionSpec,
) -> pd.Series:
    required = (anchor_asset,) if spec.signal_source == ANCHOR_SOURCE else assets
    panel = pd.DataFrame(
        {
            asset: asof_previous_close(
                _signal_on_close(_close(market[asset]), spec), calendar
            )
            for asset in required
        },
        index=calendar,
    )
    if spec.signal_source == ANCHOR_SOURCE:
        return panel[anchor_asset].rename("position_signal_at_open")
    values = [panel.at[timestamp, str(selection.loc[timestamp])] for timestamp in calendar]
    return pd.Series(
        values, index=calendar, name="position_signal_at_open", dtype=float
    )


def _exposure_from_signal(signal: pd.Series, spec: PositionSpec) -> pd.Series:
    if spec.family == TREND_BINARY:
        exposure = signal.gt(0.0).astype(float)
    elif spec.family == CONTRARIAN_TREND:
        exposure = pd.Series(
            np.where(
                signal.isna(),
                1.0,
                np.where(signal.le(0.0), 1.0, float(spec.secondary_level)),
            ),
            index=signal.index,
        )
    elif spec.family == RANGE_LOCATION:
        exposure = 1.0 - signal
    elif spec.family == RANGE_HIGH_CUT:
        exposure = pd.Series(
            np.where(
                signal.ge(float(spec.level)), float(spec.secondary_level), 1.0
            ),
            index=signal.index,
        )
    elif spec.family == VOLATILITY_TARGET:
        exposure = float(spec.level) / signal.replace(0.0, np.nan)
    elif spec.family == RELATIVE_VOLATILITY_CAP:
        exposure = signal
    elif spec.family == VOLATILITY_HIGH_CUT:
        exposure = pd.Series(
            np.where(signal.lt(1.0), float(spec.secondary_level), 1.0),
            index=signal.index,
        )
    elif spec.family == DRAWDOWN_SCALE:
        exposure = -signal / float(spec.level)
    else:
        raise ValueError(f"family has no signal exposure: {spec.family}")
    return exposure.clip(0.0, 1.0).fillna(1.0).rename("equity_weight")


def build_position_targets(
    market: Mapping[str, pd.DataFrame],
    assets: tuple[str, ...],
    defensive_asset: str,
    selection: pd.Series,
    calendar: pd.DatetimeIndex,
    anchor_asset: str,
    spec: PositionSpec,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build causal selected-dividend/511260 next-open targets."""
    if not selection.index.equals(calendar):
        raise ValueError("selection must use the target calendar")
    if defensive_asset in assets:
        raise ValueError("defensive asset cannot be a selectable equity asset")
    if spec.family == FROZEN_CHAMPION:
        schedule = champion_schedule(market[anchor_asset])
        exposure = schedule["primary_target"].reindex(calendar).ffill()
        signal = pd.Series(np.nan, index=calendar, name="position_signal_at_open")
    elif spec.family == FIXED_WEIGHT:
        exposure = pd.Series(float(spec.level), index=calendar, name="equity_weight")
        signal = pd.Series(np.nan, index=calendar, name="position_signal_at_open")
    else:
        signal = _source_signal_at_open(
            market, assets, selection, calendar, anchor_asset, spec
        )
        exposure = _exposure_from_signal(signal, spec)
    if exposure.isna().any() or exposure.lt(0.0).any() or exposure.gt(1.0).any():
        raise ValueError("position exposure must be finite and lie in [0, 1]")

    targets = pd.DataFrame(
        0.0, index=calendar, columns=[*assets, defensive_asset], dtype=float
    )
    for timestamp in calendar:
        selected = str(selection.loc[timestamp])
        if selected not in assets:
            raise ValueError(f"unknown selected asset: {selected}")
        weight = float(exposure.loc[timestamp])
        targets.at[timestamp, selected] = weight
        targets.at[timestamp, defensive_asset] = 1.0 - weight
    if not np.allclose(targets.sum(axis=1), 1.0, atol=1e-12):
        raise AssertionError("position targets must sum to one")
    diagnostics = pd.DataFrame(
        {
            "selected_asset": selection.astype(str),
            "position_signal_at_open": signal,
            "equity_weight": exposure.astype(float),
            "bond_weight": 1.0 - exposure.astype(float),
        },
        index=calendar,
    )
    diagnostics.index.name = "date"
    return targets, diagnostics
