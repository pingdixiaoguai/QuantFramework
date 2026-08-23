"""Reusable causal volatility primitives for Momentum overlay research."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from data.store import query
from research.momentum_defender_occam import _momentum_target_schedule


def load_ohlc(asset: str, end: date, start: date = date(2013, 1, 1)) -> pd.DataFrame:
    frame = query(asset, start, end).sort_values("date")
    frame = frame.drop_duplicates("date").set_index("date")
    required = ["open", "high", "low", "close"]
    if frame.empty or frame[required].isna().any().any():
        raise ValueError(f"invalid OHLC history for {asset}")
    return frame[required].astype(float)


def rogers_satchell_volatility(prices: pd.DataFrame, window: int) -> pd.Series:
    required = {"open", "high", "low", "close"}
    missing = required - set(prices.columns)
    if missing:
        raise ValueError(f"price history missing OHLC columns: {sorted(missing)}")
    if window < 2:
        raise ValueError("volatility window must be at least 2")
    ohlc = prices[["open", "high", "low", "close"]].astype(float)
    if (ohlc <= 0).any().any():
        raise ValueError("Rogers-Satchell volatility requires positive OHLC")
    variance = (
        np.log(ohlc["high"] / ohlc["close"])
        * np.log(ohlc["high"] / ohlc["open"])
        + np.log(ohlc["low"] / ohlc["close"])
        * np.log(ohlc["low"] / ohlc["open"])
    ).clip(lower=0.0)
    realized = np.sqrt(252.0 * variance.rolling(window, min_periods=window).mean())
    realized.name = f"rs_volatility_{window}"
    return realized


def expanding_volatility_cap(
    realized_volatility: pd.Series,
    quantile: float,
    *,
    step: float = 0.20,
    min_history: int = 20,
) -> pd.DataFrame:
    """Build a strict-lag expanding-quantile cap on the close calendar."""
    if not 0.0 < quantile < 1.0:
        raise ValueError("expanding quantile must be strictly between zero and one")
    if not 0.0 < step <= 1.0:
        raise ValueError("cap step must be in (0, 1]")
    volatility = realized_volatility.astype(float)
    threshold = volatility.shift(1).expanding(min_periods=min_history).quantile(quantile)
    raw_cap = (threshold / volatility).clip(upper=1.0)
    cap = np.floor(raw_cap / step + 1e-12) * step
    cap = cap.clip(lower=0.0, upper=1.0).where(raw_cap.notna(), 1.0)
    return pd.DataFrame(
        {
            "realized_volatility": volatility,
            "threshold": threshold,
            "raw_cap": raw_cap,
            "cap": cap,
        }
    )


def asof_previous_close(series: pd.Series, calendar: pd.DatetimeIndex) -> pd.Series:
    """Map every open to the latest strictly earlier close observation."""
    source = series.copy().sort_index()
    if source.index.duplicated().any():
        raise ValueError("source close series contains duplicate dates")
    source_index = pd.DatetimeIndex(source.index)
    positions = source_index.searchsorted(calendar, side="left") - 1
    values = np.full(len(calendar), np.nan, dtype=object)
    valid = positions >= 0
    source_values = source.to_numpy()
    values[valid] = source_values[positions[valid]]
    return pd.to_numeric(pd.Series(values, index=calendar, name=series.name))


def momentum_asset_at_previous_close(
    momentum_result,
    calendar: pd.DatetimeIndex,
) -> pd.Series:
    """Return the Momentum ETF owned through the close before every open."""
    prior_dates = momentum_result.daily_returns.index[
        momentum_result.daily_returns.index < calendar.min()
    ]
    if len(prior_dates) == 0:
        raise AssertionError("Momentum signal study requires a warm-up holding date")
    replay_calendar = pd.DatetimeIndex([prior_dates.max()]).append(calendar)
    targets = _momentum_target_schedule(momentum_result, replay_calendar)
    previous = targets.idxmax(axis=1).shift(1).reindex(calendar)
    if previous.isna().any():
        raise AssertionError("Momentum previous-close asset is missing")
    previous.name = "momentum_asset_at_previous_close"
    return previous


def choose_by_asset(
    values_by_asset: Mapping[str, pd.Series],
    asset_at_open: pd.Series,
) -> pd.Series:
    result = pd.Series(np.nan, index=asset_at_open.index, dtype=float)
    for asset, values in values_by_asset.items():
        held = asset_at_open.eq(asset)
        result.loc[held] = values.reindex(result.index).loc[held].astype(float)
    if result.isna().any():
        missing = sorted(asset_at_open.loc[result.isna()].unique())
        raise AssertionError(f"missing per-asset signal values for: {missing}")
    return result
