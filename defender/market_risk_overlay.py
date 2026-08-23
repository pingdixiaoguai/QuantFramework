"""Simple, causal broad-market risk overlay for Defender v2.

Only broad-market ETFs are used.  The overlay has two ratio signals and no
industry ETF, volume filter, breadth vote, or cross-sectional scoring model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from data.store import read_local


CSI300_ASSET = "510300.SH"
SP500_ASSET = "513500.SH"
NASDAQ_ASSET = "513100.SH"
RISK_REFERENCE_ASSETS: tuple[str, ...] = (
    CSI300_ASSET,
    SP500_ASSET,
    NASDAQ_ASSET,
)


@dataclass(frozen=True)
class MarketRiskOverlayParams:
    """Parameters for the broad-market cap on the 512890 sleeve."""

    enabled: bool = True
    primary_cap: float = 0.65
    relative_window: int = 504
    relative_threshold: float = 0.95
    global_leadership_window: int = 40
    global_leadership_threshold: float = 0.05
    confirmation_days: int = 10

    def __post_init__(self) -> None:
        if not 0 <= self.primary_cap <= 1:
            raise ValueError("primary_cap must be in [0, 1]")
        if self.relative_window < 60:
            raise ValueError("relative_window must be at least 60")
        if not 0 <= self.relative_threshold <= 1:
            raise ValueError("relative_threshold must be in [0, 1]")
        if self.global_leadership_window < 2:
            raise ValueError("global_leadership_window must be at least 2")
        if self.confirmation_days < 1:
            raise ValueError("confirmation_days must be positive")


def _local_close(asset: str, index: pd.DatetimeIndex) -> pd.Series:
    frame = read_local(asset)
    if frame is None or frame.empty or "close" not in frame:
        raise RuntimeError(f"missing local close data for risk reference {asset}")
    local = frame.copy()
    local["date"] = pd.to_datetime(local["date"])
    close = local.set_index("date")["close"].astype(float).sort_index()
    return close.reindex(index).ffill()


def rolling_percentile(series: pd.Series, window: int) -> pd.Series:
    """Percentile of the current observation in a trailing causal window."""

    def percentile(values: np.ndarray) -> float:
        return float((values <= values[-1]).mean())

    return series.rolling(window, min_periods=max(60, window // 2)).apply(
        percentile,
        raw=True,
    )


def build_market_risk_features(
    primary: pd.DataFrame,
    params: MarketRiskOverlayParams = MarketRiskOverlayParams(),
) -> pd.DataFrame:
    """Build the two broad-market ratio features on the primary calendar."""

    frame = primary.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values("date").drop_duplicates("date").set_index("date")
    index = pd.DatetimeIndex(frame.index)
    primary_close = frame["close"].astype(float)
    csi300 = _local_close(CSI300_ASSET, index)
    sp500 = _local_close(SP500_ASSET, index)
    nasdaq = _local_close(NASDAQ_ASSET, index)

    features = pd.DataFrame(index=index)
    features["relative_price_percentile"] = rolling_percentile(
        primary_close / csi300,
        params.relative_window,
    )
    window = params.global_leadership_window
    features["nasdaq_sp500_relative_return"] = (
        nasdaq.pct_change(window) - sp500.pct_change(window)
    )
    return features


def calculate_market_risk_overlay(
    primary: pd.DataFrame,
    params: MarketRiskOverlayParams = MarketRiskOverlayParams(),
) -> pd.DataFrame:
    """Return close signals and the next-open executable overlay state."""

    index = pd.DatetimeIndex(pd.to_datetime(primary["date"]))
    if not params.enabled:
        result = pd.DataFrame(index=index)
        result["relative_price_percentile"] = np.nan
        result["nasdaq_sp500_relative_return"] = np.nan
        for column in (
            "relative_stretched",
            "nasdaq_leadership",
            "overlay_signal_close",
            "overlay_active",
        ):
            result[column] = False
        return result

    features = build_market_risk_features(primary, params)
    relative_stretched = (
        features["relative_price_percentile"] >= params.relative_threshold
    )
    nasdaq_leadership = (
        features["nasdaq_sp500_relative_return"]
        >= params.global_leadership_threshold
    )
    raw = (relative_stretched | nasdaq_leadership).fillna(False)
    confirmed = raw.rolling(params.confirmation_days).sum() >= params.confirmation_days

    result = features.copy()
    result["relative_stretched"] = relative_stretched.fillna(False)
    result["nasdaq_leadership"] = nasdaq_leadership.fillna(False)
    result["overlay_signal_close"] = confirmed.fillna(False)
    result["overlay_active"] = result["overlay_signal_close"].shift(
        1,
        fill_value=False,
    )
    return result
