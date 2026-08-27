"""Read-only price/volume/fund-share warning for DingTalk diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Callable

import numpy as np
import pandas as pd

from data.fund_share import fetch_fund_share
from data.store import query
from research.momentum_defender_occam import MOMENTUM_ASSETS


CHINEXT_ASSET = "159915.SZ"
PRICE_HIGH_LOOKBACK = 200
PRICE_RETURN_WINDOW = 20
PRICE_RETURN_THRESHOLD = 0.15
VOLUME_MEDIAN_WINDOW = 20
VOLUME_RATIO_THRESHOLD = 1.50
SHARE_FLOW_WINDOW = 20


@dataclass(frozen=True)
class PeakWarning:
    asset: str
    signal_date: date
    triggered: bool
    current_close: float
    prior_high200: float
    close20ago: float
    current_volume: float
    prior_volume_median20: float
    price_breakout: float
    price_return20: float
    volume_ratio20: float
    share_filter_required: bool
    share_data_available: bool
    share_flow20: float | None
    reason: str


def evaluate_peak_warning(
    asset: str,
    signal_date: date,
    *,
    share_loader: Callable[[str, date, date], pd.Series] = fetch_fund_share,
) -> PeakWarning:
    """Evaluate the close-known warning without changing any strategy state."""

    if asset not in MOMENTUM_ASSETS:
        raise ValueError(f"peak warning requires a Momentum ETF: {asset}")
    price = (
        query(asset, date(2013, 1, 1), signal_date)
        .sort_values("date")
        .drop_duplicates("date")
        .set_index("date")
    )
    timestamp = pd.Timestamp(signal_date)
    if timestamp not in price.index:
        raise RuntimeError(f"peak warning has no price on {signal_date}: {asset}")
    history = price.loc[:timestamp]
    required = max(PRICE_HIGH_LOOKBACK, PRICE_RETURN_WINDOW, VOLUME_MEDIAN_WINDOW) + 1
    if len(history) < required:
        raise RuntimeError(f"peak warning history is too short for {asset}")
    close = history["close"].astype(float)
    volume = history["volume"].astype(float)
    current_close = float(close.iloc[-1])
    prior_high = float(close.iloc[-(PRICE_HIGH_LOOKBACK + 1) : -1].max())
    price_breakout = current_close / prior_high - 1.0
    price_return20 = current_close / float(close.iloc[-(PRICE_RETURN_WINDOW + 1)]) - 1.0
    prior_volume_median = float(
        volume.iloc[-(VOLUME_MEDIAN_WINDOW + 1) : -1].median()
    )
    if prior_volume_median <= 0.0:
        raise RuntimeError(f"peak warning prior volume is invalid for {asset}")
    volume_ratio20 = float(volume.iloc[-1]) / prior_volume_median
    price_flag = price_breakout > 0.0 and price_return20 >= PRICE_RETURN_THRESHOLD
    volume_flag = volume_ratio20 >= VOLUME_RATIO_THRESHOLD
    base_trigger = price_flag and volume_flag
    share_required = asset == CHINEXT_ASSET

    if not share_required:
        return PeakWarning(
            asset=asset,
            signal_date=signal_date,
            triggered=base_trigger,
            current_close=current_close,
            prior_high200=prior_high,
            close20ago=float(close.iloc[-(PRICE_RETURN_WINDOW + 1)]),
            current_volume=float(volume.iloc[-1]),
            prior_volume_median20=prior_volume_median,
            price_breakout=price_breakout,
            price_return20=price_return20,
            volume_ratio20=volume_ratio20,
            share_filter_required=False,
            share_data_available=True,
            share_flow20=None,
            reason=(
                "价格、20日涨幅和成交量条件同时满足"
                if base_trigger
                else "至少一个价量条件尚未满足"
            ),
        )

    try:
        shares = share_loader(
            asset,
            signal_date - timedelta(days=120),
            signal_date,
        ).sort_index()
        if shares.empty or pd.Timestamp(shares.index.max()).date() != signal_date:
            raise RuntimeError("信号日基金份额尚不可用")
        aligned = shares.reindex(pd.DatetimeIndex(history.index)).ffill().loc[:timestamp]
        aligned = aligned.dropna()
        if len(aligned) < SHARE_FLOW_WINDOW + 1:
            raise RuntimeError("基金份额历史不足20个交易区间")
        current_share = float(aligned.iloc[-1])
        prior_share = float(aligned.iloc[-(SHARE_FLOW_WINDOW + 1)])
        if not all(np.isfinite(value) and value > 0.0 for value in (current_share, prior_share)):
            raise RuntimeError("基金份额值无效")
        share_flow20 = current_share / prior_share - 1.0
    except Exception as exc:
        return PeakWarning(
            asset=asset,
            signal_date=signal_date,
            triggered=False,
            current_close=current_close,
            prior_high200=prior_high,
            close20ago=float(close.iloc[-(PRICE_RETURN_WINDOW + 1)]),
            current_volume=float(volume.iloc[-1]),
            prior_volume_median20=prior_volume_median,
            price_breakout=price_breakout,
            price_return20=price_return20,
            volume_ratio20=volume_ratio20,
            share_filter_required=True,
            share_data_available=False,
            share_flow20=None,
            reason=f"创业板份额数据不可用（{type(exc).__name__}）",
        )

    share_flag = share_flow20 > 0.0
    triggered = base_trigger and share_flag
    unmet = []
    if not price_flag:
        unmet.append("价格条件未全部满足")
    if not volume_flag:
        unmet.append("量能条件未满足")
    if not share_flag:
        unmet.append("创业板20日基金份额持平或下降")
    return PeakWarning(
        asset=asset,
        signal_date=signal_date,
        triggered=triggered,
        current_close=current_close,
        prior_high200=prior_high,
        close20ago=float(close.iloc[-(PRICE_RETURN_WINDOW + 1)]),
        current_volume=float(volume.iloc[-1]),
        prior_volume_median20=prior_volume_median,
        price_breakout=price_breakout,
        price_return20=price_return20,
        volume_ratio20=volume_ratio20,
        share_filter_required=True,
        share_data_available=True,
        share_flow20=share_flow20,
        reason=(
            "全部价量和创业板份额条件同时满足"
            if triggered
            else "；".join(unmet)
        ),
    )
