"""Next-open target calculation for the vendored formal Defender strategy."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Mapping

import pandas as pd

from .current_strategy import run_backtest as run_current_strategy
from .relative_defender_rotation import (
    DEFENSIVE_ASSET,
    ROTATION_ASSETS,
    load_rotation_market,
)


ALL_ASSETS = (*ROTATION_ASSETS, DEFENSIVE_ASSET)


@dataclass(frozen=True)
class DefenderNextOpenTarget:
    signal_date: pd.Timestamp
    execution_date: pd.Timestamp
    current_weights: Mapping[str, float]
    target_weights: Mapping[str, float]
    target_cash_weight: float
    current_selected_asset: str
    target_selected_asset: str
    selection_reason: str
    signal_reason: str


def _market_through(
    signal_date: pd.Timestamp,
    market: Mapping[str, pd.DataFrame] | None,
) -> dict[str, pd.DataFrame]:
    source = dict(market) if market is not None else load_rotation_market()
    result: dict[str, pd.DataFrame] = {}
    for asset in ALL_ASSETS:
        frame = source[asset].copy()
        frame["date"] = pd.to_datetime(frame["date"])
        frame = frame.loc[frame["date"] <= signal_date].sort_values("date")
        if frame.empty:
            raise RuntimeError(f"{asset}: no Defender history through {signal_date.date()}")
        result[asset] = frame.reset_index(drop=True)
    return result


def _append_flat_execution_day(
    market: Mapping[str, pd.DataFrame],
    execution_date: pd.Timestamp,
) -> dict[str, pd.DataFrame]:
    """Append a zero-return row so the latest close signal becomes executable."""
    extended: dict[str, pd.DataFrame] = {}
    for asset, source in market.items():
        frame = source.copy().sort_values("date").reset_index(drop=True)
        last = frame.iloc[-1].copy()
        last["date"] = execution_date
        for column in ("open", "high", "low", "close"):
            last[column] = float(frame.iloc[-1]["close"])
        if "volume" in frame.columns:
            last["volume"] = 0.0
        extended[asset] = pd.concat(
            [frame, pd.DataFrame([last], columns=frame.columns)],
            ignore_index=True,
        )
    return extended


def _weights(row: pd.Series) -> dict[str, float]:
    return {
        asset: float(row[f"target_{asset}"])
        for asset in ALL_ASSETS
        if float(row[f"target_{asset}"]) > 1e-14
    }


def build_next_open_target(
    signal_date: pd.Timestamp | date,
    execution_date: pd.Timestamp | date,
    *,
    market: Mapping[str, pd.DataFrame] | None = None,
) -> DefenderNextOpenTarget:
    """Calculate the formal Defender allocation for one future market open."""
    signal = pd.Timestamp(signal_date).normalize()
    execution = pd.Timestamp(execution_date).normalize()
    if execution <= signal:
        raise ValueError("execution_date must be after signal_date")

    applied_market = _market_through(signal, market)
    daily, _, _, _ = run_current_strategy(market=applied_market)
    if pd.Timestamp(daily.index[-1]).normalize() != signal:
        raise RuntimeError(
            "Defender union calendar does not end on the requested signal date"
        )

    extended = _append_flat_execution_day(applied_market, execution)
    extended_daily, _, _, _ = run_current_strategy(market=extended)
    if execution not in extended_daily.index:
        raise RuntimeError("Defender did not produce the requested execution row")

    current = daily.iloc[-1]
    target = extended_daily.loc[execution]
    target_weights = _weights(target)
    cash = max(0.0, 1.0 - sum(target_weights.values()))
    if abs(sum(target_weights.values()) + cash - 1.0) > 1e-12:
        raise AssertionError("Defender next-open target plus cash must sum to one")
    return DefenderNextOpenTarget(
        signal_date=signal,
        execution_date=execution,
        current_weights=_weights(current),
        target_weights=target_weights,
        target_cash_weight=cash,
        current_selected_asset=str(current["selected_asset"]),
        target_selected_asset=str(target["selected_asset"]),
        selection_reason=str(target["selection_reason"]),
        signal_reason=str(target["signal_execution_reason"]),
    )
