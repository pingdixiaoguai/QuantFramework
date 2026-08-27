"""Formal monthly W40-reversal dividend sleeve with no bond allocation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Mapping

import numpy as np
import pandas as pd

from data.store import read_local
from defender.live import (
    DefenderNextOpenTarget,
    _append_flat_execution_day,
)
from defender.relative_defender_rotation import (
    DEFENSIVE_ASSET,
)
from research.momentum_defender_occam import HELD_RETURN, performance
from research.momentum_defender_occam_defender import (
    MonthlySelectionSpec,
    build_portfolio_switch_interface,
    monthly_top1_selection,
    score_at_open,
    selected_asset_targets,
)


FORMAL_DEFENDER_STRATEGY_ID = "dividend_w40_reversal_full_equity_v2"
FORMAL_DIVIDEND_ASSETS = (
    "512890.SH",
    "513530.SH",
    "515080.SH",
    "510880.SH",
    "515450.SH",
    "513630.SH",
)
FORMAL_COST_RATES = {
    **{asset: 0.0001 for asset in FORMAL_DIVIDEND_ASSETS},
    DEFENSIVE_ASSET: 0.00001,
}
SELECTION_WINDOW = 40
SELECTION_SPEC = MonthlySelectionSpec(SELECTION_WINDOW, "return", "lowest")
FORMAL_START = pd.Timestamp("2013-01-01")
WARMUP_ASSET = "510880.SH"


@dataclass(frozen=True)
class FormalFullEquityDefenderBacktest:
    calendar: pd.DatetimeIndex
    selection: pd.DataFrame
    targets: pd.DataFrame
    interface: pd.DataFrame
    audit: Mapping[str, object]


def _union_calendar(
    market: Mapping[str, pd.DataFrame],
    *,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> pd.DatetimeIndex:
    values: set[pd.Timestamp] = set()
    for asset in (*FORMAL_DIVIDEND_ASSETS, DEFENSIVE_ASSET):
        dates = pd.to_datetime(market[asset]["date"])
        values.update(pd.Timestamp(value) for value in dates)
    calendar = pd.DatetimeIndex(sorted(values))
    if start is not None:
        calendar = calendar[calendar >= start]
    if end is not None:
        calendar = calendar[calendar <= end]
    if calendar.empty:
        raise ValueError("formal full-equity Defender calendar is empty")
    return calendar


def _load_formal_market(end: date | None = None) -> dict[str, pd.DataFrame]:
    """Load only the promoted v2 dividend universe from local HFQ storage."""
    end_timestamp = pd.Timestamp(end or date.today())
    result: dict[str, pd.DataFrame] = {}
    for asset in (*FORMAL_DIVIDEND_ASSETS, DEFENSIVE_ASSET):
        frame = read_local(asset)
        if frame is None or frame.empty:
            raise RuntimeError(f"missing local data for: {asset}")
        selected = frame.loc[frame["date"].le(end_timestamp)].copy()
        if selected.empty:
            raise RuntimeError(f"{asset}: no formal Defender history through {end_timestamp.date()}")
        result[asset] = selected.reset_index(drop=True)
    return result


def _formal_market_through(
    signal_date: pd.Timestamp,
    market: Mapping[str, pd.DataFrame] | None,
) -> dict[str, pd.DataFrame]:
    source = dict(market) if market is not None else _load_formal_market(signal_date.date())
    result: dict[str, pd.DataFrame] = {}
    for asset in (*FORMAL_DIVIDEND_ASSETS, DEFENSIVE_ASSET):
        frame = source[asset].copy()
        frame["date"] = pd.to_datetime(frame["date"])
        selected = frame.loc[frame["date"].le(signal_date)].sort_values("date")
        if selected.empty:
            raise RuntimeError(f"{asset}: no formal Defender history through {signal_date.date()}")
        result[asset] = selected.reset_index(drop=True)
    return result


def _with_initial_warmup_fallback(
    scores: pd.DataFrame,
    market: Mapping[str, pd.DataFrame],
    calendar: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Hold the only long-history dividend ETF until 40-session scores exist."""
    applied = scores.copy()
    first_execution = pd.Timestamp(calendar[0])
    if applied.loc[first_execution].notna().any():
        return applied
    traded = set(pd.to_datetime(market[WARMUP_ASSET]["date"]))
    if first_execution not in traded:
        raise RuntimeError("formal Defender warmup asset is not initially tradable")
    applied.at[first_execution, WARMUP_ASSET] = 0.0
    return applied


def build_formal_backtest(
    calendar: pd.DatetimeIndex,
    *,
    end: date,
    market: Mapping[str, pd.DataFrame] | None = None,
) -> FormalFullEquityDefenderBacktest:
    """Build the formal selected-dividend interface on a supplied calendar."""
    prices = (
        _load_formal_market(end=end)
        if market is None
        else {asset: frame.copy() for asset, frame in market.items()}
    )
    scores = score_at_open(prices, FORMAL_DIVIDEND_ASSETS, calendar, SELECTION_SPEC)
    scores = _with_initial_warmup_fallback(scores, prices, calendar)
    selection = monthly_top1_selection(
        prices, FORMAL_DIVIDEND_ASSETS, calendar, scores, SELECTION_SPEC
    )
    targets = selected_asset_targets(
        selection["selected_asset"].astype(str),
        FORMAL_DIVIDEND_ASSETS,
        selected_weight=1.0,
        residual_asset=DEFENSIVE_ASSET,
    )
    interface = build_portfolio_switch_interface(
        prices, targets, FORMAL_COST_RATES
    )
    target_error = float((targets.sum(axis=1) - 1.0).abs().max())
    bond_max = float(targets[DEFENSIVE_ASSET].abs().max())
    held = interface[HELD_RETURN].astype(float)
    audit = {
        "status": "passed",
        "strategy_id": FORMAL_DEFENDER_STRATEGY_ID,
        "selection_window": SELECTION_WINDOW,
        "selection_direction": "lowest",
        "candidate_assets": list(FORMAL_DIVIDEND_ASSETS),
        "selection_switches": int(selection["selection_changed"].sum()),
        "target_sum_max_abs_error": target_error,
        "bond_weight_max_abs": bond_max,
        "performance": performance(held),
    }
    if target_error > 1e-12 or bond_max > 1e-12:
        raise AssertionError("formal full-equity Defender target audit failed")
    return FormalFullEquityDefenderBacktest(
        calendar=calendar,
        selection=selection,
        targets=targets,
        interface=interface,
        audit=audit,
    )


def build_next_open_target(
    signal_date: pd.Timestamp | date,
    execution_date: pd.Timestamp | date,
    *,
    market: Mapping[str, pd.DataFrame] | None = None,
) -> DefenderNextOpenTarget:
    """Calculate the formal 100%-dividend target for one future open."""
    signal = pd.Timestamp(signal_date).normalize()
    execution = pd.Timestamp(execution_date).normalize()
    if execution <= signal:
        raise ValueError("execution_date must be after signal_date")
    applied = _formal_market_through(signal, market)
    current_calendar = _union_calendar(
        applied, start=FORMAL_START, end=signal
    )
    current_scores = score_at_open(
        applied, FORMAL_DIVIDEND_ASSETS, current_calendar, SELECTION_SPEC
    )
    current_scores = _with_initial_warmup_fallback(
        current_scores, applied, current_calendar
    )
    current_selection = monthly_top1_selection(
        applied,
        FORMAL_DIVIDEND_ASSETS,
        current_calendar,
        current_scores,
        SELECTION_SPEC,
    )
    if current_calendar[-1] != signal:
        raise RuntimeError("Defender union calendar does not end on signal_date")

    extended = _append_flat_execution_day(applied, execution)
    extended_calendar = _union_calendar(
        extended, start=FORMAL_START, end=execution
    )
    extended_scores = score_at_open(
        extended, FORMAL_DIVIDEND_ASSETS, extended_calendar, SELECTION_SPEC
    )
    extended_scores = _with_initial_warmup_fallback(
        extended_scores, extended, extended_calendar
    )
    target_selection = monthly_top1_selection(
        extended,
        FORMAL_DIVIDEND_ASSETS,
        extended_calendar,
        extended_scores,
        SELECTION_SPEC,
    )
    current_asset = str(current_selection.iloc[-1]["selected_asset"])
    target_asset = str(target_selection.at[execution, "selected_asset"])
    reason = str(target_selection.at[execution, "selection_reason"])
    target_weights = {target_asset: 1.0}
    if not np.isclose(sum(target_weights.values()), 1.0, atol=1e-12):
        raise AssertionError("formal full-equity Defender target must sum to one")
    return DefenderNextOpenTarget(
        signal_date=signal,
        execution_date=execution,
        current_weights={current_asset: 1.0},
        target_weights=target_weights,
        target_cash_weight=0.0,
        current_selected_asset=current_asset,
        target_selected_asset=target_asset,
        selection_reason=reason,
        signal_reason="monthly_40_day_lowest_return_full_equity",
    )
