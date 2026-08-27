"""Formal monthly QM40-reversal dividend sleeve with 100% equity."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Mapping

import numpy as np
import pandas as pd

from defender.live import DefenderNextOpenTarget, _append_flat_execution_day
from defender.relative_defender_rotation import DEFENSIVE_ASSET
from defender.w40_reversal_full_equity import (
    FORMAL_COST_RATES,
    FORMAL_DIVIDEND_ASSETS,
    FORMAL_START,
    _formal_market_through,
    _load_formal_market,
    _union_calendar,
    _with_initial_warmup_fallback,
)
from research.momentum_defender_occam import HELD_RETURN, performance
from research.momentum_defender_occam_defender import (
    MonthlySelectionSpec,
    build_portfolio_switch_interface,
    monthly_top1_selection,
    score_at_open,
    selected_asset_targets,
)


FORMAL_DEFENDER_STRATEGY_ID = "dividend_w40_qm_reversal_full_equity_v3"
SELECTION_WINDOW = 40
SELECTION_SPEC = MonthlySelectionSpec(SELECTION_WINDOW, "quality", "lowest")


@dataclass(frozen=True)
class FormalQMFullEquityDefenderBacktest:
    calendar: pd.DatetimeIndex
    selection: pd.DataFrame
    targets: pd.DataFrame
    interface: pd.DataFrame
    audit: Mapping[str, object]


def build_formal_backtest(
    calendar: pd.DatetimeIndex,
    *,
    end: date,
    market: Mapping[str, pd.DataFrame] | None = None,
) -> FormalQMFullEquityDefenderBacktest:
    prices = (
        _load_formal_market(end=end)
        if market is None
        else {asset: frame.copy() for asset, frame in market.items()}
    )
    scores = score_at_open(
        prices, FORMAL_DIVIDEND_ASSETS, calendar, SELECTION_SPEC
    )
    scores = _with_initial_warmup_fallback(scores, prices, calendar)
    selection = monthly_top1_selection(
        prices,
        FORMAL_DIVIDEND_ASSETS,
        calendar,
        scores,
        SELECTION_SPEC,
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
        "selection_score": "quality_momentum",
        "selection_direction": "lowest",
        "candidate_assets": list(FORMAL_DIVIDEND_ASSETS),
        "selection_switches": int(selection["selection_changed"].sum()),
        "target_sum_max_abs_error": target_error,
        "bond_weight_max_abs": bond_max,
        "performance": performance(held),
    }
    if target_error > 1e-12 or bond_max > 1e-12:
        raise AssertionError("formal QM40 Defender target audit failed")
    return FormalQMFullEquityDefenderBacktest(
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
    signal = pd.Timestamp(signal_date).normalize()
    execution = pd.Timestamp(execution_date).normalize()
    if execution <= signal:
        raise ValueError("execution_date must be after signal_date")
    applied = _formal_market_through(signal, market)
    current_calendar = _union_calendar(applied, start=FORMAL_START, end=signal)
    current_scores = score_at_open(
        applied,
        FORMAL_DIVIDEND_ASSETS,
        current_calendar,
        SELECTION_SPEC,
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
        extended,
        FORMAL_DIVIDEND_ASSETS,
        extended_calendar,
        SELECTION_SPEC,
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
        raise AssertionError("formal QM40 Defender target must sum to one")
    return DefenderNextOpenTarget(
        signal_date=signal,
        execution_date=execution,
        current_weights={current_asset: 1.0},
        target_weights=target_weights,
        target_cash_weight=0.0,
        current_selected_asset=current_asset,
        target_selected_asset=target_asset,
        selection_reason=reason,
        signal_reason="monthly_40_day_lowest_quality_momentum_full_equity",
    )
