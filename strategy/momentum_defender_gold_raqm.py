"""Formal no-lock confirmation/bridge signal with raw-Gold coverage."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

from strategy.momentum_defender_absolute_stability import (
    FORMAL_STRATEGY_ID,
    GOLD_ENTRY_DIFFERENCE,
    GOLD_EXIT_DIFFERENCE,
    GOLD_MIN_HOLD_DAYS,
    FormalNextOpenSignal,
    build_next_open_signal,
    raw_gold_metrics_at_open,
)


ENTRY_DIFFERENCE = GOLD_ENTRY_DIFFERENCE
EXIT_DIFFERENCE = GOLD_EXIT_DIFFERENCE
MIN_GOLD_HOLD_DAYS = GOLD_MIN_HOLD_DAYS
FormalGoldNextOpenSignal = FormalNextOpenSignal


@dataclass(frozen=True)
class GoldOverrideStep:
    active: bool
    reason: str
    held_days_at_open: int


def advance_gold_override(
    *,
    current_active: bool,
    completed_gold_days: int,
    base_next_risk_on: bool,
    metric_difference: float,
    entry_difference: float = ENTRY_DIFFERENCE,
    exit_difference: float = EXIT_DIFFERENCE,
    min_gold_hold_days: int = MIN_GOLD_HOLD_DAYS,
) -> GoldOverrideStep:
    """Advance the formal raw-Gold state exactly one market open."""
    if completed_gold_days < 0:
        raise ValueError("completed_gold_days cannot be negative")
    if not current_active:
        if not base_next_risk_on and metric_difference > entry_difference:
            return GoldOverrideStep(True, "gold_entry", 0)
        return GoldOverrideStep(False, "gold_inactive", 0)
    if completed_gold_days < min_gold_hold_days:
        return GoldOverrideStep(True, "gold_hard_min_hold", completed_gold_days)
    if base_next_risk_on:
        return GoldOverrideStep(
            False,
            "gold_to_momentum_after_min_hold",
            completed_gold_days,
        )
    if metric_difference <= exit_difference:
        return GoldOverrideStep(
            False,
            "gold_to_defender_after_min_hold",
            completed_gold_days,
        )
    return GoldOverrideStep(True, "gold_hold", completed_gold_days)


def _next_open_metrics(context, execution_date: date) -> pd.Series:
    timestamp = pd.Timestamp(execution_date)
    if timestamp <= context.curves.index.max():
        raise ValueError("execution_date must be after the latest close")
    extended = pd.concat(
        [
            context.curves,
            pd.DataFrame([context.curves.iloc[-1].to_dict()], index=[timestamp]),
        ]
    )
    return raw_gold_metrics_at_open(extended).loc[timestamp]


def build_formal_gold_next_open_signal(
    root: Path,
    signal_date: date,
    execution_date: date,
) -> FormalGoldNextOpenSignal:
    return build_next_open_signal(root, signal_date, execution_date)
