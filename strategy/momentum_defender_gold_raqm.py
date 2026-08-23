"""Formal next-open signal for C2 plus the frozen Gold RAQM-W5 override."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Mapping

import pandas as pd

from research.gold_min5_risk_adjusted_momentum import (
    risk_adjusted_momentum_at_open,
)
from research.gold_min5_risk_adjusted_momentum_w5 import (
    GoldRAQMW5Params,
    run_gold_raqm_w5,
)
from research.momentum_defender_gold_override import (
    GOLD_ASSET,
    build_gold_override_context,
)
from strategy.momentum_defender import (
    IntegratedNextOpenSignal,
    MomentumNextOpenTarget,
    build_integrated_next_open_signal_from_backtest,
)
from defender.live import DefenderNextOpenTarget


FORMAL_STRATEGY_ID = "momentum_defender_c2_gold_raqm_w5_v1"
ENTRY_DIFFERENCE = 2.20
EXIT_DIFFERENCE = 0.60
MIN_GOLD_HOLD_DAYS = 5


@dataclass(frozen=True)
class GoldOverrideStep:
    active: bool
    reason: str
    held_days_at_open: int


@dataclass(frozen=True)
class FormalGoldNextOpenSignal:
    strategy_id: str
    defender_strategy_id: str
    signal_date: date
    execution_date: date
    current_model_sleeve: str
    target_sleeve: str
    state_reason: str
    held_days_at_open: int
    slow_gate_return: float
    slow_gate_risk_on: bool
    emergency_asset: str
    emergency_cap: float
    emergency_alert: bool
    momentum: MomentumNextOpenTarget
    defender: DefenderNextOpenTarget
    target_weights: Mapping[str, float]
    target_cash_weight: float
    base_c2_target_sleeve: str
    base_c2_state_reason: str
    gold_current_active: bool
    gold_target_active: bool
    gold_held_days_at_open: int
    gold_metric: float
    defender_metric: float
    metric_difference: float
    gold_entry_threshold: float
    gold_exit_threshold: float
    gold_min_hold_days: int


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
    """Advance the formal Gold state exactly one market open."""
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
            pd.DataFrame(
                [context.curves.iloc[-1].to_dict()], index=[timestamp]
            ),
        ]
    )
    return risk_adjusted_momentum_at_open(extended, window=5).loc[timestamp]


def build_formal_gold_next_open_signal(
    root: Path,
    signal_date: date,
    execution_date: date,
) -> FormalGoldNextOpenSignal:
    """Replay the frozen strategy and calculate one causal next-open target."""
    context = build_gold_override_context(root, end=signal_date)
    if context.calendar.max().date() != signal_date:
        raise RuntimeError("formal replay does not end on signal_date")
    params = GoldRAQMW5Params(ENTRY_DIFFERENCE, EXIT_DIFFERENCE)
    historical = run_gold_raqm_w5(context, params)
    base = build_integrated_next_open_signal_from_backtest(
        root,
        context.integrated,
        signal_date,
        execution_date,
    )
    latest_state = historical.state.iloc[-1]
    current_active = bool(latest_state["gold_active"])
    completed_days = (
        int(latest_state["gold_held_days_at_open"]) + 1
        if current_active
        else 0
    )
    metrics = _next_open_metrics(context, execution_date)
    difference = float(metrics["difference"])
    step = advance_gold_override(
        current_active=current_active,
        completed_gold_days=completed_days,
        base_next_risk_on=base.target_sleeve == "momentum",
        metric_difference=difference,
    )

    if step.active:
        target_weights = {GOLD_ASSET: 1.0}
        target_cash = 0.0
        target_sleeve = "gold_override"
    else:
        target_weights = dict(base.target_weights)
        target_cash = base.target_cash_weight
        target_sleeve = base.target_sleeve
    if abs(sum(target_weights.values()) + target_cash - 1.0) > 1e-12:
        raise AssertionError("formal Gold target plus cash must sum to one")

    if current_active:
        current_sleeve = "gold_override"
    else:
        current_sleeve = base.current_model_sleeve
    return FormalGoldNextOpenSignal(
        strategy_id=FORMAL_STRATEGY_ID,
        defender_strategy_id=base.defender_strategy_id,
        signal_date=signal_date,
        execution_date=execution_date,
        current_model_sleeve=current_sleeve,
        target_sleeve=target_sleeve,
        state_reason=step.reason,
        held_days_at_open=base.held_days_at_open,
        slow_gate_return=base.slow_gate_return,
        slow_gate_risk_on=base.slow_gate_risk_on,
        emergency_asset=base.emergency_asset,
        emergency_cap=base.emergency_cap,
        emergency_alert=base.emergency_alert,
        momentum=base.momentum,
        defender=base.defender,
        target_weights=target_weights,
        target_cash_weight=target_cash,
        base_c2_target_sleeve=base.target_sleeve,
        base_c2_state_reason=base.state_reason,
        gold_current_active=current_active,
        gold_target_active=step.active,
        gold_held_days_at_open=step.held_days_at_open,
        gold_metric=float(metrics[GOLD_ASSET]),
        defender_metric=float(metrics["DEFENDER"]),
        metric_difference=difference,
        gold_entry_threshold=ENTRY_DIFFERENCE,
        gold_exit_threshold=EXIT_DIFFERENCE,
        gold_min_hold_days=MIN_GOLD_HOLD_DAYS,
    )
