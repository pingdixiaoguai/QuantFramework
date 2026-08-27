"""Formal W40/full-equity strategy with Gold-only QM20 lock escape."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from defender.live import DefenderNextOpenTarget
from defender.w40_reversal_full_equity import (
    FORMAL_DEFENDER_STRATEGY_ID,
    build_next_open_target as build_defender_next_open_target,
)
from research.momentum_defender_downside_raqm import downside_raqm_state_schedule
from research.momentum_defender_occam import MOMENTUM_ASSETS, performance
from research.momentum_defender_w40_asset_specific_escape import (
    AssetSpecificW40EscapeBacktest,
    AssetXYPolicy,
    asset_specific_escape_schedule,
    run_asset_specific_w40_escape,
)
from research.momentum_defender_w40_top1_escape import quality_metrics_at_open
from strategy.momentum_defender import MomentumNextOpenTarget, _momentum_next_open_target
from strategy.signal_performance import (
    SignalPerformanceSnapshot,
    build_signal_performance,
)
from strategy.momentum_defender_w40_full_equity import (
    FORMAL_BACKTEST_START,
    FormalW40FullEquityBacktest,
    _features,
    run_formal_strategy as run_base_formal,
)
from strategy.momentum_defender_w40_loss import (
    DEFENDER_ENTRY_CONFIRMATION_DAYS,
    DEFENDER_ENTRY_PERCENTILE,
    DEFENDER_LOCK_DAYS,
    MOMENTUM_LOCK_DAYS,
    MOMENTUM_RECOVERY_CONFIRMATION_DAYS,
    MOMENTUM_RECOVERY_PERCENTILE,
    formal_spec,
)


FORMAL_STRATEGY_ID = "momentum_defender_w40_gold_qm20_escape_v3"
GOLD_ASSET = "518880.SH"
GOLD_ENTRY_X = 0.005
GOLD_EXIT_Y = -0.020
DEFENDER_ELIGIBILITY_DAYS = 5
GOLD_HARD_HOLD_DAYS = 5
IMMEDIATE_DEFENDER_ENTRY_GOLD_VETO = True


def formal_policies() -> dict[str, AssetXYPolicy | None]:
    result: dict[str, AssetXYPolicy | None] = {
        asset: None for asset in MOMENTUM_ASSETS
    }
    result[GOLD_ASSET] = AssetXYPolicy(GOLD_ENTRY_X, GOLD_EXIT_Y)
    return result


@dataclass(frozen=True)
class FormalW40GoldEscapeBacktest:
    base: FormalW40FullEquityBacktest
    context: object
    raw_loss_at_open: pd.Series
    score_at_open: pd.Series
    state: pd.DataFrame
    escape: AssetSpecificW40EscapeBacktest
    daily: pd.DataFrame
    audit: Mapping[str, object]


@dataclass(frozen=True)
class FormalW40GoldEscapeNextOpenSignal:
    strategy_id: str
    defender_strategy_id: str
    signal_date: date
    execution_date: date
    current_model_sleeve: str
    target_sleeve: str
    state_reason: str
    held_days_at_open: int
    w40_downside_log_loss: float
    w40_loss_percentile: float
    defender_entry_percentile: float
    momentum_recovery_percentile: float
    entry_confirmation_streak: int
    recovery_confirmation_streak: int
    defender_entry_confirmation_days: int
    momentum_recovery_confirmation_days: int
    momentum_lock_days: int
    defender_lock_days: int
    escape_active: bool
    escape_entry: bool
    escape_return_to_defender: bool
    escape_entry_asset: str | None
    escape_held_days_at_open: int
    actual_defender_held_days_at_open: int
    momentum_top1: str
    top1_metric_at_open: float
    defender_metric_at_open: float
    metric_difference_at_open: float
    gold_entry_x: float
    gold_exit_y: float
    defender_eligibility_days: int
    gold_hard_hold_days: int
    immediate_defender_entry_gold_veto_enabled: bool
    immediate_defender_entry_gold_veto_triggered: bool
    current_candidate: str
    target_candidate: str
    momentum: MomentumNextOpenTarget
    defender: DefenderNextOpenTarget
    target_weights: Mapping[str, float]
    target_cash_weight: float
    performance_snapshot: SignalPerformanceSnapshot | None
    performance_error: str | None


def run_formal_strategy(
    root: Path,
    *,
    end: date,
    start: date = FORMAL_BACKTEST_START,
) -> FormalW40GoldEscapeBacktest:
    base = run_base_formal(root, end=end, start=start)
    metrics = quality_metrics_at_open(base.context)
    escape = run_asset_specific_w40_escape(
        base.context,
        base.state,
        formal_policies(),
        metrics=metrics,
        immediate_entry_veto=IMMEDIATE_DEFENDER_ENTRY_GOLD_VETO,
    )
    daily = escape.daily
    audit = {
        "status": "passed",
        "strategy_id": FORMAL_STRATEGY_ID,
        "defender_strategy_id": FORMAL_DEFENDER_STRATEGY_ID,
        "base_strategy_id": base.audit["strategy_id"],
        "escape_entries": escape.audit["escape_entries"],
        "lock_break_entries": escape.audit["lock_break_entries"],
        "escape_days": escape.audit["escape_days"],
        "immediate_entry_veto_entries": escape.audit[
            "immediate_entry_veto_entries"
        ],
        "candidate_switches": int(daily["switched"].sum()),
        "nav_reconstruction_max_abs_error": float(
            ((1.0 + daily["return"]).cumprod() - daily["nav"]).abs().max()
        ),
        "performance": performance(daily["return"].astype(float)),
        "base_w40_audit": base.audit,
        "escape_audit": escape.audit,
    }
    if audit["nav_reconstruction_max_abs_error"] > 1e-12:
        raise AssertionError("formal Gold escape NAV reconstruction failed")
    return FormalW40GoldEscapeBacktest(
        base=base,
        context=base.context,
        raw_loss_at_open=base.raw_loss_at_open,
        score_at_open=base.score_at_open,
        state=base.state,
        escape=escape,
        daily=daily,
        audit=audit,
    )


def _extended_context(
    historical: FormalW40GoldEscapeBacktest,
    execution: pd.Timestamp,
    momentum_asset: str,
):
    context = historical.context
    calendar = context.calendar.append(pd.DatetimeIndex([execution]))
    curves = context.curves.reindex(calendar).ffill()
    momentum_target = pd.concat(
        [
            context.momentum_target,
            pd.Series(
                [momentum_asset], index=pd.DatetimeIndex([execution]),
                name=context.momentum_target.name,
            ),
        ]
    )
    return replace(
        context,
        calendar=calendar,
        curves=curves,
        momentum_target=momentum_target,
    )


def build_next_open_signal(
    root: Path,
    signal_date: date,
    execution_date: date,
) -> FormalW40GoldEscapeNextOpenSignal:
    if execution_date <= signal_date:
        raise ValueError("execution date must follow signal date")
    historical = run_formal_strategy(root, end=signal_date)
    execution = pd.Timestamp(execution_date)
    momentum = _momentum_next_open_target(
        root, historical.context.integrated, signal_date
    )
    momentum_asset = next(iter(momentum.effective_weights))
    extended = _extended_context(historical, execution, momentum_asset)
    raw_loss, score = _features(extended.calendar, end=signal_date)
    base_state = downside_raqm_state_schedule(score, formal_spec().state_spec())
    metrics = quality_metrics_at_open(extended)
    escape_state = asset_specific_escape_schedule(
        extended,
        base_state,
        metrics,
        formal_policies(),
        immediate_entry_veto=IMMEDIATE_DEFENDER_ENTRY_GOLD_VETO,
    )
    row = escape_state.loc[execution]
    target_candidate = str(row["target_candidate"])
    defender = build_defender_next_open_target(signal_date, execution_date)
    if target_candidate == "DEFENDER":
        target_weights = dict(defender.target_weights)
        target_cash = defender.target_cash_weight
        target_sleeve = "defender"
    else:
        target_weights = {target_candidate: 1.0}
        target_cash = 0.0
        target_sleeve = (
            "gold_escape"
            if bool(row["escape_active"])
            else "momentum"
        )
    if abs(sum(target_weights.values()) + target_cash - 1.0) > 1e-12:
        raise AssertionError("formal Gold escape target plus cash must sum to one")
    current_candidate = str(historical.daily.iloc[-1]["candidate"])
    current_escape = bool(historical.escape.state.iloc[-1]["escape_active"])
    current_sleeve = (
        "defender"
        if current_candidate == "DEFENDER"
        else "gold_escape" if current_escape else "momentum"
    )
    values = (
        raw_loss.loc[execution],
        score.loc[execution],
        row["top1_metric_at_open"],
        row["defender_metric_at_open"],
        row["metric_difference_at_open"],
    )
    if not all(np.isfinite(value) for value in values):
        raise RuntimeError("formal Gold escape next-open metric is unavailable")
    try:
        performance_snapshot = build_signal_performance(
            root, historical, signal_date
        )
        performance_error = None
    except Exception as exc:
        performance_snapshot = None
        performance_error = type(exc).__name__
    return FormalW40GoldEscapeNextOpenSignal(
        strategy_id=FORMAL_STRATEGY_ID,
        defender_strategy_id=FORMAL_DEFENDER_STRATEGY_ID,
        signal_date=signal_date,
        execution_date=execution_date,
        current_model_sleeve=current_sleeve,
        target_sleeve=target_sleeve,
        state_reason=str(row["state_reason"]),
        held_days_at_open=int(base_state.at[execution, "held_days_at_open"]),
        w40_downside_log_loss=float(raw_loss.loc[execution]),
        w40_loss_percentile=float(score.loc[execution]),
        defender_entry_percentile=DEFENDER_ENTRY_PERCENTILE,
        momentum_recovery_percentile=MOMENTUM_RECOVERY_PERCENTILE,
        entry_confirmation_streak=int(
            base_state.at[execution, "entry_confirmation_streak"]
        ),
        recovery_confirmation_streak=int(
            base_state.at[execution, "recovery_confirmation_streak"]
        ),
        defender_entry_confirmation_days=DEFENDER_ENTRY_CONFIRMATION_DAYS,
        momentum_recovery_confirmation_days=MOMENTUM_RECOVERY_CONFIRMATION_DAYS,
        momentum_lock_days=MOMENTUM_LOCK_DAYS,
        defender_lock_days=DEFENDER_LOCK_DAYS,
        escape_active=bool(row["escape_active"]),
        escape_entry=bool(row["escape_entry"]),
        escape_return_to_defender=bool(row["escape_return_to_defender"]),
        escape_entry_asset=(
            str(row["escape_entry_asset"])
            if pd.notna(row["escape_entry_asset"])
            else None
        ),
        escape_held_days_at_open=int(row["escape_held_days_at_open"]),
        actual_defender_held_days_at_open=int(
            row["actual_defender_held_days_at_open"]
        ),
        momentum_top1=str(row["momentum_top1"]),
        top1_metric_at_open=float(row["top1_metric_at_open"]),
        defender_metric_at_open=float(row["defender_metric_at_open"]),
        metric_difference_at_open=float(row["metric_difference_at_open"]),
        gold_entry_x=GOLD_ENTRY_X,
        gold_exit_y=GOLD_EXIT_Y,
        defender_eligibility_days=DEFENDER_ELIGIBILITY_DAYS,
        gold_hard_hold_days=GOLD_HARD_HOLD_DAYS,
        immediate_defender_entry_gold_veto_enabled=(
            IMMEDIATE_DEFENDER_ENTRY_GOLD_VETO
        ),
        immediate_defender_entry_gold_veto_triggered=bool(
            row["immediate_entry_veto_qualified"]
        ),
        current_candidate=current_candidate,
        target_candidate=target_candidate,
        momentum=momentum,
        defender=defender,
        target_weights=target_weights,
        target_cash_weight=target_cash,
        performance_snapshot=performance_snapshot,
        performance_error=performance_error,
    )
