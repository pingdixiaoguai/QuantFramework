"""Formal v4: W40 champion, QM40 Defender, signed-QM40 recovery exit."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from defender.w40_qm_reversal_full_equity import (
    FORMAL_DEFENDER_STRATEGY_ID,
    FormalQMFullEquityDefenderBacktest,
    build_formal_backtest as build_formal_defender,
    build_next_open_target as build_defender_next_open_target,
)
from factors.quality_momentum import compute as quality_momentum
from research.defender_curve_momentum import DEFENDER_CANDIDATE
from research.momentum_defender_downside_raqm import strict_lag_percentile
from research.momentum_defender_gold_override import simulate_candidate_schedule
from research.momentum_defender_occam import MOMENTUM_ASSETS, performance
from research.momentum_defender_w40_asset_specific_escape import (
    AssetSpecificW40EscapeBacktest,
    AssetXYPolicy,
    asset_specific_escape_schedule,
    run_asset_specific_w40_escape,
)
from research.momentum_defender_w40_loss_gate import downside_log_loss
from research.momentum_defender_w40_top1_escape import quality_metrics_at_open
from research.momentum_volatility import asof_previous_close, load_ohlc
from strategy.momentum_defender import MomentumNextOpenTarget, _momentum_next_open_target
from strategy.momentum_defender_w40_full_equity import (
    FORMAL_BACKTEST_START,
    build_formal_context,
)
from strategy.momentum_defender_w40_gold_escape import (
    DEFENDER_ELIGIBILITY_DAYS,
    GOLD_ASSET,
    GOLD_ENTRY_X,
    GOLD_EXIT_Y,
    GOLD_HARD_HOLD_DAYS,
    IMMEDIATE_DEFENDER_ENTRY_GOLD_VETO,
    FormalW40GoldEscapeNextOpenSignal,
)
from strategy.signal_performance import (
    SignalPerformanceSnapshot,
    build_signal_performance,
)


FORMAL_STRATEGY_ID = "momentum_defender_w40_qm40_signed_exit_v4"
W40_WINDOW = 40
W40_PERCENTILE_HISTORY = 756
W40_PERCENTILE_MIN_HISTORY = 252
DEFENDER_ENTRY_PERCENTILE = 0.60
MOMENTUM_RECOVERY_PERCENTILE = 0.35
MOMENTUM_LOCK_DAYS = 30
DEFENDER_FALLBACK_LOCK_DAYS = 30
DEFENDER_MINIMUM_DAYS = 5
QM40_RECOVERY_CONFIRMATION_DAYS = 10


def formal_policies() -> dict[str, AssetXYPolicy | None]:
    result: dict[str, AssetXYPolicy | None] = {
        asset: None for asset in MOMENTUM_ASSETS
    }
    result[GOLD_ASSET] = AssetXYPolicy(GOLD_ENTRY_X, GOLD_EXIT_Y)
    return result


@dataclass(frozen=True)
class FormalW40QM40BaseBacktest:
    context: object
    defender: FormalQMFullEquityDefenderBacktest
    raw_loss_at_open: pd.Series
    score_at_open: pd.Series
    anchor_qm40_at_open: pd.Series
    anchor_log_return40_at_open: pd.Series
    state: pd.DataFrame
    daily: pd.DataFrame
    audit: Mapping[str, object]


@dataclass(frozen=True)
class FormalW40QM40Backtest:
    base: FormalW40QM40BaseBacktest
    context: object
    raw_loss_at_open: pd.Series
    score_at_open: pd.Series
    anchor_qm40_at_open: pd.Series
    anchor_log_return40_at_open: pd.Series
    state: pd.DataFrame
    escape: AssetSpecificW40EscapeBacktest
    daily: pd.DataFrame
    audit: Mapping[str, object]


@dataclass(frozen=True)
class FormalW40QM40NextOpenSignal(FormalW40GoldEscapeNextOpenSignal):
    w40_percentile_history: int
    base_state_reason: str
    base_defender_held_days_before_decision: int
    anchor_qm40_at_open: float
    anchor_log_return40_at_open: float
    qm40_recovery_threshold: float
    qm40_recovery_streak_before_decision: int
    qm40_recovery_confirmation_days: int
    base_defender_minimum_days: int
    qm40_early_exit_qualified: bool
    fallback_recovery_qualified: bool


def _w40_features(
    calendar: pd.DatetimeIndex,
    *,
    end: date,
) -> tuple[pd.Series, pd.Series]:
    close = load_ohlc("510300.SH", end)["close"].astype(float)
    raw = downside_log_loss(close, W40_WINDOW)
    percentile = strict_lag_percentile(
        raw,
        history_window=W40_PERCENTILE_HISTORY,
        min_history=W40_PERCENTILE_MIN_HISTORY,
    )
    return (
        asof_previous_close(raw, calendar).rename(
            "w40_downside_log_loss_at_open"
        ),
        asof_previous_close(percentile, calendar).rename(
            "w40_loss_percentile_at_open"
        ),
    )


def _anchor_qm40_features(
    calendar: pd.DatetimeIndex,
    *,
    end: date,
) -> tuple[pd.Series, pd.Series]:
    close = load_ohlc("510300.SH", end)["close"].astype(float)
    frame = pd.DataFrame({"date": close.index, "close": close.to_numpy(float)})
    qm_close = quality_momentum(frame, {"window": 40})
    log_return_close = np.log(close).diff(40)
    return (
        asof_previous_close(qm_close, calendar).rename(
            "anchor_qm40_at_open"
        ),
        asof_previous_close(log_return_close, calendar).rename(
            "anchor_log_return40_at_open"
        ),
    )


def qm40_recovery_state_schedule(
    score_at_open: pd.Series,
    qm40_at_open: pd.Series,
    *,
    qm40_recovery_threshold: float = 0.0,
) -> pd.DataFrame:
    if not score_at_open.index.equals(qm40_at_open.index):
        raise ValueError("W40 score and QM40 recovery must share one calendar")
    if not np.isfinite(qm40_recovery_threshold):
        raise ValueError("QM40 recovery threshold must be finite")
    risk_on = True
    held_days = 10**9
    recovery_streak = 0
    rows: list[dict[str, object]] = []
    for timestamp, raw_score in score_at_open.items():
        score = float(raw_score) if pd.notna(raw_score) else np.nan
        qm40 = (
            float(qm40_at_open.loc[timestamp])
            if pd.notna(qm40_at_open.loc[timestamp])
            else np.nan
        )
        previous = risk_on
        held_before = held_days
        streak_before = recovery_streak
        early_qualified = False
        fallback_qualified = False
        reason = "hold"
        if not np.isfinite(score):
            recovery_streak = 0
            reason = "insufficient_w40_history"
        elif risk_on:
            recovery_streak = 0
            if score >= DEFENDER_ENTRY_PERCENTILE:
                if held_days >= MOMENTUM_LOCK_DAYS:
                    risk_on = False
                    held_days = 0
                    reason = "w40_to_defender"
                else:
                    reason = "defender_entry_blocked_by_momentum_lock"
        else:
            evidence = (
                np.isfinite(qm40) and qm40 > qm40_recovery_threshold
            )
            recovery_streak = recovery_streak + 1 if evidence else 0
            streak_before = recovery_streak
            early_qualified = bool(
                held_days >= DEFENDER_MINIMUM_DAYS
                and held_days < DEFENDER_FALLBACK_LOCK_DAYS
                and recovery_streak >= QM40_RECOVERY_CONFIRMATION_DAYS
            )
            fallback_qualified = bool(
                held_days >= DEFENDER_FALLBACK_LOCK_DAYS
                and score <= MOMENTUM_RECOVERY_PERCENTILE
            )
            if early_qualified or fallback_qualified:
                risk_on = True
                held_days = 0
                recovery_streak = 0
                reason = (
                    "qm40_recovery_to_momentum"
                    if early_qualified
                    else "w40_fallback_to_momentum"
                )
            elif score <= MOMENTUM_RECOVERY_PERCENTILE:
                reason = "w40_recovery_blocked_by_defender_policy"
        rows.append(
            {
                "date": timestamp,
                "risk_on": risk_on,
                "state_changed": risk_on != previous,
                "state_reason": reason,
                "downside_raqm_percentile_at_open": score,
                "entry_confirmation_streak": 0,
                "recovery_confirmation_streak": recovery_streak,
                "held_days_at_open": held_days,
                "held_days_before_decision": held_before,
                "qm40_at_open": qm40,
                "qm40_recovery_threshold": qm40_recovery_threshold,
                "qm40_recovery_streak_before_decision": streak_before,
                "qm40_early_exit_qualified": early_qualified,
                "fallback_recovery_qualified": fallback_qualified,
            }
        )
        held_days += 1
    return pd.DataFrame(rows).set_index("date")


def run_base_strategy(
    root: Path,
    *,
    end: date,
    start: date = FORMAL_BACKTEST_START,
    qm40_recovery_threshold: float = 0.0,
    base_strategy_id: str = "momentum_defender_w40_qm40_base_v4",
) -> FormalW40QM40BaseBacktest:
    context, defender = build_formal_context(
        root,
        end=end,
        start=start,
        defender_builder=build_formal_defender,
        defender_strategy_id=FORMAL_DEFENDER_STRATEGY_ID,
    )
    raw, score = _w40_features(context.calendar, end=end)
    qm40, log_return40 = _anchor_qm40_features(context.calendar, end=end)
    state = qm40_recovery_state_schedule(
        score,
        qm40,
        qm40_recovery_threshold=qm40_recovery_threshold,
    )
    requested = context.momentum_target.where(
        state["risk_on"].astype(bool), DEFENDER_CANDIDATE
    )
    daily = simulate_candidate_schedule(
        requested,
        context.interfaces,
        context.initial_previous_candidate,
    )
    audit = {
        "status": "passed",
        "strategy_id": base_strategy_id,
        "defender_strategy_id": FORMAL_DEFENDER_STRATEGY_ID,
        "defender_entries": int(
            (state["state_changed"] & ~state["risk_on"]).sum()
        ),
        "momentum_recoveries": int(
            (state["state_changed"] & state["risk_on"]).sum()
        ),
        "qm40_early_recoveries": int(
            state["state_reason"].eq("qm40_recovery_to_momentum").sum()
        ),
        "qm40_recovery_threshold": qm40_recovery_threshold,
        "candidate_switches": int(daily["switched"].sum()),
        "nav_reconstruction_max_abs_error": float(
            ((1.0 + daily["return"]).cumprod() - daily["nav"]).abs().max()
        ),
        "performance": performance(daily["return"].astype(float)),
        "defender_audit": defender.audit,
    }
    if audit["nav_reconstruction_max_abs_error"] > 1e-12:
        raise AssertionError("formal v4 base NAV reconstruction failed")
    return FormalW40QM40BaseBacktest(
        context=context,
        defender=defender,
        raw_loss_at_open=raw,
        score_at_open=score,
        anchor_qm40_at_open=qm40,
        anchor_log_return40_at_open=log_return40,
        state=state,
        daily=daily,
        audit=audit,
    )


def run_formal_strategy(
    root: Path,
    *,
    end: date,
    start: date = FORMAL_BACKTEST_START,
    qm40_recovery_threshold: float = 0.0,
    strategy_id: str = FORMAL_STRATEGY_ID,
    base_strategy_id: str = "momentum_defender_w40_qm40_base_v4",
) -> FormalW40QM40Backtest:
    base = run_base_strategy(
        root,
        end=end,
        start=start,
        qm40_recovery_threshold=qm40_recovery_threshold,
        base_strategy_id=base_strategy_id,
    )
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
        "strategy_id": strategy_id,
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
        "base_audit": base.audit,
        "escape_audit": escape.audit,
    }
    if audit["nav_reconstruction_max_abs_error"] > 1e-12:
        raise AssertionError("formal v4 NAV reconstruction failed")
    return FormalW40QM40Backtest(
        base=base,
        context=base.context,
        raw_loss_at_open=base.raw_loss_at_open,
        score_at_open=base.score_at_open,
        anchor_qm40_at_open=base.anchor_qm40_at_open,
        anchor_log_return40_at_open=base.anchor_log_return40_at_open,
        state=base.state,
        escape=escape,
        daily=daily,
        audit=audit,
    )


def _extended_context(
    historical: FormalW40QM40Backtest,
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
                [momentum_asset],
                index=pd.DatetimeIndex([execution]),
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
    *,
    qm40_recovery_threshold: float = 0.0,
    strategy_id: str = FORMAL_STRATEGY_ID,
    base_strategy_id: str = "momentum_defender_w40_qm40_base_v4",
) -> FormalW40QM40NextOpenSignal:
    if execution_date <= signal_date:
        raise ValueError("execution date must follow signal date")
    historical = run_formal_strategy(
        root,
        end=signal_date,
        qm40_recovery_threshold=qm40_recovery_threshold,
        strategy_id=strategy_id,
        base_strategy_id=base_strategy_id,
    )
    execution = pd.Timestamp(execution_date)
    momentum = _momentum_next_open_target(
        root, historical.context.integrated, signal_date
    )
    momentum_asset = next(iter(momentum.effective_weights))
    extended = _extended_context(historical, execution, momentum_asset)
    raw_loss, score = _w40_features(extended.calendar, end=signal_date)
    qm40, log_return40 = _anchor_qm40_features(
        extended.calendar, end=signal_date
    )
    base_state = qm40_recovery_state_schedule(
        score,
        qm40,
        qm40_recovery_threshold=qm40_recovery_threshold,
    )
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
    if target_candidate == DEFENDER_CANDIDATE:
        target_weights = dict(defender.target_weights)
        target_cash = defender.target_cash_weight
        target_sleeve = "defender"
    else:
        target_weights = {target_candidate: 1.0}
        target_cash = 0.0
        target_sleeve = (
            "gold_escape" if bool(row["escape_active"]) else "momentum"
        )
    if abs(sum(target_weights.values()) + target_cash - 1.0) > 1e-12:
        raise AssertionError("formal v4 target plus cash must sum to one")
    current_candidate = str(historical.daily.iloc[-1]["candidate"])
    current_escape = bool(historical.escape.state.iloc[-1]["escape_active"])
    current_sleeve = (
        "defender"
        if current_candidate == DEFENDER_CANDIDATE
        else "gold_escape" if current_escape else "momentum"
    )
    values = (
        raw_loss.loc[execution],
        score.loc[execution],
        qm40.loc[execution],
        log_return40.loc[execution],
        row["top1_metric_at_open"],
        row["defender_metric_at_open"],
        row["metric_difference_at_open"],
    )
    if not all(np.isfinite(value) for value in values):
        raise RuntimeError("formal v4 next-open metric is unavailable")
    try:
        performance_snapshot = build_signal_performance(
            root, historical, signal_date
        )
        performance_error = None
    except Exception as exc:
        performance_snapshot = None
        performance_error = type(exc).__name__
    base_row = base_state.loc[execution]
    return FormalW40QM40NextOpenSignal(
        strategy_id=strategy_id,
        defender_strategy_id=FORMAL_DEFENDER_STRATEGY_ID,
        signal_date=signal_date,
        execution_date=execution_date,
        current_model_sleeve=current_sleeve,
        target_sleeve=target_sleeve,
        state_reason=str(row["state_reason"]),
        held_days_at_open=int(base_row["held_days_at_open"]),
        w40_downside_log_loss=float(raw_loss.loc[execution]),
        w40_loss_percentile=float(score.loc[execution]),
        defender_entry_percentile=DEFENDER_ENTRY_PERCENTILE,
        momentum_recovery_percentile=MOMENTUM_RECOVERY_PERCENTILE,
        entry_confirmation_streak=0,
        recovery_confirmation_streak=int(
            base_row["recovery_confirmation_streak"]
        ),
        defender_entry_confirmation_days=1,
        momentum_recovery_confirmation_days=1,
        momentum_lock_days=MOMENTUM_LOCK_DAYS,
        defender_lock_days=DEFENDER_FALLBACK_LOCK_DAYS,
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
        w40_percentile_history=W40_PERCENTILE_HISTORY,
        base_state_reason=str(base_row["state_reason"]),
        base_defender_held_days_before_decision=int(
            base_row["held_days_before_decision"]
        ),
        anchor_qm40_at_open=float(qm40.loc[execution]),
        anchor_log_return40_at_open=float(log_return40.loc[execution]),
        qm40_recovery_threshold=qm40_recovery_threshold,
        qm40_recovery_streak_before_decision=int(
            base_row["qm40_recovery_streak_before_decision"]
        ),
        qm40_recovery_confirmation_days=QM40_RECOVERY_CONFIRMATION_DAYS,
        base_defender_minimum_days=DEFENDER_MINIMUM_DAYS,
        qm40_early_exit_qualified=bool(
            base_row["qm40_early_exit_qualified"]
        ),
        fallback_recovery_qualified=bool(
            base_row["fallback_recovery_qualified"]
        ),
    )
