"""Formal single-W40 510300 downside-log-loss Momentum/Defender strategy."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from defender.live import DefenderNextOpenTarget, build_next_open_target
from research.defender_curve_momentum import DEFENDER_CANDIDATE
from research.momentum_defender_downside_raqm import (
    build_exact_execution_data,
    downside_raqm_state_schedule,
)
from research.momentum_defender_gold_override import (
    GoldOverrideContext,
    build_gold_override_context,
    simulate_candidate_schedule,
)
from research.momentum_defender_integrated import DEFENDER_STRATEGY_ID
from research.momentum_defender_occam import performance
from research.momentum_defender_w40_loss_gate import (
    W40LossGateSpec,
    run_w40_loss_gate,
    w40_loss_percentile_at_open,
)
from research.momentum_volatility import load_ohlc
from strategy.momentum_defender import MomentumNextOpenTarget, _momentum_next_open_target


FORMAL_STRATEGY_ID = "momentum_defender_w40_loss_excluding_extremes_v1"
ANCHOR_ASSET = "510300.SH"
WINDOW = 40
HISTORY_WINDOW = 504
MIN_HISTORY = 252
DEFENDER_ENTRY_PERCENTILE = 0.55
MOMENTUM_RECOVERY_PERCENTILE = 0.40
DEFENDER_ENTRY_CONFIRMATION_DAYS = 1
MOMENTUM_RECOVERY_CONFIRMATION_DAYS = 1
MOMENTUM_LOCK_DAYS = 30
DEFENDER_LOCK_DAYS = 30


@dataclass(frozen=True)
class FormalW40LossBacktest:
    context: GoldOverrideContext
    raw_loss_at_open: pd.Series
    score_at_open: pd.Series
    state: pd.DataFrame
    daily: pd.DataFrame
    audit: Mapping[str, object]


@dataclass(frozen=True)
class FormalW40LossNextOpenSignal:
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
    momentum: MomentumNextOpenTarget
    defender: DefenderNextOpenTarget
    target_weights: Mapping[str, float]
    target_cash_weight: float


def formal_spec() -> W40LossGateSpec:
    return W40LossGateSpec(
        entry_percentile=DEFENDER_ENTRY_PERCENTILE,
        recovery_percentile=MOMENTUM_RECOVERY_PERCENTILE,
        entry_confirmation_days=DEFENDER_ENTRY_CONFIRMATION_DAYS,
        recovery_confirmation_days=MOMENTUM_RECOVERY_CONFIRMATION_DAYS,
        momentum_lock_days=MOMENTUM_LOCK_DAYS,
        defender_lock_days=DEFENDER_LOCK_DAYS,
    )


def _features(
    calendar: pd.DatetimeIndex,
    *,
    end: date,
) -> tuple[pd.Series, pd.Series]:
    close = load_ohlc(ANCHOR_ASSET, end)["close"]
    return w40_loss_percentile_at_open(
        close,
        calendar,
        history_window=HISTORY_WINDOW,
        min_history=MIN_HISTORY,
    )


def _state(score_at_open: pd.Series) -> pd.DataFrame:
    return downside_raqm_state_schedule(score_at_open, formal_spec().state_spec())


def run_formal_strategy(root: Path, *, end: date) -> FormalW40LossBacktest:
    """Replay the immutable W40 rule through ``end`` with exact switch legs."""
    context = build_gold_override_context(root, end=end)
    data = build_exact_execution_data(context)
    raw_loss, score = _features(context.calendar, end=end)
    run = run_w40_loss_gate(data, score, formal_spec())
    requested = pd.Series(
        [data.candidates[value] for value in run.requested_target],
        index=data.calendar,
    )
    daily = simulate_candidate_schedule(
        requested,
        context.interfaces,
        context.initial_previous_candidate,
    )
    parity = float(
        np.max(np.abs(daily["return"].to_numpy(float) - run.returns))
    )
    audit = {
        "status": "passed",
        "strategy_id": FORMAL_STRATEGY_ID,
        "defender_entries": run.defender_entries,
        "defender_days": run.defender_days,
        "sleeve_switches": run.sleeve_switches,
        "candidate_switches": run.candidate_switches,
        "dense_exact_return_parity_max_abs_error": parity,
        "nav_reconstruction_max_abs_error": float(
            ((1.0 + daily["return"]).cumprod() - daily["nav"]).abs().max()
        ),
        "performance": performance(daily["return"].astype(float)),
    }
    if parity > 1e-14 or audit["nav_reconstruction_max_abs_error"] > 1e-12:
        raise AssertionError("formal W40 loss execution parity failed")
    return FormalW40LossBacktest(
        context=context,
        raw_loss_at_open=raw_loss,
        score_at_open=score,
        state=run.state,
        daily=daily,
        audit=audit,
    )


def build_next_open_signal(
    root: Path,
    signal_date: date,
    execution_date: date,
) -> FormalW40LossNextOpenSignal:
    """Replay history and advance the frozen W40 state one market open."""
    if execution_date <= signal_date:
        raise ValueError("execution date must follow signal date")
    historical = run_formal_strategy(root, end=signal_date)
    context = historical.context
    execution = pd.Timestamp(execution_date)
    calendar = context.calendar.append(pd.DatetimeIndex([execution]))
    raw_loss, score = _features(calendar, end=signal_date)
    state = _state(score)
    momentum = _momentum_next_open_target(root, context.integrated, signal_date)
    defender = build_next_open_target(signal_date, execution_date)
    target_risk_on = bool(state.at[execution, "risk_on"])
    target_weights = dict(
        momentum.effective_weights if target_risk_on else defender.target_weights
    )
    target_cash = 0.0 if target_risk_on else defender.target_cash_weight
    if abs(sum(target_weights.values()) + target_cash - 1.0) > 1e-12:
        raise AssertionError("formal W40 next-open target plus cash must sum to one")
    values = (raw_loss.loc[execution], score.loc[execution])
    if not all(np.isfinite(value) for value in values):
        raise RuntimeError("formal W40 next-open factor is unavailable")
    current_risk_on = bool(historical.state.iloc[-1]["risk_on"])
    return FormalW40LossNextOpenSignal(
        strategy_id=FORMAL_STRATEGY_ID,
        defender_strategy_id=DEFENDER_STRATEGY_ID,
        signal_date=signal_date,
        execution_date=execution_date,
        current_model_sleeve="momentum" if current_risk_on else "defender",
        target_sleeve="momentum" if target_risk_on else "defender",
        state_reason=str(state.at[execution, "state_reason"]),
        held_days_at_open=int(state.at[execution, "held_days_at_open"]),
        w40_downside_log_loss=float(raw_loss.loc[execution]),
        w40_loss_percentile=float(score.loc[execution]),
        defender_entry_percentile=DEFENDER_ENTRY_PERCENTILE,
        momentum_recovery_percentile=MOMENTUM_RECOVERY_PERCENTILE,
        entry_confirmation_streak=int(
            state.at[execution, "entry_confirmation_streak"]
        ),
        recovery_confirmation_streak=int(
            state.at[execution, "recovery_confirmation_streak"]
        ),
        defender_entry_confirmation_days=DEFENDER_ENTRY_CONFIRMATION_DAYS,
        momentum_recovery_confirmation_days=MOMENTUM_RECOVERY_CONFIRMATION_DAYS,
        momentum_lock_days=MOMENTUM_LOCK_DAYS,
        defender_lock_days=DEFENDER_LOCK_DAYS,
        momentum=momentum,
        defender=defender,
        target_weights=target_weights,
        target_cash_weight=target_cash,
    )
