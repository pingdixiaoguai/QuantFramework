"""Formal frozen 510300 downside-RAQM Momentum/Defender strategy.

The production rule has one state machine.  It observes only the previous
close's 510300 downside-RAQM percentiles, keeps the validated 30-session lock
on both sleeves, and never applies a Gold, rapid-reversal, or emergency layer.
"""

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
    DownsideRAQMFeatures,
    DownsideRAQMSpec,
    FactorProfile,
    build_downside_raqm_features,
    downside_raqm_state_schedule,
)
from research.momentum_defender_gold_override import (
    GoldOverrideContext,
    build_gold_override_context,
    simulate_candidate_schedule,
)
from research.momentum_defender_integrated import DEFENDER_STRATEGY_ID
from research.momentum_defender_occam import performance
from research.momentum_volatility import load_ohlc
from strategy.momentum_defender import MomentumNextOpenTarget, _momentum_next_open_target


FORMAL_STRATEGY_ID = "momentum_defender_downside_raqm_weighted_v1"
ANCHOR_ASSET = "510300.SH"
PROFILE_ID = "w30_40_25_75"
HORIZONS = (30, 40)
WEIGHTS = (0.25, 0.75)
VOLATILITY_FLOOR_ANNUAL = 0.08
WINSOR_LIMIT = 3.0
PERCENTILE_HISTORY_WINDOW = 504
PERCENTILE_MIN_HISTORY = 252
DEFENDER_ENTRY_PERCENTILE = 0.55
MOMENTUM_RECOVERY_PERCENTILE = 0.20
MOMENTUM_LOCK_DAYS = 30
DEFENDER_LOCK_DAYS = 30
DEFENDER_ENTRY_CONFIRMATION_DAYS = 3
MOMENTUM_RECOVERY_CONFIRMATION_DAYS = 1


@dataclass(frozen=True)
class FormalDownsideRAQMBacktest:
    context: GoldOverrideContext
    features: DownsideRAQMFeatures
    state: pd.DataFrame
    daily: pd.DataFrame
    audit: Mapping[str, object]


@dataclass(frozen=True)
class FormalDownsideRAQMNextOpenSignal:
    strategy_id: str
    defender_strategy_id: str
    signal_date: date
    execution_date: date
    current_model_sleeve: str
    target_sleeve: str
    state_reason: str
    held_days_at_open: int
    downside_raqm_percentile: float
    downside_raqm_30: float
    downside_raqm_40: float
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


def formal_profile() -> FactorProfile:
    return FactorProfile(PROFILE_ID, HORIZONS, WEIGHTS)


def formal_spec() -> DownsideRAQMSpec:
    return DownsideRAQMSpec(
        profile=formal_profile(),
        history_mode="rolling_504_strict_lag",
        entry_percentile=DEFENDER_ENTRY_PERCENTILE,
        exit_percentile=MOMENTUM_RECOVERY_PERCENTILE,
        momentum_lock_days=MOMENTUM_LOCK_DAYS,
        defender_lock_days=DEFENDER_LOCK_DAYS,
        entry_confirmation_days=DEFENDER_ENTRY_CONFIRMATION_DAYS,
        recovery_confirmation_days=MOMENTUM_RECOVERY_CONFIRMATION_DAYS,
    )


def _features(
    calendar: pd.DatetimeIndex,
    *,
    end: date,
) -> DownsideRAQMFeatures:
    profile = formal_profile()
    close = load_ohlc(ANCHOR_ASSET, end)["close"]
    return build_downside_raqm_features(
        close,
        calendar,
        {profile.profile_id: profile},
        {"rolling_504_strict_lag": PERCENTILE_HISTORY_WINDOW},
        min_history=PERCENTILE_MIN_HISTORY,
        volatility_floor_annual=VOLATILITY_FLOOR_ANNUAL,
        winsor_limit=WINSOR_LIMIT,
    )


def _state(features: DownsideRAQMFeatures) -> pd.DataFrame:
    score = features.composite_at_open[
        PROFILE_ID, "rolling_504_strict_lag"
    ]
    return downside_raqm_state_schedule(score, formal_spec())


def run_formal_strategy(root: Path, *, end: date) -> FormalDownsideRAQMBacktest:
    """Replay the immutable formal rule through ``end`` with exact switch legs."""
    context = build_gold_override_context(root, end=end)
    features = _features(context.calendar, end=end)
    state = _state(features)
    requested = context.momentum_target.where(
        state["risk_on"].astype(bool), DEFENDER_CANDIDATE
    )
    daily = simulate_candidate_schedule(
        requested,
        context.interfaces,
        context.initial_previous_candidate,
    )
    entries = state["state_changed"].astype(bool) & ~state["risk_on"].astype(bool)
    audit = {
        "status": "passed",
        "strategy_id": FORMAL_STRATEGY_ID,
        "defender_entries": int(entries.sum()),
        "defender_days": int((~state["risk_on"].astype(bool)).sum()),
        "sleeve_switches": int(state["state_changed"].sum()),
        "candidate_switches": int(daily["switched"].sum()),
        "nav_reconstruction_max_abs_error": float(
            ((1.0 + daily["return"]).cumprod() - daily["nav"]).abs().max()
        ),
        "performance": performance(daily["return"].astype(float)),
    }
    if audit["nav_reconstruction_max_abs_error"] > 1e-12:
        raise AssertionError("formal downside-RAQM NAV reconstruction failed")
    return FormalDownsideRAQMBacktest(context, features, state, daily, audit)


def build_next_open_signal(
    root: Path,
    signal_date: date,
    execution_date: date,
) -> FormalDownsideRAQMNextOpenSignal:
    """Replay history and advance the frozen state exactly one market open."""
    if execution_date <= signal_date:
        raise ValueError("execution date must follow signal date")
    historical = run_formal_strategy(root, end=signal_date)
    context = historical.context
    execution = pd.Timestamp(execution_date)
    extended_calendar = context.calendar.append(pd.DatetimeIndex([execution]))
    features = _features(extended_calendar, end=signal_date)
    state = _state(features)
    momentum = _momentum_next_open_target(root, context.integrated, signal_date)
    defender = build_next_open_target(signal_date, execution_date)
    target_risk_on = bool(state.at[execution, "risk_on"])
    target_weights = dict(
        momentum.effective_weights if target_risk_on else defender.target_weights
    )
    target_cash = 0.0 if target_risk_on else defender.target_cash_weight
    if abs(sum(target_weights.values()) + target_cash - 1.0) > 1e-12:
        raise AssertionError("formal next-open target plus cash must sum to one")
    current_risk_on = bool(historical.state.iloc[-1]["risk_on"])
    raw_30 = features.raw_at_open[30].loc[execution]
    raw_40 = features.raw_at_open[40].loc[execution]
    score = features.composite_at_open[
        PROFILE_ID, "rolling_504_strict_lag"
    ].loc[execution]
    if not all(np.isfinite(value) for value in (raw_30, raw_40, score)):
        raise RuntimeError("formal downside-RAQM next-open factor is unavailable")
    return FormalDownsideRAQMNextOpenSignal(
        strategy_id=FORMAL_STRATEGY_ID,
        defender_strategy_id=DEFENDER_STRATEGY_ID,
        signal_date=signal_date,
        execution_date=execution_date,
        current_model_sleeve="momentum" if current_risk_on else "defender",
        target_sleeve="momentum" if target_risk_on else "defender",
        state_reason=str(state.at[execution, "state_reason"]),
        held_days_at_open=int(state.at[execution, "held_days_at_open"]),
        downside_raqm_percentile=float(score),
        downside_raqm_30=float(raw_30),
        downside_raqm_40=float(raw_40),
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
