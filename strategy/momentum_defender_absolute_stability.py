"""Production no-lock confirmation state with rapid-reversal and raw-Gold layers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from defender.live import DefenderNextOpenTarget, build_next_open_target
from research.defender_curve_momentum import DEFENDER_CANDIDATE
from research.momentum_defender_gold_override import (
    GOLD_ASSET,
    GoldOverrideContext,
    build_gold_override_context,
    simulate_candidate_schedule,
)
from research.momentum_defender_integrated import DEFENDER_STRATEGY_ID
from research.momentum_defender_occam import MOMENTUM_ASSETS, performance
from research.momentum_volatility import asof_previous_close, load_ohlc
from strategy.momentum_defender import MomentumNextOpenTarget, _momentum_next_open_target


FORMAL_STRATEGY_ID = "momentum_defender_confirmation_bridge_raw_gold_v4"
MOMENTUM_TREND_WINDOW = 120
RISK_OFF_CONFIRMATION_DAYS = 20
RISK_ON_CONFIRMATION_DAYS = 10
EMERGENCY_TREND_WINDOW = 5
EMERGENCY_VOLATILITY_WINDOW = 20
EMERGENCY_QUANTILE = 0.95
RAW_GOLD_WINDOW = 5
RAPID_REVERSAL_ENTRY_DIFFERENCE = 2.0
RAPID_REVERSAL_EXIT_DIFFERENCE = 0.75
GOLD_ENTRY_DIFFERENCE = 2.0
GOLD_EXIT_DIFFERENCE = 0.75
GOLD_MIN_HOLD_DAYS = 5


@dataclass(frozen=True)
class FormalStrategyBacktest:
    context: GoldOverrideContext
    indicators: pd.DataFrame
    base_state: pd.DataFrame
    rapid_reversal_metrics: pd.DataFrame
    rapid_reversal_state: pd.DataFrame
    gold_metrics: pd.DataFrame
    gold_state: pd.DataFrame
    base_daily: pd.DataFrame
    daily: pd.DataFrame
    audit: Mapping[str, object]


@dataclass(frozen=True)
class FormalNextOpenSignal:
    strategy_id: str
    defender_strategy_id: str
    signal_date: date
    execution_date: date
    current_model_sleeve: str
    target_sleeve: str
    state_reason: str
    held_days_at_open: int
    risk_on_confirmation_streak: int
    risk_off_confirmation_streak: int
    risk_on_confirmation_days: int
    risk_off_confirmation_days: int
    anchor_log_return_120: float
    held_momentum_asset: str
    held_asset_log_return_120: float
    emergency_log_return_5: float
    emergency_downside_volatility_20: float
    emergency_downside_threshold_q95: float
    emergency_alert: bool
    momentum: MomentumNextOpenTarget
    defender: DefenderNextOpenTarget
    target_weights: Mapping[str, float]
    target_cash_weight: float
    base_target_sleeve: str
    base_state_reason: str
    rapid_reversal_current_active: bool
    rapid_reversal_target_active: bool
    rapid_reversal_asset: str
    rapid_reversal_metric: float
    rapid_reversal_defender_metric: float
    rapid_reversal_difference: float
    rapid_reversal_entry_threshold: float
    rapid_reversal_exit_threshold: float
    gold_current_active: bool
    gold_target_active: bool
    gold_held_days_at_open: int
    gold_metric: float
    defender_metric: float
    metric_difference: float
    gold_entry_threshold: float
    gold_exit_threshold: float
    gold_min_hold_days: int


def _held_feature(panel: pd.DataFrame, held_asset: pd.Series) -> pd.Series:
    result = pd.Series(np.nan, index=panel.index, dtype=float)
    for asset in MOMENTUM_ASSETS:
        held = held_asset.eq(asset)
        result.loc[held] = panel.loc[held, asset].astype(float)
    return result


def _indicator_frame(
    calendar: pd.DatetimeIndex,
    held_asset: pd.Series,
    *,
    end: date,
) -> pd.DataFrame:
    returns_120 = pd.DataFrame(index=calendar, columns=MOMENTUM_ASSETS, dtype=float)
    returns_5 = pd.DataFrame(index=calendar, columns=MOMENTUM_ASSETS, dtype=float)
    downside = pd.DataFrame(index=calendar, columns=MOMENTUM_ASSETS, dtype=float)
    thresholds = pd.DataFrame(index=calendar, columns=MOMENTUM_ASSETS, dtype=float)
    for asset in MOMENTUM_ASSETS:
        prices = load_ohlc(asset, end)
        log_close = np.log(prices["close"].astype(float))
        daily_log = log_close.diff()
        close_return_120 = log_close - log_close.shift(MOMENTUM_TREND_WINDOW)
        close_return_5 = log_close - log_close.shift(EMERGENCY_TREND_WINDOW)
        close_downside = np.sqrt(
            252.0
            * daily_log.clip(upper=0.0)
            .pow(2)
            .rolling(EMERGENCY_VOLATILITY_WINDOW)
            .mean()
        )
        close_threshold = close_downside.shift(1).expanding(
            min_periods=20
        ).quantile(EMERGENCY_QUANTILE)
        returns_120[asset] = asof_previous_close(close_return_120, calendar)
        returns_5[asset] = asof_previous_close(close_return_5, calendar)
        downside[asset] = asof_previous_close(close_downside, calendar)
        thresholds[asset] = asof_previous_close(close_threshold, calendar)
    held_return_120 = _held_feature(returns_120, held_asset)
    held_return_5 = _held_feature(returns_5, held_asset)
    held_downside = _held_feature(downside, held_asset)
    held_threshold = _held_feature(thresholds, held_asset)
    gate = returns_120["510300.SH"].gt(0.0) & held_return_120.gt(0.0)
    emergency = (
        held_return_5.lt(0.0)
        & held_downside.gt(held_threshold)
        & held_threshold.notna()
    )
    return pd.DataFrame(
        {
            "anchor_log_return_120": returns_120["510300.SH"],
            "held_momentum_asset": held_asset.astype(str),
            "held_asset_log_return_120": held_return_120,
            "held_asset_log_return_5": held_return_5,
            "held_asset_downside_volatility_20": held_downside,
            "held_asset_downside_threshold_q95": held_threshold,
            "wanted_risk_on": gate.where(returns_120.notna().any(axis=1)),
            "emergency_alert": emergency.fillna(False),
        },
        index=calendar,
    )


def base_state_schedule(indicators: pd.DataFrame) -> pd.DataFrame:
    """Confirm dual-trend evidence without imposing a sleeve holding lock."""
    state = True
    held_days = 10**9
    on_streak = 0
    off_streak = 0
    rows = []
    for timestamp, indicator in indicators.iterrows():
        wanted = indicator["wanted_risk_on"]
        if pd.isna(wanted):
            on_streak = 0
            off_streak = 0
        elif bool(wanted):
            on_streak += 1
            off_streak = 0
        else:
            off_streak += 1
            on_streak = 0
        previous = state
        reason = "hold"
        if state and bool(indicator["emergency_alert"]):
            state = False
            held_days = 0
            on_streak = 0
            off_streak = 0
            reason = "downside_emergency_exit"
        elif state and off_streak >= RISK_OFF_CONFIRMATION_DAYS:
            state = False
            held_days = 0
            on_streak = 0
            off_streak = 0
            reason = "dual_trend_confirmed_to_defender"
        elif not state and on_streak >= RISK_ON_CONFIRMATION_DAYS:
            state = True
            held_days = 0
            on_streak = 0
            off_streak = 0
            reason = "dual_trend_confirmed_to_momentum"
        rows.append(
            {
                "date": timestamp,
                "risk_on": state,
                "state_changed": state != previous,
                "state_reason": reason,
                "held_days_at_open": held_days,
                "risk_on_streak": on_streak,
                "risk_off_streak": off_streak,
            }
        )
        held_days += 1
    return pd.DataFrame(rows).set_index("date")


def raw_raqm_score(curve: pd.Series, window: int = RAW_GOLD_WINDOW) -> pd.Series:
    values = curve.astype(float)
    daily_log = np.log(values).diff()
    total_log = np.log(values).diff(window)
    path = daily_log.abs().rolling(window).sum()
    efficiency = total_log.abs() / path.replace(0.0, np.nan)
    volatility = daily_log.rolling(window).std(ddof=1) * np.sqrt(window)
    return (total_log / volatility.replace(0.0, np.nan) * efficiency).astype(float)


def raw_gold_metrics_at_open(curves: pd.DataFrame) -> pd.DataFrame:
    result = pd.DataFrame(index=curves.index)
    for candidate in (GOLD_ASSET, DEFENDER_CANDIDATE):
        result[candidate] = raw_raqm_score(curves[candidate]).shift(1)
    result["difference"] = result[GOLD_ASSET] - result[DEFENDER_CANDIDATE]
    return result


def raw_top1_metrics_at_open(
    curves: pd.DataFrame,
    momentum_target: pd.Series,
) -> pd.DataFrame:
    """Compare the causal Momentum Top1 raw RAQM with Defender at each open."""
    scores = pd.DataFrame(index=curves.index)
    for candidate in MOMENTUM_ASSETS:
        scores[candidate] = raw_raqm_score(curves[candidate]).shift(1)
    defender = raw_raqm_score(curves[DEFENDER_CANDIDATE]).shift(1)
    selected = _held_feature(scores, momentum_target.reindex(curves.index))
    return pd.DataFrame(
        {
            "rapid_reversal_asset": momentum_target.reindex(curves.index).astype(str),
            "rapid_reversal_metric_at_open": selected,
            "rapid_reversal_defender_metric_at_open": defender,
            "rapid_reversal_difference_at_open": selected - defender,
        },
        index=curves.index,
    )


def rapid_reversal_state_schedule(
    base_risk_on: pd.Series,
    emergency: pd.Series,
    metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Bridge confirmed Defender to Momentum with evidence hysteresis and no hold."""
    active = False
    rows = []
    for timestamp in metrics.index:
        previous = active
        difference = metrics.at[timestamp, "rapid_reversal_difference_at_open"]
        reason = "hold"
        if bool(base_risk_on.loc[timestamp]):
            if active:
                reason = "rapid_reversal_to_confirmed_momentum"
            active = False
        elif bool(emergency.loc[timestamp]):
            if active:
                reason = "rapid_reversal_emergency_exit"
            active = False
        elif not active:
            if (
                pd.notna(difference)
                and float(difference) > RAPID_REVERSAL_ENTRY_DIFFERENCE
            ):
                active = True
                reason = "rapid_reversal_entry"
        elif pd.isna(difference) or float(difference) <= RAPID_REVERSAL_EXIT_DIFFERENCE:
            active = False
            reason = "rapid_reversal_exit"
        rows.append(
            {
                "date": timestamp,
                "rapid_reversal_active": active,
                "rapid_reversal_changed": active != previous,
                "state_reason": reason,
                "effective_risk_on": bool(base_risk_on.loc[timestamp]) or active,
            }
        )
    return pd.DataFrame(rows).set_index("date")


def gold_state_schedule(
    calendar: pd.DatetimeIndex,
    base_risk_on: pd.Series,
    momentum_target: pd.Series,
    metrics: pd.DataFrame,
) -> pd.DataFrame:
    active = False
    held_days = 10**9
    rows = []
    for timestamp in calendar:
        previous = active
        difference = metrics.at[timestamp, "difference"]
        reason = "hold"
        if not active:
            if (
                not bool(base_risk_on.loc[timestamp])
                and pd.notna(difference)
                and float(difference) > GOLD_ENTRY_DIFFERENCE
            ):
                active = True
                held_days = 0
                reason = "gold_entry"
        elif held_days >= GOLD_MIN_HOLD_DAYS:
            if bool(base_risk_on.loc[timestamp]):
                active = False
                held_days = 0
                reason = "gold_to_momentum_after_min_hold"
            elif pd.notna(difference) and float(difference) <= GOLD_EXIT_DIFFERENCE:
                active = False
                held_days = 0
                reason = "gold_to_defender_after_min_hold"
        else:
            reason = "gold_hard_min_hold"
        if active:
            target = GOLD_ASSET
        elif bool(base_risk_on.loc[timestamp]):
            target = str(momentum_target.loc[timestamp])
        else:
            target = DEFENDER_CANDIDATE
        rows.append(
            {
                "date": timestamp,
                "base_risk_on": bool(base_risk_on.loc[timestamp]),
                "gold_active": active,
                "gold_changed": active != previous,
                "state_reason": reason,
                "gold_held_days_at_open": held_days,
                "gold_metric_at_open": metrics.at[timestamp, GOLD_ASSET],
                "defender_metric_at_open": metrics.at[timestamp, DEFENDER_CANDIDATE],
                "metric_difference_at_open": difference,
                "target_candidate": target,
            }
        )
        if active:
            held_days += 1
    return pd.DataFrame(rows).set_index("date")


def run_formal_strategy(root: Path, *, end: date) -> FormalStrategyBacktest:
    context = build_gold_override_context(root, end=end)
    calendar = context.calendar
    indicators = _indicator_frame(
        calendar,
        context.integrated.result.previous_asset,
        end=end,
    )
    base_state = base_state_schedule(indicators)
    rapid_reversal_metrics = raw_top1_metrics_at_open(
        context.curves,
        context.momentum_target,
    )
    rapid_reversal_state = rapid_reversal_state_schedule(
        base_state["risk_on"],
        indicators["emergency_alert"],
        rapid_reversal_metrics,
    )
    base_target = context.momentum_target.where(
        rapid_reversal_state["effective_risk_on"], DEFENDER_CANDIDATE
    )
    base_daily = simulate_candidate_schedule(
        base_target,
        context.interfaces,
        context.initial_previous_candidate,
    )
    gold_metrics = raw_gold_metrics_at_open(context.curves)
    gold_state = gold_state_schedule(
        calendar,
        rapid_reversal_state["effective_risk_on"],
        context.momentum_target,
        gold_metrics,
    )
    daily = simulate_candidate_schedule(
        gold_state["target_candidate"],
        context.interfaces,
        context.initial_previous_candidate,
    )
    audit = {
        "status": "passed",
        "strategy_id": FORMAL_STRATEGY_ID,
        "base_defender_entries": int(
            (base_state["state_changed"] & ~base_state["risk_on"]).sum()
        ),
        "base_defender_days": int((~base_state["risk_on"]).sum()),
        "base_switches": int(base_state["state_changed"].sum()),
        "rapid_reversal_entries": int(
            rapid_reversal_state["state_reason"].eq("rapid_reversal_entry").sum()
        ),
        "rapid_reversal_days": int(
            rapid_reversal_state["rapid_reversal_active"].sum()
        ),
        "effective_defender_days": int(
            (~rapid_reversal_state["effective_risk_on"]).sum()
        ),
        "gold_entries": int(gold_state["state_reason"].eq("gold_entry").sum()),
        "gold_days": int(gold_state["gold_active"].sum()),
        "switches": int(daily["switched"].sum()),
        "nav_reconstruction_max_abs_error": float(
            ((1.0 + daily["return"]).cumprod() - daily["nav"]).abs().max()
        ),
        "performance": performance(daily["return"].astype(float)),
    }
    if audit["nav_reconstruction_max_abs_error"] > 1e-12:
        raise AssertionError("formal strategy NAV reconstruction failed")
    return FormalStrategyBacktest(
        context,
        indicators,
        base_state,
        rapid_reversal_metrics,
        rapid_reversal_state,
        gold_metrics,
        gold_state,
        base_daily,
        daily,
        audit,
    )


def build_next_open_signal(
    root: Path,
    signal_date: date,
    execution_date: date,
) -> FormalNextOpenSignal:
    historical = run_formal_strategy(root, end=signal_date)
    context = historical.context
    execution = pd.Timestamp(execution_date)
    extended_calendar = context.calendar.append(pd.DatetimeIndex([execution]))
    extended_held = pd.concat(
        [
            context.integrated.result.previous_asset,
            pd.Series(
                [str(context.momentum_target.iloc[-1])],
                index=[execution],
            ),
        ]
    )
    indicators = _indicator_frame(
        extended_calendar,
        extended_held,
        end=signal_date,
    )
    base_state = base_state_schedule(indicators)
    extended_curves = pd.concat(
        [
            context.curves,
            pd.DataFrame([context.curves.iloc[-1].to_dict()], index=[execution]),
        ]
    )
    gold_metrics = raw_gold_metrics_at_open(extended_curves)
    extended_momentum_target = pd.concat(
        [
            context.momentum_target,
            pd.Series([str(context.momentum_target.iloc[-1])], index=[execution]),
        ]
    )
    rapid_reversal_metrics = raw_top1_metrics_at_open(
        extended_curves,
        extended_momentum_target,
    )
    rapid_reversal_state = rapid_reversal_state_schedule(
        base_state["risk_on"],
        indicators["emergency_alert"],
        rapid_reversal_metrics,
    )
    gold_state = gold_state_schedule(
        extended_calendar,
        rapid_reversal_state["effective_risk_on"],
        extended_momentum_target,
        gold_metrics,
    )
    momentum = _momentum_next_open_target(
        root,
        context.integrated,
        signal_date,
    )
    defender = build_next_open_target(signal_date, execution_date)
    target_gold = bool(gold_state.at[execution, "gold_active"])
    target_risk_on = bool(
        rapid_reversal_state.at[execution, "effective_risk_on"]
    )
    if target_gold:
        target_weights = {GOLD_ASSET: 1.0}
        target_cash = 0.0
        target_sleeve = "gold_override"
    elif target_risk_on:
        target_weights = dict(momentum.effective_weights)
        target_cash = 0.0
        target_sleeve = "momentum"
    else:
        target_weights = dict(defender.target_weights)
        target_cash = defender.target_cash_weight
        target_sleeve = "defender"
    if abs(sum(target_weights.values()) + target_cash - 1.0) > 1e-12:
        raise AssertionError("next-open target plus cash must sum to one")
    current_gold = bool(historical.gold_state.iloc[-1]["gold_active"])
    current_risk_on = bool(
        historical.rapid_reversal_state.iloc[-1]["effective_risk_on"]
    )
    current_sleeve = (
        "gold_override"
        if current_gold
        else ("momentum" if current_risk_on else "defender")
    )
    indicator = indicators.loc[execution]
    gold_reason = str(gold_state.at[execution, "state_reason"])
    rapid_reason = str(rapid_reversal_state.at[execution, "state_reason"])
    base_reason = str(base_state.at[execution, "state_reason"])
    state_reason = (
        gold_reason
        if gold_reason != "hold"
        else (rapid_reason if rapid_reason != "hold" else base_reason)
    )
    return FormalNextOpenSignal(
        strategy_id=FORMAL_STRATEGY_ID,
        defender_strategy_id=DEFENDER_STRATEGY_ID,
        signal_date=signal_date,
        execution_date=execution_date,
        current_model_sleeve=current_sleeve,
        target_sleeve=target_sleeve,
        state_reason=state_reason,
        held_days_at_open=int(base_state.at[execution, "held_days_at_open"]),
        risk_on_confirmation_streak=int(
            base_state.at[execution, "risk_on_streak"]
        ),
        risk_off_confirmation_streak=int(
            base_state.at[execution, "risk_off_streak"]
        ),
        risk_on_confirmation_days=RISK_ON_CONFIRMATION_DAYS,
        risk_off_confirmation_days=RISK_OFF_CONFIRMATION_DAYS,
        anchor_log_return_120=float(indicator["anchor_log_return_120"]),
        held_momentum_asset=str(indicator["held_momentum_asset"]),
        held_asset_log_return_120=float(indicator["held_asset_log_return_120"]),
        emergency_log_return_5=float(indicator["held_asset_log_return_5"]),
        emergency_downside_volatility_20=float(
            indicator["held_asset_downside_volatility_20"]
        ),
        emergency_downside_threshold_q95=float(
            indicator["held_asset_downside_threshold_q95"]
        ),
        emergency_alert=bool(indicator["emergency_alert"]),
        momentum=momentum,
        defender=defender,
        target_weights=target_weights,
        target_cash_weight=target_cash,
        base_target_sleeve=(
            "momentum"
            if bool(base_state.at[execution, "risk_on"])
            else "defender"
        ),
        base_state_reason=base_reason,
        rapid_reversal_current_active=bool(
            historical.rapid_reversal_state.iloc[-1]["rapid_reversal_active"]
        ),
        rapid_reversal_target_active=bool(
            rapid_reversal_state.at[execution, "rapid_reversal_active"]
        ),
        rapid_reversal_asset=str(
            rapid_reversal_metrics.at[execution, "rapid_reversal_asset"]
        ),
        rapid_reversal_metric=float(
            rapid_reversal_metrics.at[
                execution, "rapid_reversal_metric_at_open"
            ]
        ),
        rapid_reversal_defender_metric=float(
            rapid_reversal_metrics.at[
                execution, "rapid_reversal_defender_metric_at_open"
            ]
        ),
        rapid_reversal_difference=float(
            rapid_reversal_metrics.at[
                execution, "rapid_reversal_difference_at_open"
            ]
        ),
        rapid_reversal_entry_threshold=RAPID_REVERSAL_ENTRY_DIFFERENCE,
        rapid_reversal_exit_threshold=RAPID_REVERSAL_EXIT_DIFFERENCE,
        gold_current_active=current_gold,
        gold_target_active=target_gold,
        gold_held_days_at_open=int(
            gold_state.at[execution, "gold_held_days_at_open"]
        ),
        gold_metric=float(gold_metrics.at[execution, GOLD_ASSET]),
        defender_metric=float(gold_metrics.at[execution, DEFENDER_CANDIDATE]),
        metric_difference=float(gold_metrics.at[execution, "difference"]),
        gold_entry_threshold=GOLD_ENTRY_DIFFERENCE,
        gold_exit_threshold=GOLD_EXIT_DIFFERENCE,
        gold_min_hold_days=GOLD_MIN_HOLD_DAYS,
    )
