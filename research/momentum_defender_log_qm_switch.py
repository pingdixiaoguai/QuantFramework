"""Causal switching primitives for the frozen log-MOM/log-ER Momentum sleeve."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping

import numpy as np
import pandas as pd

from research.defender_curve_momentum import DEFENDER_CANDIDATE
from research.gold_min5_risk_adjusted_momentum import risk_adjusted_momentum_at_open
from research.gold_min5_risk_adjusted_momentum_w5 import (
    GoldRAQMW5Params,
    run_gold_raqm_w5,
)
from research.momentum_defender_gold_override import (
    GoldOverrideContext,
    simulate_candidate_schedule,
)
from research.momentum_defender_occam import (
    MOMENTUM_ASSETS,
    apply_state_schedule,
    simulate_switch,
)
from research.momentum_volatility import asof_previous_close


SIMPLE_RETURN = "simple"
LOG_RETURN = "log"
ROGERS_SATCHELL = "rogers_satchell"
CLOSE_LOG_STD = "close_log_std"
EXPANDING_HISTORY = "expanding_strict_lag"
ROLLING_HISTORY = "rolling_504_strict_lag"
NO_EMERGENCY = "no_emergency"


@dataclass(frozen=True)
class SwitchSpec:
    slow_return_mode: str
    slow_lookback: int
    slow_threshold: float
    min_hold_days: int
    emergency_enabled: bool
    volatility_estimator: str = ROGERS_SATCHELL
    volatility_window: int = 10
    quantile_history: str = EXPANDING_HISTORY
    quantile_scheme: str = "current_asset_specific"
    cap_trigger_maximum: float = 0.80

    def __post_init__(self) -> None:
        if self.slow_return_mode not in {SIMPLE_RETURN, LOG_RETURN}:
            raise ValueError("unsupported slow return mode")
        if self.slow_lookback < 2 or self.min_hold_days < 1:
            raise ValueError("invalid slow lookback or holding period")
        if self.emergency_enabled:
            if self.volatility_estimator not in {ROGERS_SATCHELL, CLOSE_LOG_STD}:
                raise ValueError("unsupported volatility estimator")
            if self.volatility_window < 2:
                raise ValueError("volatility window must be at least two")
            if self.quantile_history not in {EXPANDING_HISTORY, ROLLING_HISTORY}:
                raise ValueError("unsupported quantile history")
            if not 0.0 <= self.cap_trigger_maximum <= 1.0:
                raise ValueError("cap trigger must lie in [0, 1]")

    def candidate_id(self) -> str:
        slow = (
            f"slow_{self.slow_return_mode}_w{self.slow_lookback}_"
            f"t{self.slow_threshold:+.4f}_h{self.min_hold_days}"
        )
        if not self.emergency_enabled:
            return f"{slow}__{NO_EMERGENCY}"
        return (
            f"{slow}__{self.volatility_estimator}_w{self.volatility_window}_"
            f"{self.quantile_history}_{self.quantile_scheme}_"
            f"cap{self.cap_trigger_maximum:.1f}"
        )


@dataclass(frozen=True)
class SwitchRun:
    spec: SwitchSpec
    state: pd.DataFrame
    base_daily: pd.DataFrame
    formal_state: pd.DataFrame
    formal_daily: pd.DataFrame
    audit: Mapping[str, object]


@dataclass(frozen=True)
class FastSwitchData:
    """Array-backed immutable inputs shared by every searched candidate."""

    calendar: pd.DatetimeIndex
    candidates: tuple[str, ...]
    candidate_index: Mapping[str, int]
    momentum_target: np.ndarray
    held_returns: np.ndarray
    enter_returns: np.ndarray
    exit_returns: np.ndarray
    initial_candidate: int
    gold_difference: np.ndarray


@dataclass(frozen=True)
class FastSwitchResult:
    returns: np.ndarray
    risk_on: np.ndarray
    target_candidate: np.ndarray
    emergency_entries: int
    defender_days: int
    base_switches: int
    gold_entries: int
    gold_days: int
    formal_switches: int


def slow_regime_at_open(
    close: pd.Series,
    calendar: pd.DatetimeIndex,
    *,
    mode: str,
    lookback: int,
    threshold: float,
) -> pd.Series:
    """Return a strictly previous-close slow regime on the execution calendar."""
    values = close.astype(float).sort_index()
    if mode == SIMPLE_RETURN:
        trailing = values / values.shift(lookback) - 1.0
    elif mode == LOG_RETURN:
        trailing = np.log(values / values.shift(lookback))
    else:
        raise ValueError(f"unsupported slow return mode: {mode}")
    known_after_close = trailing.gt(threshold).where(trailing.notna())
    return known_after_close.shift(1).reindex(calendar).ffill().rename(
        "slow_regime_at_open"
    )


def realized_volatility(
    prices: pd.DataFrame,
    *,
    estimator: str,
    window: int,
) -> pd.Series:
    """Calculate an annualized close-known volatility series."""
    frame = prices[["open", "high", "low", "close"]].astype(float)
    if (frame <= 0.0).any().any():
        raise ValueError("volatility estimators require positive OHLC")
    if estimator == ROGERS_SATCHELL:
        variance = (
            np.log(frame["high"] / frame["close"])
            * np.log(frame["high"] / frame["open"])
            + np.log(frame["low"] / frame["close"])
            * np.log(frame["low"] / frame["open"])
        ).clip(lower=0.0)
        result = np.sqrt(252.0 * variance.rolling(window).mean())
    elif estimator == CLOSE_LOG_STD:
        result = np.log(frame["close"]).diff().rolling(window).std(ddof=1) * np.sqrt(
            252.0
        )
    else:
        raise ValueError(f"unsupported volatility estimator: {estimator}")
    result.name = f"{estimator}_{window}"
    return result


def strict_lag_volatility_cap(
    volatility: pd.Series,
    quantile: float,
    *,
    history: str,
    step: float = 0.20,
    minimum_history: int = 20,
    rolling_history: int = 504,
) -> pd.DataFrame:
    """Build a discretized cap whose quantile never includes the current close."""
    values = volatility.astype(float)
    lagged = values.shift(1)
    if history == EXPANDING_HISTORY:
        threshold = lagged.expanding(min_periods=minimum_history).quantile(quantile)
    elif history == ROLLING_HISTORY:
        threshold = lagged.rolling(
            rolling_history, min_periods=minimum_history
        ).quantile(quantile)
    else:
        raise ValueError(f"unsupported quantile history: {history}")
    raw = (threshold / values).clip(upper=1.0)
    cap = (np.floor(raw / step + 1e-12) * step).clip(0.0, 1.0)
    cap = cap.where(raw.notna(), 1.0)
    return pd.DataFrame(
        {"volatility": values, "threshold": threshold, "raw_cap": raw, "cap": cap}
    )


def held_asset_alert(
    caps: Mapping[str, pd.Series],
    previous_asset: pd.Series,
    trigger: float,
) -> pd.Series:
    """Select only the cap of the Momentum asset owned through prior close."""
    result = pd.Series(False, index=previous_asset.index, dtype=bool)
    for asset in MOMENTUM_ASSETS:
        held = previous_asset.eq(asset)
        selected = caps[asset].reindex(result.index)
        if selected.loc[held].isna().any():
            raise ValueError(f"missing held-asset cap for {asset}")
        result.loc[held] = selected.loc[held].le(trigger)
    return result.rename("emergency_alert_at_open")


def build_fast_switch_data(
    context: GoldOverrideContext,
    gold_metrics: pd.DataFrame,
) -> FastSwitchData:
    """Convert fixed candidate interfaces to dense arrays once per experiment."""
    from research.momentum_defender_occam import ENTER_RETURN, EXIT_RETURN, HELD_RETURN

    candidates = tuple(context.interfaces)
    candidate_index = {candidate: index for index, candidate in enumerate(candidates)}
    momentum_target = context.momentum_target.map(candidate_index).to_numpy(int)
    held = np.vstack(
        [context.interfaces[candidate][HELD_RETURN].to_numpy(float) for candidate in candidates]
    )
    enter = np.vstack(
        [context.interfaces[candidate][ENTER_RETURN].to_numpy(float) for candidate in candidates]
    )
    exit_ = np.vstack(
        [context.interfaces[candidate][EXIT_RETURN].to_numpy(float) for candidate in candidates]
    )
    return FastSwitchData(
        calendar=context.calendar,
        candidates=candidates,
        candidate_index=candidate_index,
        momentum_target=momentum_target,
        held_returns=held,
        enter_returns=enter,
        exit_returns=exit_,
        initial_candidate=candidate_index[context.initial_previous_candidate],
        gold_difference=gold_metrics["difference"].to_numpy(float),
    )


def fast_state_schedule(
    slow: np.ndarray,
    emergency: np.ndarray,
    min_hold_days: int,
) -> tuple[np.ndarray, int, int]:
    """Array implementation of the exact C2 state lock and emergency priority."""
    risk_on = np.empty(len(slow), dtype=bool)
    state = True
    held_days = 10**9
    emergency_entries = 0
    switches = 0
    for position in range(len(slow)):
        previous = state
        if bool(emergency[position]):
            if state:
                state = False
                held_days = 0
                emergency_entries += 1
        else:
            wanted = slow[position]
            if not pd.isna(wanted) and bool(wanted) != state and held_days >= min_hold_days:
                state = bool(wanted)
                held_days = 0
        risk_on[position] = state
        switches += int(state != previous)
        held_days += 1
    return risk_on, emergency_entries, switches


def fast_gold_targets(
    data: FastSwitchData,
    risk_on: np.ndarray,
    *,
    entry_difference: float = 2.20,
    exit_difference: float = 0.60,
    minimum_hold_days: int = 5,
) -> tuple[np.ndarray, int, int]:
    """Apply the frozen hard-five-day Gold state to an arbitrary C2 schedule."""
    target = np.empty(len(risk_on), dtype=int)
    gold = data.candidate_index["518880.SH"]
    defender = data.candidate_index[DEFENDER_CANDIDATE]
    active = False
    held_days = 10**9
    entries = 0
    active_days = 0
    for position in range(len(risk_on)):
        difference = data.gold_difference[position]
        if not active:
            if (
                not bool(risk_on[position])
                and np.isfinite(difference)
                and float(difference) > entry_difference
            ):
                active = True
                held_days = 0
                entries += 1
        elif held_days >= minimum_hold_days:
            if bool(risk_on[position]) or (
                np.isfinite(difference) and float(difference) <= exit_difference
            ):
                active = False
                held_days = 0
        if active:
            target[position] = gold
            active_days += 1
            held_days += 1
        elif bool(risk_on[position]):
            target[position] = data.momentum_target[position]
        else:
            target[position] = defender
    return target, entries, active_days


def fast_candidate_schedule(
    data: FastSwitchData,
    requested_target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Execute target candidates with exact held and open switch return legs."""
    returns = np.empty(len(requested_target), dtype=float)
    actual = np.empty(len(requested_target), dtype=int)
    current = int(data.initial_candidate)
    switches = 0
    for position, requested_value in enumerate(requested_target):
        requested = int(requested_value)
        switched = requested != current
        if switched and (
            not np.isfinite(data.exit_returns[current, position])
            or not np.isfinite(data.enter_returns[requested, position])
        ):
            switched = False
            requested = current
        if switched:
            returns[position] = (
                (1.0 + data.exit_returns[current, position])
                * (1.0 + data.enter_returns[requested, position])
                - 1.0
            )
            current = requested
            switches += 1
        else:
            returns[position] = data.held_returns[current, position]
        actual[position] = current
    return returns, actual, switches


def fast_run_switch_spec(
    data: FastSwitchData,
    spec: SwitchSpec,
    slow: pd.Series,
    emergency: pd.Series | None,
) -> FastSwitchResult:
    """Fast exact-return path used during parameter search."""
    emergency_values = (
        np.zeros(len(data.calendar), dtype=bool)
        if emergency is None
        else emergency.reindex(data.calendar).fillna(False).to_numpy(bool)
    )
    risk_on, emergency_entries, base_switches = fast_state_schedule(
        slow.reindex(data.calendar).to_numpy(),
        emergency_values,
        spec.min_hold_days,
    )
    target, gold_entries, gold_days = fast_gold_targets(data, risk_on)
    returns, actual, formal_switches = fast_candidate_schedule(data, target)
    return FastSwitchResult(
        returns=returns,
        risk_on=risk_on,
        target_candidate=actual,
        emergency_entries=emergency_entries,
        defender_days=int((~risk_on).sum()),
        base_switches=base_switches,
        gold_entries=gold_entries,
        gold_days=gold_days,
        formal_switches=formal_switches,
    )


def candidate_gold_context(
    context: GoldOverrideContext,
    state: pd.DataFrame,
    simulated: pd.DataFrame,
) -> GoldOverrideContext:
    """Replace only the base-C2 state while retaining fixed interfaces and curves."""
    replaced_result = replace(
        context.integrated.result,
        state=state,
        simulated=simulated,
    )
    replaced_integrated = replace(context.integrated, result=replaced_result)
    baseline_target = context.momentum_target.where(
        state["risk_on"].astype(bool), DEFENDER_CANDIDATE
    ).rename("baseline_target_at_open")
    replay = simulate_candidate_schedule(
        baseline_target,
        context.interfaces,
        context.initial_previous_candidate,
    )
    parity = float(
        (replay["return"].astype(float) - simulated["return"].astype(float))
        .abs()
        .max()
    )
    if parity > 5e-8:
        raise AssertionError(f"candidate base replay parity failed: {parity:.3e}")
    return replace(
        context,
        integrated=replaced_integrated,
        baseline_target=baseline_target,
        baseline_parity_max_abs_error=parity,
    )


def run_switch_spec(
    context: GoldOverrideContext,
    spec: SwitchSpec,
    caps: Mapping[str, pd.Series] | None,
    *,
    gold_metrics: pd.DataFrame | None = None,
) -> SwitchRun:
    """Run one causal switch rule and the frozen formal Gold overlay."""
    calendar = context.calendar
    slow = slow_regime_at_open(
        context.integrated.result.inputs.risk_close,
        calendar,
        mode=spec.slow_return_mode,
        lookback=spec.slow_lookback,
        threshold=spec.slow_threshold,
    )
    if spec.emergency_enabled:
        if caps is None:
            raise ValueError("enabled emergency requires cap series")
        emergency = held_asset_alert(
            caps,
            context.integrated.result.previous_asset,
            spec.cap_trigger_maximum,
        )
    else:
        emergency = pd.Series(False, index=calendar, name="emergency_alert_at_open")
    state = apply_state_schedule(
        slow,
        emergency,
        calendar,
        spec.min_hold_days,
        emergency_override=True,
    )
    base_daily = simulate_switch(
        context.integrated.result.inputs.momentum,
        context.integrated.result.inputs.defender,
        state["risk_on"],
    )
    candidate_context = candidate_gold_context(context, state, base_daily)
    metrics = (
        risk_adjusted_momentum_at_open(candidate_context.curves, window=5)
        if gold_metrics is None
        else gold_metrics
    )
    formal = run_gold_raqm_w5(
        candidate_context,
        GoldRAQMW5Params(2.20, 0.60),
        metrics=metrics,
    )
    audit = {
        "candidate_id": spec.candidate_id(),
        "emergency_entries": int(
            (
                state["state_changed"].astype(bool)
                & state["state_reason"].eq("emergency_exit")
            ).sum()
        ),
        "defender_days": int((~state["risk_on"].astype(bool)).sum()),
        "base_switches": int(base_daily["sleeve_switch"].sum()),
        "gold_entries": int(formal.audit["gold_entries"]),
        "gold_days": int(formal.audit["gold_days"]),
        "formal_switches": int(formal.audit["switches"]),
        "base_replay_parity_max_abs_error": candidate_context.baseline_parity_max_abs_error,
    }
    return SwitchRun(
        spec=spec,
        state=state,
        base_daily=base_daily,
        formal_state=formal.state,
        formal_daily=formal.daily,
        audit=audit,
    )


def pareto_frontier(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    """Return rows not dominated across metrics where larger is always better."""
    values = frame[columns].to_numpy(float)
    frontier = np.ones(len(frame), dtype=bool)
    for index in range(len(frame)):
        if not frontier[index]:
            continue
        dominated = np.all(values >= values[index], axis=1) & np.any(
            values > values[index], axis=1
        )
        if dominated.any():
            frontier[index] = False
    return pd.Series(frontier, index=frame.index, name="pareto_frontier")
