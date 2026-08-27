"""Causal downside-RAQM regime switching for log-quality Momentum.

Only the positive magnitude of a negative regularized RAQM observation is
used.  Every close-known factor value is mapped to the next executable open,
and neither sleeve may bypass its configured 20-30 session lock.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from research.defender_curve_momentum import DEFENDER_CANDIDATE
from research.momentum_defender_gold_override import GoldOverrideContext
from research.momentum_defender_occam import ENTER_RETURN, EXIT_RETURN, HELD_RETURN
from research.momentum_volatility import asof_previous_close


EXPANDING_STRICT_LAG = "expanding_strict_lag"
ROLLING_504_STRICT_LAG = "rolling_504_strict_lag"
SUPPORTED_HISTORY_MODES = {EXPANDING_STRICT_LAG, ROLLING_504_STRICT_LAG}


@dataclass(frozen=True)
class FactorProfile:
    profile_id: str
    horizons: tuple[int, ...]
    weights: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.profile_id:
            raise ValueError("profile_id must be non-empty")
        if not self.horizons or len(self.horizons) != len(self.weights):
            raise ValueError("factor horizons and weights must be non-empty and aligned")
        if any(horizon < 20 for horizon in self.horizons):
            raise ValueError("every downside-RAQM horizon must be at least 20 days")
        if any(weight <= 0.0 for weight in self.weights):
            raise ValueError("factor weights must be positive")
        if not np.isclose(sum(self.weights), 1.0, atol=1e-12):
            raise ValueError("factor weights must sum to one")


@dataclass(frozen=True)
class DownsideRAQMSpec:
    profile: FactorProfile
    history_mode: str
    entry_percentile: float
    exit_percentile: float
    momentum_lock_days: int
    defender_lock_days: int
    entry_confirmation_days: int
    recovery_confirmation_days: int

    def __post_init__(self) -> None:
        if self.history_mode not in SUPPORTED_HISTORY_MODES:
            raise ValueError(f"unsupported history mode: {self.history_mode}")
        if not 0.0 <= self.exit_percentile < self.entry_percentile <= 1.0:
            raise ValueError("exit percentile must be below entry percentile")
        if not 20 <= self.momentum_lock_days <= 30:
            raise ValueError("Momentum lock must lie in [20, 30]")
        if not 20 <= self.defender_lock_days <= 30:
            raise ValueError("Defender lock must lie in [20, 30]")
        if min(self.entry_confirmation_days, self.recovery_confirmation_days) < 1:
            raise ValueError("confirmation counts must be positive")

    def candidate_id(self) -> str:
        history = "exp" if self.history_mode == EXPANDING_STRICT_LAG else "r504"
        return (
            f"draqm_{self.profile.profile_id}_{history}_"
            f"en{self.entry_percentile:.2f}_ex{self.exit_percentile:.2f}_"
            f"mh{self.momentum_lock_days}_dh{self.defender_lock_days}_"
            f"ec{self.entry_confirmation_days}_rc{self.recovery_confirmation_days}"
        )


@dataclass(frozen=True)
class ExactExecutionData:
    calendar: pd.DatetimeIndex
    candidates: tuple[str, ...]
    candidate_index: Mapping[str, int]
    momentum_target: np.ndarray
    held_returns: np.ndarray
    enter_returns: np.ndarray
    exit_returns: np.ndarray
    initial_candidate: int


@dataclass(frozen=True)
class DownsideRAQMFeatures:
    calendar: pd.DatetimeIndex
    raw_at_open: Mapping[int, pd.Series]
    percentile_at_open: Mapping[tuple[int, str], pd.Series]
    composite_at_open: Mapping[tuple[str, str], pd.Series]


@dataclass(frozen=True)
class DownsideRAQMRun:
    spec: DownsideRAQMSpec
    state: pd.DataFrame
    returns: np.ndarray
    requested_target: np.ndarray
    actual_target: np.ndarray
    defender_entries: int
    defender_days: int
    sleeve_switches: int
    candidate_switches: int


def downside_regularized_raqm(
    close: pd.Series,
    window: int,
    *,
    volatility_floor_annual: float = 0.08,
    winsor_limit: float = 3.0,
) -> pd.Series:
    """Return positive strength only when signed regularized RAQM is negative."""
    if window < 20:
        raise ValueError("downside-RAQM window must be at least 20")
    if volatility_floor_annual < 0.0 or winsor_limit <= 0.0:
        raise ValueError("invalid RAQM regularization")
    values = close.astype(float)
    if (values <= 0.0).any():
        raise ValueError("RAQM requires positive close prices")
    log_close = np.log(values)
    daily_log = log_close.diff()
    total_log = log_close.diff(window)
    path = daily_log.abs().rolling(window).sum()
    efficiency = total_log.abs() / path.replace(0.0, np.nan)
    volatility = daily_log.rolling(window).std(ddof=1) * np.sqrt(window)
    floor = volatility_floor_annual * np.sqrt(window / 252.0)
    adjusted = np.maximum(volatility, floor)
    signed = (total_log / adjusted).clip(-winsor_limit, winsor_limit) * efficiency
    result = (-signed).clip(lower=0.0).astype(float)
    result.name = f"downside_raqm_{window}"
    return result


def strict_lag_percentile(
    values: pd.Series,
    *,
    history_window: int | None,
    min_history: int,
) -> pd.Series:
    """Percentile of the current value against prior observations only.

    A zero downside score is assigned percentile zero rather than inheriting
    the large point mass created by non-negative trends.
    """
    if min_history < 1:
        raise ValueError("min_history must be positive")
    if history_window is not None and history_window < min_history:
        raise ValueError("history window cannot be shorter than min_history")
    source = values.astype(float)
    raw = source.to_numpy(float)
    result = np.full(len(source), np.nan, dtype=float)
    finite_positions: list[int] = []
    for position, current in enumerate(raw):
        if not np.isfinite(current):
            continue
        start = 0 if history_window is None else max(0, position - history_window)
        prior_positions = [
            item for item in finite_positions if item >= start
        ]
        if len(prior_positions) >= min_history:
            if current <= 0.0:
                result[position] = 0.0
            else:
                prior = raw[np.asarray(prior_positions, dtype=int)]
                result[position] = float(np.mean(prior <= current))
        finite_positions.append(position)
    return pd.Series(result, index=source.index, name=f"{source.name}_strict_lag_pct")


def build_downside_raqm_features(
    close: pd.Series,
    calendar: pd.DatetimeIndex,
    profiles: Mapping[str, FactorProfile],
    history_modes: Mapping[str, int | None],
    *,
    min_history: int,
    volatility_floor_annual: float,
    winsor_limit: float,
) -> DownsideRAQMFeatures:
    """Build prior-close raw, percentile and weighted composite features."""
    for mode in history_modes:
        if mode not in SUPPORTED_HISTORY_MODES:
            raise ValueError(f"unsupported history mode: {mode}")
    horizons = sorted(
        {horizon for profile in profiles.values() for horizon in profile.horizons}
    )
    raw_close = {
        horizon: downside_regularized_raqm(
            close,
            horizon,
            volatility_floor_annual=volatility_floor_annual,
            winsor_limit=winsor_limit,
        )
        for horizon in horizons
    }
    percentile_close: dict[tuple[int, str], pd.Series] = {}
    raw_open: dict[int, pd.Series] = {}
    percentile_open: dict[tuple[int, str], pd.Series] = {}
    for horizon, values in raw_close.items():
        raw_open[horizon] = asof_previous_close(values, calendar)
        for mode, history_window in history_modes.items():
            percentile = strict_lag_percentile(
                values,
                history_window=history_window,
                min_history=min_history,
            )
            percentile_close[horizon, mode] = percentile
            percentile_open[horizon, mode] = asof_previous_close(
                percentile, calendar
            )
    composites: dict[tuple[str, str], pd.Series] = {}
    for profile in profiles.values():
        for mode in history_modes:
            columns = [
                percentile_open[horizon, mode].rename(str(horizon))
                for horizon in profile.horizons
            ]
            panel = pd.concat(columns, axis=1)
            weighted = panel.mul(np.asarray(profile.weights), axis=1).sum(
                axis=1, min_count=len(profile.horizons)
            )
            weighted.name = f"{profile.profile_id}_{mode}_at_open"
            composites[profile.profile_id, mode] = weighted
    return DownsideRAQMFeatures(
        calendar=calendar,
        raw_at_open=raw_open,
        percentile_at_open=percentile_open,
        composite_at_open=composites,
    )


def build_exact_execution_data(context: GoldOverrideContext) -> ExactExecutionData:
    """Convert frozen candidate interfaces to dense exact-return arrays."""
    candidates = tuple(context.interfaces)
    candidate_index = {candidate: index for index, candidate in enumerate(candidates)}
    momentum_target = context.momentum_target.map(candidate_index)
    if momentum_target.isna().any():
        raise ValueError("Momentum target contains an unknown candidate")
    return ExactExecutionData(
        calendar=context.calendar,
        candidates=candidates,
        candidate_index=candidate_index,
        momentum_target=momentum_target.to_numpy(int),
        held_returns=np.vstack(
            [
                context.interfaces[candidate][HELD_RETURN].to_numpy(float)
                for candidate in candidates
            ]
        ),
        enter_returns=np.vstack(
            [
                context.interfaces[candidate][ENTER_RETURN].to_numpy(float)
                for candidate in candidates
            ]
        ),
        exit_returns=np.vstack(
            [
                context.interfaces[candidate][EXIT_RETURN].to_numpy(float)
                for candidate in candidates
            ]
        ),
        initial_candidate=candidate_index[context.initial_previous_candidate],
    )


def downside_raqm_state_schedule(
    score_at_open: pd.Series,
    spec: DownsideRAQMSpec,
) -> pd.DataFrame:
    """Apply dual thresholds, confirmations and non-bypassable sleeve locks."""
    state = True
    held_days = 10**9
    entry_streak = 0
    recovery_streak = 0
    rows: list[dict[str, object]] = []
    for timestamp, raw_score in score_at_open.items():
        score = float(raw_score) if pd.notna(raw_score) else np.nan
        if not np.isfinite(score):
            entry_streak = 0
            recovery_streak = 0
        else:
            entry_streak = entry_streak + 1 if score >= spec.entry_percentile else 0
            recovery_streak = (
                recovery_streak + 1 if score <= spec.exit_percentile else 0
            )
        previous = state
        reason = "hold"
        if not np.isfinite(score):
            reason = "insufficient_factor_history"
        elif state and entry_streak >= spec.entry_confirmation_days:
            if held_days >= spec.momentum_lock_days:
                state = False
                held_days = 0
                entry_streak = 0
                recovery_streak = 0
                reason = "downside_raqm_to_defender"
            else:
                reason = "defender_entry_blocked_by_momentum_lock"
        elif not state and recovery_streak >= spec.recovery_confirmation_days:
            if held_days >= spec.defender_lock_days:
                state = True
                held_days = 0
                entry_streak = 0
                recovery_streak = 0
                reason = "downside_raqm_to_momentum"
            else:
                reason = "momentum_recovery_blocked_by_defender_lock"
        rows.append(
            {
                "date": timestamp,
                "risk_on": state,
                "state_changed": state != previous,
                "state_reason": reason,
                "downside_raqm_percentile_at_open": score,
                "entry_confirmation_streak": entry_streak,
                "recovery_confirmation_streak": recovery_streak,
                "held_days_at_open": held_days,
            }
        )
        held_days += 1
    return pd.DataFrame(rows).set_index("date")


def exact_candidate_schedule(
    data: ExactExecutionData,
    requested_target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Execute requested candidates with exact exit and entry return legs."""
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
        if not np.isfinite(returns[position]) or returns[position] <= -1.0:
            raise ValueError(f"invalid executed return at position {position}")
        actual[position] = current
    return returns, actual, switches


def run_downside_raqm_spec(
    data: ExactExecutionData,
    features: DownsideRAQMFeatures,
    spec: DownsideRAQMSpec,
) -> DownsideRAQMRun:
    score = features.composite_at_open[spec.profile.profile_id, spec.history_mode]
    state = downside_raqm_state_schedule(score, spec)
    risk_on = state["risk_on"].to_numpy(bool)
    defender = data.candidate_index[DEFENDER_CANDIDATE]
    requested = np.where(risk_on, data.momentum_target, defender).astype(int)
    returns, actual, candidate_switches = exact_candidate_schedule(data, requested)
    entries = state["state_changed"].astype(bool) & ~state["risk_on"].astype(bool)
    return DownsideRAQMRun(
        spec=spec,
        state=state,
        returns=returns,
        requested_target=requested,
        actual_target=actual,
        defender_entries=int(entries.sum()),
        defender_days=int((~state["risk_on"].astype(bool)).sum()),
        sleeve_switches=int(state["state_changed"].sum()),
        candidate_switches=candidate_switches,
    )
