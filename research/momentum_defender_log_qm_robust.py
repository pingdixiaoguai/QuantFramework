"""Broader robust regime mechanisms for frozen log-MOM/log-ER Momentum."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from research.momentum_defender_log_qm_switch import (
    EXPANDING_HISTORY,
    ROLLING_HISTORY,
    FastSwitchData,
    fast_candidate_schedule,
    fast_gold_targets,
)
from research.momentum_defender_occam import MOMENTUM_ASSETS
from research.momentum_volatility import asof_previous_close, load_ohlc


ANCHOR = "anchor"
HELD = "held"
ANCHOR_OR_HELD = "anchor_or_held"
ANCHOR_AND_HELD = "anchor_and_held"
BREADTH = "breadth"
ANCHOR_OR_BREADTH = "anchor_or_breadth"
HELD_OR_BREADTH = "held_or_breadth"
TREND_VOTE = "trend_vote"
RELATIVE_VOTE = "relative_vote"
COMBINED_VOTE = "combined_vote"

NO_EMERGENCY = "none"
ONE_DAY_LOSS = "one_day_loss"
DRAWDOWN = "drawdown"
DOWNSIDE_VOL = "downside_vol"
NEGATIVE_RS_VOL = "negative_rs_vol"


@dataclass(frozen=True)
class StatePolicy:
    policy_id: str
    min_momentum_days: int
    min_defender_days: int
    risk_off_confirmation: int
    risk_on_confirmation: int

    def __post_init__(self) -> None:
        if min(
            self.min_momentum_days,
            self.min_defender_days,
            self.risk_off_confirmation,
            self.risk_on_confirmation,
        ) < 1:
            raise ValueError("state-policy values must be positive")


@dataclass(frozen=True)
class GateSpec:
    mode: str
    lookback: int
    return_threshold: float
    breadth_required: int
    policy: StatePolicy

    def __post_init__(self) -> None:
        if self.mode not in {
            ANCHOR,
            HELD,
            ANCHOR_OR_HELD,
            ANCHOR_AND_HELD,
            BREADTH,
            ANCHOR_OR_BREADTH,
            HELD_OR_BREADTH,
        }:
            raise ValueError("unsupported gate mode")
        if self.lookback < 2 or not 1 <= self.breadth_required <= 4:
            raise ValueError("invalid gate lookback or breadth")

    def candidate_id(self) -> str:
        return (
            f"gate_{self.mode}_w{self.lookback}_t{self.return_threshold:+.3f}_"
            f"b{self.breadth_required}_{self.policy.policy_id}"
        )


@dataclass(frozen=True)
class EmergencySpec:
    mode: str = NO_EMERGENCY
    window: int = 1
    threshold: float = 0.0
    quantile: float = 0.90
    history: str = EXPANDING_HISTORY

    def __post_init__(self) -> None:
        if self.mode not in {
            NO_EMERGENCY,
            ONE_DAY_LOSS,
            DRAWDOWN,
            DOWNSIDE_VOL,
            NEGATIVE_RS_VOL,
        }:
            raise ValueError("unsupported emergency mode")
        if self.window < 1:
            raise ValueError("emergency window must be positive")
        if self.mode in {DOWNSIDE_VOL, NEGATIVE_RS_VOL}:
            if not 0.0 < self.quantile < 1.0:
                raise ValueError("emergency quantile must lie in (0, 1)")
            if self.history not in {EXPANDING_HISTORY, ROLLING_HISTORY}:
                raise ValueError("unsupported emergency history")

    def candidate_id(self) -> str:
        if self.mode == NO_EMERGENCY:
            return "em_none"
        if self.mode in {ONE_DAY_LOSS, DRAWDOWN}:
            return f"em_{self.mode}_w{self.window}_t{self.threshold:+.3f}"
        return (
            f"em_{self.mode}_w{self.window}_q{self.quantile:.2f}_{self.history}"
        )


@dataclass(frozen=True)
class RobustSpec:
    gate: GateSpec
    emergency: EmergencySpec

    def candidate_id(self) -> str:
        return f"{self.gate.candidate_id()}__{self.emergency.candidate_id()}"


@dataclass(frozen=True)
class EnsembleGateSpec:
    mode: str
    horizons: tuple[int, ...]
    vote_fraction: float
    relative_threshold: float
    policy: StatePolicy

    def __post_init__(self) -> None:
        if self.mode not in {TREND_VOTE, RELATIVE_VOTE, COMBINED_VOTE}:
            raise ValueError("unsupported ensemble mode")
        if not self.horizons or any(value < 2 for value in self.horizons):
            raise ValueError("invalid ensemble horizons")
        if not 0.0 < self.vote_fraction <= 1.0:
            raise ValueError("vote fraction must lie in (0, 1]")

    def candidate_id(self) -> str:
        horizons = "-".join(map(str, self.horizons))
        return (
            f"ensemble_{self.mode}_w{horizons}_v{self.vote_fraction:.2f}_"
            f"rt{self.relative_threshold:+.3f}_{self.policy.policy_id}"
        )


@dataclass(frozen=True)
class RobustEnsembleSpec:
    gate: EnsembleGateSpec
    emergency: EmergencySpec

    def candidate_id(self) -> str:
        return f"{self.gate.candidate_id()}__{self.emergency.candidate_id()}"


@dataclass(frozen=True)
class FeatureBundle:
    calendar: pd.DatetimeIndex
    previous_asset: pd.Series
    log_returns: Mapping[int, pd.DataFrame]
    drawdowns: Mapping[int, pd.DataFrame]
    downside_alerts: Mapping[tuple[int, float, str], pd.DataFrame]
    rs_alerts: Mapping[tuple[int, float, str], pd.DataFrame]
    relative_returns: Mapping[int, pd.Series]


@dataclass(frozen=True)
class RobustFastResult:
    returns: np.ndarray
    risk_on: np.ndarray
    target_candidate: np.ndarray
    defender_entries: int
    defender_days: int
    base_switches: int
    gold_entries: int
    gold_days: int
    formal_switches: int


def _strict_lag_threshold(
    values: pd.Series,
    quantile: float,
    history: str,
    *,
    minimum_history: int,
    rolling_history: int,
) -> pd.Series:
    lagged = values.shift(1)
    if history == EXPANDING_HISTORY:
        return lagged.expanding(min_periods=minimum_history).quantile(quantile)
    if history == ROLLING_HISTORY:
        return lagged.rolling(
            rolling_history, min_periods=minimum_history
        ).quantile(quantile)
    raise ValueError(f"unsupported quantile history: {history}")


def _rs_volatility(prices: pd.DataFrame, window: int) -> pd.Series:
    frame = prices[["open", "high", "low", "close"]].astype(float)
    variance = (
        np.log(frame["high"] / frame["close"])
        * np.log(frame["high"] / frame["open"])
        + np.log(frame["low"] / frame["close"])
        * np.log(frame["low"] / frame["open"])
    ).clip(lower=0.0)
    return np.sqrt(252.0 * variance.rolling(window).mean())


def build_feature_bundle(
    calendar: pd.DatetimeIndex,
    previous_asset: pd.Series,
    *,
    end,
    return_lookbacks: list[int],
    drawdown_windows: list[int],
    volatility_windows: list[int],
    quantiles: list[float],
    histories: list[str],
    minimum_history: int,
    rolling_history: int,
    momentum_curve: pd.Series | None = None,
    defender_curve: pd.Series | None = None,
) -> FeatureBundle:
    """Build strictly previous-close cross-asset features and emergency flags."""
    log_returns: dict[int, pd.DataFrame] = {
        lookback: pd.DataFrame(index=calendar, columns=MOMENTUM_ASSETS, dtype=float)
        for lookback in return_lookbacks
    }
    drawdowns: dict[int, pd.DataFrame] = {
        window: pd.DataFrame(index=calendar, columns=MOMENTUM_ASSETS, dtype=float)
        for window in drawdown_windows
    }
    downside_alerts: dict[tuple[int, float, str], pd.DataFrame] = {
        (window, quantile, history): pd.DataFrame(
            False, index=calendar, columns=MOMENTUM_ASSETS, dtype=bool
        )
        for window in volatility_windows
        for quantile in quantiles
        for history in histories
    }
    rs_alerts = {
        key: frame.copy() for key, frame in downside_alerts.items()
    }
    relative_returns: dict[int, pd.Series] = {}
    for asset in MOMENTUM_ASSETS:
        prices = load_ohlc(asset, end)
        log_close = np.log(prices["close"].astype(float))
        daily_log = log_close.diff()
        for lookback in return_lookbacks:
            close_signal = log_close - log_close.shift(lookback)
            log_returns[lookback][asset] = asof_previous_close(
                close_signal, calendar
            )
        for window in drawdown_windows:
            close_signal = prices["close"] / prices["close"].rolling(window).max() - 1.0
            drawdowns[window][asset] = asof_previous_close(close_signal, calendar)
        for window in volatility_windows:
            downside = np.sqrt(
                252.0
                * daily_log.clip(upper=0.0).pow(2).rolling(window).mean()
            )
            rs = _rs_volatility(prices, window)
            for quantile in quantiles:
                for history in histories:
                    key = (window, quantile, history)
                    downside_line = _strict_lag_threshold(
                        downside,
                        quantile,
                        history,
                        minimum_history=minimum_history,
                        rolling_history=rolling_history,
                    )
                    rs_line = _strict_lag_threshold(
                        rs,
                        quantile,
                        history,
                        minimum_history=minimum_history,
                        rolling_history=rolling_history,
                    )
                    downside_alerts[key][asset] = asof_previous_close(
                        (downside > downside_line).astype(float), calendar
                    ).fillna(0.0).astype(bool)
                    rs_alerts[key][asset] = asof_previous_close(
                        (rs > rs_line).astype(float), calendar
                    ).fillna(0.0).astype(bool)
    if momentum_curve is not None and defender_curve is not None:
        momentum_log = np.log(momentum_curve.astype(float))
        defender_log = np.log(defender_curve.astype(float))
        for lookback in return_lookbacks:
            relative_returns[lookback] = (
                (momentum_log - momentum_log.shift(lookback))
                - (defender_log - defender_log.shift(lookback))
            ).shift(1).reindex(calendar).ffill()
    return FeatureBundle(
        calendar=calendar,
        previous_asset=previous_asset,
        log_returns=log_returns,
        drawdowns=drawdowns,
        downside_alerts=downside_alerts,
        rs_alerts=rs_alerts,
        relative_returns=relative_returns,
    )


def held_feature(panel: pd.DataFrame, previous_asset: pd.Series) -> pd.Series:
    result = pd.Series(np.nan, index=panel.index, dtype=float)
    for asset in MOMENTUM_ASSETS:
        held = previous_asset.eq(asset)
        result.loc[held] = panel.loc[held, asset].astype(float)
    return result


def gate_signal(features: FeatureBundle, spec: GateSpec) -> pd.Series:
    panel = features.log_returns[spec.lookback]
    anchor = panel["510300.SH"].gt(spec.return_threshold)
    held = held_feature(panel, features.previous_asset).gt(spec.return_threshold)
    breadth = panel.gt(0.0).sum(axis=1).ge(spec.breadth_required)
    if spec.mode == ANCHOR:
        wanted = anchor
    elif spec.mode == HELD:
        wanted = held
    elif spec.mode == ANCHOR_OR_HELD:
        wanted = anchor | held
    elif spec.mode == ANCHOR_AND_HELD:
        wanted = anchor & held
    elif spec.mode == BREADTH:
        wanted = breadth
    elif spec.mode == ANCHOR_OR_BREADTH:
        wanted = anchor | breadth
    elif spec.mode == HELD_OR_BREADTH:
        wanted = held | breadth
    else:
        raise ValueError(f"unsupported gate mode: {spec.mode}")
    valid = panel.notna().any(axis=1)
    return wanted.where(valid).rename("wanted_risk_on")


def ensemble_gate_signal(
    features: FeatureBundle,
    spec: EnsembleGateSpec,
) -> pd.Series:
    trend_votes: list[pd.Series] = []
    relative_votes: list[pd.Series] = []
    valid = pd.Series(False, index=features.calendar)
    for horizon in spec.horizons:
        panel = features.log_returns[horizon]
        valid |= panel.notna().any(axis=1)
        trend_votes.extend(
            [
                panel["510300.SH"].gt(0.0),
                held_feature(panel, features.previous_asset).gt(0.0),
                panel.gt(0.0).sum(axis=1).ge(2),
            ]
        )
        if horizon not in features.relative_returns:
            raise ValueError("relative curve feature missing for ensemble horizon")
        relative_votes.append(
            features.relative_returns[horizon].gt(spec.relative_threshold)
        )
    if spec.mode == TREND_VOTE:
        votes = trend_votes
    elif spec.mode == RELATIVE_VOTE:
        votes = relative_votes
    else:
        votes = [*trend_votes, *relative_votes]
    required = int(np.ceil(len(votes) * spec.vote_fraction - 1e-12))
    wanted = pd.concat(votes, axis=1).sum(axis=1).ge(required)
    return wanted.where(valid).rename("wanted_risk_on")


def emergency_signal(
    features: FeatureBundle,
    spec: EmergencySpec,
    *,
    negative_trend_window: int,
) -> pd.Series:
    if spec.mode == NO_EMERGENCY:
        return pd.Series(False, index=features.calendar, dtype=bool)
    if spec.mode == ONE_DAY_LOSS:
        values = held_feature(features.log_returns[1], features.previous_asset)
        return values.le(spec.threshold).fillna(False)
    if spec.mode == DRAWDOWN:
        values = held_feature(features.drawdowns[spec.window], features.previous_asset)
        return values.le(spec.threshold).fillna(False)
    key = (spec.window, spec.quantile, spec.history)
    alerts = (
        features.downside_alerts[key]
        if spec.mode == DOWNSIDE_VOL
        else features.rs_alerts[key]
    )
    selected = held_feature(alerts.astype(float), features.previous_asset).gt(0.5)
    negative = held_feature(
        features.log_returns[negative_trend_window], features.previous_asset
    ).lt(0.0)
    return (selected & negative).fillna(False)


def asymmetric_state_schedule(
    wanted: pd.Series,
    emergency: pd.Series,
    policy: StatePolicy,
) -> tuple[np.ndarray, int, int]:
    """Apply asymmetric holds and independent entry/exit confirmation counts."""
    wanted_values = wanted.to_numpy()
    emergency_values = emergency.reindex(wanted.index).fillna(False).to_numpy(bool)
    risk_on = np.empty(len(wanted), dtype=bool)
    state = True
    held_days = 10**9
    on_streak = 0
    off_streak = 0
    defender_entries = 0
    switches = 0
    for position in range(len(wanted_values)):
        value = wanted_values[position]
        if pd.isna(value):
            on_streak = 0
            off_streak = 0
        elif bool(value):
            on_streak += 1
            off_streak = 0
        else:
            off_streak += 1
            on_streak = 0
        previous = state
        if state and emergency_values[position]:
            state = False
            held_days = 0
            defender_entries += 1
            on_streak = 0
            off_streak = 0
        elif state:
            if (
                off_streak >= policy.risk_off_confirmation
                and held_days >= policy.min_momentum_days
            ):
                state = False
                held_days = 0
                defender_entries += 1
                on_streak = 0
                off_streak = 0
        elif (
            on_streak >= policy.risk_on_confirmation
            and held_days >= policy.min_defender_days
        ):
            state = True
            held_days = 0
            on_streak = 0
            off_streak = 0
        risk_on[position] = state
        switches += int(state != previous)
        held_days += 1
    return risk_on, defender_entries, switches


def run_robust_spec(
    data: FastSwitchData,
    features: FeatureBundle,
    spec: RobustSpec,
    *,
    negative_trend_window: int,
) -> RobustFastResult:
    wanted = gate_signal(features, spec.gate)
    emergency = emergency_signal(
        features,
        spec.emergency,
        negative_trend_window=negative_trend_window,
    )
    risk_on, entries, base_switches = asymmetric_state_schedule(
        wanted, emergency, spec.gate.policy
    )
    target, gold_entries, gold_days = fast_gold_targets(data, risk_on)
    returns, actual, formal_switches = fast_candidate_schedule(data, target)
    return RobustFastResult(
        returns=returns,
        risk_on=risk_on,
        target_candidate=actual,
        defender_entries=entries,
        defender_days=int((~risk_on).sum()),
        base_switches=base_switches,
        gold_entries=gold_entries,
        gold_days=gold_days,
        formal_switches=formal_switches,
    )


def run_ensemble_spec(
    data: FastSwitchData,
    features: FeatureBundle,
    spec: RobustEnsembleSpec,
    *,
    negative_trend_window: int,
) -> RobustFastResult:
    wanted = ensemble_gate_signal(features, spec.gate)
    emergency = emergency_signal(
        features,
        spec.emergency,
        negative_trend_window=negative_trend_window,
    )
    risk_on, entries, base_switches = asymmetric_state_schedule(
        wanted, emergency, spec.gate.policy
    )
    target, gold_entries, gold_days = fast_gold_targets(data, risk_on)
    returns, actual, formal_switches = fast_candidate_schedule(data, target)
    return RobustFastResult(
        returns=returns,
        risk_on=risk_on,
        target_candidate=actual,
        defender_entries=entries,
        defender_days=int((~risk_on).sum()),
        base_switches=base_switches,
        gold_entries=gold_entries,
        gold_days=gold_days,
        formal_switches=formal_switches,
    )


def robust_leave_year_metrics(
    returns: pd.DataFrame,
    baseline: pd.Series,
    years: list[int],
) -> pd.DataFrame:
    """Measure each fixed candidate after deleting every calendar year in turn."""
    from research.momentum_defender_gold_override_overfit import full_metrics

    deltas: dict[str, list[np.ndarray]] = {
        "annualized_return_252": [],
        "sharpe": [],
        "max_drawdown": [],
    }
    for year in years:
        mask = returns.index.year != year
        measured = full_metrics(returns.loc[mask], baseline.loc[mask])
        for field in deltas:
            deltas[field].append(measured[f"delta_{field}"].to_numpy(float))
    result = pd.DataFrame(index=returns.columns)
    for field, samples in deltas.items():
        matrix = np.vstack(samples)
        result[f"leave_year_{field}_median"] = np.median(matrix, axis=0)
        result[f"leave_year_{field}_q25"] = np.quantile(matrix, 0.25, axis=0)
        result[f"leave_year_{field}_worst"] = matrix.min(axis=0)
    return result
