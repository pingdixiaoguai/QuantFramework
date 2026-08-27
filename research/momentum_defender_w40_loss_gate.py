"""Causal single-window 510300 downside-log-loss percentile gate."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from research.defender_curve_momentum import DEFENDER_CANDIDATE
from research.momentum_defender_downside_raqm import (
    DownsideRAQMSpec,
    ExactExecutionData,
    FactorProfile,
    downside_raqm_state_schedule,
    exact_candidate_schedule,
    strict_lag_percentile,
)
from research.momentum_volatility import asof_previous_close


ANCHOR_ASSET = "510300.SH"
WINDOW = 40
HISTORY_WINDOW = 504
MIN_HISTORY = 252


@dataclass(frozen=True)
class W40LossGateSpec:
    entry_percentile: float
    recovery_percentile: float
    entry_confirmation_days: int
    recovery_confirmation_days: int
    momentum_lock_days: int
    defender_lock_days: int

    def __post_init__(self) -> None:
        if not 0.0 <= self.recovery_percentile < self.entry_percentile <= 1.0:
            raise ValueError("recovery percentile must be below entry percentile")
        if min(self.entry_confirmation_days, self.recovery_confirmation_days) < 1:
            raise ValueError("confirmation counts must be positive")
        for label, value in (
            ("Momentum", self.momentum_lock_days),
            ("Defender", self.defender_lock_days),
        ):
            if not 20 <= value <= 30 or value % 5:
                raise ValueError(
                    f"{label} lock must be a five-day multiple in [20, 30]"
                )

    def candidate_id(self) -> str:
        return (
            f"w40_loss_r504_en{self.entry_percentile:.2f}_"
            f"ex{self.recovery_percentile:.2f}_"
            f"ec{self.entry_confirmation_days}_rc{self.recovery_confirmation_days}_"
            f"mh{self.momentum_lock_days}_dh{self.defender_lock_days}"
        )

    def state_spec(self) -> DownsideRAQMSpec:
        return DownsideRAQMSpec(
            profile=FactorProfile("w40_loss", (40,), (1.0,)),
            history_mode="rolling_504_strict_lag",
            entry_percentile=self.entry_percentile,
            exit_percentile=self.recovery_percentile,
            momentum_lock_days=self.momentum_lock_days,
            defender_lock_days=self.defender_lock_days,
            entry_confirmation_days=self.entry_confirmation_days,
            recovery_confirmation_days=self.recovery_confirmation_days,
        )


@dataclass(frozen=True)
class W40LossGateRun:
    spec: W40LossGateSpec
    state: pd.DataFrame
    returns: np.ndarray
    requested_target: np.ndarray
    actual_target: np.ndarray
    defender_entries: int
    defender_days: int
    sleeve_switches: int
    candidate_switches: int


def downside_log_loss(close: pd.Series, window: int = WINDOW) -> pd.Series:
    """Return positive log loss for downtrends and zero otherwise."""
    if window < 2:
        raise ValueError("loss window must be at least two")
    values = close.astype(float)
    if (values <= 0.0).any():
        raise ValueError("log loss requires positive closes")
    result = (-np.log(values).diff(window)).clip(lower=0.0).astype(float)
    result.name = f"downside_log_loss_{window}"
    return result


def w40_loss_percentile_at_open(
    close: pd.Series,
    calendar: pd.DatetimeIndex,
    *,
    history_window: int = HISTORY_WINDOW,
    min_history: int = MIN_HISTORY,
) -> tuple[pd.Series, pd.Series]:
    """Build raw loss and strict-lag rolling percentile at executable opens."""
    raw_close = downside_log_loss(close, WINDOW)
    percentile_close = strict_lag_percentile(
        raw_close,
        history_window=history_window,
        min_history=min_history,
    )
    raw_open = asof_previous_close(raw_close, calendar)
    percentile_open = asof_previous_close(percentile_close, calendar)
    raw_open.name = "w40_downside_log_loss_at_open"
    percentile_open.name = "w40_loss_percentile_at_open"
    return raw_open, percentile_open


def run_w40_loss_gate(
    data: ExactExecutionData,
    score_at_open: pd.Series,
    spec: W40LossGateSpec,
) -> W40LossGateRun:
    state = downside_raqm_state_schedule(score_at_open, spec.state_spec())
    defender = data.candidate_index[DEFENDER_CANDIDATE]
    requested = np.where(
        state["risk_on"].to_numpy(bool), data.momentum_target, defender
    ).astype(int)
    returns, actual, switches = exact_candidate_schedule(data, requested)
    entries = state["state_changed"].astype(bool) & ~state["risk_on"].astype(bool)
    return W40LossGateRun(
        spec=spec,
        state=state,
        returns=returns,
        requested_target=requested,
        actual_target=actual,
        defender_entries=int(entries.sum()),
        defender_days=int((~state["risk_on"].astype(bool)).sum()),
        sleeve_switches=int(state["state_changed"].sum()),
        candidate_switches=switches,
    )
