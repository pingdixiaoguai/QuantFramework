"""Independent universal anchor state plus a non-mutating Gold overlay."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from research.defender_curve_momentum import DEFENDER_CANDIDATE
from research.momentum_defender_downside_raqm import (
    DownsideRAQMFeatures,
    ExactExecutionData,
    exact_candidate_schedule,
)
from research.momentum_defender_gold_exception_gate import (
    GOLD_BIDIRECTIONAL,
    GOLD_EXEMPTION_ONLY,
    OVERRIDE_MODES,
)
from research.momentum_defender_selected_asset_draqm import AssetDRAQMPolicy


@dataclass(frozen=True)
class GoldOverlaySpec:
    anchor_policy: AssetDRAQMPolicy
    gold_policy: AssetDRAQMPolicy
    momentum_lock_days: int
    defender_lock_days: int
    override_mode: str

    def __post_init__(self) -> None:
        if self.anchor_policy.asset != "510300.SH":
            raise ValueError("anchor policy must use 510300.SH")
        if self.gold_policy.asset != "518880.SH":
            raise ValueError("Gold policy must use 518880.SH")
        for label, value in (
            ("Momentum", self.momentum_lock_days),
            ("Defender", self.defender_lock_days),
        ):
            if value < 0 or value % 5:
                raise ValueError(f"{label} lock must be a non-negative multiple of 5")
        if self.override_mode not in OVERRIDE_MODES:
            raise ValueError("unsupported Gold overlay mode")

    def candidate_id(self) -> str:
        mode = "exempt" if self.override_mode == GOLD_EXEMPTION_ONLY else "bidir"
        return (
            f"base={self.anchor_policy.policy_id()}_mh{self.momentum_lock_days}_"
            f"dh{self.defender_lock_days}__gold={self.gold_policy.policy_id()}_{mode}"
        )


@dataclass(frozen=True)
class GoldOverlayRun:
    spec: GoldOverlaySpec
    state: pd.DataFrame
    returns: np.ndarray
    requested_target: np.ndarray
    actual_target: np.ndarray
    base_defender_entries: int
    effective_defender_days: int
    candidate_switches: int


def _score(features, policy, timestamp):
    value = features[policy.asset].composite_at_open[
        policy.profile.profile_id, "rolling_504_strict_lag"
    ].loc[timestamp]
    return float(value) if pd.notna(value) else np.nan


def independent_anchor_state(
    calendar: pd.DatetimeIndex,
    features: Mapping[str, DownsideRAQMFeatures],
    spec: GoldOverlaySpec,
) -> pd.DataFrame:
    """Run the 510300 base state without observing Gold or Momentum Top-1."""
    state = True
    held_days = 10**9
    entry_streak = 0
    recovery_streak = 0
    rows = []
    policy = spec.anchor_policy
    for timestamp in calendar:
        score = _score(features, policy, timestamp)
        entry = bool(np.isfinite(score) and score >= policy.entry_percentile)
        recovery = bool(
            np.isfinite(score) and score <= policy.recovery_percentile
        )
        previous = state
        reason = "base_hold"
        if state:
            recovery_streak = 0
            entry_streak = entry_streak + 1 if entry else 0
            if entry_streak >= policy.entry_confirmation_days:
                if held_days >= spec.momentum_lock_days:
                    state = False
                    held_days = 0
                    entry_streak = 0
                    reason = "base_to_defender"
                else:
                    reason = "base_entry_blocked_by_lock"
        else:
            entry_streak = 0
            recovery_streak = recovery_streak + 1 if recovery else 0
            if recovery_streak >= policy.recovery_confirmation_days:
                if held_days >= spec.defender_lock_days:
                    state = True
                    held_days = 0
                    recovery_streak = 0
                    reason = "base_to_momentum"
                else:
                    reason = "base_recovery_blocked_by_lock"
        rows.append(
            {
                "date": timestamp,
                "base_risk_on": state,
                "base_state_changed": state != previous,
                "base_state_reason": reason,
                "anchor_score_at_open": score,
                "base_held_days_at_open": held_days,
            }
        )
        held_days += 1
    return pd.DataFrame(rows).set_index("date")


def independent_gold_state(
    calendar: pd.DatetimeIndex,
    momentum_target: pd.Series,
    features: Mapping[str, DownsideRAQMFeatures],
    policy: AssetDRAQMPolicy,
) -> pd.DataFrame:
    """Run the reset-on-exit Gold state independently of the base gate."""
    targets = momentum_target.reindex(calendar)
    if targets.isna().any():
        raise ValueError("Momentum Top-1 is missing")
    gold_risk_on = True
    gold_entry_streak = 0
    gold_recovery_streak = 0
    previous_gold_top1 = False
    rows = []
    for timestamp in calendar:
        gold_top1 = str(targets.loc[timestamp]) == "518880.SH"
        score = _score(features, policy, timestamp)
        entry = bool(np.isfinite(score) and score >= policy.entry_percentile)
        recovery = bool(
            np.isfinite(score) and score <= policy.recovery_percentile
        )
        if not gold_top1:
            gold_risk_on = True
            gold_entry_streak = 0
            gold_recovery_streak = 0
        else:
            if not previous_gold_top1:
                gold_risk_on = True
                gold_entry_streak = 0
                gold_recovery_streak = 0
            if gold_risk_on:
                gold_recovery_streak = 0
                gold_entry_streak = gold_entry_streak + 1 if entry else 0
                if gold_entry_streak >= policy.entry_confirmation_days:
                    gold_risk_on = False
                    gold_entry_streak = 0
            else:
                gold_entry_streak = 0
                gold_recovery_streak = gold_recovery_streak + 1 if recovery else 0
                if gold_recovery_streak >= policy.recovery_confirmation_days:
                    gold_risk_on = True
                    gold_recovery_streak = 0
        rows.append(
            {
                "date": timestamp,
                "momentum_top1_at_open": str(targets.loc[timestamp]),
                "gold_top1": gold_top1,
                "gold_score_at_open": score,
                "gold_risk_on": gold_risk_on,
                "gold_entry_streak": gold_entry_streak,
                "gold_recovery_streak": gold_recovery_streak,
            }
        )
        previous_gold_top1 = gold_top1
    return pd.DataFrame(rows).set_index("date")


def gold_overlay_state_schedule(
    calendar: pd.DatetimeIndex,
    momentum_target: pd.Series,
    features: Mapping[str, DownsideRAQMFeatures],
    spec: GoldOverlaySpec,
    base_state: pd.DataFrame | None = None,
    gold_state: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Combine independent base and Gold states without mutating either."""
    base = (
        independent_anchor_state(calendar, features, spec)
        if base_state is None
        else base_state.reindex(calendar)
    )
    gold = (
        independent_gold_state(calendar, momentum_target, features, spec.gold_policy)
        if gold_state is None
        else gold_state.reindex(calendar)
    )
    if base[["base_risk_on", "base_state_changed"]].isna().any().any():
        raise ValueError("precomputed base state is missing execution rows")
    if gold[["gold_top1", "gold_risk_on"]].isna().any().any():
        raise ValueError("precomputed Gold state is missing execution rows")
    result = base.join(gold)
    base_risk_on = result["base_risk_on"].astype(bool)
    gold_top1 = result["gold_top1"].astype(bool)
    gold_risk_on = result["gold_risk_on"].astype(bool)
    if spec.override_mode == GOLD_EXEMPTION_ONLY:
        gold_effective = base_risk_on | gold_risk_on
    else:
        gold_effective = gold_risk_on
    result["effective_risk_on"] = base_risk_on.where(~gold_top1, gold_effective)
    result["gold_overrides_base"] = gold_top1 & result[
        "effective_risk_on"
    ].ne(base_risk_on)
    return result


def run_gold_overlay_spec(
    data: ExactExecutionData,
    momentum_target: pd.Series,
    features: Mapping[str, DownsideRAQMFeatures],
    spec: GoldOverlaySpec,
    base_state: pd.DataFrame | None = None,
    gold_state: pd.DataFrame | None = None,
) -> GoldOverlayRun:
    state = gold_overlay_state_schedule(
        data.calendar,
        momentum_target,
        features,
        spec,
        base_state=base_state,
        gold_state=gold_state,
    )
    defender = data.candidate_index[DEFENDER_CANDIDATE]
    requested = np.where(
        state["effective_risk_on"].to_numpy(bool), data.momentum_target, defender
    ).astype(int)
    returns, actual, switches = exact_candidate_schedule(data, requested)
    entries = state["base_state_changed"].astype(bool) & ~state[
        "base_risk_on"
    ].astype(bool)
    return GoldOverlayRun(
        spec=spec,
        state=state,
        returns=returns,
        requested_target=requested,
        actual_target=actual,
        base_defender_entries=int(entries.sum()),
        effective_defender_days=int((~state["effective_risk_on"].astype(bool)).sum()),
        candidate_switches=switches,
    )
