"""Asset-selected downside-RAQM switching for the frozen Momentum Top-1.

Only configured Momentum assets may trigger Defender.  Other Momentum targets
remain completely ungated while the strategy is in the Momentum sleeve.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from research.defender_curve_momentum import DEFENDER_CANDIDATE
from research.momentum_defender_downside_raqm import (
    DownsideRAQMFeatures,
    ExactExecutionData,
    FactorProfile,
    exact_candidate_schedule,
)


STICKY_ENTRY_ASSET = "sticky_entry_asset"
SHADOW_TOP1_RECOVER_OTHER = "shadow_top1_recover_other"
RECOVERY_MODES = {STICKY_ENTRY_ASSET, SHADOW_TOP1_RECOVER_OTHER}


@dataclass(frozen=True)
class AssetDRAQMPolicy:
    asset: str
    profile: FactorProfile
    entry_percentile: float
    recovery_percentile: float
    entry_confirmation_days: int
    recovery_confirmation_days: int

    def __post_init__(self) -> None:
        if self.asset not in {"510300.SH", "518880.SH"}:
            raise ValueError("only 510300.SH and 518880.SH may have DRAQM policies")
        if not 0.0 <= self.recovery_percentile < self.entry_percentile <= 1.0:
            raise ValueError("recovery percentile must be below entry percentile")
        if min(self.entry_confirmation_days, self.recovery_confirmation_days) < 1:
            raise ValueError("confirmation counts must be positive")

    def policy_id(self) -> str:
        code = self.asset.split(".")[0]
        return (
            f"{code}_{self.profile.profile_id}_"
            f"en{self.entry_percentile:.2f}_re{self.recovery_percentile:.2f}_"
            f"ec{self.entry_confirmation_days}_rc{self.recovery_confirmation_days}"
        )


@dataclass(frozen=True)
class SelectedAssetDRAQMSpec:
    policies: Mapping[str, AssetDRAQMPolicy | None]
    momentum_lock_days: int
    defender_lock_days: int
    recovery_mode: str

    def __post_init__(self) -> None:
        if set(self.policies) != {"510300.SH", "518880.SH"}:
            raise ValueError("policies must contain exactly 510300.SH and 518880.SH")
        for asset, policy in self.policies.items():
            if policy is not None and policy.asset != asset:
                raise ValueError("policy asset does not match mapping key")
        if self.momentum_lock_days < 0 or self.momentum_lock_days % 5:
            raise ValueError("Momentum lock must be a non-negative multiple of 5")
        if self.defender_lock_days < 0 or self.defender_lock_days % 5:
            raise ValueError("Defender lock must be a non-negative multiple of 5")
        if self.recovery_mode not in RECOVERY_MODES:
            raise ValueError(f"unsupported recovery mode: {self.recovery_mode}")

    def candidate_id(self) -> str:
        policy = "__".join(
            self.policies[asset].policy_id() if self.policies[asset] else f"{asset}=off"
            for asset in ("510300.SH", "518880.SH")
        )
        mode = "sticky" if self.recovery_mode == STICKY_ENTRY_ASSET else "shadow"
        return (
            f"{policy}__mh{self.momentum_lock_days}_dh{self.defender_lock_days}_{mode}"
        )


@dataclass(frozen=True)
class SelectedAssetDRAQMRun:
    spec: SelectedAssetDRAQMSpec
    state: pd.DataFrame
    returns: np.ndarray
    requested_target: np.ndarray
    actual_target: np.ndarray
    defender_entries: int
    defender_days: int
    sleeve_switches: int
    candidate_switches: int


def _score_at_open(
    features: Mapping[str, DownsideRAQMFeatures],
    policy: AssetDRAQMPolicy,
    timestamp: pd.Timestamp,
) -> float:
    series = features[policy.asset].composite_at_open[
        policy.profile.profile_id, "rolling_504_strict_lag"
    ]
    value = series.loc[timestamp]
    return float(value) if pd.notna(value) else np.nan


def selected_asset_state_schedule(
    calendar: pd.DatetimeIndex,
    momentum_target: pd.Series,
    features: Mapping[str, DownsideRAQMFeatures],
    spec: SelectedAssetDRAQMSpec,
) -> pd.DataFrame:
    """Apply asset-specific entry/recovery evidence with top-level sleeve locks."""
    target = momentum_target.reindex(calendar)
    if target.isna().any():
        raise ValueError("Momentum target is missing on the execution calendar")
    state = True
    held_days = 10**9
    trigger_asset: str | None = None
    entry_reference: str | None = None
    recovery_reference: str | None = None
    entry_streak = 0
    recovery_streak = 0
    rows: list[dict[str, object]] = []
    for timestamp in calendar:
        top1 = str(target.loc[timestamp])
        previous_state = state
        policy = spec.policies.get(top1)
        score = np.nan
        evidence_asset: str | None = None
        entry_qualified = False
        recovery_qualified = False
        reason = "hold"

        if state:
            recovery_streak = 0
            recovery_reference = None
            if policy is None:
                entry_streak = 0
                entry_reference = None
                reason = "other_momentum_asset_not_gated"
            else:
                evidence_asset = top1
                score = _score_at_open(features, policy, timestamp)
                if entry_reference != top1:
                    entry_reference = top1
                    entry_streak = 0
                entry_qualified = bool(
                    np.isfinite(score) and score >= policy.entry_percentile
                )
                entry_streak = entry_streak + 1 if entry_qualified else 0
                if entry_streak >= policy.entry_confirmation_days:
                    if held_days >= spec.momentum_lock_days:
                        state = False
                        trigger_asset = top1
                        held_days = 0
                        entry_streak = 0
                        recovery_streak = 0
                        recovery_reference = None
                        reason = "selected_asset_draqm_to_defender"
                    else:
                        reason = "defender_entry_blocked_by_momentum_lock"
        else:
            entry_streak = 0
            entry_reference = None
            if trigger_asset is None:
                raise AssertionError("Defender state must retain its trigger asset")
            if spec.recovery_mode == STICKY_ENTRY_ASSET:
                reference = trigger_asset
                recovery_policy = spec.policies[reference]
                assert recovery_policy is not None
                evidence_asset = reference
                score = _score_at_open(features, recovery_policy, timestamp)
                recovery_qualified = bool(
                    np.isfinite(score)
                    and score <= recovery_policy.recovery_percentile
                )
            elif policy is None:
                reference = "OTHER"
                recovery_policy = None
                recovery_qualified = True
            else:
                reference = top1
                recovery_policy = policy
                evidence_asset = reference
                score = _score_at_open(features, recovery_policy, timestamp)
                recovery_qualified = bool(
                    np.isfinite(score)
                    and score <= recovery_policy.recovery_percentile
                )
            if recovery_reference != reference:
                recovery_reference = reference
                recovery_streak = 0
            recovery_streak = recovery_streak + 1 if recovery_qualified else 0
            required = (
                1
                if recovery_policy is None
                else recovery_policy.recovery_confirmation_days
            )
            if recovery_streak >= required:
                if held_days >= spec.defender_lock_days:
                    state = True
                    held_days = 0
                    trigger_asset = None
                    recovery_streak = 0
                    recovery_reference = None
                    reason = (
                        "shadow_other_asset_to_momentum"
                        if recovery_policy is None
                        else "selected_asset_draqm_to_momentum"
                    )
                else:
                    reason = "momentum_recovery_blocked_by_defender_lock"

        rows.append(
            {
                "date": timestamp,
                "risk_on": state,
                "state_changed": state != previous_state,
                "state_reason": reason,
                "momentum_top1_at_open": top1,
                "trigger_asset": trigger_asset,
                "evidence_asset": evidence_asset,
                "draqm_percentile_at_open": score,
                "entry_qualified": entry_qualified,
                "recovery_qualified": recovery_qualified,
                "entry_confirmation_streak": entry_streak,
                "recovery_confirmation_streak": recovery_streak,
                "held_days_at_open": held_days,
            }
        )
        held_days += 1
    return pd.DataFrame(rows).set_index("date")


def run_selected_asset_draqm_spec(
    data: ExactExecutionData,
    momentum_target: pd.Series,
    features: Mapping[str, DownsideRAQMFeatures],
    spec: SelectedAssetDRAQMSpec,
) -> SelectedAssetDRAQMRun:
    state = selected_asset_state_schedule(
        data.calendar, momentum_target, features, spec
    )
    defender = data.candidate_index[DEFENDER_CANDIDATE]
    requested = np.where(
        state["risk_on"].to_numpy(bool), data.momentum_target, defender
    ).astype(int)
    returns, actual, candidate_switches = exact_candidate_schedule(data, requested)
    entries = state["state_changed"].astype(bool) & ~state["risk_on"].astype(bool)
    return SelectedAssetDRAQMRun(
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
