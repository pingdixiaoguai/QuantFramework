"""Universal 510300 gate with a minimal Gold-specific exception."""

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
from research.momentum_defender_selected_asset_draqm import AssetDRAQMPolicy


GOLD_EXEMPTION_ONLY = "gold_exemption_only"
GOLD_BIDIRECTIONAL = "gold_bidirectional"
OVERRIDE_MODES = {GOLD_EXEMPTION_ONLY, GOLD_BIDIRECTIONAL}


@dataclass(frozen=True)
class GoldExceptionSpec:
    anchor_policy: AssetDRAQMPolicy
    gold_policy: AssetDRAQMPolicy
    momentum_lock_days: int
    defender_lock_days: int
    override_mode: str

    def __post_init__(self) -> None:
        if self.anchor_policy.asset != "510300.SH":
            raise ValueError("anchor policy must use 510300.SH")
        if self.gold_policy.asset != "518880.SH":
            raise ValueError("gold policy must use 518880.SH")
        for label, value in (
            ("Momentum", self.momentum_lock_days),
            ("Defender", self.defender_lock_days),
        ):
            if value < 0 or value % 5:
                raise ValueError(f"{label} lock must be a non-negative multiple of 5")
        if self.override_mode not in OVERRIDE_MODES:
            raise ValueError(f"unsupported Gold override mode: {self.override_mode}")

    def candidate_id(self) -> str:
        mode = "exempt" if self.override_mode == GOLD_EXEMPTION_ONLY else "bidir"
        return (
            f"anchor={self.anchor_policy.policy_id()}__gold={self.gold_policy.policy_id()}__"
            f"mh{self.momentum_lock_days}_dh{self.defender_lock_days}_{mode}"
        )


@dataclass(frozen=True)
class GoldExceptionRun:
    spec: GoldExceptionSpec
    state: pd.DataFrame
    returns: np.ndarray
    requested_target: np.ndarray
    actual_target: np.ndarray
    defender_entries: int
    defender_days: int
    sleeve_switches: int
    candidate_switches: int


def _score(
    features: Mapping[str, DownsideRAQMFeatures],
    policy: AssetDRAQMPolicy,
    timestamp: pd.Timestamp,
) -> float:
    series = features[policy.asset].composite_at_open[
        policy.profile.profile_id, "rolling_504_strict_lag"
    ]
    value = series.loc[timestamp]
    return float(value) if pd.notna(value) else np.nan


def gold_exception_state_schedule(
    calendar: pd.DatetimeIndex,
    momentum_target: pd.Series,
    features: Mapping[str, DownsideRAQMFeatures],
    spec: GoldExceptionSpec,
) -> pd.DataFrame:
    """Use 510300 for non-Gold Top-1 and Gold evidence only for Gold Top-1."""
    targets = momentum_target.reindex(calendar)
    if targets.isna().any():
        raise ValueError("Momentum Top-1 is missing on the execution calendar")
    state = True
    held_days = 10**9
    entry_streak = 0
    recovery_streak = 0
    entry_source: str | None = None
    recovery_source: str | None = None
    rows = []
    for timestamp in calendar:
        top1 = str(targets.loc[timestamp])
        gold_top1 = top1 == "518880.SH"
        anchor_score = _score(features, spec.anchor_policy, timestamp)
        gold_score = _score(features, spec.gold_policy, timestamp)
        anchor_entry = bool(
            np.isfinite(anchor_score)
            and anchor_score >= spec.anchor_policy.entry_percentile
        )
        gold_entry = bool(
            np.isfinite(gold_score)
            and gold_score >= spec.gold_policy.entry_percentile
        )
        anchor_recovery = bool(
            np.isfinite(anchor_score)
            and anchor_score <= spec.anchor_policy.recovery_percentile
        )
        gold_recovery = bool(
            np.isfinite(gold_score)
            and gold_score <= spec.gold_policy.recovery_percentile
        )
        previous = state
        reason = "hold"
        evidence_source = "gold" if gold_top1 else "anchor"
        entry_qualified = False
        recovery_qualified = False
        if state:
            recovery_streak = 0
            recovery_source = None
            if gold_top1:
                if spec.override_mode == GOLD_EXEMPTION_ONLY:
                    entry_qualified = anchor_entry and gold_entry
                    source = "anchor_and_gold"
                    required = max(
                        spec.anchor_policy.entry_confirmation_days,
                        spec.gold_policy.entry_confirmation_days,
                    )
                else:
                    entry_qualified = gold_entry
                    source = "gold"
                    required = spec.gold_policy.entry_confirmation_days
            else:
                entry_qualified = anchor_entry
                source = "anchor"
                required = spec.anchor_policy.entry_confirmation_days
            if entry_source != source:
                entry_source = source
                entry_streak = 0
            entry_streak = entry_streak + 1 if entry_qualified else 0
            if entry_streak >= required:
                if held_days >= spec.momentum_lock_days:
                    state = False
                    held_days = 0
                    entry_streak = 0
                    recovery_streak = 0
                    reason = "gold_exception_to_defender"
                else:
                    reason = "defender_entry_blocked_by_momentum_lock"
            elif gold_top1 and anchor_entry and not gold_entry:
                reason = "gold_exception_blocks_anchor_defender"
        else:
            entry_streak = 0
            entry_source = None
            if gold_top1:
                recovery_qualified = gold_recovery
                source = "gold"
                required = spec.gold_policy.recovery_confirmation_days
            else:
                recovery_qualified = anchor_recovery
                source = "anchor"
                required = spec.anchor_policy.recovery_confirmation_days
            if recovery_source != source:
                recovery_source = source
                recovery_streak = 0
            recovery_streak = recovery_streak + 1 if recovery_qualified else 0
            if recovery_streak >= required:
                if held_days >= spec.defender_lock_days:
                    state = True
                    held_days = 0
                    entry_streak = 0
                    recovery_streak = 0
                    reason = "gold_exception_to_momentum"
                else:
                    reason = "momentum_recovery_blocked_by_defender_lock"
        rows.append(
            {
                "date": timestamp,
                "risk_on": state,
                "state_changed": state != previous,
                "state_reason": reason,
                "momentum_top1_at_open": top1,
                "evidence_source": evidence_source,
                "anchor_score_at_open": anchor_score,
                "gold_score_at_open": gold_score,
                "anchor_entry_qualified": anchor_entry,
                "gold_entry_qualified": gold_entry,
                "entry_qualified": entry_qualified,
                "recovery_qualified": recovery_qualified,
                "entry_confirmation_streak": entry_streak,
                "recovery_confirmation_streak": recovery_streak,
                "held_days_at_open": held_days,
            }
        )
        held_days += 1
    return pd.DataFrame(rows).set_index("date")


def run_gold_exception_spec(
    data: ExactExecutionData,
    momentum_target: pd.Series,
    features: Mapping[str, DownsideRAQMFeatures],
    spec: GoldExceptionSpec,
) -> GoldExceptionRun:
    state = gold_exception_state_schedule(
        data.calendar, momentum_target, features, spec
    )
    defender = data.candidate_index[DEFENDER_CANDIDATE]
    requested = np.where(
        state["risk_on"].to_numpy(bool), data.momentum_target, defender
    ).astype(int)
    returns, actual, switches = exact_candidate_schedule(data, requested)
    entries = state["state_changed"].astype(bool) & ~state["risk_on"].astype(bool)
    return GoldExceptionRun(
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
