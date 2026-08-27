"""Relative Gold-versus-Defender overlay on a frozen universal gate.

The overlay never mutates the universal 510300 state.  It is evaluated only
while Gold is Momentum Top-1 and uses previous-close signed RAQM for both the
Gold continuous curve and the whole-Defender continuous NAV.  Gold may bypass
the universal Defender state only when it is itself non-negative and has a
configured RAQM advantage over Defender.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from research.defender_curve_momentum import DEFENDER_CANDIDATE
from research.gold_raqm_regularization import RAQMSpec, raqm_score
from research.momentum_defender_downside_raqm import (
    ExactExecutionData,
    FactorProfile,
    exact_candidate_schedule,
)
from research.momentum_defender_gold_exception_gate import (
    GOLD_BIDIRECTIONAL,
    GOLD_EXEMPTION_ONLY,
    OVERRIDE_MODES,
)
from research.momentum_defender_gold_override import GOLD_ASSET


@dataclass(frozen=True)
class RelativeGoldOverlaySpec:
    profile: FactorProfile
    entry_difference: float
    exit_difference: float
    entry_confirmation_days: int
    exit_confirmation_days: int
    minimum_gold_hold_days: int
    override_mode: str

    def __post_init__(self) -> None:
        if self.exit_difference >= self.entry_difference:
            raise ValueError("exit difference must be below entry difference")
        if min(self.entry_confirmation_days, self.exit_confirmation_days) < 1:
            raise ValueError("confirmation counts must be positive")
        if not 0 <= self.minimum_gold_hold_days <= 30:
            raise ValueError("Gold hold must lie in [0, 30]")
        if self.minimum_gold_hold_days % 5:
            raise ValueError("Gold hold must be a multiple of five")
        if self.override_mode not in OVERRIDE_MODES:
            raise ValueError("unsupported Gold overlay mode")

    def candidate_id(self) -> str:
        mode = "exempt" if self.override_mode == GOLD_EXEMPTION_ONLY else "bidir"
        return (
            f"relative_gold_{self.profile.profile_id}_"
            f"en{self.entry_difference:+.2f}_ex{self.exit_difference:+.2f}_"
            f"ec{self.entry_confirmation_days}_xc{self.exit_confirmation_days}_"
            f"h{self.minimum_gold_hold_days}_{mode}"
        )


@dataclass(frozen=True)
class RelativeGoldOverlayRun:
    spec: RelativeGoldOverlaySpec
    state: pd.DataFrame
    returns: np.ndarray
    requested_target: np.ndarray
    actual_target: np.ndarray
    gold_entries: int
    gold_allowed_days: int
    override_days: int
    candidate_switches: int


@dataclass(frozen=True)
class FastRelativeGoldState:
    effective_risk_on: np.ndarray
    gold_overlay_active: np.ndarray
    gold_overlay_changed: np.ndarray
    gold_overrides_base: np.ndarray


def signed_raqm_profiles_at_open(
    curves: pd.DataFrame,
    profiles: Mapping[str, FactorProfile],
    *,
    volatility_floor_annual: float = 0.08,
    winsor_limit: float = 3.0,
) -> dict[str, pd.DataFrame]:
    """Build same-profile signed RAQM for Gold and whole Defender at open."""
    missing = {GOLD_ASSET, DEFENDER_CANDIDATE} - set(curves)
    if missing:
        raise ValueError(f"missing continuous curves: {sorted(missing)}")
    horizons = sorted(
        {horizon for profile in profiles.values() for horizon in profile.horizons}
    )
    raw: dict[tuple[str, int], pd.Series] = {}
    for candidate in (GOLD_ASSET, DEFENDER_CANDIDATE):
        for horizon in horizons:
            raw[candidate, horizon] = raqm_score(
                curves[candidate],
                RAQMSpec(
                    family="relative_gold_signed_raqm",
                    window=horizon,
                    volatility_floor_annual=volatility_floor_annual,
                    winsor_limit=winsor_limit,
                    extra_numeric_parameters=2,
                ),
            ).shift(1)
    result: dict[str, pd.DataFrame] = {}
    for profile_id, profile in profiles.items():
        frame = pd.DataFrame(index=curves.index)
        for candidate in (GOLD_ASSET, DEFENDER_CANDIDATE):
            panel = pd.concat(
                [raw[candidate, horizon] for horizon in profile.horizons],
                axis=1,
            )
            frame[candidate] = panel.mul(
                np.asarray(profile.weights), axis=1
            ).sum(axis=1, min_count=len(profile.horizons))
        frame["difference"] = frame[GOLD_ASSET] - frame[DEFENDER_CANDIDATE]
        result[profile_id] = frame
    return result


def relative_gold_overlay_state(
    calendar: pd.DatetimeIndex,
    momentum_target: pd.Series,
    base_risk_on: pd.Series,
    metrics_at_open: pd.DataFrame,
    spec: RelativeGoldOverlaySpec,
) -> pd.DataFrame:
    """Apply a reset-on-Gold-exit relative overlay without changing base state.

    The hold protects an active exemption from a noisy relative-score exit.
    Confirmed negative Gold RAQM is a hard safety exit and bypasses that hold.
    """
    target = momentum_target.reindex(calendar)
    base = base_risk_on.reindex(calendar)
    metrics = metrics_at_open.reindex(calendar)
    if target.isna().any() or base.isna().any():
        raise ValueError("Momentum target or base state is missing")
    required = [GOLD_ASSET, DEFENDER_CANDIDATE, "difference"]
    if not set(required).issubset(metrics):
        raise ValueError("relative Gold metrics are incomplete")

    active = False
    active_days = 0
    entry_streak = 0
    exit_streak = 0
    previous_gold_top1 = False
    rows: list[dict[str, object]] = []
    for timestamp in calendar:
        gold_top1 = str(target.loc[timestamp]) == GOLD_ASSET
        gold_value = float(metrics.at[timestamp, GOLD_ASSET])
        defender_value = float(metrics.at[timestamp, DEFENDER_CANDIDATE])
        difference = float(metrics.at[timestamp, "difference"])
        finite = bool(
            np.isfinite(gold_value)
            and np.isfinite(defender_value)
            and np.isfinite(difference)
        )
        gold_healthy = bool(finite and gold_value > 0.0)
        entry_qualified = bool(
            gold_top1
            and gold_healthy
            and difference >= spec.entry_difference
        )
        relative_exit = bool(
            gold_top1 and finite and difference <= spec.exit_difference
        )
        gold_weak = bool(gold_top1 and finite and gold_value <= 0.0)
        changed = False
        reason = "overlay_hold"

        if not gold_top1:
            if active or previous_gold_top1:
                reason = "reset_outside_gold_top1"
            active = False
            active_days = 0
            entry_streak = 0
            exit_streak = 0
        elif not previous_gold_top1:
            active = False
            active_days = 0
            entry_streak = 0
            exit_streak = 0

        if gold_top1 and not active:
            exit_streak = 0
            entry_streak = entry_streak + 1 if entry_qualified else 0
            if entry_streak >= spec.entry_confirmation_days:
                active = True
                changed = True
                active_days = 0
                entry_streak = 0
                reason = "relative_advantage_gold_allowed"
            elif not finite:
                reason = "insufficient_relative_factor_history"
            elif not gold_healthy:
                reason = "gold_not_healthy"
            elif not entry_qualified:
                reason = "relative_entry_not_met"
        elif gold_top1 and active:
            entry_streak = 0
            exit_qualified = gold_weak or relative_exit
            exit_streak = exit_streak + 1 if exit_qualified else 0
            if exit_streak >= spec.exit_confirmation_days:
                if gold_weak:
                    active = False
                    changed = True
                    active_days = 0
                    exit_streak = 0
                    reason = "weak_gold_hard_exit"
                elif active_days >= spec.minimum_gold_hold_days:
                    active = False
                    changed = True
                    active_days = 0
                    exit_streak = 0
                    reason = "relative_advantage_exit"
                else:
                    reason = "relative_exit_blocked_by_gold_hold"

        base_value = bool(base.loc[timestamp])
        if gold_top1:
            effective = (
                base_value or active
                if spec.override_mode == GOLD_EXEMPTION_ONLY
                else active
            )
        else:
            effective = base_value
        rows.append(
            {
                "date": timestamp,
                "momentum_top1_at_open": str(target.loc[timestamp]),
                "gold_top1": gold_top1,
                "base_risk_on": base_value,
                "gold_raqm_at_open": gold_value,
                "defender_raqm_at_open": defender_value,
                "relative_raqm_at_open": difference,
                "gold_healthy": gold_healthy,
                "relative_entry_qualified": entry_qualified,
                "relative_exit_qualified": relative_exit,
                "gold_overlay_active": active,
                "gold_overlay_changed": changed,
                "gold_overlay_reason": reason,
                "entry_confirmation_streak": entry_streak,
                "exit_confirmation_streak": exit_streak,
                "gold_active_days_at_open": active_days,
                "effective_risk_on": effective,
                "gold_overrides_base": gold_top1 and effective != base_value,
            }
        )
        if active:
            active_days += 1
        previous_gold_top1 = gold_top1
    return pd.DataFrame(rows).set_index("date")


def fast_relative_gold_state(
    gold_top1: np.ndarray,
    base_risk_on: np.ndarray,
    gold_raqm: np.ndarray,
    defender_raqm: np.ndarray,
    difference: np.ndarray,
    spec: RelativeGoldOverlaySpec,
) -> FastRelativeGoldState:
    """Array-equivalent overlay state for broad searches.

    Final candidates should still use :func:`relative_gold_overlay_state` so
    that their reason ledger remains human-auditable.
    """
    lengths = {
        len(gold_top1),
        len(base_risk_on),
        len(gold_raqm),
        len(defender_raqm),
        len(difference),
    }
    if len(lengths) != 1:
        raise ValueError("fast relative Gold arrays must have equal length")
    observations = len(gold_top1)
    effective = np.empty(observations, dtype=bool)
    active_values = np.empty(observations, dtype=bool)
    changed_values = np.zeros(observations, dtype=bool)
    override_values = np.empty(observations, dtype=bool)
    active = False
    active_days = 0
    entry_streak = 0
    exit_streak = 0
    previous_gold_top1 = False
    for position in range(observations):
        is_gold = bool(gold_top1[position])
        gold_value = float(gold_raqm[position])
        defender_value = float(defender_raqm[position])
        difference_value = float(difference[position])
        finite = bool(
            np.isfinite(gold_value)
            and np.isfinite(defender_value)
            and np.isfinite(difference_value)
        )
        healthy = bool(finite and gold_value > 0.0)
        entry_qualified = bool(
            is_gold and healthy and difference_value >= spec.entry_difference
        )
        relative_exit = bool(
            is_gold and finite and difference_value <= spec.exit_difference
        )
        weak = bool(is_gold and finite and gold_value <= 0.0)
        changed = False
        if not is_gold:
            active = False
            active_days = 0
            entry_streak = 0
            exit_streak = 0
        elif not previous_gold_top1:
            active = False
            active_days = 0
            entry_streak = 0
            exit_streak = 0
        if is_gold and not active:
            exit_streak = 0
            entry_streak = entry_streak + 1 if entry_qualified else 0
            if entry_streak >= spec.entry_confirmation_days:
                active = True
                changed = True
                active_days = 0
                entry_streak = 0
        elif is_gold and active:
            entry_streak = 0
            exit_qualified = weak or relative_exit
            exit_streak = exit_streak + 1 if exit_qualified else 0
            if exit_streak >= spec.exit_confirmation_days:
                if weak or active_days >= spec.minimum_gold_hold_days:
                    active = False
                    changed = True
                    active_days = 0
                    exit_streak = 0
        base_value = bool(base_risk_on[position])
        if is_gold:
            effective_value = (
                base_value or active
                if spec.override_mode == GOLD_EXEMPTION_ONLY
                else active
            )
        else:
            effective_value = base_value
        effective[position] = effective_value
        active_values[position] = active
        changed_values[position] = changed
        override_values[position] = is_gold and effective_value != base_value
        if active:
            active_days += 1
        previous_gold_top1 = is_gold
    return FastRelativeGoldState(
        effective_risk_on=effective,
        gold_overlay_active=active_values,
        gold_overlay_changed=changed_values,
        gold_overrides_base=override_values,
    )


def run_relative_gold_overlay(
    data: ExactExecutionData,
    momentum_target: pd.Series,
    base_risk_on: pd.Series,
    metrics_at_open: pd.DataFrame,
    spec: RelativeGoldOverlaySpec,
) -> RelativeGoldOverlayRun:
    state = relative_gold_overlay_state(
        data.calendar,
        momentum_target,
        base_risk_on,
        metrics_at_open,
        spec,
    )
    defender = data.candidate_index[DEFENDER_CANDIDATE]
    requested = np.where(
        state["effective_risk_on"].to_numpy(bool), data.momentum_target, defender
    ).astype(int)
    returns, actual, switches = exact_candidate_schedule(data, requested)
    entries = state["gold_overlay_changed"].astype(bool) & state[
        "gold_overlay_active"
    ].astype(bool)
    return RelativeGoldOverlayRun(
        spec=spec,
        state=state,
        returns=returns,
        requested_target=requested,
        actual_target=actual,
        gold_entries=int(entries.sum()),
        gold_allowed_days=int(state["gold_overlay_active"].sum()),
        override_days=int(state["gold_overrides_base"].sum()),
        candidate_switches=switches,
    )
