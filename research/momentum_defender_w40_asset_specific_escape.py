"""Asset-specific X/Y policies for the formal W40 Top1 escape overlay."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from research.defender_curve_momentum import DEFENDER_CANDIDATE
from research.momentum_defender_gold_override import (
    GoldOverrideContext,
    simulate_candidate_schedule,
)
from research.momentum_defender_occam import MOMENTUM_ASSETS, performance
from research.momentum_defender_w40_top1_escape import (
    DEFENDER_ELIGIBILITY_DAYS,
    TOP1_HARD_HOLD_DAYS,
    quality_metrics_at_open,
)


@dataclass(frozen=True)
class AssetXYPolicy:
    entry_x: float
    exit_y: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.entry_x) or not np.isfinite(self.exit_y):
            raise ValueError("asset X/Y must be finite")
        if self.exit_y > self.entry_x:
            raise ValueError("asset Y must not exceed X")

    @property
    def policy_id(self) -> str:
        return f"x{self.entry_x:+.4f}_y{self.exit_y:+.4f}"


@dataclass(frozen=True)
class AssetSpecificW40EscapeBacktest:
    policies: Mapping[str, AssetXYPolicy | None]
    state: pd.DataFrame
    daily: pd.DataFrame
    audit: Mapping[str, object]


def _validate_policies(
    policies: Mapping[str, AssetXYPolicy | None],
) -> None:
    if set(policies) != set(MOMENTUM_ASSETS):
        raise ValueError("policies must contain exactly the four Momentum ETFs")


def policy_set_id(policies: Mapping[str, AssetXYPolicy | None]) -> str:
    _validate_policies(policies)
    return "__".join(
        f"{asset.split('.')[0]}="
        f"{policies[asset].policy_id if policies[asset] else 'off'}"
        for asset in MOMENTUM_ASSETS
    )


def asset_specific_escape_schedule(
    context: GoldOverrideContext,
    formal_state: pd.DataFrame,
    metrics: pd.DataFrame,
    policies: Mapping[str, AssetXYPolicy | None],
    *,
    immediate_entry_veto: bool = False,
) -> pd.DataFrame:
    """Use Top1 X/Y, optionally vetoing the first executable Defender entry."""
    _validate_policies(policies)
    calendar = context.calendar
    if not (
        calendar.equals(formal_state.index) and calendar.equals(metrics.index)
    ):
        raise ValueError("asset-specific inputs must share the formal calendar")
    active = False
    entry_asset: str | None = None
    escape_held_days = 0
    defender_held_days = 0
    previous_target = str(context.initial_previous_candidate)
    rows = []
    for timestamp in calendar:
        base_risk_on = bool(formal_state.at[timestamp, "risk_on"])
        base_state_changed = bool(
            formal_state.at[timestamp, "state_changed"]
            if "state_changed" in formal_state
            else False
        )
        base_defender_entry = base_state_changed and not base_risk_on
        top1 = str(context.momentum_target.loc[timestamp])
        policy = policies[top1]
        top1_metric = metrics.at[timestamp, top1]
        defender_metric = metrics.at[timestamp, DEFENDER_CANDIDATE]
        difference = top1_metric - defender_metric
        metric_entry_qualified = bool(
            policy is not None
            and not base_risk_on
            and pd.notna(difference)
            and float(difference) > policy.entry_x
        )
        immediate_entry_veto_qualified = bool(
            immediate_entry_veto
            and base_defender_entry
            and metric_entry_qualified
        )
        entry_qualified = bool(
            metric_entry_qualified
            and (
                defender_held_days >= DEFENDER_ELIGIBILITY_DAYS
                or immediate_entry_veto_qualified
            )
        )
        exit_qualified = bool(
            active
            and escape_held_days >= TOP1_HARD_HOLD_DAYS
            and not base_risk_on
            and (
                policy is None
                or (
                    pd.notna(difference)
                    and float(difference) < policy.exit_y
                )
            )
        )
        previous_active = active
        reason = "hold"
        if active:
            if escape_held_days < TOP1_HARD_HOLD_DAYS:
                assert entry_asset is not None
                target = entry_asset
                reason = "asset_escape_hard_hold"
            elif base_risk_on:
                active = False
                entry_asset = None
                target = top1
                reason = "base_w40_recovered_to_momentum"
            elif exit_qualified:
                active = False
                entry_asset = None
                target = DEFENDER_CANDIDATE
                reason = (
                    "asset_escape_return_disabled_top1"
                    if policy is None
                    else "asset_escape_return_below_y"
                )
            else:
                target = top1
                reason = (
                    "asset_escape_normal_rotation"
                    if top1 != previous_target
                    else "asset_escape_momentum_hold"
                )
        elif base_risk_on:
            target = top1
            reason = "base_w40_momentum"
        elif entry_qualified:
            active = True
            entry_asset = top1
            escape_held_days = 0
            target = entry_asset
            reason = (
                "asset_escape_veto_defender_entry"
                if immediate_entry_veto_qualified
                else "asset_escape_break_defender_lock"
            )
        else:
            target = DEFENDER_CANDIDATE
            reason = "base_w40_defender"

        entry = active and not previous_active
        returned = previous_active and not active and target == DEFENDER_CANDIDATE
        rows.append(
            {
                "date": timestamp,
                "base_w40_risk_on": base_risk_on,
                "base_w40_defender_entry": base_defender_entry,
                "base_w40_held_days_at_open": int(
                    formal_state.at[timestamp, "held_days_at_open"]
                ),
                "momentum_top1": top1,
                "top1_metric_at_open": top1_metric,
                "defender_metric_at_open": defender_metric,
                "metric_difference_at_open": difference,
                "current_top1_policy_enabled": policy is not None,
                "current_top1_entry_x": policy.entry_x if policy else np.nan,
                "current_top1_exit_y": policy.exit_y if policy else np.nan,
                "actual_defender_held_days_at_open": defender_held_days,
                "escape_active": active,
                "escape_entry": entry,
                "escape_return_to_defender": returned,
                "escape_entry_asset": entry_asset,
                "escape_held_days_at_open": escape_held_days,
                "entry_qualified": entry_qualified,
                "immediate_entry_veto_qualified": immediate_entry_veto_qualified,
                "exit_qualified": exit_qualified,
                "state_reason": reason,
                "target_candidate": target,
            }
        )
        if target == DEFENDER_CANDIDATE:
            defender_held_days = (
                defender_held_days + 1
                if previous_target == DEFENDER_CANDIDATE
                else 1
            )
        else:
            defender_held_days = 0
        if active:
            escape_held_days += 1
        else:
            escape_held_days = 0
        previous_target = str(target)
    return pd.DataFrame(rows).set_index("date")


def run_asset_specific_w40_escape(
    context: GoldOverrideContext,
    formal_state: pd.DataFrame,
    policies: Mapping[str, AssetXYPolicy | None],
    *,
    metrics: pd.DataFrame | None = None,
    immediate_entry_veto: bool = False,
) -> AssetSpecificW40EscapeBacktest:
    applied = quality_metrics_at_open(context) if metrics is None else metrics
    state = asset_specific_escape_schedule(
        context,
        formal_state,
        applied,
        policies,
        immediate_entry_veto=immediate_entry_veto,
    )
    daily = simulate_candidate_schedule(
        state["target_candidate"],
        context.interfaces,
        context.initial_previous_candidate,
    )
    entries = state["escape_entry"].astype(bool)
    returns = state["escape_return_to_defender"].astype(bool)
    invalid_entries = int(
        (
            state.loc[entries, "base_w40_risk_on"].astype(bool)
            | ~state.loc[entries, "current_top1_policy_enabled"].astype(bool)
            | (
                state.loc[entries, "actual_defender_held_days_at_open"].lt(
                    DEFENDER_ELIGIBILITY_DAYS
                )
                & ~state.loc[
                    entries, "immediate_entry_veto_qualified"
                ].astype(bool)
            )
            | state.loc[entries, "metric_difference_at_open"].le(
                state.loc[entries, "current_top1_entry_x"]
            )
        ).sum()
    )
    invalid_returns = int(
        (
            state.loc[returns, "base_w40_risk_on"].astype(bool)
            | state.loc[returns, "escape_held_days_at_open"].lt(
                TOP1_HARD_HOLD_DAYS
            )
        ).sum()
    )
    hard_hold_violations = 0
    for timestamp in state.index[entries]:
        start = state.index.get_loc(timestamp)
        interval = state.iloc[start : start + TOP1_HARD_HOLD_DAYS]
        asset = str(state.at[timestamp, "escape_entry_asset"])
        hard_hold_violations += int(
            interval["target_candidate"].astype(str).ne(asset).sum()
        )
    blocked = int(daily["switch_blocked_untradable"].sum())
    nav_error = float(
        ((1.0 + daily["return"]).cumprod() - daily["nav"]).abs().max()
    )
    if invalid_entries or invalid_returns or hard_hold_violations or blocked or nav_error > 1e-12:
        raise AssertionError(
            "asset-specific W40 escape audit failed: "
            f"entries={invalid_entries}, returns={invalid_returns}, "
            f"hard_hold={hard_hold_violations}, blocked={blocked}, nav={nav_error:.3e}"
        )
    escape_rows = state["escape_active"].astype(bool)
    asset_days = state.loc[escape_rows, "target_candidate"].value_counts()
    entry_assets = state.loc[entries, "escape_entry_asset"].value_counts()
    lock_breaks = entries & state["base_w40_held_days_at_open"].lt(30)
    audit = {
        "status": "passed",
        "policy_set_id": policy_set_id(policies),
        "enabled_assets": int(sum(policy is not None for policy in policies.values())),
        "escape_entries": int(entries.sum()),
        "immediate_entry_veto_enabled": bool(immediate_entry_veto),
        "immediate_entry_veto_entries": int(
            state["immediate_entry_veto_qualified"].astype(bool).sum()
        ),
        "escape_returns_to_defender": int(returns.sum()),
        "escape_days": int(escape_rows.sum()),
        "lock_break_entries": int(lock_breaks.sum()),
        "escape_normal_rotations": int(
            state["state_reason"].eq("asset_escape_normal_rotation").sum()
        ),
        "entry_count_by_asset": {
            asset: int(entry_assets.get(asset, 0)) for asset in MOMENTUM_ASSETS
        },
        "escape_days_by_asset": {
            asset: int(asset_days.get(asset, 0)) for asset in MOMENTUM_ASSETS
        },
        "nav_reconstruction_max_abs_error": nav_error,
        "performance": performance(daily["return"].astype(float)),
    }
    return AssetSpecificW40EscapeBacktest(
        dict(policies), state, daily, audit
    )
