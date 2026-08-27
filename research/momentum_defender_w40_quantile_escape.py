"""Rolling-500 quantile thresholds for asset-specific W40 Top1 escapes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

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


HISTORY_WINDOW = 500
MIN_HISTORY = 252


@dataclass(frozen=True)
class QuantileXYPolicy:
    entry_a: float
    exit_b: float

    def __post_init__(self) -> None:
        if not 0.0 < self.exit_b <= self.entry_a < 1.0:
            raise ValueError("quantile policy requires 0 < B <= A < 1")

    @property
    def policy_id(self) -> str:
        return f"a{self.entry_a:.2f}_b{self.exit_b:.2f}"


def _validate_policies(
    policies: Mapping[str, QuantileXYPolicy | None],
) -> None:
    if set(policies) != set(MOMENTUM_ASSETS):
        raise ValueError("quantile policies must contain all four Momentum ETFs")


def policy_set_id(
    defender_c: float,
    policies: Mapping[str, QuantileXYPolicy | None],
) -> str:
    _validate_policies(policies)
    return f"c{defender_c:.2f}__" + "__".join(
        f"{asset.split('.')[0]}="
        f"{policies[asset].policy_id if policies[asset] else 'off'}"
        for asset in MOMENTUM_ASSETS
    )


def rolling_quantiles_at_open(
    metrics_at_open: pd.DataFrame,
    quantiles: Sequence[float],
    *,
    history_window: int = HISTORY_WINDOW,
    min_history: int = MIN_HISTORY,
) -> dict[tuple[str, float], pd.Series]:
    """Return strict-lag quantiles of already prior-close-aligned QM values."""
    if history_window < min_history or min_history < 1:
        raise ValueError("invalid rolling quantile history")
    result = {}
    for asset in (*MOMENTUM_ASSETS, DEFENDER_CANDIDATE):
        values = metrics_at_open[asset].astype(float)
        for quantile in quantiles:
            q = float(quantile)
            if not 0.0 < q < 1.0:
                raise ValueError("quantiles must lie in (0, 1)")
            result[asset, q] = values.shift(1).rolling(
                history_window, min_periods=min_history
            ).quantile(q)
    return result


def quantile_escape_schedule(
    context: GoldOverrideContext,
    formal_state: pd.DataFrame,
    metrics: pd.DataFrame,
    quantile_frames: Mapping[tuple[str, float], pd.Series],
    defender_c: float,
    policies: Mapping[str, QuantileXYPolicy | None],
) -> pd.DataFrame:
    """Use dynamic QA-QC entry and QB-QC exit lines for current Top1."""
    _validate_policies(policies)
    calendar = context.calendar
    if not (
        calendar.equals(formal_state.index) and calendar.equals(metrics.index)
    ):
        raise ValueError("quantile escape inputs must share formal calendar")
    defender_anchor = quantile_frames[DEFENDER_CANDIDATE, float(defender_c)]
    active = False
    entry_asset: str | None = None
    escape_held_days = 0
    defender_held_days = 0
    previous_target = str(context.initial_previous_candidate)
    rows = []
    for timestamp in calendar:
        base_risk_on = bool(formal_state.at[timestamp, "risk_on"])
        top1 = str(context.momentum_target.loc[timestamp])
        policy = policies[top1]
        top1_metric = metrics.at[timestamp, top1]
        defender_metric = metrics.at[timestamp, DEFENDER_CANDIDATE]
        difference = top1_metric - defender_metric
        if policy is None:
            entry_line = np.nan
            exit_line = np.nan
            top1_entry_anchor = np.nan
            top1_exit_anchor = np.nan
            defender_quantile_anchor = defender_anchor.loc[timestamp]
        else:
            top1_entry_anchor = quantile_frames[
                top1, policy.entry_a
            ].loc[timestamp]
            top1_exit_anchor = quantile_frames[
                top1, policy.exit_b
            ].loc[timestamp]
            defender_quantile_anchor = defender_anchor.loc[timestamp]
            entry_line = top1_entry_anchor - defender_quantile_anchor
            exit_line = top1_exit_anchor - defender_quantile_anchor
        entry_qualified = bool(
            policy is not None
            and not base_risk_on
            and defender_held_days >= DEFENDER_ELIGIBILITY_DAYS
            and all(pd.notna(value) for value in (difference, entry_line))
            and float(difference) > float(entry_line)
        )
        exit_qualified = bool(
            active
            and escape_held_days >= TOP1_HARD_HOLD_DAYS
            and not base_risk_on
            and (
                policy is None
                or (
                    all(pd.notna(value) for value in (difference, exit_line))
                    and float(difference) < float(exit_line)
                )
            )
        )
        previous_active = active
        reason = "hold"
        if active:
            if escape_held_days < TOP1_HARD_HOLD_DAYS:
                assert entry_asset is not None
                target = entry_asset
                reason = "quantile_escape_hard_hold"
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
                    "quantile_escape_return_disabled_top1"
                    if policy is None
                    else "quantile_escape_return_below_dynamic_y"
                )
            else:
                target = top1
                reason = (
                    "quantile_escape_normal_rotation"
                    if top1 != previous_target
                    else "quantile_escape_momentum_hold"
                )
        elif base_risk_on:
            target = top1
            reason = "base_w40_momentum"
        elif entry_qualified:
            active = True
            entry_asset = top1
            escape_held_days = 0
            target = entry_asset
            reason = "quantile_escape_break_defender_lock"
        else:
            target = DEFENDER_CANDIDATE
            reason = "base_w40_defender"
        entry = active and not previous_active
        returned = previous_active and not active and target == DEFENDER_CANDIDATE
        rows.append(
            {
                "date": timestamp,
                "base_w40_risk_on": base_risk_on,
                "base_w40_held_days_at_open": int(
                    formal_state.at[timestamp, "held_days_at_open"]
                ),
                "momentum_top1": top1,
                "top1_metric_at_open": top1_metric,
                "defender_metric_at_open": defender_metric,
                "metric_difference_at_open": difference,
                "current_top1_policy_enabled": policy is not None,
                "current_top1_entry_a": policy.entry_a if policy else np.nan,
                "current_top1_exit_b": policy.exit_b if policy else np.nan,
                "common_defender_c": defender_c,
                "top1_entry_quantile_at_open": top1_entry_anchor,
                "top1_exit_quantile_at_open": top1_exit_anchor,
                "defender_quantile_at_open": defender_quantile_anchor,
                "dynamic_entry_line_at_open": entry_line,
                "dynamic_exit_line_at_open": exit_line,
                "actual_defender_held_days_at_open": defender_held_days,
                "escape_active": active,
                "escape_entry": entry,
                "escape_return_to_defender": returned,
                "escape_entry_asset": entry_asset,
                "escape_held_days_at_open": escape_held_days,
                "entry_qualified": entry_qualified,
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


@dataclass(frozen=True)
class QuantileEscapeBacktest:
    defender_c: float
    policies: Mapping[str, QuantileXYPolicy | None]
    state: pd.DataFrame
    daily: pd.DataFrame
    audit: Mapping[str, object]


def run_quantile_escape(
    context: GoldOverrideContext,
    formal_state: pd.DataFrame,
    defender_c: float,
    policies: Mapping[str, QuantileXYPolicy | None],
    *,
    metrics: pd.DataFrame | None = None,
    quantile_frames: Mapping[tuple[str, float], pd.Series] | None = None,
) -> QuantileEscapeBacktest:
    applied = quality_metrics_at_open(context) if metrics is None else metrics
    required_quantiles = {float(defender_c)}
    for policy in policies.values():
        if policy is not None:
            required_quantiles.update((policy.entry_a, policy.exit_b))
    frames = (
        rolling_quantiles_at_open(applied, sorted(required_quantiles))
        if quantile_frames is None
        else quantile_frames
    )
    state = quantile_escape_schedule(
        context, formal_state, applied, frames, defender_c, policies
    )
    daily = simulate_candidate_schedule(
        state["target_candidate"],
        context.interfaces,
        context.initial_previous_candidate,
    )
    entries = state["escape_entry"].astype(bool)
    returns = state["escape_return_to_defender"].astype(bool)
    hard_hold_violations = 0
    for timestamp in state.index[entries]:
        start = state.index.get_loc(timestamp)
        interval = state.iloc[start : start + TOP1_HARD_HOLD_DAYS]
        asset = str(state.at[timestamp, "escape_entry_asset"])
        hard_hold_violations += int(
            interval["target_candidate"].astype(str).ne(asset).sum()
        )
    invalid_entries = int(
        (
            state.loc[entries, "actual_defender_held_days_at_open"].lt(
                DEFENDER_ELIGIBILITY_DAYS
            )
            | state.loc[entries, "metric_difference_at_open"].le(
                state.loc[entries, "dynamic_entry_line_at_open"]
            )
        ).sum()
    )
    invalid_returns = int(
        (
            state.loc[returns, "escape_held_days_at_open"].lt(
                TOP1_HARD_HOLD_DAYS
            )
        ).sum()
    )
    blocked = int(daily["switch_blocked_untradable"].sum())
    nav_error = float(
        ((1.0 + daily["return"]).cumprod() - daily["nav"]).abs().max()
    )
    if invalid_entries or invalid_returns or hard_hold_violations or blocked or nav_error > 1e-12:
        raise AssertionError(
            "quantile escape audit failed: "
            f"entries={invalid_entries}, returns={invalid_returns}, "
            f"hard_hold={hard_hold_violations}, blocked={blocked}, nav={nav_error:.3e}"
        )
    escape_rows = state["escape_active"].astype(bool)
    entry_assets = state.loc[entries, "escape_entry_asset"].value_counts()
    asset_days = state.loc[escape_rows, "target_candidate"].value_counts()
    lock_breaks = entries & state["base_w40_held_days_at_open"].lt(30)
    audit = {
        "status": "passed",
        "policy_set_id": policy_set_id(defender_c, policies),
        "enabled_assets": int(sum(policy is not None for policy in policies.values())),
        "escape_entries": int(entries.sum()),
        "escape_returns_to_defender": int(returns.sum()),
        "escape_days": int(escape_rows.sum()),
        "lock_break_entries": int(lock_breaks.sum()),
        "entry_count_by_asset": {
            asset: int(entry_assets.get(asset, 0)) for asset in MOMENTUM_ASSETS
        },
        "escape_days_by_asset": {
            asset: int(asset_days.get(asset, 0)) for asset in MOMENTUM_ASSETS
        },
        "nav_reconstruction_max_abs_error": nav_error,
        "performance": performance(daily["return"].astype(float)),
    }
    return QuantileEscapeBacktest(
        defender_c, dict(policies), state, daily, audit
    )
