"""Top-1 quality-momentum escape from the formal W40 Defender sleeve."""

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
from research.momentum_top1_defender_escape import (
    QUALITY_METRIC,
    all_metrics_at_open,
)


QUALITY_WINDOW = 20
DEFENDER_ELIGIBILITY_DAYS = 5
TOP1_HARD_HOLD_DAYS = 5


@dataclass(frozen=True)
class W40Top1EscapeSpec:
    entry_difference: float
    exit_difference: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.entry_difference) or not np.isfinite(
            self.exit_difference
        ):
            raise ValueError("escape thresholds must be finite")
        if self.exit_difference > self.entry_difference:
            raise ValueError("exit threshold Y must not exceed entry threshold X")

    @property
    def candidate_id(self) -> str:
        return (
            f"w40_top1_qm20_escape_x{self.entry_difference:+.4f}_"
            f"y{self.exit_difference:+.4f}_d5_h5"
        )


@dataclass(frozen=True)
class W40Top1EscapeBacktest:
    spec: W40Top1EscapeSpec
    metrics_at_open: pd.DataFrame
    state: pd.DataFrame
    daily: pd.DataFrame
    audit: Mapping[str, object]


def quality_metrics_at_open(context: GoldOverrideContext) -> pd.DataFrame:
    """Use the registered log/log QM20 on all ETFs and formal Defender NAV."""
    return all_metrics_at_open(
        context.curves,
        QUALITY_METRIC,
        QUALITY_WINDOW,
    )


def top1_escape_schedule(
    context: GoldOverrideContext,
    formal_state: pd.DataFrame,
    metrics: pd.DataFrame,
    spec: W40Top1EscapeSpec,
) -> pd.DataFrame:
    """Apply the five-day eligibility and five-day hard-hold escape state."""
    calendar = context.calendar
    if not (
        calendar.equals(formal_state.index) and calendar.equals(metrics.index)
    ):
        raise ValueError("escape inputs must share the formal calendar")
    active = False
    entry_asset: str | None = None
    escape_held_days = 0
    defender_held_days = 0
    previous_target = str(context.initial_previous_candidate)
    rows: list[dict[str, object]] = []

    for timestamp in calendar:
        base_risk_on = bool(formal_state.at[timestamp, "risk_on"])
        top1 = str(context.momentum_target.loc[timestamp])
        top1_metric = metrics.at[timestamp, top1]
        defender_metric = metrics.at[timestamp, DEFENDER_CANDIDATE]
        difference = top1_metric - defender_metric
        entry_qualified = bool(
            not base_risk_on
            and defender_held_days >= DEFENDER_ELIGIBILITY_DAYS
            and pd.notna(difference)
            and float(difference) > spec.entry_difference
        )
        exit_qualified = bool(
            not base_risk_on
            and escape_held_days >= TOP1_HARD_HOLD_DAYS
            and pd.notna(difference)
            and float(difference) < spec.exit_difference
        )
        previous_active = active
        previous_entry_asset = entry_asset
        reason = "hold"

        if active:
            if escape_held_days < TOP1_HARD_HOLD_DAYS:
                assert entry_asset is not None
                target = entry_asset
                reason = "top1_escape_hard_hold"
            elif base_risk_on:
                active = False
                entry_asset = None
                target = top1
                reason = "base_w40_recovered_to_momentum"
            elif exit_qualified:
                active = False
                entry_asset = None
                target = DEFENDER_CANDIDATE
                reason = "top1_escape_return_to_defender"
            else:
                target = top1
                reason = (
                    "top1_escape_normal_rotation"
                    if top1 != previous_target
                    else "top1_escape_momentum_hold"
                )
        elif base_risk_on:
            target = top1
            reason = "base_w40_momentum"
        elif entry_qualified:
            active = True
            entry_asset = top1
            escape_held_days = 0
            target = entry_asset
            reason = "top1_escape_break_defender_lock"
        else:
            target = DEFENDER_CANDIDATE
            reason = "base_w40_defender"

        entry_changed = active and not previous_active
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
                "actual_defender_held_days_at_open": defender_held_days,
                "escape_active": active,
                "escape_entry": entry_changed,
                "escape_return_to_defender": returned,
                "escape_entry_asset": entry_asset,
                "previous_escape_entry_asset": previous_entry_asset,
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


def _validate(
    context: GoldOverrideContext,
    formal_state: pd.DataFrame,
    spec: W40Top1EscapeSpec,
    state: pd.DataFrame,
    daily: pd.DataFrame,
) -> dict[str, object]:
    entries = state["escape_entry"].astype(bool)
    returns = state["escape_return_to_defender"].astype(bool)
    invalid_entries = int(
        (
            state.loc[entries, "base_w40_risk_on"].astype(bool)
            | state.loc[entries, "actual_defender_held_days_at_open"].lt(
                DEFENDER_ELIGIBILITY_DAYS
            )
            | state.loc[entries, "metric_difference_at_open"].le(
                spec.entry_difference
            )
        ).sum()
    )
    invalid_returns = int(
        (
            state.loc[returns, "base_w40_risk_on"].astype(bool)
            | state.loc[returns, "escape_held_days_at_open"].lt(
                TOP1_HARD_HOLD_DAYS
            )
            | state.loc[returns, "metric_difference_at_open"].ge(
                spec.exit_difference
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
    blocked_switches = int(daily["switch_blocked_untradable"].sum())
    nav_error = float(
        ((1.0 + daily["return"]).cumprod() - daily["nav"]).abs().max()
    )
    if (
        invalid_entries
        or invalid_returns
        or hard_hold_violations
        or blocked_switches
        or nav_error > 1e-12
    ):
        raise AssertionError(
            "W40 Top1 escape audit failed: "
            f"entries={invalid_entries}, returns={invalid_returns}, "
            f"hard_hold={hard_hold_violations}, blocked={blocked_switches}, "
            f"nav={nav_error:.3e}"
        )
    escape_rows = state["escape_active"].astype(bool)
    asset_days = state.loc[escape_rows, "target_candidate"].value_counts()
    lock_breaks = entries & state["base_w40_held_days_at_open"].lt(30)
    return {
        "status": "passed",
        "candidate_id": spec.candidate_id,
        "escape_entries": int(entries.sum()),
        "escape_returns_to_defender": int(returns.sum()),
        "escape_days": int(escape_rows.sum()),
        "lock_break_entries": int(lock_breaks.sum()),
        "escape_normal_rotations": int(
            state["state_reason"].eq("top1_escape_normal_rotation").sum()
        ),
        "escape_asset_days": {
            str(asset): int(asset_days.get(asset, 0)) for asset in MOMENTUM_ASSETS
        },
        "nav_reconstruction_max_abs_error": nav_error,
        "performance": performance(daily["return"].astype(float)),
    }


def run_w40_top1_escape(
    context: GoldOverrideContext,
    formal_state: pd.DataFrame,
    spec: W40Top1EscapeSpec,
    *,
    metrics: pd.DataFrame | None = None,
) -> W40Top1EscapeBacktest:
    applied = quality_metrics_at_open(context) if metrics is None else metrics
    state = top1_escape_schedule(context, formal_state, applied, spec)
    daily = simulate_candidate_schedule(
        state["target_candidate"],
        context.interfaces,
        context.initial_previous_candidate,
    )
    audit = _validate(context, formal_state, spec, state, daily)
    return W40Top1EscapeBacktest(spec, applied, state, daily, audit)
