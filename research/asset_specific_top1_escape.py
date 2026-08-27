"""Asset-specific escape policies for Momentum Top-1 while C2 holds Defender."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from research.defender_curve_momentum import DEFENDER_CANDIDATE
from research.momentum_defender_gold_override import (
    METRICS,
    GoldOverrideContext,
    simulate_candidate_schedule,
)
from research.momentum_defender_occam import MOMENTUM_ASSETS, performance
from research.momentum_top1_defender_escape import all_metrics_at_open


@dataclass(frozen=True)
class AssetEscapePolicy:
    metric: str
    window: int
    entry_difference: float
    exit_difference: float
    absolute_minimum: float = 0.0
    min_hold_days: int = 1

    def __post_init__(self) -> None:
        if self.metric not in METRICS:
            raise ValueError(f"metric must be one of {METRICS}")
        if self.window < 2 or self.min_hold_days < 1:
            raise ValueError("window must be >=2 and min_hold_days positive")
        if self.exit_difference > self.entry_difference:
            raise ValueError("exit_difference must not exceed entry_difference")

    def policy_id(self) -> str:
        return (
            f"{self.metric}_w{self.window}_en{self.entry_difference:+.4f}_"
            f"ex{self.exit_difference:+.4f}_abs{self.absolute_minimum:+.4f}_"
            f"h{self.min_hold_days}"
        )


@dataclass(frozen=True)
class AssetSpecificEscapeBacktest:
    policies: Mapping[str, AssetEscapePolicy | None]
    state: pd.DataFrame
    daily: pd.DataFrame
    audit: dict[str, object]


def _qualified(
    metrics: pd.DataFrame,
    timestamp: pd.Timestamp,
    asset: str,
    policy: AssetEscapePolicy,
    *,
    entry: bool,
) -> tuple[bool, float, float, float]:
    asset_metric = metrics.at[timestamp, asset]
    defender_metric = metrics.at[timestamp, DEFENDER_CANDIDATE]
    difference = asset_metric - defender_metric
    threshold = policy.entry_difference if entry else policy.exit_difference
    qualified = bool(
        pd.notna(asset_metric)
        and pd.notna(difference)
        and float(asset_metric) > policy.absolute_minimum
        and float(difference) > threshold
    )
    return qualified, float(asset_metric), float(defender_metric), float(difference)


def asset_specific_schedule(
    context: GoldOverrideContext,
    policies: Mapping[str, AssetEscapePolicy | None],
    metric_frames: Mapping[tuple[str, int], pd.DataFrame],
) -> pd.DataFrame:
    """Apply each asset's own policy only when that asset is Momentum Top-1."""
    if set(policies) != set(MOMENTUM_ASSETS):
        raise ValueError("policies must contain exactly the four Momentum assets")
    base_risk_on = context.integrated.result.state["risk_on"].astype(bool)
    active_asset: str | None = None
    held_days = 10**9
    rows: list[dict[str, object]] = []
    for timestamp in context.calendar:
        previous_asset = active_asset
        top1 = str(context.momentum_target.loc[timestamp])
        policy = policies[top1]
        entry_qualified = False
        stay_qualified = False
        top1_metric = np.nan
        defender_metric = np.nan
        difference = np.nan
        if policy is not None:
            frame = metric_frames[(policy.metric, policy.window)]
            entry_qualified, top1_metric, defender_metric, difference = _qualified(
                frame, timestamp, top1, policy, entry=True
            )
            stay_qualified = _qualified(
                frame, timestamp, top1, policy, entry=False
            )[0]
        reason = "hold"
        if bool(base_risk_on.loc[timestamp]):
            active_asset = None
            if previous_asset is not None:
                reason = "base_c2_returned_to_momentum"
        elif active_asset is None:
            if policy is not None and entry_qualified:
                active_asset = top1
                held_days = 0
                reason = "asset_escape_entry"
        else:
            current_policy = policies[active_asset]
            assert current_policy is not None
            if held_days >= current_policy.min_hold_days:
                if top1 == active_asset:
                    if not stay_qualified:
                        active_asset = None
                        held_days = 0
                        reason = "asset_escape_exit"
                elif policy is not None and entry_qualified:
                    active_asset = top1
                    held_days = 0
                    reason = "asset_escape_rotation"
                else:
                    active_asset = None
                    held_days = 0
                    reason = "asset_escape_exit_on_top1_change"

        target = (
            top1
            if bool(base_risk_on.loc[timestamp])
            else active_asset if active_asset is not None else DEFENDER_CANDIDATE
        )
        rows.append(
            {
                "date": timestamp,
                "base_c2_risk_on": bool(base_risk_on.loc[timestamp]),
                "momentum_top1": top1,
                "active_escape_asset": active_asset,
                "escape_active": active_asset is not None,
                "escape_asset_changed": active_asset != previous_asset,
                "state_reason": reason,
                "held_days_at_open": held_days,
                "top1_metric_at_open": top1_metric,
                "defender_metric_at_open": defender_metric,
                "metric_difference_at_open": difference,
                "entry_qualified": entry_qualified,
                "stay_qualified": stay_qualified,
                "target_candidate": target,
            }
        )
        if active_asset is not None:
            held_days += 1
    return pd.DataFrame(rows).set_index("date")


def policy_set_id(policies: Mapping[str, AssetEscapePolicy | None]) -> str:
    return "__".join(
        f"{asset.split('.')[0]}={policies[asset].policy_id() if policies[asset] else 'off'}"
        for asset in MOMENTUM_ASSETS
    )


def run_asset_specific_escape(
    context: GoldOverrideContext,
    policies: Mapping[str, AssetEscapePolicy | None],
    *,
    metric_frames: Mapping[tuple[str, int], pd.DataFrame] | None = None,
) -> AssetSpecificEscapeBacktest:
    frames = dict(metric_frames or {})
    for policy in policies.values():
        if policy is None:
            continue
        key = (policy.metric, policy.window)
        if key not in frames:
            frames[key] = all_metrics_at_open(
                context.curves, policy.metric, policy.window
            )
    state = asset_specific_schedule(context, policies, frames)
    daily = simulate_candidate_schedule(
        state["target_candidate"],
        context.interfaces,
        context.initial_previous_candidate,
    )
    entries = state["state_reason"].eq("asset_escape_entry")
    invalid_entries = int(
        (
            state.loc[entries, "base_c2_risk_on"].astype(bool)
            | ~state.loc[entries, "entry_qualified"].astype(bool)
        ).sum()
    )
    momentum_rows = state["base_c2_risk_on"].astype(bool)
    momentum_matches = bool(
        state.loc[momentum_rows, "target_candidate"].equals(
            context.momentum_target.loc[momentum_rows]
        )
    )
    nav_error = float(
        ((1.0 + daily["return"]).cumprod() - daily["nav"]).abs().max()
    )
    if invalid_entries or not momentum_matches or nav_error > 1e-12:
        raise AssertionError("asset-specific escape audit failed")
    asset_days = state.loc[state["escape_active"], "active_escape_asset"].value_counts()
    audit = {
        "status": "passed",
        "policy_set_id": policy_set_id(policies),
        "invalid_entries": invalid_entries,
        "momentum_rows_match_base_c2": momentum_matches,
        "nav_reconstruction_max_abs_error": nav_error,
        "escape_entries": int(entries.sum()),
        "escape_rotations": int(state["state_reason"].eq("asset_escape_rotation").sum()),
        "escape_days": int(state["escape_active"].sum()),
        "escape_asset_days": {str(key): int(value) for key, value in asset_days.items()},
        "switches": int(daily["switched"].sum()),
        "performance": performance(daily["return"]),
    }
    return AssetSpecificEscapeBacktest(dict(policies), state, daily, audit)


def build_policy_grid(settings: Mapping[str, Mapping[str, Sequence[float | int]]]) -> list[AssetEscapePolicy]:
    policies: list[AssetEscapePolicy] = []
    for metric, values in settings.items():
        for window, entry, exit_, hold in product(
            values["windows"],
            values["entry_differences"],
            values["exit_differences"],
            values["min_hold_days"],
        ):
            if float(exit_) > float(entry):
                continue
            policies.append(
                AssetEscapePolicy(
                    metric=str(metric),
                    window=int(window),
                    entry_difference=float(entry),
                    exit_difference=float(exit_),
                    absolute_minimum=0.0,
                    min_hold_days=int(hold),
                )
            )
    return policies


def single_asset_search(
    context: GoldOverrideContext,
    asset: str,
    policy_grid: Sequence[AssetEscapePolicy],
    metric_frames: Mapping[tuple[str, int], pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    records: list[dict[str, object]] = []
    returns: dict[str, np.ndarray] = {}
    for policy in policy_grid:
        policies = {
            candidate: policy if candidate == asset else None
            for candidate in MOMENTUM_ASSETS
        }
        run = run_asset_specific_escape(
            context, policies, metric_frames=metric_frames
        )
        candidate_id = f"{asset}|{policy.policy_id()}"
        records.append(
            {
                "candidate_id": candidate_id,
                "asset": asset,
                **asdict(policy),
                "escape_entries": run.audit["escape_entries"],
                "escape_days": run.audit["escape_days"],
            }
        )
        returns[candidate_id] = run.daily["return"].to_numpy(float)
    return (
        pd.DataFrame(records).set_index("candidate_id"),
        pd.DataFrame(returns, index=context.calendar),
    )


def combination_search(
    context: GoldOverrideContext,
    options: Mapping[str, Sequence[AssetEscapePolicy | None]],
    metric_frames: Mapping[tuple[str, int], pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Mapping[str, AssetEscapePolicy | None]]]:
    records: list[dict[str, object]] = []
    returns: dict[str, np.ndarray] = {}
    policy_sets: dict[str, Mapping[str, AssetEscapePolicy | None]] = {}
    ordered_options = [options[asset] for asset in MOMENTUM_ASSETS]
    for selected in product(*ordered_options):
        policies = dict(zip(MOMENTUM_ASSETS, selected, strict=True))
        run = run_asset_specific_escape(
            context, policies, metric_frames=metric_frames
        )
        candidate_id = policy_set_id(policies)
        policy_sets[candidate_id] = policies
        records.append(
            {
                "candidate_id": candidate_id,
                "enabled_assets": int(sum(value is not None for value in selected)),
                "escape_entries": run.audit["escape_entries"],
                "escape_rotations": run.audit["escape_rotations"],
                "escape_days": run.audit["escape_days"],
                **{
                    f"policy_{asset}": policies[asset].policy_id()
                    if policies[asset]
                    else "off"
                    for asset in MOMENTUM_ASSETS
                },
            }
        )
        returns[candidate_id] = run.daily["return"].to_numpy(float)
    return (
        pd.DataFrame(records).set_index("candidate_id"),
        pd.DataFrame(returns, index=context.calendar),
        policy_sets,
    )
