"""Common-formula RAQM override across all four Momentum ETFs."""

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


VOL_FLOOR_ANNUAL = 0.08
WINSOR_LIMIT = 3.0
HARD_MIN_HOLD_DAYS = 5


@dataclass(frozen=True)
class CommonRAQMSpec:
    """One factor definition shared by every Momentum ETF and Defender."""

    window: int = 5
    efficiency_power: float = 1.0
    vol_floor_annual: float = VOL_FLOOR_ANNUAL

    def __post_init__(self) -> None:
        if self.window < 2:
            raise ValueError("window must be at least two")
        if self.efficiency_power not in {0.0, 0.5, 1.0}:
            raise ValueError("efficiency_power must be 0, 0.5, or 1")
        if self.vol_floor_annual <= 0.0:
            raise ValueError("volatility floor must be positive")

    def spec_id(self) -> str:
        return (
            f"raqm_w{self.window}_erpow{self.efficiency_power:.1f}_"
            f"floor{self.vol_floor_annual:.2f}"
        )


@dataclass(frozen=True)
class AssetRAQMThresholds:
    entry_difference: float
    exit_difference: float

    def __post_init__(self) -> None:
        if self.exit_difference > self.entry_difference:
            raise ValueError("exit difference must not exceed entry difference")

    def policy_id(self) -> str:
        return (
            f"en{self.entry_difference:+.2f}_"
            f"ex{self.exit_difference:+.2f}_h{HARD_MIN_HOLD_DAYS}"
        )


@dataclass(frozen=True)
class AllAssetRAQMBacktest:
    spec: CommonRAQMSpec
    policies: Mapping[str, AssetRAQMThresholds | None]
    metrics_at_open: pd.DataFrame
    state: pd.DataFrame
    daily: pd.DataFrame
    audit: dict[str, object]


def common_raqm_at_open(
    curves: pd.DataFrame,
    spec: CommonRAQMSpec,
) -> pd.DataFrame:
    """Compute one causal RAQM definition for all four ETFs and Defender."""

    values: dict[str, pd.Series] = {}
    floor_n = spec.vol_floor_annual * np.sqrt(spec.window / 252.0)
    for candidate in (*MOMENTUM_ASSETS, DEFENDER_CANDIDATE):
        close = curves[candidate].astype(float)
        log_return = np.log(close).diff()
        trailing = np.log(close).diff(spec.window)
        path = log_return.abs().rolling(spec.window).sum()
        volatility = log_return.rolling(spec.window).std(ddof=1) * np.sqrt(
            spec.window
        )
        adjusted_volatility = np.maximum(volatility, floor_n)
        risk_adjusted = (trailing / adjusted_volatility).clip(
            lower=-WINSOR_LIMIT,
            upper=WINSOR_LIMIT,
        )
        if spec.efficiency_power == 0.0:
            efficiency_adjustment = pd.Series(1.0, index=close.index)
        else:
            efficiency = trailing.abs() / path.replace(0.0, np.nan)
            efficiency_adjustment = efficiency.pow(spec.efficiency_power)
        values[candidate] = (risk_adjusted * efficiency_adjustment).shift(1)
    result = pd.DataFrame(values, index=curves.index)
    result.index.name = "date"
    return result


def policy_set_id(
    spec: CommonRAQMSpec,
    policies: Mapping[str, AssetRAQMThresholds | None],
) -> str:
    return "__".join(
        [
            spec.spec_id(),
            *[
                f"{asset.split('.')[0]}="
                f"{policies[asset].policy_id() if policies[asset] else 'off'}"
                for asset in MOMENTUM_ASSETS
            ],
        ]
    )


def _best_entry_candidate(
    timestamp: pd.Timestamp,
    context: GoldOverrideContext,
    metrics: pd.DataFrame,
    policies: Mapping[str, AssetRAQMThresholds | None],
) -> tuple[str | None, dict[str, float]]:
    defender_metric = metrics.at[timestamp, DEFENDER_CANDIDATE]
    margins: dict[str, float] = {}
    for asset in MOMENTUM_ASSETS:
        policy = policies[asset]
        value = metrics.at[timestamp, asset]
        difference = value - defender_metric
        if (
            policy is not None
            and pd.notna(difference)
            and float(difference) > policy.entry_difference
        ):
            margins[asset] = float(difference - policy.entry_difference)
    if not margins:
        return None, margins
    maximum = max(margins.values())
    tied = {
        asset
        for asset, margin in margins.items()
        if np.isclose(margin, maximum, atol=1e-14)
    }
    top1 = str(context.momentum_target.loc[timestamp])
    if top1 in tied:
        return top1, margins
    return next(asset for asset in MOMENTUM_ASSETS if asset in tied), margins


def all_asset_raqm_schedule(
    context: GoldOverrideContext,
    metrics: pd.DataFrame,
    policies: Mapping[str, AssetRAQMThresholds | None],
) -> pd.DataFrame:
    """Generalize the formal hard-min-5 Gold override to all four ETFs."""

    if set(policies) != set(MOMENTUM_ASSETS):
        raise ValueError("policies must contain exactly the four Momentum assets")
    base_risk_on = context.integrated.result.state["risk_on"].astype(bool)
    active_asset: str | None = None
    held_days = 10**9
    rows: list[dict[str, object]] = []
    for timestamp in context.calendar:
        previous_asset = active_asset
        held_before_decision = held_days
        defender_metric = metrics.at[timestamp, DEFENDER_CANDIDATE]
        best_entry, margins = _best_entry_candidate(
            timestamp,
            context,
            metrics,
            policies,
        )
        reason = "hold"
        if active_asset is None:
            if not bool(base_risk_on.loc[timestamp]) and best_entry is not None:
                active_asset = best_entry
                held_days = 0
                reason = "raqm_entry"
        elif held_days >= HARD_MIN_HOLD_DAYS:
            if bool(base_risk_on.loc[timestamp]):
                active_asset = None
                held_days = 0
                reason = "raqm_to_base_momentum_after_min_hold"
            else:
                active_policy = policies[active_asset]
                assert active_policy is not None
                active_difference = (
                    metrics.at[timestamp, active_asset] - defender_metric
                )
                if (
                    pd.isna(active_difference)
                    or float(active_difference) <= active_policy.exit_difference
                ):
                    if best_entry is not None and best_entry != active_asset:
                        active_asset = best_entry
                        held_days = 0
                        reason = "raqm_rotation_after_exit"
                    else:
                        active_asset = None
                        held_days = 0
                        reason = "raqm_to_defender_after_min_hold"

        top1 = str(context.momentum_target.loc[timestamp])
        target = (
            active_asset
            if active_asset is not None
            else top1 if bool(base_risk_on.loc[timestamp]) else DEFENDER_CANDIDATE
        )
        rows.append(
            {
                "date": timestamp,
                "base_c2_risk_on": bool(base_risk_on.loc[timestamp]),
                "momentum_top1": top1,
                "raqm_active_asset": active_asset,
                "raqm_active": active_asset is not None,
                "raqm_asset_changed": active_asset != previous_asset,
                "state_reason": reason,
                "held_days_at_open": (
                    held_before_decision
                    if reason
                    in {
                        "raqm_to_base_momentum_after_min_hold",
                        "raqm_to_defender_after_min_hold",
                        "raqm_rotation_after_exit",
                    }
                    else held_days
                ),
                "active_metric_at_open": (
                    metrics.at[timestamp, active_asset]
                    if active_asset is not None
                    else np.nan
                ),
                "defender_metric_at_open": defender_metric,
                "active_metric_difference_at_open": (
                    metrics.at[timestamp, active_asset] - defender_metric
                    if active_asset is not None
                    else np.nan
                ),
                "best_entry_candidate": best_entry,
                "best_entry_margin": (
                    margins[best_entry] if best_entry is not None else np.nan
                ),
                "target_candidate": target,
            }
        )
        if active_asset is not None:
            held_days += 1
    return pd.DataFrame(rows).set_index("date")


def validate_all_asset_raqm(
    context: GoldOverrideContext,
    spec: CommonRAQMSpec,
    policies: Mapping[str, AssetRAQMThresholds | None],
    state: pd.DataFrame,
    daily: pd.DataFrame,
) -> dict[str, object]:
    entries = state["state_reason"].eq("raqm_entry")
    rotations = state["state_reason"].eq("raqm_rotation_after_exit")
    exits = state["state_reason"].isin(
        [
            "raqm_to_base_momentum_after_min_hold",
            "raqm_to_defender_after_min_hold",
            "raqm_rotation_after_exit",
        ]
    )
    invalid_entries = 0
    for timestamp in state.index[entries | rotations]:
        asset = state.at[timestamp, "raqm_active_asset"]
        policy = policies[str(asset)]
        assert policy is not None
        difference = (
            state.at[timestamp, "active_metric_at_open"]
            - state.at[timestamp, "defender_metric_at_open"]
        )
        if (
            bool(state.at[timestamp, "base_c2_risk_on"])
            or not float(difference) > policy.entry_difference
        ):
            invalid_entries += 1
    early_exits = int(
        state.loc[exits, "held_days_at_open"].lt(HARD_MIN_HOLD_DAYS).sum()
    )
    active = state["raqm_active"].astype(bool)
    active_valid = bool(
        state.loc[active, "target_candidate"]
        .astype(str)
        .eq(state.loc[active, "raqm_active_asset"].astype(str))
        .all()
    )
    base_momentum = state["base_c2_risk_on"].astype(bool) & ~active
    momentum_matches = bool(
        state.loc[base_momentum, "target_candidate"]
        .astype(str)
        .eq(context.momentum_target.loc[base_momentum].astype(str))
        .all()
    )
    nav_error = float(
        ((1.0 + daily["return"]).cumprod() - daily["nav"]).abs().max()
    )
    if (
        invalid_entries
        or early_exits
        or not active_valid
        or not momentum_matches
        or nav_error > 1e-12
    ):
        raise AssertionError(
            "all-asset RAQM audit failed: "
            f"entries={invalid_entries}, early={early_exits}, "
            f"active={active_valid}, momentum={momentum_matches}, "
            f"nav={nav_error:.3e}"
        )
    asset_days = state.loc[active, "raqm_active_asset"].value_counts()
    entry_assets = state.loc[entries, "raqm_active_asset"].value_counts()
    return {
        "status": "passed",
        "candidate_id": policy_set_id(spec, policies),
        "invalid_entries": invalid_entries,
        "early_exits": early_exits,
        "active_targets_valid": active_valid,
        "base_momentum_targets_match": momentum_matches,
        "nav_reconstruction_max_abs_error": nav_error,
        "raqm_entries": int(entries.sum()),
        "raqm_rotations": int(rotations.sum()),
        "raqm_days": int(active.sum()),
        "raqm_asset_days": {
            str(asset): int(days) for asset, days in asset_days.items()
        },
        "raqm_entry_assets": {
            str(asset): int(count) for asset, count in entry_assets.items()
        },
        "switches": int(daily["switched"].sum()),
        "performance": performance(daily["return"]),
    }


def run_all_asset_raqm(
    context: GoldOverrideContext,
    spec: CommonRAQMSpec,
    policies: Mapping[str, AssetRAQMThresholds | None],
    *,
    metrics: pd.DataFrame | None = None,
) -> AllAssetRAQMBacktest:
    applied = common_raqm_at_open(context.curves, spec) if metrics is None else metrics
    state = all_asset_raqm_schedule(context, applied, policies)
    daily = simulate_candidate_schedule(
        state["target_candidate"],
        context.interfaces,
        context.initial_previous_candidate,
    )
    audit = validate_all_asset_raqm(context, spec, policies, state, daily)
    return AllAssetRAQMBacktest(spec, dict(policies), applied, state, daily, audit)
