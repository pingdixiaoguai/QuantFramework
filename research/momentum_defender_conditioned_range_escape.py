"""Conditioned partial Momentum escape from a high-range Defender sleeve.

This research overlay extends the simpler range-escape audit with two explicit
guards requested after its rejection:

* the Momentum tranche has a hard holding period; and
* entry requires the formal Momentum Top-1 quality-momentum score, its
  advantage over the continuous Defender curve, or both, to clear a floor.

The formal v5 state, Gold overlay, Momentum ranking, and Defender selector are
inputs and are never refit here.  All signals are previous-close values used at
the next open.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping
import warnings

import numpy as np
import pandas as pd

from defender.defender_opt_v2 import (
    _execute_portfolio_target,
    _indexed_market,
)
from research.defender_curve_momentum import DEFENDER_CANDIDATE
from research.momentum_defender_defender_range_escape import (
    ANCHOR_FIXED_512890,
    ANCHOR_MODES,
    FIXED_ANCHOR_ASSET,
    DefenderRangeEscapeContext,
)
from research.momentum_defender_occam import INTERNAL_COST, performance
from research.momentum_defender_occam_defender import (
    _market_prices_at,
    _normalized_target,
)


HOLD_DAILY = "daily"
HOLD_MINIMUM_UNTIL_FAIL = "minimum_until_fail"
HOLD_FIXED_PULSE_REARM = "fixed_pulse_rearm"
HOLD_POLICIES = (
    HOLD_DAILY,
    HOLD_MINIMUM_UNTIL_FAIL,
    HOLD_FIXED_PULSE_REARM,
)
QUALITY_AGGREGATIONS = ("single", "mean", "median", "minimum")


@dataclass(frozen=True)
class MomentumQualityGate:
    absolute_floor: float | None = None
    relative_floor: float | None = None

    def __post_init__(self) -> None:
        for name in ("absolute_floor", "relative_floor"):
            value = getattr(self, name)
            if value is not None and not np.isfinite(value):
                raise ValueError(f"{name} must be finite when configured")

    @property
    def family(self) -> str:
        if self.absolute_floor is None and self.relative_floor is None:
            return "none"
        if self.relative_floor is None:
            return "absolute"
        if self.absolute_floor is None:
            return "relative"
        return "joint"

    @property
    def gate_id(self) -> str:
        absolute = (
            "off" if self.absolute_floor is None else f"{self.absolute_floor:+.4f}"
        )
        relative = (
            "off" if self.relative_floor is None else f"{self.relative_floor:+.4f}"
        )
        return f"abs{absolute}_rel{relative}"

    def qualified(self, top1_metric: float, difference: float) -> bool:
        if self.absolute_floor is None and self.relative_floor is None:
            return True
        if self.absolute_floor is not None and not np.isfinite(top1_metric):
            return False
        if self.relative_floor is not None and not np.isfinite(difference):
            return False
        absolute_ok = (
            self.absolute_floor is None or top1_metric > self.absolute_floor
        )
        relative_ok = (
            self.relative_floor is None or difference > self.relative_floor
        )
        return bool(absolute_ok and relative_ok)


@dataclass(frozen=True)
class ConditionedRangeEscapeParams:
    anchor_mode: str
    range_window: int
    upper_threshold: float
    momentum_weight: float
    quality_window: int
    gate: MomentumQualityGate
    hold_policy: str
    hold_days: int
    quality_windows: tuple[int, ...] = ()
    quality_aggregation: str = "single"

    def __post_init__(self) -> None:
        if self.anchor_mode not in ANCHOR_MODES:
            raise ValueError(f"unsupported anchor mode: {self.anchor_mode}")
        if self.range_window < 2 or self.quality_window < 2:
            raise ValueError("range and quality windows must be at least two")
        if not 0.0 <= self.upper_threshold <= 1.0:
            raise ValueError("upper_threshold must lie in [0, 1]")
        if not 0.0 < self.momentum_weight < 1.0:
            raise ValueError("momentum_weight must lie in (0, 1)")
        if self.hold_policy not in HOLD_POLICIES:
            raise ValueError(f"unsupported hold policy: {self.hold_policy}")
        if self.hold_days < 1:
            raise ValueError("hold_days must be positive")
        if self.hold_policy == HOLD_DAILY and self.hold_days != 1:
            raise ValueError("daily policy requires hold_days=1")
        if self.quality_aggregation not in QUALITY_AGGREGATIONS:
            raise ValueError(
                f"unsupported quality aggregation: {self.quality_aggregation}"
            )
        if self.quality_aggregation == "single" and self.quality_windows:
            raise ValueError("single quality aggregation cannot configure windows")
        if self.quality_aggregation != "single":
            if len(self.quality_windows) < 2:
                raise ValueError("ensemble quality aggregation needs at least two windows")
            if any(window < 2 for window in self.quality_windows):
                raise ValueError("ensemble quality windows must be at least two")
            if tuple(sorted(set(self.quality_windows))) != self.quality_windows:
                raise ValueError("ensemble quality windows must be unique and ordered")

    @property
    def candidate_id(self) -> str:
        anchor = "a512890" if self.anchor_mode == ANCHOR_FIXED_512890 else "aselected"
        policy = {
            HOLD_DAILY: "daily",
            HOLD_MINIMUM_UNTIL_FAIL: "minhold",
            HOLD_FIXED_PULSE_REARM: "pulse",
        }[self.hold_policy]
        quality = (
            f"qw{self.quality_window}"
            if self.quality_aggregation == "single"
            else (
                f"q{self.quality_aggregation}"
                + "-".join(str(window) for window in self.quality_windows)
            )
        )
        return (
            f"conditioned_range_{anchor}_rw{self.range_window}_"
            f"hi{self.upper_threshold:.2f}_mw{self.momentum_weight:.2f}_"
            f"{quality}_{self.gate.gate_id}_{policy}{self.hold_days}"
        )


@dataclass(frozen=True)
class WeightExecutionCache:
    calendar: pd.DatetimeIndex
    open_prices: tuple[Mapping[str, float], ...]
    mark_open_prices: tuple[Mapping[str, float], ...]
    close_prices: tuple[Mapping[str, float], ...]


@dataclass(frozen=True)
class ConditionedRangeEscapeBacktest:
    params: ConditionedRangeEscapeParams
    state: pd.DataFrame
    targets: pd.DataFrame
    daily: pd.DataFrame
    audit: Mapping[str, object]


def aggregate_quality_metrics(
    panels: Mapping[int, pd.DataFrame],
    windows: tuple[int, ...],
    aggregation: str,
) -> pd.DataFrame:
    """Combine identical per-window QM panels without fitting weights."""
    if aggregation not in {"mean", "median", "minimum"}:
        raise ValueError("ensemble aggregation must be mean, median, or minimum")
    missing = set(windows) - set(panels)
    if missing:
        raise ValueError(f"missing quality metric panels: {sorted(missing)}")
    first = panels[windows[0]]
    if any(not first.index.equals(panels[window].index) for window in windows[1:]):
        raise ValueError("quality metric panels must share one calendar")
    columns = list(first.columns)
    values = np.stack(
        [panels[window][columns].to_numpy(float) for window in windows],
        axis=0,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        if aggregation == "mean":
            combined = np.nanmean(values, axis=0)
        elif aggregation == "median":
            combined = np.nanmedian(values, axis=0)
        else:
            combined = np.nanmin(values, axis=0)
    result = pd.DataFrame(combined, index=first.index, columns=columns)
    result.index.name = first.index.name
    return result


def build_weight_execution_cache(
    context: DefenderRangeEscapeContext,
) -> WeightExecutionCache:
    calendar = pd.DatetimeIndex([context.previous_date]).append(context.calendar)
    indexed = _indexed_market(context.market)
    opens: list[Mapping[str, float]] = []
    marks: list[Mapping[str, float]] = []
    closes: list[Mapping[str, float]] = []
    for timestamp in calendar:
        open_prices, mark_open, close_prices = _market_prices_at(indexed, timestamp)
        opens.append(open_prices)
        marks.append(mark_open)
        closes.append(close_prices)
    return WeightExecutionCache(
        calendar=calendar,
        open_prices=tuple(opens),
        mark_open_prices=tuple(marks),
        close_prices=tuple(closes),
    )


def execute_targets_cached(
    context: DefenderRangeEscapeContext,
    targets: pd.DataFrame,
    cache: WeightExecutionCache,
    *,
    cost_multiplier: float = 1.0,
) -> pd.DataFrame:
    """Execute a target schedule exactly without rebuilding price interfaces."""
    if cost_multiplier < 0.0:
        raise ValueError("cost_multiplier must be non-negative")
    if not context.calendar.equals(targets.index):
        raise ValueError("targets must use the formal evaluation calendar")
    if not cache.calendar[1:].equals(context.calendar):
        raise ValueError("execution cache does not match the formal calendar")
    initial = pd.DataFrame(0.0, index=cache.calendar[:1], columns=context.assets)
    initial.at[
        context.previous_date,
        context.formal.context.initial_previous_candidate,
    ] = 1.0
    full_targets = pd.concat([initial, targets])
    costs = {
        asset: float(rate) * cost_multiplier
        for asset, rate in context.cost_rates.items()
    }

    cash = 1.0
    shares: dict[str, float] = {}
    previous_target: dict[str, float] = {}
    previous_nav = 1.0
    rows: list[dict[str, object]] = []
    for position, timestamp in enumerate(cache.calendar):
        policy_target = _normalized_target(full_targets.loc[timestamp])
        executions: list[dict[str, float | str]] = []
        if policy_target != previous_target:
            cash, shares, executions = _execute_portfolio_target(
                cash,
                shares,
                policy_target,
                cache.open_prices[position],
                cache.mark_open_prices[position],
                costs,
            )
            previous_target = policy_target
        nav = cash + sum(
            quantity * cache.close_prices[position].get(asset, 0.0)
            for asset, quantity in shares.items()
        )
        daily_return = nav / previous_nav - 1.0
        internal_cost = sum(float(item["cost"]) for item in executions) / previous_nav
        rows.append(
            {
                "date": timestamp,
                "return": daily_return,
                "nav": nav,
                INTERNAL_COST: internal_cost,
                "internal_rebalanced": bool(executions),
            }
        )
        previous_nav = nav
    sample = pd.DataFrame(rows).set_index("date").loc[context.calendar].copy()
    sample["portfolio_nav"] = sample["nav"].astype(float)
    sample["nav"] = (1.0 + sample["return"].astype(float)).cumprod()
    return sample


def conditioned_range_escape_state(
    formal_candidates: pd.Series,
    momentum_targets: pd.Series,
    selected_defender_assets: pd.Series,
    locations_at_open: pd.DataFrame,
    quality_metrics_at_open: pd.DataFrame,
    params: ConditionedRangeEscapeParams,
) -> pd.DataFrame:
    """Build the causal conditioned escape state at every formal open."""
    calendar = pd.DatetimeIndex(formal_candidates.index)
    if not all(
        calendar.equals(value.index)
        for value in (
            momentum_targets,
            selected_defender_assets,
            locations_at_open,
            quality_metrics_at_open,
        )
    ):
        raise ValueError("conditioned escape inputs must share one calendar")

    active = False
    active_asset: str | None = None
    held_days = 0
    armed = True
    rows: list[dict[str, object]] = []
    for timestamp in calendar:
        formal_candidate = str(formal_candidates.loc[timestamp])
        formal_defender = formal_candidate == DEFENDER_CANDIDATE
        top1 = str(momentum_targets.loc[timestamp])
        selected_defender = str(selected_defender_assets.loc[timestamp])
        anchor_asset = (
            FIXED_ANCHOR_ASSET
            if params.anchor_mode == ANCHOR_FIXED_512890
            else selected_defender
        )
        raw_location = locations_at_open.at[timestamp, anchor_asset]
        location = float(raw_location) if pd.notna(raw_location) else np.nan
        raw_top1 = quality_metrics_at_open.at[timestamp, top1]
        raw_defender = quality_metrics_at_open.at[timestamp, DEFENDER_CANDIDATE]
        top1_metric = float(raw_top1) if pd.notna(raw_top1) else np.nan
        defender_metric = float(raw_defender) if pd.notna(raw_defender) else np.nan
        difference = top1_metric - defender_metric
        range_high = bool(
            np.isfinite(location) and location >= params.upper_threshold
        )
        gate_qualified = params.gate.qualified(top1_metric, difference)
        qualified = bool(formal_defender and range_high and gate_qualified)
        previous_active = active
        previous_asset = active_asset
        reason = "hold"

        if params.hold_policy == HOLD_DAILY:
            active = qualified
            active_asset = top1 if active else None
            held_days = 0
            reason = "daily_qualified" if active else "daily_inactive"
        elif active:
            if not formal_defender:
                active = False
                active_asset = None
                held_days = 0
                armed = True
                reason = "formal_left_defender"
            elif held_days < params.hold_days:
                reason = "momentum_hard_hold"
            elif params.hold_policy == HOLD_FIXED_PULSE_REARM:
                active = False
                active_asset = None
                held_days = 0
                armed = False
                reason = "fixed_pulse_complete"
            elif not qualified:
                active = False
                active_asset = None
                held_days = 0
                reason = "condition_failed_after_hold"
            elif top1 != active_asset:
                active_asset = top1
                held_days = 0
                reason = "qualified_top1_rotation"
            else:
                reason = "qualified_momentum_hold"
        else:
            if params.hold_policy == HOLD_FIXED_PULSE_REARM and not qualified:
                armed = True
            can_enter = qualified and (
                params.hold_policy != HOLD_FIXED_PULSE_REARM or armed
            )
            if can_enter:
                active = True
                active_asset = top1
                held_days = 0
                reason = "conditioned_escape_entry"
            else:
                reason = "conditioned_escape_wait"

        if active:
            assert active_asset is not None
            momentum_weight = params.momentum_weight
            defender_weight = 1.0 - momentum_weight
        else:
            momentum_weight = 0.0
            defender_weight = 1.0 if formal_defender else 0.0
        rows.append(
            {
                "date": timestamp,
                "formal_candidate": formal_candidate,
                "formal_defender": formal_defender,
                "selected_defender_asset": selected_defender,
                "range_anchor_asset": anchor_asset,
                "range_location_at_open": location,
                "range_high": range_high,
                "momentum_top1": top1,
                "top1_quality_momentum_at_open": top1_metric,
                "defender_quality_momentum_at_open": defender_metric,
                "quality_momentum_difference_at_open": difference,
                "gate_qualified": gate_qualified,
                "entry_qualified": qualified,
                "escape_active": active,
                "escape_changed": active != previous_active,
                "escape_asset_changed": active_asset != previous_asset,
                "escape_asset": active_asset,
                "escape_held_days_before_decision": held_days,
                "pulse_armed": armed,
                "defender_weight": defender_weight,
                "momentum_weight": momentum_weight,
                "state_reason": reason,
            }
        )
        if active:
            held_days += 1
    return pd.DataFrame(rows).set_index("date")


def conditioned_target_schedule(
    context: DefenderRangeEscapeContext,
    state: pd.DataFrame,
) -> pd.DataFrame:
    calendar = context.calendar
    targets = pd.DataFrame(0.0, index=calendar, columns=context.assets)
    formal_candidates = context.formal.daily["candidate"].astype(str)
    defender_targets = context.formal.base.defender.targets
    for timestamp in calendar:
        formal_candidate = str(formal_candidates.loc[timestamp])
        if formal_candidate != DEFENDER_CANDIDATE:
            targets.at[timestamp, formal_candidate] = 1.0
            continue
        momentum_weight = float(state.at[timestamp, "momentum_weight"])
        defender_weight = 1.0 - momentum_weight
        targets.loc[timestamp, defender_targets.columns] += (
            defender_targets.loc[timestamp].to_numpy(float) * defender_weight
        )
        if momentum_weight > 0.0:
            asset = str(state.at[timestamp, "escape_asset"])
            targets.at[timestamp, asset] += momentum_weight
    error = float((targets.sum(axis=1) - 1.0).abs().max())
    if error > 1e-12:
        raise AssertionError(f"conditioned targets do not sum to one: {error:.3e}")
    targets.index.name = "date"
    return targets


def run_conditioned_range_escape(
    context: DefenderRangeEscapeContext,
    params: ConditionedRangeEscapeParams,
    *,
    locations_at_open: pd.DataFrame,
    quality_metrics_at_open: pd.DataFrame,
    execution_cache: WeightExecutionCache,
    cost_multiplier: float = 1.0,
) -> ConditionedRangeEscapeBacktest:
    state = conditioned_range_escape_state(
        context.formal.daily["candidate"].astype(str),
        context.formal.context.momentum_target.astype(str),
        context.formal.base.defender.selection["selected_asset"].astype(str),
        locations_at_open,
        quality_metrics_at_open,
        params,
    )
    targets = conditioned_target_schedule(context, state)
    daily = execute_targets_cached(
        context,
        targets,
        execution_cache,
        cost_multiplier=cost_multiplier,
    )
    returns = daily["return"].astype(float)
    entries = (
        state["escape_changed"].astype(bool)
        & state["escape_active"].astype(bool)
    )
    active = state["escape_active"].astype(bool)
    audit = {
        "status": "passed",
        "candidate_id": params.candidate_id,
        "params": {
            **asdict(params),
            "gate_family": params.gate.family,
            "gate_id": params.gate.gate_id,
        },
        "performance": performance(returns),
        "escape_entries": int(entries.sum()),
        "escape_days": int(active.sum()),
        "escape_asset_rotations": int(
            (active & state["escape_asset_changed"].astype(bool) & ~entries).sum()
        ),
        "average_momentum_weight_on_formal_defender_days": float(
            state.loc[state["formal_defender"].astype(bool), "momentum_weight"].mean()
        ),
        "nav_reconstruction_max_abs_error": float(
            ((1.0 + returns).cumprod() - daily["nav"]).abs().max()
        ),
        "cost_multiplier": float(cost_multiplier),
    }
    if audit["nav_reconstruction_max_abs_error"] > 1e-12:
        raise AssertionError("conditioned range escape NAV reconstruction failed")
    return ConditionedRangeEscapeBacktest(
        params=params,
        state=state,
        targets=targets,
        daily=daily,
        audit=audit,
    )
