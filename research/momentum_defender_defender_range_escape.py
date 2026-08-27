"""Research-only Defender high-range partial escape into Momentum.

The overlay keeps the formal v5 state machine, Gold escape, Momentum Top-1,
and monthly Defender selector frozen.  It changes weights only on opens where
the formal executable candidate is Defender:

* the Defender weight is controlled by a causal range-position rule; and
* the residual is assigned to the already-known formal Momentum Top-1.

Signals use closes strictly before the execution open.  The supplied 40-day
10%/95%/20% state machine is represented by ``continuous_grid``.  Two simpler
diagnostics, ``defender_episode_grid`` and ``binary_partial``, deliberately
remove long-lived state rather than add new predictors.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from defender.relative_defender_rotation import DEFENSIVE_ASSET
from defender.w40_reversal_full_equity import (
    FORMAL_COST_RATES,
    FORMAL_DIVIDEND_ASSETS,
    _load_formal_market,
)
from research.defender_curve_momentum import DEFENDER_CANDIDATE
from research.momentum_defender_occam import (
    HELD_RETURN,
    INTERNAL_COST,
    MOMENTUM_ASSETS,
    _load_prices,
    performance,
)
from research.momentum_defender_occam_defender import (
    build_portfolio_switch_interface,
)
from research.momentum_volatility import asof_previous_close
from strategy.momentum_defender_w40_full_equity import FORMAL_BACKTEST_START
from strategy.momentum_defender_w40_qm40_threshold import run_formal_strategy


ANCHOR_FIXED_512890 = "fixed_512890"
ANCHOR_SELECTED_DEFENDER = "selected_defender"
ANCHOR_MODES = (ANCHOR_FIXED_512890, ANCHOR_SELECTED_DEFENDER)

POLICY_CONTINUOUS_GRID = "continuous_grid"
POLICY_DEFENDER_EPISODE_GRID = "defender_episode_grid"
POLICY_BINARY_PARTIAL = "binary_partial"
POLICIES = (
    POLICY_CONTINUOUS_GRID,
    POLICY_DEFENDER_EPISODE_GRID,
    POLICY_BINARY_PARTIAL,
)

FIXED_ANCHOR_ASSET = "512890.SH"


@dataclass(frozen=True)
class DefenderRangeEscapeParams:
    anchor_mode: str = ANCHOR_FIXED_512890
    policy: str = POLICY_CONTINUOUS_GRID
    range_window: int = 40
    lower_threshold: float = 0.10
    upper_threshold: float = 0.95
    position_step: float = 0.20

    def __post_init__(self) -> None:
        if self.anchor_mode not in ANCHOR_MODES:
            raise ValueError(f"unsupported anchor mode: {self.anchor_mode}")
        if self.policy not in POLICIES:
            raise ValueError(f"unsupported policy: {self.policy}")
        if self.range_window < 2:
            raise ValueError("range_window must be at least two")
        if not 0.0 <= self.lower_threshold < self.upper_threshold <= 1.0:
            raise ValueError("range thresholds must satisfy 0 <= lower < upper <= 1")
        if not 0.0 < self.position_step < 1.0:
            raise ValueError("position_step must lie in (0, 1)")
        if self.policy != POLICY_BINARY_PARTIAL:
            levels = round(1.0 / self.position_step)
            if not np.isclose(
                levels * self.position_step, 1.0, atol=1e-12
            ):
                raise ValueError("grid position_step must divide one exactly")

    def candidate_id(self) -> str:
        anchor = "a512890" if self.anchor_mode == ANCHOR_FIXED_512890 else "aselected"
        policy = {
            POLICY_CONTINUOUS_GRID: "continuous",
            POLICY_DEFENDER_EPISODE_GRID: "episode",
            POLICY_BINARY_PARTIAL: "binary",
        }[self.policy]
        return (
            f"range_escape_{anchor}_{policy}_w{self.range_window}_"
            f"lo{self.lower_threshold:.2f}_hi{self.upper_threshold:.2f}_"
            f"step{self.position_step:.2f}"
        )


@dataclass(frozen=True)
class DefenderRangeEscapeContext:
    formal: object
    calendar: pd.DatetimeIndex
    previous_date: pd.Timestamp
    market: Mapping[str, pd.DataFrame]
    assets: tuple[str, ...]
    cost_rates: Mapping[str, float]
    baseline_targets: pd.DataFrame
    direct_baseline_daily: pd.DataFrame
    baseline_parity_max_abs_error: float


@dataclass(frozen=True)
class DefenderRangeEscapeBacktest:
    params: DefenderRangeEscapeParams
    state: pd.DataFrame
    targets: pd.DataFrame
    daily: pd.DataFrame
    audit: Mapping[str, object]


def _normalise_market_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if "date" not in result.columns:
        result = result.rename_axis("date").reset_index()
    result["date"] = pd.to_datetime(result["date"])
    return result.sort_values("date").drop_duplicates("date").reset_index(drop=True)


def _all_assets() -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            (*MOMENTUM_ASSETS, *FORMAL_DIVIDEND_ASSETS, DEFENSIVE_ASSET)
        )
    )


def _target_schedule(
    formal_candidates: pd.Series,
    momentum_targets: pd.Series,
    defender_targets: pd.DataFrame,
    defender_weights: pd.Series,
    assets: tuple[str, ...],
) -> pd.DataFrame:
    calendar = pd.DatetimeIndex(formal_candidates.index)
    if not (
        calendar.equals(momentum_targets.index)
        and calendar.equals(defender_targets.index)
        and calendar.equals(defender_weights.index)
    ):
        raise ValueError("all target inputs must share one calendar")
    targets = pd.DataFrame(0.0, index=calendar, columns=assets)
    for timestamp in calendar:
        formal_candidate = str(formal_candidates.loc[timestamp])
        if formal_candidate != DEFENDER_CANDIDATE:
            if formal_candidate not in targets.columns:
                raise ValueError(f"unknown formal candidate: {formal_candidate}")
            targets.at[timestamp, formal_candidate] = 1.0
            continue
        defender_weight = float(defender_weights.loc[timestamp])
        momentum_weight = 1.0 - defender_weight
        targets.loc[timestamp, defender_targets.columns] += (
            defender_targets.loc[timestamp].to_numpy(float) * defender_weight
        )
        momentum_asset = str(momentum_targets.loc[timestamp])
        targets.at[timestamp, momentum_asset] += momentum_weight
    error = float((targets.sum(axis=1) - 1.0).abs().max())
    if error > 1e-12 or targets.lt(-1e-14).any().any():
        raise AssertionError(f"range escape targets are invalid: {error:.3e}")
    targets.index.name = "date"
    return targets


def _with_initial_previous_target(
    context: DefenderRangeEscapeContext,
    targets: pd.DataFrame,
) -> pd.DataFrame:
    initial = pd.DataFrame(
        0.0,
        index=pd.DatetimeIndex([context.previous_date]),
        columns=context.assets,
    )
    initial.at[
        context.previous_date,
        context.formal.context.initial_previous_candidate,
    ] = 1.0
    return pd.concat([initial, targets])


def _execute_targets(
    context: DefenderRangeEscapeContext,
    targets: pd.DataFrame,
    *,
    cost_multiplier: float = 1.0,
) -> pd.DataFrame:
    if cost_multiplier < 0.0:
        raise ValueError("cost_multiplier must be non-negative")
    costs = {
        asset: float(rate) * cost_multiplier
        for asset, rate in context.cost_rates.items()
    }
    interface = build_portfolio_switch_interface(
        context.market,
        _with_initial_previous_target(context, targets),
        costs,
    )
    sample = interface.loc[context.calendar].copy()
    sample["return"] = sample[HELD_RETURN].astype(float)
    sample["nav"] = (1.0 + sample["return"]).cumprod()
    return sample


def build_range_escape_context(
    root: Path,
    *,
    start: date,
    end: date,
) -> DefenderRangeEscapeContext:
    """Build the frozen formal v5 baseline and an exact weight-level replay."""
    formal = run_formal_strategy(root, start=start, end=end)
    calendar = formal.context.calendar
    raw_market = {
        **_load_formal_market(end),
        **_load_prices(MOMENTUM_ASSETS, FORMAL_BACKTEST_START, end),
    }
    market = {
        asset: _normalise_market_frame(frame)
        for asset, frame in raw_market.items()
    }
    assets = _all_assets()
    anchor_calendar = pd.DatetimeIndex(market["510300.SH"]["date"])
    prior = anchor_calendar[anchor_calendar < calendar.min()]
    if prior.empty:
        raise RuntimeError("missing prior trading day for exact initial holdings")
    previous_date = pd.Timestamp(prior.max())
    formal_candidates = formal.daily["candidate"].astype(str)
    baseline_weights = pd.Series(1.0, index=calendar)
    baseline_targets = _target_schedule(
        formal_candidates,
        formal.context.momentum_target.astype(str),
        formal.base.defender.targets,
        baseline_weights,
        assets,
    )
    cost_rates = {
        asset: float(FORMAL_COST_RATES.get(asset, 0.0001))
        for asset in assets
    }
    provisional = DefenderRangeEscapeContext(
        formal=formal,
        calendar=calendar,
        previous_date=previous_date,
        market=market,
        assets=assets,
        cost_rates=cost_rates,
        baseline_targets=baseline_targets,
        direct_baseline_daily=pd.DataFrame(),
        baseline_parity_max_abs_error=np.nan,
    )
    direct = _execute_targets(provisional, baseline_targets)
    parity = float(
        direct["return"].sub(formal.daily["return"].astype(float)).abs().max()
    )
    if parity > 2e-8:
        raise AssertionError(
            f"weight-level formal baseline parity failed: {parity:.3e}"
        )
    return DefenderRangeEscapeContext(
        formal=formal,
        calendar=calendar,
        previous_date=previous_date,
        market=market,
        assets=assets,
        cost_rates=cost_rates,
        baseline_targets=baseline_targets,
        direct_baseline_daily=direct,
        baseline_parity_max_abs_error=parity,
    )


def range_locations_at_open(
    context: DefenderRangeEscapeContext,
    window: int,
) -> pd.DataFrame:
    """Return close-known range locations for every Defender equity ETF."""
    if window < 2:
        raise ValueError("range window must be at least two")
    result: dict[str, pd.Series] = {}
    for asset in FORMAL_DIVIDEND_ASSETS:
        frame = context.market[asset].set_index("date")
        close = frame["close"].astype(float)
        rolling_low = close.rolling(window, min_periods=window).min()
        rolling_high = close.rolling(window, min_periods=window).max()
        width = rolling_high - rolling_low
        location = (close - rolling_low) / width.replace(0.0, np.nan)
        location = location.where(width.ne(0.0), 0.5)
        result[asset] = asof_previous_close(location, context.calendar)
    locations = pd.DataFrame(result, index=context.calendar)
    locations.index.name = "date"
    return locations


def range_escape_state(
    formal_candidates: pd.Series,
    momentum_targets: pd.Series,
    selected_defender_assets: pd.Series,
    locations_at_open: pd.DataFrame,
    params: DefenderRangeEscapeParams,
) -> pd.DataFrame:
    """Advance the range-position state and expose executable open weights."""
    calendar = pd.DatetimeIndex(formal_candidates.index)
    if not (
        calendar.equals(momentum_targets.index)
        and calendar.equals(selected_defender_assets.index)
        and calendar.equals(locations_at_open.index)
    ):
        raise ValueError("range escape state inputs must share one calendar")
    level_count = int(round(1.0 / params.position_step))
    level = level_count
    rows: list[dict[str, object]] = []
    for position, timestamp in enumerate(calendar):
        formal_candidate = str(formal_candidates.loc[timestamp])
        formal_defender = formal_candidate == DEFENDER_CANDIDATE
        selected_asset = str(selected_defender_assets.loc[timestamp])
        anchor_asset = (
            FIXED_ANCHOR_ASSET
            if params.anchor_mode == ANCHOR_FIXED_512890
            else selected_asset
        )
        raw_location = locations_at_open.at[timestamp, anchor_asset]
        location = float(raw_location) if pd.notna(raw_location) else np.nan
        reason = "hold"

        # A 2019 restart begins fully allocated and does not import a grid
        # level from the unobserved pre-sample path.
        if position == 0:
            level = level_count
            reason = "initial_full"
        elif params.policy == POLICY_BINARY_PARTIAL:
            level = (
                level_count - 1
                if formal_defender
                and np.isfinite(location)
                and location >= params.upper_threshold
                else level_count
            )
            reason = "high_partial" if level < level_count else "binary_full"
        else:
            if params.policy == POLICY_DEFENDER_EPISODE_GRID and not formal_defender:
                if level != level_count:
                    reason = "outside_defender_reset"
                level = level_count
            elif np.isfinite(location) and location <= params.lower_threshold:
                level = min(level_count, level + 1)
                reason = "relative_low_add"
            elif np.isfinite(location) and location >= params.upper_threshold:
                level = max(0, level - 1)
                reason = "relative_high_reduce"

        raw_defender_weight = level / level_count
        defender_weight = raw_defender_weight if formal_defender else 0.0
        momentum_weight = 1.0 - raw_defender_weight if formal_defender else 0.0
        rows.append(
            {
                "date": timestamp,
                "formal_candidate": formal_candidate,
                "formal_defender": formal_defender,
                "momentum_target": str(momentum_targets.loc[timestamp]),
                "selected_defender_asset": selected_asset,
                "range_anchor_asset": anchor_asset,
                "range_location_at_open": location,
                "grid_level": level,
                "raw_defender_weight": raw_defender_weight,
                "defender_weight": defender_weight,
                "momentum_weight": momentum_weight,
                "overlay_active": formal_defender and raw_defender_weight < 1.0,
                "state_reason": reason,
            }
        )
    return pd.DataFrame(rows).set_index("date")


def run_range_escape(
    context: DefenderRangeEscapeContext,
    params: DefenderRangeEscapeParams,
    *,
    locations_at_open: pd.DataFrame | None = None,
    cost_multiplier: float = 1.0,
) -> DefenderRangeEscapeBacktest:
    locations = (
        range_locations_at_open(context, params.range_window)
        if locations_at_open is None
        else locations_at_open
    )
    formal_candidates = context.formal.daily["candidate"].astype(str)
    momentum_targets = context.formal.context.momentum_target.astype(str)
    selected_assets = context.formal.base.defender.selection[
        "selected_asset"
    ].astype(str)
    state = range_escape_state(
        formal_candidates,
        momentum_targets,
        selected_assets,
        locations,
        params,
    )
    targets = _target_schedule(
        formal_candidates,
        momentum_targets,
        context.formal.base.defender.targets,
        state["raw_defender_weight"].astype(float),
        context.assets,
    )
    daily = _execute_targets(
        context,
        targets,
        cost_multiplier=cost_multiplier,
    )
    returns = daily["return"].astype(float)
    active = state["overlay_active"].astype(bool)
    formal_defender = state["formal_defender"].astype(bool)
    audit = {
        "status": "passed",
        "candidate_id": params.candidate_id(),
        "params": asdict(params),
        "performance": performance(returns),
        "overlay_days": int(active.sum()),
        "formal_defender_days": int(formal_defender.sum()),
        "average_defender_weight_on_formal_defender_days": float(
            state.loc[formal_defender, "raw_defender_weight"].mean()
        ),
        "zero_defender_days": int(
            (formal_defender & state["raw_defender_weight"].eq(0.0)).sum()
        ),
        "high_reduce_observations": int(
            state["state_reason"].isin(
                ["relative_high_reduce", "high_partial"]
            ).sum()
        ),
        "low_add_observations": int(
            state["state_reason"].eq("relative_low_add").sum()
        ),
        "target_sum_max_abs_error": float(
            (targets.sum(axis=1) - 1.0).abs().max()
        ),
        "nav_reconstruction_max_abs_error": float(
            ((1.0 + returns).cumprod() - daily["nav"]).abs().max()
        ),
        "cost_multiplier": float(cost_multiplier),
        "mean_internal_cost_rate": float(daily[INTERNAL_COST].mean()),
    }
    if (
        audit["target_sum_max_abs_error"] > 1e-12
        or audit["nav_reconstruction_max_abs_error"] > 1e-12
    ):
        raise AssertionError("range escape execution audit failed")
    return DefenderRangeEscapeBacktest(
        params=params,
        state=state,
        targets=targets,
        daily=daily,
        audit=audit,
    )


def execute_formal_targets_at_cost(
    context: DefenderRangeEscapeContext,
    cost_multiplier: float,
) -> pd.Series:
    """Return the direct formal schedule under a scaled transaction-cost rate."""
    return _execute_targets(
        context,
        context.baseline_targets,
        cost_multiplier=cost_multiplier,
    )["return"].astype(float)
