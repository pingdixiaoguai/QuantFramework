"""Causal research harness for changing only the formal Defender ETF universe."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from data.store import read_local
from defender.relative_defender_rotation import DEFENSIVE_ASSET
from research.defender_curve_momentum import DEFENDER_CANDIDATE
from research.momentum_defender_downside_raqm import build_exact_execution_data
from research.momentum_defender_gold_override import (
    GoldOverrideContext,
    build_gold_override_context,
)
from research.momentum_defender_occam_defender import (
    MonthlySelectionSpec,
    build_portfolio_switch_interface,
    monthly_top1_selection,
    score_at_open,
    selected_asset_targets,
)
from research.momentum_defender_w40_asset_specific_escape import (
    AssetSpecificW40EscapeBacktest,
    run_asset_specific_w40_escape,
)
from research.momentum_defender_w40_loss_gate import run_w40_loss_gate
from research.momentum_defender_w40_top1_escape import quality_metrics_at_open
from strategy.momentum_defender_w40_full_equity import _features
from strategy.momentum_defender_w40_gold_escape import formal_policies
from strategy.momentum_defender_w40_loss import formal_spec


SELECTION_SPEC = MonthlySelectionSpec(40, "return", "lowest")


@dataclass(frozen=True)
class DividendUniverseBacktest:
    assets: tuple[str, ...]
    context: GoldOverrideContext
    selection: pd.DataFrame
    defender_interface: pd.DataFrame
    escape: AssetSpecificW40EscapeBacktest

    @property
    def returns(self) -> pd.Series:
        return self.escape.daily["return"].astype(float)


@dataclass(frozen=True)
class DividendUniverseHarness:
    base_context: GoldOverrideContext
    market: Mapping[str, pd.DataFrame]
    scores_at_open: pd.DataFrame
    gate_score_at_open: pd.Series


@dataclass(frozen=True)
class StandaloneDividendUniverseHarness:
    calendar: pd.DatetimeIndex
    market: Mapping[str, pd.DataFrame]
    scores_at_open: pd.DataFrame


@dataclass(frozen=True)
class StandaloneDividendUniverseBacktest:
    assets: tuple[str, ...]
    selection: pd.DataFrame
    targets: pd.DataFrame
    interface: pd.DataFrame

    @property
    def returns(self) -> pd.Series:
        from research.momentum_defender_occam import HELD_RETURN

        return self.interface[HELD_RETURN].astype(float)


def load_harness(
    root: Path,
    assets: Sequence[str],
    *,
    end: date,
) -> DividendUniverseHarness:
    """Load shared formal state once for a family of universe-only paths."""
    ordered = tuple(dict.fromkeys(str(asset) for asset in assets))
    if not ordered:
        raise ValueError("dividend universe research requires at least one ETF")
    market: dict[str, pd.DataFrame] = {}
    for asset in (*ordered, DEFENSIVE_ASSET):
        frame = read_local(asset)
        if frame is None or frame.empty:
            raise RuntimeError(f"missing local data for: {asset}")
        market[asset] = frame.loc[frame["date"].le(pd.Timestamp(end))].copy()
    base = build_gold_override_context(root, end=end)
    scores = score_at_open(market, ordered, base.calendar, SELECTION_SPEC)
    _, gate_score = _features(base.calendar, end=end)
    return DividendUniverseHarness(base, market, scores, gate_score)


def load_standalone_harness(
    assets: Sequence[str],
    *,
    start: date,
    end: date,
    market_overrides: Mapping[str, pd.DataFrame] | None = None,
) -> StandaloneDividendUniverseHarness:
    """Load a listing-aware Defender-only calendar without Momentum history."""
    ordered = tuple(dict.fromkeys(str(asset) for asset in assets))
    if not ordered:
        raise ValueError("standalone Defender research requires at least one ETF")
    market: dict[str, pd.DataFrame] = {}
    calendar_values: set[pd.Timestamp] = set()
    start_timestamp = pd.Timestamp(start)
    end_timestamp = pd.Timestamp(end)
    overrides = market_overrides or {}
    for asset in (*ordered, DEFENSIVE_ASSET):
        frame = overrides.get(asset)
        if frame is None:
            frame = read_local(asset)
        if frame is None or frame.empty:
            raise RuntimeError(f"missing local data for: {asset}")
        selected = frame.loc[
            frame["date"].ge(start_timestamp) & frame["date"].le(end_timestamp)
        ].copy()
        market[asset] = selected
        if asset != DEFENSIVE_ASSET:
            calendar_values.update(pd.to_datetime(selected["date"]))
    calendar = pd.DatetimeIndex(sorted(calendar_values))
    if calendar.empty:
        raise ValueError("standalone Defender calendar is empty")
    scores = score_at_open(market, ordered, calendar, SELECTION_SPEC)
    return StandaloneDividendUniverseHarness(calendar, market, scores)


def run_standalone_universe(
    harness: StandaloneDividendUniverseHarness,
    assets: Sequence[str],
    *,
    initial_asset: str = "510880.SH",
    defender_cost_multiplier: float = 1.0,
) -> StandaloneDividendUniverseBacktest:
    """Replay monthly reversal from the first ETF listing.

    Before any ETF owns 40 prior closes, the only listed initial ETF is held as
    the full-equity warmup fallback. Once scores exist, the unchanged formal
    monthly lowest-40-session-return rule takes over.
    """
    ordered = tuple(str(asset) for asset in assets)
    if not ordered or len(set(ordered)) != len(ordered):
        raise ValueError("assets must be non-empty and unique")
    if initial_asset not in ordered:
        raise ValueError("initial_asset must be part of the candidate universe")
    unknown = set(ordered) - set(harness.scores_at_open.columns)
    if unknown:
        raise ValueError(f"assets were not loaded by the harness: {sorted(unknown)}")
    if defender_cost_multiplier <= 0.0:
        raise ValueError("defender_cost_multiplier must be positive")

    market = {
        asset: harness.market[asset]
        for asset in (*ordered, DEFENSIVE_ASSET)
    }
    first_execution = harness.calendar[0]
    initial_open_dates = set(pd.to_datetime(market[initial_asset]["date"]))
    if first_execution not in initial_open_dates:
        raise RuntimeError("initial Defender ETF is not tradable on the first date")
    scores = harness.scores_at_open.loc[:, list(ordered)].copy()
    if scores.loc[first_execution].notna().any():
        raise AssertionError("standalone warmup expected no 40-session score")
    scores.at[first_execution, initial_asset] = 0.0
    selection = monthly_top1_selection(
        market,
        ordered,
        harness.calendar,
        scores,
        SELECTION_SPEC,
    )
    targets = selected_asset_targets(
        selection["selected_asset"].astype(str),
        ordered,
        selected_weight=1.0,
        residual_asset=DEFENSIVE_ASSET,
    )
    costs = {
        **{
            asset: 0.0001 * defender_cost_multiplier
            for asset in ordered
        },
        DEFENSIVE_ASSET: 0.00001 * defender_cost_multiplier,
    }
    interface = build_portfolio_switch_interface(market, targets, costs)
    return StandaloneDividendUniverseBacktest(
        assets=ordered,
        selection=selection,
        targets=targets,
        interface=interface,
    )


def run_universe(
    harness: DividendUniverseHarness,
    assets: Sequence[str],
    *,
    defender_cost_multiplier: float = 1.0,
) -> DividendUniverseBacktest:
    """Replay the formal composite while replacing only Defender candidates."""
    ordered = tuple(str(asset) for asset in assets)
    if not ordered or len(set(ordered)) != len(ordered):
        raise ValueError("assets must be non-empty and unique")
    unknown = set(ordered) - set(harness.scores_at_open.columns)
    if unknown:
        raise ValueError(f"assets were not loaded by the harness: {sorted(unknown)}")
    if defender_cost_multiplier <= 0.0:
        raise ValueError("defender_cost_multiplier must be positive")

    market = {
        asset: harness.market[asset]
        for asset in (*ordered, DEFENSIVE_ASSET)
    }
    scores = harness.scores_at_open.loc[:, list(ordered)]
    selection = monthly_top1_selection(
        market,
        ordered,
        harness.base_context.calendar,
        scores,
        SELECTION_SPEC,
    )
    targets = selected_asset_targets(
        selection["selected_asset"].astype(str),
        ordered,
        selected_weight=1.0,
        residual_asset=DEFENSIVE_ASSET,
    )
    costs = {
        **{
            asset: 0.0001 * defender_cost_multiplier
            for asset in ordered
        },
        DEFENSIVE_ASSET: 0.00001 * defender_cost_multiplier,
    }
    interface = build_portfolio_switch_interface(market, targets, costs)
    curves = harness.base_context.curves.copy()
    curves[DEFENDER_CANDIDATE] = interface["nav_if_held"].astype(float)
    context = replace(
        harness.base_context,
        curves=curves,
        interfaces={
            **harness.base_context.interfaces,
            DEFENDER_CANDIDATE: interface,
        },
    )
    gate = run_w40_loss_gate(
        build_exact_execution_data(context),
        harness.gate_score_at_open,
        formal_spec(),
    )
    escape = run_asset_specific_w40_escape(
        context,
        gate.state,
        formal_policies(),
        metrics=quality_metrics_at_open(context),
    )
    return DividendUniverseBacktest(
        assets=ordered,
        context=context,
        selection=selection,
        defender_interface=interface,
        escape=escape,
    )


def dedupe_pools(
    pools: Mapping[str, Sequence[str]],
) -> dict[str, tuple[str, ...]]:
    """Preserve the first label for every unique ordered candidate path."""
    seen: set[tuple[str, ...]] = set()
    result: dict[str, tuple[str, ...]] = {}
    for label, values in pools.items():
        pool = tuple(str(value) for value in values)
        if pool not in seen:
            result[str(label)] = pool
            seen.add(pool)
    return result


def difference_events(
    candidate: pd.Series,
    baseline: pd.Series,
    *,
    tolerance: float = 1e-15,
) -> pd.DataFrame:
    """Return contiguous daily blocks where the two executable paths differ."""
    aligned = pd.concat(
        [candidate.rename("candidate"), baseline.rename("baseline")], axis=1
    ).dropna()
    changed = aligned["candidate"].sub(aligned["baseline"]).abs().gt(tolerance)
    groups = changed.ne(changed.shift(fill_value=False)).cumsum()
    rows: list[dict[str, object]] = []
    for event_id, (_, sample) in enumerate(
        aligned.loc[changed].groupby(groups.loc[changed]), start=1
    ):
        candidate_log = float(np.log1p(sample["candidate"]).sum())
        baseline_log = float(np.log1p(sample["baseline"]).sum())
        rows.append(
            {
                "event_id": event_id,
                "start": sample.index.min(),
                "end": sample.index.max(),
                "observations": int(len(sample)),
                "candidate_return": float((1.0 + sample["candidate"]).prod() - 1.0),
                "baseline_return": float((1.0 + sample["baseline"]).prod() - 1.0),
                "log_excess": candidate_log - baseline_log,
            }
        )
    return pd.DataFrame(rows)
