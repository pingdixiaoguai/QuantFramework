"""RAQM-W5 bridge from locked Defender back to the Momentum Top-1 sleeve.

The production Gold override remains authoritative.  This research overlay is
only allowed to replace a formal ``DEFENDER`` target when all of the following
are already known at the previous close:

* the C2 slow gate wants Momentum but its state machine is still in Defender;
* the emergency volatility gate is not active;
* the current Momentum Top-1 has an extreme registered
  ``risk_adjusted_quality_momentum(window=5)`` reading for the configured
  number of consecutive observations.

Once opened, the bridge follows the ordinary Momentum Top-1 schedule.  It does
not freeze the entry ETF and it never overrides either an emergency signal or
the formal Gold override.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from factors.risk_adjusted_quality_momentum import compute
from research.defender_curve_momentum import DEFENDER_CANDIDATE
from research.gold_min5_risk_adjusted_momentum_w5 import (
    GoldRAQMW5Params,
    run_gold_raqm_w5,
)
from research.momentum_defender_gold_override import (
    GoldOverrideContext,
    simulate_candidate_schedule,
)
from research.momentum_defender_occam import MOMENTUM_ASSETS, performance


WINDOW = 5
VOL_FLOOR_ANNUAL = 0.08
FORMAL_GOLD_ENTRY_DIFFERENCE = 2.20
FORMAL_GOLD_EXIT_DIFFERENCE = 0.60


@dataclass(frozen=True)
class Top1RAQMW5BridgeParams:
    """Parameters for the extreme-trend bridge.

    ``minimum_difference=None`` disables the relative-to-Defender condition.
    This is important because the registered factor is capped: in a broad
    rally both Top-1 and Defender can equal +3 even though Top-1 is genuinely
    extreme.
    """

    entry_minimum: float = 2.20
    confirmation_days: int = 1
    minimum_difference: float | None = None

    def __post_init__(self) -> None:
        if not -3.0 <= self.entry_minimum <= 3.0:
            raise ValueError("entry_minimum must be inside the factor's [-3, 3] range")
        if self.confirmation_days < 1:
            raise ValueError("confirmation_days must be positive")

    def candidate_id(self) -> str:
        difference = (
            "off"
            if self.minimum_difference is None
            else f"{self.minimum_difference:+.2f}"
        )
        return (
            f"top1_raqm_w5_abs{self.entry_minimum:+.2f}_"
            f"confirm{self.confirmation_days}_diff{difference}_"
            "slow_true_emergency_blocked"
        )


@dataclass(frozen=True)
class Top1RAQMW5BridgeBacktest:
    params: Top1RAQMW5BridgeParams
    metrics_at_open: pd.DataFrame
    state: pd.DataFrame
    daily: pd.DataFrame
    audit: dict[str, object]


def registered_raqm_w5_at_open(curves: pd.DataFrame) -> pd.DataFrame:
    """Apply the exact registered RAQM factor and lag it to the next open."""

    values: dict[str, pd.Series] = {}
    for candidate in (*MOMENTUM_ASSETS, DEFENDER_CANDIDATE):
        frame = pd.DataFrame(
            {
                "date": curves.index,
                "close": curves[candidate].to_numpy(float),
            }
        )
        values[candidate] = compute(
            frame,
            {"window": WINDOW, "vol_floor_annual": VOL_FLOOR_ANNUAL},
        ).reindex(curves.index).shift(1)
    result = pd.DataFrame(values, index=curves.index)
    result.index.name = "date"
    return result


def _entry_qualification(
    context: GoldOverrideContext,
    formal_target: pd.Series,
    metrics: pd.DataFrame,
    params: Top1RAQMW5BridgeParams,
) -> pd.DataFrame:
    base_state = context.integrated.result.state
    slow = base_state["slow_signal_asof_previous_close"].fillna(False).astype(bool)
    emergency = base_state["emergency_asof_previous_close"].fillna(False).astype(bool)
    top1 = context.momentum_target.astype(str)
    top1_metric = pd.Series(
        [metrics.at[timestamp, top1.loc[timestamp]] for timestamp in context.calendar],
        index=context.calendar,
        dtype=float,
        name="top1_metric_at_open",
    )
    defender_metric = metrics[DEFENDER_CANDIDATE].astype(float).rename(
        "defender_metric_at_open"
    )
    difference = (top1_metric - defender_metric).rename(
        "metric_difference_at_open"
    )
    qualified = (
        formal_target.eq(DEFENDER_CANDIDATE)
        & slow
        & ~emergency
        & top1_metric.gt(params.entry_minimum)
    )
    if params.minimum_difference is not None:
        qualified &= difference.gt(params.minimum_difference)

    confirmed = pd.Series(False, index=context.calendar, name="entry_confirmed")
    confirmation_days = params.confirmation_days
    for position, timestamp in enumerate(context.calendar):
        if position + 1 < confirmation_days:
            continue
        window = context.calendar[position - confirmation_days + 1 : position + 1]
        same_top1 = top1.loc[window].eq(top1.loc[timestamp]).all()
        confirmed.loc[timestamp] = bool(same_top1 and qualified.loc[window].all())
    return pd.DataFrame(
        {
            "momentum_top1": top1,
            "top1_metric_at_open": top1_metric,
            "defender_metric_at_open": defender_metric,
            "metric_difference_at_open": difference,
            "slow_gate_risk_on": slow,
            "emergency_active": emergency,
            "entry_qualified": qualified,
            "entry_confirmed": confirmed,
        },
        index=context.calendar,
    )


def top1_raqm_w5_bridge_schedule(
    context: GoldOverrideContext,
    formal_target: pd.Series,
    metrics: pd.DataFrame,
    params: Top1RAQMW5BridgeParams,
) -> pd.DataFrame:
    """Overlay the bridge without modifying formal Gold or emergency targets."""

    signal = _entry_qualification(context, formal_target, metrics, params)
    active = False
    rows: list[dict[str, object]] = []
    for timestamp in context.calendar:
        previous = active
        reason = "hold"
        formal_candidate = str(formal_target.loc[timestamp])
        if formal_candidate != DEFENDER_CANDIDATE:
            active = False
            if previous:
                reason = "formal_strategy_priority"
        elif active and (
            not bool(signal.at[timestamp, "slow_gate_risk_on"])
            or bool(signal.at[timestamp, "emergency_active"])
        ):
            active = False
            reason = "bridge_safety_exit"
        elif not active and bool(signal.at[timestamp, "entry_confirmed"]):
            active = True
            reason = "top1_bridge_entry"

        target = (
            str(signal.at[timestamp, "momentum_top1"])
            if active
            else formal_candidate
        )
        rows.append(
            {
                "date": timestamp,
                "formal_target_candidate": formal_candidate,
                "momentum_top1": signal.at[timestamp, "momentum_top1"],
                "top1_bridge_active": active,
                "top1_bridge_changed": active != previous,
                "state_reason": reason,
                "top1_metric_at_open": signal.at[
                    timestamp, "top1_metric_at_open"
                ],
                "defender_metric_at_open": signal.at[
                    timestamp, "defender_metric_at_open"
                ],
                "metric_difference_at_open": signal.at[
                    timestamp, "metric_difference_at_open"
                ],
                "slow_gate_risk_on": signal.at[timestamp, "slow_gate_risk_on"],
                "emergency_active": signal.at[timestamp, "emergency_active"],
                "entry_qualified": signal.at[timestamp, "entry_qualified"],
                "entry_confirmed": signal.at[timestamp, "entry_confirmed"],
                "target_candidate": target,
            }
        )
    return pd.DataFrame(rows).set_index("date")


def validate_top1_raqm_w5_bridge(
    context: GoldOverrideContext,
    params: Top1RAQMW5BridgeParams,
    state: pd.DataFrame,
    daily: pd.DataFrame,
) -> dict[str, object]:
    entries = state["state_reason"].eq("top1_bridge_entry")
    invalid_entries = int(
        (
            ~state.loc[entries, "entry_confirmed"].astype(bool)
            | state.loc[entries, "emergency_active"].astype(bool)
            | ~state.loc[entries, "slow_gate_risk_on"].astype(bool)
            | ~state.loc[entries, "formal_target_candidate"].eq(
                DEFENDER_CANDIDATE
            )
        ).sum()
    )
    emergency_violations = int(
        (
            state["top1_bridge_active"].astype(bool)
            & state["emergency_active"].astype(bool)
        ).sum()
    )
    active = state["top1_bridge_active"].astype(bool)
    active_matches = bool(
        state.loc[active, "target_candidate"].equals(
            context.momentum_target.loc[active]
        )
    )
    inactive_matches = bool(
        state.loc[~active, "target_candidate"].equals(
            state.loc[~active, "formal_target_candidate"]
        )
    )
    formal_priority = ~state["formal_target_candidate"].eq(DEFENDER_CANDIDATE)
    formal_priority_violations = int(
        state.loc[formal_priority, "top1_bridge_active"].astype(bool).sum()
    )
    nav_error = float(
        ((1.0 + daily["return"]).cumprod() - daily["nav"]).abs().max()
    )
    if (
        invalid_entries
        or emergency_violations
        or formal_priority_violations
        or not active_matches
        or not inactive_matches
        or nav_error > 1e-12
    ):
        raise AssertionError(
            "Top1 RAQM-W5 bridge audit failed: "
            f"entries={invalid_entries}, emergency={emergency_violations}, "
            f"formal={formal_priority_violations}, active={active_matches}, "
            f"inactive={inactive_matches}, nav={nav_error:.3e}"
        )
    asset_days = state.loc[active, "target_candidate"].value_counts()
    return {
        "status": "passed",
        "candidate_id": params.candidate_id(),
        "baseline_parity_max_abs_error": context.baseline_parity_max_abs_error,
        "invalid_entries": invalid_entries,
        "emergency_override_violations": emergency_violations,
        "formal_priority_violations": formal_priority_violations,
        "active_targets_match_momentum_top1": active_matches,
        "inactive_targets_match_formal_strategy": inactive_matches,
        "nav_reconstruction_max_abs_error": nav_error,
        "bridge_entries": int(entries.sum()),
        "bridge_exits": int(state["top1_bridge_changed"].sum() - entries.sum()),
        "bridge_days": int(active.sum()),
        "bridge_asset_days": {
            str(asset): int(days) for asset, days in asset_days.items()
        },
        "switches": int(daily["switched"].sum()),
        "performance": performance(daily["return"]),
    }


def run_top1_raqm_w5_bridge(
    context: GoldOverrideContext,
    params: Top1RAQMW5BridgeParams,
    *,
    metrics: pd.DataFrame | None = None,
    formal_run=None,
) -> Top1RAQMW5BridgeBacktest:
    applied_metrics = (
        registered_raqm_w5_at_open(context.curves)
        if metrics is None
        else metrics
    )
    applied_formal = (
        run_gold_raqm_w5(
            context,
            GoldRAQMW5Params(
                FORMAL_GOLD_ENTRY_DIFFERENCE,
                FORMAL_GOLD_EXIT_DIFFERENCE,
            ),
        )
        if formal_run is None
        else formal_run
    )
    state = top1_raqm_w5_bridge_schedule(
        context,
        applied_formal.state["target_candidate"],
        applied_metrics,
        params,
    )
    daily = simulate_candidate_schedule(
        state["target_candidate"],
        context.interfaces,
        context.initial_previous_candidate,
    )
    audit = validate_top1_raqm_w5_bridge(context, params, state, daily)
    return Top1RAQMW5BridgeBacktest(
        params=params,
        metrics_at_open=applied_metrics,
        state=state,
        daily=daily,
        audit=audit,
    )


def collect_grid(
    context: GoldOverrideContext,
    entry_minimums: Iterable[float],
    confirmation_days: Iterable[int],
    minimum_differences: Iterable[float | None],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = registered_raqm_w5_at_open(context.curves)
    formal_run = run_gold_raqm_w5(
        context,
        GoldRAQMW5Params(
            FORMAL_GOLD_ENTRY_DIFFERENCE,
            FORMAL_GOLD_EXIT_DIFFERENCE,
        ),
    )
    records: list[dict[str, object]] = []
    returns: dict[str, np.ndarray] = {}
    for entry_minimum in entry_minimums:
        for confirmation in confirmation_days:
            for difference in minimum_differences:
                params = Top1RAQMW5BridgeParams(
                    entry_minimum=float(entry_minimum),
                    confirmation_days=int(confirmation),
                    minimum_difference=(
                        None if difference is None else float(difference)
                    ),
                )
                run = run_top1_raqm_w5_bridge(
                    context,
                    params,
                    metrics=metrics,
                    formal_run=formal_run,
                )
                candidate_id = params.candidate_id()
                records.append(
                    {
                        "candidate_id": candidate_id,
                        **asdict(params),
                        "bridge_entries": run.audit["bridge_entries"],
                        "bridge_days": run.audit["bridge_days"],
                        "switches": run.audit["switches"],
                        **{
                            f"bridge_days_{asset}": run.audit[
                                "bridge_asset_days"
                            ].get(asset, 0)
                            for asset in MOMENTUM_ASSETS
                        },
                    }
                )
                returns[candidate_id] = run.daily["return"].to_numpy(float)
    metadata = pd.DataFrame(records).set_index("candidate_id")
    matrix = pd.DataFrame(returns, index=context.calendar)
    if metadata.index.duplicated().any() or matrix.columns.duplicated().any():
        raise AssertionError("Top1 RAQM-W5 grid contains duplicate IDs")
    return metadata, matrix
