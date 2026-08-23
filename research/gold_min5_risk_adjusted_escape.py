"""Fixed 10-day risk-adjusted Gold escape with a hard five-day holding rule."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from research.defender_curve_momentum import DEFENDER_CANDIDATE
from research.momentum_defender_gold_override import (
    GOLD_ASSET,
    GoldOverrideContext,
    metric_at_open,
    simulate_candidate_schedule,
)
from research.momentum_defender_occam import performance


WINDOW = 10
MIN_GOLD_HOLD_DAYS = 5


@dataclass(frozen=True)
class GoldMin5Params:
    entry_difference: float
    exit_difference: float

    def __post_init__(self) -> None:
        if self.entry_difference < 0.0:
            raise ValueError("entry_difference must be non-negative")
        if self.exit_difference > self.entry_difference:
            raise ValueError("exit_difference must not exceed entry_difference")

    def candidate_id(self) -> str:
        return (
            f"risk_adjusted_return_w10_en{self.entry_difference:+.3f}_"
            f"ex{self.exit_difference:+.3f}_hard_h5"
        )


@dataclass(frozen=True)
class GoldMin5Backtest:
    params: GoldMin5Params
    state: pd.DataFrame
    daily: pd.DataFrame
    audit: dict[str, object]


def gold_min5_schedule(
    context: GoldOverrideContext,
    metrics: pd.DataFrame,
    params: GoldMin5Params,
) -> pd.DataFrame:
    """Hold Gold for five complete sessions before any C2 or exit decision."""
    base_risk_on = context.integrated.result.state["risk_on"].astype(bool)
    active = False
    held_days = 10**9
    rows: list[dict[str, object]] = []
    for timestamp in context.calendar:
        previous = active
        held_days_before_decision = held_days
        difference = metrics.at[timestamp, "difference"]
        reason = "hold"
        if not active:
            if (
                not bool(base_risk_on.loc[timestamp])
                and pd.notna(difference)
                and float(difference) > params.entry_difference
            ):
                active = True
                held_days = 0
                reason = "gold_entry"
        elif held_days >= MIN_GOLD_HOLD_DAYS:
            if bool(base_risk_on.loc[timestamp]):
                active = False
                held_days = 0
                reason = "gold_to_momentum_after_min_hold"
            elif (
                pd.notna(difference)
                and float(difference) <= params.exit_difference
            ):
                active = False
                held_days = 0
                reason = "gold_to_defender_after_min_hold"

        if active:
            target = GOLD_ASSET
        elif bool(base_risk_on.loc[timestamp]):
            target = str(context.momentum_target.loc[timestamp])
        else:
            target = DEFENDER_CANDIDATE
        rows.append(
            {
                "date": timestamp,
                "base_c2_risk_on": bool(base_risk_on.loc[timestamp]),
                "gold_active": active,
                "gold_changed": active != previous,
                "state_reason": reason,
                "gold_held_days_at_open": (
                    held_days_before_decision
                    if reason in {
                        "gold_to_momentum_after_min_hold",
                        "gold_to_defender_after_min_hold",
                    }
                    else held_days
                ),
                "gold_metric_at_open": metrics.at[timestamp, GOLD_ASSET],
                "defender_metric_at_open": metrics.at[
                    timestamp, DEFENDER_CANDIDATE
                ],
                "metric_difference_at_open": difference,
                "target_candidate": target,
            }
        )
        if active:
            held_days += 1
    return pd.DataFrame(rows).set_index("date")


def run_gold_min5(
    context: GoldOverrideContext,
    params: GoldMin5Params,
    *,
    metrics: pd.DataFrame | None = None,
) -> GoldMin5Backtest:
    applied = (
        metric_at_open(context.curves, "risk_adjusted_return", WINDOW)
        if metrics is None
        else metrics
    )
    state = gold_min5_schedule(context, applied, params)
    daily = simulate_candidate_schedule(
        state["target_candidate"],
        context.interfaces,
        context.initial_previous_candidate,
    )
    entries = state["state_reason"].eq("gold_entry")
    exits = state["state_reason"].isin(
        ["gold_to_momentum_after_min_hold", "gold_to_defender_after_min_hold"]
    )
    invalid_entries = int(
        (
            state.loc[entries, "base_c2_risk_on"].astype(bool)
            | state.loc[entries, "metric_difference_at_open"].le(
                params.entry_difference
            )
        ).sum()
    )
    early_exits = int(
        state.loc[exits, "gold_held_days_at_open"].lt(
            MIN_GOLD_HOLD_DAYS
        ).sum()
    )
    post_min_momentum = (
        state["base_c2_risk_on"].astype(bool)
        & ~state["gold_active"].astype(bool)
    )
    momentum_matches = bool(
        state.loc[post_min_momentum, "target_candidate"].equals(
            context.momentum_target.loc[post_min_momentum]
        )
    )
    nav_error = float(
        ((1.0 + daily["return"]).cumprod() - daily["nav"]).abs().max()
    )
    if invalid_entries or early_exits or not momentum_matches or nav_error > 1e-12:
        raise AssertionError("Gold min-5 audit failed")
    audit = {
        "status": "passed",
        "candidate_id": params.candidate_id(),
        "baseline_parity_max_abs_error": context.baseline_parity_max_abs_error,
        "invalid_entries": invalid_entries,
        "early_exits": early_exits,
        "post_min_hold_momentum_matches_top1": momentum_matches,
        "nav_reconstruction_max_abs_error": nav_error,
        "gold_entries": int(entries.sum()),
        "gold_to_momentum_exits": int(
            state["state_reason"].eq("gold_to_momentum_after_min_hold").sum()
        ),
        "gold_to_defender_exits": int(
            state["state_reason"].eq("gold_to_defender_after_min_hold").sum()
        ),
        "gold_days": int(state["gold_active"].sum()),
        "switches": int(daily["switched"].sum()),
        "performance": performance(daily["return"]),
    }
    return GoldMin5Backtest(params, state, daily, audit)


def collect_grid(
    context: GoldOverrideContext,
    entry_values: Iterable[float],
    exit_values: Iterable[float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = metric_at_open(context.curves, "risk_adjusted_return", WINDOW)
    records: list[dict[str, object]] = []
    returns: dict[str, np.ndarray] = {}
    for entry_value in entry_values:
        for exit_value in exit_values:
            if float(exit_value) > float(entry_value):
                continue
            params = GoldMin5Params(float(entry_value), float(exit_value))
            run = run_gold_min5(context, params, metrics=metrics)
            candidate_id = params.candidate_id()
            records.append(
                {
                    "candidate_id": candidate_id,
                    **asdict(params),
                    "gold_entries": run.audit["gold_entries"],
                    "gold_to_momentum_exits": run.audit[
                        "gold_to_momentum_exits"
                    ],
                    "gold_to_defender_exits": run.audit[
                        "gold_to_defender_exits"
                    ],
                    "gold_days": run.audit["gold_days"],
                    "switches": run.audit["switches"],
                }
            )
            returns[candidate_id] = run.daily["return"].to_numpy(float)
    return (
        pd.DataFrame(records).set_index("candidate_id"),
        pd.DataFrame(returns, index=context.calendar),
    )
