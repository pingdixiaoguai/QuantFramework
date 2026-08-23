"""Five-day registered risk-adjusted momentum for the Gold min-5 escape."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from research.gold_min5_risk_adjusted_escape import run_gold_min5
from research.gold_min5_risk_adjusted_momentum import (
    risk_adjusted_momentum_at_open,
)
from research.momentum_defender_gold_override import GoldOverrideContext


WINDOW = 5


@dataclass(frozen=True)
class GoldRAQMW5Params:
    entry_difference: float
    exit_difference: float

    def __post_init__(self) -> None:
        if self.entry_difference < 0.0:
            raise ValueError("entry_difference must be non-negative")
        if self.exit_difference > self.entry_difference:
            raise ValueError("exit_difference must not exceed entry_difference")

    def candidate_id(self) -> str:
        return (
            f"risk_adjusted_quality_momentum_w5_"
            f"en{self.entry_difference:+.3f}_ex{self.exit_difference:+.3f}_hard_h5"
        )


def run_gold_raqm_w5(
    context: GoldOverrideContext,
    params: GoldRAQMW5Params,
    *,
    metrics: pd.DataFrame | None = None,
):
    applied = (
        risk_adjusted_momentum_at_open(context.curves, window=WINDOW)
        if metrics is None
        else metrics
    )
    return run_gold_min5(context, params, metrics=applied)


def collect_grid(
    context: GoldOverrideContext,
    entry_values: Iterable[float],
    exit_values: Iterable[float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = risk_adjusted_momentum_at_open(context.curves, window=WINDOW)
    records: list[dict[str, object]] = []
    returns: dict[str, np.ndarray] = {}
    for entry_value in entry_values:
        for exit_value in exit_values:
            if float(exit_value) > float(entry_value):
                continue
            params = GoldRAQMW5Params(float(entry_value), float(exit_value))
            run = run_gold_raqm_w5(context, params, metrics=metrics)
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
