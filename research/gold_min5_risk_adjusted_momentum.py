"""Gold min-5 escape using the registered 20-day risk-adjusted momentum factor."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from factors.risk_adjusted_quality_momentum import compute
from research.defender_curve_momentum import DEFENDER_CANDIDATE
from research.gold_min5_risk_adjusted_escape import (
    run_gold_min5,
)
from research.momentum_defender_gold_override import (
    GOLD_ASSET,
    GoldOverrideContext,
)


WINDOW = 20
VOL_FLOOR_ANNUAL = 0.08


@dataclass(frozen=True)
class GoldRAQMParams:
    entry_difference: float
    exit_difference: float

    def __post_init__(self) -> None:
        if self.entry_difference < 0.0:
            raise ValueError("entry_difference must be non-negative")
        if self.exit_difference > self.entry_difference:
            raise ValueError("exit_difference must not exceed entry_difference")

    def candidate_id(self) -> str:
        return (
            f"risk_adjusted_quality_momentum_w20_"
            f"en{self.entry_difference:+.3f}_ex{self.exit_difference:+.3f}_hard_h5"
        )


def risk_adjusted_momentum_at_open(
    curves: pd.DataFrame,
    *,
    window: int = WINDOW,
) -> pd.DataFrame:
    """Apply the exact registered factor to Gold and whole-Defender NAV."""
    values: dict[str, pd.Series] = {}
    for candidate in (GOLD_ASSET, DEFENDER_CANDIDATE):
        frame = pd.DataFrame(
            {
                "date": curves.index,
                "close": curves[candidate].to_numpy(float),
            }
        )
        values[candidate] = compute(
            frame,
            {"window": window, "vol_floor_annual": VOL_FLOOR_ANNUAL},
        ).reindex(curves.index).shift(1)
    result = pd.DataFrame(values, index=curves.index)
    result["difference"] = result[GOLD_ASSET] - result[DEFENDER_CANDIDATE]
    result.index.name = "date"
    return result


def run_gold_raqm(
    context: GoldOverrideContext,
    params: GoldRAQMParams,
    *,
    metrics: pd.DataFrame | None = None,
):
    applied = risk_adjusted_momentum_at_open(context.curves) if metrics is None else metrics
    # The fixed state machine only requires entry/exit fields and candidate_id;
    # passing the stricter RAQM parameter type preserves one implementation of
    # the five-day execution semantics.
    return run_gold_min5(context, params, metrics=applied)


def collect_grid(
    context: GoldOverrideContext,
    entry_values: Iterable[float],
    exit_values: Iterable[float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = risk_adjusted_momentum_at_open(context.curves)
    records: list[dict[str, object]] = []
    returns: dict[str, np.ndarray] = {}
    for entry_value in entry_values:
        for exit_value in exit_values:
            if float(exit_value) > float(entry_value):
                continue
            params = GoldRAQMParams(float(entry_value), float(exit_value))
            run = run_gold_raqm(context, params, metrics=metrics)
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
