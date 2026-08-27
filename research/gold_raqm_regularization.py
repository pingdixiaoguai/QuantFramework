"""Optional-floor/optional-winsor Gold RAQM research primitives."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from research.defender_curve_momentum import DEFENDER_CANDIDATE
from research.momentum_defender_gold_override import GOLD_ASSET
from research.momentum_defender_log_qm_switch import (
    FastSwitchData,
    fast_candidate_schedule,
    fast_gold_targets,
)


@dataclass(frozen=True)
class RAQMSpec:
    family: str
    window: int
    volatility_floor_annual: float | None
    winsor_limit: float | None
    extra_numeric_parameters: int

    def __post_init__(self) -> None:
        if self.window < 2:
            raise ValueError("RAQM window must be at least two")
        if self.volatility_floor_annual is not None and self.volatility_floor_annual <= 0:
            raise ValueError("volatility floor must be positive")
        if self.winsor_limit is not None and self.winsor_limit <= 0:
            raise ValueError("winsor limit must be positive")
        if self.extra_numeric_parameters not in {0, 1, 2}:
            raise ValueError("invalid extra parameter count")

    def factor_id(self) -> str:
        floor = "none" if self.volatility_floor_annual is None else f"{self.volatility_floor_annual:.2f}"
        winsor = "none" if self.winsor_limit is None else f"{self.winsor_limit:.1f}"
        return f"{self.family}_w{self.window}_floor{floor}_clip{winsor}"


@dataclass(frozen=True)
class GoldRuleSpec:
    factor: RAQMSpec
    entry_difference: float
    exit_difference: float
    hard_min_hold_days: int = 5

    def __post_init__(self) -> None:
        if self.exit_difference > self.entry_difference:
            raise ValueError("exit difference cannot exceed entry difference")
        if self.hard_min_hold_days < 1:
            raise ValueError("hard hold must be positive")

    def candidate_id(self) -> str:
        return (
            f"{self.factor.factor_id()}_en{self.entry_difference:+.2f}_"
            f"ex{self.exit_difference:+.2f}_h{self.hard_min_hold_days}"
        )


@dataclass(frozen=True)
class GoldRuleResult:
    returns: np.ndarray
    target_candidate: np.ndarray
    gold_entries: int
    gold_days: int
    switches: int


def raqm_score(curve: pd.Series, spec: RAQMSpec) -> pd.Series:
    """Calculate log-return RAQM with independently removable regularizers."""
    values = curve.astype(float)
    daily_log = np.log(values).diff()
    total_log = np.log(values).diff(spec.window)
    path = daily_log.abs().rolling(spec.window).sum()
    efficiency = total_log.abs() / path.replace(0.0, np.nan)
    volatility = daily_log.rolling(spec.window).std(ddof=1) * np.sqrt(spec.window)
    adjusted = volatility
    if spec.volatility_floor_annual is not None:
        floor = spec.volatility_floor_annual * np.sqrt(spec.window / 252.0)
        adjusted = np.maximum(adjusted, floor)
    risk_adjusted = total_log / adjusted.replace(0.0, np.nan)
    if spec.winsor_limit is not None:
        risk_adjusted = risk_adjusted.clip(
            lower=-spec.winsor_limit,
            upper=spec.winsor_limit,
        )
    return (risk_adjusted * efficiency).astype(float)


def metric_at_open(curves: pd.DataFrame, spec: RAQMSpec) -> pd.DataFrame:
    result = pd.DataFrame(index=curves.index)
    for candidate in (GOLD_ASSET, DEFENDER_CANDIDATE):
        result[candidate] = raqm_score(curves[candidate], spec).shift(1)
    result["difference"] = result[GOLD_ASSET] - result[DEFENDER_CANDIDATE]
    return result


def run_gold_rule(
    data: FastSwitchData,
    risk_on: np.ndarray,
    difference: pd.Series,
    spec: GoldRuleSpec,
) -> GoldRuleResult:
    applied = replace(data, gold_difference=difference.reindex(data.calendar).to_numpy(float))
    target, entries, days = fast_gold_targets(
        applied,
        risk_on,
        entry_difference=spec.entry_difference,
        exit_difference=spec.exit_difference,
        minimum_hold_days=spec.hard_min_hold_days,
    )
    returns, actual, switches = fast_candidate_schedule(applied, target)
    return GoldRuleResult(returns, actual, entries, days, switches)


def run_no_gold(data: FastSwitchData, risk_on: np.ndarray) -> np.ndarray:
    defender = data.candidate_index[DEFENDER_CANDIDATE]
    target = np.where(risk_on, data.momentum_target, defender).astype(int)
    returns, _, _ = fast_candidate_schedule(data, target)
    return returns
