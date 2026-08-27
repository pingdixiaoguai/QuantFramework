"""Monthly underperformance attribution for integrated C2 versus Momentum."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from research.momentum_defender_integrated import IntegratedC2Backtest, run_integrated_c2
from research.momentum_defender_occam import performance, simulate_switch


@dataclass(frozen=True)
class CounterfactualResult:
    name: str
    state: pd.DataFrame
    simulated: pd.DataFrame


def _compound(values: pd.Series) -> float:
    return float((1.0 + values.astype(float)).prod() - 1.0)


def _bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().eq("true")


def classify_daily_excess(daily: pd.DataFrame) -> pd.DataFrame:
    """Assign each daily C2-vs-Momentum log excess to a causal state bucket."""
    result = daily.copy()
    risk_on = _bool(result["risk_on"])
    slow = _bool(result["slow_signal_asof_previous_close"])
    emergency = _bool(result["emergency_asof_previous_close"])
    held_days = result["held_days_at_open"].astype(int)
    category = pd.Series("momentum_hold", index=result.index, dtype=object)
    defender = ~risk_on
    category.loc[defender & ~slow] = "slow_gate_defender"
    category.loc[defender & slow & emergency] = "emergency_cap_hold"
    category.loc[defender & slow & ~emergency & held_days.lt(30)] = (
        "defender_exit_lock_delay"
    )
    category.loc[
        defender
        & slow
        & ~emergency
        & held_days.ge(30)
    ] = "defender_other"
    transitions = result["transition"].astype(str).isin(
        ["momentum_to_defender", "defender_to_momentum"]
    )
    category.loc[transitions] = "sleeve_transition"
    result["cause_category"] = category
    result["log_excess"] = np.log1p(result["return"].astype(float)) - np.log1p(
        result["momentum_exact_return"].astype(float)
    )
    result["negative_log_excess"] = result["log_excess"].clip(upper=0.0)
    result["month"] = result.index.to_period("M").astype(str)
    return result


def monthly_returns(daily: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for month, sample in daily.groupby(daily.index.to_period("M")):
        c2_return = _compound(sample["return"])
        momentum_return = _compound(sample["momentum_exact_return"])
        relative = (1.0 + c2_return) / (1.0 + momentum_return) - 1.0
        category_net = sample.groupby("cause_category")["log_excess"].sum()
        negative = category_net.loc[category_net.lt(0.0)]
        primary_cause = str(negative.idxmin()) if not negative.empty else "none"
        defender = ~_bool(sample["risk_on"])
        momentum_asset = sample.loc[defender, "momentum_target_at_open"]
        dominant_asset = (
            str(momentum_asset.value_counts().idxmax())
            if not momentum_asset.empty
            else "none"
        )
        rows.append(
            {
                "month": str(month),
                "start": sample.index.min().date().isoformat(),
                "end": sample.index.max().date().isoformat(),
                "observations": int(len(sample)),
                "c2_return": c2_return,
                "momentum_return": momentum_return,
                "relative_return": relative,
                "underperformed": relative < -1e-12,
                "defender_days": int(defender.sum()),
                "defender_day_share": float(defender.mean()),
                "slow_gate_defender_days": int(
                    sample["cause_category"].eq("slow_gate_defender").sum()
                ),
                "exit_lock_delay_days": int(
                    sample["cause_category"].eq("defender_exit_lock_delay").sum()
                ),
                "emergency_cap_days": int(
                    sample["cause_category"].eq("emergency_cap_hold").sum()
                ),
                "sleeve_transition_days": int(
                    sample["cause_category"].eq("sleeve_transition").sum()
                ),
                "dominant_momentum_asset_during_defender": dominant_asset,
                "gold_momentum_days_during_defender": int(
                    momentum_asset.eq("518880.SH").sum()
                ),
                "primary_cause": primary_cause,
            }
        )
    return pd.DataFrame(rows).set_index("month")


def apply_asymmetric_state_schedule(
    slow_at_open: pd.Series,
    emergency_at_open: pd.Series,
    calendar: pd.DatetimeIndex,
    *,
    momentum_min_hold_days: int = 30,
    defender_min_hold_days: int = 30,
    emergency_override: bool = True,
) -> pd.DataFrame:
    """Replay C2 while allowing a separate Defender exit lock."""
    if momentum_min_hold_days < 1 or defender_min_hold_days < 1:
        raise ValueError("hold periods must be positive")
    slow = slow_at_open.reindex(calendar)
    emergency = emergency_at_open.reindex(calendar).fillna(False).astype(bool)
    state = True
    held_days = 10**9
    rows: list[dict[str, object]] = []
    for timestamp in calendar:
        previous = state
        reason = "hold"
        if bool(emergency.loc[timestamp]):
            if state and (emergency_override or held_days >= momentum_min_hold_days):
                state = False
                held_days = 0
                reason = "emergency_exit"
            elif not state:
                reason = "emergency_hold"
        else:
            wanted = slow.loc[timestamp]
            required = momentum_min_hold_days if state else defender_min_hold_days
            if pd.notna(wanted) and bool(wanted) != state and held_days >= required:
                state = bool(wanted)
                held_days = 0
                reason = "slow_regime_switch"
        rows.append(
            {
                "date": timestamp,
                "risk_on": state,
                "state_changed": state != previous,
                "state_reason": reason,
                "slow_signal_asof_previous_close": slow.loc[timestamp],
                "emergency_asof_previous_close": bool(emergency.loc[timestamp]),
                "held_days_at_open": held_days,
            }
        )
        held_days += 1
    return pd.DataFrame(rows).set_index("date")


def run_counterfactuals(integrated: IntegratedC2Backtest) -> list[CounterfactualResult]:
    result = integrated.result
    calendar = result.inputs.calendar
    variants: list[CounterfactualResult] = []
    for exit_lock in (30, 20, 10, 5, 1):
        state = apply_asymmetric_state_schedule(
            result.slow_signal,
            result.emergency_alert,
            calendar,
            momentum_min_hold_days=30,
            defender_min_hold_days=exit_lock,
        )
        simulated = simulate_switch(
            result.inputs.momentum,
            result.inputs.defender,
            state["risk_on"],
            initial_previous_state=result.config.initial_previous_sleeve,
        )
        variants.append(CounterfactualResult(f"defender_exit_lock_{exit_lock}", state, simulated))
    no_emergency_state = apply_asymmetric_state_schedule(
        result.slow_signal,
        pd.Series(False, index=calendar),
        calendar,
        momentum_min_hold_days=30,
        defender_min_hold_days=30,
    )
    no_emergency = simulate_switch(
        result.inputs.momentum,
        result.inputs.defender,
        no_emergency_state["risk_on"],
        initial_previous_state=result.config.initial_previous_sleeve,
    )
    variants.append(CounterfactualResult("no_emergency_cap", no_emergency_state, no_emergency))
    return variants


def build_analysis(root: Path) -> tuple[
    IntegratedC2Backtest,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    integrated = run_integrated_c2(root)
    result = integrated.result
    daily = integrated.daily.copy()
    momentum_weights = result.inputs.momentum[
        [
            "target_weight_510300.SH",
            "target_weight_159915.SZ",
            "target_weight_513100.SH",
            "target_weight_518880.SH",
        ]
    ].astype(float)
    daily["momentum_target_at_open"] = momentum_weights.idxmax(axis=1).str.removeprefix(
        "target_weight_"
    )
    attributed = classify_daily_excess(daily)
    monthly = monthly_returns(attributed)

    cause_rows = []
    losing_months = set(monthly.index[monthly["underperformed"]])
    losing_daily = attributed.loc[attributed["month"].isin(losing_months)]
    for category, sample in losing_daily.groupby("cause_category"):
        cause_rows.append(
            {
                "cause_category": category,
                "days": int(len(sample)),
                "net_log_excess": float(sample["log_excess"].sum()),
                "negative_log_excess": float(sample["negative_log_excess"].sum()),
                "months_present": int(sample["month"].nunique()),
            }
        )
    cause_summary = pd.DataFrame(cause_rows).set_index("cause_category")

    variants = run_counterfactuals(integrated)
    counterfactual_rows = []
    counterfactual_monthly_rows = []
    momentum = result.inputs.momentum["daily_net_return_if_held"].astype(float)
    for variant in variants:
        metrics = performance(variant.simulated["return"])
        frame = pd.DataFrame(
            {
                "return": variant.simulated["return"],
                "momentum_exact_return": momentum,
                "risk_on": variant.state["risk_on"],
            }
        )
        rows = []
        for month, sample in frame.groupby(frame.index.to_period("M")):
            strategy_return = _compound(sample["return"])
            momentum_return = _compound(sample["momentum_exact_return"])
            rows.append(
                {
                    "variant": variant.name,
                    "month": str(month),
                    "strategy_return": strategy_return,
                    "momentum_return": momentum_return,
                    "relative_return": (1.0 + strategy_return) / (1.0 + momentum_return) - 1.0,
                }
            )
        variant_monthly = pd.DataFrame(rows)
        counterfactual_monthly_rows.extend(rows)
        counterfactual_rows.append(
            {
                "variant": variant.name,
                **metrics,
                "underperform_months": int(
                    variant_monthly["relative_return"].lt(-1e-12).sum()
                ),
                "defender_days": int((~variant.state["risk_on"].astype(bool)).sum()),
                "sleeve_switches": int(variant.simulated["sleeve_switch"].sum()),
            }
        )
    counterfactual_metrics = pd.DataFrame(counterfactual_rows).set_index("variant")
    counterfactual_monthly = pd.DataFrame(counterfactual_monthly_rows)
    return integrated, attributed, monthly, cause_summary, counterfactual_metrics, counterfactual_monthly
