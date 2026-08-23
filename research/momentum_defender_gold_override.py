"""Gold override layered on top of the frozen integrated C2 state machine."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from factors.quality_momentum import compute as quality_momentum
from research.defender_curve_momentum import (
    DEFENDER_CANDIDATE,
    build_candidate_bundle,
)
from research.momentum_defender_integrated import IntegratedC2Backtest, run_integrated_c2
from research.momentum_defender_occam import (
    ENTER_RETURN,
    ENTRY_COST,
    EXIT_COST,
    EXIT_RETURN,
    HELD_RETURN,
    INTERNAL_COST,
    MOMENTUM_ASSETS,
    performance,
)


GOLD_ASSET = "518880.SH"
RETURN_METRIC = "return"
QUALITY_METRIC = "quality_momentum"
RISK_ADJUSTED_METRIC = "risk_adjusted_return"
METRICS = (RETURN_METRIC, QUALITY_METRIC, RISK_ADJUSTED_METRIC)


@dataclass(frozen=True)
class GoldOverrideParams:
    metric: str = QUALITY_METRIC
    window: int = 20
    entry_threshold: float = 0.0
    exit_threshold: float = 0.0
    min_gold_hold_days: int = 1

    def __post_init__(self) -> None:
        if self.metric not in METRICS:
            raise ValueError(f"metric must be one of {METRICS}")
        if self.window < 2 or self.min_gold_hold_days < 1:
            raise ValueError("window must be >=2 and min_gold_hold_days positive")
        if self.exit_threshold > self.entry_threshold:
            raise ValueError("exit_threshold must not exceed entry_threshold")

    def candidate_id(self) -> str:
        return (
            f"{self.metric}_w{self.window}_en{self.entry_threshold:+.4f}_"
            f"ex{self.exit_threshold:+.4f}_h{self.min_gold_hold_days}"
        )


@dataclass(frozen=True)
class GoldOverrideContext:
    integrated: IntegratedC2Backtest
    calendar: pd.DatetimeIndex
    curves: pd.DataFrame
    interfaces: Mapping[str, pd.DataFrame]
    momentum_target: pd.Series
    baseline_target: pd.Series
    initial_previous_candidate: str
    baseline_parity_max_abs_error: float


@dataclass(frozen=True)
class GoldOverrideBacktest:
    params: GoldOverrideParams
    metrics_at_open: pd.DataFrame
    state: pd.DataFrame
    daily: pd.DataFrame
    audit: dict[str, object]


def _chain(left: float, right: float) -> float:
    return (1.0 + float(left)) * (1.0 + float(right)) - 1.0


def simulate_candidate_schedule(
    target: pd.Series,
    interfaces: Mapping[str, pd.DataFrame],
    initial_previous_candidate: str,
) -> pd.DataFrame:
    """Execute a causal candidate schedule with exact open switch legs."""
    calendar = pd.DatetimeIndex(target.index)
    current = str(initial_previous_candidate)
    nav = 1.0
    rows: list[dict[str, object]] = []
    for timestamp in calendar:
        requested = str(target.loc[timestamp])
        actual_target = requested
        switched = requested != current
        blocked = False
        held_leg = np.nan
        exit_leg = np.nan
        enter_leg = np.nan
        if switched:
            exit_ok = pd.notna(interfaces[current].at[timestamp, EXIT_RETURN])
            entry_ok = pd.notna(interfaces[requested].at[timestamp, ENTER_RETURN])
            if not exit_ok or not entry_ok:
                blocked = True
                switched = False
                actual_target = current

        if switched:
            exit_leg = float(interfaces[current].at[timestamp, EXIT_RETURN])
            enter_leg = float(interfaces[actual_target].at[timestamp, ENTER_RETURN])
            daily_return = _chain(exit_leg, enter_leg)
            exit_cost = float(interfaces[current].at[timestamp, EXIT_COST])
            entry_cost = float(interfaces[actual_target].at[timestamp, ENTRY_COST])
            cost_rate = 1.0 - (1.0 - exit_cost) * (1.0 - entry_cost)
            transition = f"{current}_to_{actual_target}"
            current = actual_target
        else:
            held_leg = float(interfaces[current].at[timestamp, HELD_RETURN])
            daily_return = held_leg
            cost_rate = float(interfaces[current].at[timestamp, INTERNAL_COST])
            transition = f"{current}_hold"
        if not np.isfinite(daily_return) or daily_return <= -1.0:
            raise ValueError(f"invalid candidate return on {timestamp}: {daily_return}")
        nav *= 1.0 + daily_return
        rows.append(
            {
                "date": timestamp,
                "return": daily_return,
                "nav": nav,
                "requested_candidate": requested,
                "candidate": current,
                "transition": transition,
                "switched": switched,
                "switch_blocked_untradable": blocked,
                "cost_rate_at_open": cost_rate,
                "held_return_leg_used": held_leg,
                "exit_return_leg_used": exit_leg,
                "enter_return_leg_used": enter_leg,
            }
        )
    return pd.DataFrame(rows).set_index("date")


def build_gold_override_context(
    root: Path,
    *,
    end: date | None = None,
) -> GoldOverrideContext:
    integrated = run_integrated_c2(root, end=end)
    calendar = integrated.result.inputs.calendar
    _, curves, _, interfaces = build_candidate_bundle(
        end=calendar.max().date(), window=20
    )
    curves = curves.reindex(calendar)
    sliced_interfaces = {
        candidate: frame.reindex(calendar)
        for candidate, frame in interfaces.items()
    }
    momentum_weights = integrated.result.inputs.momentum[
        [f"target_weight_{asset}" for asset in MOMENTUM_ASSETS]
    ].astype(float)
    momentum_target = momentum_weights.idxmax(axis=1).str.removeprefix(
        "target_weight_"
    )
    momentum_target.name = "momentum_target_at_open"
    baseline_target = momentum_target.where(
        integrated.result.state["risk_on"].astype(bool), DEFENDER_CANDIDATE
    ).rename("baseline_target_at_open")
    initial_previous = str(integrated.result.previous_asset.iloc[0])
    replay = simulate_candidate_schedule(
        baseline_target, sliced_interfaces, initial_previous
    )
    parity_error = float(
        (
            replay["return"].astype(float)
            - integrated.result.simulated["return"].astype(float)
        )
        .abs()
        .max()
    )
    # The candidate-level ledger compounds the two one-way switch legs, while
    # the historical Momentum sleeve solves its cash scaling in one portfolio
    # operation.  Their only difference is floating-point cost ordering.
    if parity_error > 5e-8:
        raise AssertionError(
            f"candidate-level replay does not match C2: {parity_error:.3e}"
        )
    return GoldOverrideContext(
        integrated=integrated,
        calendar=calendar,
        curves=curves,
        interfaces=sliced_interfaces,
        momentum_target=momentum_target,
        baseline_target=baseline_target,
        initial_previous_candidate=initial_previous,
        baseline_parity_max_abs_error=parity_error,
    )


def _quality_metric(curve: pd.Series, window: int) -> pd.Series:
    frame = pd.DataFrame(
        {"date": curve.index, "close": curve.to_numpy(float)}
    )
    return quality_momentum(frame, {"window": window}).reindex(curve.index)


def metric_at_open(
    curves: pd.DataFrame,
    metric: str,
    window: int,
) -> pd.DataFrame:
    """Return close-known Gold/Defender metrics and their next-open difference."""
    if metric not in METRICS:
        raise ValueError(f"unsupported metric: {metric}")
    values: dict[str, pd.Series] = {}
    for candidate in (GOLD_ASSET, DEFENDER_CANDIDATE):
        curve = curves[candidate].astype(float)
        if metric == RETURN_METRIC:
            signal = curve.pct_change(window)
        elif metric == QUALITY_METRIC:
            signal = _quality_metric(curve, window)
        else:
            trailing_return = curve.pct_change(window)
            annualized_volatility = (
                curve.pct_change().rolling(window).std(ddof=1) * np.sqrt(252.0)
            )
            signal = trailing_return / annualized_volatility.replace(0.0, np.nan)
        values[candidate] = signal.shift(1)
    frame = pd.DataFrame(values, index=curves.index)
    frame["difference"] = frame[GOLD_ASSET] - frame[DEFENDER_CANDIDATE]
    frame.index.name = "date"
    return frame


def gold_override_schedule(
    context: GoldOverrideContext,
    metrics: pd.DataFrame,
    params: GoldOverrideParams,
) -> pd.DataFrame:
    """Overlay Gold only while the base C2 state remains in Defender."""
    difference = metrics["difference"].reindex(context.calendar)
    base_risk_on = context.integrated.result.state["risk_on"].astype(bool)
    active = False
    gold_held_days = 10**9
    rows: list[dict[str, object]] = []
    for timestamp in context.calendar:
        previous = active
        reason = "hold"
        value = difference.loc[timestamp]
        if bool(base_risk_on.loc[timestamp]):
            active = False
            if previous:
                reason = "base_c2_returned_to_momentum"
        elif not active:
            if pd.notna(value) and float(value) > params.entry_threshold:
                active = True
                gold_held_days = 0
                reason = "gold_override_entry"
        elif (
            pd.notna(value)
            and float(value) <= params.exit_threshold
            and gold_held_days >= params.min_gold_hold_days
        ):
            active = False
            gold_held_days = 0
            reason = "gold_override_exit"

        if bool(base_risk_on.loc[timestamp]):
            target = str(context.momentum_target.loc[timestamp])
        else:
            target = GOLD_ASSET if active else DEFENDER_CANDIDATE
        rows.append(
            {
                "date": timestamp,
                "base_c2_risk_on": bool(base_risk_on.loc[timestamp]),
                "gold_override_active": active,
                "gold_override_changed": active != previous,
                "state_reason": reason,
                "gold_held_days_at_open": gold_held_days,
                "gold_metric_at_open": metrics.at[timestamp, GOLD_ASSET],
                "defender_metric_at_open": metrics.at[
                    timestamp, DEFENDER_CANDIDATE
                ],
                "metric_difference_at_open": value,
                "target_candidate": target,
            }
        )
        if active:
            gold_held_days += 1
    return pd.DataFrame(rows).set_index("date")


def validate_gold_override(
    context: GoldOverrideContext,
    params: GoldOverrideParams,
    state: pd.DataFrame,
    daily: pd.DataFrame,
) -> dict[str, object]:
    entries = (
        state["gold_override_changed"].astype(bool)
        & state["gold_override_active"].astype(bool)
    )
    invalid_entries = int(
        (
            state.loc[entries, "base_c2_risk_on"].astype(bool)
            | state.loc[entries, "metric_difference_at_open"].le(
                params.entry_threshold
            )
        ).sum()
    )
    momentum_rows = state["base_c2_risk_on"].astype(bool)
    momentum_target_matches = bool(
        state.loc[momentum_rows, "target_candidate"].equals(
            context.momentum_target.loc[momentum_rows]
        )
    )
    nav_error = float(
        ((1.0 + daily["return"]).cumprod() - daily["nav"]).abs().max()
    )
    if invalid_entries or not momentum_target_matches or nav_error > 1e-12:
        raise AssertionError(
            "gold override audit failed: "
            f"entries={invalid_entries}, momentum={momentum_target_matches}, "
            f"nav={nav_error:.3e}"
        )
    return {
        "status": "passed",
        "candidate_id": params.candidate_id(),
        "baseline_parity_max_abs_error": context.baseline_parity_max_abs_error,
        "invalid_gold_override_entries": invalid_entries,
        "momentum_rows_match_base_c2": momentum_target_matches,
        "nav_reconstruction_max_abs_error": nav_error,
        "gold_override_entries": int(entries.sum()),
        "gold_override_days": int(state["gold_override_active"].sum()),
        "switches": int(daily["switched"].sum()),
        "performance": performance(daily["return"]),
    }


def run_gold_override(
    context: GoldOverrideContext,
    params: GoldOverrideParams,
    *,
    metrics: pd.DataFrame | None = None,
) -> GoldOverrideBacktest:
    applied_metrics = (
        metric_at_open(context.curves, params.metric, params.window)
        if metrics is None
        else metrics
    )
    state = gold_override_schedule(context, applied_metrics, params)
    daily = simulate_candidate_schedule(
        state["target_candidate"],
        context.interfaces,
        context.initial_previous_candidate,
    )
    audit = validate_gold_override(context, params, state, daily)
    return GoldOverrideBacktest(params, applied_metrics, state, daily, audit)


def candidate_record(
    backtest: GoldOverrideBacktest,
    periods: dict[str, tuple[date, date]],
) -> dict[str, object]:
    row: dict[str, object] = {
        "candidate_id": backtest.params.candidate_id(),
        **asdict(backtest.params),
        "gold_override_entries": backtest.audit["gold_override_entries"],
        "gold_override_days": backtest.audit["gold_override_days"],
        "switches": backtest.audit["switches"],
    }
    for label, (start, end) in periods.items():
        returns = backtest.daily.loc[pd.Timestamp(start) : pd.Timestamp(end), "return"]
        metrics = performance(returns)
        for field in (
            "observations",
            "total_return",
            "annualized_return_252",
            "annualized_volatility",
            "sharpe",
            "max_drawdown",
        ):
            row[f"{label}_{field}"] = metrics[field]
    row["worst_split_sharpe"] = min(
        float(row[f"{label}_sharpe"])
        for label in periods
        if label != "full"
    )
    return row


def search_grid(
    context: GoldOverrideContext,
    periods: dict[str, tuple[date, date]],
    grid: Mapping[str, Mapping[str, Iterable[float | int]]],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    metric_cache: dict[tuple[str, int], pd.DataFrame] = {}
    for metric, settings in grid.items():
        for window in settings["windows"]:
            cache_key = (metric, int(window))
            metric_cache[cache_key] = metric_at_open(
                context.curves, metric, int(window)
            )
            for entry in settings["entry_thresholds"]:
                for exit_ in settings["exit_thresholds"]:
                    if float(exit_) > float(entry):
                        continue
                    for hold_days in settings["min_gold_hold_days"]:
                        params = GoldOverrideParams(
                            metric=metric,
                            window=int(window),
                            entry_threshold=float(entry),
                            exit_threshold=float(exit_),
                            min_gold_hold_days=int(hold_days),
                        )
                        candidate = run_gold_override(
                            context,
                            params,
                            metrics=metric_cache[cache_key],
                        )
                        rows.append(candidate_record(candidate, periods))
    result = pd.DataFrame(rows)
    if result["candidate_id"].duplicated().any():
        raise AssertionError("gold override search produced duplicate candidates")
    return result
