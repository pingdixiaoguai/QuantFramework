"""Unified Momentum Top-1 escape gate while the base C2 holds Defender."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from factors.quality_momentum import compute as quality_momentum
from research.defender_curve_momentum import DEFENDER_CANDIDATE
from research.momentum_defender_gold_override import (
    METRICS,
    QUALITY_METRIC,
    RISK_ADJUSTED_METRIC,
    RETURN_METRIC,
    GoldOverrideContext,
    simulate_candidate_schedule,
)
from research.momentum_defender_occam import MOMENTUM_ASSETS, performance


@dataclass(frozen=True)
class Top1EscapeParams:
    metric: str = QUALITY_METRIC
    window: int = 20
    entry_difference: float = 0.0
    exit_difference: float = 0.0
    absolute_minimum: float = 0.0
    min_escape_hold_days: int = 1

    def __post_init__(self) -> None:
        if self.metric not in METRICS:
            raise ValueError(f"metric must be one of {METRICS}")
        if self.window < 2 or self.min_escape_hold_days < 1:
            raise ValueError("window must be >=2 and hold days positive")
        if self.exit_difference > self.entry_difference:
            raise ValueError("exit_difference must not exceed entry_difference")

    def candidate_id(self) -> str:
        return (
            f"{self.metric}_w{self.window}_en{self.entry_difference:+.4f}_"
            f"ex{self.exit_difference:+.4f}_abs{self.absolute_minimum:+.4f}_"
            f"h{self.min_escape_hold_days}"
        )


@dataclass(frozen=True)
class Top1EscapeBacktest:
    params: Top1EscapeParams
    metrics_at_open: pd.DataFrame
    state: pd.DataFrame
    daily: pd.DataFrame
    audit: dict[str, object]


def _quality(curve: pd.Series, window: int) -> pd.Series:
    frame = pd.DataFrame({"date": curve.index, "close": curve.to_numpy(float)})
    return quality_momentum(frame, {"window": window}).reindex(curve.index)


def all_metrics_at_open(
    curves: pd.DataFrame,
    metric: str,
    window: int,
) -> pd.DataFrame:
    """Compute one identical close-known metric for four ETFs and Defender."""
    if metric not in METRICS:
        raise ValueError(f"unsupported metric: {metric}")
    close_metrics: dict[str, pd.Series] = {}
    for candidate in (*MOMENTUM_ASSETS, DEFENDER_CANDIDATE):
        curve = curves[candidate].astype(float)
        if metric == RETURN_METRIC:
            signal = curve.pct_change(window)
        elif metric == QUALITY_METRIC:
            signal = _quality(curve, window)
        else:
            trailing_return = curve.pct_change(window)
            volatility = (
                curve.pct_change().rolling(window).std(ddof=1) * np.sqrt(252.0)
            )
            signal = trailing_return / volatility.replace(0.0, np.nan)
        close_metrics[candidate] = signal.shift(1)
    result = pd.DataFrame(close_metrics, index=curves.index)
    result.index.name = "date"
    return result


def top1_escape_schedule(
    context: GoldOverrideContext,
    metrics: pd.DataFrame,
    params: Top1EscapeParams,
) -> pd.DataFrame:
    """Apply one common escape rule to whichever Momentum ETF is Top-1."""
    base_risk_on = context.integrated.result.state["risk_on"].astype(bool)
    active = False
    active_candidate: str | None = None
    held_days = 10**9
    rows: list[dict[str, object]] = []
    for timestamp in context.calendar:
        previous_active = active
        previous_candidate = active_candidate
        top1 = str(context.momentum_target.loc[timestamp])
        top1_metric = metrics.at[timestamp, top1]
        defender_metric = metrics.at[timestamp, DEFENDER_CANDIDATE]
        difference = top1_metric - defender_metric
        entry_qualified = bool(
            pd.notna(difference)
            and pd.notna(top1_metric)
            and float(difference) > params.entry_difference
            and float(top1_metric) > params.absolute_minimum
        )
        stay_qualified = bool(
            pd.notna(difference)
            and pd.notna(top1_metric)
            and float(difference) > params.exit_difference
            and float(top1_metric) > params.absolute_minimum
        )
        reason = "hold"
        if bool(base_risk_on.loc[timestamp]):
            active = False
            active_candidate = None
            if previous_active:
                reason = "base_c2_returned_to_momentum"
        elif not active:
            if entry_qualified:
                active = True
                active_candidate = top1
                held_days = 0
                reason = "top1_escape_entry"
        elif held_days >= params.min_escape_hold_days:
            if not stay_qualified:
                active = False
                active_candidate = None
                held_days = 0
                reason = "top1_escape_exit"
            elif top1 != active_candidate:
                active_candidate = top1
                held_days = 0
                reason = "top1_escape_rotation"

        if bool(base_risk_on.loc[timestamp]):
            target = top1
        else:
            target = active_candidate if active else DEFENDER_CANDIDATE
        rows.append(
            {
                "date": timestamp,
                "base_c2_risk_on": bool(base_risk_on.loc[timestamp]),
                "momentum_top1": top1,
                "top1_escape_active": active,
                "top1_escape_changed": active != previous_active,
                "active_escape_candidate": active_candidate,
                "escape_candidate_changed": active_candidate != previous_candidate,
                "state_reason": reason,
                "escape_held_days_at_open": held_days,
                "top1_metric_at_open": top1_metric,
                "defender_metric_at_open": defender_metric,
                "metric_difference_at_open": difference,
                "entry_qualified": entry_qualified,
                "stay_qualified": stay_qualified,
                "target_candidate": target,
            }
        )
        if active:
            held_days += 1
    return pd.DataFrame(rows).set_index("date")


def validate_top1_escape(
    context: GoldOverrideContext,
    params: Top1EscapeParams,
    state: pd.DataFrame,
    daily: pd.DataFrame,
) -> dict[str, object]:
    entries = (
        state["top1_escape_changed"].astype(bool)
        & state["top1_escape_active"].astype(bool)
    )
    invalid_entries = int(
        (
            state.loc[entries, "base_c2_risk_on"].astype(bool)
            | ~state.loc[entries, "entry_qualified"].astype(bool)
        ).sum()
    )
    momentum_rows = state["base_c2_risk_on"].astype(bool)
    momentum_matches = bool(
        state.loc[momentum_rows, "target_candidate"].equals(
            context.momentum_target.loc[momentum_rows]
        )
    )
    nav_error = float(
        ((1.0 + daily["return"]).cumprod() - daily["nav"]).abs().max()
    )
    if invalid_entries or not momentum_matches or nav_error > 1e-12:
        raise AssertionError(
            "Top1 escape audit failed: "
            f"entries={invalid_entries}, momentum={momentum_matches}, nav={nav_error:.3e}"
        )
    asset_days = state.loc[state["top1_escape_active"], "active_escape_candidate"].value_counts()
    return {
        "status": "passed",
        "candidate_id": params.candidate_id(),
        "baseline_parity_max_abs_error": context.baseline_parity_max_abs_error,
        "invalid_escape_entries": invalid_entries,
        "momentum_rows_match_base_c2": momentum_matches,
        "nav_reconstruction_max_abs_error": nav_error,
        "escape_entries": int(entries.sum()),
        "escape_rotations": int(state["state_reason"].eq("top1_escape_rotation").sum()),
        "escape_days": int(state["top1_escape_active"].sum()),
        "escape_asset_days": {str(key): int(value) for key, value in asset_days.items()},
        "switches": int(daily["switched"].sum()),
        "performance": performance(daily["return"]),
    }


def run_top1_escape(
    context: GoldOverrideContext,
    params: Top1EscapeParams,
    *,
    metrics: pd.DataFrame | None = None,
) -> Top1EscapeBacktest:
    applied = (
        all_metrics_at_open(context.curves, params.metric, params.window)
        if metrics is None
        else metrics
    )
    state = top1_escape_schedule(context, applied, params)
    daily = simulate_candidate_schedule(
        state["target_candidate"],
        context.interfaces,
        context.initial_previous_candidate,
    )
    audit = validate_top1_escape(context, params, state, daily)
    return Top1EscapeBacktest(params, applied, state, daily, audit)


def candidate_record(
    backtest: Top1EscapeBacktest,
    periods: Mapping[str, tuple[date, date]],
) -> dict[str, object]:
    row: dict[str, object] = {
        "candidate_id": backtest.params.candidate_id(),
        **asdict(backtest.params),
        "escape_entries": backtest.audit["escape_entries"],
        "escape_rotations": backtest.audit["escape_rotations"],
        "escape_days": backtest.audit["escape_days"],
        **{
            f"escape_days_{asset}": backtest.audit["escape_asset_days"].get(asset, 0)
            for asset in MOMENTUM_ASSETS
        },
    }
    for label, (start, end) in periods.items():
        metrics = performance(
            backtest.daily.loc[pd.Timestamp(start) : pd.Timestamp(end), "return"]
        )
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
    periods: Mapping[str, tuple[date, date]],
    grid: Mapping[str, Mapping[str, Iterable[float | int]]],
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    cache: dict[tuple[str, int], pd.DataFrame] = {}
    for metric, settings in grid.items():
        for window_value in settings["windows"]:
            window = int(window_value)
            key = (str(metric), window)
            cache[key] = all_metrics_at_open(context.curves, str(metric), window)
            for entry_value in settings["entry_differences"]:
                for exit_value in settings["exit_differences"]:
                    entry = float(entry_value)
                    exit_ = float(exit_value)
                    if exit_ > entry:
                        continue
                    for hold_value in settings["min_escape_hold_days"]:
                        params = Top1EscapeParams(
                            metric=str(metric),
                            window=window,
                            entry_difference=entry,
                            exit_difference=exit_,
                            absolute_minimum=0.0,
                            min_escape_hold_days=int(hold_value),
                        )
                        run = run_top1_escape(context, params, metrics=cache[key])
                        records.append(candidate_record(run, periods))
    result = pd.DataFrame(records)
    if result["candidate_id"].duplicated().any():
        raise AssertionError("Top1 escape grid contains duplicate candidates")
    return result


def collect_candidate_returns(
    context: GoldOverrideContext,
    grid: Mapping[str, Mapping[str, Iterable[float | int]]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return metadata and daily returns for every unified-grid candidate."""
    metadata: list[dict[str, object]] = []
    returns: dict[str, np.ndarray] = {}
    cache: dict[tuple[str, int], pd.DataFrame] = {}
    for metric, settings in grid.items():
        for window_value in settings["windows"]:
            window = int(window_value)
            key = (str(metric), window)
            cache[key] = all_metrics_at_open(context.curves, str(metric), window)
            for entry_value in settings["entry_differences"]:
                for exit_value in settings["exit_differences"]:
                    entry = float(entry_value)
                    exit_ = float(exit_value)
                    if exit_ > entry:
                        continue
                    for hold_value in settings["min_escape_hold_days"]:
                        params = Top1EscapeParams(
                            metric=str(metric),
                            window=window,
                            entry_difference=entry,
                            exit_difference=exit_,
                            absolute_minimum=0.0,
                            min_escape_hold_days=int(hold_value),
                        )
                        run = run_top1_escape(context, params, metrics=cache[key])
                        candidate_id = params.candidate_id()
                        returns[candidate_id] = run.daily["return"].to_numpy(float)
                        metadata.append(
                            {
                                "candidate_id": candidate_id,
                                **asdict(params),
                                "escape_entries": run.audit["escape_entries"],
                                "escape_rotations": run.audit["escape_rotations"],
                                "escape_days": run.audit["escape_days"],
                                **{
                                    f"escape_days_{asset}": run.audit[
                                        "escape_asset_days"
                                    ].get(asset, 0)
                                    for asset in MOMENTUM_ASSETS
                                },
                            }
                        )
    return (
        pd.DataFrame(metadata).set_index("candidate_id"),
        pd.DataFrame(returns, index=context.calendar),
    )
