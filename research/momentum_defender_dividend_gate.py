"""Joint slow-return and Defender-dividend-allocation gate research.

The family enters Defender only when both the 510300 trailing return is below
its threshold and Defender's causal next-open dividend-equity target is at
least the configured minimum.  No emergency rule may bypass this entry gate.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
from typing import Iterable

import pandas as pd

from defender.relative_defender_rotation import ROTATION_ASSETS
from research.momentum_defender_integrated import IntegratedC2Backtest
from research.momentum_defender_occam import performance, simulate_switch


ENTRY_ONLY = "entry_only"
CONJUNCTION = "conjunction"
EXIT_MODES = (ENTRY_ONLY, CONJUNCTION)


@dataclass(frozen=True)
class DividendGateParams:
    lookback: int = 40
    slow_return_threshold: float = 0.025
    defender_primary_minimum: float = 0.80
    min_hold_days: int = 30
    exit_mode: str = ENTRY_ONLY

    def __post_init__(self) -> None:
        if self.lookback < 1 or self.min_hold_days < 1:
            raise ValueError("lookback and min_hold_days must be positive")
        if not 0.0 <= self.defender_primary_minimum <= 1.0:
            raise ValueError("defender_primary_minimum must be in [0, 1]")
        if self.exit_mode not in EXIT_MODES:
            raise ValueError(f"exit_mode must be one of {EXIT_MODES}")

    def candidate_id(self) -> str:
        return (
            f"lb{self.lookback}_r{self.slow_return_threshold:+.3f}_"
            f"p{self.defender_primary_minimum:.1f}_h{self.min_hold_days}_"
            f"{self.exit_mode}"
        )


@dataclass(frozen=True)
class DividendGateBacktest:
    params: DividendGateParams
    slow_return_at_open: pd.Series
    defender_primary_target: pd.Series
    joint_condition: pd.Series
    state: pd.DataFrame
    simulated: pd.DataFrame


def defender_primary_target_at_open(defender: pd.DataFrame) -> pd.Series:
    """Sum the executable next-open weights of all Defender dividend ETFs."""
    columns = [
        f"target_weight_{asset.split('.', maxsplit=1)[0]}"
        for asset in ROTATION_ASSETS
    ]
    missing = sorted(set(columns) - set(defender.columns))
    if missing:
        raise ValueError(f"Defender interface lacks primary weights: {missing}")
    primary = defender[columns].astype(float).sum(axis=1)
    if primary.isna().any() or primary.lt(-1e-12).any() or primary.gt(1.0 + 1e-12).any():
        raise AssertionError("invalid Defender primary target")
    return primary.rename("defender_dividend_primary_target_at_open")


def trailing_return_at_open(
    close: pd.Series,
    calendar: pd.DatetimeIndex,
    lookback: int,
) -> pd.Series:
    """Map a close-known trailing return to the immediately following open."""
    if lookback < 1:
        raise ValueError("lookback must be positive")
    ordered = close.astype(float).sort_index()
    close_signal = ordered / ordered.shift(lookback) - 1.0
    at_open = close_signal.shift(1).reindex(calendar).ffill()
    return at_open.rename(f"return_{lookback}_at_open")


def apply_dividend_gate_schedule(
    slow_return: pd.Series,
    primary_target: pd.Series,
    calendar: pd.DatetimeIndex,
    params: DividendGateParams,
    *,
    initial_risk_on: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    """Apply the joint entry gate and selected exit semantics with a state lock."""
    slow = slow_return.reindex(calendar)
    primary = primary_target.reindex(calendar)
    if primary.isna().any():
        raise ValueError("Defender primary target does not cover the calendar")
    condition = (
        slow.lt(params.slow_return_threshold)
        & primary.ge(params.defender_primary_minimum)
        & slow.notna()
    ).rename("joint_defender_condition_at_open")

    risk_on = bool(initial_risk_on)
    held_days = 10**9
    rows: list[dict[str, object]] = []
    for timestamp in calendar:
        previous = risk_on
        reason = "hold"
        if risk_on:
            if bool(condition.loc[timestamp]) and held_days >= params.min_hold_days:
                risk_on = False
                held_days = 0
                reason = "joint_gate_enter_defender"
        else:
            if params.exit_mode == ENTRY_ONLY:
                exit_now = (
                    pd.notna(slow.loc[timestamp])
                    and float(slow.loc[timestamp]) >= params.slow_return_threshold
                )
                exit_reason = "slow_gate_exit_defender"
            else:
                exit_now = not bool(condition.loc[timestamp])
                exit_reason = "joint_gate_exit_defender"
            if exit_now and held_days >= params.min_hold_days:
                risk_on = True
                held_days = 0
                reason = exit_reason
        rows.append(
            {
                "date": timestamp,
                "risk_on": risk_on,
                "state_changed": risk_on != previous,
                "state_reason": reason,
                "held_days_at_open": held_days,
                "slow_return_asof_previous_close": slow.loc[timestamp],
                "defender_primary_target_at_open": primary.loc[timestamp],
                "joint_condition_at_open": bool(condition.loc[timestamp]),
            }
        )
        held_days += 1
    return pd.DataFrame(rows).set_index("date"), condition


def run_dividend_gate(
    integrated: IntegratedC2Backtest,
    params: DividendGateParams,
) -> DividendGateBacktest:
    result = integrated.result
    calendar = result.inputs.calendar
    slow = trailing_return_at_open(
        result.inputs.risk_close,
        calendar,
        params.lookback,
    )
    primary = defender_primary_target_at_open(result.inputs.defender)
    state, condition = apply_dividend_gate_schedule(
        slow,
        primary,
        calendar,
        params,
    )
    simulated = simulate_switch(
        result.inputs.momentum,
        result.inputs.defender,
        state["risk_on"],
        initial_previous_state="momentum",
    )
    return DividendGateBacktest(params, slow, primary, condition, state, simulated)


def period_metrics(
    returns: pd.Series,
    start: date,
    end: date,
) -> dict[str, float | int | str]:
    sample = returns.loc[pd.Timestamp(start) : pd.Timestamp(end)]
    return performance(sample)


def candidate_record(
    backtest: DividendGateBacktest,
    periods: dict[str, tuple[date, date]],
) -> dict[str, object]:
    returns = backtest.simulated["return"].astype(float)
    record: dict[str, object] = {
        "candidate_id": backtest.params.candidate_id(),
        **asdict(backtest.params),
        "switches": int(backtest.simulated["sleeve_switch"].sum()),
        "defender_entries": int(
            (
                backtest.state["state_changed"].astype(bool)
                & ~backtest.state["risk_on"].astype(bool)
            ).sum()
        ),
        "defender_days": int((~backtest.state["risk_on"].astype(bool)).sum()),
        "joint_condition_days": int(backtest.joint_condition.sum()),
    }
    for label, (start, end) in periods.items():
        metrics = period_metrics(returns, start, end)
        for field in (
            "observations",
            "total_return",
            "annualized_return_252",
            "annualized_volatility",
            "sharpe",
            "max_drawdown",
        ):
            record[f"{label}_{field}"] = metrics[field]
    record["worst_split_sharpe"] = min(
        float(record[f"{label}_sharpe"])
        for label in periods
        if label != "full"
    )
    return record


def search_grid(
    integrated: IntegratedC2Backtest,
    periods: dict[str, tuple[date, date]],
    *,
    lookbacks: Iterable[int],
    slow_thresholds: Iterable[float],
    primary_minimums: Iterable[float],
    min_hold_days: Iterable[int],
    exit_modes: Iterable[str] = EXIT_MODES,
) -> pd.DataFrame:
    """Evaluate the declared joint-gate grid without dynamic parameter mutation."""
    rows: list[dict[str, object]] = []
    for lookback in lookbacks:
        for slow_threshold in slow_thresholds:
            for primary_minimum in primary_minimums:
                for hold_days in min_hold_days:
                    for exit_mode in exit_modes:
                        params = DividendGateParams(
                            lookback=int(lookback),
                            slow_return_threshold=float(slow_threshold),
                            defender_primary_minimum=float(primary_minimum),
                            min_hold_days=int(hold_days),
                            exit_mode=str(exit_mode),
                        )
                        candidate = run_dividend_gate(integrated, params)
                        rows.append(candidate_record(candidate, periods))
    frame = pd.DataFrame(rows)
    if frame["candidate_id"].duplicated().any():
        raise AssertionError("candidate grid produced duplicate IDs")
    return frame


def validate_dividend_gate(backtest: DividendGateBacktest) -> dict[str, object]:
    """Prove every Momentum-to-Defender transition satisfies both entry gates."""
    entries = (
        backtest.state["state_changed"].astype(bool)
        & ~backtest.state["risk_on"].astype(bool)
    )
    invalid_entries = int((~backtest.joint_condition.loc[entries]).sum())
    returns = backtest.simulated["return"].astype(float)
    nav_error = float(
        ((1.0 + returns).cumprod() - backtest.simulated["nav"].astype(float))
        .abs()
        .max()
    )
    if invalid_entries or nav_error > 1e-12:
        raise AssertionError(
            f"dividend gate audit failed: invalid_entries={invalid_entries}, "
            f"nav_error={nav_error:.3e}"
        )
    return {
        "status": "passed",
        "candidate_id": backtest.params.candidate_id(),
        "invalid_defender_entries": invalid_entries,
        "nav_reconstruction_max_abs_error": nav_error,
        "defender_entries": int(entries.sum()),
        "switches": int(backtest.simulated["sleeve_switch"].sum()),
    }
