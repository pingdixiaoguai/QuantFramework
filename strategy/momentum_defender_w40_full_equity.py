"""Formal W40 Momentum gate with monthly reversal and 100% dividend Defender."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Mapping

import numpy as np
import pandas as pd

from backtest.runner import run as run_backtest
from defender.relative_defender_rotation import DEFENSIVE_ASSET
from defender.w40_reversal_full_equity import (
    FORMAL_DIVIDEND_ASSETS,
    FORMAL_DEFENDER_STRATEGY_ID,
    FormalFullEquityDefenderBacktest,
    build_formal_backtest as build_formal_defender,
    build_next_open_target as build_defender_next_open_target,
)
from research.defender_curve_momentum import DEFENDER_CANDIDATE
from research.momentum_defender_downside_raqm import build_exact_execution_data
from research.defender_curve_momentum import _single_etf_interface
from research.momentum_defender_gold_override import GoldOverrideContext, simulate_candidate_schedule
from research.momentum_defender_occam import (
    HELD_RETURN,
    MOMENTUM_ASSETS,
    ResearchInputs,
    _load_momentum_config,
    _load_prices,
    _momentum_target_schedule,
    build_momentum_interface,
    performance,
)
from research.momentum_defender_w40_loss_gate import (
    run_w40_loss_gate,
    w40_loss_percentile_at_open,
)
from research.momentum_volatility import load_ohlc
from strategy.momentum_defender import MomentumNextOpenTarget, _momentum_next_open_target
from strategy.momentum_defender_w40_loss import (
    ANCHOR_ASSET,
    DEFENDER_ENTRY_CONFIRMATION_DAYS,
    DEFENDER_ENTRY_PERCENTILE,
    DEFENDER_LOCK_DAYS,
    HISTORY_WINDOW,
    MIN_HISTORY,
    MOMENTUM_LOCK_DAYS,
    MOMENTUM_RECOVERY_CONFIRMATION_DAYS,
    MOMENTUM_RECOVERY_PERCENTILE,
    FormalW40LossNextOpenSignal,
    formal_spec,
)


FORMAL_STRATEGY_ID = "momentum_defender_w40_reversal_full_equity_v2"
FORMAL_BACKTEST_START = date(2013, 1, 1)
MOMENTUM_INTERFACE_PARITY_TOLERANCE = 2e-5


@dataclass(frozen=True)
class FormalW40FullEquityBacktest:
    context: GoldOverrideContext
    defender: FormalFullEquityDefenderBacktest
    raw_loss_at_open: pd.Series
    score_at_open: pd.Series
    state: pd.DataFrame
    daily: pd.DataFrame
    audit: Mapping[str, object]


def _features(
    calendar: pd.DatetimeIndex,
    *,
    end: date,
) -> tuple[pd.Series, pd.Series]:
    close = load_ohlc(ANCHOR_ASSET, end)["close"]
    return w40_loss_percentile_at_open(
        close,
        calendar,
        history_window=HISTORY_WINDOW,
        min_history=MIN_HISTORY,
    )


def build_formal_context(
    root: Path,
    *,
    end: date,
    start: date = FORMAL_BACKTEST_START,
    defender_builder: Callable[..., FormalFullEquityDefenderBacktest] = (
        build_formal_defender
    ),
    defender_strategy_id: str = FORMAL_DEFENDER_STRATEGY_ID,
) -> tuple[GoldOverrideContext, FormalFullEquityDefenderBacktest]:
    momentum_config = _load_momentum_config(
        root / "strategy/configs/quality_momentum_top1.yaml", end
    )
    momentum_config["start"] = FORMAL_BACKTEST_START
    momentum_result = run_backtest(momentum_config)
    full_momentum_calendar = pd.DatetimeIndex(momentum_result.daily_returns.index)
    calendar = full_momentum_calendar[
        full_momentum_calendar >= pd.Timestamp(start)
    ]
    if calendar.empty:
        raise ValueError("formal evaluation calendar is empty")
    defender = defender_builder(calendar, end=end)

    prior_dates = full_momentum_calendar[full_momentum_calendar < calendar.min()]
    replay_calendar = (
        pd.DatetimeIndex([prior_dates.max()]).append(calendar)
        if len(prior_dates)
        else calendar
    )
    replay_targets = _momentum_target_schedule(momentum_result, replay_calendar)
    targets = replay_targets.reindex(calendar)
    prices = _load_prices(MOMENTUM_ASSETS, FORMAL_BACKTEST_START, end)
    momentum_interface = build_momentum_interface(
        replay_targets, prices
    ).reindex(calendar)
    parity = float(
        momentum_interface[HELD_RETURN]
        .sub(momentum_result.daily_returns.reindex(calendar))
        .abs()
        .max()
    )
    if parity > MOMENTUM_INTERFACE_PARITY_TOLERANCE:
        raise AssertionError(f"formal Momentum interface parity failed: {parity:.3e}")

    interfaces: dict[str, pd.DataFrame] = {
        DEFENDER_CANDIDATE: defender.interface,
    }
    curves: dict[str, pd.Series] = {
        DEFENDER_CANDIDATE: defender.interface["nav_if_held"].astype(float),
    }
    for asset in MOMENTUM_ASSETS:
        interface, close = _single_etf_interface(asset, calendar, end)
        interfaces[asset] = interface
        curves[asset] = close
    curve_frame = pd.DataFrame(curves, index=calendar)

    momentum_target = targets.idxmax(axis=1).astype(str)
    momentum_target.name = "momentum_target_at_open"
    defender_input = defender.interface.copy()
    for asset in (*FORMAL_DIVIDEND_ASSETS, DEFENSIVE_ASSET):
        code = asset.split(".", maxsplit=1)[0]
        defender_input[f"target_weight_{code}"] = defender.targets[asset]
    defender_input["strategy_id"] = defender_strategy_id
    risk_close = prices["510300.SH"]["close"].astype(float)
    inputs = ResearchInputs(
        calendar=calendar,
        momentum=momentum_interface,
        defender=defender_input,
        risk_close=risk_close,
        momentum_result=momentum_result,
    )
    placeholder_state = pd.DataFrame({"risk_on": True}, index=calendar)
    placeholder_simulated = pd.DataFrame(
        {
            "return": momentum_interface[HELD_RETURN].astype(float),
            "nav": (1.0 + momentum_interface[HELD_RETURN].astype(float)).cumprod(),
        },
        index=calendar,
    )
    integrated = SimpleNamespace(
        result=SimpleNamespace(
            inputs=inputs,
            state=placeholder_state,
            simulated=placeholder_simulated,
            previous_asset=momentum_target,
        )
    )
    context = GoldOverrideContext(
        integrated=integrated,
        calendar=calendar,
        curves=curve_frame,
        interfaces=interfaces,
        momentum_target=momentum_target,
        baseline_target=momentum_target.rename("baseline_target_at_open"),
        initial_previous_candidate=str(
            replay_targets.iloc[0].idxmax()
        ),
        baseline_parity_max_abs_error=parity,
    )
    return context, defender


def run_formal_strategy(
    root: Path,
    *,
    end: date,
    start: date = FORMAL_BACKTEST_START,
) -> FormalW40FullEquityBacktest:
    """Replay the promoted W40/full-equity rule through ``end``."""
    context, defender = build_formal_context(root, end=end, start=start)
    data = build_exact_execution_data(context)
    raw_loss, score = _features(context.calendar, end=end)
    run = run_w40_loss_gate(data, score, formal_spec())
    requested = pd.Series(
        [data.candidates[value] for value in run.requested_target],
        index=data.calendar,
    )
    daily = simulate_candidate_schedule(
        requested, context.interfaces, context.initial_previous_candidate
    )
    parity = float(
        np.max(np.abs(daily["return"].to_numpy(float) - run.returns))
    )
    audit = {
        "status": "passed",
        "strategy_id": FORMAL_STRATEGY_ID,
        "defender_strategy_id": FORMAL_DEFENDER_STRATEGY_ID,
        "defender_entries": run.defender_entries,
        "defender_days": run.defender_days,
        "sleeve_switches": run.sleeve_switches,
        "candidate_switches": run.candidate_switches,
        "dense_exact_return_parity_max_abs_error": parity,
        "nav_reconstruction_max_abs_error": float(
            ((1.0 + daily["return"]).cumprod() - daily["nav"]).abs().max()
        ),
        "performance": performance(daily["return"].astype(float)),
        "defender_audit": defender.audit,
    }
    if parity > 1e-14 or audit["nav_reconstruction_max_abs_error"] > 1e-12:
        raise AssertionError("formal W40/full-equity execution parity failed")
    return FormalW40FullEquityBacktest(
        context=context,
        defender=defender,
        raw_loss_at_open=raw_loss,
        score_at_open=score,
        state=run.state,
        daily=daily,
        audit=audit,
    )


def build_next_open_signal(
    root: Path,
    signal_date: date,
    execution_date: date,
) -> FormalW40LossNextOpenSignal:
    """Replay history and advance the promoted state one future open."""
    if execution_date <= signal_date:
        raise ValueError("execution date must follow signal date")
    historical = run_formal_strategy(root, end=signal_date)
    execution = pd.Timestamp(execution_date)
    calendar = historical.context.calendar.append(pd.DatetimeIndex([execution]))
    raw_loss, score = _features(calendar, end=signal_date)
    from research.momentum_defender_downside_raqm import downside_raqm_state_schedule

    state = downside_raqm_state_schedule(score, formal_spec().state_spec())
    momentum = _momentum_next_open_target(
        root, historical.context.integrated, signal_date
    )
    defender = build_defender_next_open_target(signal_date, execution_date)
    target_risk_on = bool(state.at[execution, "risk_on"])
    target_weights = dict(
        momentum.effective_weights if target_risk_on else defender.target_weights
    )
    target_cash = 0.0 if target_risk_on else defender.target_cash_weight
    if abs(sum(target_weights.values()) + target_cash - 1.0) > 1e-12:
        raise AssertionError("formal next-open target plus cash must sum to one")
    values = (raw_loss.loc[execution], score.loc[execution])
    if not all(np.isfinite(value) for value in values):
        raise RuntimeError("formal W40 next-open factor is unavailable")
    current_risk_on = bool(historical.state.iloc[-1]["risk_on"])
    return FormalW40LossNextOpenSignal(
        strategy_id=FORMAL_STRATEGY_ID,
        defender_strategy_id=FORMAL_DEFENDER_STRATEGY_ID,
        signal_date=signal_date,
        execution_date=execution_date,
        current_model_sleeve="momentum" if current_risk_on else "defender",
        target_sleeve="momentum" if target_risk_on else "defender",
        state_reason=str(state.at[execution, "state_reason"]),
        held_days_at_open=int(state.at[execution, "held_days_at_open"]),
        w40_downside_log_loss=float(raw_loss.loc[execution]),
        w40_loss_percentile=float(score.loc[execution]),
        defender_entry_percentile=DEFENDER_ENTRY_PERCENTILE,
        momentum_recovery_percentile=MOMENTUM_RECOVERY_PERCENTILE,
        entry_confirmation_streak=int(
            state.at[execution, "entry_confirmation_streak"]
        ),
        recovery_confirmation_streak=int(
            state.at[execution, "recovery_confirmation_streak"]
        ),
        defender_entry_confirmation_days=DEFENDER_ENTRY_CONFIRMATION_DAYS,
        momentum_recovery_confirmation_days=MOMENTUM_RECOVERY_CONFIRMATION_DAYS,
        momentum_lock_days=MOMENTUM_LOCK_DAYS,
        defender_lock_days=DEFENDER_LOCK_DAYS,
        momentum=momentum,
        defender=defender,
        target_weights=target_weights,
        target_cash_weight=target_cash,
    )
