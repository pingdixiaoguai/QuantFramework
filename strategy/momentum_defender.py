"""Live next-open signal for the integrated Momentum/Defender C2 strategy."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Mapping

import pandas as pd

from data.store import query
from defender.live import DefenderNextOpenTarget, build_next_open_target
from factors.registry import load_registered_factors
from research.momentum_defender_integrated import (
    DEFENDER_STRATEGY_ID,
    INTEGRATED_STRATEGY_ID,
    IntegratedC2Backtest,
    run_integrated_c2,
)
from research.momentum_defender_occam import MOMENTUM_ASSETS
from research.momentum_volatility import (
    asof_previous_close,
    expanding_volatility_cap,
    load_ohlc,
    momentum_asset_at_previous_close,
    rogers_satchell_volatility,
)
from run_daily import _build_signal_snapshot, _compute_factor_values, _load_config
from strategy.rebalance import normalize_rebalance_mode, should_hold_position


@dataclass(frozen=True)
class MomentumNextOpenTarget:
    raw_weights: Mapping[str, float]
    effective_weights: Mapping[str, float]
    held_asset: str
    holding_days: int
    hold_filter_active: bool


@dataclass(frozen=True)
class IntegratedNextOpenSignal:
    strategy_id: str
    defender_strategy_id: str
    signal_date: date
    execution_date: date
    current_model_sleeve: str
    target_sleeve: str
    state_reason: str
    held_days_at_open: int
    slow_gate_return: float
    slow_gate_risk_on: bool
    emergency_asset: str
    emergency_cap: float
    emergency_alert: bool
    momentum: MomentumNextOpenTarget
    defender: DefenderNextOpenTarget
    target_weights: Mapping[str, float]
    target_cash_weight: float


def _momentum_next_open_target(
    root: Path,
    integrated: IntegratedC2Backtest,
    signal_date: date,
) -> MomentumNextOpenTarget:
    config_path = root / "strategy/configs/quality_momentum_top1.yaml"
    config = _load_config(config_path)
    config["end"] = signal_date
    price_data = {
        asset: frame
        for asset in MOMENTUM_ASSETS
        if len(
            frame := query(
                asset,
                config.get("start", date(2013, 7, 1)),
                signal_date,
            )
        )
    }
    factor_values = _compute_factor_values(
        config,
        price_data,
        signal_date,
        load_registered_factors(),
    )
    raw = _build_signal_snapshot(config, factor_values).weights

    result = integrated.result.inputs.momentum_result
    if result.positions.empty:
        return MomentumNextOpenTarget(
            raw_weights=raw,
            effective_weights=raw,
            held_asset=max(raw, key=raw.get),
            holding_days=0,
            hold_filter_active=False,
        )
    positions = result.positions.sort_index().ffill().fillna(0.0)
    latest = positions.iloc[-1]
    current = {
        asset: float(latest.get(asset, 0.0))
        for asset in MOMENTUM_ASSETS
        if float(latest.get(asset, 0.0)) > 1e-14
    }
    entry_date = pd.Timestamp(positions.index[-1])
    holding_days = int(
        result.daily_returns.loc[
            (result.daily_returns.index >= entry_date)
            & (result.daily_returns.index <= pd.Timestamp(signal_date))
        ].shape[0]
    )
    rebalance_days = int(config.get("rebalance_days", 1))
    mode = normalize_rebalance_mode(config.get("rebalance_mode"))
    hold = should_hold_position(current, holding_days, rebalance_days, mode)
    effective = current if hold else raw
    if not effective:
        raise RuntimeError("Momentum produced no next-open target")
    return MomentumNextOpenTarget(
        raw_weights=raw,
        effective_weights=effective,
        held_asset=max(current or effective, key=(current or effective).get),
        holding_days=holding_days,
        hold_filter_active=hold,
    )


def _next_sleeve_state(
    integrated: IntegratedC2Backtest,
    execution_date: date,
) -> tuple[bool, str, int, float, bool, str, float, bool]:
    result = integrated.result
    config = result.config
    current = bool(result.state.iloc[-1]["risk_on"])
    held_days = int(result.state.iloc[-1]["held_days_at_open"]) + 1

    close = result.inputs.risk_close.astype(float).sort_index()
    trailing = close / close.shift(config.slow_lookback) - 1.0
    latest_trailing = float(trailing.dropna().iloc[-1])
    wanted = latest_trailing > config.slow_risk_on_threshold

    calendar = pd.DatetimeIndex([pd.Timestamp(execution_date)])
    previous_asset = str(
        momentum_asset_at_previous_close(
            result.inputs.momentum_result,
            calendar,
        ).iloc[0]
    )
    prices = load_ohlc(previous_asset, integrated.result.inputs.calendar.max().date())
    volatility = rogers_satchell_volatility(prices, config.volatility_window)
    cap_close = expanding_volatility_cap(
        volatility,
        config.asset_quantiles[previous_asset],
        step=config.cap_step,
        min_history=config.quantile_min_history,
    )["cap"]
    selected_cap = float(asof_previous_close(cap_close, calendar).fillna(1.0).iloc[0])
    emergency = selected_cap <= config.trigger_maximum

    target = current
    reason = "hold"
    if emergency:
        if current and (config.emergency_override or held_days >= config.min_hold_days):
            target = False
            reason = "emergency_exit"
        elif current:
            reason = "emergency_blocked_by_min_hold"
        else:
            reason = "emergency_hold"
    elif wanted != current and held_days >= config.min_hold_days:
        target = wanted
        reason = "slow_regime_switch"
    return (
        target,
        reason,
        held_days,
        latest_trailing,
        wanted,
        previous_asset,
        selected_cap,
        emergency,
    )


def build_integrated_next_open_signal(
    root: Path,
    signal_date: date,
    execution_date: date,
) -> IntegratedNextOpenSignal:
    """Replay history, advance one causal state step, and emit exact weights."""
    integrated = run_integrated_c2(root, end=signal_date)
    return build_integrated_next_open_signal_from_backtest(
        root,
        integrated,
        signal_date,
        execution_date,
    )


def build_integrated_next_open_signal_from_backtest(
    root: Path,
    integrated: IntegratedC2Backtest,
    signal_date: date,
    execution_date: date,
) -> IntegratedNextOpenSignal:
    """Advance one next-open C2 signal from an existing historical replay."""
    if integrated.result.inputs.calendar.max().date() != signal_date:
        raise RuntimeError("integrated backtest does not end on signal_date")
    momentum = _momentum_next_open_target(root, integrated, signal_date)
    defender = build_next_open_target(signal_date, execution_date)
    (
        target_risk_on,
        reason,
        held_days,
        slow_return,
        slow_wanted,
        emergency_asset,
        emergency_cap,
        emergency,
    ) = _next_sleeve_state(integrated, execution_date)

    target_weights = dict(
        momentum.effective_weights if target_risk_on else defender.target_weights
    )
    target_cash = 0.0 if target_risk_on else defender.target_cash_weight
    if abs(sum(target_weights.values()) + target_cash - 1.0) > 1e-12:
        raise AssertionError("composite next-open target plus cash must sum to one")
    current_sleeve = (
        "momentum" if bool(integrated.result.state.iloc[-1]["risk_on"]) else "defender"
    )
    return IntegratedNextOpenSignal(
        strategy_id=INTEGRATED_STRATEGY_ID,
        defender_strategy_id=DEFENDER_STRATEGY_ID,
        signal_date=signal_date,
        execution_date=execution_date,
        current_model_sleeve=current_sleeve,
        target_sleeve="momentum" if target_risk_on else "defender",
        state_reason=reason,
        held_days_at_open=held_days,
        slow_gate_return=slow_return,
        slow_gate_risk_on=slow_wanted,
        emergency_asset=emergency_asset,
        emergency_cap=emergency_cap,
        emergency_alert=emergency,
        momentum=momentum,
        defender=defender,
        target_weights=target_weights,
        target_cash_weight=target_cash,
    )
