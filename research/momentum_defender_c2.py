"""Frozen Momentum/Defender C2 strategy definition and causal simulator.

This module is the single implementation source for the current research
branch candidate.  Parameter search and ablation scripts remain historical
evidence and must not supply parameters to this module at runtime.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
import yaml

from research.momentum_defender_occam import (
    HELD_RETURN,
    MOMENTUM_ASSETS,
    OccamParams,
    ResearchInputs,
    apply_state_schedule,
    build_inputs,
    performance,
    simulate_switch,
    slow_regime_at_open,
)
from research.momentum_volatility import (
    asof_previous_close,
    choose_by_asset,
    expanding_volatility_cap,
    load_ohlc,
    momentum_asset_at_previous_close,
    rogers_satchell_volatility,
)


DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parent
    / "configs"
    / "momentum_defender_c2_frozen_v2.yaml"
)


@dataclass(frozen=True)
class FrozenC2Config:
    strategy_id: str
    status: str
    frozen_on: date
    research_cutoff: date
    momentum_config_path: Path
    defender_deliverable_dir: Path
    defender_switch_returns_file: str
    slow_anchor_asset: str
    slow_lookback: int
    slow_risk_on_threshold: float
    volatility_estimator: str
    volatility_window: int
    annualization: int
    quantile_history: str
    quantile_min_history: int
    cap_step: float
    trigger_operator: str
    trigger_maximum: float
    asset_quantiles: Mapping[str, float]
    signal_timing: str
    min_hold_days: int
    emergency_override: bool
    initial_previous_sleeve: str
    checkpoint: Mapping[str, object]

    def variant_id(self) -> str:
        quantiles = self.asset_quantiles
        return (
            f"C2_vw{self.volatility_window}_cap{self.trigger_maximum:.1f}_"
            f"qc300{quantiles['510300.SH']:.2f}_"
            f"qcyb{quantiles['159915.SZ']:.2f}_"
            f"qndx{quantiles['513100.SH']:.2f}_"
            f"qau{quantiles['518880.SH']:.2f}"
        )

    def slow_params(self) -> OccamParams:
        return OccamParams(
            lookback=self.slow_lookback,
            risk_on_threshold=self.slow_risk_on_threshold,
            min_hold_days=self.min_hold_days,
            emergency_daily_loss=None,
        )

    def serializable(self) -> dict[str, object]:
        values = asdict(self)
        values["frozen_on"] = self.frozen_on.isoformat()
        values["research_cutoff"] = self.research_cutoff.isoformat()
        values["momentum_config_path"] = str(self.momentum_config_path)
        values["defender_deliverable_dir"] = str(self.defender_deliverable_dir)
        values["asset_quantiles"] = dict(self.asset_quantiles)
        values["checkpoint"] = dict(self.checkpoint)
        values["variant_id"] = self.variant_id()
        return values


@dataclass(frozen=True)
class FrozenC2Backtest:
    config: FrozenC2Config
    inputs: ResearchInputs
    previous_asset: pd.Series
    caps_by_asset: Mapping[str, pd.Series]
    selected_cap: pd.Series
    emergency_alert: pd.Series
    slow_signal: pd.Series
    state: pd.DataFrame
    simulated: pd.DataFrame
    daily: pd.DataFrame


def _mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a mapping")
    return value


def load_frozen_c2_config(path: Path = DEFAULT_CONFIG_PATH) -> FrozenC2Config:
    with path.open(encoding="utf-8") as handle:
        raw = _mapping(yaml.safe_load(handle), "config")
    strategy = _mapping(raw.get("strategy"), "strategy")
    momentum = _mapping(raw.get("momentum"), "momentum")
    defender = _mapping(raw.get("defender"), "defender")
    slow = _mapping(raw.get("slow_gate"), "slow_gate")
    cap = _mapping(raw.get("emergency_cap"), "emergency_cap")
    execution = _mapping(raw.get("execution"), "execution")
    checkpoint = _mapping(raw.get("checkpoint"), "checkpoint")
    quantiles_raw = _mapping(cap.get("asset_quantiles"), "asset_quantiles")
    quantiles = {str(asset): float(value) for asset, value in quantiles_raw.items()}

    config = FrozenC2Config(
        strategy_id=str(strategy["id"]),
        status=str(strategy["status"]),
        frozen_on=date.fromisoformat(str(strategy["frozen_on"])),
        research_cutoff=date.fromisoformat(str(strategy["research_cutoff"])),
        momentum_config_path=Path(str(momentum["config_path"])),
        defender_deliverable_dir=Path(str(defender["deliverable_dir"])),
        defender_switch_returns_file=str(defender["switch_returns_file"]),
        slow_anchor_asset=str(slow["anchor_asset"]),
        slow_lookback=int(slow["lookback"]),
        slow_risk_on_threshold=float(slow["risk_on_threshold"]),
        volatility_estimator=str(cap["estimator"]),
        volatility_window=int(cap["volatility_window"]),
        annualization=int(cap["annualization"]),
        quantile_history=str(cap["quantile_history"]),
        quantile_min_history=int(cap["quantile_min_history"]),
        cap_step=float(cap["cap_step"]),
        trigger_operator=str(cap["trigger_operator"]),
        trigger_maximum=float(cap["trigger_maximum"]),
        asset_quantiles=quantiles,
        signal_timing=str(execution["signal_timing"]),
        min_hold_days=int(execution["min_hold_days"]),
        emergency_override=bool(execution["emergency_override"]),
        initial_previous_sleeve=str(execution["initial_previous_sleeve"]),
        checkpoint=dict(checkpoint),
    )
    validate_frozen_c2_config(config)
    return config


def validate_frozen_c2_config(config: FrozenC2Config) -> None:
    if set(config.asset_quantiles) != set(MOMENTUM_ASSETS):
        raise ValueError("asset_quantiles must contain exactly the four Momentum ETFs")
    if any(not 0.0 < value < 1.0 for value in config.asset_quantiles.values()):
        raise ValueError("all asset quantiles must be strictly between zero and one")
    if config.momentum_config_path != Path(
        "strategy/configs/quality_momentum_top1.yaml"
    ):
        raise ValueError("frozen C2 requires quality_momentum_top1.yaml")
    if config.slow_anchor_asset != "510300.SH":
        raise ValueError("frozen C2 slow gate must use 510300.SH")
    if config.volatility_estimator != "rogers_satchell":
        raise ValueError("frozen C2 requires Rogers-Satchell volatility")
    if config.annualization != 252:
        raise ValueError("frozen C2 volatility annualization must be 252")
    if config.quantile_history != "expanding_all_available_strict_lag":
        raise ValueError("frozen C2 requires strict-lag expanding quantiles")
    if config.trigger_operator != "<=":
        raise ValueError("frozen C2 cap trigger operator must be <=")
    if config.signal_timing != "previous_close_to_next_open":
        raise ValueError("frozen C2 signal timing must be previous close to next open")
    if config.initial_previous_sleeve != "momentum":
        raise ValueError("frozen C2 initial previous sleeve must be momentum")
    if config.min_hold_days < 1 or config.volatility_window < 2:
        raise ValueError("invalid frozen C2 lookback or holding period")


def held_asset_cap_alert(
    caps_by_asset: Mapping[str, pd.Series],
    previous_asset: pd.Series,
    trigger_maximum: float,
) -> pd.Series:
    """Use only the cap of the Momentum ETF owned through prior close."""
    missing = set(MOMENTUM_ASSETS) - set(caps_by_asset)
    if missing:
        raise ValueError(f"missing cap series for {sorted(missing)}")
    alert = pd.Series(False, index=previous_asset.index)
    for asset in MOMENTUM_ASSETS:
        held = previous_asset.eq(asset)
        cap = caps_by_asset[asset].reindex(alert.index)
        if cap.loc[held].isna().any():
            raise ValueError(f"cap series for {asset} is missing held dates")
        alert.loc[held] = cap.loc[held].le(trigger_maximum)
    alert.name = "frozen_c2_emergency_alert_at_open"
    return alert.astype(bool)


def run_frozen_c2(
    root: Path,
    config: FrozenC2Config,
    *,
    defender_dir: Path | None = None,
    end: date | None = None,
) -> FrozenC2Backtest:
    cutoff = end or config.research_cutoff
    deliverable = defender_dir or config.defender_deliverable_dir
    inputs = build_inputs(
        root,
        deliverable / config.defender_switch_returns_file,
        cutoff,
    )
    return run_c2_with_inputs(config, inputs)


def run_c2_with_inputs(
    config: FrozenC2Config,
    inputs: ResearchInputs,
) -> FrozenC2Backtest:
    """Run the frozen C2 state machine against already-aligned sleeves.

    Keeping the signal/state logic here lets the historical frozen checkpoint
    retain its CSV input while the integrated production candidate supplies an
    in-memory interface from the vendored Defender implementation.
    """
    cutoff = inputs.calendar.max().date()
    calendar = inputs.calendar
    previous_asset = momentum_asset_at_previous_close(
        inputs.momentum_result,
        calendar,
    )
    caps_by_asset: dict[str, pd.Series] = {}
    for asset in MOMENTUM_ASSETS:
        prices = load_ohlc(asset, cutoff)
        volatility = rogers_satchell_volatility(prices, config.volatility_window)
        close_cap = expanding_volatility_cap(
            volatility,
            config.asset_quantiles[asset],
            step=config.cap_step,
            min_history=config.quantile_min_history,
        )["cap"]
        caps_by_asset[asset] = asof_previous_close(close_cap, calendar).fillna(1.0)

    selected_cap = choose_by_asset(caps_by_asset, previous_asset).rename(
        "frozen_c2_selected_cap_at_open"
    )
    emergency = held_asset_cap_alert(
        caps_by_asset,
        previous_asset,
        config.trigger_maximum,
    )
    slow = slow_regime_at_open(
        inputs.risk_close,
        calendar,
        config.slow_lookback,
        config.slow_risk_on_threshold,
    )
    state = apply_state_schedule(
        slow,
        emergency,
        calendar,
        config.min_hold_days,
        emergency_override=config.emergency_override,
        initial_risk_on=True,
    )
    simulated = simulate_switch(
        inputs.momentum,
        inputs.defender,
        state["risk_on"],
        initial_previous_state=config.initial_previous_sleeve,
    )
    daily = state.join(simulated.drop(columns=["risk_on"]))
    daily["emergency_alert"] = emergency
    daily["selected_cap"] = selected_cap
    daily["momentum_asset_at_previous_close"] = previous_asset
    daily["momentum_exact_return"] = inputs.momentum[HELD_RETURN].astype(float)
    daily["strategy_id"] = config.strategy_id
    daily["variant_id"] = config.variant_id()
    daily.index.name = "date"
    return FrozenC2Backtest(
        config=config,
        inputs=inputs,
        previous_asset=previous_asset,
        caps_by_asset=caps_by_asset,
        selected_cap=selected_cap,
        emergency_alert=emergency,
        slow_signal=slow,
        state=state,
        simulated=simulated,
        daily=daily,
    )


def daily_return_sha256(returns: pd.Series) -> str:
    values = returns.to_numpy(dtype="<f8")
    return hashlib.sha256(values.tobytes()).hexdigest()


def validate_frozen_checkpoint(result: FrozenC2Backtest) -> dict[str, object]:
    config = result.config
    expected = config.checkpoint
    measured = performance(result.simulated["return"])
    emergency_entries = (
        result.state["state_changed"].astype(bool)
        & result.state["state_reason"].eq("emergency_exit")
    )
    actual: dict[str, object] = {
        **measured,
        "alert_days": int(result.emergency_alert.sum()),
        "emergency_entries": int(emergency_entries.sum()),
        "defender_days": int((~result.state["risk_on"]).sum()),
        "sleeve_switches": int(result.simulated["sleeve_switch"].sum()),
        "daily_return_sha256_float64_le": daily_return_sha256(
            result.simulated["return"]
        ),
    }
    exact_fields = {
        "start",
        "end",
        "observations",
        "alert_days",
        "emergency_entries",
        "defender_days",
        "sleeve_switches",
        "daily_return_sha256_float64_le",
    }
    numeric_fields = {
        "total_return",
        "annualized_return_252",
        "annualized_volatility",
        "sharpe",
        "max_drawdown",
    }
    failures: list[str] = []
    for field in exact_fields:
        if actual[field] != expected[field]:
            failures.append(f"{field}: {actual[field]!r} != {expected[field]!r}")
    for field in numeric_fields:
        error = abs(float(actual[field]) - float(expected[field]))
        if error > 1e-12:
            failures.append(f"{field}: abs error {error:.3e}")
    if failures:
        raise AssertionError("frozen C2 checkpoint mismatch: " + "; ".join(failures))
    return {
        "status": "passed",
        "strategy_id": config.strategy_id,
        "variant_id": config.variant_id(),
        "tolerance": 1e-12,
        "actual": actual,
        "expected": dict(expected),
    }
