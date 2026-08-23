"""Integrated Momentum/Defender C2 using the vendored Defender main code.

Unlike the historical frozen-v2 checkpoint, this module never reads an
external Defender deliverable.  It builds the exact open-switch interface in
memory from QuantFramework's own HFQ data and the pinned upstream algorithm.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from defender.relative_defender_rotation import DEFENSIVE_ASSET, ROTATION_ASSETS
from defender.relative_defender_rotation_2013_export import (
    STRATEGY_ID as DEFENDER_STRATEGY_ID,
    build_switch_return_frame,
)
from research.momentum_defender_c2 import (
    DEFAULT_CONFIG_PATH,
    FrozenC2Backtest,
    FrozenC2Config,
    load_frozen_c2_config,
    run_c2_with_inputs,
)
from research.momentum_defender_occam import (
    MOMENTUM_ASSETS,
    build_inputs_from_defender_interface,
    performance,
)


UPSTREAM_COMMIT = "b5e34191a7d445521de330e998bfe0804d6ebd43"
INTEGRATED_STRATEGY_ID = "momentum_defender_c2_defender_main_b5e3419"
DEFAULT_START = date(2019, 1, 18)
DEFENDER_ASSETS = (*ROTATION_ASSETS, DEFENSIVE_ASSET)
ALL_ASSETS = (*MOMENTUM_ASSETS, *DEFENDER_ASSETS)


@dataclass(frozen=True)
class IntegratedC2Backtest:
    result: FrozenC2Backtest
    targets: pd.DataFrame
    daily: pd.DataFrame
    defender_metrics: dict[str, object]
    audit: dict[str, object]


def integrated_config(
    path: Path = DEFAULT_CONFIG_PATH,
) -> FrozenC2Config:
    """Reuse the frozen C2 gate/cap parameters with the new Defender sleeve."""
    frozen = load_frozen_c2_config(path)
    return replace(
        frozen,
        strategy_id=INTEGRATED_STRATEGY_ID,
        status="integrated_production_candidate",
    )


def _target_column(asset: str) -> str:
    return f"target_weight_{asset.split('.', maxsplit=1)[0]}"


def composite_target_schedule(result: FrozenC2Backtest) -> pd.DataFrame:
    """Return the executable open target for the active sleeve on every day."""
    calendar = result.inputs.calendar
    risk_on = result.state["risk_on"].astype(bool)
    targets = pd.DataFrame(0.0, index=calendar, columns=list(ALL_ASSETS))

    for asset in MOMENTUM_ASSETS:
        column = f"target_weight_{asset}"
        targets.loc[risk_on, asset] = result.inputs.momentum.loc[
            risk_on, column
        ].astype(float)
    for asset in DEFENDER_ASSETS:
        targets.loc[~risk_on, asset] = result.inputs.defender.loc[
            ~risk_on, _target_column(asset)
        ].astype(float)

    cash = pd.Series(0.0, index=calendar, name="target_cash_weight")
    cash.loc[~risk_on] = result.inputs.defender.loc[
        ~risk_on, "target_cash_weight"
    ].astype(float)
    total = targets.sum(axis=1) + cash
    if not np.allclose(total, 1.0, atol=1e-12):
        bad = total.loc[~np.isclose(total, 1.0, atol=1e-12)]
        raise AssertionError(
            "integrated target plus cash does not sum to one: "
            f"{bad.head().to_dict()}"
        )
    targets["target_cash_weight"] = cash
    targets.index.name = "date"
    return targets


def validate_integrated_result(
    result: FrozenC2Backtest,
    targets: pd.DataFrame,
) -> dict[str, object]:
    """Mechanically validate the sleeve composition and causal handoff."""
    calendar = result.inputs.calendar
    if not (
        calendar.equals(result.inputs.momentum.index)
        and calendar.equals(result.inputs.defender.index)
        and calendar.equals(result.state.index)
        and calendar.equals(result.simulated.index)
        and calendar.equals(targets.index)
    ):
        raise AssertionError("integrated calendars are not identical")

    strategy_ids = set(
        result.inputs.defender["strategy_id"].dropna().astype(str)
    )
    if strategy_ids != {DEFENDER_STRATEGY_ID}:
        raise AssertionError(
            f"unexpected Defender strategy IDs: {sorted(strategy_ids)}"
        )

    target_error = float(
        (
            targets[list(ALL_ASSETS)].sum(axis=1)
            + targets["target_cash_weight"]
            - 1.0
        ).abs().max()
    )
    returns = result.simulated["return"].astype(float)
    if not np.isfinite(returns.to_numpy()).all() or returns.le(-1.0).any():
        raise AssertionError("integrated daily returns contain invalid values")

    reconstructed_nav = (1.0 + returns).cumprod()
    nav_error = float(
        (reconstructed_nav - result.simulated["nav"].astype(float)).abs().max()
    )
    signal_dates = pd.to_datetime(result.inputs.defender["signal_date"])
    causal = bool((signal_dates.dropna() < signal_dates.dropna().index).all())
    if target_error > 1e-12 or nav_error > 1e-12 or not causal:
        raise AssertionError(
            "integrated audit failed: "
            f"target_error={target_error:.3e}, nav_error={nav_error:.3e}, "
            f"causal={causal}"
        )

    return {
        "status": "passed",
        "strategy_id": INTEGRATED_STRATEGY_ID,
        "defender_strategy_id": DEFENDER_STRATEGY_ID,
        "defender_upstream_commit": UPSTREAM_COMMIT,
        "observations": int(len(calendar)),
        "start": calendar.min().date().isoformat(),
        "end": calendar.max().date().isoformat(),
        "target_sum_max_abs_error": target_error,
        "nav_reconstruction_max_abs_error": nav_error,
        "signal_timing_causal": causal,
        "sleeve_switches": int(result.simulated["sleeve_switch"].sum()),
        "defender_days": int((~result.state["risk_on"].astype(bool)).sum()),
        "performance": performance(returns),
    }


def run_integrated_c2(
    root: Path,
    *,
    end: date | None = None,
    start: date = DEFAULT_START,
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> IntegratedC2Backtest:
    """Run C2 with an in-memory interface built by the vendored Defender."""
    switch, defender_metrics = build_switch_return_frame()
    available_end = switch.index.max().date()
    cutoff = available_end if end is None else min(end, available_end)
    config = integrated_config(config_path)
    inputs = build_inputs_from_defender_interface(
        root,
        switch,
        cutoff,
        start=start,
    )
    result = run_c2_with_inputs(config, inputs)
    targets = composite_target_schedule(result)
    audit = validate_integrated_result(result, targets)
    daily = result.daily.join(
        targets.rename(columns={asset: f"target_{asset}" for asset in ALL_ASSETS})
    )
    daily["defender_strategy_id"] = DEFENDER_STRATEGY_ID
    daily["defender_upstream_commit"] = UPSTREAM_COMMIT
    daily.index.name = "date"
    return IntegratedC2Backtest(
        result=result,
        targets=targets,
        daily=daily,
        defender_metrics=dict(defender_metrics),
        audit=audit,
    )
