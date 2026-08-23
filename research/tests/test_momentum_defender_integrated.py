from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from research.momentum_defender_integrated import (
    ALL_ASSETS,
    DEFENDER_STRATEGY_ID,
    composite_target_schedule,
    validate_integrated_result,
)


def _dummy_result():
    calendar = pd.DatetimeIndex(["2026-08-20", "2026-08-21"], name="date")
    momentum = pd.DataFrame(index=calendar)
    for asset in ("510300.SH", "159915.SZ", "513100.SH", "518880.SH"):
        momentum[f"target_weight_{asset}"] = 0.0
    momentum.loc[calendar[0], "target_weight_510300.SH"] = 1.0
    momentum.loc[calendar[1], "target_weight_518880.SH"] = 1.0
    defender = pd.DataFrame(index=calendar)
    for code in ("512890", "159545", "513530", "515080", "510880", "563020", "511260"):
        defender[f"target_weight_{code}"] = 0.0
    defender["target_weight_510880"] = 0.2
    defender["target_weight_511260"] = 0.8
    defender["target_cash_weight"] = 0.0
    defender["strategy_id"] = DEFENDER_STRATEGY_ID
    defender["signal_date"] = pd.DatetimeIndex(["2026-08-19", "2026-08-20"])
    state = pd.DataFrame({"risk_on": [True, False]}, index=calendar)
    simulated = pd.DataFrame(
        {"return": [0.01, -0.005], "nav": [1.01, 1.00495], "sleeve_switch": [False, True]},
        index=calendar,
    )
    inputs = SimpleNamespace(calendar=calendar, momentum=momentum, defender=defender)
    return SimpleNamespace(inputs=inputs, state=state, simulated=simulated)


def test_composite_targets_use_only_the_active_sleeve() -> None:
    result = _dummy_result()
    targets = composite_target_schedule(result)

    assert targets.loc["2026-08-20", "510300.SH"] == pytest.approx(1.0)
    assert targets.loc["2026-08-20", "510880.SH"] == pytest.approx(0.0)
    assert targets.loc["2026-08-21", "510880.SH"] == pytest.approx(0.2)
    assert targets.loc["2026-08-21", "511260.SH"] == pytest.approx(0.8)
    assert targets[list(ALL_ASSETS)].sum(axis=1).to_numpy() == pytest.approx(1.0)


def test_integrated_audit_reconstructs_nav_and_checks_causality() -> None:
    result = _dummy_result()
    targets = composite_target_schedule(result)

    audit = validate_integrated_result(result, targets)

    assert audit["status"] == "passed"
    assert audit["target_sum_max_abs_error"] == pytest.approx(0.0)
    assert audit["nav_reconstruction_max_abs_error"] == pytest.approx(0.0)
    assert audit["signal_timing_causal"] is True
