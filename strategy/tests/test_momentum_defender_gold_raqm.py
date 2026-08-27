from __future__ import annotations

from datetime import date
from types import SimpleNamespace

import pandas as pd

from strategy.momentum_defender_gold_raqm import (
    _next_open_metrics,
    advance_gold_override,
)


def test_hard_five_day_hold_overrides_base_momentum() -> None:
    step = advance_gold_override(
        current_active=True,
        completed_gold_days=4,
        base_next_risk_on=True,
        metric_difference=-10.0,
    )

    assert step.active
    assert step.reason == "gold_hard_min_hold"


def test_after_five_days_base_momentum_takes_precedence() -> None:
    step = advance_gold_override(
        current_active=True,
        completed_gold_days=5,
        base_next_risk_on=True,
        metric_difference=10.0,
    )

    assert not step.active
    assert step.reason == "gold_to_momentum_after_min_hold"


def test_defender_entry_and_exit_use_frozen_thresholds() -> None:
    entered = advance_gold_override(
        current_active=False,
        completed_gold_days=0,
        base_next_risk_on=False,
        metric_difference=2.01,
    )
    exited = advance_gold_override(
        current_active=True,
        completed_gold_days=5,
        base_next_risk_on=False,
        metric_difference=0.75,
    )

    assert entered.active and entered.reason == "gold_entry"
    assert not exited.active and exited.reason == "gold_to_defender_after_min_hold"


def test_next_open_metrics_explicitly_use_five_day_window(monkeypatch) -> None:
    index = pd.DatetimeIndex(["2026-08-20", "2026-08-21"])
    context = SimpleNamespace(
        curves=pd.DataFrame(
            {"518880.SH": [1.0, 1.1], "DEFENDER": [1.0, 1.01]},
            index=index,
        )
    )
    captured = {}

    def fake_metric(curves):
        captured["called"] = True
        return pd.DataFrame(
            {"518880.SH": 1.0, "DEFENDER": 0.5, "difference": 0.5},
            index=curves.index,
        )

    monkeypatch.setattr(
        "strategy.momentum_defender_gold_raqm.raw_gold_metrics_at_open",
        fake_metric,
    )

    result = _next_open_metrics(context, date(2026, 8, 24))

    assert captured["called"]
    assert result["difference"] == 0.5
