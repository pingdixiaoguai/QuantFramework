"""Regression coverage for live signals that span a market holiday."""

from __future__ import annotations

from datetime import date

import pandas as pd

from execution.interfaces import diff
from execution.position import PositionPeriod, PositionState
from run_daily import (
    _is_unpriced_pending_target,
    _next_entry_date,
    _priced_state_as_of,
    _save_or_update_rebalance_target,
)


def _pending_switch() -> PositionState:
    return PositionState(
        weights={"159915.SZ": 1.0},
        entry_date="2026-06-19",  # holiday; originally scheduled incorrectly
        entry_prices=None,
        ytd_history=[
            PositionPeriod(
                weights={"513100.SH": 1.0},
                entry_date="2026-05-12",
                exit_date="2026-06-19",
                entry_prices={"513100.SH": 10.0},
                exit_prices=None,
            )
        ],
    )


def test_next_entry_date_uses_exchange_calendar_not_next_calendar_day(monkeypatch):
    class FakePro:
        def trade_cal(self, **kwargs):
            assert kwargs["exchange"] == "SSE"
            assert kwargs["start_date"] == "20260619"
            assert kwargs["is_open"] == "1"
            return pd.DataFrame({"cal_date": ["20260622"]})

    monkeypatch.setattr("run_daily.get_tushare_token", lambda: "token")
    monkeypatch.setattr("run_daily.ts.pro_api", lambda token: FakePro())

    assert _next_entry_date(date(2026, 6, 18)) == date(2026, 6, 22)


def test_unpriced_target_uses_outgoing_holding_for_repeat_signal():
    pending = _pending_switch()

    actual = _priced_state_as_of(pending, date(2026, 6, 18))
    orders = diff({"159915.SZ": 1.0}, actual.weights)

    assert actual.weights == {"513100.SH": 1.0}
    assert {order.action for order in orders} == {"buy", "sell"}


def test_repeat_signal_updates_pending_target_without_duplicating_history(monkeypatch):
    pending = _pending_switch()
    saved: list[PositionState] = []

    monkeypatch.setattr("run_daily.write_position", lambda state, _: saved.append(state))
    monkeypatch.setattr(
        "run_daily.save_position",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not append history")),
    )

    updated = _save_or_update_rebalance_target(
        pending,
        {"159915.SZ": 1.0},
        date(2026, 6, 22),
        date(2026, 6, 18),
        "quality_momentum_top1",
    )

    assert updated is True
    assert _is_unpriced_pending_target(pending, date(2026, 6, 18)) is True
    assert len(saved) == 1
    assert saved[0].entry_date == "2026-06-22"
    assert len(saved[0].ytd_history) == 1
    assert saved[0].ytd_history[0].exit_date == "2026-06-22"
