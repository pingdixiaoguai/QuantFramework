"""Regression coverage for live signals that span a market holiday."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from execution.interfaces import diff
from execution.position import PositionPeriod, PositionState
from run_daily import (
    _is_sse_trading_day,
    _is_unpriced_pending_target,
    _next_entry_date,
    _priced_state_as_of,
    _save_or_update_rebalance_target,
    run,
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


def test_trading_day_status_uses_requested_sse_calendar_date(monkeypatch):
    class FakePro:
        def trade_cal(self, **kwargs):
            assert kwargs == {
                "exchange": "SSE",
                "start_date": "20261001",
                "end_date": "20261001",
            }
            return pd.DataFrame({"cal_date": ["20261001"], "is_open": [0]})

    monkeypatch.setattr("run_daily.get_tushare_token", lambda: "token")
    monkeypatch.setattr("run_daily.ts.pro_api", lambda token: FakePro())

    assert _is_sse_trading_day(date(2026, 10, 1)) is False


def test_trading_day_status_rejects_unusable_calendar_response(monkeypatch):
    class FakePro:
        def trade_cal(self, **kwargs):
            return pd.DataFrame()

    monkeypatch.setattr("run_daily.get_tushare_token", lambda: "token")
    monkeypatch.setattr("run_daily.ts.pro_api", lambda token: FakePro())

    with pytest.raises(RuntimeError, match="Cannot determine SSE trading status"):
        _is_sse_trading_day(date(2026, 10, 1))


def test_closed_market_skips_sync_and_state_changes(monkeypatch, capsys):
    config = {
        "strategy_name": "quality_momentum_top1",
        "asset_pool": ["510300.SH"],
        "factors": [{"name": "quality_momentum"}],
        "enable_dingtalk": False,
    }
    monkeypatch.setattr("run_daily._is_sse_trading_day", lambda _: False)
    monkeypatch.setattr(
        "run_daily._sync_and_check",
        lambda *_: (_ for _ in ()).throw(AssertionError("must not sync")),
    )

    run(config)

    assert "daily signal skipped" in capsys.readouterr().out


def test_enabled_dingtalk_configuration_error_is_not_swallowed(monkeypatch):
    config = {
        "strategy_name": "quality_momentum_top1",
        "asset_pool": ["510300.SH"],
        "factors": [{"name": "quality_momentum"}],
        "enable_dingtalk": True,
    }

    def fail_notifier():
        raise ValueError("DingTalk webhook URL required")

    monkeypatch.setattr("run_daily.DingTalkNotifier", fail_notifier)

    with pytest.raises(ValueError, match="webhook URL required"):
        run(config)


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
