"""Tests for DingTalk YTD return calculation helpers."""

from __future__ import annotations

from datetime import date

import pytest

import pandas as pd

from execution.position import PositionPeriod, PositionState
from run_daily import (
    _compute_position_return,
    _compute_ytd_return,
    _latest_common_data_date,
    _load_config,
    _priced_state_as_of,
)


def test_ytd_chains_closed_open_to_open_and_current_open_to_close_returns():
    """DingTalk YTD follows the live open-execution ledger.

    A closed period runs from entry open to exit open. The current period then
    runs from the same open to the latest close, so a rebalance day keeps the
    old holding's overnight PnL and the new holding's intraday PnL.
    """
    closed = PositionPeriod(
        weights={"A.SH": 1.0},
        entry_date="2026-01-05",
        exit_date="2026-01-06",
        entry_prices={"A.SH": 100.0},
        exit_prices={"A.SH": 110.0},
    )

    assert _compute_ytd_return([closed], current_return=0.10) == pytest.approx(0.21)


def test_latest_common_data_date_uses_priced_date_not_calendar_today(monkeypatch):
    """Morning runs may only have yesterday's bars; label the signal by data."""

    def fake_read_local(asset):
        latest = {
            "A.SH": "2026-05-14",
            "B.SH": "2026-05-13",
        }[asset]
        return pd.DataFrame({"date": pd.to_datetime(["2026-05-12", latest])})

    monkeypatch.setattr("run_daily.read_local", fake_read_local)

    assert _latest_common_data_date(["A.SH", "B.SH"], date(2026, 5, 14)) == date(
        2026, 5, 13
    )


def test_pending_next_open_position_values_old_holding_until_entry_is_priced(
    monkeypatch,
):
    """A saved T+1 target must not make the T close -> T+1 pending gap disappear."""
    pending_state = PositionState(
        weights={"B.SH": 1.0},
        entry_date="2026-01-06",
        entry_prices=None,
        ytd_history=[
            PositionPeriod(
                weights={"C.SH": 1.0},
                entry_date="2026-01-02",
                exit_date="2026-01-03",
                entry_prices={"C.SH": 10.0},
                exit_prices={"C.SH": 11.0},
            ),
            PositionPeriod(
                weights={"A.SH": 1.0},
                entry_date="2026-01-03",
                exit_date="2026-01-06",
                entry_prices={"A.SH": 100.0},
                exit_prices=None,
            ),
        ],
    )

    def fake_query(asset, start, end):
        assert asset == "A.SH"
        return pd.DataFrame(
            {
                "date": pd.to_datetime(["2026-01-05"]),
                "open": [100.0],
                "close": [120.0],
            }
        )

    monkeypatch.setattr("run_daily.query", fake_query)

    priced_state = _priced_state_as_of(pending_state, date(2026, 1, 5))
    current_return = _compute_position_return(
        priced_state.weights,
        priced_state.entry_prices,
        date(2026, 1, 5),
    )

    assert priced_state.weights == {"A.SH": 1.0}
    assert priced_state.ytd_history == pending_state.ytd_history[:1]
    assert current_return == pytest.approx(0.20)
    assert _compute_ytd_return(priced_state.ytd_history, current_return) == pytest.approx(
        1.10 * 1.20 - 1
    )


def test_load_config_accepts_dynamic_today_end(tmp_path, monkeypatch):
    class FixedDate(date):
        @classmethod
        def today(cls):
            return cls(2026, 5, 11)

    monkeypatch.setattr("run_daily.date", FixedDate)
    path = tmp_path / "cfg.yaml"
    path.write_text(
        "strategy_name: test\n"
        "asset_pool: []\n"
        "start: '2016-01-01'\n"
        "end: 'today'\n"
        "factors: []\n",
        encoding="utf-8",
    )

    config = _load_config(path)

    assert config["start"] == date(2016, 1, 1)
    assert config["end"] == date(2026, 5, 11)
