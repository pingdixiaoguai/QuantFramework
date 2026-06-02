"""Tests for the YTD state backfill replay logic."""

from __future__ import annotations

from datetime import date

import pytest

from backfill_ytd import (
    ReplayResult,
    _clip_result_to_ytd,
    _load_config,
    _replay_signals_to_state,
    _state_as_of,
)
from execution.position import PositionPeriod, PositionState


def _price_lookup(prices: dict[tuple[str, date], float]):
    def lookup(assets: list[str], d: date) -> dict[str, float]:
        return {asset: prices[(asset, d)] for asset in assets if (asset, d) in prices}

    return lookup


def test_load_config_accepts_dynamic_today_end(tmp_path, monkeypatch):
    class FixedDate(date):
        @classmethod
        def today(cls):
            return cls(2026, 5, 11)

    monkeypatch.setattr("backfill_ytd.date", FixedDate)
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


def test_replay_books_trades_on_next_trading_day_open():
    trading_days = [
        date(2026, 1, 2),
        date(2026, 1, 5),
        date(2026, 1, 6),
    ]
    signals = [
        (date(2026, 1, 2), {"A.SH": 1.0}),
        (date(2026, 1, 5), {"B.SH": 1.0}),
        (date(2026, 1, 6), {"B.SH": 1.0}),
    ]
    prices = {
        ("A.SH", date(2026, 1, 5)): 10.0,
        ("A.SH", date(2026, 1, 6)): 11.0,
        ("B.SH", date(2026, 1, 6)): 20.0,
    }

    result = _replay_signals_to_state(
        signals,
        trading_days,
        _price_lookup(prices),
        rebalance_days=1,
    )

    assert len(result.state.ytd_history) == 1
    closed = result.state.ytd_history[0]
    assert closed.weights == {"A.SH": 1.0}
    assert closed.entry_date == "2026-01-05"
    assert closed.exit_date == "2026-01-06"
    assert closed.entry_prices == {"A.SH": 10.0}
    assert closed.exit_prices == {"A.SH": 11.0}
    assert result.state.weights == {"B.SH": 1.0}
    assert result.state.entry_date == "2026-01-06"
    assert result.state.entry_prices == {"B.SH": 20.0}


def test_replay_honors_rebalance_days_hold_window():
    trading_days = [
        date(2026, 1, 2),
        date(2026, 1, 5),
        date(2026, 1, 6),
        date(2026, 1, 7),
    ]
    signals = [
        (date(2026, 1, 2), {"A.SH": 1.0}),
        (date(2026, 1, 5), {"B.SH": 1.0}),
        (date(2026, 1, 6), {"B.SH": 1.0}),
        (date(2026, 1, 7), {"B.SH": 1.0}),
    ]
    prices = {
        ("A.SH", date(2026, 1, 5)): 10.0,
        ("A.SH", date(2026, 1, 7)): 12.0,
        ("B.SH", date(2026, 1, 7)): 20.0,
    }

    result = _replay_signals_to_state(
        signals,
        trading_days,
        _price_lookup(prices),
        rebalance_days=2,
    )

    assert len(result.state.ytd_history) == 1
    closed = result.state.ytd_history[0]
    assert closed.weights == {"A.SH": 1.0}
    assert closed.entry_date == "2026-01-05"
    assert closed.exit_date == "2026-01-07"
    assert result.state.weights == {"B.SH": 1.0}
    assert result.state.entry_date == "2026-01-07"


def test_replay_fixed_cycle_skips_non_boundary_signals():
    trading_days = [
        date(2026, 1, 2),
        date(2026, 1, 5),
        date(2026, 1, 6),
        date(2026, 1, 7),
        date(2026, 1, 8),
        date(2026, 1, 9),
    ]
    signals = [
        (date(2026, 1, 2), {"A.SH": 1.0}),
        (date(2026, 1, 5), {"A.SH": 1.0}),
        (date(2026, 1, 6), {"A.SH": 1.0}),
        (date(2026, 1, 7), {"B.SH": 1.0}),
        (date(2026, 1, 8), {"B.SH": 1.0}),
        (date(2026, 1, 9), {"B.SH": 1.0}),
    ]
    prices = {
        ("A.SH", date(2026, 1, 5)): 10.0,
        ("A.SH", date(2026, 1, 9)): 12.0,
        ("B.SH", date(2026, 1, 9)): 20.0,
    }

    result = _replay_signals_to_state(
        signals,
        trading_days,
        _price_lookup(prices),
        rebalance_days=2,
        rebalance_mode="fixed_cycle",
    )

    assert len(result.state.ytd_history) == 1
    closed = result.state.ytd_history[0]
    assert closed.weights == {"A.SH": 1.0}
    assert closed.entry_date == "2026-01-05"
    assert closed.exit_date == "2026-01-09"
    assert result.state.weights == {"B.SH": 1.0}
    assert result.state.entry_date == "2026-01-09"


def test_state_as_of_preserves_before_date_and_activates_spanning_period():
    before = PositionPeriod(
        weights={"A.SH": 1.0},
        entry_date="2026-01-05",
        exit_date="2026-01-07",
        entry_prices={"A.SH": 10.0},
        exit_prices={"A.SH": 11.0},
    )
    spanning = PositionPeriod(
        weights={"B.SH": 1.0},
        entry_date="2026-01-07",
        exit_date="2026-01-12",
        entry_prices={"B.SH": 20.0},
        exit_prices={"B.SH": 21.0},
    )
    future = PositionPeriod(
        weights={"C.SH": 1.0},
        entry_date="2026-01-12",
        exit_date="2026-01-15",
        entry_prices={"C.SH": 30.0},
        exit_prices={"C.SH": 31.0},
    )
    state = PositionState(
        weights={"D.SH": 1.0},
        entry_date="2026-01-15",
        entry_prices={"D.SH": 40.0},
        ytd_history=[before, spanning, future],
    )

    sliced = _state_as_of(state, date(2026, 1, 9))

    assert sliced.ytd_history == [before]
    assert sliced.weights == {"B.SH": 1.0}
    assert sliced.entry_date == "2026-01-07"
    assert sliced.entry_prices == {"B.SH": 20.0}


def test_replay_from_initial_state_keeps_existing_history():
    preserved = PositionPeriod(
        weights={"A.SH": 1.0},
        entry_date="2026-01-05",
        exit_date="2026-01-07",
        entry_prices={"A.SH": 10.0},
        exit_prices={"A.SH": 11.0},
    )
    initial_state = PositionState(
        weights={"B.SH": 1.0},
        entry_date="2026-01-07",
        entry_prices={"B.SH": 20.0},
        ytd_history=[preserved],
    )
    trading_days = [
        date(2026, 1, 9),
        date(2026, 1, 12),
    ]
    holding_calendar = [
        date(2026, 1, 7),
        date(2026, 1, 8),
        *trading_days,
    ]
    signals = [
        (date(2026, 1, 9), {"C.SH": 1.0}),
        (date(2026, 1, 12), {"C.SH": 1.0}),
    ]
    prices = {
        ("B.SH", date(2026, 1, 12)): 22.0,
        ("C.SH", date(2026, 1, 12)): 30.0,
    }

    result = _replay_signals_to_state(
        signals,
        trading_days,
        _price_lookup(prices),
        rebalance_days=2,
        initial_state=initial_state,
        holding_calendar=holding_calendar,
    )

    assert len(result.state.ytd_history) == 2
    assert result.state.ytd_history[0] == preserved
    replayed = result.state.ytd_history[1]
    assert replayed.weights == {"B.SH": 1.0}
    assert replayed.entry_date == "2026-01-07"
    assert replayed.exit_date == "2026-01-12"
    assert replayed.exit_prices == {"B.SH": 22.0}
    assert result.state.weights == {"C.SH": 1.0}
    assert result.state.entry_date == "2026-01-12"


def test_clip_result_to_ytd_rebases_cross_year_period_and_current_position():
    before_year = PositionPeriod(
        weights={"A.SH": 1.0},
        entry_date="2025-12-26",
        exit_date="2026-01-02",
        entry_prices={"A.SH": 10.0},
        exit_prices={"A.SH": 11.0},
    )
    spanning = PositionPeriod(
        weights={"B.SH": 1.0},
        entry_date="2025-12-30",
        exit_date="2026-01-07",
        entry_prices={"B.SH": 20.0},
        exit_prices={"B.SH": 24.0},
    )
    same_year = PositionPeriod(
        weights={"C.SH": 1.0},
        entry_date="2026-01-07",
        exit_date="2026-01-09",
        entry_prices={"C.SH": 30.0},
        exit_prices={"C.SH": 33.0},
    )
    result = ReplayResult(
        state=PositionState(
            weights={"D.SH": 1.0},
            entry_date="2025-12-31",
            entry_prices={"D.SH": 40.0},
            ytd_history=[before_year, spanning, same_year],
        ),
        closed_returns=[],
        latest_trading_day=date(2026, 1, 12),
    )
    prices = {
        ("B.SH", date(2026, 1, 5)): 21.0,
        ("D.SH", date(2026, 1, 5)): 41.0,
    }

    clipped = _clip_result_to_ytd(
        result,
        date(2026, 1, 5),
        _price_lookup(prices),
    )

    assert len(clipped.state.ytd_history) == 2
    rebased = clipped.state.ytd_history[0]
    assert rebased.weights == {"B.SH": 1.0}
    assert rebased.entry_date == "2026-01-05"
    assert rebased.entry_prices == {"B.SH": 21.0}
    assert rebased.exit_date == "2026-01-07"
    assert clipped.closed_returns[0][1] == pytest.approx(24.0 / 21.0 - 1)

    assert clipped.state.ytd_history[1] == same_year
    assert clipped.state.entry_date == "2025-12-31"
    assert clipped.state.entry_prices == {"D.SH": 40.0}
    assert clipped.state.ytd_entry_date == "2026-01-05"
    assert clipped.state.ytd_entry_prices == {"D.SH": 41.0}


def test_clip_result_uses_prior_day_close_when_carry_in_basis_supplied():
    """When the wider replay knows the prev-year last trading day, the YTD
    basis should be that day's close. This makes the chained YTD include the
    close[prev] -> open[ytd_start] overnight gap that the backtest captures
    via its close-to-close daily return on the first day of the year."""
    spanning = PositionPeriod(
        weights={"B.SH": 1.0},
        entry_date="2025-12-30",
        exit_date="2026-01-07",
        entry_prices={"B.SH": 20.0},
        exit_prices={"B.SH": 24.0},
    )
    result = ReplayResult(
        state=PositionState(
            weights={"D.SH": 1.0},
            entry_date="2025-12-31",
            entry_prices={"D.SH": 40.0},
            ytd_history=[spanning],
        ),
        closed_returns=[],
        latest_trading_day=date(2026, 1, 12),
    )
    open_prices = {
        ("B.SH", date(2026, 1, 5)): 21.0,
        ("D.SH", date(2026, 1, 5)): 41.0,
    }
    close_prices = {
        ("B.SH", date(2025, 12, 31)): 20.5,
        ("D.SH", date(2025, 12, 31)): 40.5,
    }

    clipped = _clip_result_to_ytd(
        result,
        date(2026, 1, 5),
        _price_lookup(open_prices),
        carry_in_basis_date=date(2025, 12, 31),
        carry_in_basis_lookup=_price_lookup(close_prices),
    )

    rebased = clipped.state.ytd_history[0]
    assert rebased.entry_date == "2026-01-05"  # date label is still YTD-bound
    assert rebased.entry_prices == {"B.SH": 20.5}  # but basis is prev-day close
    assert clipped.closed_returns[0][1] == pytest.approx(24.0 / 20.5 - 1)
    assert clipped.state.ytd_entry_prices == {"D.SH": 40.5}
