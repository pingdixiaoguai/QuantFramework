"""Tests for the YTD state backfill replay logic."""

from __future__ import annotations

from datetime import date

from backfill_ytd import _replay_signals_to_state


def _price_lookup(prices: dict[tuple[str, date], float]):
    def lookup(assets: list[str], d: date) -> dict[str, float]:
        return {asset: prices[(asset, d)] for asset in assets if (asset, d) in prices}

    return lookup


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
