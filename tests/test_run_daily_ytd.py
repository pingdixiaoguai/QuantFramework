"""Tests for DingTalk YTD return calculation helpers."""

from __future__ import annotations

import pytest

from execution.position import PositionPeriod
from run_daily import _compute_ytd_return


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
