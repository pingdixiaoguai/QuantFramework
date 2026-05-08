"""Tests for run_daily._should_hold — rebalance-window decision logic."""

from __future__ import annotations

from run_daily import _should_hold


class TestRebalanceDaysOne:
    """rebalance_days=1 means daily — never hold."""

    def test_with_position_never_holds(self):
        assert _should_hold({"X": 1.0}, holding_days=1, rebalance_days=1) is False
        assert _should_hold({"X": 1.0}, holding_days=10, rebalance_days=1) is False

    def test_empty_position_never_holds(self):
        assert _should_hold({}, holding_days=None, rebalance_days=1) is False


class TestRebalanceDaysFive:
    def test_inside_window_holds(self):
        for d in (1, 2, 3, 4):
            assert _should_hold({"X": 1.0}, holding_days=d, rebalance_days=5) is True

    def test_at_window_boundary_rebalances(self):
        # holding_days >= rebalance_days → rebalance allowed
        assert _should_hold({"X": 1.0}, holding_days=5, rebalance_days=5) is False
        assert _should_hold({"X": 1.0}, holding_days=6, rebalance_days=5) is False

    def test_empty_position_never_holds(self):
        # First-time entry: must allow strategy to take a position even if
        # holding_days reads as None or 0.
        assert _should_hold({}, holding_days=None, rebalance_days=5) is False
        assert _should_hold({}, holding_days=0, rebalance_days=5) is False

    def test_holding_days_none_with_position_holds(self):
        # Just bought yesterday; entry_date is set but today's bar not yet
        # reflected → holding_days reads None. Must hold (don't churn).
        assert _should_hold({"X": 1.0}, holding_days=None, rebalance_days=5) is True
