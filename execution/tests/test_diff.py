"""Tests for execution.interfaces.diff."""

from execution.interfaces import Order, diff


class TestNewPositions:
    def test_buy_into_empty_portfolio(self):
        orders = diff({"A.SH": 1.0}, {})
        assert len(orders) == 1
        assert orders[0] == Order(asset="A.SH", action="buy", weight_delta=1.0)

    def test_multiple_new_positions(self):
        orders = diff({"A.SH": 0.6, "B.SH": 0.4}, {})
        by_asset = {o.asset: o for o in orders}
        assert by_asset["A.SH"].action == "buy"
        assert by_asset["A.SH"].weight_delta == 0.6
        assert by_asset["B.SH"].action == "buy"
        assert by_asset["B.SH"].weight_delta == 0.4


class TestExitPositions:
    def test_sell_all_when_target_empty(self):
        orders = diff({}, {"A.SH": 0.5, "B.SH": 0.5})
        by_asset = {o.asset: o for o in orders}
        assert by_asset["A.SH"].action == "sell"
        assert by_asset["A.SH"].weight_delta == -0.5
        assert by_asset["B.SH"].action == "sell"
        assert by_asset["B.SH"].weight_delta == -0.5

    def test_exit_one_asset(self):
        orders = diff({"A.SH": 1.0}, {"A.SH": 0.5, "B.SH": 0.5})
        by_asset = {o.asset: o for o in orders}
        assert by_asset["B.SH"].action == "sell"
        assert by_asset["B.SH"].weight_delta == -0.5


class TestRebalance:
    def test_increase_and_decrease(self):
        orders = diff({"A.SH": 0.7, "B.SH": 0.3}, {"A.SH": 0.5, "B.SH": 0.5})
        by_asset = {o.asset: o for o in orders}
        assert by_asset["A.SH"].action == "buy"
        assert abs(by_asset["A.SH"].weight_delta - 0.2) < 1e-10
        assert by_asset["B.SH"].action == "sell"
        assert abs(by_asset["B.SH"].weight_delta - (-0.2)) < 1e-10

    def test_mixed_new_exit_rebalance(self):
        """target has A,B; current has A,C → buy B, sell C, rebalance A."""
        orders = diff(
            {"A.SH": 0.6, "B.SH": 0.4},
            {"A.SH": 0.5, "C.SH": 0.5},
        )
        by_asset = {o.asset: o for o in orders}
        assert by_asset["A.SH"].action == "buy"
        assert abs(by_asset["A.SH"].weight_delta - 0.1) < 1e-10
        assert by_asset["B.SH"].action == "buy"
        assert abs(by_asset["B.SH"].weight_delta - 0.4) < 1e-10
        assert by_asset["C.SH"].action == "sell"
        assert abs(by_asset["C.SH"].weight_delta - (-0.5)) < 1e-10


class TestNoChange:
    def test_identical_weights_produce_hold(self):
        orders = diff({"A.SH": 1.0}, {"A.SH": 1.0})
        assert len(orders) == 1
        assert orders[0].action == "hold"
        assert orders[0].weight_delta == 0.0

    def test_both_empty(self):
        assert diff({}, {}) == []
