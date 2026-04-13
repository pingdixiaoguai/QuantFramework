"""Tests for notification.formatter."""

from datetime import date

from execution.interfaces import Order
from notification.formatter import format_orders


class TestFormatOrders:
    def test_empty_orders(self):
        msg = format_orders([], {}, run_date=date(2026, 4, 13))
        assert "无需调仓" in msg
        assert "2026-04-13" in msg

    def test_buy_and_sell(self):
        orders = [
            Order(asset="A.SH", action="buy", weight_delta=0.1),
            Order(asset="B.SH", action="sell", weight_delta=-0.1),
        ]
        target = {"A.SH": 0.6, "B.SH": 0.4}
        msg = format_orders(orders, target, run_date=date(2026, 4, 13))
        assert "买入: 1" in msg
        assert "卖出: 1" in msg
        assert "A.SH" in msg
        assert "B.SH" in msg
        assert "+10.00%" in msg
        assert "-10.00%" in msg

    def test_target_weight_shown(self):
        orders = [Order(asset="X.SH", action="buy", weight_delta=1.0)]
        msg = format_orders(orders, {"X.SH": 1.0}, run_date=date(2026, 1, 1))
        assert "100.00%" in msg

    def test_hold_counted(self):
        orders = [Order(asset="A.SH", action="hold", weight_delta=0.0)]
        msg = format_orders(orders, {"A.SH": 0.5}, run_date=date(2026, 1, 1))
        assert "持有: 1" in msg

    def test_exited_asset_shows_zero_target(self):
        orders = [Order(asset="OLD.SH", action="sell", weight_delta=-0.5)]
        msg = format_orders(orders, {}, run_date=date(2026, 1, 1))
        assert "0.00%" in msg
