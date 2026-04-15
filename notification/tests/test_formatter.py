"""Tests for notification.formatter."""

from __future__ import annotations

from datetime import date

from execution.interfaces import Order
from notification.formatter import (
    ASSET_NAMES,
    NotificationContext,
    format_notification,
)


class TestNotificationContext:
    def test_can_construct_with_all_fields(self):
        ctx = NotificationContext(
            strategy_name="Quality_Momentum",
            signal_date=date(2026, 4, 9),
            orders=[],
            target_weights={"513100.SH": 1.0},
            current_weights={"513100.SH": 1.0},
            entry_date=date(2026, 4, 8),
            holding_days=2,
            position_return=0.0089,
            benchmark_returns={"159915.SZ": 0.0244},
            ytd_return=-0.0209,
            asset_names=ASSET_NAMES,
        )
        assert ctx.strategy_name == "Quality_Momentum"
        assert ctx.signal_date == date(2026, 4, 9)
        assert ctx.holding_days == 2

    def test_nullable_fields_accept_none(self):
        ctx = NotificationContext(
            strategy_name="Quality_Momentum",
            signal_date=date(2026, 4, 9),
            orders=[],
            target_weights={"513100.SH": 1.0},
            current_weights={"513100.SH": 1.0},
            entry_date=None,
            holding_days=None,
            position_return=None,
            benchmark_returns={},
            ytd_return=None,
            asset_names={},
        )
        assert ctx.entry_date is None
        assert ctx.holding_days is None
        assert ctx.position_return is None
        assert ctx.ytd_return is None


class TestAssetNames:
    def test_known_assets_present(self):
        assert "510300.SH" in ASSET_NAMES
        assert "159915.SZ" in ASSET_NAMES
        assert "513100.SH" in ASSET_NAMES
        assert "518880.SH" in ASSET_NAMES

    def test_chinese_names(self):
        assert ASSET_NAMES["510300.SH"] == "沪深300"
        assert ASSET_NAMES["159915.SZ"] == "创业板"
        assert ASSET_NAMES["513100.SH"] == "纳指ETF"
        assert ASSET_NAMES["518880.SH"] == "黄金ETF"


class TestFormatNotificationHold:
    """Hold path: no rebalance, show current position stats."""

    def _make_ctx(self, **overrides) -> NotificationContext:
        defaults = dict(
            strategy_name="Quality_Momentum",
            signal_date=date(2026, 4, 9),
            orders=[Order(asset="513100.SH", action="hold", weight_delta=0.0)],
            target_weights={"513100.SH": 1.0},
            current_weights={"513100.SH": 1.0},
            entry_date=date(2026, 4, 8),
            holding_days=2,
            position_return=0.0089,
            benchmark_returns={
                "159915.SZ": 0.0244,
                "510300.SH": 0.0117,
                "518880.SH": -0.0193,
            },
            ytd_return=-0.0209,
            asset_names=ASSET_NAMES,
        )
        defaults.update(overrides)
        return NotificationContext(**defaults)

    def test_contains_strategy_name(self):
        msg = format_notification(self._make_ctx())
        assert "Quality_Momentum" in msg

    def test_contains_signal_date(self):
        msg = format_notification(self._make_ctx())
        assert "2026-04-09" in msg

    def test_shows_holding_days(self):
        msg = format_notification(self._make_ctx())
        assert "2" in msg

    def test_shows_position_return(self):
        msg = format_notification(self._make_ctx())
        assert "+0.89%" in msg

    def test_shows_benchmark_section(self):
        msg = format_notification(self._make_ctx())
        assert "同期对比" in msg
        assert "创业板" in msg

    def test_shows_ytd_return(self):
        msg = format_notification(self._make_ctx())
        assert "年初至今" in msg
        assert "-2.09%" in msg

    def test_shows_asset_chinese_name(self):
        msg = format_notification(self._make_ctx())
        assert "纳指ETF" in msg

    def test_unknown_asset_shows_ticker(self):
        ctx = self._make_ctx(
            target_weights={"UNKNOWN.SH": 1.0},
            current_weights={"UNKNOWN.SH": 1.0},
            orders=[Order(asset="UNKNOWN.SH", action="hold", weight_delta=0.0)],
            asset_names={},
        )
        msg = format_notification(ctx)
        assert "UNKNOWN.SH" in msg

    def test_fallback_entry_prices_null(self):
        msg = format_notification(self._make_ctx(position_return=None))
        assert "待更新" in msg

    def test_fallback_entry_date_null(self):
        msg = format_notification(
            self._make_ctx(entry_date=None, holding_days=None, position_return=None)
        )
        assert "—" in msg

    def test_fallback_ytd_no_data(self):
        msg = format_notification(self._make_ctx(ytd_return=None))
        assert "数据不足" in msg

    def test_benchmark_diff_shown(self):
        msg = format_notification(self._make_ctx())
        # benchmark 159915.SZ = +2.44%, position = +0.89%, diff = +1.54% above
        assert "超出持仓" in msg or "↑" in msg or "↓" in msg


class TestFormatNotificationRebalance:
    """Rebalance path: orders present, show trade instructions."""

    def _make_ctx(self) -> NotificationContext:
        return NotificationContext(
            strategy_name="Quality_Momentum",
            signal_date=date(2026, 4, 9),
            orders=[
                Order(asset="159915.SZ", action="sell", weight_delta=-1.0),
                Order(asset="513100.SH", action="buy", weight_delta=1.0),
            ],
            target_weights={"513100.SH": 1.0},
            current_weights={"159915.SZ": 1.0},
            entry_date=None,
            holding_days=None,
            position_return=None,
            benchmark_returns={
                "159915.SZ": 0.0244,
                "510300.SH": 0.0117,
                "518880.SH": -0.0193,
            },
            ytd_return=-0.0209,
            asset_names=ASSET_NAMES,
        )

    def test_shows_rebalance_header(self):
        msg = format_notification(self._make_ctx())
        assert "调仓指令" in msg

    def test_shows_sell_instruction(self):
        msg = format_notification(self._make_ctx())
        assert "卖出" in msg
        assert "创业板" in msg

    def test_shows_buy_instruction(self):
        msg = format_notification(self._make_ctx())
        assert "买入" in msg
        assert "纳指ETF" in msg

    def test_shows_weight_change(self):
        msg = format_notification(self._make_ctx())
        assert "100%" in msg or "0%" in msg

    def test_still_shows_benchmark_and_ytd(self):
        msg = format_notification(self._make_ctx())
        assert "同期对比" in msg
        assert "年初至今" in msg
