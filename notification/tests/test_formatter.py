"""Tests for notification.formatter."""

from __future__ import annotations

from datetime import date

from execution.interfaces import Order
from notification.formatter import (
    ASSET_NAMES,
    NotificationContext,
    StrategySignalView,
    format_notification,
)


ASSETS = ["510300.SH", "159915.SZ", "513100.SH", "518880.SH"]


def _signal(
    strategy_name: str,
    target: str,
    scores: dict[str, float],
    confidence: dict[str, float],
) -> StrategySignalView:
    return StrategySignalView(
        strategy_name=strategy_name,
        target=target,
        scores=scores,
        confidence=confidence,
        expected_assets=ASSETS,
    )


def _production_signal(target: str = "513100.SH") -> StrategySignalView:
    return _signal(
        "quality_momentum_top1",
        target,
        {
            "510300.SH": 0.01,
            "159915.SZ": 0.02,
            "513100.SH": 0.05,
            "518880.SH": 0.03,
        },
        {
            "510300.SH": 0.10,
            "159915.SZ": 0.15,
            "513100.SH": 0.45,
            "518880.SH": 0.30,
        },
    )


def _shadow_signal(target: str = "518880.SH") -> StrategySignalView:
    return _signal(
        "quality_momentum_top1_ohlc_er",
        target,
        {
            "510300.SH": 0.01,
            "159915.SZ": 0.015,
            "513100.SH": 0.025,
            "518880.SH": 0.06,
        },
        {
            "510300.SH": 0.08,
            "159915.SZ": 0.12,
            "513100.SH": 0.25,
            "518880.SH": 0.55,
        },
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
            rebalance_days=5,
            production_signal=_production_signal(),
            shadow_signal=_shadow_signal(),
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
            rebalance_days=5,
            production_signal=_production_signal(),
            shadow_signal=_shadow_signal(),
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
        assert "同期表现" in msg
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
        assert "自 2026-04-09 开盘起" in msg

    def test_fallback_ytd_no_data(self):
        msg = format_notification(self._make_ctx(ytd_return=None))
        assert "数据不足" in msg

    def test_benchmark_diff_shown(self):
        msg = format_notification(self._make_ctx())
        # benchmark 159915.SZ = +2.44%, position = +0.89%, diff = +1.54% above
        assert "超出持仓" in msg or "↑" in msg or "↓" in msg

    def test_shows_symmetric_strategy_comparison(self):
        msg = format_notification(self._make_ctx())

        assert "生产策略 · quality_momentum_top1" in msg
        assert "影子策略 · quality_momentum_top1_ohlc_er" in msg
        assert "#1 513100 纳指ETF" in msg
        assert "#1 518880 黄金ETF" in msg
        assert "得分 +5.00%" in msg
        assert "得分 +6.00%" in msg
        assert "相对强度 45.0%" in msg
        assert "相对强度 55.0%" in msg
        assert "Top1 领先：15.0 个百分点" in msg
        assert "Top1 领先：30.0 个百分点" in msg
        assert "原始目标不同" in msg
        assert "纳指ETF：生产 #1 → 影子 #2" in msg
        assert "黄金ETF：生产 #2 → 影子 #1" in msg
        assert "影子信号仅作观察" in msg
        assert "不代表上涨概率" in msg

    def test_distinguishes_raw_signal_from_held_execution_target(self):
        production = _production_signal(target="518880.SH")
        msg = format_notification(self._make_ctx(production_signal=production))

        assert "生产策略原始目标：518880 黄金ETF" in msg
        assert "实际交易目标：513100 纳指ETF" in msg
        assert "持有期约束：原始目标暂未执行" in msg

    def test_warns_when_shadow_signal_uses_incomplete_universe(self):
        shadow = _shadow_signal()
        shadow.scores.pop("510300.SH")
        shadow.confidence.pop("510300.SH")

        msg = format_notification(self._make_ctx(shadow_signal=shadow))

        assert "⚠️ 数据完整性：3/4，缺少 510300 沪深300" in msg


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
            rebalance_days=5,
            production_signal=_production_signal(),
            shadow_signal=_shadow_signal(),
        )

    def test_shows_rebalance_header(self):
        msg = format_notification(self._make_ctx())
        assert "实际执行（调仓）" in msg

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
        assert "同期表现" in msg
        assert "年初至今" in msg

    def test_shows_raw_and_actual_target_on_rebalance_days(self):
        msg = format_notification(self._make_ctx())

        assert "生产策略原始目标：513100 纳指ETF" in msg
        assert "实际交易目标：513100 纳指ETF" in msg
        assert "持有期约束" not in msg
