from __future__ import annotations

from datetime import date
from types import SimpleNamespace

from defender.live import DefenderNextOpenTarget
from execution.position import PositionState
from notification.peak_warning import PeakWarning
from strategy.momentum_defender import (
    IntegratedNextOpenSignal,
    MomentumNextOpenTarget,
)
from strategy.momentum_defender_w40_loss import FormalW40LossNextOpenSignal
from strategy.signal_performance import (
    PeriodPerformance,
    SignalPerformanceSnapshot,
)


def _signal() -> IntegratedNextOpenSignal:
    defender = DefenderNextOpenTarget(
        signal_date=date(2026, 8, 21),
        execution_date=date(2026, 8, 24),
        current_weights={"510880.SH": 0.2, "511260.SH": 0.8},
        target_weights={"510880.SH": 0.2, "511260.SH": 0.8},
        target_cash_weight=0.0,
        current_selected_asset="510880.SH",
        target_selected_asset="510880.SH",
        selection_reason="long_term_trend",
        signal_reason="hold",
    )
    momentum = MomentumNextOpenTarget(
        raw_weights={"518880.SH": 1.0},
        effective_weights={"518880.SH": 1.0},
        held_asset="510300.SH",
        holding_days=10,
        hold_filter_active=False,
    )
    return IntegratedNextOpenSignal(
        strategy_id="momentum_defender_c2_defender_main_b5e3419",
        defender_strategy_id="relative_defender_rotation_2013_listing_aware",
        signal_date=date(2026, 8, 21),
        execution_date=date(2026, 8, 24),
        current_model_sleeve="defender",
        target_sleeve="defender",
        state_reason="hold",
        held_days_at_open=60,
        slow_gate_return=-0.04,
        slow_gate_risk_on=False,
        emergency_asset="518880.SH",
        emergency_cap=1.0,
        emergency_alert=False,
        momentum=momentum,
        defender=defender,
        target_weights={"510880.SH": 0.2, "511260.SH": 0.8},
        target_cash_weight=0.0,
    )


def _w40_signal(asset: str = "159915.SZ") -> FormalW40LossNextOpenSignal:
    base = _signal()
    momentum = MomentumNextOpenTarget(
        raw_weights={asset: 1.0},
        effective_weights={asset: 1.0},
        held_asset=asset,
        holding_days=10,
        hold_filter_active=False,
    )
    return FormalW40LossNextOpenSignal(
        strategy_id="momentum_defender_w40_reversal_full_equity_v2",
        defender_strategy_id="dividend_w40_reversal_full_equity_v2",
        signal_date=date(2026, 8, 21),
        execution_date=date(2026, 8, 24),
        current_model_sleeve="momentum",
        target_sleeve="momentum",
        state_reason="hold",
        held_days_at_open=40,
        w40_downside_log_loss=0.0,
        w40_loss_percentile=0.0,
        defender_entry_percentile=0.55,
        momentum_recovery_percentile=0.40,
        entry_confirmation_streak=0,
        recovery_confirmation_streak=0,
        defender_entry_confirmation_days=1,
        momentum_recovery_confirmation_days=1,
        momentum_lock_days=30,
        defender_lock_days=30,
        momentum=momentum,
        defender=base.defender,
        target_weights={asset: 1.0},
        target_cash_weight=0.0,
    )


def test_production_runner_sends_and_persists_exact_target(monkeypatch) -> None:
    import run_daily_momentum_defender as runner

    class FixedDate(date):
        @classmethod
        def today(cls):
            return cls(2026, 8, 24)

    sent: list[tuple[str, str]] = []
    saved: list[dict[str, float]] = []

    class FakeNotifier:
        def send(self, message, *, title="", alert_text=""):
            sent.append((title, message))

    monkeypatch.setattr(runner, "date", FixedDate)
    monkeypatch.setattr(runner, "_is_sse_trading_day", lambda _: True)
    monkeypatch.setattr(runner, "_latest_common_data_date", lambda *_: date(2026, 8, 21))
    monkeypatch.setattr(runner, "_next_entry_date", lambda _: date(2026, 8, 24))
    monkeypatch.setattr(runner, "read_position", lambda _: PositionState())
    monkeypatch.setattr(runner, "_backfill_open_prices", lambda state, *_: state)
    monkeypatch.setattr(runner, "build_integrated_next_open_signal", lambda *_: _signal())
    monkeypatch.setattr(
        runner,
        "_save_or_update_rebalance_target",
        lambda _state, target, *_args: saved.append(target.copy()) or False,
    )

    runner.run(
        {
            "strategy_name": "momentum_defender_c2_defender_main_b5e3419",
            "asset_pool": ["510880.SH", "511260.SH"],
            "enable_dingtalk": True,
        },
        skip_sync=True,
        notifier_factory=FakeNotifier,
    )

    assert len(sent) == 1
    assert "Defender" in sent[0][0]
    assert "510880 红利ETF 20%" in sent[0][1]
    assert saved == [{"510880.SH": 0.2, "511260.SH": 0.8}]


def test_dry_run_neither_sends_nor_persists(monkeypatch) -> None:
    import run_daily_momentum_defender as runner

    monkeypatch.setattr(runner, "_latest_common_data_date", lambda *_: date(2026, 8, 21))
    monkeypatch.setattr(runner, "_next_entry_date", lambda _: date(2026, 8, 24))
    monkeypatch.setattr(runner, "read_position", lambda _: PositionState())
    monkeypatch.setattr(runner, "build_integrated_next_open_signal", lambda *_: _signal())
    monkeypatch.setattr(
        runner,
        "_save_or_update_rebalance_target",
        lambda *_: (_ for _ in ()).throw(AssertionError("must not persist")),
    )

    signal, message = runner.run(
        {
            "strategy_name": "momentum_defender_c2_defender_main_b5e3419",
            "asset_pool": ["510880.SH", "511260.SH"],
            "enable_dingtalk": True,
        },
        dry_run=True,
        skip_sync=True,
        notifier_factory=lambda: (_ for _ in ()).throw(AssertionError("must not send")),
    )

    assert signal.target_sleeve == "defender"
    assert "未发送、未写持仓" in message


def test_confirmation_bridge_mode_dispatches_formal_builder(monkeypatch) -> None:
    import run_daily_momentum_defender as runner

    expected = object()
    monkeypatch.setattr(
        runner,
        "build_formal_gold_next_open_signal",
        lambda *_: expected,
    )

    actual = runner._build_signal(
        {"strategy_mode": "confirmation_bridge_raw_gold"},
        date(2026, 8, 21),
        date(2026, 8, 24),
    )

    assert actual is expected


def test_downside_raqm_mode_dispatches_new_formal_builder(monkeypatch) -> None:
    import run_daily_momentum_defender as runner

    expected = object()
    monkeypatch.setattr(
        runner,
        "build_downside_raqm_next_open_signal",
        lambda *_: expected,
    )

    actual = runner._build_signal(
        {"strategy_mode": "downside_raqm"},
        date(2026, 8, 21),
        date(2026, 8, 24),
    )

    assert actual is expected


def test_w40_loss_mode_dispatches_current_formal_builder(monkeypatch) -> None:
    import run_daily_momentum_defender as runner

    expected = object()
    monkeypatch.setattr(runner, "build_w40_loss_next_open_signal", lambda *_: expected)
    actual = runner._build_signal(
        {"strategy_mode": "w40_loss"},
        date(2026, 8, 21),
        date(2026, 8, 24),
    )
    assert actual is expected


def test_w40_full_equity_mode_dispatches_promoted_builder(monkeypatch) -> None:
    import run_daily_momentum_defender as runner

    expected = object()
    monkeypatch.setattr(
        runner, "build_w40_full_equity_next_open_signal", lambda *_: expected
    )
    actual = runner._build_signal(
        {"strategy_mode": "w40_reversal_full_equity"},
        date(2026, 8, 21),
        date(2026, 8, 24),
    )
    assert actual is expected


def test_w40_gold_escape_mode_dispatches_latest_formal_builder(monkeypatch) -> None:
    import run_daily_momentum_defender as runner

    expected = object()
    monkeypatch.setattr(
        runner, "build_w40_gold_escape_next_open_signal", lambda *_: expected
    )
    actual = runner._build_signal(
        {"strategy_mode": "w40_gold_qm20_escape"},
        date(2026, 8, 21),
        date(2026, 8, 24),
    )
    assert actual is expected


def test_w40_qm40_signed_exit_mode_dispatches_v4_builder(monkeypatch) -> None:
    import run_daily_momentum_defender as runner

    expected = object()
    monkeypatch.setattr(
        runner,
        "build_w40_qm40_signed_exit_next_open_signal",
        lambda *_: expected,
    )
    actual = runner._build_signal(
        {"strategy_mode": "w40_qm40_signed_exit"},
        date(2026, 8, 21),
        date(2026, 8, 24),
    )
    assert actual is expected


def test_w40_qm40_threshold_mode_dispatches_v5_builder(monkeypatch) -> None:
    import run_daily_momentum_defender as runner

    expected = object()
    monkeypatch.setattr(
        runner,
        "build_w40_qm40_threshold_next_open_signal",
        lambda *_: expected,
    )
    actual = runner._build_signal(
        {"strategy_mode": "w40_qm40_threshold"},
        date(2026, 8, 26),
        date(2026, 8, 27),
    )
    assert actual is expected


def test_w40_notification_includes_read_only_peak_warning(monkeypatch) -> None:
    import run_daily_momentum_defender as runner

    sent: list[str] = []

    class FakeNotifier:
        def send(self, message, *, title="", alert_text=""):
            sent.append(message)

    monkeypatch.setattr(
        runner, "_latest_common_data_date", lambda *_: date(2026, 8, 21)
    )
    monkeypatch.setattr(
        runner, "_next_entry_date", lambda _: date(2026, 8, 24)
    )
    monkeypatch.setattr(runner, "read_position", lambda _: PositionState())
    monkeypatch.setattr(runner, "_build_signal", lambda *_: _w40_signal())
    warning = PeakWarning(
        asset="159915.SZ",
        signal_date=date(2026, 8, 21),
        triggered=True,
        current_close=9.187,
        prior_high200=8.859,
        close20ago=7.966,
        current_volume=9_193_141.0,
        prior_volume_median20=4_486_818.0,
        price_breakout=0.03,
        price_return20=0.20,
        volume_ratio20=2.0,
        share_filter_required=True,
        share_data_available=True,
        share_flow20=0.08,
        reason="价量条件满足，创业板20日基金份额增长",
    )

    signal, message = runner.run(
        {
            "strategy_name": "momentum_defender_w40_reversal_full_equity_v2",
            "asset_pool": ["159915.SZ"],
            "enable_dingtalk": True,
        },
        notification_only=True,
        skip_sync=True,
        notifier_factory=FakeNotifier,
        peak_warning_evaluator=lambda *_: warning,
    )

    assert signal.target_weights == {"159915.SZ": 1.0}
    assert len(sent) == 1
    assert "价格×量能顶部预警（只读）" in message
    assert "预警状态：⚠️ 已触发" in message
    assert "突破200日前高：已满足" in message
    assert "20日涨幅≥15%：已满足" in message
    assert "成交量/此前20日中位数≥1.50倍：已满足" in message
    assert "创业板20日基金份额增长：已满足；当前+8.00%" in message
    assert "**袖套状态：** 动量（保持）" in message
    assert "**状态原因：** 保持当前W40基础状态" in message
    assert "继续持有 159915 创业板 100%" in message
    assert "买入 159915" not in message
    assert "当前实盘持仓" not in message
    assert "不改变正式目标" not in message
    assert "信号仅使用截至信号日收盘的数据" not in message


def test_w40_notification_formats_unavailable_warning_without_nan() -> None:
    import run_daily_momentum_defender as runner

    warning = PeakWarning(
        asset="159915.SZ",
        signal_date=date(2026, 8, 21),
        triggered=False,
        current_close=float("nan"),
        prior_high200=float("nan"),
        close20ago=float("nan"),
        current_volume=float("nan"),
        prior_volume_median20=float("nan"),
        price_breakout=float("nan"),
        price_return20=float("nan"),
        volume_ratio20=float("nan"),
        share_filter_required=True,
        share_data_available=False,
        share_flow20=None,
        reason="预警计算不可用（RuntimeError）",
    )

    message = runner.format_w40_loss_notification(
        _w40_signal(), {}, [], warning
    )

    assert "突破200日前高：无法评估" in message
    assert "20日涨幅≥15%：无法评估" in message
    assert "创业板20日基金份额增长：无法评估" in message
    assert "nan" not in message.lower()


def test_w40_notification_shows_each_condition_for_2025_10_14() -> None:
    import run_daily_momentum_defender as runner

    warning = PeakWarning(
        asset="518880.SH",
        signal_date=date(2025, 10, 14),
        triggered=False,
        current_close=8.976,
        prior_high200=8.859,
        close20ago=7.847,
        current_volume=16_679_605.06,
        prior_volume_median20=4_420_063.92,
        price_breakout=0.0132069082289199,
        price_return20=0.14387664075442852,
        volume_ratio20=3.7736117309362354,
        share_filter_required=False,
        share_data_available=True,
        share_flow20=None,
        reason="至少一个价量条件尚未满足",
    )

    message = runner.format_w40_loss_notification(
        _w40_signal("518880.SH"), {}, [], warning
    )

    assert "突破200日前高：已满足；严格滞后前高 8.859" in message
    assert "20日涨幅≥15%：不满足；当前14.39%，还差0.61个百分点" in message
    assert "成交量/此前20日中位数≥1.50倍：已满足；当前3.77倍" in message
    assert "预警状态：未触发" in message


def test_current_formal_reason_labels_are_fixed_chinese() -> None:
    import run_daily_momentum_defender as runner

    expected = {
        "base_w40_momentum": "W40基础状态保持动量",
        "base_w40_defender": "W40基础状态保持防守",
        "asset_escape_break_defender_lock": "黄金满足逃生条件，打破防守锁",
        "asset_escape_hard_hold": "黄金逃生硬持有期",
        "asset_escape_momentum_hold": "黄金逃生继续持有",
        "asset_escape_return_disabled_top1": "Momentum Top1不再是黄金，返回防守",
        "asset_escape_return_below_y": "黄金相对Defender指标跌破退出线，返回防守",
        "base_w40_recovered_to_momentum": "W40基础状态恢复动量，结束黄金逃生",
    }

    assert {
        reason: runner.W40_STATE_REASON_LABELS[reason] for reason in expected
    } == expected


def test_signal_performance_sections_are_appended_with_relative_comparison() -> None:
    import run_daily_momentum_defender as runner

    snapshot = SignalPerformanceSnapshot(
        since_date=date(2026, 8, 11),
        current_holding_label="518880.SH",
        current_holding_return=0.10,
        concurrent_returns={
            "MOMENTUM": 0.08,
            "DEFENDER": 0.12,
            "510300.SH": 0.01,
        },
        period_returns={
            key: PeriodPerformance(0.01, 0.02, 0.03)
            for key in (
                "FORMAL",
                "LEGACY_MOMENTUM",
                "PURE_MOMENTUM",
                "PURE_DEFENDER",
            )
        },
    )
    signal = SimpleNamespace(
        performance_snapshot=snapshot,
        performance_error=None,
    )

    lines = runner._format_signal_performance_lines(signal)
    message = "\n".join(lines)

    assert "同期表现（自 2026-08-11 开盘起）" in message
    assert "518880 黄金ETF\u3000+10.00%\u3000|\u3000当前持仓" in message
    assert "Momentum\u3000+8.00%\u3000|\u3000落后持仓 -2.00% ↓" in message
    assert "Defender\u3000+12.00%\u3000|\u3000领先持仓 +2.00% ↑" in message
    assert "当前完整策略\u3000本月 +1.00%" in message
    assert "原非对数Momentum（模型回放）" in message
    assert "纯Momentum" in message
    assert "纯Defender" in message
