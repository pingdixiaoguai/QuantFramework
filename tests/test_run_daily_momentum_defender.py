from __future__ import annotations

from datetime import date

from defender.live import DefenderNextOpenTarget
from execution.position import PositionState
from strategy.momentum_defender import (
    IntegratedNextOpenSignal,
    MomentumNextOpenTarget,
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


def test_gold_strategy_mode_dispatches_formal_builder(monkeypatch) -> None:
    import run_daily_momentum_defender as runner

    expected = object()
    monkeypatch.setattr(
        runner,
        "build_formal_gold_next_open_signal",
        lambda *_: expected,
    )

    actual = runner._build_signal(
        {"strategy_mode": "gold_raqm_w5"},
        date(2026, 8, 21),
        date(2026, 8, 24),
    )

    assert actual is expected
