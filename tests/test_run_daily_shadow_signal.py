"""The rolling OHLC ER shadow must never alter production orders."""

from __future__ import annotations

from datetime import date

import pandas as pd

from execution.interfaces import Order
from execution.position import PositionState
from strategy.rolling_ohlc_er import RollingWeights, SignalComparison


def test_shadow_target_is_not_used_for_execution(monkeypatch):
    import run_daily

    signal_date = date(2026, 7, 27)
    dates = pd.bdate_range("2026-06-01", periods=41)
    frame = pd.DataFrame(
        {
            "date": dates,
            "open": range(100, 141),
            "high": range(101, 142),
            "low": range(99, 140),
            "close": range(100, 141),
            "volume": [1_000.0] * len(dates),
        }
    )
    captured: dict[str, dict[str, float]] = {}

    class ProductionStrategy:
        def generate_weights(self, _):
            return {"510300.SH": 1.0}

    factor_metadata = {
        "name": "quality_momentum",
        "author": "test",
        "version": "1",
        "params": {},
        "min_history": 1,
        "direction": "higher_better",
        "description": "test",
    }

    def factor_compute(df, params=None):
        return pd.Series(1.0, index=df["date"], dtype=float)

    def capture_diff(target, current):
        captured["target"] = target
        captured["current"] = current
        return []

    shadow = SignalComparison(
        weights=RollingWeights(
            effective_date=date(2026, 7, 1),
            training_start=date(2022, 5, 5),
            training_end=date(2026, 6, 30),
            values=(0.853, 0.337, 0.029, 0.281),
        ),
        old_target="510300.SH",
        new_target="518880.SH",
        assets={
            asset: {
                "momentum": 0.01,
                "old_er": 0.2,
                "old_score": 0.002,
                "old_confidence": 0.25,
                "new_er": 0.1,
                "new_score": 0.001,
                "new_confidence": 0.25,
            }
            for asset in (
                "510300.SH",
                "159915.SZ",
                "513100.SH",
                "518880.SH",
            )
        },
    )

    monkeypatch.setattr(run_daily, "_sync_and_check", lambda *_: None)
    monkeypatch.setattr(
        run_daily,
        "_latest_common_data_date",
        lambda *_: signal_date,
    )
    monkeypatch.setattr(run_daily, "read_position", lambda *_: PositionState())
    monkeypatch.setattr(
        run_daily,
        "_backfill_open_prices",
        lambda state, *_: state,
    )
    monkeypatch.setattr(run_daily, "query", lambda *_: frame)
    monkeypatch.setattr(
        run_daily,
        "load_registered_factors",
        lambda: {
            "quality_momentum": {
                "METADATA": factor_metadata,
                "compute": factor_compute,
            }
        },
    )
    monkeypatch.setattr(run_daily, "validate", lambda *_: None)
    monkeypatch.setattr(run_daily, "load_strategy", lambda *_: ProductionStrategy())
    monkeypatch.setattr(
        run_daily,
        "build_signal_comparison",
        lambda *_args, **_kwargs: shadow,
    )
    monkeypatch.setattr(run_daily, "diff", capture_diff)
    monkeypatch.setattr(run_daily, "_compute_position_return", lambda *_: None)
    monkeypatch.setattr(run_daily, "_compute_benchmark_returns", lambda *_: {})
    monkeypatch.setattr(run_daily, "_compute_ytd_return", lambda *_: None)
    monkeypatch.setattr(run_daily, "format_notification", lambda *_: "message")

    run_daily.run(
        {
            "strategy_name": "quality_momentum_top1",
            "strategy_class": "strategy.top1.Top1",
            "asset_pool": [
                "510300.SH",
                "159915.SZ",
                "513100.SH",
                "518880.SH",
            ],
            "start": date(2026, 6, 1),
            "factors": [{"name": "quality_momentum"}],
            "rebalance_days": 5,
            "enable_dingtalk": False,
            "shadow_rolling_ohlc_er": {
                "enabled": True,
                "asset_pool": [
                    "510300.SH",
                    "159915.SZ",
                    "513100.SH",
                    "518880.SH",
                ],
            },
        }
    )

    assert captured["target"] == {"510300.SH": 1.0}
    assert captured["current"] == {}
    assert shadow.new_target == "518880.SH"


def test_notification_only_sends_test_without_writing_position(monkeypatch):
    import run_daily

    signal_date = date(2026, 7, 27)
    dates = pd.bdate_range("2026-06-01", periods=41)
    frame = pd.DataFrame(
        {
            "date": dates,
            "open": range(100, 141),
            "high": range(101, 142),
            "low": range(99, 140),
            "close": range(100, 141),
            "volume": [1_000.0] * len(dates),
        }
    )
    sent: dict[str, str] = {}

    class ProductionStrategy:
        def generate_weights(self, _):
            return {"510300.SH": 1.0}

    class FakeNotifier:
        def send(self, message, *, title, alert_text):
            sent.update(
                message=message,
                title=title,
                alert_text=alert_text,
            )

    factor_metadata = {
        "name": "quality_momentum",
        "author": "test",
        "version": "1",
        "params": {},
        "min_history": 1,
        "direction": "higher_better",
        "description": "test",
    }

    monkeypatch.setattr(run_daily, "_sync_and_check", lambda *_: None)
    monkeypatch.setattr(
        run_daily,
        "_latest_common_data_date",
        lambda *_: signal_date,
    )
    monkeypatch.setattr(run_daily, "read_position", lambda *_: PositionState())
    monkeypatch.setattr(
        run_daily,
        "_backfill_open_prices",
        lambda *_: (_ for _ in ()).throw(AssertionError("backfill called")),
    )
    monkeypatch.setattr(run_daily, "query", lambda *_: frame)
    monkeypatch.setattr(
        run_daily,
        "load_registered_factors",
        lambda: {
            "quality_momentum": {
                "METADATA": factor_metadata,
                "compute": lambda df, params=None: pd.Series(
                    1.0,
                    index=df["date"],
                    dtype=float,
                ),
            }
        },
    )
    monkeypatch.setattr(run_daily, "validate", lambda *_: None)
    monkeypatch.setattr(run_daily, "load_strategy", lambda *_: ProductionStrategy())
    monkeypatch.setattr(
        run_daily,
        "diff",
        lambda *_: [Order("510300.SH", "buy", 1.0)],
    )
    monkeypatch.setattr(run_daily, "_compute_position_return", lambda *_: None)
    monkeypatch.setattr(run_daily, "_compute_benchmark_returns", lambda *_: {})
    monkeypatch.setattr(run_daily, "_compute_ytd_return", lambda *_: None)
    monkeypatch.setattr(run_daily, "format_notification", lambda *_: "message")
    monkeypatch.setattr(run_daily, "DingTalkNotifier", FakeNotifier)
    monkeypatch.setattr(
        run_daily,
        "_next_entry_date",
        lambda *_: (_ for _ in ()).throw(AssertionError("persistence reached")),
    )

    run_daily.run(
        {
            "strategy_name": "quality_momentum_top1",
            "strategy_class": "strategy.top1.Top1",
            "asset_pool": ["510300.SH"],
            "start": date(2026, 6, 1),
            "factors": [{"name": "quality_momentum"}],
            "rebalance_days": 1,
            "enable_dingtalk": True,
        },
        notification_only=True,
    )

    assert sent["title"] == "通知测试（无需操作）"
    assert sent["alert_text"] == "钉钉消息测试，请忽略，无需操作。"
    assert "不保存或回填持仓" in sent["message"]
