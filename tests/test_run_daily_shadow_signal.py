"""Read-only standard-pipeline shadow signals must not alter production orders."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from execution.position import PositionState


ASSETS = ["510300.SH", "159915.SZ", "513100.SH", "518880.SH"]


def _frame() -> pd.DataFrame:
    dates = pd.bdate_range("2026-06-01", periods=41)
    return pd.DataFrame(
        {
            "date": dates,
            "open": range(100, 141),
            "high": range(101, 142),
            "low": range(99, 140),
            "close": range(100, 141),
            "volume": [1_000.0] * len(dates),
        }
    )


def _factor_module():
    metadata = {
        "name": "quality_momentum",
        "author": "test",
        "version": "1",
        "params": {},
        "min_history": 1,
        "direction": "higher_better",
        "description": "test",
    }

    def compute(df, params=None):
        # Make the shadow strategy's values valid without touching the network.
        values = {
            "510300.SH": 0.10,
            "159915.SZ": 0.20,
            "513100.SH": 0.30,
            "518880.SH": 0.40,
        }
        value = values.get(str(df.attrs.get("asset", "")), 1.0)
        return pd.Series(value, index=df["date"], dtype=float)

    return {"METADATA": metadata, "compute": compute}


def _base_config():
    return {
        "strategy_name": "quality_momentum_top1",
        "strategy_class": "strategy.top1.Top1",
        "asset_pool": ASSETS,
        "start": date(2026, 6, 1),
        "factors": [{"name": "quality_momentum"}],
        "rebalance_days": 5,
        "enable_dingtalk": False,
    }


def _patch_common(monkeypatch, run_daily, captured, shadow_target="518880.SH"):
    signal_date = date(2026, 7, 27)
    frame = _frame()

    class Strategy:
        def __init__(self, target):
            self.target = target

        def generate_weights(self, _):
            return {self.target: 1.0}

    monkeypatch.setattr(run_daily, "_is_sse_trading_day", lambda _: True)
    monkeypatch.setattr(run_daily, "_sync_and_check", lambda *_: None)
    monkeypatch.setattr(
        run_daily, "_latest_common_data_date", lambda *_: signal_date
    )
    monkeypatch.setattr(run_daily, "read_position", lambda *_: PositionState())
    monkeypatch.setattr(run_daily, "query", lambda *_: frame)
    monkeypatch.setattr(
        run_daily,
        "load_registered_factors",
        lambda: {"quality_momentum": _factor_module()},
    )
    monkeypatch.setattr(run_daily, "validate", lambda *_: None)
    monkeypatch.setattr(
        run_daily,
        "_load_config",
        lambda path: {
            "strategy_name": "quality_momentum_top1_ohlc_er",
            "strategy_class": "strategy.top1.Top1",
            "asset_pool": ASSETS,
            "start": date(2026, 6, 1),
            "factors": [{"name": "quality_momentum"}],
        },
    )
    monkeypatch.setattr(
        run_daily,
        "load_strategy",
        lambda config: Strategy(
            shadow_target
            if config["strategy_name"] == "quality_momentum_top1_ohlc_er"
            else "510300.SH"
        ),
    )
    monkeypatch.setattr(run_daily, "diff", lambda target, current: captured.update(
        target=target, current=current
    ) or [])
    monkeypatch.setattr(run_daily, "_compute_position_return", lambda *_: None)
    monkeypatch.setattr(run_daily, "_compute_benchmark_returns", lambda *_: {})
    monkeypatch.setattr(run_daily, "_compute_ytd_return", lambda *_: None)
    monkeypatch.setattr(
        run_daily,
        "format_notification",
        lambda context: captured.update(context=context) or "message",
    )


def test_shadow_target_is_not_used_for_execution(monkeypatch):
    import run_daily

    captured = {}
    _patch_common(monkeypatch, run_daily, captured)

    run_daily.run(
        _base_config(),
        shadow_config_paths=[Path("shadow.yaml")],
    )

    assert captured["target"] == {"510300.SH": 1.0}
    assert captured["current"] == {}
    assert captured["context"].old_signal_target == "510300.SH"
    assert captured["context"].new_signal_target == "518880.SH"


def test_notification_only_works_on_holiday_without_state_write(monkeypatch):
    import run_daily

    captured = {}
    _patch_common(monkeypatch, run_daily, captured)
    monkeypatch.setattr(run_daily, "_is_sse_trading_day", lambda _: False)
    monkeypatch.setattr(
        run_daily,
        "DingTalkNotifier",
        lambda: type(
            "Notifier",
            (),
            {"send": lambda self, *args, **kwargs: captured.update(sent=True)},
        )(),
    )
    monkeypatch.setattr(
        run_daily,
        "_backfill_open_prices",
        lambda *_: (_ for _ in ()).throw(AssertionError("backfill called")),
    )
    monkeypatch.setattr(
        run_daily,
        "_next_entry_date",
        lambda *_: (_ for _ in ()).throw(AssertionError("persistence reached")),
    )

    config = _base_config()
    config["enable_dingtalk"] = True
    run_daily.run(
        config,
        notification_only=True,
        shadow_config_paths=[Path("shadow.yaml")],
    )

    assert captured["sent"] is True
    assert captured["context"].signal_date == date(2026, 7, 27)
