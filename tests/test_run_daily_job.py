"""Tests for the unattended daily-job wrapper."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from scripts import run_daily_job


def test_run_job_returns_on_first_success_without_alert(monkeypatch):
    calls = []
    alerts = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(run_daily_job.subprocess, "run", fake_run)
    monkeypatch.setattr(
        run_daily_job,
        "_send_failure_alert",
        lambda *args: alerts.append(args),
    )

    result = run_daily_job.run_job(
        Path("strategy.yaml"),
        shadow_config=run_daily_job.DEFAULT_SHADOW_CONFIG,
        retry_delay=0,
    )

    assert result == 0
    assert len(calls) == 1
    assert calls[0][0][-4:] == [
        "--config",
        "strategy.yaml",
        "--shadow-config",
        str(run_daily_job.DEFAULT_SHADOW_CONFIG),
    ]
    assert calls[0][1]["cwd"] == run_daily_job.PROJECT_ROOT
    assert calls[0][1]["check"] is False
    assert alerts == []


def test_run_job_retries_then_sends_one_final_alert(monkeypatch):
    return_codes = iter([2, 3, 4])
    sleeps = []
    alerts = []

    monkeypatch.setattr(
        run_daily_job.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=next(return_codes)),
    )
    monkeypatch.setattr(
        run_daily_job.time,
        "sleep",
        lambda seconds: sleeps.append(seconds),
    )
    monkeypatch.setattr(
        run_daily_job,
        "_send_failure_alert",
        lambda *args: alerts.append(args),
    )

    result = run_daily_job.run_job(
        Path("strategy.yaml"),
        attempts=3,
        retry_delay=10,
    )

    assert result == 4
    assert sleeps == [10, 10]
    assert alerts == [(Path("strategy.yaml"), 3, "exit code 4")]


def test_final_alert_uses_plain_text_dingtalk_alert(monkeypatch):
    messages = []

    class FakeNotifier:
        def send_alert(self, message):
            messages.append(message)

    monkeypatch.setattr(run_daily_job, "DingTalkNotifier", FakeNotifier)
    monkeypatch.setattr(run_daily_job.socket, "gethostname", lambda: "quant-host")

    run_daily_job._send_failure_alert(
        Path("strategy.yaml"),
        3,
        "exit code 1",
    )

    assert len(messages) == 1
    assert "quant-host" in messages[0]
    assert "exit code 1" in messages[0]
    assert "journalctl -u quant-daily.service" in messages[0]


def test_formal_config_dispatches_composite_runner(monkeypatch, tmp_path):
    calls = []
    config = tmp_path / "formal.yaml"
    config.write_text("strategy_mode: gold_raqm_w5\n", encoding="utf-8")
    monkeypatch.setattr(
        run_daily_job.subprocess,
        "run",
        lambda command, **kwargs: calls.append(command)
        or SimpleNamespace(returncode=0),
    )

    result = run_daily_job.run_job(config, retry_delay=0)

    assert result == 0
    assert calls[0][1].endswith("run_daily_momentum_defender.py")


def test_downside_raqm_config_dispatches_composite_runner(monkeypatch, tmp_path):
    calls = []
    config = tmp_path / "formal.yaml"
    config.write_text("strategy_mode: downside_raqm\n", encoding="utf-8")
    monkeypatch.setattr(
        run_daily_job.subprocess,
        "run",
        lambda command, **kwargs: calls.append(command)
        or SimpleNamespace(returncode=0),
    )

    result = run_daily_job.run_job(config, retry_delay=0)

    assert result == 0
    assert calls[0][1].endswith("run_daily_momentum_defender.py")


def test_w40_loss_config_dispatches_composite_runner(monkeypatch, tmp_path):
    calls = []
    config = tmp_path / "formal.yaml"
    config.write_text("strategy_mode: w40_loss\n", encoding="utf-8")
    monkeypatch.setattr(
        run_daily_job.subprocess,
        "run",
        lambda command, **kwargs: calls.append(command)
        or SimpleNamespace(returncode=0),
    )
    assert run_daily_job.run_job(config, retry_delay=0) == 0
    assert calls[0][1].endswith("run_daily_momentum_defender.py")


def test_w40_full_equity_config_dispatches_composite_runner(monkeypatch, tmp_path):
    calls = []
    config = tmp_path / "formal.yaml"
    config.write_text(
        "strategy_mode: w40_reversal_full_equity\n", encoding="utf-8"
    )
    monkeypatch.setattr(
        run_daily_job.subprocess,
        "run",
        lambda command, **kwargs: calls.append(command)
        or SimpleNamespace(returncode=0),
    )
    assert run_daily_job.run_job(config, retry_delay=0) == 0
    assert calls[0][1].endswith("run_daily_momentum_defender.py")


def test_w40_gold_escape_config_dispatches_composite_runner(monkeypatch, tmp_path):
    calls = []
    config = tmp_path / "formal.yaml"
    config.write_text("strategy_mode: w40_gold_qm20_escape\n", encoding="utf-8")
    monkeypatch.setattr(
        run_daily_job.subprocess,
        "run",
        lambda command, **kwargs: calls.append(command)
        or SimpleNamespace(returncode=0),
    )
    assert run_daily_job.run_job(config, retry_delay=0) == 0
    assert calls[0][1].endswith("run_daily_momentum_defender.py")


def test_w40_qm40_signed_exit_config_dispatches_composite_runner(
    monkeypatch, tmp_path
):
    calls = []
    config = tmp_path / "formal.yaml"
    config.write_text("strategy_mode: w40_qm40_signed_exit\n", encoding="utf-8")
    monkeypatch.setattr(
        run_daily_job.subprocess,
        "run",
        lambda command, **kwargs: calls.append(command)
        or SimpleNamespace(returncode=0),
    )
    assert run_daily_job.run_job(config, retry_delay=0) == 0
    assert calls[0][1].endswith("run_daily_momentum_defender.py")


def test_w40_qm40_threshold_config_dispatches_composite_runner(
    monkeypatch, tmp_path
):
    calls = []
    config = tmp_path / "formal.yaml"
    config.write_text("strategy_mode: w40_qm40_threshold\n", encoding="utf-8")
    monkeypatch.setattr(
        run_daily_job.subprocess,
        "run",
        lambda command, **kwargs: calls.append(command)
        or SimpleNamespace(returncode=0),
    )
    assert run_daily_job.run_job(config, retry_delay=0) == 0
    assert calls[0][1].endswith("run_daily_momentum_defender.py")


def test_formal_config_rejects_shadow_config(tmp_path):
    import pytest

    config = tmp_path / "formal.yaml"
    config.write_text("strategy_mode: gold_raqm_w5\n", encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported"):
        run_daily_job.run_job(
            config,
            shadow_config=Path("shadow.yaml"),
            retry_delay=0,
        )
