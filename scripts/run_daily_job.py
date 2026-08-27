"""Production wrapper for the unattended daily signal job.

Runs ``run_daily.py`` in a child process, retries transient failures, and sends
one DingTalk alert only after the final failed attempt.  Child output is left
attached to stdout/stderr so systemd journals retain the full traceback.
"""

from __future__ import annotations

import argparse
import socket
import subprocess
import sys
import time
from pathlib import Path

import yaml

from notification.dingtalk import DingTalkNotifier


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = Path(
    "strategy/configs/momentum_defender_w40_gold_escape.yaml"
)
DEFAULT_SHADOW_CONFIG = Path("strategy/configs/quality_momentum_top1_ohlc_er.yaml")


def _send_failure_alert(
    config: Path,
    attempts: int,
    failure_detail: str,
) -> None:
    message = "\n".join(
        [
            "⚠️ QuantFramework 早盘任务最终失败",
            f"主机：{socket.gethostname()}",
            f"策略配置：{config}",
            f"尝试次数：{attempts}",
            f"最后结果：{failure_detail}",
            "请登录服务器执行：",
            "journalctl -u quant-daily.service -n 200 --no-pager",
        ]
    )
    try:
        DingTalkNotifier().send_alert(message)
        print("Final failure alert sent to DingTalk.", file=sys.stderr, flush=True)
    except Exception as exc:
        print(
            f"Unable to send final DingTalk failure alert: {exc}",
            file=sys.stderr,
            flush=True,
        )


def run_job(
    config: Path,
    *,
    shadow_config: Path | None = None,
    attempts: int = 3,
    retry_delay: float = 600,
) -> int:
    """Run the daily entry point and return its final process exit code."""
    if attempts < 1:
        raise ValueError("attempts must be >= 1")
    if retry_delay < 0:
        raise ValueError("retry_delay must be >= 0")

    config_path = config if config.is_absolute() else PROJECT_ROOT / config
    entrypoint = PROJECT_ROOT / "run_daily.py"
    if config_path.exists():
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        composite_modes = {
            "gold_raqm_w5",
            "absolute_stability_raw_gold",
            "confirmation_bridge_raw_gold",
            "downside_raqm",
            "w40_loss",
            "w40_reversal_full_equity",
            "w40_gold_qm20_escape",
            "w40_qm40_signed_exit",
            "w40_qm40_threshold",
        }
        if isinstance(loaded, dict) and loaded.get("strategy_mode") in composite_modes:
            entrypoint = PROJECT_ROOT / "run_daily_momentum_defender.py"
            if shadow_config is not None:
                raise ValueError(
                    "shadow_config is unsupported by the formal composite runner"
                )
    command = [
        sys.executable,
        str(entrypoint),
        "--config",
        str(config),
    ]
    if shadow_config is not None:
        command.extend(["--shadow-config", str(shadow_config)])
    last_exit_code = 1
    failure_detail = "not started"

    for attempt in range(1, attempts + 1):
        print(
            f"Starting daily signal attempt {attempt}/{attempts}.",
            flush=True,
        )
        try:
            result = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
            last_exit_code = result.returncode if result.returncode > 0 else 1
            if result.returncode == 0:
                print("Daily signal job completed successfully.", flush=True)
                return 0
            failure_detail = f"exit code {result.returncode}"
        except OSError as exc:
            last_exit_code = 1
            failure_detail = f"{type(exc).__name__}: {exc}"
            print(
                f"Unable to start daily signal process: {exc}",
                file=sys.stderr,
                flush=True,
            )

        if attempt < attempts:
            print(
                f"Attempt {attempt} failed ({failure_detail}); "
                f"retrying in {retry_delay:g} seconds.",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(retry_delay)

    _send_failure_alert(config, attempts, failure_detail)
    return last_exit_code


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the daily signal with retries and final failure alerting."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Strategy YAML passed to run_daily.py.",
    )
    parser.add_argument(
        "--attempts",
        type=int,
        default=3,
        help="Total attempts before sending the failure alert.",
    )
    parser.add_argument(
        "--shadow-config",
        type=Path,
        default=None,
        help=(
            "Optional independent read-only strategy YAML used for notification "
            "comparison."
        ),
    )
    parser.add_argument(
        "--retry-delay",
        type=float,
        default=600,
        help="Seconds between attempts.",
    )
    args = parser.parse_args()
    raise SystemExit(
        run_job(
            args.config,
            shadow_config=args.shadow_config,
            attempts=args.attempts,
            retry_delay=args.retry_delay,
        )
    )


if __name__ == "__main__":
    main()
