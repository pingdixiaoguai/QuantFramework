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

from notification.dingtalk import DingTalkNotifier


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = Path("strategy/configs/quality_momentum_top1.yaml")


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
    attempts: int = 3,
    retry_delay: float = 600,
) -> int:
    """Run the daily entry point and return its final process exit code."""
    if attempts < 1:
        raise ValueError("attempts must be >= 1")
    if retry_delay < 0:
        raise ValueError("retry_delay must be >= 0")

    command = [
        sys.executable,
        str(PROJECT_ROOT / "run_daily.py"),
        "--config",
        str(config),
    ]
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
        "--retry-delay",
        type=float,
        default=600,
        help="Seconds between attempts.",
    )
    args = parser.parse_args()
    raise SystemExit(
        run_job(
            args.config,
            attempts=args.attempts,
            retry_delay=args.retry_delay,
        )
    )


if __name__ == "__main__":
    main()
