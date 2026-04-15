"""DingTalk robot webhook notifier."""

from __future__ import annotations

import hashlib
import hmac
import base64
import time
import urllib.parse
import urllib.request
import json
import os

from notification.interfaces import Notifier


class DingTalkNotifier(Notifier):
    """Send messages via DingTalk custom robot webhook.

    Args:
        webhook_url: DingTalk robot webhook URL.
            Defaults to DINGTALK_WEBHOOK env var.
        secret: HMAC-SHA256 signing secret for the robot.
            Defaults to DINGTALK_SECRET env var (optional).
    """

    def __init__(
        self,
        webhook_url: str | None = None,
        secret: str | None = None,
    ):
        self.webhook_url = webhook_url or os.environ.get("DINGTALK_WEBHOOK", "")
        self.secret = secret or os.environ.get("DINGTALK_SECRET")

        if not self.webhook_url:
            raise ValueError(
                "DingTalk webhook URL required. "
                "Set DINGTALK_WEBHOOK env var or pass webhook_url."
            )

    def _sign_url(self) -> str:
        """Append HMAC-SHA256 timestamp+sign to webhook URL if secret is set."""
        if not self.secret:
            return self.webhook_url

        timestamp = str(int(time.time() * 1000))
        string_to_sign = f"{timestamp}\n{self.secret}"
        hmac_code = hmac.new(
            self.secret.encode("utf-8"),
            string_to_sign.encode("utf-8"),
            digestmod=hashlib.sha256,
        ).digest()
        sign = urllib.parse.quote_plus(base64.b64encode(hmac_code).decode())
        return f"{self.webhook_url}&timestamp={timestamp}&sign={sign}"

    def _post(self, payload: bytes) -> None:
        """POST a JSON payload to the signed webhook URL."""
        url = self._sign_url()
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req) as resp:
            result = json.loads(resp.read().decode())
            if result.get("errcode") != 0:
                raise RuntimeError(
                    f"DingTalk API error: {result.get('errmsg', 'unknown')}"
                )

    def send(self, message: str) -> None:
        """Send a markdown card followed by a separate @所有人 text message."""
        # 1. Send the markdown card (no @mention inside)
        self._post(json.dumps({
            "msgtype": "markdown",
            "markdown": {
                "title": "调仓信号",
                "text": message,
            },
        }).encode("utf-8"))

        # 2. Send a plain text message to trigger @所有人 notification
        #    DingTalk only reliably fires the group-wide alert for text type.
        self._post(json.dumps({
            "msgtype": "text",
            "text": {"content": "请查看今日调仓信号，及时操作！"},
            "at": {"isAtAll": True},
        }).encode("utf-8"))
