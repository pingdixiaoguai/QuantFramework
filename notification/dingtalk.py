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

    def send(self, message: str) -> None:
        """Send a markdown message to DingTalk."""
        url = self._sign_url()
        payload = json.dumps({
            "msgtype": "markdown",
            "markdown": {
                "title": "调仓通知",
                "text": message,
            },
        }).encode("utf-8")

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
