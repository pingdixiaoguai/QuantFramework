"""Tests for notification.dingtalk."""

import json

import pytest

from notification.dingtalk import DingTalkNotifier


class TestDingTalkInit:
    def test_raises_without_url(self, monkeypatch):
        monkeypatch.delenv("DINGTALK_WEBHOOK", raising=False)
        with pytest.raises(ValueError, match="webhook URL required"):
            DingTalkNotifier()

    def test_accepts_explicit_url(self):
        n = DingTalkNotifier(webhook_url="https://example.com/webhook")
        assert n.webhook_url == "https://example.com/webhook"

    def test_reads_env_var(self, monkeypatch):
        monkeypatch.setenv("DINGTALK_WEBHOOK", "https://env.example.com")
        n = DingTalkNotifier()
        assert n.webhook_url == "https://env.example.com"


class TestDingTalkSign:
    def test_no_secret_returns_original_url(self):
        n = DingTalkNotifier(webhook_url="https://example.com/webhook")
        assert n._sign_url() == "https://example.com/webhook"

    def test_secret_appends_timestamp_and_sign(self):
        n = DingTalkNotifier(
            webhook_url="https://example.com/webhook",
            secret="SEC_test_secret",
        )
        url = n._sign_url()
        assert "timestamp=" in url
        assert "sign=" in url


class TestDingTalkSend:
    def test_send_posts_correct_payload(self, monkeypatch):
        """Verify send() builds the right JSON payload."""
        captured = {}

        class FakeResponse:
            def read(self):
                return json.dumps({"errcode": 0, "errmsg": "ok"}).encode()
            def __enter__(self):
                return self
            def __exit__(self, *a):
                pass

        def fake_urlopen(req):
            captured["url"] = req.full_url
            captured["data"] = json.loads(req.data.decode())
            captured["headers"] = dict(req.headers)
            return FakeResponse()

        monkeypatch.setattr("notification.dingtalk.urllib.request.urlopen", fake_urlopen)

        n = DingTalkNotifier(webhook_url="https://example.com/webhook")
        n.send("hello world")

        assert captured["data"]["msgtype"] == "markdown"
        assert captured["data"]["markdown"]["text"] == "hello world"
        assert captured["headers"]["Content-type"] == "application/json"
        assert captured["data"]["at"]["isAtAll"] is True

    def test_send_raises_on_api_error(self, monkeypatch):
        class FakeResponse:
            def read(self):
                return json.dumps({"errcode": 310000, "errmsg": "bad token"}).encode()
            def __enter__(self):
                return self
            def __exit__(self, *a):
                pass

        monkeypatch.setattr(
            "notification.dingtalk.urllib.request.urlopen",
            lambda req: FakeResponse(),
        )

        n = DingTalkNotifier(webhook_url="https://example.com/webhook")
        with pytest.raises(RuntimeError, match="bad token"):
            n.send("test")
