"""Tests for notification.dingtalk."""

import json

import pytest

from notification.dingtalk import DingTalkNotifier


class TestDingTalkInit:
    def test_raises_without_url(self, monkeypatch):
        monkeypatch.delenv("DINGTALK_WEBHOOK", raising=False)
        monkeypatch.setattr("notification.dingtalk.load_dotenv", lambda: None)
        with pytest.raises(ValueError, match="webhook URL required"):
            DingTalkNotifier()

    def test_accepts_explicit_url(self):
        n = DingTalkNotifier(webhook_url="https://example.com/webhook")
        assert n.webhook_url == "https://example.com/webhook"

    def test_reads_env_var(self, monkeypatch):
        monkeypatch.setenv("DINGTALK_WEBHOOK", "https://env.example.com")
        n = DingTalkNotifier()
        assert n.webhook_url == "https://env.example.com"

    def test_rejects_example_placeholders(self):
        with pytest.raises(ValueError, match="access-token placeholder"):
            DingTalkNotifier(
                webhook_url=(
                    "https://oapi.dingtalk.com/robot/send"
                    "?access_token=your_token_here"
                ),
            )

        with pytest.raises(ValueError, match="signing secret"):
            DingTalkNotifier(
                webhook_url="https://example.com/webhook",
                secret="your_secret_here",
            )

    def test_loads_dotenv_before_reading_env(self, monkeypatch):
        monkeypatch.delenv("DINGTALK_WEBHOOK", raising=False)

        def fake_load_dotenv():
            monkeypatch.setenv("DINGTALK_WEBHOOK", "https://dotenv.example.com")

        monkeypatch.setattr("notification.dingtalk.load_dotenv", fake_load_dotenv)

        n = DingTalkNotifier()

        assert n.webhook_url == "https://dotenv.example.com"


class TestDingTalkSign:
    def test_no_secret_returns_original_url(self, monkeypatch):
        monkeypatch.delenv("DINGTALK_SECRET", raising=False)
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
        """Verify send() posts the markdown card followed by the @所有人 text ping."""
        captured = []

        class FakeResponse:
            def read(self):
                return json.dumps({"errcode": 0, "errmsg": "ok"}).encode()
            def __enter__(self):
                return self
            def __exit__(self, *a):
                pass

        def fake_urlopen(req):
            captured.append({
                "url": req.full_url,
                "data": json.loads(req.data.decode()),
                "headers": dict(req.headers),
            })
            return FakeResponse()

        monkeypatch.setattr("notification.dingtalk.urllib.request.urlopen", fake_urlopen)

        n = DingTalkNotifier(webhook_url="https://example.com/webhook")
        n.send("hello world")

        assert len(captured) == 2

        markdown_call = captured[0]
        assert markdown_call["data"]["msgtype"] == "markdown"
        assert markdown_call["data"]["markdown"]["text"] == "hello world"
        assert markdown_call["headers"]["Content-type"] == "application/json"

        at_all_call = captured[1]
        assert at_all_call["data"]["msgtype"] == "text"
        assert at_all_call["data"]["at"]["isAtAll"] is True

    def test_send_accepts_test_title_and_alert(self, monkeypatch):
        captured = []

        class FakeResponse:
            def read(self):
                return json.dumps({"errcode": 0, "errmsg": "ok"}).encode()
            def __enter__(self):
                return self
            def __exit__(self, *a):
                pass

        def fake_urlopen(req):
            captured.append(json.loads(req.data.decode()))
            return FakeResponse()

        monkeypatch.setattr("notification.dingtalk.urllib.request.urlopen", fake_urlopen)

        n = DingTalkNotifier(webhook_url="https://example.com/webhook")
        n.send(
            "test body",
            title="通知测试（无需操作）",
            alert_text="钉钉消息测试，请忽略，无需操作。",
        )

        assert captured[0]["markdown"]["title"] == "通知测试（无需操作）"
        assert captured[1]["text"]["content"] == "钉钉消息测试，请忽略，无需操作。"

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

    def test_send_alert_posts_one_text_at_all_payload(self, monkeypatch):
        captured = []

        class FakeResponse:
            def read(self):
                return json.dumps({"errcode": 0, "errmsg": "ok"}).encode()
            def __enter__(self):
                return self
            def __exit__(self, *a):
                pass

        def fake_urlopen(req):
            captured.append(json.loads(req.data.decode()))
            return FakeResponse()

        monkeypatch.setattr("notification.dingtalk.urllib.request.urlopen", fake_urlopen)

        n = DingTalkNotifier(webhook_url="https://example.com/webhook")
        n.send_alert("任务失败")

        assert captured == [
            {
                "msgtype": "text",
                "text": {"content": "任务失败"},
                "at": {"isAtAll": True},
            }
        ]
