"""Tests for data.config token loading."""

import pytest

from data.config import get_tushare_token


class TestTokenFromEnvVar:
    def test_returns_token_from_env(self, monkeypatch):
        monkeypatch.setenv("TUSHARE_TOKEN", "fake")
        assert get_tushare_token() == "fake"


class TestTokenMissingRaises:
    def test_raises_when_no_token(self, monkeypatch):
        monkeypatch.delenv("TUSHARE_TOKEN", raising=False)
        # Also prevent dotenv from loading a real .env
        monkeypatch.setattr("data.config.load_dotenv", lambda: None)
        with pytest.raises(RuntimeError, match="TUSHARE_TOKEN missing"):
            get_tushare_token()
