"""Tests for execution.position."""

from pathlib import Path

import pytest

from execution.position import read_position, save_position


@pytest.fixture
def tmp_state(monkeypatch, tmp_path):
    """Redirect STATE_FILE to a temp directory."""
    state_file = tmp_path / "current_position.json"
    monkeypatch.setattr("execution.position.STATE_FILE", state_file)
    return state_file


class TestReadPosition:
    def test_returns_empty_when_missing(self, tmp_state):
        assert read_position() == {}

    def test_reads_saved_weights(self, tmp_state):
        save_position({"A.SH": 0.6, "B.SH": 0.4})
        result = read_position()
        assert result == {"A.SH": 0.6, "B.SH": 0.4}


class TestSavePosition:
    def test_creates_file(self, tmp_state):
        save_position({"X.SH": 1.0})
        assert tmp_state.exists()

    def test_overwrites_previous(self, tmp_state):
        save_position({"A.SH": 1.0})
        save_position({"B.SH": 1.0})
        assert read_position() == {"B.SH": 1.0}
