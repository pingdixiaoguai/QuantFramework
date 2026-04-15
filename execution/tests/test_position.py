"""Tests for execution.position."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from execution.position import PositionPeriod, PositionState, read_position, save_position


@pytest.fixture
def tmp_state(monkeypatch, tmp_path):
    """Redirect STATE_FILE to a temp directory."""
    state_file = tmp_path / "current_position.json"
    monkeypatch.setattr("execution.position.STATE_FILE", state_file)
    return state_file


class TestReadPosition:
    def test_returns_empty_state_when_missing(self, tmp_state):
        result = read_position()
        assert isinstance(result, PositionState)
        assert result.weights == {}
        assert result.entry_date is None
        assert result.entry_prices is None
        assert result.ytd_history == []

    def test_migrates_old_format(self, tmp_state):
        # Old format: flat dict of weights, no "weights" key
        tmp_state.write_text('{"159915.SZ": 1.0}')
        result = read_position()
        assert isinstance(result, PositionState)
        assert result.weights == {"159915.SZ": 1.0}
        assert result.entry_date is None
        assert result.entry_prices is None
        assert result.ytd_history == []

    def test_reads_new_format_with_entry_metadata(self, tmp_state):
        data = {
            "weights": {"513100.SH": 1.0},
            "entry_date": "2026-04-08",
            "entry_prices": {"513100.SH": 1.234},
            "ytd_history": [],
        }
        tmp_state.write_text(json.dumps(data))
        result = read_position()
        assert result.weights == {"513100.SH": 1.0}
        assert result.entry_date == "2026-04-08"
        assert result.entry_prices == {"513100.SH": 1.234}
        assert result.ytd_history == []

    def test_reads_null_entry_fields(self, tmp_state):
        data = {
            "weights": {"513100.SH": 1.0},
            "entry_date": None,
            "entry_prices": None,
            "ytd_history": [],
        }
        tmp_state.write_text(json.dumps(data))
        result = read_position()
        assert result.entry_date is None
        assert result.entry_prices is None

    def test_reads_ytd_history(self, tmp_state):
        data = {
            "weights": {"513100.SH": 1.0},
            "entry_date": "2026-04-08",
            "entry_prices": {"513100.SH": 1.234},
            "ytd_history": [
                {
                    "weights": {"159915.SZ": 1.0},
                    "entry_date": "2026-01-02",
                    "exit_date": "2026-04-07",
                    "entry_prices": {"159915.SZ": 3.012},
                    "exit_prices": {"159915.SZ": 2.988},
                }
            ],
        }
        tmp_state.write_text(json.dumps(data))
        result = read_position()
        assert len(result.ytd_history) == 1
        period = result.ytd_history[0]
        assert isinstance(period, PositionPeriod)
        assert period.weights == {"159915.SZ": 1.0}
        assert period.entry_date == "2026-01-02"
        assert period.exit_date == "2026-04-07"
        assert period.entry_prices == {"159915.SZ": 3.012}
        assert period.exit_prices == {"159915.SZ": 2.988}


class TestSavePosition:
    def test_creates_file(self, tmp_state):
        save_position({"X.SH": 1.0})
        assert tmp_state.exists()

    def test_overwrites_previous(self, tmp_state):
        save_position({"A.SH": 1.0})
        save_position({"B.SH": 1.0})
        result = read_position()
        assert result.weights == {"B.SH": 1.0}
