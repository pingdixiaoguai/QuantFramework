"""Tests for execution.position."""

from __future__ import annotations

import json
from datetime import date

import pytest

from execution.position import PositionPeriod, PositionState, read_position, save_position

_STRAT = "test_strategy"


@pytest.fixture
def tmp_state(monkeypatch, tmp_path):
    """Redirect state directory to a temp directory."""
    monkeypatch.setattr("execution.position._STATE_DIR", tmp_path)
    return tmp_path / f"{_STRAT}_position.json"


class TestReadPosition:
    def test_returns_empty_state_when_missing(self, tmp_state):
        result = read_position(_STRAT)
        assert isinstance(result, PositionState)
        assert result.weights == {}
        assert result.entry_date is None
        assert result.entry_prices is None
        assert result.ytd_history == []

    def test_migrates_old_format(self, tmp_state):
        # Old format: flat dict of weights, no "weights" key
        tmp_state.write_text('{"159915.SZ": 1.0}')
        result = read_position(_STRAT)
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
        result = read_position(_STRAT)
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
        result = read_position(_STRAT)
        assert result.entry_date is None
        assert result.entry_prices is None

    def test_reads_ytd_entry_basis(self, tmp_state):
        data = {
            "weights": {"159915.SZ": 1.0},
            "entry_date": "2025-12-30",
            "entry_prices": {"159915.SZ": 3.0},
            "ytd_entry_date": "2026-01-05",
            "ytd_entry_prices": {"159915.SZ": 3.2},
            "ytd_history": [],
        }
        tmp_state.write_text(json.dumps(data))
        result = read_position(_STRAT)
        assert result.entry_date == "2025-12-30"
        assert result.entry_prices == {"159915.SZ": 3.0}
        assert result.ytd_entry_date == "2026-01-05"
        assert result.ytd_entry_prices == {"159915.SZ": 3.2}

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
        result = read_position(_STRAT)
        assert len(result.ytd_history) == 1
        period = result.ytd_history[0]
        assert isinstance(period, PositionPeriod)
        assert period.weights == {"159915.SZ": 1.0}
        assert period.entry_date == "2026-01-02"
        assert period.exit_date == "2026-04-07"
        assert period.entry_prices == {"159915.SZ": 3.012}
        assert period.exit_prices == {"159915.SZ": 2.988}

    def test_reads_ytd_history_with_null_exit_prices(self, tmp_state):
        # exit_prices may be null while awaiting backfill
        data = {
            "weights": {"513100.SH": 1.0},
            "entry_date": "2026-04-09",
            "entry_prices": None,
            "ytd_history": [
                {
                    "weights": {"159915.SZ": 1.0},
                    "entry_date": "2026-01-02",
                    "exit_date": "2026-04-09",
                    "entry_prices": {"159915.SZ": 3.012},
                    "exit_prices": None,
                }
            ],
        }
        tmp_state.write_text(json.dumps(data))
        result = read_position(_STRAT)
        assert result.ytd_history[0].exit_prices is None


class TestSavePosition:
    def test_creates_file(self, tmp_state):
        save_position({"X.SH": 1.0}, date(2026, 4, 9), _STRAT)
        assert tmp_state.exists()

    def test_writes_new_format(self, tmp_state):
        save_position({"513100.SH": 1.0}, date(2026, 4, 9), _STRAT)
        result = read_position(_STRAT)
        assert result.weights == {"513100.SH": 1.0}
        assert result.entry_date == "2026-04-09"
        assert result.entry_prices is None
        assert result.ytd_history == []

    def test_entry_prices_stored_when_provided(self, tmp_state):
        save_position(
            {"513100.SH": 1.0},
            date(2026, 4, 9),
            _STRAT,
            entry_prices={"513100.SH": 1.234},
        )
        result = read_position(_STRAT)
        assert result.entry_prices == {"513100.SH": 1.234}

    def test_archives_old_position_to_ytd_history(self, tmp_state):
        # Seed existing position in new format
        existing = {
            "weights": {"159915.SZ": 1.0},
            "entry_date": "2026-01-02",
            "entry_prices": {"159915.SZ": 3.012},
            "ytd_history": [],
        }
        tmp_state.write_text(json.dumps(existing))

        save_position({"513100.SH": 1.0}, date(2026, 4, 9), _STRAT)

        result = read_position(_STRAT)
        assert result.weights == {"513100.SH": 1.0}
        assert len(result.ytd_history) == 1
        archived = result.ytd_history[0]
        assert archived.weights == {"159915.SZ": 1.0}
        assert archived.entry_date == "2026-01-02"
        assert archived.exit_date == "2026-04-09"
        assert archived.entry_prices == {"159915.SZ": 3.012}
        assert archived.exit_prices is None  # backfilled on next run

    def test_no_archive_when_no_current_weights(self, tmp_state):
        # File missing → no archiving
        save_position({"513100.SH": 1.0}, date(2026, 4, 9), _STRAT)
        result = read_position(_STRAT)
        assert result.ytd_history == []

    def test_preserves_existing_ytd_history(self, tmp_state):
        existing = {
            "weights": {"159915.SZ": 1.0},
            "entry_date": "2026-03-01",
            "entry_prices": {"159915.SZ": 3.0},
            "ytd_history": [
                {
                    "weights": {"518880.SH": 1.0},
                    "entry_date": "2026-01-02",
                    "exit_date": "2026-03-01",
                    "entry_prices": {"518880.SH": 6.0},
                    "exit_prices": {"518880.SH": 6.1},
                }
            ],
        }
        tmp_state.write_text(json.dumps(existing))

        save_position({"513100.SH": 1.0}, date(2026, 4, 9), _STRAT)

        result = read_position(_STRAT)
        assert len(result.ytd_history) == 2
        assert result.ytd_history[0].weights == {"518880.SH": 1.0}
        assert result.ytd_history[1].weights == {"159915.SZ": 1.0}

    def test_archives_ytd_basis_for_cross_year_current_position(self, tmp_state):
        existing = {
            "weights": {"159915.SZ": 1.0},
            "entry_date": "2025-12-30",
            "entry_prices": {"159915.SZ": 3.0},
            "ytd_entry_date": "2026-01-05",
            "ytd_entry_prices": {"159915.SZ": 3.2},
            "ytd_history": [],
        }
        tmp_state.write_text(json.dumps(existing))

        save_position({"513100.SH": 1.0}, date(2026, 1, 9), _STRAT)

        result = read_position(_STRAT)
        archived = result.ytd_history[0]
        assert archived.entry_date == "2026-01-05"
        assert archived.entry_prices == {"159915.SZ": 3.2}

    def test_overwrites_previous(self, tmp_state):
        save_position({"A.SH": 1.0}, date(2026, 1, 2), _STRAT)
        save_position({"B.SH": 1.0}, date(2026, 4, 9), _STRAT)
        result = read_position(_STRAT)
        assert result.weights == {"B.SH": 1.0}


class TestPerStrategyIsolation:
    def test_two_strategies_use_different_files(self, tmp_state, tmp_path):
        save_position({"A.SH": 1.0}, date(2026, 4, 9), "strat_a")
        save_position({"B.SH": 1.0}, date(2026, 4, 9), "strat_b")

        a = read_position("strat_a")
        b = read_position("strat_b")

        assert a.weights == {"A.SH": 1.0}
        assert b.weights == {"B.SH": 1.0}
        assert (tmp_path / "strat_a_position.json").exists()
        assert (tmp_path / "strat_b_position.json").exists()
