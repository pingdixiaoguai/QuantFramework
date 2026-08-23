from __future__ import annotations

import json

import pytest

from strategy.prospective_ledger import append_signal_record


def _record():
    return {
        "strategy_id": "formal",
        "signal_date": "2026-08-21",
        "execution_date": "2026-08-24",
        "target_weights": {"518880.SH": 1.0},
    }


def test_append_is_idempotent_for_exact_signal(tmp_path) -> None:
    path = tmp_path / "ledger.jsonl"

    assert append_signal_record(path, _record())
    assert not append_signal_record(path, _record())
    assert len(path.read_text(encoding="utf-8").splitlines()) == 1


def test_same_identity_with_changed_target_fails_closed(tmp_path) -> None:
    path = tmp_path / "ledger.jsonl"
    append_signal_record(path, _record())
    changed = _record()
    changed["target_weights"] = {"511260.SH": 1.0}

    with pytest.raises(RuntimeError, match="different data"):
        append_signal_record(path, changed)

    stored = json.loads(path.read_text(encoding="utf-8"))
    assert stored["target_weights"] == {"518880.SH": 1.0}
