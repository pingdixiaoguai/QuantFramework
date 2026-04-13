"""Current position persistence — state/current_position.json."""

from __future__ import annotations

import json
from pathlib import Path

STATE_FILE = Path(__file__).resolve().parent.parent / "state" / "current_position.json"


def read_position() -> dict[str, float]:
    """Load current position weights from disk. Returns empty dict if missing."""
    if not STATE_FILE.exists():
        return {}
    with open(STATE_FILE) as f:
        return json.load(f)


def save_position(weights: dict[str, float]) -> None:
    """Persist target weights to disk."""
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(weights, f, indent=2)
