"""Current position persistence — state/current_position.json."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

STATE_FILE = Path(__file__).resolve().parent.parent / "state" / "current_position.json"


@dataclass
class PositionPeriod:
    """A closed position interval recorded in ytd_history."""

    weights: dict[str, float]
    entry_date: str
    exit_date: str
    entry_prices: dict[str, float]
    exit_prices: dict[str, float]


@dataclass
class PositionState:
    """Full state of the current position file."""

    weights: dict[str, float] = field(default_factory=dict)
    entry_date: str | None = None
    entry_prices: dict[str, float] | None = None
    ytd_history: list[PositionPeriod] = field(default_factory=list)


def _parse_period(d: dict) -> PositionPeriod:
    return PositionPeriod(
        weights=d["weights"],
        entry_date=d["entry_date"],
        exit_date=d["exit_date"],
        entry_prices=d["entry_prices"],
        exit_prices=d["exit_prices"],
    )


def read_position() -> PositionState:
    """Load position state from disk. Auto-migrates old flat-dict format.

    Returns empty PositionState if the file is missing.
    """
    if not STATE_FILE.exists():
        return PositionState()

    with open(STATE_FILE) as f:
        data = json.load(f)

    # Old format: flat dict of {asset: weight} with no "weights" key
    if "weights" not in data:
        return PositionState(
            weights=data,
            entry_date=None,
            entry_prices=None,
            ytd_history=[],
        )

    return PositionState(
        weights=data["weights"],
        entry_date=data.get("entry_date"),
        entry_prices=data.get("entry_prices"),
        ytd_history=[_parse_period(p) for p in data.get("ytd_history", [])],
    )


def save_position(weights: dict[str, float]) -> None:
    """Persist target weights to disk (legacy signature, T2 will extend this)."""
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(weights, f, indent=2)
