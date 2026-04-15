"""Current position persistence — state/current_position.json."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

STATE_FILE = Path(__file__).resolve().parent.parent / "state" / "current_position.json"


@dataclass
class PositionPeriod:
    """A closed position interval recorded in ytd_history."""

    weights: dict[str, float]
    entry_date: str
    exit_date: str
    entry_prices: dict[str, float] | None
    exit_prices: dict[str, float] | None  # null until backfilled on next run


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
        entry_prices=d.get("entry_prices"),
        exit_prices=d.get("exit_prices"),
    )


def _state_to_dict(state: PositionState) -> dict:
    return {
        "weights": state.weights,
        "entry_date": state.entry_date,
        "entry_prices": state.entry_prices,
        "ytd_history": [
            {
                "weights": p.weights,
                "entry_date": p.entry_date,
                "exit_date": p.exit_date,
                "entry_prices": p.entry_prices,
                "exit_prices": p.exit_prices,
            }
            for p in state.ytd_history
        ],
    }


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


def write_position(state: PositionState) -> None:
    """Persist a PositionState directly to disk (used for backfill updates)."""
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(_state_to_dict(state), f, indent=2)


def save_position(
    target_weights: dict[str, float],
    entry_date: date,
    entry_prices: dict[str, float] | None = None,
) -> None:
    """Persist new target weights, archiving the outgoing position to ytd_history.

    The archived period has exit_prices=None; these are backfilled on the next run.
    """
    current = read_position()

    new_history = list(current.ytd_history)
    if current.weights:
        new_history.append(
            PositionPeriod(
                weights=current.weights,
                entry_date=current.entry_date or "",
                exit_date=entry_date.isoformat(),
                entry_prices=current.entry_prices,
                exit_prices=None,
            )
        )

    new_state = PositionState(
        weights=target_weights,
        entry_date=entry_date.isoformat(),
        entry_prices=entry_prices,
        ytd_history=new_history,
    )
    write_position(new_state)
