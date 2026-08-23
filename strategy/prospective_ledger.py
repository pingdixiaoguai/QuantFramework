"""Idempotent append-only prospective signal ledger."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping


def append_signal_record(path: Path, record: Mapping[str, object]) -> bool:
    """Append one unique signal/execution record; return False for exact replay."""
    required = {"strategy_id", "signal_date", "execution_date"}
    missing = required - set(record)
    if missing:
        raise ValueError(f"prospective record missing: {sorted(missing)}")
    identity = tuple(str(record[field]) for field in sorted(required))
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            existing = json.loads(line)
            existing_identity = tuple(
                str(existing.get(field)) for field in sorted(required)
            )
            if existing_identity == identity:
                if dict(existing) != dict(record):
                    raise RuntimeError(
                        "prospective signal identity already exists with different data"
                    )
                return False
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(record), ensure_ascii=False, sort_keys=True) + "\n")
    return True
