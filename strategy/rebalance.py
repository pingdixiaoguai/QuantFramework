"""Rebalance timing policies shared by live run, backfill, and backtest."""

from __future__ import annotations


VALID_REBALANCE_MODES = {"min_hold", "fixed_cycle"}


def normalize_rebalance_mode(rebalance_mode: str | None) -> str:
    mode = (rebalance_mode or "min_hold").strip().lower()
    if mode not in VALID_REBALANCE_MODES:
        allowed = ", ".join(sorted(VALID_REBALANCE_MODES))
        raise ValueError(f"rebalance_mode must be one of: {allowed}; got {rebalance_mode!r}")
    return mode


def should_hold_position(
    current_weights: dict[str, float],
    holding_days: int | None,
    rebalance_days: int,
    rebalance_mode: str | None = "min_hold",
) -> bool:
    """Return True when today's signal should be suppressed."""
    if rebalance_days < 1:
        raise ValueError(f"rebalance_days must be >= 1, got {rebalance_days}")

    mode = normalize_rebalance_mode(rebalance_mode)
    if rebalance_days <= 1:
        return False
    if not current_weights:
        return False
    if holding_days is None:
        return True
    if mode == "min_hold":
        return holding_days < rebalance_days
    return holding_days % rebalance_days != 0
