"""Formal v5 wrapper with the promoted absolute QM40 recovery threshold."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from strategy.momentum_defender_w40_qm40_signed_exit import (
    FormalW40QM40Backtest,
    FormalW40QM40NextOpenSignal,
    build_next_open_signal as build_v4_next_open_signal,
    run_formal_strategy as run_v4_formal_strategy,
)


FORMAL_STRATEGY_ID = "momentum_defender_w40_qm40_threshold_v5"
BASE_STRATEGY_ID = "momentum_defender_w40_qm40_base_v5"
QM40_RECOVERY_THRESHOLD = 0.0075


def run_formal_strategy(
    root: Path,
    *,
    end: date,
    start: date = date(2013, 1, 1),
) -> FormalW40QM40Backtest:
    return run_v4_formal_strategy(
        root,
        end=end,
        start=start,
        qm40_recovery_threshold=QM40_RECOVERY_THRESHOLD,
        strategy_id=FORMAL_STRATEGY_ID,
        base_strategy_id=BASE_STRATEGY_ID,
    )


def build_next_open_signal(
    root: Path,
    signal_date: date,
    execution_date: date,
) -> FormalW40QM40NextOpenSignal:
    return build_v4_next_open_signal(
        root,
        signal_date,
        execution_date,
        qm40_recovery_threshold=QM40_RECOVERY_THRESHOLD,
        strategy_id=FORMAL_STRATEGY_ID,
        base_strategy_id=BASE_STRATEGY_ID,
    )
