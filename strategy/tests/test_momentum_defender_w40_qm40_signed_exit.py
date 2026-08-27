from __future__ import annotations

import hashlib
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from strategy.momentum_defender_w40_qm40_signed_exit import (
    DEFENDER_ENTRY_PERCENTILE,
    FORMAL_STRATEGY_ID,
    MOMENTUM_RECOVERY_PERCENTILE,
    QM40_RECOVERY_CONFIRMATION_DAYS,
    W40_PERCENTILE_HISTORY,
    qm40_recovery_state_schedule,
    run_formal_strategy,
)


def test_formal_v4_matches_requested_2019_combination() -> None:
    root = Path(__file__).resolve().parents[2]
    formal = run_formal_strategy(
        root,
        start=date(2019, 1, 18),
        end=date(2026, 8, 26),
    )
    research = pd.read_parquet(
        root
        / "experiments/20260826_w40_defender_qm_signed_exit_combination_2019"
        / "factorial_daily_returns.parquet"
    )["requested_all_three"].astype(float)
    actual = formal.daily["return"].astype(float)

    np.testing.assert_allclose(actual, research, atol=1e-14)
    assert FORMAL_STRATEGY_ID == "momentum_defender_w40_qm40_signed_exit_v4"
    assert W40_PERCENTILE_HISTORY == 756
    assert DEFENDER_ENTRY_PERCENTILE == 0.60
    assert MOMENTUM_RECOVERY_PERCENTILE == 0.35
    assert QM40_RECOVERY_CONFIRMATION_DAYS == 10
    assert formal.audit["escape_entries"] == 24
    assert formal.base.audit["qm40_early_recoveries"] == 3
    assert hashlib.sha256(actual.to_numpy(dtype="<f8").tobytes()).hexdigest() == (
        "80e979b8db7c430f1208ac2a3a671e7cf728e88f3f87c3186af25d1d1f1b3d91"
    )


def test_formal_v4_2013_checkpoint_is_frozen() -> None:
    root = Path(__file__).resolve().parents[2]
    formal = run_formal_strategy(root, end=date(2026, 8, 26))
    actual = formal.daily["return"].astype(float)

    assert formal.daily.index.min() == pd.Timestamp("2013-02-04")
    assert formal.audit["escape_entries"] == 34
    assert formal.audit["lock_break_entries"] == 27
    assert formal.audit["immediate_entry_veto_entries"] == 8
    assert formal.audit["candidate_switches"] == 291
    assert formal.base.defender.audit["selection_score"] == "quality_momentum"
    assert formal.base.defender.audit["selection_switches"] == 59
    assert hashlib.sha256(actual.to_numpy(dtype="<f8").tobytes()).hexdigest() == (
        "11bb60cbeeb013235491312dc623a0ed1ebb904120599a72309df8633c10afd0"
    )


def test_positive_qm40_threshold_delays_weak_recovery() -> None:
    index = pd.bdate_range("2024-01-02", periods=20)
    score = pd.Series([0.70] + [0.50] * 19, index=index, dtype=float)
    qm40 = pd.Series([0.0] + [0.005] * 19, index=index, dtype=float)

    zero = qm40_recovery_state_schedule(
        score, qm40, qm40_recovery_threshold=0.0
    )
    higher = qm40_recovery_state_schedule(
        score, qm40, qm40_recovery_threshold=0.0075
    )

    assert zero["state_reason"].eq("qm40_recovery_to_momentum").sum() == 1
    assert not higher["state_reason"].eq("qm40_recovery_to_momentum").any()
