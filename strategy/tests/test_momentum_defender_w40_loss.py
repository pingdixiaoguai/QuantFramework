from __future__ import annotations

import hashlib
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from strategy.momentum_defender_w40_loss import (
    DEFENDER_ENTRY_CONFIRMATION_DAYS,
    DEFENDER_ENTRY_PERCENTILE,
    DEFENDER_LOCK_DAYS,
    FORMAL_STRATEGY_ID,
    MOMENTUM_LOCK_DAYS,
    MOMENTUM_RECOVERY_PERCENTILE,
    run_formal_strategy,
)


def test_formal_constants_match_excluding_extremes_selection() -> None:
    assert FORMAL_STRATEGY_ID == "momentum_defender_w40_loss_excluding_extremes_v1"
    assert DEFENDER_ENTRY_PERCENTILE == 0.55
    assert MOMENTUM_RECOVERY_PERCENTILE == 0.40
    assert DEFENDER_ENTRY_CONFIRMATION_DAYS == 1
    assert MOMENTUM_LOCK_DAYS == DEFENDER_LOCK_DAYS == 30


def test_formal_replay_matches_selected_research_checkpoint() -> None:
    root = Path(__file__).resolve().parents[2]
    result = run_formal_strategy(root, end=date(2026, 8, 21))
    selected = pd.read_parquet(
        root
        / "experiments/20260825_momentum_defender_w40_loss_occam_search/selected_excluding_extremes_daily.parquet"
    )
    actual = result.daily["return"].astype(float)
    np.testing.assert_allclose(
        actual.to_numpy(float), selected["return"].to_numpy(float), atol=1e-14
    )
    assert result.audit["defender_entries"] == 20
    assert result.audit["candidate_switches"] == 125
    assert hashlib.sha256(actual.to_numpy(dtype="<f8").tobytes()).hexdigest() == (
        "1ec2fcef5d4646e62af8a98a5377184e92845620b47193d7737423e2c9a9fc39"
    )
