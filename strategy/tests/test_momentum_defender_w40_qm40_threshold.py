from __future__ import annotations

import hashlib
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from strategy.momentum_defender_w40_qm40_threshold import (
    FORMAL_STRATEGY_ID,
    QM40_RECOVERY_THRESHOLD,
    run_formal_strategy,
)


def test_formal_v5_matches_selected_threshold_path() -> None:
    root = Path(__file__).resolve().parents[2]
    formal = run_formal_strategy(
        root,
        start=date(2019, 1, 18),
        end=date(2026, 8, 26),
    )
    selected = pd.read_parquet(
        root
        / "experiments/20260826_qm40_recovery_threshold_search_2019"
        / "daily_returns.parquet"
    )["qm40_threshold_+0.00750"].astype(float)
    actual = formal.daily["return"].astype(float)

    np.testing.assert_allclose(actual, selected, atol=1e-14)
    assert FORMAL_STRATEGY_ID == "momentum_defender_w40_qm40_threshold_v5"
    assert QM40_RECOVERY_THRESHOLD == 0.0075
    assert formal.base.audit["qm40_early_recoveries"] == 1
    assert hashlib.sha256(actual.to_numpy(dtype="<f8").tobytes()).hexdigest() == (
        "0c6039ee3e80daabe7fbadff4163d51fc772fa96bbd242bb13b22e029634d1fb"
    )


def test_formal_v5_2013_checkpoint_is_frozen() -> None:
    root = Path(__file__).resolve().parents[2]
    formal = run_formal_strategy(root, end=date(2026, 8, 26))
    actual = formal.daily["return"].astype(float)

    assert formal.audit["escape_entries"] == 35
    assert formal.audit["lock_break_entries"] == 28
    assert formal.audit["escape_days"] == 438
    assert formal.audit["candidate_switches"] == 289
    assert formal.base.audit["qm40_early_recoveries"] == 3
    assert hashlib.sha256(actual.to_numpy(dtype="<f8").tobytes()).hexdigest() == (
        "6a45479ffe5da9b081e53e36c7a0b137656ed9a61cdf3c7d8044aa278700f4d3"
    )
