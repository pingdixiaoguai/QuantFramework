from __future__ import annotations

import hashlib
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from strategy.momentum_defender_w40_gold_escape import (
    FORMAL_STRATEGY_ID,
    GOLD_ENTRY_X,
    GOLD_EXIT_Y,
    run_formal_strategy,
)


def test_formal_gold_escape_matches_research_checkpoint() -> None:
    root = Path(__file__).resolve().parents[2]
    result = run_formal_strategy(root, end=date(2026, 8, 26))
    selected = pd.read_parquet(
        root
        / "experiments/20260826_momentum_defender_immediate_gold_entry_veto/candidate_daily.parquet"
    )["candidate_return"].astype(float)
    actual = result.daily["return"].astype(float)
    np.testing.assert_allclose(actual, selected, atol=1e-14)
    assert FORMAL_STRATEGY_ID == "momentum_defender_w40_gold_qm20_escape_v3"
    assert GOLD_ENTRY_X == 0.005
    assert GOLD_EXIT_Y == -0.020
    assert result.daily.index.min() == pd.Timestamp("2013-02-04")
    assert result.audit["escape_entries"] == 33
    assert result.audit["lock_break_entries"] == 27
    assert result.audit["immediate_entry_veto_entries"] == 7
    assert hashlib.sha256(actual.to_numpy(dtype="<f8").tobytes()).hexdigest() == (
        "2e746404983f979dd638c982e3d0e9cfdc571c038f8333f04ce8de2f9016af88"
    )
