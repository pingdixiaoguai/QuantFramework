from __future__ import annotations

import hashlib
from datetime import date
from pathlib import Path

import pytest
import pandas as pd
import yaml

from strategy.momentum_defender_w40_full_equity import (
    FORMAL_STRATEGY_ID,
    run_formal_strategy,
)


def test_formal_replay_matches_promoted_research_checkpoint() -> None:
    root = Path(__file__).resolve().parents[2]
    config = yaml.safe_load(
        (root / "strategy/configs/momentum_defender_w40_full_equity.yaml").read_text(
            encoding="utf-8"
        )
    )
    result = run_formal_strategy(root, end=date(2026, 8, 26))
    actual = result.daily["return"].astype(float)
    assert FORMAL_STRATEGY_ID == "momentum_defender_w40_reversal_full_equity_v2"
    assert actual.index.min() == pd.Timestamp("2013-02-04")
    assert result.audit["defender_entries"] == 31
    assert result.audit["performance"]["annualized_return_252"] == pytest.approx(
        config["checkpoint"]["annualized_return_252"], abs=1e-12
    )
    assert hashlib.sha256(actual.to_numpy(dtype="<f8").tobytes()).hexdigest() == (
        "d977b9707ba7ee004390e255220a4e9aea3b392bb0a52f2ab71227690599bda8"
    )
