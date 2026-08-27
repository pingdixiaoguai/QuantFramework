from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np

from defender.relative_defender_rotation import DEFENSIVE_ASSET
from defender.w40_reversal_full_equity import (
    FORMAL_DIVIDEND_ASSETS,
    FORMAL_DEFENDER_STRATEGY_ID,
)
from strategy.momentum_defender_w40_full_equity import run_formal_strategy


def test_formal_defender_is_always_full_equity_and_zero_bond() -> None:
    root = Path(__file__).resolve().parents[2]
    formal = run_formal_strategy(root, end=date(2026, 8, 26))
    result = formal.defender
    assert result.audit["strategy_id"] == FORMAL_DEFENDER_STRATEGY_ID
    assert tuple(result.audit["candidate_assets"]) == FORMAL_DIVIDEND_ASSETS
    assert np.allclose(result.targets.sum(axis=1), 1.0)
    assert np.allclose(result.targets[DEFENSIVE_ASSET], 0.0)
    equity = result.targets.drop(columns=[DEFENSIVE_ASSET])
    assert np.allclose((equity.gt(0.0)).sum(axis=1), 1)
    assert np.allclose(equity.max(axis=1), 1.0)
