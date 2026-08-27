from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np

from defender.relative_defender_rotation import DEFENSIVE_ASSET
from defender.w40_qm_reversal_full_equity import (
    FORMAL_DEFENDER_STRATEGY_ID,
)
from strategy.momentum_defender_w40_qm40_signed_exit import run_formal_strategy


def test_formal_qm40_defender_is_full_equity_and_quality_ranked() -> None:
    root = Path(__file__).resolve().parents[2]
    formal = run_formal_strategy(root, end=date(2026, 8, 26))
    defender = formal.base.defender

    assert defender.audit["strategy_id"] == FORMAL_DEFENDER_STRATEGY_ID
    assert defender.audit["selection_score"] == "quality_momentum"
    assert defender.audit["selection_window"] == 40
    assert np.allclose(defender.targets.sum(axis=1), 1.0)
    assert np.allclose(defender.targets[DEFENSIVE_ASSET], 0.0)
    equity = defender.targets.drop(columns=[DEFENSIVE_ASSET])
    assert np.allclose(equity.gt(0.0).sum(axis=1), 1)
    assert np.allclose(equity.max(axis=1), 1.0)
