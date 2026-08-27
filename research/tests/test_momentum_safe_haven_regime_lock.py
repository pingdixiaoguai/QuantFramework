from __future__ import annotations

import pandas as pd

from research.momentum_safe_haven_regime_lock import all_sleeve_locked_choice


def test_all_sleeve_lock_blocks_changes_for_thirty_days() -> None:
    dates = pd.date_range("2026-01-01", periods=65, freq="D")
    raw_risk_on = pd.Series(False, index=dates)
    raw_risk_on.iloc[1:] = True
    safe_choice = pd.Series("defender", index=dates)

    actual = all_sleeve_locked_choice(raw_risk_on, safe_choice, min_days=30)

    assert actual.iloc[:30].eq("defender").all()
    assert actual.iloc[30:].eq("momentum").all()


def test_safe_haven_changes_are_also_locked_in_alternative_mode() -> None:
    dates = pd.date_range("2026-01-01", periods=40, freq="D")
    raw_risk_on = pd.Series(False, index=dates)
    safe_choice = pd.Series("defender", index=dates)
    safe_choice.iloc[1:] = "gold"

    actual = all_sleeve_locked_choice(raw_risk_on, safe_choice, min_days=30)

    assert actual.iloc[:30].eq("defender").all()
    assert actual.iloc[30:].eq("gold").all()
