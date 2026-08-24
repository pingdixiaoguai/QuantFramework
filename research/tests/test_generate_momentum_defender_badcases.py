from __future__ import annotations

import pandas as pd
import pytest

from research.defender_curve_momentum import DEFENDER_CANDIDATE
from research.generate_momentum_defender_badcases import (
    _render_document,
    defender_episode_windows,
)


def test_defender_episode_windows_include_exit_open_and_mark_open_case() -> None:
    index = pd.date_range("2026-01-01", periods=7, freq="B")
    candidate = pd.Series(
        [
            "510300.SH",
            DEFENDER_CANDIDATE,
            DEFENDER_CANDIDATE,
            "518880.SH",
            DEFENDER_CANDIDATE,
            DEFENDER_CANDIDATE,
            DEFENDER_CANDIDATE,
        ],
        index=index,
    )

    windows = defender_episode_windows(candidate)

    assert windows[0] == (index[1], index[2], index[3], False)
    assert windows[1] == (index[4], index[6], index[6], True)


def test_context_coverage_must_match_detected_badcases_exactly() -> None:
    badcases = pd.DataFrame(
        {
            "case_start": ["2026-01-02"],
            "base_entry_reason": ["slow_regime_switch"],
            "immediate_reason": ["hold"],
            "dominant_momentum_asset": ["510300.SH"],
        }
    )
    badcases.attrs["details"] = {"2026-01-02": {}}
    stale_config = {
        "strategy_id": "example",
        "evidence_cutoff": "2026-01-07",
        "asset_names": {},
        "cases": {"2026-01-03": {}},
    }

    with pytest.raises(AssertionError, match="coverage mismatch"):
        _render_document(badcases, stale_config)
