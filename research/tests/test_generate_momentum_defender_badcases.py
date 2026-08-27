from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from research.defender_curve_momentum import DEFENDER_CANDIDATE
from research.generate_momentum_defender_badcases import (
    _gold_lock_break_counts,
    _render_document,
    build_gold_lock_break_evidence,
    defender_episode_windows,
)
from research.momentum_defender_occam import (
    ENTER_RETURN,
    ENTRY_COST,
    EXIT_COST,
    EXIT_RETURN,
    HELD_RETURN,
    INTERNAL_COST,
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


def test_gold_lock_break_evidence_uses_exact_defender_counterfactual() -> None:
    index = pd.date_range("2026-01-01", periods=4, freq="B")
    state = pd.DataFrame(
        {
            "escape_entry": [False, True, False, False],
            "escape_active": [False, True, True, False],
            "base_w40_held_days_at_open": [4, 5, 6, 7],
            "state_reason": [
                "base_w40_defender",
                "asset_escape_break_defender_lock",
                "asset_escape_hard_hold",
                "base_w40_defender",
            ],
        },
        index=index,
    )
    daily = pd.DataFrame(
        {
            "candidate": [
                DEFENDER_CANDIDATE,
                "518880.SH",
                "518880.SH",
                DEFENDER_CANDIDATE,
            ],
            "return": [0.0, 0.02, 0.01, 0.0],
        },
        index=index,
    )
    interface = pd.DataFrame(
        {
            HELD_RETURN: [0.005] * 4,
            ENTER_RETURN: [0.005] * 4,
            EXIT_RETURN: [0.0] * 4,
            INTERNAL_COST: [0.0] * 4,
            ENTRY_COST: [0.0] * 4,
            EXIT_COST: [0.0] * 4,
        },
        index=index,
    )
    formal = SimpleNamespace(
        escape=SimpleNamespace(state=state, daily=daily),
        context=SimpleNamespace(
            initial_previous_candidate=DEFENDER_CANDIDATE,
            interfaces={DEFENDER_CANDIDATE: interface},
        ),
        audit={"lock_break_entries": 1},
    )

    evidence = build_gold_lock_break_evidence(formal)
    counts = _gold_lock_break_counts(evidence)

    assert len(evidence) == 1
    assert evidence.iloc[0]["gold_return"] == pytest.approx(1.02 * 1.01 - 1.0)
    assert evidence.iloc[0]["defender_return"] == pytest.approx(1.005**2 - 1.0)
    assert counts == {
        "total": 1,
        "wins": 1,
        "win_rate": 1.0,
        "held_total": 1,
        "held_wins": 1,
        "veto_total": 0,
        "veto_wins": 0,
        "open_events": 0,
    }
