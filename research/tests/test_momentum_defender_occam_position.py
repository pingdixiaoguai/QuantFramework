from __future__ import annotations

import numpy as np
import pandas as pd

from research.momentum_defender_occam_position import (
    PositionSpec,
    build_position_targets,
)


def _market(closes: list[float]) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=len(closes))
    values = np.asarray(closes, dtype=float)
    return pd.DataFrame(
        {
            "date": dates,
            "open": values,
            "high": values,
            "low": values,
            "close": values,
        }
    )


def test_fixed_weight_targets_split_selected_equity_and_bond() -> None:
    market = {"A": _market([1, 1, 1]), "BOND": _market([1, 1, 1])}
    calendar = pd.DatetimeIndex(market["A"]["date"])
    selection = pd.Series("A", index=calendar)
    targets, diagnostics = build_position_targets(
        market,
        ("A",),
        "BOND",
        selection,
        calendar,
        "A",
        PositionSpec("fixed_weight", level=0.25),
    )
    assert np.allclose(targets["A"], 0.25)
    assert np.allclose(targets["BOND"], 0.75)
    assert np.allclose(diagnostics["equity_weight"], 0.25)


def test_trend_binary_uses_strictly_previous_close() -> None:
    market = {
        "A": _market([100, 101, 102, 103, 50]),
        "BOND": _market([1, 1, 1, 1, 1]),
    }
    calendar = pd.DatetimeIndex(market["A"]["date"])
    selection = pd.Series("A", index=calendar)
    targets, _ = build_position_targets(
        market,
        ("A",),
        "BOND",
        selection,
        calendar,
        "A",
        PositionSpec("trend_binary", "selected", 2),
    )
    assert targets.iloc[-1]["A"] == 1.0
    assert targets.iloc[-1]["BOND"] == 0.0


def test_range_location_is_direct_contrarian_exposure() -> None:
    market = {
        "A": _market([100, 110, 105, 100]),
        "BOND": _market([1, 1, 1, 1]),
    }
    calendar = pd.DatetimeIndex(market["A"]["date"])
    selection = pd.Series("A", index=calendar)
    targets, _ = build_position_targets(
        market,
        ("A",),
        "BOND",
        selection,
        calendar,
        "A",
        PositionSpec("range_location", "selected", 3),
    )
    assert targets.iloc[-1]["A"] == 0.5
    assert targets.iloc[-1]["BOND"] == 0.5


def test_range_high_cut_only_reduces_at_the_threshold() -> None:
    market = {
        "A": _market([100, 110, 109, 110]),
        "BOND": _market([1, 1, 1, 1]),
    }
    calendar = pd.DatetimeIndex(market["A"]["date"])
    selection = pd.Series("A", index=calendar)
    targets, _ = build_position_targets(
        market,
        ("A",),
        "BOND",
        selection,
        calendar,
        "A",
        PositionSpec("range_high_cut", "anchor", 3, 0.80, 0.50),
    )
    assert targets.iloc[-1]["A"] == 0.5
    assert targets.iloc[-1]["BOND"] == 0.5
