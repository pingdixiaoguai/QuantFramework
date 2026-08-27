from __future__ import annotations

import numpy as np
import pandas as pd

from research.momentum_defender_occam import (
    ENTER_RETURN,
    EXIT_RETURN,
    HELD_RETURN,
)
from research.momentum_defender_occam_defender import (
    MonthlySelectionSpec,
    build_portfolio_switch_interface,
    monthly_top1_selection,
    score_at_open,
    selected_asset_targets,
)


def _prices(start: str, closes: list[float]) -> pd.DataFrame:
    dates = pd.bdate_range(start, periods=len(closes))
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


def test_score_at_open_is_strictly_prior_close() -> None:
    market = {
        "A": _prices("2024-01-01", [100, 101, 102, 103, 104]),
        "B": _prices("2024-01-01", [100, 100, 100, 100, 200]),
    }
    calendar = pd.DatetimeIndex(market["A"]["date"])
    scores = score_at_open(
        market,
        ("A", "B"),
        calendar,
        MonthlySelectionSpec(window=2, direction="highest"),
    )
    last = calendar[-1]
    assert np.isclose(scores.at[last, "A"], np.log(103.0 / 101.0))
    assert np.isclose(scores.at[last, "B"], 0.0)


def test_monthly_selection_uses_previous_close_and_stable_ties() -> None:
    dates = pd.to_datetime(["2024-01-31", "2024-02-01", "2024-02-02"])
    market = {
        asset: pd.DataFrame(
            {
                "date": dates,
                "open": [1.0, 1.0, 1.0],
                "high": [1.0, 1.0, 1.0],
                "low": [1.0, 1.0, 1.0],
                "close": [1.0, 1.0, 1.0],
            }
        )
        for asset in ("A", "B")
    }
    scores = pd.DataFrame(
        {"A": [1.0, 2.0, -100.0], "B": [1.0, 1.0, 100.0]},
        index=dates,
    )
    selected = monthly_top1_selection(
        market,
        ("A", "B"),
        pd.DatetimeIndex(dates),
        scores,
        MonthlySelectionSpec(window=2, direction="highest"),
    )
    assert selected["selected_asset"].tolist() == ["A", "A", "A"]


def test_portfolio_interface_matches_single_asset_legs() -> None:
    market = {"A": _prices("2024-01-01", [100.0, 110.0, 99.0])}
    calendar = pd.DatetimeIndex(market["A"]["date"])
    targets = pd.DataFrame({"A": 1.0}, index=calendar)
    interface = build_portfolio_switch_interface(market, targets, {"A": 0.0})
    assert np.allclose(interface[HELD_RETURN], [0.0, 0.10, -0.10])
    assert np.allclose(interface[ENTER_RETURN], [0.0, 0.0, 0.0])
    assert np.isnan(interface.iloc[0][EXIT_RETURN])
    assert np.allclose(interface[EXIT_RETURN].iloc[1:], [0.10, -0.10])
    assert np.allclose(interface["nav_if_held"], [1.0, 1.1, 0.99])


def test_selected_asset_targets_assign_fixed_residual() -> None:
    index = pd.date_range("2024-01-01", periods=2)
    selection = pd.Series(["A", "B"], index=index)
    targets = selected_asset_targets(
        selection,
        ("A", "B"),
        selected_weight=0.75,
        residual_asset="BOND",
    )
    assert targets.loc[index[0]].to_dict() == {"A": 0.75, "B": 0.0, "BOND": 0.25}
    assert targets.loc[index[1]].to_dict() == {"A": 0.0, "B": 0.75, "BOND": 0.25}
