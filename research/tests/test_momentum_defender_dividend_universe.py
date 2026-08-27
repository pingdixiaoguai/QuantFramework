import numpy as np
import pandas as pd

from defender.relative_defender_rotation import DEFENSIVE_ASSET
from research.momentum_defender_dividend_universe import (
    SELECTION_SPEC,
    StandaloneDividendUniverseHarness,
    dedupe_pools,
    difference_events,
    run_standalone_universe,
)
from research.momentum_defender_occam_defender import score_at_open


def test_dedupe_pools_preserves_first_ordered_path_label():
    pools = {
        "first": ("A", "B"),
        "duplicate": ("A", "B"),
        "different_order": ("B", "A"),
    }

    assert dedupe_pools(pools) == {
        "first": ("A", "B"),
        "different_order": ("B", "A"),
    }


def test_difference_events_groups_contiguous_executable_differences():
    index = pd.date_range("2026-01-01", periods=7, freq="D")
    baseline = pd.Series(0.0, index=index)
    candidate = pd.Series([0.0, 0.01, 0.02, 0.0, -0.01, -0.02, 0.0], index=index)

    events = difference_events(candidate, baseline)

    assert list(events["observations"]) == [2, 2]
    assert list(events["start"]) == [index[1], index[4]]
    assert list(events["end"]) == [index[2], index[5]]
    assert events.iloc[0]["candidate_return"] == (1.01 * 1.02) - 1.0


def test_standalone_universe_holds_only_listed_etf_during_score_warmup():
    dates = pd.bdate_range("2026-01-05", periods=60)

    def market_frame(selected_dates: pd.DatetimeIndex, start: float) -> pd.DataFrame:
        close = start + np.arange(len(selected_dates), dtype=float) * 0.01
        return pd.DataFrame(
            {
                "date": selected_dates,
                "open": close,
                "high": close,
                "low": close,
                "close": close,
                "volume": 1_000_000.0,
            }
        )

    market = {
        "510880.SH": market_frame(dates, 1.0),
        "512890.SH": market_frame(dates[45:], 1.2),
        DEFENSIVE_ASSET: market_frame(dates, 100.0),
    }
    assets = ("510880.SH", "512890.SH")
    scores = score_at_open(market, assets, dates, SELECTION_SPEC)
    harness = StandaloneDividendUniverseHarness(dates, market, scores)

    result = run_standalone_universe(harness, assets)

    assert result.selection.iloc[0]["selected_asset"] == "510880.SH"
    assert result.targets.iloc[0]["510880.SH"] == 1.0
    assert result.targets.sum(axis=1).eq(1.0).all()
