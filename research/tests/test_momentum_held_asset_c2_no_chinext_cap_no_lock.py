import pandas as pd

from research.run_momentum_held_asset_c2_no_chinext_cap_no_lock import (
    _risk_off_episode_lengths,
)


def test_risk_off_episode_lengths_counts_contiguous_runs() -> None:
    state = pd.DataFrame(
        {"risk_on": [True, False, False, True, False, True]},
        index=pd.date_range("2024-01-01", periods=6),
    )

    assert _risk_off_episode_lengths(state) == [2, 1]
