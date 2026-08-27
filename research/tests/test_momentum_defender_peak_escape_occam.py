from __future__ import annotations

import pandas as pd
import pytest

from research.momentum_defender_peak_escape_occam import (
    PeakEscapeFeatures,
    PeakEscapeParams,
    peak_escape_schedule,
    top_drawdown_summary,
)
from research.run_momentum_defender_peak_escape_occam import (
    _scale_interface_net_costs,
    _select_candidate,
)


def _features(index: pd.DatetimeIndex) -> PeakEscapeFeatures:
    columns = ["510300.SH", "159915.SZ", "513100.SH", "518880.SH"]
    breakout = pd.DataFrame(-0.01, index=index, columns=columns)
    return20 = pd.DataFrame(0.10, index=index, columns=columns)
    volume = pd.DataFrame(1.0, index=index, columns=columns)
    share = pd.DataFrame(0.0, index=index, columns=columns)
    breakout.loc[index[0], "159915.SZ"] = 0.02
    return20.loc[index[0], "159915.SZ"] = 0.25
    volume.loc[index[0], "159915.SZ"] = 2.0
    return PeakEscapeFeatures(
        calendar=index,
        price_breakout_at_open=breakout,
        price_return20_at_open=return20,
        volume_ratio20_at_open=volume,
        adjusted_share_flow20_at_open=share,
        coverage=pd.DataFrame(),
    )


def test_peak_escape_holds_defender_for_minimum_sessions() -> None:
    index = pd.date_range("2026-01-01", periods=7, freq="B")
    baseline = pd.Series("159915.SZ", index=index)
    params = PeakEscapeParams("price_volume", 0.20, 1.50, 0.05, 5)

    state = peak_escape_schedule(baseline, _features(index), params)

    assert state.loc[index[:5], "target_candidate"].eq("DEFENDER").all()
    assert state.at[index[5], "target_candidate"] == "159915.SZ"
    assert state.at[index[0], "state_reason"] == "peak_escape_entry"
    assert state.at[index[5], "state_reason"] == "peak_escape_exit"


def test_price_crowding_allows_scale_evidence_without_volume() -> None:
    index = pd.date_range("2026-01-01", periods=2, freq="B")
    features = _features(index)
    features.volume_ratio20_at_open.loc[index[0], "159915.SZ"] = 1.0
    features.adjusted_share_flow20_at_open.loc[index[0], "159915.SZ"] = 0.08
    baseline = pd.Series("159915.SZ", index=index)

    pv = peak_escape_schedule(
        baseline,
        features,
        PeakEscapeParams("price_volume", 0.20, 1.50, 0.05, 1),
    )
    crowding = peak_escape_schedule(
        baseline,
        features,
        PeakEscapeParams("price_crowding", 0.20, 1.50, 0.05, 1),
    )

    assert not bool(pv.at[index[0], "peak_escape_active"])
    assert bool(crowding.at[index[0], "peak_escape_active"])
    assert bool(crowding.at[index[0], "scale_flag"])


def test_top_drawdown_summary_uses_distinct_underwater_episodes() -> None:
    index = pd.date_range("2026-01-01", periods=6, freq="B")
    nav = pd.Series([1.10, 0.99, 1.11, 1.00, 0.90, 1.12], index=index)
    prior = pd.Series([1.0, *nav.iloc[:-1]], index=index)
    returns = nav / prior - 1.0

    summary, episodes = top_drawdown_summary(returns, top_n=20)

    assert summary["top_drawdown_count"] == 2
    assert summary["top_mean_drawdown"] == pytest.approx(
        ((0.99 / 1.10 - 1.0) + (0.90 / 1.11 - 1.0)) / 2.0
    )
    assert list(episodes["trough_date"]) == [index[4], index[1]]


def test_cost_stress_reconstructs_net_legs_without_gross_columns() -> None:
    frame = pd.DataFrame(
        {
            "daily_net_return_if_held": [0.01],
            "enter_open_to_close_net_return": [0.019796],
            "exit_prev_close_to_open_net_return": [-0.010099],
            "internal_cost_rate_at_open": [0.0],
            "fresh_entry_cost_rate_at_open": [0.0002],
            "fresh_exit_cost_rate_at_open": [0.0001],
        }
    )

    identity = _scale_interface_net_costs(frame, 1.0)
    stressed = _scale_interface_net_costs(frame, 3.0)

    assert identity["enter_open_to_close_net_return"].iloc[0] == pytest.approx(
        frame["enter_open_to_close_net_return"].iloc[0]
    )
    assert stressed["enter_open_to_close_net_return"].iloc[0] < identity[
        "enter_open_to_close_net_return"
    ].iloc[0]
    assert stressed["fresh_entry_cost_rate_at_open"].iloc[0] == pytest.approx(
        0.0006
    )


def test_preregistered_split_failure_returns_diagnostic_status() -> None:
    index = pd.DatetimeIndex(
        [
            "2019-01-02",
            "2019-01-03",
            "2020-01-02",
            "2020-01-03",
        ]
    )
    candidate_id = "example"
    metrics = pd.DataFrame(
        {
            "escape_entries": [5],
            "development_delta_annualized_return_252": [0.10],
            "validation_delta_annualized_return_252": [0.10],
            "development_delta_sharpe": [0.10],
            "validation_delta_sharpe": [0.10],
            "development_delta_top20_mean_drawdown": [-0.001],
            "validation_delta_top20_mean_drawdown": [0.01],
            "policy": ["price_volume"],
            "escape_days": [10],
        },
        index=[candidate_id],
    )
    returns = pd.DataFrame({candidate_id: [0.01, -0.01, 0.01, -0.01]}, index=index)
    baseline = pd.Series([0.0, 0.0, 0.0, 0.0], index=index)
    config = {
        "objective": {"top_n": 20},
        "periods": {
            "development": ["2019-01-01", "2019-12-31"],
            "validation": ["2020-01-01", "2020-12-31"],
        },
        "selection": {
            "minimum_escape_entries": 3,
            "annualized_return_delta_floor": -0.03,
            "sharpe_delta_floor": -0.05,
            "top20_mean_drawdown_delta_floor": 0.0,
        },
    }

    diagnostic, eligible, status = _select_candidate(
        metrics, returns, baseline, config
    )

    assert status == "no_eligible_candidate_diagnostic_leader_only"
    assert eligible.empty
    assert diagnostic.name == candidate_id
