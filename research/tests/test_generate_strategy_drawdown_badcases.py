from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from research.formal_strategy_holdings import build_formal_target_schedule
from research.generate_strategy_drawdown_badcases import (
    _candidate_runs,
    _volume_classification,
    distinct_drawdown_episodes,
    fixed_sleeve_returns,
)


def test_distinct_drawdowns_do_not_double_count_nested_troughs() -> None:
    index = pd.date_range("2026-01-01", periods=6, freq="B")
    nav = pd.Series([1.10, 0.99, 1.11, 1.00, 0.90, 1.12], index=index)
    prior = pd.Series([1.0, *nav.iloc[:-1]], index=index)
    daily = pd.DataFrame({"return": nav / prior - 1.0, "nav": nav})

    episodes = distinct_drawdown_episodes(daily, top_n=2)

    assert episodes.attrs["all_episode_count"] == 2
    assert episodes.loc[0, "peak_date"] == index[2]
    assert episodes.loc[0, "trough_date"] == index[4]
    assert episodes.loc[0, "recovery_date"] == index[5]
    assert episodes.loc[0, "max_drawdown"] == pytest.approx(
        0.90 / 1.11 - 1.0
    )
    assert episodes.loc[1, "peak_date"] == index[0]
    assert episodes.loc[1, "trough_date"] == index[1]


def test_formal_target_schedule_uses_w40_candidate_not_old_c2_target() -> None:
    index = pd.date_range("2026-01-01", periods=2, freq="B")
    daily = pd.DataFrame(
        {"candidate": ["159915.SZ", "DEFENDER"]}, index=index
    )
    defender = pd.DataFrame(index=index)
    for code in ("512890", "159545", "513530", "515080", "510880", "563020", "511260"):
        defender[f"target_weight_{code}"] = 0.0
    defender.loc[index[1], "target_weight_511260"] = 0.8
    defender.loc[index[1], "target_weight_512890"] = 0.2
    defender["target_cash_weight"] = 0.0
    formal_run = SimpleNamespace(
        daily=daily,
        context=SimpleNamespace(
            interfaces={"DEFENDER": pd.DataFrame(index=index)},
            integrated=SimpleNamespace(
                result=SimpleNamespace(
                    inputs=SimpleNamespace(defender=defender)
                )
            )
        ),
    )

    targets = build_formal_target_schedule(formal_run)

    assert targets.at[index[0], "159915.SZ"] == 1.0
    assert targets.at[index[0], "511260.SH"] == 0.0
    assert targets.at[index[1], "512890.SH"] == 0.2
    assert targets.at[index[1], "511260.SH"] == 0.8
    assert np.allclose(targets.sum(axis=1), 1.0)


def test_fixed_sleeve_returns_assume_both_sleeves_are_already_held() -> None:
    index = pd.date_range("2026-01-01", periods=3, freq="B")
    momentum = pd.DataFrame(
        {"daily_net_return_if_held": [0.10, -0.05, 0.02]}, index=index
    )
    defender = pd.DataFrame(
        {"daily_net_return_if_held": [0.01, 0.02, -0.01]}, index=index
    )
    formal_run = SimpleNamespace(
        context=SimpleNamespace(
            interfaces={"DEFENDER": defender},
            integrated=SimpleNamespace(
                result=SimpleNamespace(
                    inputs=SimpleNamespace(
                        momentum=momentum,
                        defender=defender,
                    )
                )
            )
        )
    )

    pure_momentum, pure_defender = fixed_sleeve_returns(
        formal_run, index[0], index[-1]
    )

    assert pure_momentum == pytest.approx(1.10 * 0.95 * 1.02 - 1.0)
    assert pure_defender == pytest.approx(1.01 * 1.02 * 0.99 - 1.0)


def test_mixed_candidate_runs_preserve_every_top_level_switch() -> None:
    index = pd.date_range("2026-01-01", periods=5, freq="B")
    daily = pd.DataFrame(
        {
            "candidate": [
                "518880.SH",
                "518880.SH",
                "DEFENDER",
                "DEFENDER",
                "159915.SZ",
            ],
            "return": [0.01, -0.02, 0.005, 0.005, -0.03],
        },
        index=index,
    )

    runs = _candidate_runs(daily, index[0], index[-1], {})

    assert [run["candidate"] for run in runs] == [
        "518880.SH",
        "DEFENDER",
        "159915.SZ",
    ]
    assert runs[0]["strategy_return"] == pytest.approx(1.01 * 0.98 - 1.0)
    assert runs[1]["strategy_return"] == pytest.approx(1.005**2 - 1.0)


def test_volume_classification_distinguishes_climax_and_quiet_peaks() -> None:
    strong = {
        "volume_ratio_to_prior20_median": 2.0,
        "log_volume_z60": 2.5,
        "volume_percentile_prior252": 0.95,
        "prior5_max_volume_ratio": 2.0,
    }
    quiet = {
        "volume_ratio_to_prior20_median": 0.9,
        "log_volume_z60": 0.2,
        "volume_percentile_prior252": 0.5,
        "prior5_max_volume_ratio": 1.2,
    }

    assert _volume_classification(strong) == "峰值日强放量"
    assert _volume_classification(quiet) == "无明显放量（例外）"
