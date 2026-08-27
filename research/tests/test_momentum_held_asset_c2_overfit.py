from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.momentum_defender_occam import performance
from research.run_momentum_held_asset_c2_overfit import (
    _choose_candidate,
    _circular_block_indices,
    _cscv_pbo,
    _matrix_metrics,
    _unique_paths,
    _white_reality_check,
)


def test_matrix_metrics_match_reference_performance() -> None:
    dates = pd.date_range("2026-01-05", periods=5, freq="B")
    values = np.array([0.01, -0.02, 0.03, 0.00, 0.01])
    measured = _matrix_metrics(values)
    expected = performance(pd.Series(values, index=dates))
    for key in (
        "total_return",
        "annualized_return_252",
        "sharpe",
        "max_drawdown",
    ):
        assert measured[key][0] == pytest.approx(expected[key])


def test_selection_excludes_inactive_and_non_improving_candidates() -> None:
    metrics = {
        "annualized_return_252": np.array([0.30, 0.25, 0.22]),
        "sharpe": np.array([2.0, 1.8, 1.6]),
        "max_drawdown": np.array([-0.20, -0.10, -0.12]),
    }
    baseline = {
        "annualized_return_252": np.array([0.20]),
        "sharpe": np.array([1.5]),
        "max_drawdown": np.array([-0.15]),
    }
    chosen, pool, count = _choose_candidate(
        metrics,
        baseline,
        np.array([2, 0, 1]),
        np.array([5, 3, 4]),
        ["bad_mdd", "inactive", "balanced"],
    )
    assert chosen == 2
    assert pool == "beats_no_cap_and_active"
    assert count == 1


def test_unique_paths_keep_first_identical_column() -> None:
    values = np.array(
        [
            [0.01, 0.01, 0.02],
            [0.02, 0.02, 0.03],
        ]
    )
    assert _unique_paths(values).tolist() == [0, 2]


def test_circular_block_indices_preserve_local_order_and_bounds() -> None:
    sample = _circular_block_indices(
        observations=17,
        block_length=5,
        rng=np.random.default_rng(7),
    )
    assert len(sample) == 17
    assert sample.min() >= 0
    assert sample.max() < 17
    for position in range(1, len(sample)):
        if position % 5:
            assert sample[position] == (sample[position - 1] + 1) % 17


def test_cscv_reports_every_balanced_split_and_valid_probability() -> None:
    values = np.array(
        [
            [0.01, 0.00, -0.01],
            [0.01, 0.00, -0.01],
            [-0.01, 0.00, 0.01],
            [-0.01, 0.00, 0.01],
            [0.02, 0.00, -0.02],
            [0.02, 0.00, -0.02],
            [-0.02, 0.00, 0.02],
            [-0.02, 0.00, 0.02],
        ]
    )
    details, summary = _cscv_pbo(values, ["a", "b", "c"], blocks=4)
    assert len(details) == 6
    assert summary["splits"] == 6
    assert 0.0 <= summary["pbo"] <= 1.0
    assert details["test_rank_percentile"].between(0.0, 1.0).all()


def test_reality_check_distinguishes_null_from_uniform_positive_edge() -> None:
    rng = np.random.default_rng(11)
    null = _white_reality_check(
        np.zeros((40, 3)), block_length=5, repetitions=99, rng=rng
    )
    positive = _white_reality_check(
        np.column_stack([np.full(40, 0.01), np.zeros(40)]),
        block_length=5,
        repetitions=99,
        rng=np.random.default_rng(11),
    )
    assert null["bootstrap_p_value"] == 1.0
    assert positive["bootstrap_p_value"] == pytest.approx(0.01)
