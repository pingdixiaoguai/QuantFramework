from __future__ import annotations

import numpy as np
import pandas as pd

from research.momentum_defender_gold_override_overfit import (
    cscv_pbo,
    expanding_walk_forward,
    full_metrics,
    leave_one_year_selection,
    paired_block_bootstrap,
    yearly_reality_check,
)


def _returns() -> tuple[pd.DataFrame, pd.Series]:
    index = pd.bdate_range("2020-01-01", "2023-12-31")
    x = np.arange(len(index), dtype=float)
    baseline = pd.Series(0.0003 + 0.001 * np.sin(x / 9.0), index=index)
    candidates = pd.DataFrame(
        {
            "stable": baseline + 0.0001,
            "cyclical": baseline + 0.0003 * np.sin(x / 23.0),
            "weak": baseline - 0.0001,
        },
        index=index,
    )
    return candidates, baseline


def test_full_metrics_identifies_stable_positive_candidate() -> None:
    candidates, baseline = _returns()

    metrics = full_metrics(candidates, baseline)

    assert metrics.at["stable", "delta_annualized_return_252"] > 0.0
    assert metrics.at["stable", "delta_sharpe"] > 0.0
    assert metrics.at["weak", "delta_annualized_return_252"] < 0.0


def test_cscv_and_temporal_selection_return_bounded_diagnostics() -> None:
    candidates, baseline = _returns()

    splits, summary = cscv_pbo(candidates, baseline, block_count=4)
    walk = expanding_walk_forward(candidates, baseline)
    leave = leave_one_year_selection(candidates, baseline)

    assert len(splits) == 3
    assert 0.0 <= summary["pbo"] <= 1.0
    assert not walk.empty
    assert len(leave) == 4
    assert walk["selected_candidate"].eq("stable").all()


def test_bootstrap_and_reality_check_are_deterministic() -> None:
    candidates, baseline = _returns()

    bootstrap, summary = paired_block_bootstrap(
        candidates["stable"],
        baseline,
        block_size=10,
        repetitions=30,
        seed=7,
    )
    reality = yearly_reality_check(
        candidates,
        baseline,
        repetitions=50,
        seed=7,
    )

    assert len(bootstrap) == 30
    assert summary["annualized_return_delta_positive_probability"] == 1.0
    assert reality["observed_best_candidate"] == "stable"
    assert 0.0 <= reality["p_value"] <= 1.0
