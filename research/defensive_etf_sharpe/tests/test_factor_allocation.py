import numpy as np

from research.defensive_etf_sharpe.factor_allocation import (
    MECHANISM_SPECS,
    SLEEVE_GROUPS,
    MechanismSpec,
    adjusted_weights,
)
from research.defensive_etf_sharpe.strategy import STATIC_BENCHMARK_TARGET


def test_all_adjustment_mechanisms_are_long_only_and_fully_invested() -> None:
    ranks = {asset: value for asset, value in zip(STATIC_BENCHMARK_TARGET, np.linspace(-1.0, 1.0, 8))}
    for mechanism in MECHANISM_SPECS:
        weights = adjusted_weights(ranks, mechanism)
        assert np.isclose(sum(weights.values()), 1.0)
        assert min(weights.values()) >= 0.0


def test_sleeve_tilt_preserves_each_baseline_sleeve_budget() -> None:
    ranks = {asset: value for asset, value in zip(STATIC_BENCHMARK_TARGET, np.linspace(-1.0, 1.0, 8))}
    mechanisms = [item for item in MECHANISM_SPECS if item.kind == "sleeve_tilt"]
    for mechanism in mechanisms:
        weights = adjusted_weights(ranks, mechanism)
        for group in SLEEVE_GROUPS:
            expected = sum(STATIC_BENCHMARK_TARGET[asset] for asset in group)
            actual = sum(weights[asset] for asset in group)
            assert np.isclose(actual, expected)


MAGNITUDE_MECHANISMS = (
    MechanismSpec("exp", "exp_tilt", 0.5, ""),
    MechanismSpec("sigmoid", "sigmoid_tilt", 1.0, ""),
    MechanismSpec("additive", "additive_tilt", 0.05, ""),
)


def test_magnitude_mechanisms_are_long_only_and_fully_invested() -> None:
    scores = {asset: value for asset, value in zip(STATIC_BENCHMARK_TARGET, np.linspace(-2.0, 2.0, 8))}
    for mechanism in MAGNITUDE_MECHANISMS:
        weights = adjusted_weights(scores, mechanism)
        assert np.isclose(sum(weights.values()), 1.0)
        assert min(weights.values()) >= 0.0


def test_magnitude_mechanisms_reduce_to_baseline_when_scores_are_tied() -> None:
    scores = {asset: 0.0 for asset in STATIC_BENCHMARK_TARGET}
    for mechanism in MAGNITUDE_MECHANISMS:
        weights = adjusted_weights(scores, mechanism)
        for asset, weight in STATIC_BENCHMARK_TARGET.items():
            assert np.isclose(weights[asset], weight)


def test_magnitude_mechanisms_respect_score_ordering() -> None:
    scores = {asset: value for asset, value in zip(STATIC_BENCHMARK_TARGET, np.linspace(-1.5, 1.5, 8))}
    for mechanism in MAGNITUDE_MECHANISMS:
        weights = adjusted_weights(scores, mechanism)
        ordered = [weights[asset] / STATIC_BENCHMARK_TARGET[asset] for asset in STATIC_BENCHMARK_TARGET]
        assert ordered == sorted(ordered)
