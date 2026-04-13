"""Tests for strategy.momentum_rotation."""

import pytest

from strategy.momentum_rotation import MomentumRotation


def _config(factors=None):
    return {
        "factors": factors or [
            {"name": "momentum", "weight": 0.7, "params": {"window": 20}},
            {"name": "volatility", "weight": 0.3, "direction_flip": True, "params": {"window": 20}},
        ],
    }


class TestMomentumRotation:
    def test_empty_input_returns_empty(self):
        s = MomentumRotation(_config())
        assert s.generate_weights({}) == {}

    def test_single_asset_gets_full_weight(self):
        s = MomentumRotation(_config())
        result = s.generate_weights({
            "A.SH": {"momentum": 0.5, "volatility": 0.3},
        })
        assert result == {"A.SH": 1.0}

    def test_weights_sum_to_one(self):
        s = MomentumRotation(_config())
        result = s.generate_weights({
            "A.SH": {"momentum": 10.0, "volatility": 5.0},
            "B.SH": {"momentum": 20.0, "volatility": 3.0},
            "C.SH": {"momentum": 15.0, "volatility": 8.0},
        })
        assert abs(sum(result.values()) - 1.0) < 1e-10

    def test_ranking_order(self):
        """Higher momentum → higher rank → higher weight (no flip)."""
        config = _config([{"name": "momentum", "weight": 1.0}])
        s = MomentumRotation(config)
        result = s.generate_weights({
            "LOW.SH": {"momentum": 1.0},
            "HIGH.SH": {"momentum": 10.0},
        })
        assert result["HIGH.SH"] > result["LOW.SH"]

    def test_direction_flip(self):
        """With direction_flip, lower volatility → higher weight."""
        config = _config([{"name": "volatility", "weight": 1.0, "direction_flip": True}])
        s = MomentumRotation(config)
        result = s.generate_weights({
            "LOW_VOL.SH": {"volatility": 1.0},
            "HIGH_VOL.SH": {"volatility": 10.0},
        })
        assert result["LOW_VOL.SH"] > result["HIGH_VOL.SH"]

    def test_multi_factor_weighted(self):
        """Two factors with different weights produce blended scores."""
        config = _config([
            {"name": "alpha", "weight": 0.8},
            {"name": "beta", "weight": 0.2, "direction_flip": True},
        ])
        s = MomentumRotation(config)
        result = s.generate_weights({
            "A.SH": {"alpha": 100.0, "beta": 1.0},   # high alpha, low beta
            "B.SH": {"alpha": 1.0, "beta": 100.0},    # low alpha, high beta
        })
        # A should win: high alpha (big weight) + low beta (flipped, also good)
        assert result["A.SH"] > result["B.SH"]
