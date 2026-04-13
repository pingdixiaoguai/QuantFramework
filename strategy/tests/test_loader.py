"""Tests for strategy.loader."""

import pytest

from strategy.base import BaseStrategy
from strategy.loader import load_strategy
from strategy.momentum_rotation import MomentumRotation


class TestLoader:
    def test_default_loads_momentum_rotation(self):
        config = {"factors": []}
        strategy = load_strategy(config)
        assert isinstance(strategy, MomentumRotation)

    def test_explicit_class_path(self):
        config = {
            "strategy_class": "strategy.momentum_rotation.MomentumRotation",
            "factors": [],
        }
        strategy = load_strategy(config)
        assert isinstance(strategy, MomentumRotation)

    def test_invalid_class_path_raises(self):
        config = {"strategy_class": "nonexistent.module.Cls"}
        with pytest.raises(ModuleNotFoundError):
            load_strategy(config)

    def test_loaded_strategy_has_config(self):
        config = {"factors": [{"name": "x", "weight": 1.0}]}
        strategy = load_strategy(config)
        assert strategy.config is config
