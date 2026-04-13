"""Dynamic strategy loader."""

from __future__ import annotations

import importlib

from strategy.base import BaseStrategy


def load_strategy(config: dict) -> BaseStrategy:
    """Load a strategy class from config and return an instance.

    Uses config["strategy_class"] if present, otherwise defaults to
    MomentumRotation for backward compatibility.
    """
    class_path = config.get(
        "strategy_class",
        "strategy.momentum_rotation.MomentumRotation",
    )
    module_path, class_name = class_path.rsplit(".", 1)
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)
    return cls(config)
