"""Tests for strategy.composite_top1."""

from strategy.composite_top1 import CompositeTop1


def _strategy(factors=None) -> CompositeTop1:
    return CompositeTop1({
        "factors": factors
        or [
            {"name": "quality_momentum", "weight": 1.0},
            {"name": "rsi", "weight": 0.6, "direction_flip": True},
        ]
    })


def test_empty_input_returns_empty_weights() -> None:
    assert _strategy().generate_weights({}) == {}


def test_empty_factor_config_returns_empty_weights() -> None:
    assert CompositeTop1({"factors": []}).generate_weights({
        "A.SH": {"quality_momentum": 1.0}
    }) == {}


def test_single_asset_gets_full_weight() -> None:
    weights = _strategy().generate_weights({
        "A.SH": {"quality_momentum": 0.8, "rsi": 65.0}
    })
    assert weights == {"A.SH": 1.0}


def test_two_rank_quality_lead_is_not_overturned() -> None:
    weights = _strategy().generate_weights({
        "A.SH": {"quality_momentum": 0.9, "rsi": 80.0},
        "B.SH": {"quality_momentum": 0.3, "rsi": 50.0},
        "C.SH": {"quality_momentum": 0.6, "rsi": 70.0},
        "D.SH": {"quality_momentum": 0.1, "rsi": 60.0},
    })
    assert weights == {"A.SH": 1.0}


def test_inverse_rsi_rank_breaks_close_primary_rank_choice() -> None:
    weights = _strategy().generate_weights({
        "A.SH": {"quality_momentum": 0.9, "rsi": 80.0},
        "B.SH": {"quality_momentum": 0.8, "rsi": 40.0},
        "C.SH": {"quality_momentum": 0.3, "rsi": 60.0},
        "D.SH": {"quality_momentum": 0.1, "rsi": 70.0},
    })
    assert weights == {"B.SH": 1.0}


def test_factor_without_direction_flip_rewards_higher_values() -> None:
    strategy = _strategy([
        {"name": "quality_momentum", "weight": 1.0},
        {"name": "rsi", "weight": 1.0},
    ])
    weights = strategy.generate_weights({
        "A.SH": {"quality_momentum": 0.9, "rsi": 80.0},
        "B.SH": {"quality_momentum": 0.8, "rsi": 40.0},
    })
    assert weights == {"A.SH": 1.0}
