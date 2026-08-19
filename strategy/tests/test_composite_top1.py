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


def test_centering_does_not_change_a_uniform_weight_ranking() -> None:
    values = {
        "A.SH": {"quality_momentum": 0.9, "rsi": 80.0},
        "B.SH": {"quality_momentum": 0.8, "rsi": 40.0},
        "C.SH": {"quality_momentum": 0.3, "rsi": 60.0},
        "D.SH": {"quality_momentum": 0.1, "rsi": 70.0},
    }
    uncentered = _strategy().generate_weights(values)
    centered = _strategy([
        {"name": "quality_momentum", "weight": 1.0},
        {
            "name": "rsi",
            "weight": 0.6,
            "direction_flip": True,
            "center_rank": True,
        },
    ]).generate_weights(values)

    assert centered == uncentered


def test_asset_specific_weights_use_centered_rank_contributions() -> None:
    strategy = _strategy([
        {"name": "quality_momentum", "weight": 1.0},
        {
            "name": "rsi",
            "weight": 0.0,
            "asset_weights": {"B.SH": 0.9},
            "direction_flip": True,
            "center_rank": True,
        },
    ])
    weights = strategy.generate_weights({
        "A.SH": {"quality_momentum": 0.9, "rsi": 80.0},
        "B.SH": {"quality_momentum": 0.8, "rsi": 40.0},
        "C.SH": {"quality_momentum": 0.3, "rsi": 60.0},
        "D.SH": {"quality_momentum": 0.1, "rsi": 70.0},
    })

    assert weights == {"B.SH": 1.0}


def test_centered_value_signal_can_adjust_primary_rank() -> None:
    strategy = _strategy([
        {"name": "quality_momentum", "weight": 1.0},
        {
            "name": "drawdown_percentile",
            "weight": 1.0,
            "score_mode": "centered_value",
            "center": 0.5,
            "scale": 2.0,
        },
    ])
    weights = strategy.generate_weights({
        "A.SH": {"quality_momentum": 0.9, "drawdown_percentile": 0.0},
        "B.SH": {"quality_momentum": 0.8, "drawdown_percentile": 1.0},
    })

    assert weights == {"B.SH": 1.0}


def test_unknown_score_mode_is_rejected() -> None:
    strategy = _strategy([
        {"name": "quality_momentum", "weight": 1.0, "score_mode": "raw"}
    ])

    try:
        strategy.generate_weights({"A.SH": {"quality_momentum": 0.9}})
    except ValueError as exc:
        assert "unknown score_mode" in str(exc)
    else:
        raise AssertionError("unknown score_mode should fail")
