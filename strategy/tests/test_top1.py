"""Tests for strategy.top1."""

from strategy.top1 import Top1


def _config(factors=None):
    return {
        "factors": factors or [{"name": "qmom", "weight": 1.0}],
    }


class TestTop1:
    def test_empty_input(self):
        s = Top1(_config())
        assert s.generate_weights({}) == {}

    def test_single_asset(self):
        s = Top1(_config())
        result = s.generate_weights({"A.SH": {"qmom": 0.5}})
        assert result == {"A.SH": 1.0}

    def test_picks_highest(self):
        s = Top1(_config())
        result = s.generate_weights({
            "A.SH": {"qmom": 0.1},
            "B.SH": {"qmom": 0.9},
            "C.SH": {"qmom": 0.5},
        })
        assert result == {"B.SH": 1.0}

    def test_direction_flip_picks_lowest(self):
        config = _config([{"name": "vol", "weight": 1.0, "direction_flip": True}])
        s = Top1(config)
        result = s.generate_weights({
            "A.SH": {"vol": 0.1},
            "B.SH": {"vol": 0.9},
        })
        assert result == {"A.SH": 1.0}

    def test_weights_sum_to_one(self):
        s = Top1(_config())
        result = s.generate_weights({
            "A.SH": {"qmom": 0.3},
            "B.SH": {"qmom": 0.7},
            "C.SH": {"qmom": 0.5},
            "D.SH": {"qmom": 0.1},
        })
        assert sum(result.values()) == 1.0
        assert len(result) == 1

    def test_hysteresis_keeps_incumbent_below_threshold(self):
        s = Top1({**_config(), "hysteresis_threshold": 0.05})
        result = s.generate_weights(
            {
                "A.SH": {"qmom": 0.50},
                "B.SH": {"qmom": 0.53},
            },
            current_weights={"A.SH": 1.0},
        )
        assert result == {"A.SH": 1.0}

    def test_hysteresis_switches_when_challenger_clears_threshold(self):
        s = Top1({**_config(), "hysteresis_threshold": 0.05})
        result = s.generate_weights(
            {
                "A.SH": {"qmom": 0.50},
                "B.SH": {"qmom": 0.56},
            },
            current_weights={"A.SH": 1.0},
        )
        assert result == {"B.SH": 1.0}

    def test_hysteresis_falls_back_when_incumbent_score_missing(self):
        s = Top1({**_config(), "hysteresis_threshold": 0.05})
        result = s.generate_weights(
            {"B.SH": {"qmom": 0.53}},
            current_weights={"A.SH": 1.0},
        )
        assert result == {"B.SH": 1.0}

    def test_direction_flip_hysteresis_requires_lower_challenger(self):
        config = _config([{"name": "vol", "weight": 1.0, "direction_flip": True}])
        s = Top1({**config, "hysteresis_threshold": 0.05})
        result = s.generate_weights(
            {
                "A.SH": {"vol": 0.50},
                "B.SH": {"vol": 0.46},
            },
            current_weights={"A.SH": 1.0},
        )
        assert result == {"A.SH": 1.0}
