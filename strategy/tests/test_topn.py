"""Tests for strategy.topn."""

import pytest

from strategy.topn import TopN


def _config(top_n=5, factors=None):
    return {
        "top_n": top_n,
        "factors": factors or [{"name": "qmom", "weight": 1.0}],
    }


class TestTopN:
    def test_empty_input(self):
        s = TopN(_config())
        assert s.generate_weights({}) == {}

    def test_no_factors_configured(self):
        s = TopN({"top_n": 5, "factors": []})
        result = s.generate_weights({"A.SH": {"qmom": 0.5}})
        assert result == {}

    def test_picks_top_n(self):
        s = TopN(_config(top_n=3))
        result = s.generate_weights({
            "A.SH": {"qmom": 0.1},
            "B.SH": {"qmom": 0.9},
            "C.SH": {"qmom": 0.5},
            "D.SH": {"qmom": 0.7},
            "E.SH": {"qmom": 0.3},
        })
        # Top 3 by descending qmom: B (0.9), D (0.7), C (0.5)
        assert set(result.keys()) == {"B.SH", "D.SH", "C.SH"}
        for w in result.values():
            assert w == pytest.approx(1.0 / 3)

    def test_weights_sum_to_one(self):
        s = TopN(_config(top_n=3))
        result = s.generate_weights({
            "A.SH": {"qmom": 0.1},
            "B.SH": {"qmom": 0.9},
            "C.SH": {"qmom": 0.5},
            "D.SH": {"qmom": 0.7},
        })
        assert sum(result.values()) == pytest.approx(1.0)

    def test_direction_flip_picks_lowest(self):
        cfg = _config(top_n=2, factors=[{"name": "vol", "weight": 1.0, "direction_flip": True}])
        s = TopN(cfg)
        result = s.generate_weights({
            "A.SH": {"vol": 0.1},
            "B.SH": {"vol": 0.9},
            "C.SH": {"vol": 0.5},
            "D.SH": {"vol": 0.3},
        })
        # Lowest 2: A (0.1), D (0.3)
        assert set(result.keys()) == {"A.SH", "D.SH"}
        for w in result.values():
            assert w == pytest.approx(0.5)

    def test_n_greater_than_candidates(self):
        s = TopN(_config(top_n=10))
        result = s.generate_weights({
            "A.SH": {"qmom": 0.1},
            "B.SH": {"qmom": 0.9},
        })
        assert set(result.keys()) == {"A.SH", "B.SH"}
        for w in result.values():
            assert w == pytest.approx(0.5)

    def test_drops_assets_missing_factor(self):
        s = TopN(_config(top_n=2))
        result = s.generate_weights({
            "A.SH": {"qmom": 0.5},
            "B.SH": {"other": 0.9},  # missing qmom
            "C.SH": {"qmom": 0.7},
        })
        assert set(result.keys()) == {"A.SH", "C.SH"}

    def test_default_top_n_is_5(self):
        # Omit top_n in config
        s = TopN({"factors": [{"name": "qmom", "weight": 1.0}]})
        scores = {f"A{i}.SH": {"qmom": float(i)} for i in range(10)}
        result = s.generate_weights(scores)
        assert len(result) == 5
        # Top 5 by descending value: 9, 8, 7, 6, 5
        assert set(result.keys()) == {"A9.SH", "A8.SH", "A7.SH", "A6.SH", "A5.SH"}
