"""Tests for removable Gold RAQM regularization parameters."""

import numpy as np
import pandas as pd

from research.gold_raqm_regularization import RAQMSpec, raqm_score


def test_raw_raqm_has_no_floor_or_clip():
    curve = pd.Series([1.0, 1.2, 1.0, 1.5])
    spec = RAQMSpec("raw", 3, None, None, 0)
    daily = np.diff(np.log(curve.to_numpy()))
    total = np.log(1.5)
    expected = (
        total / (daily.std(ddof=1) * np.sqrt(3.0))
        * abs(total) / np.abs(daily).sum()
    )
    assert np.isclose(raqm_score(curve, spec).iloc[-1], expected)


def test_floor_and_clip_are_independently_optional():
    daily = np.array([0.0010, 0.0011, 0.0009])
    curve = pd.Series(np.exp(np.r_[0.0, daily.cumsum()]))
    raw = raqm_score(curve, RAQMSpec("raw", 3, None, None, 0)).iloc[-1]
    floor_only = raqm_score(
        curve, RAQMSpec("floor_only", 3, 0.12, None, 1)
    ).iloc[-1]
    winsor_only = raqm_score(
        curve, RAQMSpec("winsor_only", 3, None, 2.0, 1)
    ).iloc[-1]
    assert floor_only < raw
    assert winsor_only <= raw
