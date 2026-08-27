"""Tests for the frozen historical Momentum baseline factor."""

import numpy as np
import pandas as pd

from factors.legacy_quality_momentum import compute


def test_legacy_formula_uses_simple_momentum_and_price_path():
    prices = [100.0, 200.0, 100.0, 150.0]
    frame = pd.DataFrame(
        {"date": pd.bdate_range("2024-01-01", periods=4), "close": prices}
    )
    result = compute(frame, {"window": 3})
    expected_momentum = 150.0 / 100.0 - 1.0
    expected_er = abs(150.0 - 100.0) / (100.0 + 100.0 + 50.0)
    assert np.isclose(result.iloc[-1], expected_momentum * expected_er)
