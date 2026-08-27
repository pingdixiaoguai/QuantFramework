import pandas as pd
import pytest

from research.run_momentum_held_asset_c2_chinext_csi300_confirmation import (
    confirm_chinext_alert_with_csi300,
)


def test_confirmation_applies_only_while_chinext_is_held() -> None:
    index = pd.date_range("2024-01-01", periods=5)
    original = pd.Series([True, True, True, False, True], index=index)
    previous_asset = pd.Series(
        ["159915.SZ", "159915.SZ", "510300.SH", "159915.SZ", "518880.SH"],
        index=index,
    )
    csi300 = pd.Series([True, False, False, True, False], index=index)

    result = confirm_chinext_alert_with_csi300(original, previous_asset, csi300)

    expected = pd.Series(
        [True, False, True, False, True],
        index=index,
        name="c2_chinext_q90_csi300_q70_confirmation_alert",
    )
    pd.testing.assert_series_equal(result, expected)


def test_confirmation_requires_aligned_inputs() -> None:
    original = pd.Series([True], index=[pd.Timestamp("2024-01-01")])
    previous_asset = pd.Series(["159915.SZ"], index=original.index)
    csi300 = pd.Series([True], index=[pd.Timestamp("2024-01-02")])

    with pytest.raises(ValueError, match="identical indexes"):
        confirm_chinext_alert_with_csi300(original, previous_asset, csi300)
