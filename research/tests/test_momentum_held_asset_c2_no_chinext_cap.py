import pandas as pd
import pytest

from research.run_momentum_held_asset_c2_no_chinext_cap import (
    suppress_alert_for_asset,
)


def test_suppress_alert_for_asset_changes_only_matching_asset_alerts() -> None:
    index = pd.date_range("2024-01-01", periods=5)
    alert = pd.Series([True, True, False, True, True], index=index, name="alert")
    previous_asset = pd.Series(
        ["159915.SZ", "510300.SH", "159915.SZ", "159915.SZ", "518880.SH"],
        index=index,
    )

    result = suppress_alert_for_asset(alert, previous_asset, "159915.SZ")

    expected = pd.Series(
        [False, True, False, False, True], index=index, name="alert"
    )
    pd.testing.assert_series_equal(result, expected)


def test_suppress_alert_for_asset_requires_identical_index() -> None:
    alert = pd.Series([True], index=[pd.Timestamp("2024-01-01")])
    previous_asset = pd.Series(["159915.SZ"], index=[pd.Timestamp("2024-01-02")])

    with pytest.raises(ValueError, match="identical indexes"):
        suppress_alert_for_asset(alert, previous_asset, "159915.SZ")
