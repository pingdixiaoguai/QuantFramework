from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from defender.current_strategy import FORMAL_STRATEGY_ID, run_backtest
from defender.relative_defender_rotation import DEFENSIVE_ASSET, ROTATION_ASSETS
from defender.relative_defender_rotation_2013_report import BRIDGE_SIGNAL_ASSET


def _prices(dates: pd.DatetimeIndex, drift: float, phase: float) -> pd.DataFrame:
    x = np.arange(len(dates), dtype=float)
    close = 100.0 * np.exp(drift * x + 0.02 * np.sin(x / 17.0 + phase))
    open_ = close * (1.0 + 0.001 * np.sin(x + phase))
    return pd.DataFrame(
        {
            "date": dates,
            "open": open_,
            "high": np.maximum(open_, close) * 1.004,
            "low": np.minimum(open_, close) * 0.996,
            "close": close,
            "volume": 1.0,
        }
    )


def _listing_aware_market() -> dict[str, pd.DataFrame]:
    dates = pd.date_range("2013-01-02", periods=1000, freq="B")
    starts = {
        "510880.SH": 0,
        "512890.SH": 320,
        "159545.SZ": 700,
        "513530.SH": 600,
        "515080.SH": 500,
        "563020.SH": 800,
        DEFENSIVE_ASSET: 220,
    }
    market: dict[str, pd.DataFrame] = {}
    for index, asset in enumerate((*ROTATION_ASSETS, DEFENSIVE_ASSET)):
        start = starts[asset]
        market[asset] = _prices(
            dates[start:],
            0.00005 + index * 0.00004,
            index * 0.4,
        )
    return market


def test_vendored_formal_strategy_is_causal_and_listing_aware() -> None:
    market = _listing_aware_market()
    daily, _, metrics, events = run_backtest(market=market)

    assert metrics["strategy"] == FORMAL_STRATEGY_ID
    anchor_close = pd.Timestamp(metrics["anchor_first_close"])
    anchor_execution = pd.Timestamp(metrics["anchor_first_execution"])
    assert anchor_execution > anchor_close
    assert daily.loc[
        daily.index < anchor_execution, "signal_source_asset"
    ].eq(BRIDGE_SIGNAL_ASSET).all()
    assert daily.loc[
        daily.index >= anchor_execution, "signal_source_asset"
    ].eq("512890.SH").all()

    first_dates = {
        asset: pd.Timestamp(frame["date"].min()) for asset, frame in market.items()
    }
    for asset in ROTATION_ASSETS:
        chosen = daily.index[daily["selected_asset"].eq(asset)]
        if len(chosen):
            assert chosen.min() >= first_dates[asset]
    actual_weight_columns = [
        f"weight_{asset}" for asset in (*ROTATION_ASSETS, DEFENSIVE_ASSET)
    ]
    assert (
        daily[actual_weight_columns].sum(axis=1) + daily["cash_weight"]
    ).to_numpy() == pytest.approx(1.0, abs=1e-12)
    assert events["listed_assets"].str.len().gt(0).all()
