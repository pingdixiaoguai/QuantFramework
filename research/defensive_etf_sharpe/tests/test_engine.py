from datetime import date

import numpy as np
import pandas as pd

from research.defensive_etf_sharpe.engine import (
    MarketData,
    StrategyParams,
    simulate,
    simulate_buy_and_hold,
    simulate_static_allocation,
)
from research.defensive_etf_sharpe.engine import _signal


def _market() -> MarketData:
    dates = pd.date_range("2013-01-01", periods=260, freq="B")
    close = pd.Series(np.linspace(10.0, 20.0, len(dates)), index=dates)
    opens = close * 0.999
    return MarketData(
        opens={"A": opens},
        closes={"A": close},
        dates=list(dates),
    )


def test_monthly_deposits_and_integer_lots_are_modelled() -> None:
    result = simulate(
        _market(),
        StrategyParams(momentum_window=20, trend_window=40, volatility_window=20),
        monthly_deposit=20_000,
        cost_rate=0.0005,
    )
    assert result.total_deposits == 240_000
    assert result.final_nav > result.total_deposits
    assert (result.trades["shares"] % 100 == 0).all()


def test_empty_target_allows_cash() -> None:
    market = _market()
    market.closes["A"] = pd.Series(np.linspace(20.0, 10.0, len(market.dates)), index=market.dates)
    market.opens["A"] = market.closes["A"]
    result = simulate(
        market,
        StrategyParams(momentum_window=20, trend_window=40, volatility_window=20),
    )
    assert result.daily["cash_weight"].iloc[-1] == 1.0


def test_buy_and_hold_baseline_uses_integer_lots_and_deposits() -> None:
    result = simulate_buy_and_hold(_market(), "A", monthly_deposit=20_000)
    assert result["deposit"].sum() == 240_000
    assert result["nav"].iloc[-1] > result["deposit"].sum()


def test_static_allocation_respects_cash_sleeve_and_integer_lots() -> None:
    market = _market()
    market.opens["CASH"] = pd.Series(100.0, index=market.dates)
    market.closes["CASH"] = pd.Series(101.0, index=market.dates)
    result = simulate_static_allocation(
        market,
        {"A": 0.5, "CASH": 0.5},
        cash_asset="CASH",
        monthly_deposit=20_000,
    )
    assert result.total_deposits == 240_000
    assert {"A", "CASH"}.issubset(set(result.trades["asset"]))
    assert (result.trades["shares"] % 100 == 0).all()


def test_momentum_times_er_selects_highest_positive_score() -> None:
    dates = pd.date_range("2020-01-01", periods=21, freq="B")
    # A has a smooth 20% gain (ER=1); B has a lower, choppy positive gain.
    a = pd.Series(np.linspace(100.0, 120.0, len(dates)), index=dates)
    b = pd.Series([100.0, 105.0] * 10 + [108.0], index=dates)
    market = MarketData(
        opens={"A": a, "B": b, "CASH": pd.Series(100.0, index=dates)},
        closes={"A": a, "B": b, "CASH": pd.Series(100.0, index=dates)},
        dates=list(dates),
    )
    target, diagnostics = _signal(
        market,
        dates[-1],
        StrategyParams(
            momentum_window=20,
            trend_window=1,
            volatility_window=2,
            top_n=1,
            weight_mode="equal",
            risk_assets=("A", "B"),
            cash_asset="CASH",
            rebalance_frequency="monthly",
            score_mode="momentum_times_er",
            min_score=0.0,
        ),
    )
    assert target == {"A": 1.0}
    assert diagnostics["A"]["score"] > diagnostics["B"]["score"]


def test_no_positive_score_checks_daily_until_first_positive_score() -> None:
    dates = pd.date_range("2020-01-01", periods=45, freq="B")
    a = pd.Series([100.0] * 30 + list(np.linspace(101.0, 110.0, 15)), index=dates)
    cash = pd.Series(100.0, index=dates)
    market = MarketData(
        opens={"A": a, "CASH": cash},
        closes={"A": a, "CASH": cash},
        dates=list(dates),
    )
    result = simulate(
        market,
        StrategyParams(
            momentum_window=20,
            trend_window=1,
            volatility_window=2,
            top_n=1,
            weight_mode="equal",
            risk_assets=("A", "CASH"),
            cash_asset="CASH",
            rebalance_frequency="monthly_then_daily_until_positive",
            score_mode="momentum_times_er",
            min_score=0.0,
        ),
    )
    selected_a = result.signals.loc[
        (result.signals["asset"] == "A")
        & result.signals["selected"]
        & (result.signals["trigger"] == "wait_for_positive")
    ]
    assert len(selected_a) == 1
    assert selected_a.iloc[0]["trigger"] == "wait_for_positive"
    assert (result.trades["asset"] == "A").any()
