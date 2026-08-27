import numpy as np
import pandas as pd

from research.defensive_etf_sharpe.engine import MarketData
from research.defensive_etf_sharpe.rebalance_timing import (
    _execute_target_with_min_notional,
    simulate_timed_allocation,
)


def test_monthly_deposit_is_invested_on_same_day_without_forcing_sales() -> None:
    dates = list(pd.date_range("2020-01-01", periods=45, freq="B"))
    prices = pd.Series(100.0, index=dates)
    market = MarketData(
        opens={"A": prices, "CASH": prices},
        closes={"A": prices, "CASH": prices},
        dates=dates,
    )
    result = simulate_timed_allocation(
        market,
        {},
        initial_target={"A": 0.5, "CASH": 0.5},
        cash_asset="CASH",
    )
    deposit_dates = result.daily.index[result.daily["deposit"] > 0]
    deposit_trades = result.trades.loc[result.trades["reason"] == "deposit_invest"]
    assert set(pd.to_datetime(deposit_trades["date"])).issuperset(set(deposit_dates))
    assert set(deposit_trades["side"]) == {"buy"}
    assert (deposit_trades["shares"] % 100 == 0).all()
    assert np.isclose(result.total_deposits, len(deposit_dates) * 20_000.0)


def test_close_signal_executes_at_next_open() -> None:
    dates = list(pd.date_range("2020-01-01", periods=10, freq="B"))
    prices = pd.Series(100.0, index=dates)
    market = MarketData(
        opens={"A": prices, "CASH": prices},
        closes={"A": prices, "CASH": prices},
        dates=dates,
    )
    result = simulate_timed_allocation(
        market,
        {dates[2]: {"A": 1.0, "CASH": 0.0}},
        initial_target={"A": 0.0, "CASH": 1.0},
        cash_asset="CASH",
    )
    rebalance_dates = pd.to_datetime(
        result.trades.loc[result.trades["reason"] == "rebalance", "date"]
    )
    assert dates[3] in set(rebalance_dates)
    assert dates[2] not in set(rebalance_dates)


def test_small_sells_block_unfunded_buy() -> None:
    cash, holdings, trades = _execute_target_with_min_notional(
        0.0,
        {"A": 100, "B": 100},
        {"A": 0.0, "B": 0.0, "C": 1.0},
        {"A": 60.0, "B": 60.0, "C": 60.0},
        {"A": 60.0, "B": 60.0, "C": 60.0},
        0.0,
        100,
        10_000.0,
    )
    assert cash == 0.0
    assert holdings == {"A": 100, "B": 100}
    assert trades == []


def test_eligible_sell_funds_only_affordable_buy_after_small_sell_is_blocked() -> None:
    cash, holdings, trades = _execute_target_with_min_notional(
        0.0,
        {"A": 100, "B": 100},
        {"A": 0.0, "B": 0.0, "C": 1.0},
        {"A": 100.0, "B": 20.0, "C": 100.0},
        {"A": 100.0, "B": 20.0, "C": 100.0},
        0.0,
        100,
        10_000.0,
    )
    assert cash == 0.0
    assert holdings == {"B": 100, "C": 100}
    assert [(trade["asset"], trade["side"], trade["notional"]) for trade in trades] == [
        ("A", "sell", 10_000.0),
        ("C", "buy", 10_000.0),
    ]


def test_deposit_waits_for_next_day_rebalance_when_immediate_investment_disabled() -> None:
    dates = list(pd.date_range("2020-01-01", periods=5, freq="B"))
    prices = pd.Series(100.0, index=dates)
    market = MarketData(
        opens={"A": prices, "CASH": prices},
        closes={"A": prices, "CASH": prices},
        dates=dates,
    )
    result = simulate_timed_allocation(
        market,
        {dates[0]: {"A": 0.5, "CASH": 0.5}},
        initial_target={"A": 0.5, "CASH": 0.5},
        cash_asset="CASH",
        invest_deposits_immediately=False,
    )
    assert result.trades["date"].min() == dates[1]
    assert set(result.trades["reason"]) == {"rebalance"}
