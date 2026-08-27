import pandas as pd

from research.defensive_etf_sharpe.engine import MarketData
from research.defensive_etf_sharpe.threshold_rebalance import TriggerSpec, simulate_threshold_rebalance


def _market(periods: int = 45) -> MarketData:
    dates = list(pd.date_range("2020-01-01", periods=periods, freq="B"))
    prices = pd.Series(100.0, index=dates)
    return MarketData(
        opens={"A": prices, "CASH": prices},
        closes={"A": prices, "CASH": prices},
        dates=dates,
    )


def test_monthly_deposit_uses_previous_trading_day_target() -> None:
    market = _market()
    switch_date = pd.Timestamp("2020-01-31")
    targets = {
        date: ({"A": 1.0, "CASH": 0.0} if date >= switch_date else {"A": 0.0, "CASH": 1.0})
        for date in market.dates
    }
    result = simulate_threshold_rebalance(
        market,
        targets,
        TriggerSpec("never", "target_change", 2.0),
        initial_target={"A": 0.0, "CASH": 1.0},
        cash_asset="CASH",
        min_rebalance_notional=0.0,
    )
    february_first = next(date for date in market.dates if date.month == 2)
    trades = result.trades.loc[
        (pd.to_datetime(result.trades["date"]) == february_first)
        & (result.trades["reason"] == "deposit_invest")
    ]
    assert "A" in set(trades["asset"])


def test_trigger_signal_executes_on_next_open_with_minimum_notional() -> None:
    market = _market(10)
    targets = {
        date: ({"A": 1.0, "CASH": 0.0} if index >= 2 else {"A": 0.0, "CASH": 1.0})
        for index, date in enumerate(market.dates)
    }
    result = simulate_threshold_rebalance(
        market,
        targets,
        TriggerSpec("change", "target_change", 0.10),
        initial_target={"A": 0.0, "CASH": 1.0},
        cash_asset="CASH",
        min_rebalance_notional=10_000.0,
    )
    signal_date = pd.to_datetime(result.signals.iloc[0]["date"])
    execution_date = pd.to_datetime(
        result.trades.loc[result.trades["reason"] == "threshold_rebalance", "date"]
    ).min()
    signal_position = market.dates.index(signal_date)
    assert execution_date == market.dates[signal_position + 1]
    assert result.trades.loc[result.trades["reason"] == "threshold_rebalance", "notional"].min() >= 10_000


def test_monthly_cap_allows_at_most_one_rebalance_execution_per_month() -> None:
    market = _market(65)
    targets = {
        date: ({"A": 1.0, "CASH": 0.0} if index % 2 == 0 else {"A": 0.0, "CASH": 1.0})
        for index, date in enumerate(market.dates)
    }
    result = simulate_threshold_rebalance(
        market,
        targets,
        TriggerSpec(
            "capped",
            "portfolio_drift",
            0.15,
            max_rebalances_per_month=1,
        ),
        initial_target={"A": 0.0, "CASH": 1.0},
        cash_asset="CASH",
        min_rebalance_notional=0.0,
    )
    dates = pd.to_datetime(
        result.trades.loc[result.trades["reason"] == "threshold_rebalance", "date"]
    ).drop_duplicates()
    counts = dates.groupby([dates.dt.year, dates.dt.month]).size()
    assert (counts <= 1).all()


def test_calendar_or_drift_triggers_on_month_start_below_threshold() -> None:
    market = _market(45)
    targets = {date: {"A": 1.0, "CASH": 0.0} for date in market.dates}
    result = simulate_threshold_rebalance(
        market,
        targets,
        TriggerSpec("combo", "calendar_or_portfolio_drift", 0.99),
        initial_target={"A": 1.0, "CASH": 0.0},
        cash_asset="CASH",
        min_rebalance_notional=0.0,
    )
    # drift alone never reaches 99%: every signal must fall on a month start
    first_days = {}
    for date in market.dates:
        first_days.setdefault((date.year, date.month), date)
    signal_dates = set(pd.to_datetime(result.signals["date"]))
    expected = {date for date in first_days.values() if date < market.dates[-1]}
    assert signal_dates == expected
