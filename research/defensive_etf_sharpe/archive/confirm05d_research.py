"""Backtest the 10% drift trigger with a 5-day confirmation requirement.

Compares ``portfolio_drift_10_confirm05d_monthly_cap1`` against the current
strategy (``portfolio_drift_10_monthly_cap1``) and the fixed monthly-check
baseline (``calendar_monthly_reference``) under identical deposits, costs,
lot sizes, and the 10,000 minimum rebalance notional.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .factor_research import _extended_metrics
from .rebalance_timing import daily_reversal_targets
from .reduced_pool_research import REDUCED_TARGET
from .strategy import CASH_ASSET, load_confirmed_market
from .threshold_rebalance import TRIGGER_SPECS, simulate_threshold_rebalance

ROOT = Path(__file__).parent
OUTPUT = ROOT / "threshold_rebalance_experiments"

COMPARED = (
    "calendar_monthly_reference",
    "portfolio_drift_10_monthly_cap1",
    "portfolio_drift_10_confirm05d_monthly_cap1",
)
NEW_STRATEGY = "portfolio_drift_10_confirm05d_monthly_cap1"


def run() -> tuple[pd.DataFrame, pd.DataFrame]:
    _, market = load_confirmed_market()
    daily_targets = daily_reversal_targets(market, REDUCED_TARGET)
    specs = {spec.name: spec for spec in TRIGGER_SPECS}
    OUTPUT.mkdir(parents=True, exist_ok=True)

    rows = []
    results = {}
    for name in COMPARED:
        result = simulate_threshold_rebalance(
            market,
            daily_targets,
            specs[name],
            initial_target=REDUCED_TARGET,
            cash_asset=CASH_ASSET,
            min_rebalance_notional=10_000.0,
        )
        results[name] = result
        rebalance_trades = result.trades.loc[result.trades["reason"] == "threshold_rebalance"]
        rows.append({
            "strategy": name,
            **_extended_metrics(result),
            "trigger_signals": len(result.signals),
            "rebalance_dates": pd.to_datetime(rebalance_trades["date"]).nunique()
            if not rebalance_trades.empty else 0,
            "deposit_trade_count": int((result.trades["reason"] == "deposit_invest").sum()),
            "rebalance_trade_count": int((result.trades["reason"] == "threshold_rebalance").sum()),
            "average_cash_weight": float(result.daily["cash_weight"].mean()),
        })

    metrics = pd.DataFrame(rows).set_index("strategy")
    metrics.to_csv(OUTPUT / "confirm05d_comparison_metrics.csv")

    new = results[NEW_STRATEGY]
    new.daily.to_csv(OUTPUT / f"{NEW_STRATEGY}_daily.csv")
    new.trades.to_csv(OUTPUT / f"{NEW_STRATEGY}_trades.csv", index=False)
    new.signals.to_csv(OUTPUT / f"{NEW_STRATEGY}_trigger_signals.csv", index=False)

    annual = pd.DataFrame({
        name: (1.0 + result.daily["return"].astype(float).fillna(0.0)).groupby(
            result.daily.index.year
        ).prod() - 1.0
        for name, result in results.items()
    })
    annual.to_csv(OUTPUT / "confirm05d_annual_returns.csv")
    return metrics, annual


if __name__ == "__main__":
    all_metrics, all_annual = run()
    display = all_metrics[[
        "annualized_return", "volatility", "sharpe", "max_drawdown", "sortino",
        "calmar", "final_nav", "trades", "estimated_transaction_cost",
        "trigger_signals", "rebalance_dates", "average_cash_weight",
    ]].copy()
    for column in ("annualized_return", "volatility", "max_drawdown", "average_cash_weight"):
        display[column] = display[column].map(lambda value: f"{value:.2%}")
    for column in ("sharpe", "sortino", "calmar"):
        display[column] = display[column].map(lambda value: f"{value:.2f}")
    display["final_nav"] = display["final_nav"].map(lambda value: f"{value:,.0f}")
    display["estimated_transaction_cost"] = display["estimated_transaction_cost"].map(
        lambda value: f"{value:,.0f}"
    )
    print(display.to_string())
    print("\nannual returns:")
    print(all_annual.map(lambda value: f"{value:.2%}").to_string())
