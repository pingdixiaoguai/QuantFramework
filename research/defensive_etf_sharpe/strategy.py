"""Confirmed score-rotation strategy and comparable defensive baseline."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .engine import BacktestResult, MarketData, StrategyParams, load_market_data, simulate, simulate_static_allocation


ROOT = Path(__file__).parent
CASH_ASSET = "511880.SH"


def _profile(equity: float, sovereign: float, credit: float, cash: float) -> dict[str, float]:
    return {
        "510880.SH": equity / 3,
        "512890.SH": equity / 3,
        "515450.SH": equity / 3,
        "511010.SH": sovereign / 3,
        "511260.SH": sovereign / 3,
        "511090.SH": sovereign / 3,
        "511360.SH": credit,
        CASH_ASSET: cash,
    }


# This fixed allocation is a benchmark only. It is never used to choose the
# rotation strategy's monthly holding.
STATIC_BENCHMARK_TARGET = _profile(0.35, 0.40, 0.15, 0.10)


def load_confirmed_market(end: date | None = None) -> tuple[dict[str, dict[str, str]], MarketData]:
    with open(ROOT / "universe.yaml", encoding="utf-8") as handle:
        universe = yaml.safe_load(handle)["assets"]
    market = load_market_data(list(universe), date(2013, 1, 1), end or date.today())
    return universe, market


def score_rotation_params(universe: dict[str, dict[str, str]]) -> StrategyParams:
    """Monthly top-1 rotation using the repository's quality-momentum formula.

    The score is 20-trading-day momentum times Kaufman efficiency ratio. Only
    strictly positive scores qualify; otherwise the portfolio targets the
    money-market ETF and checks again after every subsequent close that month.
    All eight confirmed ETFs are scored, including 511880.
    """
    return StrategyParams(
        momentum_window=20,
        trend_window=1,
        volatility_window=2,
        top_n=1,
        weight_mode="equal",
        risk_assets=tuple(universe),
        cash_asset=CASH_ASSET,
        max_risk_asset_weight=1.0,
        rebalance_frequency="monthly_then_daily_until_positive",
        score_mode="momentum_times_er",
        min_score=0.0,
    )


def run_score_rotation(
    universe: dict[str, dict[str, str]], market: MarketData
) -> BacktestResult:
    return simulate(market, score_rotation_params(universe))


def metrics_for_daily(frame: pd.DataFrame) -> dict[str, float]:
    returns = frame["return"].dropna().astype(float)
    if returns.empty:
        return {"annualized_return": 0.0, "volatility": 0.0, "sharpe": 0.0, "max_drawdown": 0.0}
    curve = (1.0 + returns).cumprod()
    volatility = float(returns.std(ddof=1) * np.sqrt(252.0))
    drawdown = curve / curve.cummax() - 1.0
    return {
        "annualized_return": float(curve.iloc[-1] ** (252.0 / len(returns)) - 1.0),
        "volatility": volatility,
        "sharpe": float(returns.mean() / returns.std(ddof=1) * np.sqrt(252.0)) if returns.std(ddof=1) > 0 else 0.0,
        "max_drawdown": float(drawdown.min()),
    }
