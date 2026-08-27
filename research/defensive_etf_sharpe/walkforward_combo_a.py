"""Rolling out-of-sample (walk-forward) validation of combo A strength 0.70.

Combo A = risk-only ranking universe (512890/511260/511360 share the 90%
risk budget, 511880 money sleeve fixed at 10%) + rank linear tilt.

Protocol: at each calendar year boundary Y in 2017..2026, the tilt strength
lambda*(Y) is selected on a real capital-path backtest using only data before
Y (selection rule: annualized return >= 5%, then highest Sharpe). The
per-year lambda choices are stitched into one target schedule, so the final
walk-forward backtest keeps a single continuous capital path: only the
parameter changes at year boundaries, never the holdings. Before 2017 the
baseline strength 0.50 is used (no in-sample window would be long enough).

Two selection windows are compared: expanding (2013..Y-1) and trailing 36
months. Run with: python -m research.defensive_etf_sharpe.walkforward_combo_a
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from .calendar_ablation import (
    BASELINE_TARGET,
    CALENDAR_TRIGGER,
    EXTRA_POOL_ASSETS,
    _extended_metrics,
    _risk_only_tilt_builder,
)
from .engine import MarketData, load_market_data
from .strategy import CASH_ASSET, load_confirmed_market
from .threshold_rebalance import simulate_threshold_rebalance


ROOT = Path(__file__).parent
OUTPUT = ROOT / "walkforward_experiments"

COST_RATE = 0.0005
MONTHLY_DEPOSIT = 20_000.0
MIN_REBALANCE_NOTIONAL = 10_000.0
RETURN_FLOOR = 0.05

LAMBDA_GRID = (0.25, 0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 0.90, 1.00, 1.25, 1.50, 2.00)
SELECTION_YEARS = tuple(range(2017, 2027))
DEFAULT_LAMBDA = 0.50
FULL_SAMPLE_LAMBDA = 0.70


def _slice_market(data: MarketData, end: pd.Timestamp | None = None, start: pd.Timestamp | None = None) -> MarketData:
    """Restrict a MarketData to [start, end] without touching the source."""
    def cut(series: pd.Series) -> pd.Series:
        out = series
        if start is not None:
            out = out.loc[out.index >= start]
        if end is not None:
            out = out.loc[out.index <= end]
        return out

    opens = {asset: cut(series) for asset, series in data.opens.items()}
    closes = {asset: cut(series) for asset, series in data.closes.items()}
    dates = [
        timestamp
        for timestamp in data.dates
        if (start is None or timestamp >= start) and (end is None or timestamp <= end)
    ]
    return MarketData(opens=opens, closes=closes, dates=dates)


def _run_combo_a(data: MarketData, strength: float):
    targets = _risk_only_tilt_builder(BASELINE_TARGET, strength)(data)
    return simulate_threshold_rebalance(
        data,
        targets,
        CALENDAR_TRIGGER,
        initial_target=dict(BASELINE_TARGET),
        cash_asset=CASH_ASSET,
        monthly_deposit=MONTHLY_DEPOSIT,
        cost_rate=COST_RATE,
        min_rebalance_notional=MIN_REBALANCE_NOTIONAL,
    )


def _select_lambda(
    data: MarketData,
    year: int,
    window: str,
) -> tuple[float, list[dict[str, float]]]:
    """Pick lambda on data strictly before `year` using the real engine."""
    end = pd.Timestamp(f"{year - 1}-12-31")
    if window == "expanding":
        insample = _slice_market(data, end=end)
    elif window == "trailing_36m":
        insample = _slice_market(data, start=pd.Timestamp(f"{year - 3}-01-01"), end=end)
    else:
        raise ValueError(f"unknown window: {window}")

    rows: list[dict[str, float]] = []
    for strength in LAMBDA_GRID:
        result = _run_combo_a(insample, strength)
        metrics = _extended_metrics(result)
        rows.append({
            "year": year,
            "window": window,
            "lambda": strength,
            "insample_annualized_return": metrics["annualized_return"],
            "insample_sharpe": metrics["sharpe"],
            "insample_max_drawdown": metrics["max_drawdown"],
        })

    frame = pd.DataFrame(rows)
    eligible = frame.loc[frame["insample_annualized_return"] >= RETURN_FLOOR]
    pool = eligible if not eligible.empty else frame
    chosen = float(pool.sort_values("insample_sharpe", ascending=False).iloc[0]["lambda"])
    return chosen, rows


def _stitched_schedule(
    data: MarketData,
    lambda_by_year: dict[int, float],
) -> dict[pd.Timestamp, dict[str, float]]:
    """Build daily targets whose tilt strength switches at year boundaries."""
    schedule: dict[pd.Timestamp, dict[str, float]] = {}
    years = sorted({timestamp.year for timestamp in data.dates})
    for year in years:
        strength = lambda_by_year.get(year, DEFAULT_LAMBDA)
        builder = _risk_only_tilt_builder(BASELINE_TARGET, strength)
        # The builder uses only data <= t via searchsorted, so computing the
        # targets on the full market for this year's dates leaks nothing.
        yearly = builder(data)
        for timestamp in data.dates:
            if timestamp.year == year:
                schedule[timestamp] = yearly[timestamp]
    return schedule


def _segment_metrics(result, since: pd.Timestamp) -> dict[str, float]:
    returns = result.daily["return"].dropna().astype(float)
    segment = returns.loc[returns.index >= since]
    if segment.empty:
        return {"annualized_return": np.nan, "sharpe": np.nan, "max_drawdown": np.nan}
    curve = (1.0 + segment).cumprod()
    drawdown = curve / curve.cummax() - 1.0
    volatility = float(segment.std(ddof=1))
    return {
        "annualized_return": float(curve.iloc[-1] ** (252.0 / len(segment)) - 1.0),
        "sharpe": float(segment.mean() / volatility * np.sqrt(252.0)) if volatility > 0 else 0.0,
        "max_drawdown": float(drawdown.min()),
        "volatility": float(volatility * np.sqrt(252.0)),
    }


def run() -> None:
    universe, _ = load_confirmed_market()
    all_assets = sorted(set(universe) | set(EXTRA_POOL_ASSETS))
    market = load_market_data(all_assets, date(2013, 1, 1), date.today())

    OUTPUT.mkdir(parents=True, exist_ok=True)

    # 1. yearly lambda selection on both windows
    selection_rows: list[dict[str, float]] = []
    chosen: dict[str, dict[int, float]] = {"expanding": {}, "trailing_36m": {}}
    for year in SELECTION_YEARS:
        for window in ("expanding", "trailing_36m"):
            strength, rows = _select_lambda(market, year, window)
            chosen[window][year] = strength
            selection_rows.extend(rows)
            print(f"[select] {year} {window}: lambda*={strength}")
    selection = pd.DataFrame(selection_rows)
    selection.to_csv(OUTPUT / "lambda_selection_grid.csv", index=False)

    chosen_frame = pd.DataFrame([
        {"year": year, "window": window, "lambda_star": strength}
        for window, by_year in chosen.items()
        for year, strength in by_year.items()
    ])
    chosen_frame.to_csv(OUTPUT / "lambda_selected.csv", index=False)

    # 2. stitched walk-forward backtests on one continuous capital path
    results: dict[str, object] = {}
    for window in ("expanding", "trailing_36m"):
        schedule = _stitched_schedule(market, chosen[window])
        results[f"walkforward_{window}"] = simulate_threshold_rebalance(
            market,
            schedule,
            CALENDAR_TRIGGER,
            initial_target=dict(BASELINE_TARGET),
            cash_asset=CASH_ASSET,
            monthly_deposit=MONTHLY_DEPOSIT,
            cost_rate=COST_RATE,
            min_rebalance_notional=MIN_REBALANCE_NOTIONAL,
        )
        print(f"[done] walkforward_{window}")

    # 3. references: baseline and in-sample-fixed combo A @ 0.70
    from .rebalance_timing import daily_reversal_targets
    from .calendar_ablation import REVERSAL_20, GLOBAL_TILT_050

    baseline_targets = daily_reversal_targets(market, dict(BASELINE_TARGET), REVERSAL_20, GLOBAL_TILT_050)
    results["baseline_calendar"] = simulate_threshold_rebalance(
        market, baseline_targets, CALENDAR_TRIGGER,
        initial_target=dict(BASELINE_TARGET), cash_asset=CASH_ASSET,
        monthly_deposit=MONTHLY_DEPOSIT, cost_rate=COST_RATE,
        min_rebalance_notional=MIN_REBALANCE_NOTIONAL,
    )
    results["comboA_fixed_070"] = _run_combo_a(market, FULL_SAMPLE_LAMBDA)
    print("[done] references")

    # 4. metrics: full period, pure OOS segment (>= 2017), per-year returns
    oos_start = pd.Timestamp(f"{SELECTION_YEARS[0]}-01-01")
    metric_rows: list[dict[str, object]] = []
    for name, result in results.items():
        full = _extended_metrics(result)
        oos = _segment_metrics(result, oos_start)
        metric_rows.append({
            "strategy": name,
            "full_annualized_return": full["annualized_return"],
            "full_sharpe": full["sharpe"],
            "full_max_drawdown": full["max_drawdown"],
            "full_sortino": full["sortino"],
            "full_calmar": full["calmar"],
            "full_trades": full["trades"],
            "full_estimated_cost": full["estimated_transaction_cost"],
            "oos_annualized_return": oos["annualized_return"],
            "oos_sharpe": oos["sharpe"],
            "oos_max_drawdown": oos["max_drawdown"],
            "oos_volatility": oos["volatility"],
        })
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(OUTPUT / "walkforward_metrics.csv", index=False)

    yearly = pd.DataFrame({
        name: result.daily["return"].dropna().astype(float)
        .groupby(result.daily["return"].dropna().index.year)
        .apply(lambda values: (1.0 + values).prod() - 1.0)
        for name, result in results.items()
    })
    yearly.index.name = "year"
    yearly.to_csv(OUTPUT / "yearly_returns.csv")

    (OUTPUT / "SUMMARY.md").write_text(
        _summary(metrics, chosen_frame, yearly), encoding="utf-8"
    )
    print(metrics.to_string(index=False))
    print(yearly.round(4).to_string())


def _summary(metrics: pd.DataFrame, chosen: pd.DataFrame, yearly: pd.DataFrame) -> str:
    def pct(value: float) -> str:
        return f"{value:.2%}"

    metric_cols = [
        "strategy",
        "full_annualized_return", "full_sharpe", "full_max_drawdown",
        "oos_annualized_return", "oos_sharpe", "oos_max_drawdown",
    ]
    display = metrics.loc[:, metric_cols].copy()
    for column in metric_cols[1:]:
        if "sharpe" in column:
            display[column] = display[column].map(lambda value: f"{value:.3f}")
        else:
            display[column] = display[column].map(pct)
    metric_table = display.to_markdown(index=False)

    pivot = chosen.pivot(index="year", columns="window", values="lambda_star")
    lambda_table = pivot.to_markdown()

    yearly_display = yearly.map(pct)
    yearly_table = yearly_display.to_markdown()

    return f"""# 组合 A（风险宇宙 + 排名线性倾斜）Walk-Forward 检验

规则：每个年度边界仅用该年之前的数据，从 λ ∈ {{0.25 … 2.0}} 中按「年化 ≥ 5% 后取 Sharpe 最大」选出当年参数；各年参数拼成单一连续资金路径（持仓不变，仅参数切换）。2017 年前沿用基线 λ=0.50。样本外段为 2017-01 起。

## 各年选出的 λ\\*

{lambda_table}

## 全期与样本外段指标

{metric_table}

## 分年度收益

{yearly_table}

注：全样本回测，参数选择规则本身仍含研究自由度；结果不构成未来收益保证。
"""


if __name__ == "__main__":
    run()
