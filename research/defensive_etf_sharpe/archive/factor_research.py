"""Run and summarize factor-allocation experiments without HTML reports."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .engine import simulate_static_allocation
from .factor_allocation import FACTOR_SPECS, MECHANISM_SPECS, simulate_factor_allocation
from .strategy import CASH_ASSET, STATIC_BENCHMARK_TARGET, load_confirmed_market, metrics_for_daily


ROOT = Path(__file__).parent
OUTPUT = ROOT / "factor_experiments"
COST_RATE = 0.0005
RETURN_FLOOR = 0.05


def _extended_metrics(result) -> dict[str, float | int | bool]:
    returns = result.daily["return"].dropna().astype(float)
    basic = metrics_for_daily(result.daily)
    downside = returns.loc[returns < 0].std(ddof=1)
    sortino = float(returns.mean() / downside * np.sqrt(252.0)) if downside > 0 else np.nan
    calmar = (
        basic["annualized_return"] / abs(basic["max_drawdown"])
        if basic["max_drawdown"] < 0
        else np.nan
    )
    annual = returns.groupby(returns.index.year).apply(lambda values: (1.0 + values).prod() - 1.0)
    notional = float(result.trades["notional"].sum()) if not result.trades.empty else 0.0
    return {
        **basic,
        "sortino": sortino,
        "calmar": calmar,
        "worst_calendar_return": float(annual.min()),
        "best_calendar_return": float(annual.max()),
        "final_nav": result.final_nav,
        "total_deposits": result.total_deposits,
        "trades": len(result.trades),
        "traded_notional": notional,
        "estimated_transaction_cost": notional * COST_RATE,
        "meets_5pct_return_floor": basic["annualized_return"] >= RETURN_FLOOR,
    }


def _manifest() -> pd.DataFrame:
    rows = [{
        "experiment_type": "factor",
        "name": factor.name,
        "kind": factor.kind,
        "parameter": factor.window,
        "description": factor.description,
    } for factor in FACTOR_SPECS]
    rows.extend({
        "experiment_type": "mechanism",
        "name": mechanism.name,
        "kind": mechanism.kind,
        "parameter": mechanism.strength,
        "description": mechanism.description,
    } for mechanism in MECHANISM_SPECS)
    return pd.DataFrame(rows)


def _summary(metrics: pd.DataFrame) -> str:
    eligible = metrics.loc[metrics["meets_5pct_return_floor"]].sort_values(
        ["sharpe", "annualized_return"], ascending=False
    )
    all_ranked = metrics.sort_values(["sharpe", "annualized_return"], ascending=False)

    def table(frame: pd.DataFrame, count: int = 12) -> str:
        cols = ["experiment", "annualized_return", "volatility", "sharpe", "max_drawdown", "sortino", "trades"]
        display = frame.loc[:, cols].head(count).copy()
        for col in ("annualized_return", "volatility", "max_drawdown"):
            display[col] = display[col].map(lambda value: f"{value:.2%}")
        for col in ("sharpe", "sortino"):
            display[col] = display[col].map(lambda value: f"{value:.2f}")
        return display.to_markdown(index=False)

    return f"""# 防守型基线因子倾斜：探索性回测汇总

本轮共比较固定基线与 {len(metrics) - 1} 个因子倾斜组合。所有版本均使用相同的 2013 年起始、每月 20,000 元入金、月初收盘形成目标、次日开盘执行、单边 0.05% 成本和 100 股整数手口径。因子只使用信号日及此前的 HFQ 收盘价。

这是全样本筛选结果，不是样本外结论；下一步应只对少量候选做 walk-forward 验证。

## 年化收益达到 5% 后按 Sharpe 排名

{table(eligible) if not eligible.empty else '没有实验达到 5% 年化收益门槛。'}

## 不设收益门槛的 Sharpe 排名

{table(all_ranked)}
"""


def run() -> pd.DataFrame:
    _, market = load_confirmed_market()
    OUTPUT.mkdir(parents=True, exist_ok=True)

    baseline = simulate_static_allocation(
        market, STATIC_BENCHMARK_TARGET, cash_asset=CASH_ASSET
    )
    baseline_metrics = _extended_metrics(baseline)
    rows = [{
        "experiment": "baseline_static_35_40_15_10",
        "factor": "none",
        "mechanism": "fixed",
        **baseline_metrics,
    }]
    schedules: dict[str, dict[pd.Timestamp, dict[str, float]]] = {}

    for factor in FACTOR_SPECS:
        for mechanism in MECHANISM_SPECS:
            experiment = f"{factor.name}__{mechanism.name}"
            result, schedule = simulate_factor_allocation(market, factor, mechanism)
            rows.append({
                "experiment": experiment,
                "factor": factor.name,
                "mechanism": mechanism.name,
                **_extended_metrics(result),
            })
            schedules[experiment] = schedule

    metrics = pd.DataFrame(rows)
    for column in ("annualized_return", "volatility", "sharpe", "max_drawdown", "sortino", "calmar"):
        metrics[f"delta_{column}_vs_baseline"] = metrics[column] - float(baseline_metrics[column])
    metrics = metrics.sort_values(
        ["meets_5pct_return_floor", "sharpe", "annualized_return"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    metrics.insert(0, "rank", np.arange(1, len(metrics) + 1))
    metrics.to_csv(OUTPUT / "factor_experiment_metrics.csv", index=False)
    _manifest().to_csv(OUTPUT / "factor_experiment_manifest.csv", index=False)
    (OUTPUT / "SUMMARY.md").write_text(_summary(metrics), encoding="utf-8")

    candidates = metrics.loc[metrics["factor"] != "none"]
    best_by_factor = candidates.loc[candidates.groupby("factor")["sharpe"].idxmax()].sort_values(
        "sharpe", ascending=False
    )
    best_by_factor.to_csv(OUTPUT / "best_by_factor.csv", index=False)
    best_by_mechanism = candidates.loc[
        candidates.groupby("mechanism")["sharpe"].idxmax()
    ].sort_values("sharpe", ascending=False)
    best_by_mechanism.to_csv(OUTPUT / "best_by_mechanism.csv", index=False)

    best = metrics.loc[metrics["experiment"] != "baseline_static_35_40_15_10"].iloc[0]["experiment"]
    weight_rows = []
    for timestamp, weights in schedules[str(best)].items():
        weight_rows.extend(
            {"date": timestamp.date().isoformat(), "experiment": best, "asset": asset, "target_weight": weight}
            for asset, weight in weights.items()
        )
    pd.DataFrame(weight_rows).to_csv(OUTPUT / "best_candidate_monthly_weights.csv", index=False)
    return metrics


if __name__ == "__main__":
    result = run()
    print(result.head(15).to_string(index=False))
