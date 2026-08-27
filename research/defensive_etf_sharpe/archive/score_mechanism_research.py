"""Score-to-weight mapping alternatives for the anchored drift strategy.

Two questions, one experiment matrix:

1. Do fixed rank steps {±1, ±1/3} discard useful magnitude information?
   Alternatives: exponential (softmax-family) tilt, sigmoid (tanh) tilt, and
   additive tilt, all driven by cross-sectional z-scores instead of ranks.
2. Does raw-return scoring let the high-volatility asset dominate the ranks?
   Alternative: vol-adjusted reversal (20d return divided by the window's
   expected volatility, a t-stat-like score).

All variants use the champion trigger: month-start anchored target, 10%
single-sided drift, at most one rebalance per month, 10,000 minimum notional.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .factor_allocation import (
    FactorSpec,
    MechanismSpec,
    _raw_factor,
    _history,
    adjusted_weights,
    factor_ranks,
    factor_zscores,
)
from .factor_research import _extended_metrics
from .rebalance_timing import (
    GLOBAL_TILT_050,
    REVERSAL_20,
    daily_reversal_targets,
    month_start_anchored_targets,
)
from .reduced_pool_research import REDUCED_TARGET
from .strategy import CASH_ASSET, load_confirmed_market
from .threshold_rebalance import TRIGGER_SPECS, simulate_threshold_rebalance

ROOT = Path(__file__).parent
OUTPUT = ROOT / "score_mechanism_experiments"

REVERSAL_VOLADJ_20 = FactorSpec("reversal_voladj_20", "reversal_vol_adjusted", 20, "20日反转÷窗口波动")
EXP_TILT_050 = MechanismSpec("exp_tilt_050", "exp_tilt", 0.50, "基准权重×exp(0.5·z)后归一化")
EXP_TILT_100 = MechanismSpec("exp_tilt_100", "exp_tilt", 1.00, "基准权重×exp(1.0·z)后归一化")
SIGMOID_TILT_100 = MechanismSpec("sigmoid_tilt_100", "sigmoid_tilt", 1.00, "基准权重×(1+tanh(z))后归一化")
ADDITIVE_TILT_005 = MechanismSpec("additive_tilt_005", "additive_tilt", 0.05, "基准权重+0.05·z截断后归一化")


def _daily_targets(
    market,
    factor: FactorSpec,
    mechanism: MechanismSpec,
    use_zscores: bool,
) -> dict[pd.Timestamp, dict[str, float]]:
    assets = tuple(REDUCED_TARGET)
    score_fn = factor_zscores if use_zscores else factor_ranks
    return {
        timestamp: adjusted_weights(
            score_fn(market, timestamp, factor, assets),
            mechanism,
            dict(REDUCED_TARGET),
        )
        for timestamp in market.dates
    }


def _diagnostics(market) -> dict[str, float]:
    assets = tuple(REDUCED_TARGET)
    dates = pd.DatetimeIndex(market.dates)
    dates = dates[dates >= "2019-01-01"]
    extreme_raw = 0
    extreme_voladj = 0
    gaps: list[float] = []
    counted = 0
    for timestamp in dates:
        raw = {
            asset: _raw_factor(_history(market, asset, timestamp), "reversal", 20)
            for asset in assets
        }
        if not all(np.isfinite(value) for value in raw.values()):
            continue
        counted += 1
        ranks_raw = factor_ranks(market, timestamp, REVERSAL_20, assets)
        ranks_voladj = factor_ranks(market, timestamp, REVERSAL_VOLADJ_20, assets)
        if abs(ranks_raw["512890.SH"]) == 1.0:
            extreme_raw += 1
        if abs(ranks_voladj["512890.SH"]) == 1.0:
            extreme_voladj += 1
        ordered = sorted(raw.values())
        gaps.extend(b - a for a, b in zip(ordered, ordered[1:]))
    return {
        "days": float(counted),
        "share_512890_at_rank_extreme_raw": extreme_raw / counted,
        "share_512890_at_rank_extreme_voladj": extreme_voladj / counted,
        "median_adjacent_rank_factor_gap_pct": float(np.median(gaps) * 100.0),
        "p25_adjacent_rank_factor_gap_pct": float(np.percentile(gaps, 25) * 100.0),
    }


def run() -> pd.DataFrame:
    _, market = load_confirmed_market()
    dates = pd.DatetimeIndex(market.dates)
    specs = {spec.name: spec for spec in TRIGGER_SPECS}
    anchor_spec = specs["portfolio_drift_10_monthly_cap1"]

    standard = daily_reversal_targets(market, REDUCED_TARGET)
    variants = {
        # name: (daily targets, spec, anchor_to_month_start)
        "ref_calendar_rank50": (standard, specs["calendar_monthly_reference"], False),
        "ref_rank50_anchor": (standard, anchor_spec, True),
        "exp05_anchor": (_daily_targets(market, REVERSAL_20, EXP_TILT_050, True), anchor_spec, True),
        "exp10_anchor": (_daily_targets(market, REVERSAL_20, EXP_TILT_100, True), anchor_spec, True),
        "sigmoid10_anchor": (_daily_targets(market, REVERSAL_20, SIGMOID_TILT_100, True), anchor_spec, True),
        "additive05_anchor": (_daily_targets(market, REVERSAL_20, ADDITIVE_TILT_005, True), anchor_spec, True),
        "voladj_rank50_anchor": (
            _daily_targets(market, REVERSAL_VOLADJ_20, GLOBAL_TILT_050, False),
            anchor_spec,
            True,
        ),
        "voladj_exp05_anchor": (
            _daily_targets(market, REVERSAL_VOLADJ_20, EXP_TILT_050, True),
            anchor_spec,
            True,
        ),
    }

    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = []
    results = {}
    for name, (daily_targets, spec, anchor) in variants.items():
        targets = month_start_anchored_targets(daily_targets, dates) if anchor else daily_targets
        result = simulate_threshold_rebalance(
            market,
            targets,
            spec,
            initial_target=REDUCED_TARGET,
            cash_asset=CASH_ASSET,
            min_rebalance_notional=10_000.0,
        )
        results[name] = result
        rebalance_trades = result.trades.loc[result.trades["reason"] == "threshold_rebalance"]
        rows.append({
            "variant": name,
            **_extended_metrics(result),
            "trigger_signals": len(result.signals),
            "rebalance_dates": pd.to_datetime(rebalance_trades["date"]).nunique()
            if not rebalance_trades.empty else 0,
            "average_cash_weight": float(result.daily["cash_weight"].mean()),
        })

    metrics = pd.DataFrame(rows).sort_values(
        ["meets_5pct_return_floor", "sharpe", "annualized_return"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    metrics.insert(0, "rank", np.arange(1, len(metrics) + 1))
    metrics.to_csv(OUTPUT / "score_mechanism_metrics.csv", index=False)

    diagnostics = _diagnostics(market)
    pd.Series(diagnostics).to_csv(OUTPUT / "score_mechanism_diagnostics.csv")

    conditional = metrics.loc[~metrics["variant"].isin({"ref_calendar_rank50", "ref_rank50_anchor"})]
    best_name = str(conditional.iloc[0]["variant"])
    best = results[best_name]
    best.daily.to_csv(OUTPUT / f"{best_name}_daily.csv")
    best.trades.to_csv(OUTPUT / f"{best_name}_trades.csv", index=False)
    best.signals.to_csv(OUTPUT / f"{best_name}_trigger_signals.csv", index=False)

    (OUTPUT / "SUMMARY.md").write_text(_summary(metrics, diagnostics, best_name), encoding="utf-8")
    return metrics, pd.Series(diagnostics)


def _summary(metrics: pd.DataFrame, diagnostics: dict[str, float], best_name: str) -> str:
    display = metrics[[
        "variant", "annualized_return", "volatility", "sharpe", "max_drawdown",
        "sortino", "rebalance_dates", "trades", "estimated_transaction_cost",
    ]].copy()
    for column in ("annualized_return", "volatility", "max_drawdown"):
        display[column] = display[column].map(lambda value: f"{value:.2%}")
    for column in ("sharpe", "sortino"):
        display[column] = display[column].map(lambda value: f"{value:.2f}")
    display["estimated_transaction_cost"] = display["estimated_transaction_cost"].map(
        lambda value: f"{value:,.0f}"
    )
    return f"""# 得分-权重映射机制实验

所有变体共享冠军触发器（月初锚定目标、单边偏离10%、每月最多一次、单笔10,000元门槛；ref_calendar 用月初固定触发作参照）。z 类机制使用横截面 z-score 替代固定排名档位。本轮是全样本探索，不是样本外结论。最佳非参照变体：{best_name}。

{display.to_markdown(index=False)}

## 诊断（2019年起）

- 512890 处于排名首位的天数占比：原始反转 {diagnostics["share_512890_at_rank_extreme_raw"]:.0%}，波动调整反转 {diagnostics["share_512890_at_rank_extreme_voladj"]:.0%}
- 相邻名次的因子值差距（20日收益率）：中位 {diagnostics["median_adjacent_rank_factor_gap_pct"]:.2f}%，p25 {diagnostics["p25_adjacent_rank_factor_gap_pct"]:.2f}%
"""


if __name__ == "__main__":
    all_metrics, all_diagnostics = run()
    print(all_metrics.to_string(index=False))
    print()
    print(all_diagnostics.to_string())
