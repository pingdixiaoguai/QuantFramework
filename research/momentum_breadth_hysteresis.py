"""Breadth-triggered hysteresis between production momentum and Defender.

Entry after a close requires every original momentum ETF to have a T-day
return strictly below X.  Once Defender is active, exit after a close requires
at least one of CSI 300, gold, or Nasdaq to have a T-day return strictly above
Y.  The decision is executed at the next trading day's open and there is no
minimum holding period beyond market causality.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from research.momentum_defender_switching import (
    DEFENDER_DEFENSIVE,
    DEFENDER_PRIMARY,
    MOMENTUM_ASSETS,
    ONE_WAY_COST_RATES,
    ResearchContext,
    _annual_metrics,
    _baseline_simulations,
    _paired_block_bootstrap,
    _report_result,
    _rolling_metrics,
    _validate_reproduction,
    build_context,
    performance,
)
from research.momentum_safe_haven_selector import simulate_targets


RECOVERY_ASSETS = ("510300.SH", "518880.SH", "513100.SH")


@dataclass(frozen=True)
class HysteresisParams:
    lookback: int = 40
    entry_threshold: float = -0.05
    exit_threshold: float = 0.05

    def __post_init__(self) -> None:
        if self.lookback < 1:
            raise ValueError("lookback must be positive")

    def label(self) -> str:
        def token(value: float) -> str:
            return (
                f"{value:+.3f}"
                .replace("+", "p")
                .replace("-", "m")
                .replace(".", "p")
            )

        return (
            f"t{self.lookback}_x{token(self.entry_threshold)}"
            f"_y{token(self.exit_threshold)}"
        )


def trailing_returns(context: ResearchContext, lookback: int) -> pd.DataFrame:
    """Close-to-close T-day returns on the common execution calendar."""
    closes = pd.DataFrame(
        {
            asset: context.prices[asset]["close"].reindex(context.calendar).ffill()
            for asset in MOMENTUM_ASSETS
        },
        index=context.calendar,
    )
    if closes.isna().any().any():
        raise AssertionError("momentum ETF close history lacks an initial value")
    return closes / closes.shift(lookback) - 1.0


def state_schedule(
    returns: pd.DataFrame,
    params: HysteresisParams,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return open states plus the two close-condition series.

    The loop records today's open state first and only then applies today's
    close condition to tomorrow, preventing same-day lookahead.
    """
    missing = set(MOMENTUM_ASSETS) - set(returns.columns)
    if missing:
        raise ValueError(f"missing momentum return columns: {sorted(missing)}")
    complete = returns[list(MOMENTUM_ASSETS)].notna().all(axis=1)
    enter_after_close = (
        returns[list(MOMENTUM_ASSETS)].lt(params.entry_threshold).all(axis=1)
        & complete
    ).rename("enter_defender_after_close")
    exit_after_close = (
        returns[list(RECOVERY_ASSETS)].gt(params.exit_threshold).any(axis=1)
        & complete
    ).rename("exit_defender_after_close")

    risk_on = True
    values: list[bool] = []
    for timestamp in returns.index:
        values.append(risk_on)
        if risk_on and bool(enter_after_close.at[timestamp]):
            risk_on = False
        elif not risk_on and bool(exit_after_close.at[timestamp]):
            risk_on = True
    return (
        pd.Series(values, index=returns.index, name="risk_on"),
        enter_after_close,
        exit_after_close,
    )


def evaluate(
    context: ResearchContext,
    baselines: Mapping[str, pd.DataFrame],
    params: HysteresisParams,
    cost_multiplier: float = 1.0,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    returns = trailing_returns(context, params.lookback)
    risk_on, enter_close, exit_close = state_schedule(returns, params)
    targets, sleeve = allocation_schedule(context, risk_on)
    daily, trades = simulate_targets(
        context,
        targets,
        sleeve,
        risk_on,
        cost_multiplier=cost_multiplier,
    )
    daily["enter_defender_condition_asof_previous_close"] = enter_close.shift(1)
    daily["exit_defender_condition_asof_previous_close"] = exit_close.shift(1)
    for asset in MOMENTUM_ASSETS:
        daily[f"return_{params.lookback}d_{asset}_asof_previous_close"] = returns[
            asset
        ].shift(1)

    metrics: dict[str, object] = {
        "label": params.label(),
        "lookback": params.lookback,
        "entry_threshold": params.entry_threshold,
        "exit_threshold": params.exit_threshold,
        "cost_multiplier": cost_multiplier,
        "sleeve_switches": int(daily["sleeve_switch"].sum()),
        "defender_day_share": float((~daily["risk_on"]).mean()),
        "total_transaction_cost": float(daily["transaction_cost"].sum()),
        **performance(daily["return"]),
    }
    for period, start, end in (
        ("early", "2019-01-18", "2021-12-31"),
        ("middle", "2022-01-01", "2024-12-31"),
        ("later", "2025-01-01", "2026-08-17"),
        ("development", "2019-01-18", "2024-12-31"),
    ):
        candidate = performance(daily.loc[start:end, "return"])
        momentum = performance(baselines["momentum"].loc[start:end, "return"])
        for key in ("cagr_calendar", "sharpe", "max_drawdown"):
            metrics[f"{period}_{key}"] = candidate[key]
            metrics[f"{period}_delta_{key}"] = float(candidate[key]) - float(
                momentum[key]
            )
    return metrics, daily, trades


def allocation_schedule(
    context: ResearchContext,
    risk_on: pd.Series,
) -> tuple[pd.DataFrame, pd.Series]:
    targets = pd.DataFrame(
        0.0, index=context.calendar, columns=sorted(ONE_WAY_COST_RATES)
    )
    sleeve = pd.Series(index=context.calendar, dtype="object", name="sleeve")
    for timestamp in context.calendar:
        if bool(risk_on.at[timestamp]):
            targets.loc[timestamp, list(MOMENTUM_ASSETS)] = context.momentum_targets.loc[
                timestamp, list(MOMENTUM_ASSETS)
            ]
            sleeve.at[timestamp] = "momentum"
        else:
            targets.loc[timestamp, [DEFENDER_PRIMARY, DEFENDER_DEFENSIVE]] = (
                context.defender_targets.loc[
                    timestamp, [DEFENDER_PRIMARY, DEFENDER_DEFENSIVE]
                ]
            )
            sleeve.at[timestamp] = "defender"
    if not np.allclose(targets.sum(axis=1), 1.0, atol=1e-12):
        raise AssertionError("allocation schedule is not fully invested")
    return targets, sleeve


def build_default_context(root: Path) -> tuple[ResearchContext, dict[str, pd.DataFrame]]:
    defender_schedule = (
        root
        / "experiments/20260818_momentum_defender_switching/defender_daily_targets.csv"
    )
    context = build_context(root, defender_schedule)
    return context, _baseline_simulations(context)


def coarse_grid(
    context: ResearchContext,
    baselines: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for lookback in range(20, 81, 10):
        for entry in (-0.025, 0.0, 0.025, 0.05, 0.075, 0.10, 0.125):
            for exit_ in (-0.025, 0.0, 0.025, 0.05, 0.075, 0.10, 0.125, 0.15, 0.175, 0.20):
                metrics, _, _ = evaluate(
                    context,
                    baselines,
                    HysteresisParams(lookback, entry, exit_),
                )
                rows.append(metrics)
    return pd.DataFrame(rows)


def local_grid(
    context: ResearchContext,
    baselines: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for lookback in range(35, 66, 5):
        for entry in (-0.05, -0.025, 0.0, 0.025):
            for exit_ in (0.125, 0.15, 0.175, 0.20, 0.225):
                metrics, _, _ = evaluate(
                    context,
                    baselines,
                    HysteresisParams(lookback, entry, exit_),
                )
                rows.append(metrics)
    return pd.DataFrame(rows)


def _episodes(daily: pd.DataFrame, momentum: pd.DataFrame) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "candidate": daily["return"],
            "momentum": momentum["return"],
            "defender": ~daily["risk_on"],
        }
    )
    groups = frame["defender"].ne(frame["defender"].shift()).cumsum()
    rows: list[dict[str, object]] = []
    for number, (_, episode) in enumerate(
        frame.loc[frame["defender"]].groupby(groups), start=1
    ):
        rows.append(
            {
                "episode": number,
                "start": episode.index[0].date().isoformat(),
                "end": episode.index[-1].date().isoformat(),
                "days": len(episode),
                "completed": bool(episode.index[-1] != daily.index[-1]),
                "candidate_return": float((1.0 + episode["candidate"]).prod() - 1.0),
                "momentum_return": float((1.0 + episode["momentum"]).prod() - 1.0),
                "excess_log_return": float(
                    np.log1p(episode["candidate"]).sum()
                    - np.log1p(episode["momentum"]).sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def _parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=root)
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "experiments/20260818_momentum_breadth_hysteresis",
    )
    parser.add_argument("--skip-scans", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    context, baselines = build_default_context(args.root)
    reproduction = _validate_reproduction(context, baselines)

    coarse_path = args.output / "coarse_parameter_grid.csv"
    local_path = args.output / "local_parameter_grid.csv"
    if args.skip_scans:
        if not coarse_path.exists() or not local_path.exists():
            raise FileNotFoundError("cannot skip parameter scans before CSVs exist")
        coarse = pd.read_csv(coarse_path)
        local = pd.read_csv(local_path)
    else:
        coarse = coarse_grid(context, baselines)
        coarse.to_csv(coarse_path, index=False)
        local = local_grid(context, baselines)
        local.to_csv(local_path, index=False)

    # This is the only simple path in the declared coarse family that strictly
    # improves both objectives in development and the later period.  The lower
    # of the two equivalent recovery hurdles (17.5% and 20%) is frozen to avoid
    # unnecessarily extending future Defender episodes.
    params = HysteresisParams(50, -0.025, 0.175)
    selected, daily, trades = evaluate(context, baselines, params)
    pd.DataFrame([selected]).to_csv(
        args.output / "selected_candidate_metrics.csv", index=False
    )
    daily.to_csv(args.output / "strategy_daily.csv")
    trades.to_csv(args.output / "trades.csv", index=False)
    events = daily.loc[daily["sleeve_switch"]].copy()
    events["previous_sleeve"] = daily["sleeve"].shift(1).reindex(events.index)
    events.to_csv(args.output / "switch_events.csv")
    episodes = _episodes(daily, baselines["momentum"])
    episodes.to_csv(args.output / "defender_episodes.csv", index=False)

    annual: list[dict[str, object]] = []
    for name, frame in (
        ("momentum", baselines["momentum"]),
        ("defender", baselines["defender"]),
        ("breadth_hysteresis", daily),
    ):
        annual.extend(_annual_metrics(name, frame))
    pd.DataFrame(annual).to_csv(args.output / "annual_metrics.csv", index=False)

    rolling = _rolling_metrics(daily["return"], baselines["momentum"]["return"])
    rolling.to_csv(args.output / "rolling_36m_metrics.csv", index=False)
    bootstrap = _paired_block_bootstrap(
        daily["return"], baselines["momentum"]["return"]
    )
    bootstrap.to_csv(args.output / "paired_block_bootstrap.csv", index=False)
    cost_rows: list[dict[str, object]] = []
    for multiplier in (1.0, 2.0, 5.0, 10.0):
        metrics, _, _ = evaluate(
            context, baselines, params, cost_multiplier=multiplier
        )
        cost_rows.append(metrics)
    cost_stress = pd.DataFrame(cost_rows)
    cost_stress.to_csv(args.output / "cost_stress.csv", index=False)

    momentum_metrics = performance(baselines["momentum"]["return"])
    candidate_metrics = performance(daily["return"])
    pd.DataFrame(
        [
            {"series": "original_momentum", **momentum_metrics},
            {"series": "breadth_hysteresis", **candidate_metrics},
            {"series": "always_defender", **performance(baselines["defender"]["return"])},
        ]
    ).to_csv(args.output / "performance_summary.csv", index=False)

    objective_columns = [
        "development_delta_cagr_calendar",
        "development_delta_sharpe",
        "later_delta_cagr_calendar",
        "later_delta_sharpe",
    ]
    full_pass = (coarse["cagr_calendar"] > float(momentum_metrics["cagr_calendar"])) & (
        coarse["sharpe"] > float(momentum_metrics["sharpe"])
    )
    strict_segment_pass = (coarse[objective_columns] > 1e-9).all(axis=1)
    local_full_pass = (
        local["cagr_calendar"] > float(momentum_metrics["cagr_calendar"])
    ) & (local["sharpe"] > float(momentum_metrics["sharpe"]))
    local_segment_pass = (local[objective_columns] > 1e-9).all(axis=1)
    rolling_both = (rolling["delta_annualized_return_252"] > 0) & (
        rolling["delta_sharpe"] > 0
    )
    bootstrap_both = (bootstrap["delta_annualized_return_252"] > 0) & (
        bootstrap["delta_sharpe"] > 0
    )
    robustness = pd.DataFrame(
        [
            {"check": "coarse_full_both", "passed": int(full_pass.sum()), "total": len(coarse), "rate": float(full_pass.mean())},
            {"check": "coarse_development_and_later_both", "passed": int(strict_segment_pass.sum()), "total": len(coarse), "rate": float(strict_segment_pass.mean())},
            {"check": "local_full_both", "passed": int(local_full_pass.sum()), "total": len(local), "rate": float(local_full_pass.mean())},
            {"check": "local_development_and_later_both", "passed": int(local_segment_pass.sum()), "total": len(local), "rate": float(local_segment_pass.mean())},
            {"check": "rolling_36m_both", "passed": int(rolling_both.sum()), "total": len(rolling), "rate": float(rolling_both.mean())},
            {"check": "bootstrap_both", "passed": int(bootstrap_both.sum()), "total": len(bootstrap), "rate": float(bootstrap_both.mean())},
        ]
    )
    robustness.to_csv(args.output / "robustness_summary.csv", index=False)

    target_columns = [f"target_{asset}" for asset in sorted(ONE_WAY_COST_RATES)]
    audit = pd.DataFrame(
        [
            {
                "target_sum_max_abs_error": float(
                    (daily[target_columns].sum(axis=1) - 1.0).abs().max()
                ),
                "signal_lag_trading_days": 1,
                "minimum_holding_days": 0,
                "sleeve_switches": int(daily["sleeve_switch"].sum()),
                "defender_episode_count": len(episodes),
                "completed_defender_episode_count": int(episodes["completed"].sum()),
                **reproduction,
            }
        ]
    )
    audit.to_csv(args.output / "execution_audit.csv", index=False)
    pd.DataFrame([reproduction]).to_csv(
        args.output / "reproduction_checks.csv", index=False
    )

    _report_result(
        daily["return"],
        baselines["momentum"]["return"],
        "Original Momentum Strategy",
        args.output / "breadth_hysteresis_vs_momentum.html",
        {"strategy_name": "momentum_breadth_hysteresis", **params.__dict__},
    )
    original_base_curve = (1.0 + context.momentum_result.benchmark_returns).cumprod()
    original_base = original_base_curve.reindex(context.calendar).pct_change()
    _report_result(
        daily["return"],
        original_base,
        "Original 4ETF Equal-Weight Base",
        args.output / "breadth_hysteresis_vs_original_base.html",
        {"strategy_name": "momentum_breadth_hysteresis", **params.__dict__},
    )

    current_returns = trailing_returns(context, params.lookback).iloc[-1]
    current_risk_on = bool(daily["risk_on"].iloc[-1])
    current_exit = bool(
        current_returns[list(RECOVERY_ASSETS)].gt(params.exit_threshold).any()
    )
    report = f"""# 四 ETF 广度—防守滞回策略研究

## 规则

- 动量状态：若原四只 ETF 的 {params.lookback} 日收益率全部严格低于 {params.entry_threshold:.2%}，下一交易日开盘切换到 Defender。
- 防守状态：一直持有 Defender，直到沪深300、黄金、纳指中至少一只的 {params.lookback} 日收益率严格高于 {params.exit_threshold:.2%}，下一交易日开盘回原四 ETF 动量。
- 没有额外最短持有期；所有内部调仓、顶层卖出和买入均按原费率计成本。

## 数值结果

- 全样本：2019-01-18—2026-08-17，共 {len(daily):,} 个交易日。
- 开发段：2019-01-18—2024-12-31；后段复核：2025-01-01—2026-08-17。

| 策略 | 自然年 CAGR | Sharpe | 年化波动 | 最大回撤 | 总收益 |
|---|---:|---:|---:|---:|---:|
| 原四 ETF 动量 | {float(momentum_metrics['cagr_calendar']):.2%} | {float(momentum_metrics['sharpe']):.3f} | {float(momentum_metrics['annualized_volatility']):.2%} | {float(momentum_metrics['max_drawdown']):.2%} | {float(momentum_metrics['total_return']):.2%} |
| 广度滞回 | {float(candidate_metrics['cagr_calendar']):.2%} | {float(candidate_metrics['sharpe']):.3f} | {float(candidate_metrics['annualized_volatility']):.2%} | {float(candidate_metrics['max_drawdown']):.2%} | {float(candidate_metrics['total_return']):.2%} |

全样本相对原动量：CAGR {float(candidate_metrics['cagr_calendar']) - float(momentum_metrics['cagr_calendar']):+.2%}，Sharpe {float(candidate_metrics['sharpe']) - float(momentum_metrics['sharpe']):+.3f}。开发段为 CAGR {float(selected['development_delta_cagr_calendar']):+.2%}、Sharpe {float(selected['development_delta_sharpe']):+.3f}；后段为 CAGR {float(selected['later_delta_cagr_calendar']):+.2%}、Sharpe {float(selected['later_delta_sharpe']):+.3f}。

## 过拟合审计与结论

- 490 点粗网格中，仅 {int(full_pass.sum())} 点（{float(full_pass.mean()):.2%}）在全样本同时超过 CAGR 和 Sharpe；开发段与后段均严格双目标为正的只有 {int(strict_segment_pass.sum())} 点（{float(strict_segment_pass.mean()):.2%}）。
- 140 点局部邻域中，全样本双目标通过 {int(local_full_pass.sum())} 点（{float(local_full_pass.mean()):.2%}）；开发段与后段均通过只有 {int(local_segment_pass.sum())} 点（{float(local_segment_pass.mean()):.2%}）。
- 历史仅有 {len(episodes)} 个 Defender 区间，其中完整结束的只有 {int(episodes['completed'].sum())} 个。主要改善来自单个 2022—2023 防守周期；2026 年新周期尚未结束。
- 2019—2021、2024—2025 的逐日路径与原策略完全相同；换言之，七年多历史并没有提供七个独立年份的支持证据。
- 36 个月滚动窗口双目标胜率 {float(rolling_both.mean()):.2%}；20 日成组 bootstrap 双目标为正 {float(bootstrap_both.mean()):.2%}。两者都不能弥补有效事件样本只有一个的问题。
- 2 倍基准交易成本时 CAGR {float(cost_stress.loc[cost_stress['cost_multiplier'] == 2.0, 'cagr_calendar'].iloc[0]):.2%}、Sharpe {float(cost_stress.loc[cost_stress['cost_multiplier'] == 2.0, 'sharpe'].iloc[0]):.3f}，仍高于原策略；5 倍成本时两项目标均不再达标。
- 因此该参数虽然数值达标，但不能称为“不过拟合的适宜取值”，不建议替代原策略。更可信的验证需要冻结规则后等待新的完整广泛下跌—恢复周期。

截至 2026-08-17 收盘，四只 ETF 的 50 日收益率分别为：沪深300 {float(current_returns['510300.SH']):.2%}、创业板 {float(current_returns['159915.SZ']):.2%}、纳指 {float(current_returns['513100.SH']):.2%}、黄金 {float(current_returns['518880.SH']):.2%}。当前开盘状态为 {'动量' if current_risk_on else 'Defender'}；最新收盘退出条件{'已满足' if current_exit else '未满足'}。
"""
    (args.output / "research_report.md").write_text(report, encoding="utf-8")

    print(pd.DataFrame([{"series": "momentum", **momentum_metrics}, {"series": "breadth_hysteresis", **candidate_metrics}]).to_string(index=False))
    print("selected", selected)
    print(robustness.to_string(index=False))
    print("episodes")
    print(episodes.to_string(index=False))
    print("output", args.output)


if __name__ == "__main__":
    main()
