"""No-lock selector for momentum, Defender, gold, and Nasdaq sleeves.

The first layer uses only 510300.SH closes to decide whether the original
four-ETF momentum strategy remains active.  When that risk filter is off, a
second layer chooses among the frozen Defender strategy, 518880.SH, and
513100.SH.  Every close decision becomes effective at the next trading day's
open; no sleeve has a minimum holding period.

This is a research entrypoint and does not change the production strategy.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from research.momentum_defender_switching import (
    DEFENDER_DEFENSIVE,
    DEFENDER_PRIMARY,
    INITIAL_CAPITAL,
    MOMENTUM_ASSETS,
    ONE_WAY_COST_RATES,
    ResearchContext,
    _annual_metrics,
    _baseline_simulations,
    _paired_block_bootstrap,
    _report_result,
    _rolling_metrics,
    _targets_equal,
    _validate_reproduction,
    build_context,
    performance,
)


GOLD = "518880.SH"
NASDAQ = "513100.SH"
SAFE_HAVENS = ("defender", "gold", "nasdaq")


@dataclass(frozen=True)
class SelectorParams:
    risk_windows: tuple[int, ...] = (30, 40, 50)
    risk_threshold: float = 0.0
    selector_method: str = "quality_momentum"
    selector_window: int = 20

    def label(self) -> str:
        windows = "_".join(str(item) for item in self.risk_windows)
        threshold = (
            f"{self.risk_threshold:+.3f}"
            .replace("+", "p")
            .replace("-", "m")
            .replace(".", "p")
        )
        return f"risk{windows}_th{threshold}_{self.selector_method}{self.selector_window}"


def risk_indicator(close: pd.Series, windows: Sequence[int]) -> pd.Series:
    """Median trailing return; a single window is the ordinary X-day return."""
    if not windows or any(window < 1 for window in windows):
        raise ValueError("risk windows must contain positive integers")
    returns = pd.concat(
        {f"return_{window}": close / close.shift(window) - 1.0 for window in windows},
        axis=1,
    )
    # Require every requested horizon, so a multi-window signal is well-defined.
    return returns.median(axis=1, skipna=False).rename("risk_indicator")


def risk_on_at_open(close: pd.Series, params: SelectorParams) -> pd.Series:
    """A close signal is actionable only at the next open; no holding lock."""
    indicator = risk_indicator(close, params.risk_windows)
    decision_after_close = indicator > params.risk_threshold
    valid_after_close = decision_after_close.where(indicator.notna())
    return valid_after_close.shift(1).fillna(True).astype(bool).rename("risk_on")


def _quality_momentum(values: pd.DataFrame, window: int) -> pd.DataFrame:
    momentum = values / values.shift(window) - 1.0
    path = values.diff().abs().rolling(window).sum()
    efficiency = (values - values.shift(window)).abs().div(path.replace(0.0, np.nan))
    return momentum * efficiency


def safe_haven_scores(
    context: ResearchContext,
    baselines: Mapping[str, pd.DataFrame],
    method: str,
    window: int,
) -> pd.DataFrame:
    """Scores known after each close for Defender NAV, gold, and Nasdaq."""
    if window < 1:
        raise ValueError("selector_window must be positive")
    values = pd.DataFrame(
        {
            "defender": baselines["defender"]["nav"],
            "gold": context.prices[GOLD]["close"].reindex(context.calendar),
            "nasdaq": context.prices[NASDAQ]["close"].reindex(context.calendar),
        },
        index=context.calendar,
    )
    # A suspended/missing quote carries its last close for signal purposes.  It
    # earns zero marked return until a new quote appears and cannot be traded in
    # ``simulate_targets`` because no open exists on that date.
    values = values.ffill()
    if values.isna().any().any():
        raise AssertionError("safe-haven score inputs lack an initial value")
    if method == "quality_momentum":
        return _quality_momentum(values, window)
    if method == "trailing_return":
        return values / values.shift(window) - 1.0
    raise ValueError(f"unsupported selector method: {method}")


def safe_haven_at_open(scores: pd.DataFrame) -> pd.Series:
    """Select yesterday's highest score, with Defender as warm-up fallback."""
    known_at_open = scores.shift(1)
    selected = known_at_open.fillna(-np.inf).idxmax(axis=1)
    no_complete_score = known_at_open.isna().any(axis=1)
    return selected.mask(no_complete_score, "defender").fillna("defender").rename("safe_haven")


def target_schedule(
    context: ResearchContext,
    risk_on: pd.Series,
    selected_safe_haven: pd.Series,
) -> tuple[pd.DataFrame, pd.Series]:
    assets = sorted(ONE_WAY_COST_RATES)
    targets = pd.DataFrame(0.0, index=context.calendar, columns=assets)
    sleeve = pd.Series(index=context.calendar, dtype="object", name="sleeve")
    for timestamp in context.calendar:
        if bool(risk_on.at[timestamp]):
            targets.loc[timestamp, list(MOMENTUM_ASSETS)] = context.momentum_targets.loc[
                timestamp, list(MOMENTUM_ASSETS)
            ]
            sleeve.at[timestamp] = "momentum"
            continue
        choice = str(selected_safe_haven.at[timestamp])
        if choice == "defender":
            targets.loc[timestamp, [DEFENDER_PRIMARY, DEFENDER_DEFENSIVE]] = (
                context.defender_targets.loc[timestamp, [DEFENDER_PRIMARY, DEFENDER_DEFENSIVE]]
            )
        elif choice == "gold":
            targets.at[timestamp, GOLD] = 1.0
        elif choice == "nasdaq":
            targets.at[timestamp, NASDAQ] = 1.0
        else:
            raise ValueError(f"unexpected safe-haven sleeve: {choice}")
        sleeve.at[timestamp] = choice
    if not np.allclose(targets.sum(axis=1), 1.0, atol=1e-12):
        raise AssertionError("target schedule is not fully invested")
    return targets, sleeve


def _execute_target(
    cash: float,
    shares: dict[str, float],
    target: Mapping[str, float],
    open_prices: Mapping[str, float],
    cost_multiplier: float,
) -> tuple[float, dict[str, float], list[dict[str, float | str]]]:
    if cost_multiplier < 0:
        raise ValueError("cost multiplier cannot be negative")
    nav_open = cash + sum(quantity * open_prices[asset] for asset, quantity in shares.items())
    desired = {asset: nav_open * weight for asset, weight in target.items()}
    executions: list[dict[str, float | str]] = []

    def rate(asset: str) -> float:
        return ONE_WAY_COST_RATES[asset] * cost_multiplier

    for asset in sorted(set(shares) | set(target)):
        current_value = shares.get(asset, 0.0) * open_prices[asset]
        sell_value = max(0.0, current_value - desired.get(asset, 0.0))
        if sell_value <= 1e-14:
            continue
        shares[asset] = shares.get(asset, 0.0) - sell_value / open_prices[asset]
        cash += sell_value * (1.0 - rate(asset))
        executions.append(
            {
                "asset": asset,
                "side": "sell",
                "notional": sell_value,
                "turnover": sell_value / nav_open,
                "cost_rate": rate(asset),
                "cost": sell_value * rate(asset),
            }
        )

    needs: dict[str, float] = {}
    for asset, desired_value in desired.items():
        current_value = shares.get(asset, 0.0) * open_prices[asset]
        needs[asset] = max(0.0, desired_value - current_value)
    total_need = sum(value * (1.0 + rate(asset)) for asset, value in needs.items())
    scale = min(1.0, cash / total_need) if total_need else 0.0
    for asset in sorted(needs, key=lambda item: target[item], reverse=True):
        buy_value = needs[asset] * scale
        if buy_value <= 1e-14:
            continue
        shares[asset] = shares.get(asset, 0.0) + buy_value / open_prices[asset]
        cash -= buy_value * (1.0 + rate(asset))
        executions.append(
            {
                "asset": asset,
                "side": "buy",
                "notional": buy_value,
                "turnover": buy_value / nav_open,
                "cost_rate": rate(asset),
                "cost": buy_value * rate(asset),
            }
        )
    return cash, {asset: qty for asset, qty in shares.items() if qty > 1e-14}, executions


def simulate_targets(
    context: ResearchContext,
    targets: pd.DataFrame,
    sleeve: pd.Series,
    risk_on: pd.Series,
    cost_multiplier: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cash = INITIAL_CAPITAL
    shares: dict[str, float] = {}
    previous_target: dict[str, float] = {}
    previous_sleeve: str | None = None
    last_nav = INITIAL_CAPITAL
    rows: list[dict[str, object]] = []
    trades: list[dict[str, object]] = []

    for timestamp in context.calendar:
        current_sleeve = str(sleeve.at[timestamp])
        target = {
            asset: float(weight)
            for asset, weight in targets.loc[timestamp].items()
            if float(weight) > 1e-14
        }
        open_prices = {
            asset: float(frame.at[timestamp, "open"])
            for asset, frame in context.prices.items()
            if timestamp in frame.index
        }
        close_prices = {
            asset: float(frame.at[timestamp, "close"])
            for asset, frame in context.prices.items()
            if timestamp in frame.index
        }
        if set(shares) - set(open_prices) or set(target) - set(open_prices):
            raise AssertionError(f"missing price on {timestamp.date()}")
        nav_open = cash + sum(qty * open_prices[asset] for asset, qty in shares.items())
        sleeve_changed = previous_sleeve is not None and current_sleeve != previous_sleeve
        changed = not _targets_equal(target, previous_target)
        executions: list[dict[str, float | str]] = []
        if changed:
            cash, shares, executions = _execute_target(
                cash, shares, target, open_prices, cost_multiplier
            )
            reason = "sleeve_switch" if sleeve_changed else "internal_target_change"
            for execution in executions:
                trades.append(
                    {
                        "date": timestamp,
                        "reason": reason,
                        "from_sleeve": previous_sleeve,
                        "to_sleeve": current_sleeve,
                        **execution,
                    }
                )
            previous_target = target
        nav = cash + sum(qty * close_prices[asset] for asset, qty in shares.items())
        row: dict[str, object] = {
            "date": timestamp,
            "return": nav / last_nav - 1.0,
            "nav": nav,
            "risk_on": bool(risk_on.at[timestamp]),
            "sleeve": current_sleeve,
            "sleeve_switch": sleeve_changed,
            "target_change": changed,
            "nav_open_before_cost": nav_open,
            "transaction_cost": sum(float(item["cost"]) for item in executions),
            "turnover": sum(float(item["turnover"]) for item in executions),
        }
        for asset in sorted(ONE_WAY_COST_RATES):
            row[f"target_{asset}"] = target.get(asset, 0.0)
            row[f"shares_{asset}"] = shares.get(asset, 0.0)
        rows.append(row)
        last_nav = nav
        previous_sleeve = current_sleeve
    return pd.DataFrame(rows).set_index("date"), pd.DataFrame(trades)


def evaluate(
    context: ResearchContext,
    baselines: Mapping[str, pd.DataFrame],
    params: SelectorParams,
    cost_multiplier: float = 1.0,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    risk_on = risk_on_at_open(context.risk_close, params).reindex(context.calendar)
    scores = safe_haven_scores(
        context, baselines, params.selector_method, params.selector_window
    )
    choice = safe_haven_at_open(scores).reindex(context.calendar)
    targets, sleeve = target_schedule(context, risk_on, choice)
    daily, trades = simulate_targets(
        context, targets, sleeve, risk_on, cost_multiplier=cost_multiplier
    )
    indicator = risk_indicator(context.risk_close, params.risk_windows)
    daily["risk_indicator_asof_previous_close"] = indicator.shift(1).reindex(context.calendar)
    daily["safe_haven_signal_asof_previous_close"] = choice
    for name in SAFE_HAVENS:
        daily[f"score_{name}_asof_previous_close"] = scores[name].shift(1).reindex(
            context.calendar
        )
    metrics: dict[str, object] = {
        "label": params.label(),
        "risk_windows": "/".join(str(item) for item in params.risk_windows),
        "risk_threshold": params.risk_threshold,
        "selector_method": params.selector_method,
        "selector_window": params.selector_window,
        "cost_multiplier": cost_multiplier,
        "sleeve_switches": int(daily["sleeve_switch"].sum()),
        "target_changes": int(daily["target_change"].sum()),
        "risk_off_day_share": float((~daily["risk_on"]).mean()),
        "total_transaction_cost": float(daily["transaction_cost"].sum()),
        **performance(daily["return"]),
    }
    for name in ("momentum", *SAFE_HAVENS):
        metrics[f"day_share_{name}"] = float((daily["sleeve"] == name).mean())
    for period, start, end in (
        ("development", "2019-01-18", "2024-12-31"),
        ("later_period", "2025-01-01", "2026-08-17"),
    ):
        candidate_metrics = performance(daily.loc[start:end, "return"])
        baseline_metrics = performance(baselines["momentum"].loc[start:end, "return"])
        for key in ("cagr_calendar", "sharpe", "max_drawdown"):
            metrics[f"{period}_{key}"] = candidate_metrics[key]
            metrics[f"{period}_delta_{key}"] = float(candidate_metrics[key]) - float(
                baseline_metrics[key]
            )
    return metrics, daily, trades


def candidate_grid(
    context: ResearchContext,
    baselines: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    # Small, pre-declared family: exact 40-day versus a robust 30/40/50 median;
    # natural 0% threshold versus the previously studied 2.5%; production QM20
    # versus an unadjusted 20-day return selector.
    for windows in ((40,), (30, 40, 50)):
        for threshold in (0.0, 0.025):
            for method in ("quality_momentum", "trailing_return"):
                metrics, _, _ = evaluate(
                    context,
                    baselines,
                    SelectorParams(windows, threshold, method, 20),
                )
                rows.append(metrics)
    return pd.DataFrame(rows)


def parameter_neighborhood(
    context: ResearchContext,
    baselines: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for windows in ((20, 30, 40), (30, 40, 50), (40, 50, 60)):
        for threshold in (-0.025, 0.0, 0.025):
            for selector_window in (15, 20, 25, 30):
                metrics, _, _ = evaluate(
                    context,
                    baselines,
                    SelectorParams(windows, threshold, "quality_momentum", selector_window),
                )
                rows.append(metrics)
    return pd.DataFrame(rows)


def single_lookback_neighborhood(
    context: ResearchContext,
    baselines: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    """Broad sensitivity scan used to audit, not select, the frozen 40-day rule."""
    rows: list[dict[str, object]] = []
    for lookback in range(20, 85, 5):
        for threshold in (-0.025, 0.0, 0.025):
            for selector_window in (15, 20, 25, 30):
                metrics, _, _ = evaluate(
                    context,
                    baselines,
                    SelectorParams(
                        (lookback,), threshold, "quality_momentum", selector_window
                    ),
                )
                rows.append(metrics)
    return pd.DataFrame(rows)


def _episode_attribution(daily: pd.DataFrame, momentum: pd.DataFrame) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "candidate": daily["return"],
            "momentum": momentum["return"],
            "risk_off": ~daily["risk_on"],
        }
    ).dropna()
    groups = frame["risk_off"].ne(frame["risk_off"].shift()).cumsum()
    rows: list[dict[str, object]] = []
    for episode_number, (_, episode) in enumerate(
        frame.loc[frame["risk_off"]].groupby(groups), start=1
    ):
        allocation = daily.loc[episode.index, "sleeve"].value_counts()
        candidate_return = float((1.0 + episode["candidate"]).prod() - 1.0)
        momentum_return = float((1.0 + episode["momentum"]).prod() - 1.0)
        rows.append(
            {
                "episode": episode_number,
                "start": episode.index[0].date().isoformat(),
                "end": episode.index[-1].date().isoformat(),
                "days": len(episode),
                "defender_days": int(allocation.get("defender", 0)),
                "gold_days": int(allocation.get("gold", 0)),
                "nasdaq_days": int(allocation.get("nasdaq", 0)),
                "candidate_return": candidate_return,
                "momentum_return": momentum_return,
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
        default=root / "experiments/20260818_momentum_safe_haven_selector",
    )
    parser.add_argument(
        "--defender-schedule",
        type=Path,
        default=root
        / "experiments/20260818_momentum_defender_switching/defender_daily_targets.csv",
    )
    parser.add_argument(
        "--skip-scans",
        action="store_true",
        help="reuse existing candidate and sensitivity CSVs",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    context = build_context(args.root, args.defender_schedule)
    baselines = _baseline_simulations(context)
    reproduction = _validate_reproduction(context, baselines)

    grid_path = args.output / "candidate_grid.csv"
    neighborhood_path = args.output / "parameter_neighborhood.csv"
    lookback_path = args.output / "single_lookback_neighborhood.csv"
    if args.skip_scans:
        missing = [
            path for path in (grid_path, neighborhood_path, lookback_path) if not path.exists()
        ]
        if missing:
            raise FileNotFoundError(f"cannot skip missing scans: {missing}")
        grid = pd.read_csv(grid_path)
        neighborhood = pd.read_csv(neighborhood_path)
        lookback_neighborhood = pd.read_csv(lookback_path)
    else:
        grid = candidate_grid(context, baselines)
        grid.to_csv(grid_path, index=False)
        neighborhood = parameter_neighborhood(context, baselines)
        neighborhood.to_csv(neighborhood_path, index=False)
        lookback_neighborhood = single_lookback_neighborhood(context, baselines)
        lookback_neighborhood.to_csv(lookback_path, index=False)

    # Freeze the original 40-day horizon, natural 0% threshold, and the QM20
    # factor already used by production.  It is not chosen by maximizing either
    # candidate table: the 30/40/50 rule and the 2.5% threshold score higher.
    selected_params = SelectorParams((40,), 0.0, "quality_momentum", 20)
    selected_metrics, daily, trades = evaluate(context, baselines, selected_params)
    pd.DataFrame([selected_metrics]).to_csv(
        args.output / "selected_candidate_metrics.csv", index=False
    )
    daily.to_csv(args.output / "strategy_daily.csv")
    trades.to_csv(args.output / "trades.csv", index=False)

    events = daily.loc[daily["sleeve_switch"]].copy()
    events["previous_sleeve"] = daily["sleeve"].shift(1).reindex(events.index)
    events.to_csv(args.output / "sleeve_switch_events.csv")
    _episode_attribution(daily, baselines["momentum"]).to_csv(
        args.output / "risk_off_episode_attribution.csv", index=False
    )

    annual: list[dict[str, object]] = []
    for name, frame in (
        ("momentum", baselines["momentum"]),
        ("defender", baselines["defender"]),
        ("safe_haven_selector", daily),
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
            context, baselines, selected_params, cost_multiplier=multiplier
        )
        cost_rows.append(metrics)
    pd.DataFrame(cost_rows).to_csv(args.output / "cost_stress.csv", index=False)

    baseline_metrics = performance(baselines["momentum"]["return"])
    summary_rows = [
        {"series": "original_momentum", **baseline_metrics},
        {"series": "always_defender", **performance(baselines["defender"]["return"])},
        {"series": "safe_haven_selector", **performance(daily["return"])},
    ]
    pd.DataFrame(summary_rows).to_csv(args.output / "performance_summary.csv", index=False)
    pd.DataFrame([reproduction]).to_csv(args.output / "reproduction_checks.csv", index=False)

    target_columns = [f"target_{asset}" for asset in sorted(ONE_WAY_COST_RATES)]
    switch_turnover = (
        trades.loc[trades["reason"] == "sleeve_switch"].groupby("date")["turnover"].sum()
    )
    holding_groups = daily["sleeve"].ne(daily["sleeve"].shift()).cumsum()
    holding_lengths = daily.groupby(holding_groups).size()
    audit = {
        "target_sum_max_abs_error": float(
            (daily[target_columns].sum(axis=1) - 1.0).abs().max()
        ),
        "signal_lag_trading_days": 1,
        "minimum_holding_days": 0,
        "sleeve_switch_count": int(daily["sleeve_switch"].sum()),
        "executed_switch_dates": int(len(switch_turnover)),
        "median_executed_switch_turnover": float(switch_turnover.median()),
        "max_executed_switch_turnover": float(switch_turnover.max()),
        "actual_min_sleeve_holding_days": int(holding_lengths.min()),
        "actual_median_sleeve_holding_days": float(holding_lengths.median()),
        "one_day_sleeve_intervals": int((holding_lengths == 1).sum()),
        **reproduction,
    }
    pd.DataFrame([audit]).to_csv(args.output / "execution_audit.csv", index=False)

    _report_result(
        daily["return"],
        baselines["momentum"]["return"],
        "Original Momentum Strategy",
        args.output / "safe_haven_selector_vs_momentum.html",
        {"strategy_name": "momentum_safe_haven_selector", **selected_params.__dict__},
    )
    original_base_curve = (1.0 + context.momentum_result.benchmark_returns).cumprod()
    original_base = original_base_curve.reindex(context.calendar).pct_change()
    _report_result(
        daily["return"],
        original_base,
        "Original 4ETF Equal-Weight Base",
        args.output / "safe_haven_selector_vs_original_base.html",
        {"strategy_name": "momentum_safe_haven_selector", **selected_params.__dict__},
    )

    objective_columns = [
        "development_delta_cagr_calendar",
        "development_delta_sharpe",
        "later_period_delta_cagr_calendar",
        "later_period_delta_sharpe",
    ]
    neighborhood_pass = (neighborhood[objective_columns] > 0).all(axis=1)
    lookback_full_pass = (
        (lookback_neighborhood["cagr_calendar"] > float(baseline_metrics["cagr_calendar"]))
        & (lookback_neighborhood["sharpe"] > float(baseline_metrics["sharpe"]))
    )
    local_lookback = lookback_neighborhood.loc[
        lookback_neighborhood["risk_windows"].astype(int).between(30, 50)
    ]
    local_full_pass = (
        (local_lookback["cagr_calendar"] > float(baseline_metrics["cagr_calendar"]))
        & (local_lookback["sharpe"] > float(baseline_metrics["sharpe"]))
    )
    rolling_both = (rolling["delta_annualized_return_252"] > 0) & (
        rolling["delta_sharpe"] > 0
    )
    bootstrap_both = (bootstrap["delta_annualized_return_252"] > 0) & (
        bootstrap["delta_sharpe"] > 0
    )
    selected_performance = performance(daily["return"])
    selected_annual = pd.DataFrame(annual).pivot(
        index="year", columns="series", values=["cagr_calendar", "sharpe"]
    )
    annual_both_wins = (
        (
            selected_annual["cagr_calendar", "safe_haven_selector"]
            > selected_annual["cagr_calendar", "momentum"]
        )
        & (
            selected_annual["sharpe", "safe_haven_selector"]
            > selected_annual["sharpe", "momentum"]
        )
    )
    episodes = _episode_attribution(daily, baselines["momentum"])
    positive_episode_excess = episodes.loc[
        episodes["excess_log_return"] > 0, "excess_log_return"
    ]
    top_five_positive_share = float(
        positive_episode_excess.nlargest(5).sum() / positive_episode_excess.sum()
    )
    latest_indicator = float(
        risk_indicator(context.risk_close, selected_params.risk_windows).iloc[-1]
    )
    latest_scores = safe_haven_scores(
        context,
        baselines,
        selected_params.selector_method,
        selected_params.selector_window,
    ).iloc[-1]
    latest_choice = str(latest_scores.idxmax())
    report = f"""# 无锁定动量—避险选择策略研究

## 冻结规则

- 正常状态：使用原四 ETF `quality_momentum_top1` 策略，不改变其选股和内部调仓。
- 风险指标：`510300.SH` 的 40 日收盘收益率；指标高于 0% 为正常，否则为风险状态。
- 风险状态：用生产策略已经采用的 QM20（20 日收益率 × 20 日效率系数），在冻结的 Defender 策略净值、黄金 ETF `518880.SH`、纳指 ETF `513100.SH` 三者中选分数最高者。
- 所有信号在收盘后计算，下一交易日开盘执行；没有最短持有期。风险恢复时下一开盘回原四 ETF 动量，风险状态内的三类 sleeve 也可逐日互换。
- 四只风险 ETF 与 `512890.SH` 单边成本 1bp，`511260.SH` 单边成本 0.1bp；卖出和买入均计费。

## 窗口与结果

- 全样本：2019-01-18—2026-08-17，共 {len(daily):,} 个交易日。
- 开发观察段：2019-01-18—2024-12-31。
- 后段复核：2025-01-01—2026-08-17。由于完整历史参与过机制探索，这不是严格未观察 OOS。

| 策略 | 自然年 CAGR | Sharpe | 年化波动 | 最大回撤 | 总收益 |
|---|---:|---:|---:|---:|---:|
| 原四 ETF 动量 | {float(baseline_metrics['cagr_calendar']):.2%} | {float(baseline_metrics['sharpe']):.3f} | {float(baseline_metrics['annualized_volatility']):.2%} | {float(baseline_metrics['max_drawdown']):.2%} | {float(baseline_metrics['total_return']):.2%} |
| 无锁定避险选择 | {float(selected_performance['cagr_calendar']):.2%} | {float(selected_performance['sharpe']):.3f} | {float(selected_performance['annualized_volatility']):.2%} | {float(selected_performance['max_drawdown']):.2%} | {float(selected_performance['total_return']):.2%} |

全样本相对原动量：CAGR {float(selected_performance['cagr_calendar']) - float(baseline_metrics['cagr_calendar']):+.2%}，Sharpe {float(selected_performance['sharpe']) - float(baseline_metrics['sharpe']):+.3f}。开发段相对原动量：CAGR {float(selected_metrics['development_delta_cagr_calendar']):+.2%}、Sharpe {float(selected_metrics['development_delta_sharpe']):+.3f}；后段复核：CAGR {float(selected_metrics['later_period_delta_cagr_calendar']):+.2%}、Sharpe {float(selected_metrics['later_period_delta_sharpe']):+.3f}。

策略发生 {int(selected_metrics['sleeve_switches'])} 次 sleeve 状态变化；交易日分布为：原动量 {float(selected_metrics['day_share_momentum']):.2%}、Defender {float(selected_metrics['day_share_defender']):.2%}、黄金 {float(selected_metrics['day_share_gold']):.2%}、纳指 {float(selected_metrics['day_share_nasdaq']):.2%}。

截至 2026-08-17 收盘，`510300.SH` 的 40 日收益率为 {latest_indicator:.2%}，仍处于风险状态；三类避险 QM20 分数为 Defender {float(latest_scores['defender']):.6f}、黄金 {float(latest_scores['gold']):.6f}、纳指 {float(latest_scores['nasdaq']):.6f}，所以下一交易日开盘目标为 {latest_choice}（若行情可交易）。

## 稳健性边界

- 预先限定的 8 个低复杂度候选见 `candidate_grid.csv`。冻结规则不是最高收益组合：保留原先提出的 40 日，只把阈值改为自然的 0%，选择层复用生产已有 QM20。
- 20–80 日、三个阈值、四个选择窗口的 156 点广义压力测试中，全样本同时提高 CAGR 和 Sharpe 的比例为 {float(lookback_full_pass.mean()):.2%}；在更可比的 30–50 日局部邻域中为 {float(local_full_pass.mean()):.2%}。这说明结果不是 40 日的孤立最高点，但也不是对参数完全不敏感。
- 30/40/50 三窗口中位数的 36 点替代机制中，开发段与后段均同时提高 CAGR 和 Sharpe 的比例为 {float(neighborhood_pass.mean()):.2%}，仅作结构替代检验，不用于挑参。
- 36 个月滚动窗口同时提高年化和 Sharpe 的比例为 {float(rolling_both.mean()):.2%}。
- 20 日成组、2,000 次配对 bootstrap 中，两项目标同时为正的比例为 {float(bootstrap_both.mean()):.2%}；未校正机制选择偏差。
- 8 个自然年中有 {int(annual_both_wins.sum())} 年同时提高收益和 Sharpe；风险状态的 {len(episodes)} 段中只有 {int((episodes['excess_log_return'] > 0).sum())} 段跑赢同期原动量，且前五个正贡献段占全部正贡献的 {top_five_positive_share:.2%}。优势并非在每次风险触发时稳定出现。
- 无锁定确实生效：最短 sleeve 区间为 {int(holding_lengths.min())} 个交易日，共有 {int((holding_lengths == 1).sum())} 个单日区间；中位持有期仅 {float(holding_lengths.median()):.0f} 日，因此换手显著高于带 30 日锁定的版本。
- 成本按基准的 1、2、5、10 倍压力测试见 `cost_stress.csv`：2 倍成本仍双目标领先，5 倍时优势消失。
- 这是回溯证据，不是严格的过拟合证明。参数冻结后仍需前瞻 shadow；未来不应因短期落后再调参。
"""
    (args.output / "research_report.md").write_text(report, encoding="utf-8")

    print(pd.DataFrame(summary_rows).to_string(index=False))
    print("selected", selected_metrics)
    print("candidate grid")
    print(
        grid[
            [
                "risk_windows",
                "risk_threshold",
                "selector_method",
                "cagr_calendar",
                "sharpe",
                "development_delta_cagr_calendar",
                "development_delta_sharpe",
                "later_period_delta_cagr_calendar",
                "later_period_delta_sharpe",
            ]
        ].to_string(index=False)
    )
    print("output", args.output)


if __name__ == "__main__":
    main()
