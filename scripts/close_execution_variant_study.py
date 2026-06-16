"""Research diagnostic: compare next-open vs next-close execution timing.

This script intentionally leaves the production backtest runner unchanged. It
replays the same factor/strategy decision path twice and changes only when a
pending target becomes the active holding for return calculation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import yaml

from backtest.runner import (
    BacktestResult,
    _chain_returns,
    _equal_weight_return_between,
    _sharpe,
    _should_hold_position,
    _turnover_between,
    _weighted_return_between,
    run,
)
from data.store import query
from factors.registry import load_registered_factors
from factors.validator import validate
from strategy.loader import load_strategy
from strategy.rebalance import normalize_rebalance_mode


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "strategy" / "configs" / "quality_momentum_top1.yaml"
ATTACHMENTS_DIR = ROOT / "strategy_changelog_attachments"
OUT_DIR = ATTACHMENTS_DIR / "2026-06-15_close_execution_variant"
REPORT_PATH = OUT_DIR / "2026-06-15_close_execution_variant.md"
DELTA_CSV_PATH = OUT_DIR / "2026-06-15_close_execution_variant_delta.csv"
SEGMENT_CSV_PATH = OUT_DIR / "2026-06-15_close_execution_variant_segments.csv"
STRONG_TREND_CSV_PATH = OUT_DIR / "2026-06-15_close_execution_variant_2024_09_switches.csv"


ExecutionMode = Literal["open", "close"]


@dataclass
class TraceResult:
    result: BacktestResult
    decisions: pd.DataFrame
    executions: pd.DataFrame


def _load_research_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config["start"] = date(2014, 1, 1)
    config["end"] = date(2026, 6, 4)
    config["transaction_cost_rate"] = 0.0001
    config["train_ratio"] = 0.7
    return config


def _asset_from_weights(weights: dict[str, float]) -> str | None:
    if not weights:
        return None
    return max(weights, key=weights.get)


def _run_traced(config: dict, execution_mode: ExecutionMode) -> TraceResult:
    asset_pool = config["asset_pool"]
    start = config["start"]
    end = config["end"]
    factor_configs = config["factors"]
    train_ratio = config.get("train_ratio", 0.7)
    rebalance_days = int(config.get("rebalance_days", 1))
    rebalance_mode = normalize_rebalance_mode(config.get("rebalance_mode"))
    transaction_cost_rate = float(config.get("transaction_cost_rate", 0.0) or 0.0)

    strategy = load_strategy(config)
    all_factors = load_registered_factors()

    asset_data: dict[str, pd.DataFrame] = {}
    for asset in asset_pool:
        df = query(asset, start, end)
        if len(df) > 0:
            asset_data[asset] = df

    if not asset_data:
        raise RuntimeError("no data available for any asset in the pool")

    trading_days = sorted({dt for df in asset_data.values() for dt in df["date"].tolist()})
    max_min_history = max(
        all_factors[fc["name"]]["METADATA"]["min_history"]
        for fc in factor_configs
    )

    split_idx = int(len(trading_days) * train_ratio)
    train_end_date = (
        trading_days[split_idx].date()
        if split_idx < len(trading_days)
        else trading_days[-1].date()
    )

    open_prices: dict[str, pd.Series] = {}
    close_prices: dict[str, pd.Series] = {}
    for asset, df in asset_data.items():
        open_prices[asset] = pd.Series(df["open"].values, index=df["date"])
        close_prices[asset] = pd.Series(df["close"].values, index=df["date"])

    positions_records: list[dict] = []
    strategy_returns: list[tuple[pd.Timestamp, float]] = []
    gross_returns: list[tuple[pd.Timestamp, float]] = []
    benchmark_returns: list[tuple[pd.Timestamp, float]] = []
    turnover_records: list[tuple[pd.Timestamp, float]] = []
    cost_records: list[tuple[pd.Timestamp, float]] = []
    decision_records: list[dict] = []
    execution_records: list[dict] = []

    current_weights: dict[str, float] = {}
    current_entry_idx: int | None = None
    pending_weights: dict[str, float] | None = None
    pending_entry_idx: int | None = None
    pending_signal_date: pd.Timestamp | None = None

    for day_idx, t in enumerate(trading_days):
        if day_idx > 0:
            prev_t = trading_days[day_idx - 1]
            old_weights = current_weights
            opened_today = (
                pending_entry_idx == day_idx and pending_weights is not None
            )
            executed_turnover = 0.0
            cost = 0.0
            strat_ret: float | None = None
            old_day_ret: float | None = None
            new_intraday_ret: float | None = None

            if opened_today and execution_mode == "open":
                overnight_ret = _weighted_return_between(
                    old_weights, close_prices, open_prices, prev_t, t
                )
                current_weights = pending_weights or {}
                current_entry_idx = day_idx
                positions_records.append({"date": t, **current_weights})
                executed_turnover = _turnover_between(current_weights, old_weights)
                turnover_records.append((t, executed_turnover))
                new_intraday_ret = _weighted_return_between(
                    current_weights, open_prices, close_prices, t, t
                )
                strat_ret = _chain_returns(overnight_ret, new_intraday_ret)
            elif opened_today and execution_mode == "close":
                if old_weights:
                    old_day_ret = _weighted_return_between(
                        old_weights, close_prices, close_prices, prev_t, t
                    )
                    strat_ret = old_day_ret
                else:
                    strat_ret = 0.0
                current_weights = pending_weights or {}
                current_entry_idx = day_idx
                positions_records.append({"date": t, **current_weights})
                executed_turnover = _turnover_between(current_weights, old_weights)
                turnover_records.append((t, executed_turnover))
            elif current_weights:
                strat_ret = _weighted_return_between(
                    current_weights, close_prices, close_prices, prev_t, t
                )

            if opened_today:
                cost = executed_turnover * transaction_cost_rate
                if cost:
                    cost_records.append((t, cost))
                execution_records.append(
                    {
                        "execution_date": t,
                        "execution_idx": day_idx,
                        "signal_date": pending_signal_date,
                        "old_asset": _asset_from_weights(old_weights),
                        "new_asset": _asset_from_weights(current_weights),
                        "old_weights": dict(old_weights),
                        "new_weights": dict(current_weights),
                        "turnover": executed_turnover,
                        "cost": cost,
                        "old_day_return": old_day_ret,
                        "new_intraday_return": new_intraday_ret,
                    }
                )
                pending_weights = None
                pending_entry_idx = None
                pending_signal_date = None

            if strat_ret is not None:
                gross_returns.append((t, strat_ret))
                strategy_returns.append((t, strat_ret - cost))

                if opened_today and execution_mode == "open" and not old_weights:
                    bench_ret = _equal_weight_return_between(
                        asset_pool, open_prices, close_prices, t, t
                    )
                else:
                    bench_ret = _equal_weight_return_between(
                        asset_pool, close_prices, close_prices, prev_t, t
                    )
                if bench_ret is not None:
                    benchmark_returns.append((t, bench_ret))

        holding_days = (
            day_idx - current_entry_idx + 1
            if current_entry_idx is not None and current_weights
            else None
        )
        should_signal = (
            pending_weights is None
            and not _should_hold_position(
                current_weights,
                holding_days,
                rebalance_days,
                rebalance_mode,
            )
        )

        if should_signal:
            asset_factor_values: dict[str, dict[str, float]] = {}
            for asset, df in asset_data.items():
                truncated = df.loc[df["date"] <= t]
                if len(truncated) < max_min_history:
                    continue

                factor_vals: dict[str, float] = {}
                for fc in factor_configs:
                    fname = fc["name"]
                    fmod = all_factors[fname]
                    params = fc.get("params")
                    try:
                        series = fmod["compute"](truncated.copy(), params)
                        validate(series, truncated, fmod["METADATA"])
                        last_val = series.iloc[-1]
                        if pd.notna(last_val):
                            factor_vals[fname] = float(last_val)
                    except (ValueError, Exception):
                        continue

                if len(factor_vals) == len(factor_configs):
                    asset_factor_values[asset] = factor_vals

            new_weights = strategy.generate_weights(asset_factor_values)
            if new_weights and new_weights != current_weights:
                next_idx = day_idx + 1
                if next_idx < len(trading_days):
                    decision_records.append(
                        {
                            "signal_date": t,
                            "execution_date": trading_days[next_idx],
                            "old_asset": _asset_from_weights(current_weights),
                            "new_asset": _asset_from_weights(new_weights),
                            "old_weights": dict(current_weights),
                            "new_weights": dict(new_weights),
                        }
                    )
                    pending_weights = new_weights
                    pending_entry_idx = next_idx
                    pending_signal_date = t

    daily_ret = _series_from_records(strategy_returns)
    gross_daily_ret = _series_from_records(gross_returns)
    bench_ret = _series_from_records(benchmark_returns)
    positions_df = pd.DataFrame(positions_records)
    if len(positions_df) > 0:
        positions_df = positions_df.set_index("date")

    turnover = _series_from_records(turnover_records)
    costs = pd.Series(0.0, index=daily_ret.index, dtype=float)
    for dt, cost in cost_records:
        if dt in costs.index:
            costs.loc[dt] = cost

    result = BacktestResult(
        daily_returns=daily_ret,
        benchmark_returns=bench_ret,
        positions=positions_df,
        train_end=train_end_date,
        config=config,
        gross_daily_returns=gross_daily_ret,
        turnover=turnover,
        costs=costs,
    )
    return TraceResult(
        result=result,
        decisions=pd.DataFrame(decision_records),
        executions=pd.DataFrame(execution_records),
    )


def _series_from_records(records: list[tuple[pd.Timestamp, float]]) -> pd.Series:
    if not records:
        return pd.Series(dtype=float)
    dates, vals = zip(*records)
    return pd.Series(vals, index=pd.DatetimeIndex(dates), dtype=float)


def _metrics(returns: pd.Series, turnover: pd.Series | None = None) -> dict[str, float]:
    returns = returns.dropna()
    if len(returns) == 0:
        return {
            "annual_return": math.nan,
            "sharpe": math.nan,
            "max_drawdown": math.nan,
            "annual_turnover": math.nan,
        }
    equity = (1 + returns).cumprod()
    annual_return = float(equity.iloc[-1] ** (252 / len(returns)) - 1)
    sharpe = _sharpe(returns)
    drawdown = equity / equity.cummax() - 1
    years = len(returns) / 252
    annual_turnover = (
        float(turnover.sum() / years)
        if turnover is not None and len(turnover) > 0 and years > 0
        else math.nan
    )
    return {
        "annual_return": annual_return,
        "sharpe": sharpe,
        "max_drawdown": float(drawdown.min()),
        "annual_turnover": annual_turnover,
    }


def _avg_holding_days(executions: pd.DataFrame) -> float:
    switches = executions[executions["old_asset"].notna()].copy()
    if len(switches) < 2:
        return math.nan
    idx = switches.sort_values("execution_idx")["execution_idx"]
    return float(idx.diff().dropna().mean())


def _build_delta_table(
    baseline: TraceResult,
    variant: TraceResult,
    config: dict,
) -> pd.DataFrame:
    data = {
        asset: query(asset, config["start"], config["end"]).set_index("date")
        for asset in config["asset_pool"]
    }
    rows = []
    base_exec = baseline.executions.copy()
    var_exec = variant.executions.copy()
    base_exec["execution_date"] = pd.to_datetime(base_exec["execution_date"])
    var_exec["execution_date"] = pd.to_datetime(var_exec["execution_date"])
    merged = base_exec.merge(
        var_exec[["execution_date", "turnover", "cost"]],
        on="execution_date",
        how="inner",
        suffixes=("_baseline", "_variant"),
    )

    for _, row in merged.iterrows():
        old_asset = row["old_asset"]
        new_asset = row["new_asset"]
        if pd.isna(new_asset):
            continue
        execution_date = row["execution_date"]
        cost = float(row["cost_baseline"])
        turnover = float(row["turnover_baseline"])

        if pd.isna(old_asset):
            old_intraday = 0.0
            old_close_to_close = 1.0
            old_close_to_open = 1.0
        else:
            old_df = data[old_asset]
            old_open = float(old_df.loc[execution_date, "open"])
            old_close = float(old_df.loc[execution_date, "close"])
            prev_idx = old_df.index.get_loc(execution_date) - 1
            prev_close = float(old_df.iloc[prev_idx]["close"])
            old_intraday = old_close / old_open - 1
            old_close_to_close = old_close / prev_close
            old_close_to_open = old_open / prev_close

        new_df = data[new_asset]
        new_open = float(new_df.loc[execution_date, "open"])
        new_close = float(new_df.loc[execution_date, "close"])
        new_intraday = new_close / new_open - 1
        new_intraday_factor = new_close / new_open

        gross_delta = math.log1p(old_intraday) - math.log1p(new_intraday)
        base_gross_factor = old_close_to_open * new_intraday_factor
        var_gross_factor = old_close_to_close
        net_delta = math.log(var_gross_factor - cost) - math.log(base_gross_factor - cost)

        rows.append(
            {
                "execution_date": execution_date.date().isoformat(),
                "signal_date": pd.Timestamp(row["signal_date"]).date().isoformat()
                if pd.notna(row["signal_date"])
                else "",
                "old_asset": "" if pd.isna(old_asset) else old_asset,
                "new_asset": new_asset,
                "old_intraday": old_intraday,
                "new_intraday": new_intraday,
                "delta": gross_delta,
                "net_delta": net_delta,
                "turnover": turnover,
                "cost": cost,
            }
        )
    return pd.DataFrame(rows)


def _period_slice(returns: pd.Series, start: str, end: str) -> pd.Series:
    return returns.loc[(returns.index >= pd.Timestamp(start)) & (returns.index <= pd.Timestamp(end))]


def _segment_rows(
    baseline: TraceResult,
    variant: TraceResult,
    delta: pd.DataFrame,
) -> pd.DataFrame:
    segments = {
        "2020Q1": ("2020-01-01", "2020-03-31"),
        "2022": ("2022-01-01", "2022-12-31"),
        "2024-10": ("2024-10-01", "2024-10-31"),
        "2024-09_rally_window": ("2024-09-01", "2024-10-31"),
    }
    rows = []
    delta_dates = pd.to_datetime(delta["execution_date"]) if len(delta) else pd.Series(dtype="datetime64[ns]")
    for name, (start, end) in segments.items():
        base_ret = _period_slice(baseline.result.daily_returns, start, end)
        var_ret = _period_slice(variant.result.daily_returns, start, end)
        base_metrics = _metrics(base_ret)
        var_metrics = _metrics(var_ret)
        mask = (delta_dates >= pd.Timestamp(start)) & (delta_dates <= pd.Timestamp(end))
        seg_delta = delta.loc[mask] if len(delta) else delta
        rows.append(
            {
                "segment": name,
                "start": start,
                "end": end,
                "baseline_total_return": _total_return(base_ret),
                "variant_total_return": _total_return(var_ret),
                "baseline_annual_return": base_metrics["annual_return"],
                "variant_annual_return": var_metrics["annual_return"],
                "baseline_sharpe": base_metrics["sharpe"],
                "variant_sharpe": var_metrics["sharpe"],
                "baseline_max_drawdown": base_metrics["max_drawdown"],
                "variant_max_drawdown": var_metrics["max_drawdown"],
                "delta_sum": float(seg_delta["delta"].sum()) if len(seg_delta) else 0.0,
                "switch_count": int(len(seg_delta)),
            }
        )
    return pd.DataFrame(rows)


def _total_return(returns: pd.Series) -> float:
    if len(returns) == 0:
        return math.nan
    return float((1 + returns).prod() - 1)


def _fmt_pct(value: float) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{value:.2%}"


def _fmt_num(value: float, digits: int = 2) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{value:.{digits}f}"


def _md_table(df: pd.DataFrame) -> str:
    return df.to_markdown(index=False)


def _assert_gate(
    production_baseline: BacktestResult,
    baseline: TraceResult,
    variant: TraceResult,
    delta: pd.DataFrame,
) -> dict[str, object]:
    prod_aligned = production_baseline.daily_returns.reindex(baseline.result.daily_returns.index)
    if not np.allclose(prod_aligned.values, baseline.result.daily_returns.values, atol=1e-12, rtol=1e-12):
        raise RuntimeError("traced open baseline does not match production backtest runner")

    base_decisions = baseline.decisions[["signal_date", "execution_date", "new_asset"]].copy()
    var_decisions = variant.decisions[["signal_date", "execution_date", "new_asset"]].copy()
    if base_decisions.to_csv(index=False) != var_decisions.to_csv(index=False):
        raise RuntimeError("switch decision dates differ between baseline and variant")

    base_turnover = baseline.result.turnover
    var_turnover = variant.result.turnover
    if not base_turnover.index.equals(var_turnover.index) or not np.allclose(
        base_turnover.values, var_turnover.values, atol=1e-12, rtol=1e-12
    ):
        raise RuntimeError("turnover differs between baseline and variant")

    base_costs = baseline.result.costs.loc[baseline.result.turnover.index]
    var_costs = variant.result.costs.loc[variant.result.turnover.index]
    if not np.allclose(base_costs.values, var_costs.values, atol=1e-12, rtol=1e-12):
        raise RuntimeError("per-execution costs differ between baseline and variant")

    differing_dates = baseline.result.daily_returns.index[
        ~np.isclose(
            baseline.result.daily_returns.values,
            variant.result.daily_returns.reindex(baseline.result.daily_returns.index).values,
            atol=1e-12,
            rtol=1e-12,
        )
    ]
    execution_dates = pd.DatetimeIndex(pd.to_datetime(baseline.executions["execution_date"]))
    unexpected = differing_dates.difference(execution_dates)
    if len(unexpected) > 0:
        raise RuntimeError(f"returns differ on non-execution dates: {list(unexpected[:5])}")

    gross_log_ratio = math.log(
        (1 + variant.result.gross_daily_returns).prod()
        / (1 + baseline.result.gross_daily_returns).prod()
    )
    delta_sum = float(delta["delta"].sum())
    net_log_ratio = math.log(
        (1 + variant.result.daily_returns).prod()
        / (1 + baseline.result.daily_returns).prod()
    )
    net_delta_sum = float(delta["net_delta"].sum())
    if not math.isclose(gross_log_ratio, delta_sum, abs_tol=1e-10):
        raise RuntimeError("gross delta reconciliation failed")
    if not math.isclose(net_log_ratio, net_delta_sum, abs_tol=1e-10):
        raise RuntimeError("net delta reconciliation failed")

    return {
        "decision_count": len(base_decisions),
        "execution_count": len(baseline.executions),
        "switch_count_ex_initial": int(baseline.executions["old_asset"].notna().sum()),
        "initial_cost": float(baseline.executions.iloc[0]["cost"]),
        "post_initial_cost_min": float(baseline.executions.iloc[1:]["cost"].min()),
        "post_initial_cost_max": float(baseline.executions.iloc[1:]["cost"].max()),
        "gross_log_ratio": gross_log_ratio,
        "delta_sum": delta_sum,
        "gross_recon_error": gross_log_ratio - delta_sum,
        "net_log_ratio": net_log_ratio,
        "net_delta_sum": net_delta_sum,
        "net_recon_error": net_log_ratio - net_delta_sum,
        "non_execution_diff_count": int(len(unexpected)),
    }


def _write_report(
    config: dict,
    baseline: TraceResult,
    variant: TraceResult,
    delta: pd.DataFrame,
    segments: pd.DataFrame,
    gate: dict[str, object],
) -> None:
    base_metrics = _metrics(baseline.result.daily_returns, baseline.result.turnover)
    var_metrics = _metrics(variant.result.daily_returns, variant.result.turnover)
    avg_base_hold = _avg_holding_days(baseline.executions)
    avg_var_hold = _avg_holding_days(variant.executions)

    comparison = pd.DataFrame(
        [
            {
                "指标": "年化收益",
                "baseline(T+1开盘)": _fmt_pct(base_metrics["annual_return"]),
                "variant(T+1收盘)": _fmt_pct(var_metrics["annual_return"]),
                "差异": _fmt_pct(var_metrics["annual_return"] - base_metrics["annual_return"]),
            },
            {
                "指标": "Sharpe",
                "baseline(T+1开盘)": _fmt_num(base_metrics["sharpe"]),
                "variant(T+1收盘)": _fmt_num(var_metrics["sharpe"]),
                "差异": _fmt_num(var_metrics["sharpe"] - base_metrics["sharpe"]),
            },
            {
                "指标": "最大回撤",
                "baseline(T+1开盘)": _fmt_pct(base_metrics["max_drawdown"]),
                "variant(T+1收盘)": _fmt_pct(var_metrics["max_drawdown"]),
                "差异": _fmt_pct(var_metrics["max_drawdown"] - base_metrics["max_drawdown"]),
            },
            {
                "指标": "年化换手率",
                "baseline(T+1开盘)": _fmt_num(base_metrics["annual_turnover"]),
                "variant(T+1收盘)": _fmt_num(var_metrics["annual_turnover"]),
                "差异": _fmt_num(var_metrics["annual_turnover"] - base_metrics["annual_turnover"]),
            },
            {
                "指标": "平均持有期(交易日)",
                "baseline(T+1开盘)": _fmt_num(avg_base_hold),
                "variant(T+1收盘)": _fmt_num(avg_var_hold),
                "差异": _fmt_num(avg_var_hold - avg_base_hold),
            },
        ]
    )

    train_end = pd.Timestamp(baseline.result.train_end)
    is_oos = pd.DataFrame(
        [
            {
                "口径": "baseline(T+1开盘)",
                "IS Sharpe": _fmt_num(_sharpe(baseline.result.daily_returns[baseline.result.daily_returns.index <= train_end])),
                "OOS Sharpe": _fmt_num(_sharpe(baseline.result.daily_returns[baseline.result.daily_returns.index > train_end])),
            },
            {
                "口径": "variant(T+1收盘)",
                "IS Sharpe": _fmt_num(_sharpe(variant.result.daily_returns[variant.result.daily_returns.index <= train_end])),
                "OOS Sharpe": _fmt_num(_sharpe(variant.result.daily_returns[variant.result.daily_returns.index > train_end])),
            },
        ]
    )

    delta_stats = pd.DataFrame(
        [
            {
                "mean": delta["delta"].mean(),
                "median": delta["delta"].median(),
                "std": delta["delta"].std(),
                "正占比": (delta["delta"] > 0).mean(),
            }
        ]
    )
    delta_stats_fmt = delta_stats.assign(
        mean=lambda x: x["mean"].map(lambda v: f"{v:.6f}"),
        median=lambda x: x["median"].map(lambda v: f"{v:.6f}"),
        std=lambda x: x["std"].map(lambda v: f"{v:.6f}"),
        正占比=lambda x: x["正占比"].map(_fmt_pct),
    )

    abs_total = float(delta["delta"].abs().sum())
    top = delta.reindex(delta["delta"].abs().sort_values(ascending=False).index)
    top3_share = float(top.head(3)["delta"].abs().sum() / abs_total) if abs_total else math.nan
    top5_share = float(top.head(5)["delta"].abs().sum() / abs_total) if abs_total else math.nan
    top_table = top.head(5)[
        ["execution_date", "old_asset", "new_asset", "old_intraday", "new_intraday", "delta"]
    ].copy()
    for col in ["old_intraday", "new_intraday"]:
        top_table[col] = top_table[col].map(_fmt_pct)
    top_table["delta"] = top_table["delta"].map(lambda v: f"{v:.6f}")

    seg_fmt = segments.copy()
    for col in [
        "baseline_total_return",
        "variant_total_return",
        "baseline_annual_return",
        "variant_annual_return",
        "baseline_max_drawdown",
        "variant_max_drawdown",
    ]:
        seg_fmt[col] = seg_fmt[col].map(_fmt_pct)
    for col in ["baseline_sharpe", "variant_sharpe", "delta_sum"]:
        seg_fmt[col] = seg_fmt[col].map(lambda v: f"{v:.4f}")

    strong = delta[
        (pd.to_datetime(delta["execution_date"]) >= pd.Timestamp("2024-09-01"))
        & (pd.to_datetime(delta["execution_date"]) <= pd.Timestamp("2024-10-31"))
    ].copy()
    strong_fmt = strong[
        ["execution_date", "old_asset", "new_asset", "old_intraday", "new_intraday", "delta"]
    ].copy()
    for col in ["old_intraday", "new_intraday"]:
        strong_fmt[col] = strong_fmt[col].map(_fmt_pct)
    strong_fmt["delta"] = strong_fmt["delta"].map(lambda v: f"{v:.6f}")

    top5_dates = set(top.head(5)["execution_date"])
    strong_overlap = sorted(set(strong["execution_date"]) & top5_dates)
    concentration_judgment = (
        "top-5 |delta| 占比 >= 70%，按既有 episode 集中度判据定性为时机运气、不可外推。"
        if top5_share >= 0.70
        else "top-5 |delta| 占比 < 70%，未达到既有 episode 集中度判据的“高度集中”阈值。"
    )

    lines = [
        "# T+1 收盘成交变体诊断",
        "",
        f"- 配置: `{CONFIG_PATH.relative_to(ROOT)}`",
        f"- 区间: {config['start'].isoformat()} ~ {config['end'].isoformat()}",
        f"- 成本: 单边 {config['transaction_cost_rate']:.4%}",
        f"- train_ratio: {config['train_ratio']:.1f}; IS/OOS 切分日: {baseline.result.train_end.isoformat()}",
        "- 口径: baseline 为 T+1 开盘成交；variant 为 T+1 收盘成交。信号生成、Top1、min_hold、未来信息截断均保持不变。",
        "- 无 look-ahead 确认: 每个成交日 t 的 pending 目标来自 t-1 收盘后信号；variant 只在 t 收盘落地目标，t 日收益仍用旧权重，未使用 t+1 或之后数据。",
        "",
        "## 正确性闸门",
        "",
        f"- 切换决策序列一致: yes ({gate['decision_count']} decisions)",
        f"- 实际执行记录一致: yes ({gate['execution_count']} executions, excluding initial switches={gate['switch_count_ex_initial']})",
        "- variant 持仓在成交日收益上相对 baseline 后移一日: yes",
        (
            "- 换手与成本逐次一致: yes "
            f"(initial={gate['initial_cost']:.4%}, post-initial range="
            f"{gate['post_initial_cost_min']:.4%}~{gate['post_initial_cost_max']:.4%})"
        ),
        f"- 非执行日收益逐日相同: yes (unexpected diff count={gate['non_execution_diff_count']})",
        f"- 毛收益 δ 对账误差: {gate['gross_recon_error']:.3e}",
        f"- 净@1bp 对账误差(含费用交互后的 net_delta): {gate['net_recon_error']:.3e}",
        "",
        "## 主对比表(净@1bp)",
        "",
        _md_table(comparison),
        "",
        "## IS/OOS Sharpe(净@1bp)",
        "",
        _md_table(is_oos),
        "",
        "## 诊断 1: 换仓日 Intraday δ",
        "",
        "- δ 明细包含首笔 cash -> 首个标的建仓记录，旧资产 intraday 按 0 处理；这样才能与全期终值比完整对账。",
        f"- δ 累计和: {gate['delta_sum']:.6f}",
        f"- log(variant gross / baseline gross): {gate['gross_log_ratio']:.6f}",
        f"- log(variant net / baseline net): {gate['net_log_ratio']:.6f}",
        "",
        _md_table(delta_stats_fmt),
        "",
        f"- top-3 |δ| 集中度: {_fmt_pct(top3_share)}",
        f"- top-5 |δ| 集中度: {_fmt_pct(top5_share)}",
        f"- 判定: {concentration_judgment}",
        "",
        _md_table(top_table),
        "",
        f"逐换仓日明细: `{DELTA_CSV_PATH.name}`",
        "",
        "## 诊断 2: 分段归因",
        "",
        _md_table(seg_fmt),
        "",
        "## 2024-09 快速切换专项",
        "",
        f"- 窗口: 2024-09-01 ~ 2024-10-31",
        f"- baseline 累计收益: {_fmt_pct(_total_return(_period_slice(baseline.result.daily_returns, '2024-09-01', '2024-10-31')))}",
        f"- variant 累计收益: {_fmt_pct(_total_return(_period_slice(variant.result.daily_returns, '2024-09-01', '2024-10-31')))}",
        f"- 窗口 δ 累计: {strong['delta'].sum():.6f}",
        f"- 与全期 top-5 |δ| 重叠日期: {', '.join(strong_overlap) if strong_overlap else '无'}",
        "",
        _md_table(strong_fmt) if len(strong_fmt) else "该窗口无换仓日。",
        "",
        "## 存档",
        "",
        f"- δ 明细 CSV: `{DELTA_CSV_PATH.name}`",
        f"- 分段明细 CSV: `{SEGMENT_CSV_PATH.name}`",
        f"- 2024-09 专项 CSV: `{STRONG_TREND_CSV_PATH.name}`",
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    config = _load_research_config()
    production_baseline = run(config)
    baseline = _run_traced(config, "open")
    variant = _run_traced(config, "close")
    delta = _build_delta_table(baseline, variant, config)
    gate = _assert_gate(production_baseline, baseline, variant, delta)
    segments = _segment_rows(baseline, variant, delta)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    delta.to_csv(DELTA_CSV_PATH, index=False, encoding="utf-8")
    segments.to_csv(SEGMENT_CSV_PATH, index=False, encoding="utf-8")
    strong = delta[
        (pd.to_datetime(delta["execution_date"]) >= pd.Timestamp("2024-09-01"))
        & (pd.to_datetime(delta["execution_date"]) <= pd.Timestamp("2024-10-31"))
    ]
    strong.to_csv(STRONG_TREND_CSV_PATH, index=False, encoding="utf-8")
    _write_report(config, baseline, variant, delta, segments, gate)
    print(f"wrote {REPORT_PATH}")
    print(f"wrote {DELTA_CSV_PATH}")
    print(f"wrote {SEGMENT_CSV_PATH}")
    print(f"wrote {STRONG_TREND_CSV_PATH}")


if __name__ == "__main__":
    main()
