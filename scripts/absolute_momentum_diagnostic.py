"""Research-only absolute-momentum cash overlay diagnostic.

This script does not edit production YAMLs and does not call live/backfill.
It reuses the current Top1 quality-momentum signal, then applies an
absolute-momentum cash overlay immediately after target-weight generation and
before accounting.
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import warnings
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from backtest.runner import (
    _chain_returns,
    _equal_weight_return_between,
    _turnover_between,
    _weighted_return_between,
)
from data.store import query
from factors.registry import load_registered_factors
from factors.validator import validate
from strategy.loader import load_strategy
from strategy.rebalance import normalize_rebalance_mode, should_hold_position


REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "strategy" / "configs" / "quality_momentum_top1.yaml"
OUT_DIR = REPO_ROOT / "strategy_changelog_attachments"
PREFIX = "2026-06-05_absolute_momentum_diagnostic"

START = date(2014, 1, 1)
HISTORY_START = date(2010, 1, 1)
ASSET_POOL = ["510300.SH", "159915.SZ", "513100.SH", "518880.SH"]
WINDOWS = [20, 40, 60, 120, 200, 250]
SMA_WINDOWS = [40, 60, 200]
CASH_YIELDS = [0.0, 0.02]
FEES = [0.0001, 0.0003, 0.0005, 0.001]
BASE_FEE = 0.0001
TRAIN_RATIO = 0.7
BAD_WINDOWS = [
    ("2020Q1", pd.Timestamp("2020-01-01"), pd.Timestamp("2020-03-31")),
    ("2022", pd.Timestamp("2022-01-01"), pd.Timestamp("2022-12-31")),
    (
        "2024-10-08_2024-11-15",
        pd.Timestamp("2024-10-08"),
        pd.Timestamp("2024-11-15"),
    ),
]


@dataclasses.dataclass(frozen=True)
class OverlaySpec:
    design: str
    window: int
    threshold: str
    cash_yield: float

    @property
    def label(self) -> str:
        return (
            f"{self.design}_N{self.window}_{self.threshold}_"
            f"cash{self.cash_yield:.2%}"
        )


@dataclasses.dataclass
class OverlayRun:
    spec: OverlaySpec
    gross: pd.Series
    turnover: pd.Series
    positions: pd.DataFrame
    benchmark_returns: pd.Series
    state: pd.Series
    evals: pd.DataFrame
    train_end: date


@dataclasses.dataclass
class MarketContext:
    config: dict
    asset_data: dict[str, pd.DataFrame]
    history_data: dict[str, pd.DataFrame]
    trading_days: list[pd.Timestamp]
    open_prices: dict[str, pd.Series]
    close_prices: dict[str, pd.Series]
    am_close_prices: dict[str, pd.Series]
    factor_modules: dict
    max_min_history: int


def _load_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config["asset_pool"] = list(ASSET_POOL)
    config["start"] = START
    config["end"] = date.today()
    config["rebalance_days"] = 5
    config["train_ratio"] = TRAIN_RATIO
    config["transaction_cost_rate"] = 0.0
    config.pop("rebalance_mode", None)
    config.pop("hysteresis_threshold", None)
    config.pop("tau", None)
    return config


def _build_context() -> MarketContext:
    config = _load_config()
    asset_data: dict[str, pd.DataFrame] = {}
    history_data: dict[str, pd.DataFrame] = {}
    for asset in ASSET_POOL:
        df = query(asset, config["start"], config["end"])
        if len(df) > 0:
            asset_data[asset] = df.copy()
        hist = query(asset, HISTORY_START, config["end"])
        if len(hist) > 0:
            history_data[asset] = hist.copy()
    if set(asset_data) != set(ASSET_POOL):
        missing = sorted(set(ASSET_POOL) - set(asset_data))
        raise RuntimeError(f"missing local data for assets: {missing}")
    if set(history_data) != set(ASSET_POOL):
        missing = sorted(set(ASSET_POOL) - set(history_data))
        raise RuntimeError(f"missing local history for assets: {missing}")

    trading_days = sorted({d for df in asset_data.values() for d in df["date"].tolist()})
    factor_modules = load_registered_factors()
    max_min_history = max(
        factor_modules[fc["name"]]["METADATA"]["min_history"]
        for fc in config["factors"]
    )
    open_prices = {
        asset: pd.Series(df["open"].values, index=pd.DatetimeIndex(df["date"]))
        for asset, df in asset_data.items()
    }
    close_prices = {
        asset: pd.Series(df["close"].values, index=pd.DatetimeIndex(df["date"]))
        for asset, df in asset_data.items()
    }
    am_close_prices = {
        asset: pd.Series(df["close"].values, index=pd.DatetimeIndex(df["date"]))
        for asset, df in history_data.items()
    }
    return MarketContext(
        config=config,
        asset_data=asset_data,
        history_data=history_data,
        trading_days=trading_days,
        open_prices=open_prices,
        close_prices=close_prices,
        am_close_prices=am_close_prices,
        factor_modules=factor_modules,
        max_min_history=max_min_history,
    )


def _cash_weights() -> dict[str, float]:
    return {asset: 0.0 for asset in ASSET_POOL}


def _is_cash(weights: dict[str, float]) -> bool:
    return bool(weights) and all(abs(float(weights.get(asset, 0.0))) < 1e-12 for asset in ASSET_POOL)


def _is_active_state(weights: dict[str, float]) -> bool:
    return bool(weights)


def _state_label(weights: dict[str, float]) -> str | None:
    if not weights:
        return None
    if _is_cash(weights):
        return "CASH"
    nonzero = {asset: weight for asset, weight in weights.items() if abs(float(weight)) > 1e-12}
    if not nonzero:
        return "CASH"
    return max(nonzero, key=nonzero.get)


def _factor_snapshot(ctx: MarketContext, t: pd.Timestamp) -> dict[str, dict[str, float]]:
    factor_configs = ctx.config["factors"]
    out: dict[str, dict[str, float]] = {}
    for asset, df in ctx.asset_data.items():
        truncated = df.loc[df["date"] <= t]
        if len(truncated) < ctx.max_min_history:
            continue

        vals: dict[str, float] = {}
        for fc in factor_configs:
            fname = fc["name"]
            fmod = ctx.factor_modules[fname]
            try:
                series = fmod["compute"](truncated.copy(), fc.get("params"))
                validate(series, truncated, fmod["METADATA"])
                last_val = series.iloc[-1]
                if pd.notna(last_val):
                    vals[fname] = float(last_val)
            except (ValueError, Exception) as exc:
                warnings.warn(
                    f"factor '{fname}' failed for {asset} on {t}: {exc}",
                    stacklevel=2,
                )
        if len(vals) == len(factor_configs):
            out[asset] = vals
    return out


def _top_asset(weights: dict[str, float]) -> str | None:
    if not weights or _is_cash(weights):
        return None
    nonzero = {asset: weight for asset, weight in weights.items() if abs(float(weight)) > 1e-12}
    return max(nonzero, key=nonzero.get) if nonzero else None


def _am_value(closes: pd.Series, t: pd.Timestamp, window: int, threshold: str) -> float:
    if t not in closes.index:
        return float("nan")
    loc = int(closes.index.get_loc(t))
    if threshold == "point":
        if loc < window:
            return float("nan")
        prev = float(closes.iloc[loc - window])
        if prev == 0:
            return float("nan")
        return float(closes.iloc[loc] / prev - 1.0)
    if threshold == "sma":
        if loc + 1 < window:
            return float("nan")
        sma = float(closes.iloc[loc - window + 1 : loc + 1].mean())
        if not math.isfinite(sma) or sma == 0:
            return float("nan")
        return float(closes.iloc[loc] / sma - 1.0)
    raise ValueError(f"unknown threshold: {threshold}")


def _am_snapshot(
    ctx: MarketContext,
    t: pd.Timestamp,
    window: int,
    threshold: str,
) -> dict[str, float]:
    return {
        asset: _am_value(ctx.am_close_prices[asset], t, window, threshold)
        for asset in ASSET_POOL
    }


def _passes(am: dict[str, float]) -> set[str]:
    return {asset for asset, value in am.items() if pd.notna(value) and float(value) > 0.0}


def _overlay_target(
    *,
    ctx: MarketContext,
    strategy,
    spec: OverlaySpec,
    factor_values: dict[str, dict[str, float]],
    base_weights: dict[str, float],
    am: dict[str, float],
) -> dict[str, float]:
    if not base_weights:
        return {}
    passed = _passes(am)
    if spec.design == "design1_rank_then_gate":
        winner = _top_asset(base_weights)
        return base_weights if winner in passed else _cash_weights()
    if spec.design == "design2_filter_then_rank":
        if not passed:
            return _cash_weights()
        filtered = {asset: vals for asset, vals in factor_values.items() if asset in passed}
        target = strategy.generate_weights(filtered)
        return target if target else _cash_weights()
    raise ValueError(f"unknown design: {spec.design}")


def _cash_daily_return(cash_yield: float) -> float:
    return float((1.0 + cash_yield) ** (1.0 / 252.0) - 1.0)


def _position_return_between(
    weights: dict[str, float],
    start_prices: dict[str, pd.Series],
    end_prices: dict[str, pd.Series],
    start_t: pd.Timestamp,
    end_t: pd.Timestamp,
    cash_daily: float,
) -> float | None:
    if _is_cash(weights):
        return cash_daily
    return _weighted_return_between(weights, start_prices, end_prices, start_t, end_t)


def _run_overlay(ctx: MarketContext, spec: OverlaySpec) -> OverlayRun:
    config = dict(ctx.config)
    strategy = load_strategy(config)
    rebalance_days = int(config.get("rebalance_days", 1))
    rebalance_mode = normalize_rebalance_mode(config.get("rebalance_mode"))
    split_idx = int(len(ctx.trading_days) * float(config.get("train_ratio", TRAIN_RATIO)))
    train_end = (
        ctx.trading_days[split_idx].date()
        if split_idx < len(ctx.trading_days)
        else ctx.trading_days[-1].date()
    )
    cash_daily = _cash_daily_return(spec.cash_yield)

    current_weights: dict[str, float] = {}
    current_entry_idx: int | None = None
    pending_weights: dict[str, float] | None = None
    pending_entry_idx: int | None = None

    gross_rows: list[tuple[pd.Timestamp, float]] = []
    bench_rows: list[tuple[pd.Timestamp, float]] = []
    turnover_rows: list[tuple[pd.Timestamp, float]] = []
    position_rows: list[dict[str, object]] = []
    state_rows: list[tuple[pd.Timestamp, str | None]] = []
    eval_rows: list[dict[str, object]] = []

    for day_idx, t in enumerate(ctx.trading_days):
        if day_idx > 0:
            prev_t = ctx.trading_days[day_idx - 1]
            old_weights = current_weights
            opened_today = pending_entry_idx == day_idx and pending_weights is not None

            if opened_today:
                overnight_ret = _position_return_between(
                    old_weights,
                    ctx.close_prices,
                    ctx.open_prices,
                    prev_t,
                    t,
                    cash_daily,
                )
                current_weights = pending_weights or {}
                current_entry_idx = day_idx
                position_rows.append({"date": t, **current_weights})
                executed_turnover = _turnover_between(current_weights, old_weights)
                turnover_rows.append((t, executed_turnover))
                pending_weights = None
                pending_entry_idx = None
                intraday_ret = _position_return_between(
                    current_weights,
                    ctx.open_prices,
                    ctx.close_prices,
                    t,
                    t,
                    cash_daily,
                )
                strat_ret = _chain_returns(overnight_ret, intraday_ret)
            elif _is_active_state(current_weights):
                strat_ret = _position_return_between(
                    current_weights,
                    ctx.close_prices,
                    ctx.close_prices,
                    prev_t,
                    t,
                    cash_daily,
                )
            else:
                strat_ret = None

            if strat_ret is not None:
                gross_rows.append((t, float(strat_ret)))
                if opened_today and not old_weights:
                    bench_ret = _equal_weight_return_between(
                        ASSET_POOL, ctx.open_prices, ctx.close_prices, t, t
                    )
                else:
                    bench_ret = _equal_weight_return_between(
                        ASSET_POOL, ctx.close_prices, ctx.close_prices, prev_t, t
                    )
                if bench_ret is not None:
                    bench_rows.append((t, float(bench_ret)))
                state_rows.append((t, _state_label(current_weights)))

        holding_days = (
            day_idx - current_entry_idx + 1
            if current_entry_idx is not None and _is_active_state(current_weights)
            else None
        )
        should_signal = (
            pending_weights is None
            and not should_hold_position(
                current_weights,
                holding_days,
                rebalance_days,
                rebalance_mode,
            )
        )
        if not should_signal:
            continue

        factor_values = _factor_snapshot(ctx, t)
        base_weights = strategy.generate_weights(
            factor_values,
            current_weights=(
                {} if _is_cash(current_weights) else current_weights
            ),
        )
        am = _am_snapshot(ctx, t, spec.window, spec.threshold)
        passed = _passes(am)
        target = _overlay_target(
            ctx=ctx,
            strategy=strategy,
            spec=spec,
            factor_values=factor_values,
            base_weights=base_weights,
            am=am,
        )
        target_label = _state_label(target)
        current_label = _state_label(current_weights)
        next_idx = day_idx + 1
        will_switch = bool(target and target != current_weights)
        execution_date = (
            ctx.trading_days[next_idx]
            if will_switch and next_idx < len(ctx.trading_days)
            else pd.NaT
        )
        eval_rows.append(
            {
                "eval_date": t,
                "execution_date": execution_date,
                "design": spec.design,
                "window": spec.window,
                "threshold": spec.threshold,
                "cash_yield": spec.cash_yield,
                "current": current_label,
                "base_top1": _top_asset(base_weights),
                "target": target_label,
                "am_valid_count": int(sum(pd.notna(v) for v in am.values())),
                "passes_count": int(len(passed)),
                "top1_passes": bool(_top_asset(base_weights) in passed),
                "all_four_negative": bool(
                    sum(pd.notna(v) for v in am.values()) == len(ASSET_POOL)
                    and len(passed) == 0
                ),
                "will_switch": will_switch,
                **{f"am_{asset}": am[asset] for asset in ASSET_POOL},
            }
        )
        if will_switch and next_idx < len(ctx.trading_days):
            pending_weights = target
            pending_entry_idx = next_idx

    gross = (
        pd.Series(dict(gross_rows), dtype=float)
        if gross_rows
        else pd.Series(dtype=float)
    )
    bench = (
        pd.Series(dict(bench_rows), dtype=float)
        if bench_rows
        else pd.Series(dtype=float)
    )
    turnover = (
        pd.Series(dict(turnover_rows), dtype=float)
        if turnover_rows
        else pd.Series(dtype=float)
    )
    positions = pd.DataFrame(position_rows)
    if not positions.empty:
        positions = positions.set_index("date")
    state = (
        pd.Series(dict(state_rows), dtype=object)
        if state_rows
        else pd.Series(dtype=object)
    )
    evals = pd.DataFrame(eval_rows)
    return OverlayRun(
        spec=spec,
        gross=gross,
        turnover=turnover,
        positions=positions,
        benchmark_returns=bench,
        state=state,
        evals=evals,
        train_end=train_end,
    )


def _run_plain_baseline(ctx: MarketContext) -> OverlayRun:
    config = dict(ctx.config)
    strategy = load_strategy(config)
    rebalance_days = int(config.get("rebalance_days", 1))
    rebalance_mode = normalize_rebalance_mode(config.get("rebalance_mode"))
    split_idx = int(len(ctx.trading_days) * float(config.get("train_ratio", TRAIN_RATIO)))
    train_end = (
        ctx.trading_days[split_idx].date()
        if split_idx < len(ctx.trading_days)
        else ctx.trading_days[-1].date()
    )

    current_weights: dict[str, float] = {}
    current_entry_idx: int | None = None
    pending_weights: dict[str, float] | None = None
    pending_entry_idx: int | None = None

    gross_rows: list[tuple[pd.Timestamp, float]] = []
    bench_rows: list[tuple[pd.Timestamp, float]] = []
    turnover_rows: list[tuple[pd.Timestamp, float]] = []
    position_rows: list[dict[str, object]] = []
    state_rows: list[tuple[pd.Timestamp, str | None]] = []
    eval_rows: list[dict[str, object]] = []

    for day_idx, t in enumerate(ctx.trading_days):
        if day_idx > 0:
            prev_t = ctx.trading_days[day_idx - 1]
            old_weights = current_weights
            opened_today = pending_entry_idx == day_idx and pending_weights is not None
            if opened_today:
                overnight_ret = _weighted_return_between(
                    old_weights, ctx.close_prices, ctx.open_prices, prev_t, t
                )
                current_weights = pending_weights or {}
                current_entry_idx = day_idx
                position_rows.append({"date": t, **current_weights})
                executed_turnover = _turnover_between(current_weights, old_weights)
                turnover_rows.append((t, executed_turnover))
                pending_weights = None
                pending_entry_idx = None
                intraday_ret = _weighted_return_between(
                    current_weights, ctx.open_prices, ctx.close_prices, t, t
                )
                strat_ret = _chain_returns(overnight_ret, intraday_ret)
            elif current_weights:
                strat_ret = _weighted_return_between(
                    current_weights, ctx.close_prices, ctx.close_prices, prev_t, t
                )
            else:
                strat_ret = None

            if strat_ret is not None:
                gross_rows.append((t, float(strat_ret)))
                if opened_today and not old_weights:
                    bench_ret = _equal_weight_return_between(
                        ASSET_POOL, ctx.open_prices, ctx.close_prices, t, t
                    )
                else:
                    bench_ret = _equal_weight_return_between(
                        ASSET_POOL, ctx.close_prices, ctx.close_prices, prev_t, t
                    )
                if bench_ret is not None:
                    bench_rows.append((t, float(bench_ret)))
                state_rows.append((t, _state_label(current_weights)))

        holding_days = (
            day_idx - current_entry_idx + 1
            if current_entry_idx is not None and current_weights
            else None
        )
        should_signal = (
            pending_weights is None
            and not should_hold_position(
                current_weights,
                holding_days,
                rebalance_days,
                rebalance_mode,
            )
        )
        if not should_signal:
            continue

        factor_values = _factor_snapshot(ctx, t)
        target = strategy.generate_weights(
            factor_values,
            current_weights=current_weights,
        )
        next_idx = day_idx + 1
        will_switch = bool(target and target != current_weights)
        execution_date = (
            ctx.trading_days[next_idx]
            if will_switch and next_idx < len(ctx.trading_days)
            else pd.NaT
        )
        eval_rows.append(
            {
                "eval_date": t,
                "execution_date": execution_date,
                "design": "baseline_no_cash",
                "current": _state_label(current_weights),
                "base_top1": _top_asset(target),
                "target": _state_label(target),
                "will_switch": will_switch,
            }
        )
        if will_switch and next_idx < len(ctx.trading_days):
            pending_weights = target
            pending_entry_idx = next_idx

    gross = pd.Series(dict(gross_rows), dtype=float) if gross_rows else pd.Series(dtype=float)
    bench = pd.Series(dict(bench_rows), dtype=float) if bench_rows else pd.Series(dtype=float)
    turnover = (
        pd.Series(dict(turnover_rows), dtype=float)
        if turnover_rows
        else pd.Series(dtype=float)
    )
    positions = pd.DataFrame(position_rows)
    if not positions.empty:
        positions = positions.set_index("date")
    state = pd.Series(dict(state_rows), dtype=object) if state_rows else pd.Series(dtype=object)
    return OverlayRun(
        spec=OverlaySpec("baseline_no_cash", 0, "none", 0.0),
        gross=gross,
        turnover=turnover,
        positions=positions,
        benchmark_returns=bench,
        state=state,
        evals=pd.DataFrame(eval_rows),
        train_end=train_end,
    )


def _returns_for(run: OverlayRun, fee: float) -> pd.Series:
    if run.gross.empty:
        return run.gross.copy()
    costs = run.turnover.reindex(run.gross.index, fill_value=0.0) * fee
    return run.gross - costs


def _slice(returns: pd.Series, start: pd.Timestamp | None, end: pd.Timestamp | None) -> pd.Series:
    out = returns
    if start is not None:
        out = out[out.index >= start]
    if end is not None:
        out = out[out.index <= end]
    return out


def _max_drawdown(returns: pd.Series) -> float:
    if returns.empty:
        return 0.0
    cumulative = (1.0 + returns).cumprod()
    return float((cumulative / cumulative.cummax() - 1.0).min())


def _metrics(
    returns: pd.Series,
    turnover: pd.Series,
    state: pd.Series,
    train_end: date,
) -> dict[str, object]:
    n_days = int(len(returns))
    if n_days == 0:
        base = {
            "start": "",
            "end": "",
            "trading_days": 0,
            "total_return": 0.0,
            "annual_return": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "calmar": 0.0,
            "annual_turnover_sum_abs": 0.0,
            "annual_turnover_single_side": 0.0,
            "avg_holding_days": 0.0,
            "cash_time_pct": 0.0,
            "cash_exit_count": 0,
            "avg_cash_duration": 0.0,
            "train_sharpe": 0.0,
            "train_max_drawdown": 0.0,
            "test_sharpe": 0.0,
            "test_max_drawdown": 0.0,
        }
        return base

    cumulative = (1.0 + returns).cumprod()
    total_return = float(cumulative.iloc[-1] - 1.0)
    annual_return = float((1.0 + total_return) ** (252.0 / n_days) - 1.0)
    std = returns.std()
    sharpe = float(returns.mean() / std * math.sqrt(252.0)) if std > 0 else 0.0
    max_drawdown = _max_drawdown(returns)
    years = n_days / 252.0
    turnover_sum_abs = float(turnover.sum() / years) if years else 0.0
    aligned_state = state.reindex(returns.index).ffill()
    segments = _state_segments(aligned_state)
    avg_holding_days = float(np.mean([seg["days"] for seg in segments])) if segments else 0.0
    cash_segments = [seg for seg in segments if seg["state"] == "CASH"]
    cash_days = int((aligned_state == "CASH").sum())
    cash_exit_count = int(sum(1 for seg in cash_segments if seg.get("prev_state") not in {None, "CASH"}))
    avg_cash_duration = float(np.mean([seg["days"] for seg in cash_segments])) if cash_segments else 0.0
    train_end_ts = pd.Timestamp(train_end)
    train = returns[returns.index <= train_end_ts]
    test = returns[returns.index > train_end_ts]

    return {
        "start": returns.index.min().date().isoformat(),
        "end": returns.index.max().date().isoformat(),
        "trading_days": n_days,
        "total_return": total_return,
        "annual_return": annual_return,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "calmar": annual_return / abs(max_drawdown) if max_drawdown < 0 else 0.0,
        "annual_turnover_sum_abs": turnover_sum_abs,
        "annual_turnover_single_side": turnover_sum_abs / 2.0,
        "avg_holding_days": avg_holding_days,
        "cash_time_pct": cash_days / n_days,
        "cash_exit_count": cash_exit_count,
        "avg_cash_duration": avg_cash_duration,
        "train_sharpe": _sharpe(train),
        "train_max_drawdown": _max_drawdown(train),
        "test_sharpe": _sharpe(test),
        "test_max_drawdown": _max_drawdown(test),
    }


def _sharpe(returns: pd.Series) -> float:
    if returns.empty:
        return 0.0
    std = returns.std()
    return float(returns.mean() / std * math.sqrt(252.0)) if std > 0 else 0.0


def _state_segments(state: pd.Series) -> list[dict[str, object]]:
    if state.empty:
        return []
    out: list[dict[str, object]] = []
    current = None
    start = None
    prev_state = None
    dates = list(pd.DatetimeIndex(state.index))
    values = [None if pd.isna(v) else str(v) for v in state.tolist()]
    for dt, value in zip(dates, values):
        if value != current:
            if current is not None and start is not None:
                end = dates[dates.index(dt) - 1]
                out.append(
                    {
                        "state": current,
                        "start": start,
                        "end": end,
                        "days": int((state.loc[start:end] == current).sum()),
                        "prev_state": prev_state,
                    }
                )
                prev_state = current
            current = value
            start = dt
    if current is not None and start is not None:
        out.append(
            {
                "state": current,
                "start": start,
                "end": dates[-1],
                "days": int((state.loc[start:dates[-1]] == current).sum()),
                "prev_state": prev_state,
            }
        )
    return out


def _first_all_am_valid(ctx: MarketContext, window: int, threshold: str) -> str:
    for t in ctx.trading_days:
        am = _am_snapshot(ctx, t, window, threshold)
        if all(pd.notna(value) for value in am.values()):
            return t.date().isoformat()
    return ""


def _metrics_tables(
    ctx: MarketContext,
    runs: list[OverlayRun],
    baseline: OverlayRun,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    warm_rows = []
    first_valid_cache: dict[tuple[int, str], str] = {}
    for window in sorted(set([run.spec.window for run in runs])):
        for threshold in sorted(set([run.spec.threshold for run in runs if run.spec.window == window])):
            first_valid_cache[(window, threshold)] = _first_all_am_valid(ctx, window, threshold)

    for run in runs:
        first_valid = first_valid_cache[(run.spec.window, run.spec.threshold)]
        for fee in FEES:
            returns = _returns_for(run, fee)
            row = {
                "kind": "overlay",
                "design": run.spec.design,
                "window": run.spec.window,
                "threshold": run.spec.threshold,
                "cash_yield": run.spec.cash_yield,
                "fee_one_side": fee,
                "fee_bps_one_side": fee * 10000.0,
                "first_all_am_valid": first_valid,
                **_metrics(returns, run.turnover, run.state, run.train_end),
            }
            rows.append(row)
            if first_valid:
                warm = returns[returns.index >= pd.Timestamp(first_valid)]
                warm_state = run.state[run.state.index >= pd.Timestamp(first_valid)]
                warm_turnover = run.turnover[run.turnover.index >= pd.Timestamp(first_valid)]
                warm_rows.append(
                    {
                        **{
                            k: row[k]
                            for k in [
                                "kind",
                                "design",
                                "window",
                                "threshold",
                                "cash_yield",
                                "fee_one_side",
                                "fee_bps_one_side",
                                "first_all_am_valid",
                            ]
                        },
                        **_metrics(warm, warm_turnover, warm_state, run.train_end),
                    }
                )

    for fee in FEES:
        returns = _returns_for(baseline, fee)
        rows.append(
            {
                "kind": "baseline_top1",
                "design": "baseline_no_cash",
                "window": 0,
                "threshold": "none",
                "cash_yield": 0.0,
                "fee_one_side": fee,
                "fee_bps_one_side": fee * 10000.0,
                "first_all_am_valid": "",
                **_metrics(returns, baseline.turnover, baseline.state, baseline.train_end),
            }
        )
    eq_returns = baseline.benchmark_returns.copy()
    rows.append(
        {
            "kind": "equal_weight",
            "design": "equal_weight_4asset",
            "window": 0,
            "threshold": "none",
            "cash_yield": 0.0,
            "fee_one_side": 0.0,
            "fee_bps_one_side": 0.0,
            "first_all_am_valid": "",
            **_metrics(eq_returns, pd.Series(dtype=float), pd.Series("EW", index=eq_returns.index), baseline.train_end),
        }
    )
    return pd.DataFrame(rows), pd.DataFrame(warm_rows), pd.DataFrame()


def _cash_ledger(runs: list[OverlayRun], baseline: OverlayRun) -> tuple[pd.DataFrame, pd.DataFrame]:
    segment_rows = []
    summary_rows = []
    baseline_gross = baseline.gross
    baseline_net = _returns_for(baseline, BASE_FEE)

    for run in runs:
        state = run.state.reindex(run.gross.index).ffill()
        segments = [seg for seg in _state_segments(state) if seg["state"] == "CASH"]
        avoided_sum = 0.0
        missed_sum = 0.0
        for idx, seg in enumerate(segments, start=1):
            start = pd.Timestamp(seg["start"])
            end = pd.Timestamp(seg["end"])
            gross = baseline_gross[(baseline_gross.index >= start) & (baseline_gross.index <= end)]
            net = baseline_net[(baseline_net.index >= start) & (baseline_net.index <= end)]
            base_ret = float((1.0 + gross).prod() - 1.0) if len(gross) else 0.0
            base_net_ret = float((1.0 + net).prod() - 1.0) if len(net) else 0.0
            avoided = max(-base_ret, 0.0)
            missed = max(base_ret, 0.0)
            avoided_sum += avoided
            missed_sum += missed
            segment_rows.append(
                {
                    "design": run.spec.design,
                    "window": run.spec.window,
                    "threshold": run.spec.threshold,
                    "cash_yield": run.spec.cash_yield,
                    "segment_id": idx,
                    "start": start.date().isoformat(),
                    "end": end.date().isoformat(),
                    "trading_days": int(seg["days"]),
                    "prev_state": seg.get("prev_state"),
                    "baseline_gross_return": base_ret,
                    "baseline_net_1bp_return": base_net_ret,
                    "avoided_loss": avoided,
                    "missed_gain": missed,
                }
            )
        summary_rows.append(
            {
                "design": run.spec.design,
                "window": run.spec.window,
                "threshold": run.spec.threshold,
                "cash_yield": run.spec.cash_yield,
                "cash_days": int((state == "CASH").sum()),
                "cash_segments": int(len(segments)),
                "avoided_loss_sum": avoided_sum,
                "missed_gain_sum": missed_sum,
                "avoidance_minus_missed": avoided_sum - missed_sum,
                "fragile_n_lt_5": bool(len(segments) < 5),
            }
        )
    return pd.DataFrame(segment_rows), pd.DataFrame(summary_rows)


def _bad_window_tables(runs: list[OverlayRun], baseline: OverlayRun) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    curve_rows = []
    base_net = _returns_for(baseline, BASE_FEE)
    for run in runs:
        overlay_net = _returns_for(run, BASE_FEE)
        state = run.state.reindex(overlay_net.index).ffill()
        for name, start, end in BAD_WINDOWS:
            base_slice = _slice(base_net, start, end)
            overlay_slice = _slice(overlay_net, start, end)
            overlay_state = state[(state.index >= start) & (state.index <= end)]
            base_dd = _max_drawdown(base_slice)
            overlay_dd = _max_drawdown(overlay_slice)
            rows.append(
                {
                    "window_name": name,
                    "start": start.date().isoformat(),
                    "end": end.date().isoformat(),
                    "design": run.spec.design,
                    "window": run.spec.window,
                    "threshold": run.spec.threshold,
                    "cash_yield": run.spec.cash_yield,
                    "baseline_return": float((1.0 + base_slice).prod() - 1.0) if len(base_slice) else 0.0,
                    "overlay_return": float((1.0 + overlay_slice).prod() - 1.0) if len(overlay_slice) else 0.0,
                    "baseline_max_drawdown": base_dd,
                    "overlay_max_drawdown": overlay_dd,
                    "drawdown_depth_delta": overlay_dd - base_dd,
                    "cash_days": int((overlay_state == "CASH").sum()),
                    "cash_segments": int(
                        len([seg for seg in _state_segments(overlay_state) if seg["state"] == "CASH"])
                    ),
                    "states_seen": ",".join(sorted({str(v) for v in overlay_state.dropna().unique()})),
                }
            )
            common = sorted(set(base_slice.index) & set(overlay_slice.index))
            if common:
                b_cum = (1.0 + base_slice.loc[common]).cumprod()
                o_cum = (1.0 + overlay_slice.loc[common]).cumprod()
                b_dd_series = b_cum / b_cum.cummax() - 1.0
                o_dd_series = o_cum / o_cum.cummax() - 1.0
                for dt in common:
                    curve_rows.append(
                        {
                            "window_name": name,
                            "date": pd.Timestamp(dt).date().isoformat(),
                            "design": run.spec.design,
                            "window": run.spec.window,
                            "threshold": run.spec.threshold,
                            "cash_yield": run.spec.cash_yield,
                            "baseline_drawdown": float(b_dd_series.loc[dt]),
                            "overlay_drawdown": float(o_dd_series.loc[dt]),
                            "overlay_state": state.get(dt, None),
                        }
                    )
    return pd.DataFrame(rows), pd.DataFrame(curve_rows)


def _decision_for_day(
    ctx: MarketContext,
    strategy,
    design: str,
    window: int,
    threshold: str,
    t: pd.Timestamp,
) -> tuple[str | None, dict[str, object]]:
    factor_values = _factor_snapshot(ctx, t)
    base_weights = strategy.generate_weights(factor_values)
    am = _am_snapshot(ctx, t, window, threshold)
    passed = _passes(am)
    if design == "design1_rank_then_gate":
        winner = _top_asset(base_weights)
        target = _state_label(base_weights) if winner in passed else "CASH"
    else:
        if passed:
            target = _state_label(strategy.generate_weights({a: v for a, v in factor_values.items() if a in passed}))
        else:
            target = "CASH"
    return target, {
        "base_top1": _top_asset(base_weights),
        "am_valid_count": int(sum(pd.notna(v) for v in am.values())),
        "passes_count": int(len(passed)),
        "all_four_negative": bool(
            sum(pd.notna(v) for v in am.values()) == len(ASSET_POOL) and len(passed) == 0
        ),
        "top1_passes": bool(_top_asset(base_weights) in passed),
    }


def _divergence_table(ctx: MarketContext, runs: list[OverlayRun]) -> pd.DataFrame:
    rows = []
    strategy = load_strategy(dict(ctx.config))
    by_key = {
        (run.spec.design, run.spec.window, run.spec.threshold, run.spec.cash_yield): run
        for run in runs
    }
    for threshold, windows in [("point", WINDOWS), ("sma", SMA_WINDOWS)]:
        for window in windows:
            decision_rows = []
            for t in ctx.trading_days:
                d1, meta = _decision_for_day(
                    ctx, strategy, "design1_rank_then_gate", window, threshold, t
                )
                d2, _ = _decision_for_day(
                    ctx, strategy, "design2_filter_then_rank", window, threshold, t
                )
                if meta["base_top1"] is None:
                    continue
                decision_rows.append({**meta, "design1_target": d1, "design2_target": d2})
            df = pd.DataFrame(decision_rows)
            if df.empty:
                continue
            valid4 = df[df["am_valid_count"] == len(ASSET_POOL)]
            actual_diff_pct = np.nan
            d1_run = by_key.get(("design1_rank_then_gate", window, threshold, 0.0))
            d2_run = by_key.get(("design2_filter_then_rank", window, threshold, 0.0))
            if d1_run is not None and d2_run is not None:
                common = sorted(set(d1_run.state.index) & set(d2_run.state.index))
                if common:
                    actual_diff_pct = float(
                        (d1_run.state.loc[common].astype(str) != d2_run.state.loc[common].astype(str)).mean()
                    )
            rows.append(
                {
                    "threshold": threshold,
                    "window": window,
                    "decision_days": int(len(df)),
                    "all_four_valid_days": int(len(valid4)),
                    "all_four_am_negative_days": int(valid4["all_four_negative"].sum()),
                    "all_four_am_negative_pct": float(valid4["all_four_negative"].mean()) if len(valid4) else 0.0,
                    "top1_pass_days": int(df["top1_passes"].sum()),
                    "top1_pass_pct": float(df["top1_passes"].mean()),
                    "raw_design_diff_days": int((df["design1_target"] != df["design2_target"]).sum()),
                    "raw_design_diff_pct": float((df["design1_target"] != df["design2_target"]).mean()),
                    "actual_state_diff_pct_cash0": actual_diff_pct,
                }
            )
    return pd.DataFrame(rows)


def _data_gate(ctx: MarketContext) -> pd.DataFrame:
    rows = []
    for asset in ASSET_POOL:
        df = ctx.asset_data[asset]
        hist = ctx.history_data[asset]
        rows.append(
            {
                "asset": asset,
                "first_date": pd.Timestamp(hist["date"].min()).date().isoformat(),
                "backtest_first_date": pd.Timestamp(df["date"].min()).date().isoformat(),
                "last_date": pd.Timestamp(df["date"].max()).date().isoformat(),
                "rows": int(len(df)),
            }
        )
    for threshold, windows in [("point", WINDOWS), ("sma", SMA_WINDOWS)]:
        for window in windows:
            rows.append(
                {
                    "asset": "ALL_ASSETS_AM_VALID",
                    "first_date": _first_all_am_valid(ctx, window, threshold),
                    "last_date": "",
                    "rows": 0,
                    "threshold": threshold,
                    "window": window,
                }
            )
    return pd.DataFrame(rows)


def _fmt(value: object, column: str) -> str:
    if isinstance(value, str):
        return value
    if value is None or pd.isna(value):
        return ""
    if isinstance(value, (bool, np.bool_)):
        return "yes" if bool(value) else "no"
    if column in {
        "total_return",
        "annual_return",
        "max_drawdown",
        "train_max_drawdown",
        "test_max_drawdown",
        "annual_turnover_sum_abs",
        "annual_turnover_single_side",
        "cash_time_pct",
        "baseline_return",
        "overlay_return",
        "baseline_max_drawdown",
        "overlay_max_drawdown",
        "drawdown_depth_delta",
        "avoided_loss_sum",
        "missed_gain_sum",
        "avoidance_minus_missed",
        "all_four_am_negative_pct",
        "top1_pass_pct",
        "raw_design_diff_pct",
        "actual_state_diff_pct_cash0",
        "cash_yield",
    }:
        return f"{float(value):.2%}"
    if column in {"fee_bps_one_side", "sharpe", "train_sharpe", "test_sharpe", "calmar", "avg_holding_days", "avg_cash_duration"}:
        return f"{float(value):.2f}"
    if column in {"trading_days", "cash_exit_count", "cash_days", "cash_segments", "window", "decision_days", "all_four_valid_days", "all_four_am_negative_days", "top1_pass_days", "raw_design_diff_days"}:
        return str(int(value))
    if isinstance(value, float):
        return f"{value:.10g}"
    return str(value)


def _markdown_table(df: pd.DataFrame, columns: list[str], max_rows: int | None = None) -> str:
    show = df.loc[:, columns].copy()
    if max_rows is not None:
        show = show.head(max_rows)
    lines = ["| " + " | ".join(columns) + " |"]
    lines.append("| " + " | ".join("---" for _ in columns) + " |")
    for _, row in show.iterrows():
        lines.append("| " + " | ".join(_fmt(row[col], col) for col in columns) + " |")
    return "\n".join(lines)


def _write_outputs(
    ctx: MarketContext,
    metrics: pd.DataFrame,
    warmup: pd.DataFrame,
    ledger: pd.DataFrame,
    ledger_summary: pd.DataFrame,
    bad: pd.DataFrame,
    curves: pd.DataFrame,
    divergence: pd.DataFrame,
    data_gate: pd.DataFrame,
) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(OUT_DIR / f"{PREFIX}_surface_metrics.csv", index=False, encoding="utf-8-sig")
    warmup.to_csv(OUT_DIR / f"{PREFIX}_warmup_trimmed_metrics.csv", index=False, encoding="utf-8-sig")
    ledger.to_csv(OUT_DIR / f"{PREFIX}_cash_segment_ledger.csv", index=False, encoding="utf-8-sig")
    ledger_summary.to_csv(OUT_DIR / f"{PREFIX}_trigger_stats.csv", index=False, encoding="utf-8-sig")
    bad.to_csv(OUT_DIR / f"{PREFIX}_bad_window_attribution.csv", index=False, encoding="utf-8-sig")
    curves.to_csv(OUT_DIR / f"{PREFIX}_bad_window_curves.csv", index=False, encoding="utf-8-sig")
    divergence.to_csv(OUT_DIR / f"{PREFIX}_design_divergence.csv", index=False, encoding="utf-8-sig")
    data_gate.to_csv(OUT_DIR / f"{PREFIX}_data_gate.csv", index=False, encoding="utf-8-sig")

    main_1bp = metrics[
        (metrics["kind"] == "overlay")
        & (metrics["threshold"] == "point")
        & (metrics["fee_bps_one_side"] == 1.0)
    ].sort_values(["design", "cash_yield", "window"])
    sma_1bp = metrics[
        (metrics["kind"] == "overlay")
        & (metrics["threshold"] == "sma")
        & (metrics["fee_bps_one_side"] == 1.0)
    ].sort_values(["design", "cash_yield", "window"])
    baseline_rows = metrics[
        (metrics["kind"].isin(["baseline_top1", "equal_weight"]))
        & (metrics["fee_bps_one_side"].isin([0.0, 1.0]))
    ]
    top_cols = [
        "kind",
        "design",
        "window",
        "threshold",
        "cash_yield",
        "fee_bps_one_side",
        "annual_return",
        "sharpe",
        "max_drawdown",
        "calmar",
        "annual_turnover_single_side",
        "avg_holding_days",
        "cash_time_pct",
        "cash_exit_count",
        "avg_cash_duration",
        "train_sharpe",
        "test_sharpe",
        "train_max_drawdown",
        "test_max_drawdown",
        "first_all_am_valid",
    ]
    bad_show = bad[
        (bad["threshold"] == "point")
        & (bad["cash_yield"] == 0.0)
        & (bad["window"].isin([20, 60, 200]))
    ].sort_values(["window_name", "design", "window"])
    trigger_show = ledger_summary[
        (ledger_summary["threshold"] == "point")
        & (ledger_summary["cash_yield"] == 0.0)
    ].sort_values(["design", "window"])

    lines = [
        "# Absolute Momentum Cash Overlay Diagnostic",
        "",
        f"- Run date: {date.today().isoformat()}",
        f"- Config base: `{CONFIG_PATH.relative_to(REPO_ROOT)}`; in-memory overrides only: `start=2014-01-01`, `rebalance_days=5`, `train_ratio=0.7`, `transaction_cost_rate=0`.",
        "- Scope: Mode C research overlay scan only; no live/backfill, no production YAML, no changelog edits.",
        "- Ranking signal: unchanged `quality_momentum(window=20) = 20-day momentum x ER`.",
        "- AM signal history: uses local listing-history bars before 2014 when available; return accounting and reported metrics still start from 2014-01-01.",
        "- Overlay timing: target generated by Top1 at signal close, then absolute-momentum overlay is applied before pending T+1 open accounting.",
        "- Cash representation: full zero vector over the four assets. Therefore `Σ|Δw|` naturally charges 1.0 when selling to cash and 1.0 when buying back; one complete cash round-trip costs `2 x one-side fee`, equal to a full asset-to-asset switch.",
        "- Cash yields: 0% and 2% annual, daily compounded as `(1+yield)^(1/252)-1`.",
        "- Cost grid: 1, 3, 5, 10 bps one-side, applied after each trajectory from gross returns and executed turnover.",
        "",
        "## Data And Warmup Gate",
        "",
        _markdown_table(data_gate, list(data_gate.columns)),
        "",
        "## Baselines",
        "",
        _markdown_table(baseline_rows, top_cols),
        "",
        "## Main Point-to-Point Grid At 1bp",
        "",
        _markdown_table(main_1bp, top_cols),
        "",
        "## SMA Robustness Subset At 1bp",
        "",
        _markdown_table(sma_1bp, top_cols),
        "",
        "## Warmup-Trimmed Robustness",
        "",
        "The raw CSV recomputes every overlay config after dropping dates before all four AM signals are valid. Compare this table with the main grid to check that long-window conclusions are not driven by the 2014 partial-valid warmup.",
        "",
        _markdown_table(
            warmup[
                (warmup["threshold"] == "point")
                & (warmup["fee_bps_one_side"] == 1.0)
                & (warmup["cash_yield"] == 0.0)
            ].sort_values(["design", "window"]),
            top_cols,
        ),
        "",
        "## Trigger Stats And Cash Ledger Summary",
        "",
        _markdown_table(
            trigger_show,
            [
                "design",
                "window",
                "threshold",
                "cash_yield",
                "cash_days",
                "cash_segments",
                "avoided_loss_sum",
                "missed_gain_sum",
                "avoidance_minus_missed",
                "fragile_n_lt_5",
            ],
        ),
        "",
        "Cash ledger convention: for each cash segment, the no-cash Top1 baseline return over the same dates is split into avoided loss when negative and missed gain when positive. This is a simple segment-sum attribution; portfolio compounding, cash yield, and execution costs remain in the return metrics.",
        "",
        "## Bad Window Attribution At 1bp",
        "",
        _markdown_table(
            bad_show,
            [
                "window_name",
                "design",
                "window",
                "threshold",
                "cash_yield",
                "baseline_return",
                "overlay_return",
                "baseline_max_drawdown",
                "overlay_max_drawdown",
                "drawdown_depth_delta",
                "cash_days",
                "cash_segments",
                "states_seen",
            ],
        ),
        "",
        "The full drawdown overlay curves for 2020Q1, 2022, and 2024-10-08 to 2024-11-15 are archived as CSV for plotting.",
        "",
        "## Design Divergence",
        "",
        _markdown_table(divergence, list(divergence.columns)),
        "",
        "## Raw CSV",
        "",
        f"- `{PREFIX}_surface_metrics.csv`",
        f"- `{PREFIX}_warmup_trimmed_metrics.csv`",
        f"- `{PREFIX}_cash_segment_ledger.csv`",
        f"- `{PREFIX}_trigger_stats.csv`",
        f"- `{PREFIX}_bad_window_attribution.csv`",
        f"- `{PREFIX}_bad_window_curves.csv`",
        f"- `{PREFIX}_design_divergence.csv`",
        f"- `{PREFIX}_data_gate.csv`",
        "",
    ]
    report = OUT_DIR / f"{PREFIX}.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    return report


def _run_all() -> tuple[MarketContext, list[OverlayRun], OverlayRun]:
    ctx = _build_context()
    runs: list[OverlayRun] = []
    specs = []
    for design in ["design2_filter_then_rank", "design1_rank_then_gate"]:
        for window in WINDOWS:
            for cash_yield in CASH_YIELDS:
                specs.append(OverlaySpec(design, window, "point", cash_yield))
        for window in SMA_WINDOWS:
            for cash_yield in CASH_YIELDS:
                specs.append(OverlaySpec(design, window, "sma", cash_yield))
    for spec in specs:
        runs.append(_run_overlay(ctx, spec))
    baseline = _run_plain_baseline(ctx)
    return ctx, runs, baseline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-only", action="store_true")
    args = parser.parse_args()

    ctx, runs, baseline = _run_all()
    metrics, warmup, _ = _metrics_tables(ctx, runs, baseline)
    ledger, ledger_summary = _cash_ledger(runs, baseline)
    bad, curves = _bad_window_tables(runs, baseline)
    divergence = _divergence_table(ctx, runs)
    data_gate = _data_gate(ctx)
    report = _write_outputs(
        ctx,
        metrics,
        warmup,
        ledger,
        ledger_summary,
        bad,
        curves if not args.metrics_only else curves.head(0),
        divergence,
        data_gate,
    )
    print(f"REPORT {report}")


if __name__ == "__main__":
    main()
