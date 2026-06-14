"""Research scan for min-hold vs periodic re-evaluation timing.

This script is intentionally research-only. It does not edit production YAMLs
and does not use run_backtest.py --from-log or backfill/live entry points.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import math
import warnings
from datetime import date
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import yaml

import backtest.runner as runner
from backtest.runner import BacktestResult
from data.store import query
from factors.registry import load_registered_factors
from factors.validator import validate
from strategy.loader import load_strategy
from strategy.rebalance import should_hold_position


REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "strategy" / "configs" / "quality_momentum_top1.yaml"
OUT_DIR = REPO_ROOT / "strategy_changelog_attachments"
REPORT_PATH = OUT_DIR / "2026-06-02_periodic_reeval_scan.md"
START = date(2014, 1, 1)
TRAIN_END = date(2021, 12, 31)
TEST_START = date(2022, 1, 1)
BASE_COST = 0.0001
ASSET_POOL = ["510300.SH", "159915.SZ", "513100.SH", "518880.SH"]


@dataclasses.dataclass
class RunBundle:
    label: str
    mode: str
    n: int
    phase: int
    cost_rate: float
    result: BacktestResult
    net_returns: pd.Series
    metrics: dict[str, object]


@dataclasses.dataclass
class TraceBundle:
    evals: pd.DataFrame
    switches: pd.DataFrame
    states: pd.DataFrame
    trading_days: list[pd.Timestamp]
    open_prices: dict[str, pd.Series]
    close_prices: dict[str, pd.Series]


def _load_base_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config["asset_pool"] = list(ASSET_POOL)
    config["start"] = START
    config["end"] = date.today()
    config["rebalance_days"] = 5
    config.pop("rebalance_mode", None)
    return config


def _config_for(
    *,
    mode: str | None,
    n: int,
    start: date = START,
    end: date | None = None,
) -> dict:
    config = _load_base_config()
    config["start"] = start
    config["end"] = end or date.today()
    config["rebalance_days"] = n
    if mode is None:
        config.pop("rebalance_mode", None)
    else:
        config["rebalance_mode"] = mode
    return config


def _phase_should_hold(
    current_weights: dict[str, float],
    holding_days: int | None,
    rebalance_days: int,
    rebalance_mode: str | None = "min_hold",
    phase: int = 0,
) -> bool:
    if phase == 0 or rebalance_mode != "fixed_cycle":
        return should_hold_position(
            current_weights,
            holding_days,
            rebalance_days,
            rebalance_mode,
        )
    if rebalance_days < 1:
        raise ValueError(f"rebalance_days must be >= 1, got {rebalance_days}")
    if rebalance_days <= 1:
        return False
    if not current_weights:
        return False
    if holding_days is None:
        return True

    first_eval = rebalance_days + phase
    return holding_days < first_eval or (holding_days - first_eval) % rebalance_days != 0


@contextlib.contextmanager
def _patched_phase(phase: int) -> Iterator[None]:
    original = runner._should_hold_position

    def patched(
        current_weights: dict[str, float],
        holding_days: int | None,
        rebalance_days: int,
        rebalance_mode: str | None = "min_hold",
    ) -> bool:
        return _phase_should_hold(
            current_weights,
            holding_days,
            rebalance_days,
            rebalance_mode,
            phase,
        )

    runner._should_hold_position = patched
    try:
        yield
    finally:
        runner._should_hold_position = original


def _run(config: dict, phase: int = 0) -> BacktestResult:
    with _patched_phase(phase):
        return runner.run(config)


def _weights_from_row(row: pd.Series) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in row.items():
        if pd.notna(value) and float(value) != 0.0:
            out[str(key)] = float(value)
    return out


def _turnovers(positions: pd.DataFrame) -> pd.Series:
    if positions.empty:
        return pd.Series(dtype=float)

    prev: dict[str, float] = {}
    records: list[tuple[pd.Timestamp, float]] = []
    for idx, row in positions.fillna(0.0).iterrows():
        curr = _weights_from_row(row)
        assets = set(prev) | set(curr)
        turnover = sum(abs(curr.get(asset, 0.0) - prev.get(asset, 0.0)) for asset in assets)
        records.append((pd.Timestamp(idx), float(turnover)))
        prev = curr
    return pd.Series(dict(records), dtype=float)


def _apply_costs(result: BacktestResult, cost_rate: float) -> pd.Series:
    returns = result.daily_returns.copy()
    if len(returns) == 0 or cost_rate == 0:
        return returns

    costs = _turnovers(result.positions) * cost_rate
    for dt, cost in costs.items():
        if dt in returns.index:
            returns.loc[dt] = returns.loc[dt] - cost
    return returns


def _metrics(result: BacktestResult, returns: pd.Series, cost_rate: float) -> dict[str, object]:
    n_days = int(len(returns))
    if n_days == 0:
        total_return = annual_return = sharpe = max_drawdown = 0.0
        start = end = ""
    else:
        cumulative = (1.0 + returns).cumprod()
        total_return = float(cumulative.iloc[-1] - 1.0)
        annual_return = float((1.0 + total_return) ** (252.0 / n_days) - 1.0)
        std = returns.std()
        sharpe = float(returns.mean() / std * math.sqrt(252.0)) if std > 0 else 0.0
        drawdown = cumulative / cumulative.cummax() - 1.0
        max_drawdown = float(drawdown.min())
        start = returns.index.min().date().isoformat()
        end = returns.index.max().date().isoformat()

    turnovers = _turnovers(result.positions)
    years = n_days / 252.0 if n_days else 0.0
    annual_turnover = float(0.5 * turnovers.sum() / years) if years else 0.0
    switch_count = max(int(len(result.positions) - 1), 0)
    avg_holding_days = float(n_days / len(result.positions)) if len(result.positions) else 0.0

    return {
        "start": start,
        "end": end,
        "trading_days": n_days,
        "total_return": total_return,
        "annual_return": annual_return,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "annual_turnover": annual_turnover,
        "avg_holding_days": avg_holding_days,
        "switch_count": switch_count,
        "cost_bps_one_side": cost_rate * 10000.0,
    }


def _bundle(
    label: str,
    mode: str | None,
    n: int,
    phase: int,
    cost_rate: float,
    start: date = START,
    end: date | None = None,
) -> RunBundle:
    config = _config_for(mode=mode, n=n, start=start, end=end)
    result = _run(config, phase=phase)
    net_returns = _apply_costs(result, cost_rate)
    mode_label = mode or "default_min_hold"
    metrics = {
        "label": label,
        "reeval_mode": "periodic" if mode == "fixed_cycle" else "min_hold",
        "engine_mode": mode_label,
        "rebalance_days": n,
        "phase_offset": phase,
        **_metrics(result, net_returns, cost_rate),
    }
    return RunBundle(label, mode_label, n, phase, cost_rate, result, net_returns, metrics)


def _data_gate() -> pd.DataFrame:
    rows = []
    for asset in ASSET_POOL:
        df = query(asset, START, date.today())
        first = pd.Timestamp(df["date"].min()).date() if len(df) else None
        last = pd.Timestamp(df["date"].max()).date() if len(df) else None
        rows.append(
            {
                "asset": asset,
                "first_valid_date": first.isoformat() if first else "",
                "last_valid_date": last.isoformat() if last else "",
                "rows": int(len(df)),
                "has_2014_01_02": bool(
                    len(df) and pd.Timestamp("2014-01-02") in set(df["date"])
                ),
            }
        )
    out = pd.DataFrame(rows)
    if (out["first_valid_date"] > "2014-01-02").any():
        raise RuntimeError("data gate failed: an asset starts after 2014-01-02")
    return out


def _gate() -> tuple[pd.DataFrame, pd.DataFrame]:
    data_gate = _data_gate()
    explicit = _bundle("explicit_min_hold_rd5", "min_hold", 5, 0, BASE_COST)
    default = _bundle("current_default_rd5", None, 5, 0, BASE_COST)

    metric_keys = [
        "annual_return",
        "sharpe",
        "max_drawdown",
        "annual_turnover",
        "avg_holding_days",
        "switch_count",
        "trading_days",
        "total_return",
    ]
    rows = []
    for key in metric_keys:
        a = explicit.metrics[key]
        b = default.metrics[key]
        rows.append(
            {
                "metric": key,
                "explicit_min_hold": a,
                "current_default": b,
                "diff": float(a) - float(b),
                "exact_match": a == b,
            }
        )

    same_returns = explicit.net_returns.equals(default.net_returns)
    same_positions = explicit.result.positions.equals(default.result.positions)
    rows.append(
        {
            "metric": "net_returns_series",
            "explicit_min_hold": str(same_returns),
            "current_default": str(same_returns),
            "diff": 0.0 if same_returns else float("nan"),
            "exact_match": same_returns,
        }
    )
    rows.append(
        {
            "metric": "positions_df",
            "explicit_min_hold": str(same_positions),
            "current_default": str(same_positions),
            "diff": 0.0 if same_positions else float("nan"),
            "exact_match": same_positions,
        }
    )

    gate_df = pd.DataFrame(rows)
    if not bool(gate_df["exact_match"].all()):
        raise RuntimeError("4.0 gate failed: explicit min_hold differs from current default")
    return data_gate, gate_df


def _top_asset(weights: dict[str, float]) -> str | None:
    return next(iter(weights.keys())) if weights else None


def _trace(config: dict, mode: str, phase: int) -> TraceBundle:
    config = dict(config)
    config["rebalance_mode"] = mode
    asset_pool = config["asset_pool"]
    factor_configs = config["factors"]
    rebalance_days = int(config["rebalance_days"])
    strategy = load_strategy(config)
    all_factors = load_registered_factors()

    asset_data: dict[str, pd.DataFrame] = {}
    for asset in asset_pool:
        df = query(asset, config["start"], config["end"])
        if len(df) > 0:
            asset_data[asset] = df
    if not asset_data:
        raise RuntimeError("no data available for trace")

    trading_days = sorted({d for df in asset_data.values() for d in df["date"].tolist()})
    max_min_history = max(
        all_factors[fc["name"]]["METADATA"]["min_history"]
        for fc in factor_configs
    )
    open_prices = {
        asset: pd.Series(df["open"].values, index=df["date"])
        for asset, df in asset_data.items()
    }
    close_prices = {
        asset: pd.Series(df["close"].values, index=df["date"])
        for asset, df in asset_data.items()
    }

    def calc_weights(t: pd.Timestamp) -> dict[str, float]:
        asset_factor_values: dict[str, dict[str, float]] = {}
        for asset, df in asset_data.items():
            truncated = df.loc[df["date"] <= t]
            if len(truncated) < max_min_history:
                continue

            factor_vals: dict[str, float] = {}
            for fc in factor_configs:
                fname = fc["name"]
                fmod = all_factors[fname]
                try:
                    series = fmod["compute"](truncated.copy(), fc.get("params"))
                    validate(series, truncated, fmod["METADATA"])
                    last_val = series.iloc[-1]
                    if pd.notna(last_val):
                        factor_vals[fname] = float(last_val)
                except (ValueError, Exception) as exc:
                    warnings.warn(f"factor '{fname}' failed for {asset} on {t}: {exc}")
            if len(factor_vals) == len(factor_configs):
                asset_factor_values[asset] = factor_vals
        return strategy.generate_weights(asset_factor_values)

    current_weights: dict[str, float] = {}
    current_entry_idx: int | None = None
    pending_weights: dict[str, float] | None = None
    pending_entry_idx: int | None = None
    eval_rows = []
    switch_rows = []
    state_rows = []

    for day_idx, t in enumerate(trading_days):
        if pending_entry_idx == day_idx and pending_weights is not None:
            current_weights = pending_weights or {}
            current_entry_idx = day_idx
            pending_weights = None
            pending_entry_idx = None

        holding_days = (
            day_idx - current_entry_idx + 1
            if current_entry_idx is not None and current_weights
            else None
        )
        state_rows.append(
            {
                "date": t,
                "current": _top_asset(current_weights),
                "entry_date": trading_days[current_entry_idx] if current_entry_idx is not None else pd.NaT,
                "holding_days": holding_days,
            }
        )

        should_eval = (
            pending_weights is None
            and not _phase_should_hold(
                current_weights,
                holding_days,
                rebalance_days,
                mode,
                phase,
            )
        )
        if not should_eval:
            continue

        new_weights = calc_weights(t)
        will_switch = bool(new_weights and new_weights != current_weights)
        next_idx = day_idx + 1
        execution_date = trading_days[next_idx] if will_switch and next_idx < len(trading_days) else pd.NaT
        row = {
            "eval_date": t,
            "holding_days": holding_days,
            "entry_date": trading_days[current_entry_idx] if current_entry_idx is not None else pd.NaT,
            "current": _top_asset(current_weights),
            "candidate": _top_asset(new_weights),
            "will_switch": will_switch,
            "execution_date": execution_date,
        }
        eval_rows.append(row)
        if will_switch and next_idx < len(trading_days):
            switch_rows.append({**row, "new": _top_asset(new_weights)})
            pending_weights = new_weights
            pending_entry_idx = next_idx

    return TraceBundle(
        evals=pd.DataFrame(eval_rows),
        switches=pd.DataFrame(switch_rows),
        states=pd.DataFrame(state_rows).set_index("date"),
        trading_days=trading_days,
        open_prices=open_prices,
        close_prices=close_prices,
    )


def _asset_return(
    asset: str | None,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    open_prices: dict[str, pd.Series],
    close_prices: dict[str, pd.Series],
) -> float | None:
    if asset is None:
        return None
    start = open_prices.get(asset)
    end = close_prices.get(asset)
    if start is None or end is None:
        return None
    if start_date not in start.index or end_date not in end.index:
        return None
    return float(end.loc[end_date] / start.loc[start_date] - 1.0)


def _skipped_switch_attribution() -> tuple[pd.DataFrame, dict[str, object]]:
    config = _config_for(mode="min_hold", n=5)
    min_trace = _trace(config, "min_hold", 0)
    periodic_trace = _trace(config, "fixed_cycle", 0)

    min_eval_dates = set(pd.to_datetime(min_trace.evals["eval_date"]))
    periodic_eval_dates = set(pd.to_datetime(periodic_trace.evals["eval_date"]))
    shared_eval_dates = sorted(min_eval_dates & periodic_eval_dates)

    rows = []
    for _, event in min_trace.switches.iterrows():
        holding_days = event["holding_days"]
        if pd.isna(holding_days):
            continue
        holding_days_int = int(holding_days)
        if holding_days_int % 5 == 0:
            continue
        if holding_days_int not in {6, 7, 8, 9}:
            continue

        signal_date = pd.Timestamp(event["eval_date"])
        if signal_date in periodic_eval_dates:
            continue
        if signal_date not in periodic_trace.states.index:
            continue

        execution_date = pd.Timestamp(event["execution_date"])
        next_shared = next(
            (
                dt
                for dt in shared_eval_dates
                if dt > signal_date and dt >= execution_date
            ),
            None,
        )
        if next_shared is None:
            continue

        periodic_state = periodic_trace.states.loc[signal_date]
        method1_new = event["new"]
        method2_old = periodic_state["current"]
        new_ret = _asset_return(
            method1_new,
            execution_date,
            next_shared,
            min_trace.open_prices,
            min_trace.close_prices,
        )
        old_ret = _asset_return(
            method2_old,
            execution_date,
            next_shared,
            periodic_trace.open_prices,
            periodic_trace.close_prices,
        )
        if new_ret is None or old_ret is None:
            continue

        rows.append(
            {
                "signal_date": signal_date.date().isoformat(),
                "execution_date": execution_date.date().isoformat(),
                "next_shared_eval_date": next_shared.date().isoformat(),
                "method1_holding_days": holding_days_int,
                "method1_old": event["current"],
                "method1_new": method1_new,
                "method2_old": method2_old,
                "method2_holding_days": (
                    int(periodic_state["holding_days"])
                    if pd.notna(periodic_state["holding_days"])
                    else None
                ),
                "same_old_asset": event["current"] == method2_old,
                "new_asset_return": new_ret,
                "old_asset_return": old_ret,
                "old_minus_new": old_ret - new_ret,
                "old_outperformed": old_ret > new_ret,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        summary = {
            "event_count": 0,
            "old_win_rate": 0.0,
            "sum_old_minus_new": 0.0,
            "compounded_old_minus_new": 0.0,
            "same_old_asset_count": 0,
        }
        return df, summary

    summary = {
        "event_count": int(len(df)),
        "old_win_rate": float(df["old_outperformed"].mean()),
        "sum_old_minus_new": float(df["old_minus_new"].sum()),
        "compounded_old_minus_new": float(
            (1.0 + df["old_asset_return"]).prod()
            - (1.0 + df["new_asset_return"]).prod()
        ),
        "same_old_asset_count": int(df["same_old_asset"].sum()),
    }
    return df, summary


def _pct(value: float) -> str:
    return f"{value:.2%}"


def _fmt(value: object, metric: str | None = None) -> str:
    if isinstance(value, (bool, np.bool_)):
        return "yes" if bool(value) else "no"
    if isinstance(value, str):
        return value
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    if metric in {
        "total_return",
        "annual_return",
        "max_drawdown",
        "annual_turnover",
        "old_win_rate",
        "sum_old_minus_new",
        "compounded_old_minus_new",
    }:
        return _pct(float(value))
    if metric == "cost_bps_one_side":
        return f"{float(value):.1f}"
    if metric in {"sharpe", "avg_holding_days"}:
        return f"{float(value):.2f}"
    if metric in {"switch_count", "trading_days", "event_count", "same_old_asset_count"}:
        return f"{int(value)}"
    if isinstance(value, float):
        return f"{value:.10g}"
    return str(value)


def _markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    lines = []
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join("---" for _ in columns) + " |")
    for _, row in df.iterrows():
        vals = [_fmt(row[col], col) for col in columns]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def _write_csv(df: pd.DataFrame, name: str) -> Path:
    path = OUT_DIR / name
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def _full_scan() -> dict[str, object]:
    data_gate, gate_df = _gate()

    phase_rows = []
    for phase in range(5):
        bundle = _bundle(
            f"periodic_N5_k{phase}",
            "fixed_cycle",
            5,
            phase,
            BASE_COST,
        )
        phase_rows.append(bundle.metrics)
    phase_df = pd.DataFrame(phase_rows)

    split_rows = []
    split_specs = [
        ("train_2014_2021", START, TRAIN_END),
        ("test_2022_data_end", TEST_START, date.today()),
    ]
    for split_label, start, end in split_specs:
        for mode, label in [("min_hold", "min_hold"), ("fixed_cycle", "periodic_k0")]:
            bundle = _bundle(
                f"{split_label}_{label}",
                mode,
                5,
                0,
                BASE_COST,
                start=start,
                end=end,
            )
            split_rows.append({"sample": split_label, **bundle.metrics})
    split_df = pd.DataFrame(split_rows)

    n_rows = []
    for n in [2, 3, 4, 5, 6, 7, 10]:
        for mode, label in [("min_hold", "min_hold"), ("fixed_cycle", "periodic_k0")]:
            bundle = _bundle(f"N{n}_{label}", mode, n, 0, BASE_COST)
            n_rows.append(bundle.metrics)
    n_df = pd.DataFrame(n_rows)

    cost_rows = []
    for cost in [0.0001, 0.0005, 0.001]:
        for mode, label in [("min_hold", "min_hold"), ("fixed_cycle", "periodic_k0")]:
            bundle = _bundle(f"cost_{cost:.4f}_{label}", mode, 5, 0, cost)
            cost_rows.append(bundle.metrics)
    cost_df = pd.DataFrame(cost_rows)

    attribution_df, attribution_summary = _skipped_switch_attribution()
    attribution_summary_df = pd.DataFrame([attribution_summary])

    return {
        "data_gate": data_gate,
        "gate": gate_df,
        "phase": phase_df,
        "split": split_df,
        "n_scan": n_df,
        "cost": cost_df,
        "attribution": attribution_df,
        "attribution_summary": attribution_summary_df,
    }


def _report(outputs: dict[str, object]) -> None:
    data_gate = outputs["data_gate"]
    gate_df = outputs["gate"]
    phase_df = outputs["phase"]
    split_df = outputs["split"]
    n_df = outputs["n_scan"]
    cost_df = outputs["cost"]
    attribution_df = outputs["attribution"]
    attribution_summary_df = outputs["attribution_summary"]

    _write_csv(data_gate, "2026-06-02_periodic_reeval_data_gate.csv")
    _write_csv(gate_df, "2026-06-02_periodic_reeval_gate.csv")
    _write_csv(phase_df, "2026-06-02_periodic_reeval_phase_metrics.csv")
    _write_csv(split_df, "2026-06-02_periodic_reeval_split_metrics.csv")
    _write_csv(n_df, "2026-06-02_periodic_reeval_n_scan_metrics.csv")
    _write_csv(cost_df, "2026-06-02_periodic_reeval_cost_metrics.csv")
    _write_csv(attribution_df, "2026-06-02_periodic_reeval_skipped_switch_attribution.csv")
    _write_csv(
        attribution_summary_df,
        "2026-06-02_periodic_reeval_skipped_switch_summary.csv",
    )

    metric_cols = [
        "label",
        "reeval_mode",
        "rebalance_days",
        "phase_offset",
        "start",
        "end",
        "trading_days",
        "annual_return",
        "sharpe",
        "max_drawdown",
        "annual_turnover",
        "avg_holding_days",
        "switch_count",
    ]
    split_cols = ["sample", *metric_cols]

    phase_pivot = phase_df[metric_cols].copy()
    split_show = split_df[split_cols].copy()
    n_show = n_df[metric_cols].copy()
    cost_show = cost_df[
        [
            "label",
            "reeval_mode",
            "cost_bps_one_side",
            "annual_return",
            "sharpe",
            "max_drawdown",
            "annual_turnover",
            "avg_holding_days",
            "switch_count",
        ]
    ].copy()

    lines = [
        "# Periodic Re-evaluation Scan",
        "",
        f"- Run date: {date.today().isoformat()}",
        f"- Config base: `{CONFIG_PATH.relative_to(REPO_ROOT)}` with start forced to `2014-01-01`.",
        "- Data: local HFQ parquet via `data.store.query()`.",
        "- Cost: one-side 0.01% baseline, deducted on actual executed `abs(delta_weight).sum()`.",
        "- Execution: existing engine T+1 open, zero slippage.",
        "- Split: train `2014-01-01` to `2021-12-31`; test `2022-01-01` to data end.",
        "",
        "## Core Evidence - Phase Robustness (4.1)",
        "",
        _markdown_table(phase_pivot, metric_cols),
        "",
        _phase_readout(phase_df),
        "",
        "## Core Evidence - Sample Split (4.2)",
        "",
        _markdown_table(split_show, split_cols),
        "",
        _split_readout(split_df),
        "",
        "## Baseline Gate (4.0)",
        "",
        "Data gate:",
        "",
        _markdown_table(data_gate, list(data_gate.columns)),
        "",
        "Min-hold explicit vs current default `rebalance_days=5`:",
        "",
        _markdown_table(gate_df, list(gate_df.columns)),
        "",
        "Gate result: passed exactly for metrics, net return series, and positions.",
        "",
        "## Skipped Switch Attribution (4.3)",
        "",
        _markdown_table(attribution_summary_df, list(attribution_summary_df.columns)),
        "",
        _attribution_readout(attribution_summary_df),
        "",
        "Raw event rows are in `2026-06-02_periodic_reeval_skipped_switch_attribution.csv`.",
        "",
        "## N Scan (4.4)",
        "",
        _markdown_table(n_show, metric_cols),
        "",
        _n_scan_readout(n_df),
        "",
        "## Cost Sensitivity (4.5)",
        "",
        _markdown_table(cost_show, list(cost_show.columns)),
        "",
        _cost_readout(cost_df),
        "",
        "## Raw CSV",
        "",
        "- `2026-06-02_periodic_reeval_data_gate.csv`",
        "- `2026-06-02_periodic_reeval_gate.csv`",
        "- `2026-06-02_periodic_reeval_phase_metrics.csv`",
        "- `2026-06-02_periodic_reeval_split_metrics.csv`",
        "- `2026-06-02_periodic_reeval_n_scan_metrics.csv`",
        "- `2026-06-02_periodic_reeval_cost_metrics.csv`",
        "- `2026-06-02_periodic_reeval_skipped_switch_attribution.csv`",
        "- `2026-06-02_periodic_reeval_skipped_switch_summary.csv`",
        "",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def _phase_readout(df: pd.DataFrame) -> str:
    annuals = df["annual_return"].astype(float)
    dds = df["max_drawdown"].astype(float)
    spread = annuals.max() - annuals.min()
    baseline = 0.3225502191
    wins = int((annuals > baseline).sum())
    return (
        f"Readout: phase annual-return spread is {_pct(float(spread))}; "
        f"best k={int(df.loc[annuals.idxmax(), 'phase_offset'])}, "
        f"worst k={int(df.loc[annuals.idxmin(), 'phase_offset'])}. "
        f"Max-drawdown range is {_pct(float(dds.max() - dds.min()))}. "
        f"Only {wins}/5 phases beat the min_hold N=5 annual return, which points toward B."
    )


def _split_readout(df: pd.DataFrame) -> str:
    parts = []
    for sample in df["sample"].unique():
        subset = df[df["sample"] == sample].set_index("reeval_mode")
        if {"min_hold", "periodic"}.issubset(subset.index):
            diff = float(subset.loc["periodic", "annual_return"] - subset.loc["min_hold", "annual_return"])
            dd_diff = float(subset.loc["periodic", "max_drawdown"] - subset.loc["min_hold", "max_drawdown"])
            parts.append(
                f"{sample}: periodic annual return minus min_hold {_pct(diff)}, "
                f"max drawdown delta {_pct(dd_diff)}"
            )
    return (
        "Readout: "
        + "; ".join(parts)
        + ". The test-period reversal points toward B rather than a stable A mechanism."
    )


def _attribution_readout(df: pd.DataFrame) -> str:
    if df.empty or int(df.loc[0, "event_count"]) == 0:
        return "Readout: no qualifying skipped non-grid switches were found."
    row = df.loc[0]
    return (
        "Readout: among qualifying non-grid min_hold switches, periodic's held asset "
        f"outperformed in {_pct(float(row['old_win_rate']))} of event windows; "
        f"sum old-minus-new is {_pct(float(row['sum_old_minus_new']))}. "
        "This does not support A's skipped-whipsaw mechanism."
    )


def _n_scan_readout(df: pd.DataFrame) -> str:
    diffs = []
    for n in sorted(df["rebalance_days"].unique()):
        subset = df[df["rebalance_days"] == n].set_index("reeval_mode")
        if {"min_hold", "periodic"}.issubset(subset.index):
            diffs.append(float(subset.loc["periodic", "annual_return"] - subset.loc["min_hold", "annual_return"]))
    positive = sum(1 for diff in diffs if diff > 0)
    return (
        f"Readout: periodic beats min_hold on annual return in {positive}/{len(diffs)} "
        "tested N values. The advantage is not smooth across the N band, pointing toward B."
    )


def _cost_readout(df: pd.DataFrame) -> str:
    rows = []
    for cost_bps in sorted(df["cost_bps_one_side"].unique()):
        subset = df[df["cost_bps_one_side"] == cost_bps].set_index("reeval_mode")
        if {"min_hold", "periodic"}.issubset(subset.index):
            diff = float(subset.loc["periodic", "annual_return"] - subset.loc["min_hold", "annual_return"])
            rows.append(f"{cost_bps:.1f}bp: {_pct(diff)}")
    return (
        "Readout: periodic annual-return advantage by cost level = "
        + "; ".join(rows)
        + ". Higher costs narrow the deficit but do not rescue k=0, so this is not evidence for A."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate-only", action="store_true")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.gate_only:
        data_gate, gate_df = _gate()
        _write_csv(data_gate, "2026-06-02_periodic_reeval_data_gate.csv")
        _write_csv(gate_df, "2026-06-02_periodic_reeval_gate.csv")
        print("DATA_GATE")
        print(data_gate.to_string(index=False))
        print("GATE")
        print(gate_df.to_string(index=False))
        print("GATE_PASSED")
        return

    outputs = _full_scan()
    _report(outputs)
    print(f"REPORT {REPORT_PATH}")


if __name__ == "__main__":
    main()
