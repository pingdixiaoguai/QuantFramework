"""Backtest run entry point.

Usage:
    uv run python run_backtest.py                           # default config
    uv run python run_backtest.py --config path/to/cfg.yaml # custom config
    uv run python run_backtest.py --from-log path/to/exp.yaml  # reproduce
"""

import argparse
import dataclasses
import sys
from datetime import date
from pathlib import Path

import yaml

from backtest.experiment_log import save
from backtest.runner import BacktestResult, run


def _load_config_from_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    # Normalize dates
    for key in ("start", "end"):
        if key in raw and isinstance(raw[key], str):
            raw[key] = date.fromisoformat(raw[key])
    return raw


def _load_config_from_log(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        log = yaml.safe_load(f)

    exp = log["experiment"]
    # Reconstruct config from experiment log
    data_range = exp.get("data_range", "")
    parts = data_range.split(" ~ ")
    start = date.fromisoformat(parts[0]) if len(parts) >= 1 else date(2016, 1, 1)
    end = date.fromisoformat(parts[1]) if len(parts) >= 2 else date.today()

    factors = []
    params = exp.get("params", {})
    weights = exp.get("factor_weights", {})
    for fname, fweight in weights.items():
        fc = {"name": fname, "weight": fweight}
        if fname in params:
            fc["params"] = params[fname]
        factors.append(fc)

    return {
        "strategy_name": exp.get("strategy", "unknown"),
        "asset_pool": exp.get("asset_pool", []),
        "start": start,
        "end": end,
        "factors": factors,
        "train_ratio": exp.get("train_test_split", 0.7),
        "rebalance_rule": "daily",
    }


def _apply_baseline(
    strategy_result: BacktestResult,
    baseline_result: BacktestResult,
    baseline_strategy_name: str | None,
) -> BacktestResult:
    """Replace strategy_result.benchmark_returns with baseline_result.daily_returns
    restricted to strategy_result's trading days.

    Raises RuntimeError if there is no overlap between the two series' indices.
    Returns a new BacktestResult; the input is not mutated.
    """
    overlap = strategy_result.daily_returns.index.intersection(
        baseline_result.daily_returns.index
    )
    if len(overlap) == 0:
        s_idx = strategy_result.daily_returns.index
        b_idx = baseline_result.daily_returns.index
        raise RuntimeError(
            f"No overlapping trading days between strategy and baseline. "
            f"strategy={s_idx.min()}~{s_idx.max()}, "
            f"baseline={b_idx.min()}~{b_idx.max()}"
        )
    aligned_bench = baseline_result.daily_returns.loc[overlap]

    return dataclasses.replace(
        strategy_result,
        benchmark_returns=aligned_bench,
        baseline_strategy_name=baseline_strategy_name,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run backtest")
    parser.add_argument("--config", type=Path, help="Path to config YAML")
    parser.add_argument("--from-log", type=Path, help="Reproduce from experiment log")
    parser.add_argument(
        "--baseline-config",
        type=Path,
        help="Path to a YAML config whose strategy returns will replace the default "
             "pool-mean benchmark in the HTML report.",
    )
    args = parser.parse_args()

    if args.from_log:
        config = _load_config_from_log(args.from_log)
        print(f"Reproducing from: {args.from_log}")
    elif args.config:
        config = _load_config_from_yaml(args.config)
    else:
        config = None  # use default

    # Run baseline first if requested
    baseline_result = None
    baseline_name = None
    if args.baseline_config:
        baseline_config = _load_config_from_yaml(args.baseline_config)
        baseline_name = baseline_config.get("strategy_name")
        print(f"Running baseline: {baseline_name}...")
        baseline_result = run(baseline_config)

    print("Running backtest...")
    result = run(config)

    if baseline_result is not None:
        result = _apply_baseline(result, baseline_result, baseline_name)
        print(f"Using baseline '{baseline_name}' "
              f"({len(result.benchmark_returns)} overlapping days)")

    # Print summary
    print(f"\nBacktest complete: {len(result.daily_returns)} trading days")
    print(f"Train/test split at: {result.train_end}")

    import numpy as np
    import pandas as pd

    train_end_ts = pd.Timestamp(result.train_end)
    train_ret = result.daily_returns[result.daily_returns.index <= train_end_ts]
    test_ret = result.daily_returns[result.daily_returns.index > train_end_ts]

    def _fmt(returns: pd.Series, label: str) -> str:
        if len(returns) == 0:
            return f"  {label}: no data"
        cum = (1 + returns).cumprod()
        total = cum.iloc[-1] - 1
        sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        dd = ((cum - cum.cummax()) / cum.cummax()).min()
        return f"  {label}: return={total:.2%}  sharpe={sharpe:.2f}  max_dd={dd:.2%}"

    print(_fmt(train_ret, "Train"))
    print(_fmt(test_ret, "Test "))
    bench_label = (
        f"Bench ({result.baseline_strategy_name})"
        if result.baseline_strategy_name
        else "Bench"
    )
    print(_fmt(result.benchmark_returns, bench_label))

    # Save experiment log
    log_path = save(result)
    print(f"\nExperiment log: {log_path}")

    # Try to generate HTML report
    try:
        from backtest.report import generate
        report_path = log_path.with_suffix(".html")
        generate(result, report_path, benchmark_title=result.baseline_strategy_name)
        print(f"HTML report: {report_path}")
    except Exception as exc:
        print(f"HTML report skipped: {exc}")


if __name__ == "__main__":
    main()
