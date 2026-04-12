"""Backtest run entry point.

Usage:
    uv run python run_backtest.py                           # default config
    uv run python run_backtest.py --config path/to/cfg.yaml # custom config
    uv run python run_backtest.py --from-log path/to/exp.yaml  # reproduce
"""

import argparse
import sys
from datetime import date
from pathlib import Path

import yaml

from backtest.experiment_log import save
from backtest.runner import run


def _load_config_from_yaml(path: Path) -> dict:
    with open(path) as f:
        raw = yaml.safe_load(f)

    # Normalize dates
    for key in ("start", "end"):
        if key in raw and isinstance(raw[key], str):
            raw[key] = date.fromisoformat(raw[key])
    return raw


def _load_config_from_log(path: Path) -> dict:
    with open(path) as f:
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run backtest")
    parser.add_argument("--config", type=Path, help="Path to config YAML")
    parser.add_argument("--from-log", type=Path, help="Reproduce from experiment log")
    args = parser.parse_args()

    if args.from_log:
        config = _load_config_from_log(args.from_log)
        print(f"Reproducing from: {args.from_log}")
    elif args.config:
        config = _load_config_from_yaml(args.config)
    else:
        config = None  # use default

    print("Running backtest...")
    result = run(config)

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
    print(_fmt(result.benchmark_returns, "Bench"))

    # Save experiment log
    log_path = save(result)
    print(f"\nExperiment log: {log_path}")

    # Try to generate HTML report
    try:
        from backtest.report import generate
        report_path = log_path.with_suffix(".html")
        generate(result, report_path)
        print(f"HTML report: {report_path}")
    except Exception as exc:
        print(f"HTML report skipped: {exc}")


if __name__ == "__main__":
    main()
