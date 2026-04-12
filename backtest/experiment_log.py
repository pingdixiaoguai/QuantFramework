"""Experiment logging — save backtest results as YAML snapshots."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import yaml

if TYPE_CHECKING:
    from backtest.runner import BacktestResult

EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent / "experiments"


def _next_id(output_dir: Path) -> str:
    """Generate next experiment ID: YYYYMMDD-NNN."""
    today = datetime.now().strftime("%Y%m%d")
    existing = list(output_dir.glob(f"{today}-*.yaml"))
    seq = len(existing) + 1
    return f"{today}-{seq:03d}"


def _compute_metrics(returns) -> dict:
    """Compute total_return, sharpe, max_drawdown from a return series."""
    if len(returns) == 0:
        return {"total_return": 0.0, "sharpe": 0.0, "max_drawdown": 0.0}

    cumulative = (1 + returns).cumprod()
    total_return = float(cumulative.iloc[-1] - 1)

    # Sharpe (annualized)
    if returns.std() == 0:
        sharpe = 0.0
    else:
        sharpe = float(returns.mean() / returns.std() * np.sqrt(252))

    # Max drawdown
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    max_dd = float(drawdown.min())

    return {
        "total_return": round(total_return, 4),
        "sharpe": round(sharpe, 2),
        "max_drawdown": round(max_dd, 4),
    }


def save(result: BacktestResult, output_dir: Path | None = None) -> Path:
    """Save experiment log as YAML. Returns path to the saved file."""
    import pandas as pd

    if output_dir is None:
        output_dir = EXPERIMENTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    exp_id = _next_id(output_dir)
    train_end_ts = pd.Timestamp(result.train_end)

    # Split returns for train/test metrics
    train_ret = result.daily_returns[result.daily_returns.index <= train_end_ts]
    test_ret = result.daily_returns[result.daily_returns.index > train_end_ts]
    bench_train = result.benchmark_returns[result.benchmark_returns.index <= train_end_ts]
    bench_test = result.benchmark_returns[result.benchmark_returns.index > train_end_ts]

    # Serialize config dates to strings
    config = result.config.copy()
    for key in ("start", "end"):
        if key in config and hasattr(config[key], "isoformat"):
            config[key] = config[key].isoformat()

    log = {
        "experiment": {
            "id": exp_id,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "strategy": config.get("strategy_name", "unknown"),
            "params": {
                f["name"]: f.get("params", {})
                for f in config.get("factors", [])
            },
            "factor_weights": {
                f["name"]: f["weight"]
                for f in config.get("factors", [])
            },
            "asset_pool": config.get("asset_pool", []),
            "data_range": f"{config.get('start', '?')} ~ {config.get('end', '?')}",
            "train_test_split": config.get("train_ratio", 0.7),
            "results": {
                "train": _compute_metrics(train_ret),
                "test": _compute_metrics(test_ret),
                "full": _compute_metrics(result.daily_returns),
                "benchmark": _compute_metrics(result.benchmark_returns),
                "benchmark_train": _compute_metrics(bench_train),
                "benchmark_test": _compute_metrics(bench_test),
            },
        }
    }

    path = output_dir / f"{exp_id}.yaml"
    with open(path, "w") as f:
        yaml.dump(log, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    return path
