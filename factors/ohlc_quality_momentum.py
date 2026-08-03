"""OHLC-path quality momentum factor.

The factor uses the same adjusted OHLC series as the data layer and replaces
the close-only ER path with four configurable daily path components:

    D_j = w_close |C_j-C_{j-1}|
        + w_gap |O_j-C_{j-1}|
        + w_body |C_j-O_j|
        + w_range (H_j-L_j)

The public factor contract is intentionally identical to the other factors so
live, backtest, and research code can consume it through the registry.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

METADATA = {
    "name": "ohlc_quality_momentum",
    "author": "quantframework",
    "version": "1.0.0",
    "params": {
        "window": 20,
        "weights": {
            "close": 0.853,
            "gap": 0.337,
            "body": 0.029,
            "range": 0.281,
        },
    },
    "min_history": 21,
    "direction": "higher_better",
    "description": "动量 × 后复权OHLC路径效率比率",
}


def _weights(params: dict) -> tuple[float, float, float, float]:
    raw = params.get("weights", {})
    values = tuple(float(raw[key]) for key in ("close", "gap", "body", "range"))
    if min(values) < 0:
        raise ValueError(f"OHLC ER weights must be non-negative, got {values}")
    if values[0] + values[1] < 1:
        raise ValueError("close + gap weights must be >= 1")
    if values[0] + values[2] + values[3] < 1:
        raise ValueError("close + body + range weights must be >= 1")
    return values


def compute(df: pd.DataFrame, params: dict | None = None) -> pd.Series:
    p = {**METADATA["params"], **(params or {})}
    p["weights"] = {
        **METADATA["params"]["weights"],
        **(params or {}).get("weights", {}),
    }
    window = int(p["window"])
    w_close, w_gap, w_body, w_range = _weights(p)

    close = df["close"].astype(float)
    open_price = df["open"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)

    momentum = close.pct_change(window, fill_method=None)
    displacement = (close - close.shift(window)).abs()
    daily_path = (
        w_close * close.diff().abs()
        + w_gap * (open_price - close.shift(1)).abs()
        + w_body * (close - open_price).abs()
        + w_range * (high - low)
    )
    er = displacement / daily_path.rolling(window).sum().replace(0, np.nan)
    series = (momentum * er).astype(float)
    series.index = df["date"]
    return series
