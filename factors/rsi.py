"""Relative Strength Index (RSI) using Wilder's smoothing."""

from __future__ import annotations

import numpy as np
import pandas as pd


METADATA = {
    "name": "rsi",
    "author": "quantframework",
    "version": "1.0.0",
    "params": {"window": 14},
    "min_history": 15,
    "direction": "higher_better",
    "description": "Wilder RSI; 0-100 oscillator with a default 14-day window",
}


def _wilder_average(values: pd.Series, window: int) -> pd.Series:
    """Return Wilder's recursively smoothed average without look-ahead."""
    result = pd.Series(np.nan, index=values.index, dtype=float)
    if len(values) <= window:
        return result

    initial = values.iloc[1 : window + 1]
    if initial.isna().any():
        return result

    # Wilder's recursion is an EMA with alpha=1/window, seeded by the simple
    # mean of the first `window` changes. Seeding an EWM explicitly keeps the
    # exact formula while avoiding a Python/pandas scalar loop during every
    # truncated backtest calculation.
    tail = values.iloc[window:].astype(float).copy()
    tail.iloc[0] = float(initial.mean())
    invalid_suffix = tail.isna().cummax()
    smoothed = tail.ewm(alpha=1.0 / window, adjust=False).mean()
    result.iloc[window:] = smoothed.mask(invalid_suffix).to_numpy()
    return result


def compute(df: pd.DataFrame, params: dict | None = None) -> pd.Series:
    """Compute Wilder RSI from close prices and index it by ``df['date']``."""
    p = {**METADATA["params"], **(params or {})}
    window = int(p["window"])
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")

    close = pd.to_numeric(df["close"], errors="coerce")
    delta = close.diff()
    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)

    avg_gain = _wilder_average(gains, window)
    avg_loss = _wilder_average(losses, window)
    relative_strength = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - 100.0 / (1.0 + relative_strength)

    no_loss = avg_loss.eq(0.0)
    rsi = rsi.mask(no_loss & avg_gain.gt(0.0), 100.0)
    rsi = rsi.mask(no_loss & avg_gain.eq(0.0), 50.0)
    rsi.index = df["date"]
    return rsi.astype(float)
