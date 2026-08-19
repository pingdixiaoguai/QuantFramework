"""Historical percentile of the current gain from an X-day low."""

from __future__ import annotations

import pandas as pd


METADATA = {
    "name": "rebound_percentile",
    "author": "quantframework",
    "version": "1.0.0",
    "params": {"window": 60, "history": 504},
    "min_history": 563,
    "direction": "higher_better",
    "description": "Trailing percentile of close / X-day low - 1",
}


def compute(df: pd.DataFrame, params: dict | None = None) -> pd.Series:
    p = {**METADATA["params"], **(params or {})}
    window = int(p["window"])
    history = int(p["history"])
    if window < 1 or history < 1:
        raise ValueError(
            f"window and history must be >= 1, got window={window}, history={history}"
        )

    close = pd.to_numeric(df["close"], errors="coerce")
    rolling_low = close.rolling(window, min_periods=window).min()
    rebound = close / rolling_low - 1.0
    percentile = rebound.rolling(history, min_periods=history).rank(pct=True)
    percentile.index = df["date"]
    return percentile.astype(float)
