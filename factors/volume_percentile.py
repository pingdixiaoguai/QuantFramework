"""Historical percentile of volume relative to its X-day mean."""

from __future__ import annotations

import numpy as np
import pandas as pd


METADATA = {
    "name": "volume_percentile",
    "author": "quantframework",
    "version": "1.0.0",
    "params": {"window": 60, "history": 504},
    "min_history": 563,
    "direction": "higher_better",
    "description": "Trailing percentile of volume / X-day mean volume",
}


def compute(df: pd.DataFrame, params: dict | None = None) -> pd.Series:
    p = {**METADATA["params"], **(params or {})}
    window = int(p["window"])
    history = int(p["history"])
    if window < 1 or history < 1:
        raise ValueError(
            f"window and history must be >= 1, got window={window}, history={history}"
        )

    volume = pd.to_numeric(df["volume"], errors="coerce")
    mean_volume = volume.rolling(window, min_periods=window).mean()
    relative_volume = volume / mean_volume.replace(0.0, np.nan)
    percentile = relative_volume.rolling(history, min_periods=history).rank(pct=True)
    percentile.index = df["date"]
    return percentile.astype(float)
