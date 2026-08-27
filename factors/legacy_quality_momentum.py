"""Historical quality momentum: simple-return MOM times price-path Kaufman ER."""

import numpy as np
import pandas as pd


METADATA = {
    "name": "legacy_quality_momentum",
    "author": "quantframework",
    "version": "1.0.0",
    "params": {"window": 20},
    "min_history": 21,
    "direction": "higher_better",
    "description": "历史原版：简单收益动量 × 价格路径Kaufman效率比率",
}


def compute(df: pd.DataFrame, params: dict | None = None) -> pd.Series:
    p = {**METADATA["params"], **(params or {})}
    window = int(p["window"])
    close = df["close"].astype(float)
    momentum = close.pct_change(window)
    displacement = (close - close.shift(window)).abs()
    path = close.diff().abs().rolling(window).sum()
    result = momentum * displacement / path.replace(0.0, np.nan)
    result.index = df["date"]
    return result.astype(float)
