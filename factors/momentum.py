"""动量因子 — N 日收益率。"""

import pandas as pd

METADATA = {
    "name": "momentum",
    "author": "quantframework",
    "version": "1.0.0",
    "params": {"window": 20},
    "min_history": 21,
    "direction": "higher_better",
    "description": "N 日收益率",
}


def compute(df: pd.DataFrame, params: dict | None = None) -> pd.Series:
    p = {**METADATA["params"], **(params or {})}
    series = df["close"].pct_change(periods=p["window"])
    series.index = df["date"]
    return series
