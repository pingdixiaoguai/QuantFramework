"""波动率因子 — N 日收益率的滚动标准差。"""

import pandas as pd

METADATA = {
    "name": "volatility",
    "author": "quantframework",
    "version": "1.0.0",
    "params": {"window": 20},
    "min_history": 21,
    "direction": "lower_better",
    "description": "N 日收益率的滚动标准差",
}


def compute(df: pd.DataFrame, params: dict | None = None) -> pd.Series:
    p = {**METADATA["params"], **(params or {})}
    series = df["close"].pct_change().rolling(p["window"]).std()
    series.index = df["date"]
    return series
