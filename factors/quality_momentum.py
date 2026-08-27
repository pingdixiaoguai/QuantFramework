"""质量动量因子 (Quality Momentum) — 对数收益动量 × 对数路径Kaufman效率比率。

结合两个维度:
1. 对数动量: ln(close_t / close_{t-N})
2. 效率比率 (Kaufman Efficiency Ratio): |对数价格总位移| / 对数路径总长度

效率比率取值 [0, 1]:
- 接近 1.0: 路径平滑，像一条直线（温水煮青蛙）
- 接近 0.0: 路径颠簸，靠少数大阳线拉起来的

参考:
- Wesley Gray《Quantitative Momentum》— Frog in the Pan
- Robert Carver《Systematic Trading》— 风险调整收益
"""

import numpy as np
import pandas as pd

METADATA = {
    "name": "quality_momentum",
    "author": "quantframework",
    "version": "2.0.0",
    "params": {"window": 20},
    "min_history": 21,
    "direction": "higher_better",
    "description": "对数收益动量 × 对数路径Kaufman效率比率，偏好百分比路径平滑的趋势",
}


def compute(df: pd.DataFrame, params: dict | None = None) -> pd.Series:
    p = {**METADATA["params"], **(params or {})}
    w = p["window"]
    close = df["close"]
    log_close = np.log(close.astype(float))

    # 1. 对数收益动量
    momentum = log_close - log_close.shift(w)

    # 2. 对数路径效率比率 (Efficiency Ratio)
    #    分子: 窗口内对数价格总位移（绝对值）
    #    分母: 窗口内每日对数收益绝对值之和（百分比路径总长度）
    displacement = (log_close - log_close.shift(w)).abs()
    path_length = log_close.diff().abs().rolling(w).sum()
    er = displacement / path_length.replace(0, np.nan)

    # 3. 质量动量 = 动量 × 路径效率
    series = momentum * er
    series.index = df["date"]
    return series
