"""质量动量因子 (Quality Momentum) — 动量 × Kaufman 效率比率。

结合两个维度:
1. 原始动量: (close_t / close_{t-N}) - 1
2. 效率比率 (Kaufman Efficiency Ratio): |总位移| / 路径总长度

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
    "version": "1.0.0",
    "params": {"window": 20},
    "min_history": 21,
    "direction": "higher_better",
    "description": "动量 × Kaufman效率比率，偏好路径平滑的趋势",
}


def compute(df: pd.DataFrame, params: dict | None = None) -> pd.Series:
    p = {**METADATA["params"], **(params or {})}
    w = p["window"]
    close = df["close"]

    # 1. 原始动量
    momentum = close.pct_change(w)

    # 2. 效率比率 (Efficiency Ratio)
    #    分子: 窗口内总位移（绝对值）
    #    分母: 窗口内每日变动绝对值之和（路径总长度）
    displacement = (close - close.shift(w)).abs()
    path_length = close.diff().abs().rolling(w).sum()
    er = displacement / path_length.replace(0, np.nan)

    # 3. 质量动量 = 动量 × 路径效率
    series = momentum * er
    series.index = df["date"]
    return series
