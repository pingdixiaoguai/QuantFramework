"""风险调整质量动量因子 (Risk-Adjusted Quality Momentum)

公式：
    R_N        = ln(close_t / close_{t-N})
    path_N     = sum(|ln(close_j / close_{j-1})|) over last N
    ER_N       = |R_N| / path_N
    vol_N      = std(daily log return, N) * sqrt(N)
    floor_N    = vol_floor_annual * sqrt(N / 252)
    adj_vol_N  = max(vol_N, floor_N)
    ram        = clip(R_N / adj_vol_N, -3, +3)
    score      = ram * ER_N

相对 quality_momentum 的改动：动量项从原始涨幅替换为风险调整动量，
解决跨资产（股票 / 债券 / 黄金 / 红利低波）轮动时高波动资产被天然偏好的问题。
"""

import numpy as np
import pandas as pd

METADATA = {
    "name": "risk_adjusted_quality_momentum",
    "author": "quantframework",
    "version": "1.0.0",
    "params": {"window": 60, "vol_floor_annual": 0.08},
    "min_history": 61,
    "direction": "higher_better",
    "description": "风险调整动量 × Kaufman 效率比率（对数收益、波动率地板、winsorize），跨资产可比",
}

_WINSOR_LIMIT = 3.0


def compute(df: pd.DataFrame, params: dict | None = None) -> pd.Series:
    p = {**METADATA["params"], **(params or {})}
    n = p["window"]
    vol_floor_annual = p["vol_floor_annual"]
    close = df["close"].astype(float)

    log_ret = np.log(close).diff()
    R = np.log(close).diff(n)
    path = log_ret.abs().rolling(window=n).sum()
    vol = log_ret.rolling(window=n).std(ddof=1) * np.sqrt(n)

    floor_n = vol_floor_annual * np.sqrt(n / 252.0)
    adj_vol = np.maximum(vol, floor_n)

    er = R.abs() / path.replace(0, np.nan)
    ram = (R / adj_vol).clip(lower=-_WINSOR_LIMIT, upper=_WINSOR_LIMIT)
    score = ram * er

    series = score.astype(float)
    series.index = df["date"]
    return series
