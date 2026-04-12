# 复制此文件后请改文件名、改 METADATA["name"]、改 compute 逻辑
"""
因子模板 — 复制此文件，重命名，实现 compute 方法。
"""

METADATA = {
    "name": "my_factor",
    "author": "your_name",
    "version": "1.0.0",
    "params": {"window": 20},
    "min_history": 20,
    "direction": "higher_better",  # or "lower_better"
    "description": "一句话描述因子逻辑",
}


def compute(df, params=None):
    """
    输入: df — 标准化行情 DataFrame (date, open, high, low, close, volume)
    输出: pd.Series — 因子值，日期索引，float 类型

    约束:
    - 不得修改输入 df
    - 不得访问全局状态
    - 不得引入 registry 之外的第三方库
    """
    p = {**METADATA["params"], **(params or {})}
    # 你的因子计算逻辑
    return df["close"].pct_change(p["window"])
