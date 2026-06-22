"""加载 PTrade 实盘策略文件(`deploy/ptrade_quality_momentum_top1.py`)中的纯函数,
供对账复用 —— 直接测「实盘那份代码」,而非它的复制品。

deploy 文件顶层只有 `import numpy` + 常量 + 函数定义(PTrade API 调用都在
initialize/handle_data/rebalance 内,import 时不触发),故可安全 importlib 加载。
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DEPLOY_FILE = REPO_ROOT / "deploy" / "ptrade_quality_momentum_top1.py"


def load_deploy_module():
    """importlib 加载实盘策略文件为模块对象(含 _quality_momentum_score / _should_hold / 常量)。"""
    spec = importlib.util.spec_from_file_location("ptrade_deploy_strategy", DEPLOY_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def port_score_series(closes: pd.Series, score_fn, window: int) -> pd.Series:
    """对一条后复权收盘价序列,用 deploy 的 `_quality_momentum_score` 滚动算每日 score。

    复现实盘语义:每日只用「该日及之前」的收盘价(取最后 window+1 个点)。
    数据不足 / 路径长度为 0 时 score_fn 返回 None → 记为 NaN。
    """
    vals = closes.to_numpy(dtype=float)
    out: dict = {}
    for i in range(len(vals)):
        if i >= window:
            v = score_fn(vals[i - window : i + 1], window)  # window+1 个点
        else:
            v = None
        out[closes.index[i]] = float("nan") if v is None else float(v)
    return pd.Series(out)
