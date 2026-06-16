# -*- coding: utf-8 -*-
"""
PTrade 研究环境 API 探针（数据侧）
===================================
用途：在 PTrade 研究环境（Jupyter Notebook）里**整段运行**，把**全部打印输出**复制回来，
用于确认策略所依赖的数据 API 在你的券商/版本上的真实行为。

它会探测：
  1. get_history 是否在研究环境可用，返回结构（类型/列名/索引/dtype/末尾几行）
  2. get_price 作为备选是否可用（资料称研究环境可能只支持 get_price）
  3. 代码后缀 .SS 与 .SH 哪个被接受
  4. fq='post'（后复权）与 fq=None（不复权）差异
  5. include=False / True 对「是否含当日」的影响
  6. 多标的批量取数的返回结构
  7. get_trade_days 交易日历
  8. 用能跑通的接口在真实数据上算一遍质量动量 score，打印各 ETF 分数与 Top1

注意：本脚本**不会下单**，纯数据查询，安全。order_*/context/run_daily 等交易接口
研究环境无法测，请用配套的 deploy/ptrade_backtest_probe.py 在回测里验证。
"""

import numpy as np
import pandas as pd

# ---- 探测目标 ----
ETF_SS = "510300.SS"     # 沪深300 ETF，PTrade 上交所后缀
ETF_SH = "510300.SH"     # 同一只，框架里的后缀；测试是否也被接受
POOL_SS = ["510300.SS", "159915.SZ", "513100.SS", "518880.SS"]
WINDOW = 20


def _show(tag, obj):
    print("\n----- %s -----" % tag)
    print("type :", type(obj))
    try:
        print("len  :", len(obj), "| shape:", getattr(obj, "shape", None))
    except Exception:
        pass
    if isinstance(obj, pd.DataFrame):
        print("columns:", list(obj.columns))
        print("index  :", type(obj.index).__name__, "| head:", list(obj.index[:2]), "tail:", list(obj.index[-2:]))
        print("dtypes :", dict(obj.dtypes.astype(str)))
        print("tail(3):")
        print(obj.tail(3).to_string())
    elif isinstance(obj, pd.Series):
        print("name:", obj.name, "| index head:", list(obj.index[:2]), "tail:", list(obj.index[-2:]))
        print("tail(3):", list(obj.tail(3)))
    elif isinstance(obj, dict):
        print("keys:", list(obj.keys())[:10])
        for k in list(obj.keys())[:2]:
            v = obj[k]
            print("  [%s] type=%s sample=%s" % (k, type(v).__name__, repr(v)[:120]))
    else:
        print("repr:", repr(obj)[:400])


def _probe(tag, fn):
    print("\n========== %s ==========" % tag)
    try:
        r = fn()
        _show(tag, r)
        return r
    except Exception as e:
        print("[FAILED] %s -> %s: %s" % (tag, type(e).__name__, e))
        return None


print("############### PTrade 研究环境 API 探针开始 ###############")
print("pandas:", pd.__version__, "| numpy:", np.__version__)

# ---- 1. get_history：单标的，.SS，后复权，不含当日 ----
h1 = _probe(
    "get_history(25,'1d','close','510300.SS',fq='post',include=False)",
    lambda: get_history(25, "1d", "close", ETF_SS, fq="post", include=False),
)

# ---- 2. get_history：.SH 后缀是否被接受 ----
h2 = _probe(
    "get_history(... '510300.SH' ...)  # 测 .SH 后缀",
    lambda: get_history(25, "1d", "close", ETF_SH, fq="post", include=False),
)

# ---- 3. get_history：fq=None 不复权（对照） ----
_probe(
    "get_history(... fq=None ...)  # 不复权对照",
    lambda: get_history(25, "1d", "close", ETF_SS, fq=None, include=False),
)

# ---- 4. get_history：include=True 是否把当日加进来 ----
_probe(
    "get_history(... include=True ...)  # 看是否含当日",
    lambda: get_history(5, "1d", "close", ETF_SS, fq="post", include=True),
)

# ---- 5. get_history：多标的批量 ----
_probe(
    "get_history(25,'1d','close', POOL, fq='post')  # 多标的返回结构",
    lambda: get_history(25, "1d", "close", POOL_SS, fq="post", include=False),
)

# ---- 6. get_history：is_dict=True 形式 ----
_probe(
    "get_history(... is_dict=True ...)  # OrderedDict 形式",
    lambda: get_history(25, "1d", "close", POOL_SS, fq="post", include=False, is_dict=True),
)

# ---- 7. get_price 备选（研究环境可能只支持它）----
_probe(
    "get_price('510300.SS', count=25, frequency='1d', fields='close', fq='post')",
    lambda: get_price(ETF_SS, count=25, frequency="1d", fields="close", fq="post"),
)
_probe(
    "get_price('510300.SS', count=25, frequency='1d', fields=['close'], fq='post')",
    lambda: get_price(ETF_SS, count=25, frequency="1d", fields=["close"], fq="post"),
)

# ---- 8. get_trade_days 交易日历 ----
_probe(
    "get_trade_days(count=5)",
    lambda: get_trade_days(count=5),
)

# ---- 9. 端到端：用能跑通的接口取收盘价，算质量动量 score ----
print("\n========== 端到端因子计算（真实数据） ==========")

def _get_closes(security):
    """依次尝试 get_history / get_price，返回 1D 收盘价 ndarray（升序）；失败返回 None。"""
    # 尝试 A：get_history 单标的
    try:
        df = get_history(WINDOW + 5, "1d", "close", security, fq="post", include=False)
        if df is not None and len(df) > 0:
            if isinstance(df, pd.DataFrame):
                col = "close" if "close" in df.columns else df.columns[-1]
                return np.asarray(df[col].values, dtype=float)
            return np.asarray(df, dtype=float).reshape(-1)
    except Exception as e:
        print("  [%s] get_history 不可用: %s" % (security, e))
    # 尝试 B：get_price
    try:
        df = get_price(security, count=WINDOW + 5, frequency="1d", fields="close", fq="post")
        if df is not None and len(df) > 0:
            if isinstance(df, pd.DataFrame):
                col = "close" if "close" in df.columns else df.columns[-1]
                return np.asarray(df[col].values, dtype=float)
            if isinstance(df, pd.Series):
                return np.asarray(df.values, dtype=float)
    except Exception as e:
        print("  [%s] get_price 不可用: %s" % (security, e))
    return None


def _quality_momentum_score(closes, window):
    if closes is None or len(closes) < window + 1:
        return None
    c = np.asarray(closes, dtype=float)[-(window + 1):]
    if np.any(np.isnan(c)) or c[0] == 0:
        return None
    momentum = c[-1] / c[0] - 1.0
    displacement = abs(c[-1] - c[0])
    path_length = float(np.sum(np.abs(np.diff(c))))
    if path_length == 0.0:
        return None
    return momentum * (displacement / path_length)


scores = {}
for s in POOL_SS:
    closes = _get_closes(s)
    if closes is None:
        print("  %s: 取数失败" % s)
        continue
    val = _quality_momentum_score(closes, WINDOW)
    print("  %s: n=%d last_close=%.4f score=%s" %
          (s, len(closes), closes[-1], ("%.6f" % val) if val is not None else None))
    if val is not None:
        scores[s] = val

if scores:
    best = max(scores, key=lambda k: scores[k])
    print("\n  >>> 各标的 score:", {k: round(v, 6) for k, v in scores.items()})
    print("  >>> Top1（应买入）:", best)
else:
    print("  无法计算 score —— 取数接口都没跑通，请把上面的报错贴回。")

print("\n############### 探针结束：请把以上全部输出复制回来 ###############")
