"""M3 诊断:隔离 PTrade「分红盲」回测 vs 框架 HFQ 引擎的逐资产归因差异。

背景:PTrade 回测 P&L 走原始成交价、不计现金分红(交易/持仓 CSV 无分红行,
Log 无分红事件),而因子打分走 fq='post'(干净)。框架 HFQ 引擎把分红烘焙进
复权价序列。owner 观察到 PTrade 归因里沪深300(510300,四资产中分红率最高)
贡献为负,疑似旧 qfq 污染指纹。

本脚本在**同一套框架执行模型 + 同一持仓序列**下,对每个持有日分别用
  - HFQ 日收益  = (raw_close*adj)[t] / (raw_close*adj)[t-1] - 1   (含分红,=框架/实盘)
  - 原始日收益  = raw_close[t] / raw_close[t-1] - 1                 (分红盲,≈PTrade)
归集到持仓资产上。两者之差 = 纯分红口径效应(执行模型相同,自动抵消);
原始口径下的贡献符号 = 真实 regime drag。

跑:uv run python scripts/ptrade_dividend_attribution_check.py
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backtest.runner import run  # noqa: E402
from run_backtest import _load_config_from_yaml  # noqa: E402

CONFIG_PATH = REPO_ROOT / "strategy" / "configs" / "quality_momentum_top1.yaml"
DATA_DIR = REPO_ROOT / "data" / "db"
ASSETS = ["510300.SH", "159915.SZ", "513100.SH", "518880.SH"]
START = dt.date(2014, 1, 1)
COMMISSION = 0.00005  # 万0.5,本账户真实费率(本地引擎参数名为 commission_ratio)


def _price_returns() -> tuple[pd.DataFrame, pd.DataFrame]:
    """返回 (hfq_ret, raw_ret):索引为日期,列为资产代码的逐日收益。"""
    hfq, raw = {}, {}
    for code in ASSETS:
        df = pd.read_parquet(DATA_DIR / f"{code}.parquet").set_index("date").sort_index()
        hfq_close = df["raw_close"] * df["adj_factor"]  # 后复权(锚归一在收益比率中抵消)
        hfq[code] = hfq_close.pct_change()
        raw[code] = df["raw_close"].pct_change()
    return pd.DataFrame(hfq), pd.DataFrame(raw)


def _held_assets(result) -> pd.Series:
    daily = result.positions.sort_index().fillna(0.0).reindex(result.daily_returns.index).ffill()
    daily = daily.dropna(how="all")
    held = daily.fillna(0.0).idxmax(axis=1)
    held.name = "held_asset"
    return held


def _attribute(held: pd.Series, ret: pd.DataFrame) -> pd.Series:
    """对每个持有日取持仓资产当日收益,按资产求和(匹配框架 summed_daily_return 口径)。"""
    out = {a: 0.0 for a in ASSETS}
    days = {a: 0 for a in ASSETS}
    for d, a in held.items():
        ts = pd.Timestamp(d)
        if a in ret.columns and ts in ret.index:
            v = ret.at[ts, a]
            if pd.notna(v):
                out[a] += float(v)
                days[a] += 1
    s = pd.Series(out)
    s.attrs["days"] = days
    return s


def main() -> None:
    hfq_ret, raw_ret = _price_returns()
    base = _load_config_from_yaml(CONFIG_PATH)

    for rd in (2, 5):
        cfg = dict(base)
        cfg["start"] = START
        cfg["rebalance_days"] = rd
        cfg["commission_ratio"] = COMMISSION
        result = run(cfg)
        held = _held_assets(result)

        hfq_c = _attribute(held, hfq_ret)
        raw_c = _attribute(held, raw_ret)
        gap = hfq_c - raw_c  # 纯分红效应
        days = hfq_c.attrs["days"]

        hfq_tot, raw_tot = hfq_c.sum(), raw_c.sum()
        print(f"\n===== 框架 HFQ 引擎 rd={rd} (start={START}, 万0.5) — 逐资产归因(summed daily return)=====")
        print(f"持有期 {len(held)} 日; 区间 {held.index.min().date()} ~ {held.index.max().date()}")
        print(f"{'asset':<12}{'days':>6}{'HFQ含分红':>12}{'原始分红盲':>12}{'分红gap':>10}"
              f"{'HFQ share':>11}{'raw share':>11}")
        for a in sorted(ASSETS, key=lambda x: -hfq_c[x]):
            hs = hfq_c[a] / hfq_tot * 100 if hfq_tot else 0
            rs = raw_c[a] / raw_tot * 100 if raw_tot else 0
            tag = "  <-- 分红重" if a == "510300.SH" else ""
            print(f"{a:<12}{days[a]:>6}{hfq_c[a]:>12.4f}{raw_c[a]:>12.4f}{gap[a]:>10.4f}"
                  f"{hs:>10.2f}%{rs:>10.2f}%{tag}")
        print(f"{'TOTAL':<12}{'':>6}{hfq_tot:>12.4f}{raw_tot:>12.4f}{gap.sum():>10.4f}")
        c = "510300.SH"
        print(f"\n  → 510300:原始口径(≈PTrade){raw_c[c]:+.4f} → 含分红(框架/实盘){hfq_c[c]:+.4f}"
              f";分红被漏掉 {gap[c]:+.4f}({'占其负贡献的' if raw_c[c]<0 else ''}"
              f"{abs(gap[c]/raw_c[c])*100:.0f}% 来自分红口径)" if raw_c[c] else "")


if __name__ == "__main__":
    main()
