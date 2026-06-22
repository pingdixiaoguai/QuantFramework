"""M3 闭环:PTrade vs 框架 HFQ 逐资产归因的完整口径分解。

目的:把 owner 看到的「PTrade 归因里 510300 负贡献」分解为三块互不重叠的来源:
  1) 度量口径   —— money-weighted(已实现资金 P&L 占比) vs return-summed(持有日收益求和)
  2) 执行/持仓 —— PTrade 日线同 bar 成交 vs 框架 T+1 开盘 → 实际持有的日子不同
  3) 分红口径   —— PTrade 原始价(分红盲) vs 框架 HFQ(含分红)

隔离手法:用**同一份 parquet 价格 + 同一种 return-summed 口径**,只把「每日持仓资产」
序列在「框架引擎产出」与「PTrade 持仓明细解析」之间切换。两者之差 = 纯执行/持仓效应
(价格、口径、分红处理全相同)。分红效应另由 HFQ vs 原始 给出(见 dividend_attribution_check)。

附带:核对 PTrade 持仓明细的「最新价」是否 = parquet 原始价,确认 PTrade 持仓价为原始价。

跑:uv run python scripts/ptrade_vs_framework_attribution.py
"""

from __future__ import annotations

import csv
import datetime as dt
import glob
import io
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backtest.runner import run  # noqa: E402
from run_backtest import _load_config_from_yaml  # noqa: E402
from scripts.ptrade_recon.parse import SS2SH  # noqa: E402  # 后缀映射的唯一真相源

CONFIG_PATH = REPO_ROOT / "strategy" / "configs" / "quality_momentum_top1.yaml"
DATA_DIR = REPO_ROOT / "data" / "db"
PTRADE_DIR = REPO_ROOT / "backtest" / "ptrade"
ASSETS = ["510300.SH", "159915.SZ", "513100.SH", "518880.SH"]
START = dt.date(2014, 1, 1)
COMMISSION = 0.00005


def _price_returns():
    hfq, raw, rawpx = {}, {}, {}
    for code in ASSETS:
        df = pd.read_parquet(DATA_DIR / f"{code}.parquet").set_index("date").sort_index()
        rawpx[code] = df["raw_close"]
        hfq[code] = (df["raw_close"] * df["adj_factor"]).pct_change()
        raw[code] = df["raw_close"].pct_change()
    return pd.DataFrame(hfq), pd.DataFrame(raw), pd.DataFrame(rawpx)


def _framework_held(rd, base):
    cfg = dict(base)
    cfg["start"] = START
    cfg["rebalance_days"] = rd
    cfg["transaction_cost_rate"] = COMMISSION  # main 已弃用 commission_ratio,改用此 key
    result = run(cfg)
    daily = result.positions.sort_index().fillna(0.0).reindex(result.daily_returns.index).ffill()
    held = daily.dropna(how="all").fillna(0.0).idxmax(axis=1)
    return held


def _ptrade_held(rd):
    """从 PTrade 持仓明细解析每日持仓资产(数量>0;若多仓取市值最大者)。"""
    f = glob.glob(str(PTRADE_DIR / f"rd{rd}" / "持仓明细*.csv"))[0]
    rows = list(csv.reader(io.open(f, encoding="gbk")))
    by_date = {}  # date -> {asset: mv}
    px_check = []  # (date, asset, ptrade_px)
    for r in rows[1:]:
        if len(r) < 8:
            continue
        d, code, px, amt, mv = r[0], r[2], float(r[3]), float(r[4]), float(r[7])
        if amt <= 0:
            continue
        sh = SS2SH.get(code, code)
        by_date.setdefault(d, {})[sh] = mv
        px_check.append((pd.Timestamp(d), sh, px))
    held = {pd.Timestamp(d): max(mvs, key=mvs.get) for d, mvs in by_date.items()}
    return pd.Series(held).sort_index(), px_check


def _attribute(held, ret):
    out = {a: 0.0 for a in ASSETS}
    days = {a: 0 for a in ASSETS}
    for d, a in held.items():
        ts = pd.Timestamp(d)
        if a in ret.columns and ts in ret.index and pd.notna(ret.at[ts, a]):
            out[a] += float(ret.at[ts, a])
            days[a] += 1
    return pd.Series(out), days


def _money_pnl(rd):
    """PTrade 交易详情:净已实现资金流 + 期末持仓市值(=我之前的 money-weighted 口径)。"""
    fd = glob.glob(str(PTRADE_DIR / f"rd{rd}" / "交易详情*.csv"))[0]
    flow = {a: 0.0 for a in ASSETS}
    for r in list(csv.reader(io.open(fd, encoding="gbk")))[1:]:
        if len(r) < 8:
            continue
        code, side, vol, px, fee = SS2SH.get(r[2], r[2]), r[3], float(r[5]), float(r[6]), float(r[7])
        flow[code] += (vol * px if side == "卖" else -vol * px) - fee
    fh = glob.glob(str(PTRADE_DIR / f"rd{rd}" / "持仓明细*.csv"))[0]
    last_mv = {}
    for r in list(csv.reader(io.open(fh, encoding="gbk")))[1:]:
        if len(r) < 8:
            continue
        code = SS2SH.get(r[2], r[2])
        if float(r[4]) > 0:
            last_mv[code] = float(r[7])
        else:
            last_mv.pop(code, None)
    pnl = {a: flow[a] + last_mv.get(a, 0.0) for a in ASSETS}
    return pd.Series(pnl)


def main():
    hfq_ret, raw_ret, rawpx = _price_returns()
    base = _load_config_from_yaml(CONFIG_PATH)

    for rd in (2, 5):
        fw_held = _framework_held(rd, base)
        pt_held, px_check = _ptrade_held(rd)

        fw_hfq, _ = _attribute(fw_held, hfq_ret)
        fw_raw, _ = _attribute(fw_held, raw_ret)
        pt_raw, pt_days = _attribute(pt_held, raw_ret)
        money = _money_pnl(rd)

        def share(s):
            t = s.sum()
            return s / t * 100 if t else s * 0

        print(f"\n{'='*92}\n  rd={rd}  逐资产归因口径分解(占比%,正=贡献正收益)\n{'='*92}")
        print(f"{'资产':<12}{'PTrade-money':>14}{'PTrade-raw求和':>16}{'框架-raw求和':>15}"
              f"{'框架-HFQ求和':>15}")
        print(f"{'(口径)':<12}{'(资金加权)':>14}{'(执行+原始价)':>16}{'(执行差隔离)':>15}{'(+含分红)':>15}")
        for a in ASSETS:
            tag = "  <--沪深300" if a == "510300.SH" else ""
            print(f"{a:<12}{share(money)[a]:>13.2f}%{share(pt_raw)[a]:>15.2f}%"
                  f"{share(fw_raw)[a]:>14.2f}%{share(fw_hfq)[a]:>14.2f}%{tag}")
        c = "510300.SH"
        print(f"\n  510300 持有天数: PTrade={pt_days[c]}  (框架口径见 dividend_check)")
        print(f"  510300 占比走势: money {share(money)[c]:+.2f}%  →  PTrade-raw求和 "
              f"{share(pt_raw)[c]:+.2f}%  →  框架-raw求和 {share(fw_raw)[c]:+.2f}%  →  框架-HFQ "
              f"{share(fw_hfq)[c]:+.2f}%")

    # 价格核对:PTrade 最新价 vs parquet 原始价(抽前若干个 510300 持有日)
    print(f"\n{'='*92}\n  PTrade 持仓「最新价」 vs parquet raw_close 核对 (510300, 前8个持有日)\n{'='*92}")
    _, px_check = _ptrade_held(2)
    rc = rawpx["510300.SH"]
    n = 0
    for ts, sh, ppx in px_check:
        if sh != "510300.SH" or ts not in rc.index:
            continue
        print(f"  {ts.date()}  PTrade={ppx:.3f}  parquet_raw={rc.loc[ts]:.3f}  "
              f"{'一致' if abs(ppx - rc.loc[ts]) < 0.005 else '差异!'}")
        n += 1
        if n >= 8:
            break


if __name__ == "__main__":
    main()
