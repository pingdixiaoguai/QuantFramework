"""解析 PTrade 实盘导出 CSV → 与框架 fixture 对齐的逐日 schema。

PTrade 导出文件(gbk 编码)放在 `backtest/ptrade/<folder>/` 下:
  - `持仓明细*.csv`:逐日持仓快照。列:日期, 时间, 合约代码, 最新价, 仓位, 多/空,
    持仓成本价, 市值, 累计盈亏。切换日同一日期可能有两行(旧仓 仓位=0 + 新仓 仓位>0)。
  - `交易详情*.csv`:逐笔成交。列:日期, 时间, 合约代码, 买/卖, 开/平, 成交量, 成交价, 手续费。

后缀对齐:PTrade 用 `.SS`(上交所),框架用 `.SH` → 统一映射成框架口径。

供 `tests/test_ptrade_reconciliation.py` 与框架 fixture(`backtest/ptrade/reference/`)对账。
"""

from __future__ import annotations

import csv
import glob
import io
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
PTRADE_DIR = REPO_ROOT / "backtest" / "ptrade"

# PTrade .SS(上交所)→ 框架 .SH;深交所 .SZ 两边一致
SS2SH = {
    "510300.SS": "510300.SH",
    "513100.SS": "513100.SH",
    "518880.SS": "518880.SH",
    "159915.SZ": "159915.SZ",
}

# 持仓明细列索引
_H_DATE, _H_CODE, _H_AMT, _H_MV = 0, 2, 4, 7


def _resolve_folder(folder: str | int) -> Path:
    """folder 可传 rd 整数(→ "rd{N}")或现成目录名(如 "c05_rd1")。"""
    name = f"rd{folder}" if isinstance(folder, int) else str(folder)
    path = PTRADE_DIR / name
    if not path.is_dir():
        raise FileNotFoundError(f"PTrade 导出目录不存在:{path}")
    return path


def _read_gbk_rows(pattern_dir: Path, pattern: str) -> list[list[str]]:
    matches = glob.glob(str(pattern_dir / pattern))
    if not matches:
        raise FileNotFoundError(f"未找到 {pattern_dir / pattern}")
    with io.open(matches[0], encoding="gbk") as fh:
        return list(csv.reader(fh))


def parse_holdings(folder: str | int) -> tuple[pd.Series, pd.DataFrame]:
    """解析持仓明细 → (held, positions)。

    held: date 索引、值为每日持有标的(.SH 口径;多仓取市值最大者 = Top1 实际持仓)。
    positions: date × 资产 的市值权重(每日归一,sum=1.0;空仓日不出现)。
    两者索引为 PTrade 实际记录的交易日(逐日快照,无需 ffill)。
    """
    rows = _read_gbk_rows(_resolve_folder(folder), "持仓明细*.csv")
    by_date: dict[str, dict[str, float]] = {}
    for r in rows[1:]:
        if len(r) <= _H_MV:
            continue
        try:
            amt = float(r[_H_AMT])
            mv = float(r[_H_MV])
        except ValueError:
            continue
        if amt <= 0:
            continue  # 仓位=0 的行(已清仓)跳过
        asset = SS2SH.get(r[_H_CODE], r[_H_CODE])
        by_date.setdefault(r[_H_DATE], {})[asset] = mv

    held = {pd.Timestamp(d): max(mvs, key=mvs.get) for d, mvs in by_date.items()}
    held_s = pd.Series(held).sort_index()
    held_s.index.name = "date"
    held_s.name = "held"

    pos = pd.DataFrame(
        {pd.Timestamp(d): mvs for d, mvs in by_date.items()}
    ).T.sort_index()
    pos.index.name = "date"
    pos = pos.div(pos.sum(axis=1), axis=0)  # 市值 → 权重
    return held_s, pos


if __name__ == "__main__":
    for rd in (2, 5):
        held, pos = parse_holdings(rd)
        print(
            f"rd={rd}: held {held.shape} "
            f"({held.index.min().date()} ~ {held.index.max().date()}) | "
            f"positions {pos.shape} | 资产列 {list(pos.columns)}"
        )
