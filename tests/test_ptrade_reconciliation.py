"""PTrade 移植版 vs 框架的逐日对账 harness(满足 CONTRACT.md 的迁移「完成」定义)。

两层:
  逻辑精确硬门(同一份后复权价,隔离策略逻辑,必须 bit-exact):
    - test_score_logic_parity   : 框架 compute vs deploy _quality_momentum_score 每日值
    - test_top1_parity          : 两侧每日 argmax(调仓前 Top1)
    - test_min_hold_rule_parity : deploy _should_hold vs 框架 should_hold_position(输入网格)
  持仓容差对账门(端到端,含 PTrade 真实数据/执行,作回归 tripwire):
    - test_held_reconciliation  : 引擎 held vs PTrade 持仓明细,一致率 ≥ 阈值 + 分歧段全记录

数据来源全部为 committed CSV(framework fixture + PTrade 导出),不碰 data/db → CI 可跑。
fixture 刷新见 backtest/ptrade/reference/MANIFEST.md。设计见 docs/plans/2026-06-21_*.md。
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts.ptrade_recon.parse import parse_holdings
from scripts.ptrade_recon.port import load_deploy_module
from strategy.rebalance import should_hold_position

REF = Path(__file__).resolve().parent.parent / "backtest" / "ptrade" / "reference"
RDS = [2, 5]
SCORE_TOL = 1e-9          # 因子逻辑逐位一致容差
HELD_AGREEMENT_MIN = 0.97  # held 端到端一致率下限(实测 ~98.2%,留余量;此门挡 fillna 类回归)


def _read_scores(name: str) -> pd.DataFrame:
    path = REF / name
    assert path.exists(), f"缺 fixture {path};先跑 scripts/export_framework_reference.py"
    return pd.read_csv(path, index_col="date", parse_dates=["date"])


# ----------------------------- 逻辑精确硬门 -----------------------------

def test_score_logic_parity():
    """框架 compute() 与 deploy 移植函数在真实价格上每日 score 逐位一致。"""
    fw = _read_scores("scores.csv")
    port = _read_scores("port_scores.csv")
    assert list(fw.columns) == list(port.columns), "资产列不一致"

    mask = fw.notna() & port.notna()
    n = int(mask.to_numpy().sum())
    assert n > 1000, f"可比单元格太少({n}),fixture 可能异常"
    max_err = float((fw - port).abs().where(mask).max().max())
    assert max_err < SCORE_TOL, f"score 最大误差 {max_err:.3e} ≥ {SCORE_TOL:.0e}(逐位 {n} 单元)"


def test_top1_parity():
    """两侧每日 argmax(调仓前 Top1 选择)精确一致,含 tie-break 口径。"""
    fw = _read_scores("scores.csv")
    port = _read_scores("port_scores.csv")
    valid = fw.dropna(how="all").index.intersection(port.dropna(how="all").index)
    assert len(valid) > 1000
    mism = fw.loc[valid].idxmax(axis=1) != port.loc[valid].idxmax(axis=1)
    bad = [d.date() for d in valid[mism]]
    assert not bad, f"Top1 分歧 {len(bad)} 天,前几个:{bad[:10]}"


def test_min_hold_rule_parity():
    """deploy _should_hold 与框架 should_hold_position 在输入网格上完全一致。"""
    mod = load_deploy_module()
    for mode in ("min_hold", "fixed_cycle"):
        for rd in (1, 2, 3, 5):
            for hd in (None, 0, 1, 2, 3, 4, 5, 6, 10):
                for has_pos in (True, False):
                    held = "X" if has_pos else None
                    weights = {"X": 1.0} if has_pos else {}
                    port_v = mod._should_hold(held, hd, rd, mode)
                    fw_v = should_hold_position(weights, hd, rd, mode)
                    assert port_v == fw_v, (
                        f"min_hold 规则分歧 mode={mode} rd={rd} hd={hd} "
                        f"has_pos={has_pos}: port={port_v} fw={fw_v}"
                    )


# ----------------------------- 持仓容差对账门 -----------------------------

def _disagreement_runs(a: pd.Series, b: pd.Series):
    """极大连续分歧段:list of (start, end, length, fw_asset, pt_asset)。"""
    idx = a.index
    diff = (a != b).to_numpy()
    runs = []
    i, n = 0, len(diff)
    while i < n:
        if diff[i]:
            j = i
            while j < n and diff[j]:
                j += 1
            runs.append((idx[i], idx[j - 1], j - i, a.iloc[i], b.iloc[i]))
            i = j
        else:
            i += 1
    return runs


@pytest.mark.parametrize("rd", RDS)
def test_held_reconciliation(rd, capsys):
    """引擎 held vs PTrade 持仓明细每日持仓,在索引交集上一致率 ≥ 阈值;分歧段全部记录。

    残余分歧 = 执行模型差异(框架 T+1 开盘 vs PTrade 同 bar + min_hold 相位 + 近似平手日的
    数据馈送微差),非 port 逻辑 bug(逻辑由上面三个硬门保证)。见 CONTRACT.md §1 容差。
    """
    fw_path = REF / f"rd{rd}" / "held.csv"
    assert fw_path.exists(), f"缺 fixture {fw_path}"
    fw = pd.read_csv(fw_path, index_col="date", parse_dates=["date"])["held"]
    pt, _ = parse_holdings(rd)

    idx = fw.index.intersection(pt.index).sort_values()
    a, b = fw.reindex(idx), pt.reindex(idx)
    both = a.notna() & b.notna()
    a, b = a[both], b[both]
    assert len(a) > 1000, f"可比交集太小({len(a)})"

    rate = float((a == b).mean())
    runs = _disagreement_runs(a, b)
    max_run = max((r[2] for r in runs), default=0)

    with capsys.disabled():
        print(
            f"\n[rd={rd}] held 一致率 {rate * 100:.2f}% (交集 {len(a)} 天) | "
            f"分歧 {int((a != b).sum())} 天 / {len(runs)} 段 / 最长 {max_run} 天"
        )
        for s, e, length, fa, pa in sorted(runs, key=lambda r: -r[2]):
            print(f"    {s.date()}~{e.date()} ({length}d) 框架={fa} PTrade={pa}")

    assert rate >= HELD_AGREEMENT_MIN, (
        f"held 一致率 {rate * 100:.2f}% < {HELD_AGREEMENT_MIN * 100:.0f}% "
        f"(回归?检查导出脚本 / 策略改动 / fixture 是否刷新)"
    )
