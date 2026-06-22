"""导出框架侧每日 (score, held, positions) 参考 fixture。

供 `tests/test_ptrade_reconciliation.py` 与 PTrade 实盘导出做逐日对账(方案 B)。
设计依据见 `docs/plans/2026-06-21_ptrade_reconciliation_harness.md` 与 `CONTRACT.md`。

为什么 committed CSV:`data/db/*.parquet` 被 gitignore、仅本地 junction 可见,CI 跑不了
引擎。把框架产出快照进仓 → 测试 hermetic、CI 可跑。

口径(与引擎内部严格一致):
  - score 用 `data.store.query()`(= 引擎喂因子的同一份 HFQ df),对全序列跑
    `factors.quality_momentum.compute()`。compute 因果且尺度不变,故全序列一次算 =
    引擎逐日「truncate <= t 取 last」的结果。
  - held/positions 用 `backtest.runner.run()`,positions 仅执行日有行,先 reindex 到
    daily_returns 索引再 ffill 补成逐日(见 backtest/CLAUDE.md pitfall)。

改了策略 / 因子 / data/db 数据后,重跑本脚本刷新 fixture,并回填 reference/MANIFEST.md
的「数据快照」。

跑:uv run python scripts/export_framework_reference.py
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
import subprocess
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backtest.runner import run  # noqa: E402
from data.store import query  # noqa: E402
from factors import quality_momentum  # noqa: E402
from run_backtest import _load_config_from_yaml  # noqa: E402
from scripts.ptrade_recon.port import load_deploy_module, port_score_series  # noqa: E402

CONFIG_PATH = REPO_ROOT / "strategy" / "configs" / "quality_momentum_top1.yaml"
OUT_ROOT = REPO_ROOT / "backtest" / "ptrade" / "reference"
START = dt.date(2014, 1, 1)
REBALANCE_DAYS = (2, 5)
TRANSACTION_COST_RATE = 0.0002  # 万2 / c20，对齐实盘导出口径;不影响 score/held/Top1


def _scores(asset_pool: list[str], end: dt.date) -> pd.DataFrame:
    """框架侧:逐资产 quality_momentum 每日 score 宽表(行=日期, 列=资产)。rd 无关。"""
    cols = {}
    for asset in asset_pool:
        df = query(asset, START, end)  # 引擎同源 HFQ df(date/open/high/low/close/volume)
        series = quality_momentum.compute(df)  # 因果 + 尺度不变 → 等于引擎逐日 score
        cols[asset] = series
    out = pd.DataFrame(cols).sort_index()
    out.index.name = "date"
    return out


def _port_scores(asset_pool: list[str], end: dt.date) -> pd.DataFrame:
    """PTrade 侧:用 deploy 的真实 `_quality_momentum_score` 离线滚动算每日 score。rd 无关。

    与 `_scores` 喂同一份 query() 后复权收盘价 → 测试比两份 committed CSV 即验证因子逻辑
    逐位对齐,无需测试时碰 data/db(CI 可跑)。
    """
    mod = load_deploy_module()
    cols = {}
    for asset in asset_pool:
        close = query(asset, START, end).set_index("date")["close"]
        cols[asset] = port_score_series(close, mod._quality_momentum_score, mod.WINDOW)
    out = pd.DataFrame(cols).sort_index()
    out.index.name = "date"
    return out


def _held_and_positions(base: dict, rd: int) -> tuple[pd.Series, pd.DataFrame]:
    """引擎 min_hold 调仓后的逐日 held 与逐日仓位权重。"""
    cfg = dict(base)
    cfg["start"] = START
    cfg["rebalance_days"] = rd
    cfg["rebalance_mode"] = "min_hold"
    cfg["transaction_cost_rate"] = TRANSACTION_COST_RATE  # 新 key(非 commission_ratio)
    result = run(cfg)
    # positions 稀疏:每个执行日行只记当前持仓那只(=1.0),其余资产为 NaN(非 0)。
    # 必须先 fillna(0) 把每个执行日补成完整 one-hot 向量,**再** reindex+ffill 补逐日,
    # 否则 ffill 会按列前向填充旧持仓 → 多列同时为 1.0 → idxmax 取错(见 backtest/CLAUDE.md)。
    daily = (
        result.positions.sort_index()
        .fillna(0.0)
        .reindex(result.daily_returns.index)
        .ffill()
        .dropna(how="all")
    )
    positions = daily
    positions.index.name = "date"
    held = positions.idxmax(axis=1)
    held.index.name = "date"
    held.name = "held"
    return held, positions


def _snapshot(asset_pool: list[str], end: dt.date) -> str:
    """生成「数据快照」文本块,回填进 reference/MANIFEST.md。"""
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT
        ).decode().strip()
    except Exception:
        sha = "(unknown)"
    bars = []
    last_dates = []
    for asset in asset_pool:
        df = query(asset, START, end)
        bars.append(f"{asset}={len(df)}")
        last_dates.append(df["date"].max())
    return (
        f"- 生成日期:{dt.date.today().isoformat()}\n"
        f"- 生成时所在 commit:{sha}\n"
        f"- data/db 末日:{max(last_dates).date()}\n"
        f"- 各资产 bar 数(start={START}):{', '.join(bars)}\n"
    )


def main() -> None:
    base = _load_config_from_yaml(CONFIG_PATH)
    asset_pool = list(base["asset_pool"])
    end = base["end"]  # _load_config_from_yaml 已把 "today" 规整为 date

    # rd 无关的 score(框架 compute vs PTrade port fn),写在 reference 根目录
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    scores = _scores(asset_pool, end)
    port_scores = _port_scores(asset_pool, end)
    scores.to_csv(OUT_ROOT / "scores.csv")
    port_scores.to_csv(OUT_ROOT / "port_scores.csv")
    print(f"scores {scores.shape} | port_scores {port_scores.shape} -> {OUT_ROOT.relative_to(REPO_ROOT)}")

    # rd 相关的 held / positions,写在 reference/rd{N}/
    for rd in REBALANCE_DAYS:
        out_dir = OUT_ROOT / f"rd{rd}"
        out_dir.mkdir(parents=True, exist_ok=True)
        held, positions = _held_and_positions(base, rd)
        held.to_csv(out_dir / "held.csv")
        positions.to_csv(out_dir / "positions.csv")
        print(
            f"rd={rd}: held {held.shape} "
            f"({held.index.min().date()} ~ {held.index.max().date()}) | "
            f"positions {positions.shape} -> {out_dir.relative_to(REPO_ROOT)}"
        )

    print("\n===== 回填 reference/MANIFEST.md 的「数据快照」 =====")
    print(_snapshot(asset_pool, end))


if __name__ == "__main__":
    main()
