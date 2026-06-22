# 框架参考 fixture — 生成清单(MANIFEST)

本目录(`backtest/ptrade/reference/`)存放**框架侧每日 (score, held, positions)** 的
committed CSV fixture,供 `tests/test_ptrade_reconciliation.py` 与 PTrade 实盘导出做逐日对账。

> 为什么要 committed fixture:`data/db/*.parquet` 被 gitignore、仅本地经 junction 可见,
> CI 跑不了引擎。把框架产出快照进仓,测试即可 hermetic、CI 可跑、确定性强。

## 目录结构

```
reference/
├── MANIFEST.md          # 本文件
├── scores.csv           # 框架 compute() 每日因子值(行=日期, 列=资产)。rd 无关。
├── port_scores.csv      # PTrade deploy 的 _quality_momentum_score 离线复算(同一份后复权价)。rd 无关。
│                        #   → 测试比 scores vs port_scores 验证因子逻辑逐位对齐(无需碰 data/db)
├── rd2/                 # rebalance_days=2(对齐 backtest/ptrade/rd2/ 实盘导出)
│   ├── held.csv         # 行=日期, 列=held(引擎 min_hold 调仓后每日持有标的)
│   └── positions.csv    # 行=日期, 列=资产, 值=仓位权重(先 fillna(0) 再 reindex+ffill 后逐日)
└── rd5/                 # rebalance_days=5
    └── ...
```

## 生成参数(锁定)

| 项 | 值 |
|----|----|
| 策略 | `strategy.top1.Top1`(全仓 score 最高单一 ETF) |
| 资产池 | `510300.SH, 159915.SZ, 513100.SH, 518880.SH` |
| 因子 | `quality_momentum`，`window=20` |
| 区间 | `start=2014-01-01` ~ 数据末日 |
| rebalance_mode | `min_hold` |
| rebalance_days | `{2, 5}`（每 rd 一套） |
| transaction_cost_rate | `0.0002`（万2 / c20，对齐实盘导出口径；**不影响 score/held/Top1**，仅记录） |

## 刷新流程

改了策略代码、因子、或 `data/db` 数据后,**必须重跑导出脚本刷新 fixture**:

```bash
uv run python scripts/export_framework_reference.py
```

然后 review CSV diff 并连同本 MANIFEST 的「数据快照」一并提交。

## 数据快照(每次刷新回填)

> 因 `data/db` 不入 git,这里记录「这份 fixture 由哪次数据快照 / 哪个 commit 生成」,作为复现依据。

- 生成日期:2026-06-21
- 生成时所在 commit:9087f89
- `data/db` 末日:2026-06-12
- 各资产 bar 数(start=2014-01-01):510300.SH=3024, 159915.SZ=3023, 513100.SH=3023, 518880.SH=3024
- 引擎 held 实际首尾日:2014-02-07 ~ 2026-06-12(首日 = 21bar 预热后)
