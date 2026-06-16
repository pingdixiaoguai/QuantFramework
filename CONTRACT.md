# 迁移接口契约 — “完成”的定义

QuantFramework PR #17 review 中维护者定的标准（原话）:

> deploy/(PTrade 移植):走独立 repo,不进 main。main 要保持单一执行平台真相源。
> **接口契约 = 逐日信号/持仓对账测试,与框架引擎逐行对齐才算完成。**

本文件把这个标准落成可执行的验收条件,并记录已达成 / 待建部分。

## 1. 对账测试要做什么

对**同一区间、同一参数**(universe、window=20、rebalance_days、min_hold),逐个交易日比对:

| 维度 | 框架侧来源 | PTrade 侧来源 | 通过判据 |
|------|-----------|--------------|---------|
| 因子 score | `factors/quality_momentum.py` 经引擎 future-info guard 算出的每日值 | 策略 `_quality_momentum_score()` 用 `get_history(fq="post")` 算出的每日值 | 数值逐位一致(已验证,§4 of MIGRATION) |
| Top1 选择 | 引擎每日 `held_asset` | 策略每日 `g.held` / 下单目标 | 每日选股一致 |
| 持仓序列 | 引擎 `positions`(开仓执行日) | PTrade 持仓明细每日持仓 | 在执行模型差异容差内一致 |

**容差来源(已知口径差,需在对账中显式扣除,不算失败):**

1. 执行模型:框架 T+1 开盘成交 vs PTrade 日线同 bar —— 切换日的归属可能差 1 个 bar。
2. 复权:框架 HFQ(含分红) vs PTrade 原始价 P&L(分红盲)—— 仅影响 P&L 量级,不影响 score/选股(score 两边都后复权)。
3. 停牌/交易日历对齐:`get_history(count)` 按交易日历对齐,个别标的停牌日缺 bar。

## 2. 已达成

- **因子数学**:200 组随机序列上 `framework` vs `port` 最大误差 0.000e+00,Top1 一致(`deploy/PTRADE_MIGRATION.md §4`)。
- **全周期回测风险面对齐**:年化波动 25.75% vs ~25.8%,回撤/换手吻合(`§7`)。
- **归因口径对账**:510300 负贡献分解为 度量口径 ~66% + 执行 ~31% + 分红 ~3%,确认无危险后复权污染(`backtest/ptrade/2026-06-15_attribution_reconciliation.md`)。

## 3. 待建（本仓的主要工程）

**自动化逐日对账 harness**,把上面的「score / Top1 / 持仓」三维比对做成可重复运行的测试:

- `scripts/ptrade_vs_framework_attribution.py`、`ptrade_dividend_attribution_check.py` 已经是雏形,但它们 `import backtest.runner` 且读 `data/db/*.parquet` —— **依赖 QuantFramework 仓库**,无法在本仓独立运行。
- 落地方式(择一,待定):
  1. 把 QuantFramework 作为 **git submodule** 或并列 checkout,脚本通过 `FRAMEWORK_ROOT` 环境变量定位。
  2. 从框架导出每日 (score, held, positions) 为 CSV 存到本仓,对账测试只读 CSV(去依赖,但需手动刷新)。
- 验收:`逐日对账测试` 在容差内全绿 → 迁移正式“完成”,可据此上模拟盘/实盘。

> 现状:本仓为 scaffold 阶段——文件已就位、契约已写清、对账脚本为雏形;完整 harness 是下一步。
