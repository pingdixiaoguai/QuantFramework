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

## 3. 已建 ✅（2026-06-21）

**自动化逐日对账 harness** 已落地:`tests/test_ptrade_reconciliation.py`。采用方案 B
（committed CSV fixture）—— 因 ptrade 为同仓 overlay 分支,框架代码与 `data/db` junction
就在同一工作树,但 `data/db` 被 gitignore、CI 跑不了引擎,故把框架产出快照成 CSV 提交进仓,
测试只读 committed CSV(framework fixture + PTrade 导出),CI 可跑。

测试分两层:

- **逻辑精确硬门**(同一份后复权价,隔离策略逻辑,bit-exact):
  - score:框架 `compute()` vs deploy 真实 `_quality_momentum_score` —— 实测 12014 单元 `max abs diff = 0.0`。
  - Top1:两侧每日 argmax 精确一致。
  - min_hold 规则:deploy `_should_hold` vs 框架 `should_hold_position`,输入网格全一致。
- **持仓容差对账门**(端到端 tripwire):引擎 held vs PTrade 持仓明细,索引交集一致率
  **≥ 97%**(实测 rd2=98.27% / rd5=98.20%),所有分歧连续段写入测试日志供审阅。残余分歧
  经查为执行模型差异(框架 T+1 开盘 vs PTrade 同 bar + min_hold 相位 + 近平手日数据馈送微差),
  非 port 逻辑 bug(逻辑由三个硬门保证)。

**怎么跑**:`uv run pytest tests/test_ptrade_reconciliation.py -v -s`（`-s` 看分歧段清单)。
**怎么刷新 fixture**(改了策略/因子/`data/db` 后):`uv run python scripts/export_framework_reference.py`,
然后 review CSV diff 并回填 `backtest/ptrade/reference/MANIFEST.md` 的数据快照。

**验收达成**:harness 容差内全绿 → 迁移在契约意义上“完成”,可据此上模拟盘/实盘。

> 设计与排查记录见 `docs/plans/2026-06-21_ptrade_reconciliation_harness.md`(含一个 fillna
> 顺序 bug 的发现:它曾使 held 一致率假性跌到 23%,正是容差对账门要挡的回归)。
