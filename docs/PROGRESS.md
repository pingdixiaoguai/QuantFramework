# PROGRESS — PTrade 执行端(`ptrade` 分支)

> 本文件是 `ptrade` overlay 分支的进展入口(PTrade 实盘执行端)。
> 框架研究进展在 `main` 侧,不在此处。分支模型见 [`PTRADE.md`](../PTRADE.md),迁移完成契约见 [`CONTRACT.md`](../CONTRACT.md)。

## 🔄 会话交接(Session Handoff)
> 由 /bye 覆盖式维护为最新。下一轮 PTrade 会话:开 ptrade worktree,先读此区块。

- **当前状态**:PTrade 迁移由「独立 repo」改为「同仓 `ptrade` overlay 分支」。分支 / 两个 worktree / CI 防护 / `data/db` junction 全部搭好并验证;PR #17 拆分基本完成,两个 main-PR(#23 成本测试、#26 CI 防护)待 owner review。
- **下一步**:
  1. owner review/合并 [PR #23](https://github.com/pingdixiaoguai/QuantFramework/pull/23) + [PR #26](https://github.com/pingdixiaoguai/QuantFramework/pull/26)(CI 防护合并后才生效)。
  2. 把归因 memo + `backtest/ptrade/` CSV 发给 owner(PR #17 评论 #4)。
  3. 在 ptrade 建「逐日 score / Top1 / 持仓对账」harness(见 `CONTRACT.md` 待建项)。
- **悬而未决**:实盘刻意保留 rd=2(框架已三闸门回滚 rd=5)作外部对照,长期去留待定;CLAUDE.md 仍称「docs/PROGRESS.md 单一入口」,分支拆分后 main 上无此文件,约定已漂移待修。
- **决策与理由**:
  - 同仓长期 `ptrade` overlay(= main + ptrade 文件),**单向 main→ptrade、永不回流**,main 保持框架单一真相源。(放弃独立 repo 与整支 rebase。)
  - `commission_ratio` 撤回(被 main 的 `transaction_cost_rate` 取代),仅捞出测试改造成 PR #23。
  - 实盘保留 rd=2 作外部对照,文档已显式标注为「有意分歧、非配置漂移」。
- **踩过的坑**:
  - 整支 nyxx-dev rebase→新 main = 冲突地狱(changelog/runner/CLAUDE/PROGRESS 两边都改)且与拆分意图相悖 → 放弃,改干净分支 + cherry-pick 思路。
  - git worktree **不共享 gitignored 文件** → ptrade worktree 无 `data/db` → 对账脚本跑不了;用 `mklink /J` junction 解决(**勿删 main worktree,否则 junction 悬空**)。
  - PTrade 归因里 510300 负贡献 ≠ qfq 污染:实测走原始价(分红盲),但分红只占 ~3%,主因是 money-weighted 度量口径(~66%)+ 执行差异(~31%)。
- **最后更新**:2026-06-17

---

## PTrade 执行端 — 待办(细项)

### 1. 逐日对账 harness(迁移"完成"的硬指标,见 `CONTRACT.md`)
把现有诊断脚本(`scripts/ptrade_dividend_attribution_check.py`、`ptrade_vs_framework_attribution.py`)升级成可重复的「score / Top1 / 持仓」三维逐日对账测试。两种落地方式择一:
- (a) 把 QuantFramework 作为 submodule / 并列 checkout,脚本经 `FRAMEWORK_ROOT` 定位;
- (b) 从框架导出每日 (score, held, positions) CSV 存本仓,对账只读 CSV(去依赖,需手动刷新)。
- 验收:容差内全绿 → 迁移正式完成,可上模拟盘/实盘。

### 2. 交付 owner(PR #17 评论 #4)
- 归因 memo(`backtest/ptrade/2026-06-15_attribution_reconciliation.md`)+ 可贴版结论已生成。
- `backtest/ptrade/` 交易/持仓 CSV 作为逐笔对账原料。

### 3. 待 owner review 的 main-PR
- [PR #23](https://github.com/pingdixiaoguai/QuantFramework/pull/23) — `transaction_cost_rate` 测试(default-0 + 全切换 turnover=2)。
- [PR #26](https://github.com/pingdixiaoguai/QuantFramework/pull/26) — CI 防护(挡 ptrade 文件误合 main)。

### 4. 收尾遗留(建议 `/phase-cleanup` 处理)
- CLAUDE.md(main)修正:补 ptrade 分支模型 + 修「PROGRESS 单一入口」漂移(走 main-PR)。
- main 侧 PROGRESS 归属待定。
- main worktree 里未跟踪的 `backtest/ptrade/`(含本地独有 `Log.txt`)归位到 ptrade worktree——**移而非删**。
