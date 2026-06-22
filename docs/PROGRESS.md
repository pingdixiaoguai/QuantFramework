# PROGRESS — PTrade 执行端(`ptrade` 分支)

> 本文件是 `ptrade` overlay 分支的进展入口(PTrade 实盘执行端)。
> 框架研究进展在 `main` 侧,不在此处。分支模型见 [`PTRADE.md`](../PTRADE.md),迁移完成契约见 [`CONTRACT.md`](../CONTRACT.md)。

## 🔄 会话交接(Session Handoff)
> 由 /bye 覆盖式维护为最新。下一轮 PTrade 会话:开 ptrade worktree,先读此区块。

- **当前状态**:PTrade 迁移由「独立 repo」改为「同仓 `ptrade` overlay 分支」。基建全就绪;两个 main-PR(#23、#26)已合并。**逐日对账 harness 已建成并全绿**(`tests/test_ptrade_reconciliation.py`,2026-06-21):score 逻辑 bit-exact(max diff 0.0)、Top1 / min_hold 规则精确一致、held 端到端一致率 rd2=98.27% / rd5=98.20%(残余为执行模型差异)。**迁移在契约意义上已「完成」,可据此上模拟盘/实盘。**
- **下一步**:
  1. 把归因 memo + `backtest/ptrade/` CSV 发给 owner(PR #17 评论 #4)。
  2. 上模拟盘验证(harness 已为实盘放行)。
  3. 收尾遗留(见下「待办」§4:CLAUDE.md 漂移、main 侧 PROGRESS 归属)。
- **悬而未决**:实盘刻意保留 rd=2(框架已三闸门回滚 rd=5)作外部对照,长期去留待定;CLAUDE.md 仍称「docs/PROGRESS.md 单一入口」,分支拆分后 main 上无此文件,约定已漂移待修。
- **决策与理由**:
  - 同仓长期 `ptrade` overlay(= main + ptrade 文件),**单向 main→ptrade、永不回流**,main 保持框架单一真相源。(放弃独立 repo 与整支 rebase。)
  - `commission_ratio` 撤回(被 main 的 `transaction_cost_rate` 取代),仅捞出测试改造成 PR #23。
  - 实盘保留 rd=2 作外部对照,文档已显式标注为「有意分歧、非配置漂移」。
- **踩过的坑**:
  - 整支 nyxx-dev rebase→新 main = 冲突地狱(changelog/runner/CLAUDE/PROGRESS 两边都改)且与拆分意图相悖 → 放弃,改干净分支 + cherry-pick 思路。
  - git worktree **不共享 gitignored 文件** → ptrade worktree 无 `data/db` → 对账脚本跑不了;用 `mklink /J` junction 解决(**勿删 main worktree,否则 junction 悬空**)。
  - PTrade 归因里 510300 负贡献 ≠ qfq 污染:实测走原始价(分红盲),但分红只占 ~3%,主因是 money-weighted 度量口径(~66%)+ 执行差异(~31%)。
  - 引擎 `positions` 稀疏(执行日行只记当前持仓那只=1.0,其余 NaN):派生逐日持仓必须**先 `fillna(0)` 再 `reindex+ffill`**,顺序反了会按列前向填充旧持仓→多列同时 1.0→`idxmax` 取错。此 bug 曾使对账 held 一致率假性跌到 23%(实为 98%)。
- **最后更新**:2026-06-21

---

## PTrade 执行端 — 待办(细项)

### 1. 逐日对账 harness(迁移"完成"的硬指标)— 已完成 ✅(2026-06-21)
采用方案 B(committed CSV fixture):`tests/test_ptrade_reconciliation.py` + 导出脚本 `scripts/export_framework_reference.py` + 解析模块 `scripts/ptrade_recon/`。逻辑三维 bit-exact 硬门 + held 端到端容差对账门(≥97%,实测 98.2%)全绿。跑/刷新方法见 `CONTRACT.md` §3、`backtest/ptrade/reference/MANIFEST.md`、`docs/plans/2026-06-21_ptrade_reconciliation_harness.md`。

### 2. 交付 owner(PR #17 评论 #4)
- 归因 memo(`backtest/ptrade/2026-06-15_attribution_reconciliation.md`)+ 可贴版结论已生成。
- `backtest/ptrade/` 交易/持仓 CSV 作为逐笔对账原料。

### 3. main-PR(已合并 ✅)
- [PR #23](https://github.com/pingdixiaoguai/QuantFramework/pull/23) — `transaction_cost_rate` 测试(default-0 + 全切换 turnover=2)。已于 2026-06-16 合并。
- [PR #26](https://github.com/pingdixiaoguai/QuantFramework/pull/26) — CI 防护(挡 ptrade 文件误合 main)。已于 2026-06-16 合并,防护已生效。

### 4. 收尾遗留(建议 `/phase-cleanup` 处理)
- CLAUDE.md(main)修正:补 ptrade 分支模型 + 修「PROGRESS 单一入口」漂移(走 main-PR)。
- main 侧 PROGRESS 归属待定。
- main worktree 里未跟踪的 `backtest/ptrade/`(含本地独有 `Log.txt`)归位到 ptrade worktree——**移而非删**。
