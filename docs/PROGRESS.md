# PROGRESS — PTrade 执行端(`ptrade` 分支)

> 本文件是 `ptrade` overlay 分支的进展入口(PTrade 实盘执行端)。
> 框架研究进展在 `main` 侧,不在此处。分支模型见 [`PTRADE.md`](../PTRADE.md),迁移完成契约见 [`CONTRACT.md`](../CONTRACT.md)。

## 🔄 会话交接(Session Handoff)
> 由 /bye 覆盖式维护为最新。下一轮 PTrade 会话:开 ptrade worktree,先读此区块。

- **当前状态**:PTrade 迁移由「独立 repo」改为「同仓 `ptrade` overlay 分支」。基建全就绪;两个 main-PR(#23、#26)已合并。**逐日对账 harness 已建成并全绿**(`tests/test_ptrade_reconciliation.py`,2026-06-21):score 逻辑 bit-exact(max diff 0.0)、Top1 / min_hold 规则精确一致、held 端到端一致率 rd2=98.27% / rd5=98.20%(残余为执行模型差异)。**迁移在契约意义上已「完成」,可据此上模拟盘/实盘。**
- **下一步**:
  1. 上模拟盘验证(harness 已为实盘放行)。
  2. 收尾遗留(见下「待办」§4:CLAUDE.md 漂移、main 侧 PROGRESS 归属)。
- **owner round-trip 已闭环(2026-06-23)**:[PR #17](https://github.com/pingdixiaoguai/QuantFramework/pull/17) **已关闭**(owner 要求 deploy 不进 main、移对应流程——符合 overlay「永不回流」铁律,本就不该 merge,关闭即预期结果);2026-06-21 交付物(对账完成 + 归因结论 + 4 CSV + memo,见 `backtest/ptrade/2026-06-21_owner_handoff.md`)已归位。
- **悬而未决**:实盘 rd=2 **暂时保留观察**(框架已三闸门回滚 rd=5,PTrade 留 rd=2 作外部对照),长期去留待定。(原记的「CLAUDE.md 称 PROGRESS 单一入口」漂移经 2026-06-21 复核为**幻影**:无任何 CLAUDE.md 含此声明,分支模型已在 `PTRADE.md` 记清——见 §4。)
- **决策与理由**:
  - 同仓长期 `ptrade` overlay(= main + ptrade 文件),**单向 main→ptrade、永不回流**,main 保持框架单一真相源。(放弃独立 repo 与整支 rebase。)
  - `commission_ratio` 撤回(被 main 的 `transaction_cost_rate` 取代),仅捞出测试改造成 PR #23。
  - 实盘保留 rd=2 作外部对照,文档已显式标注为「有意分歧、非配置漂移」。
- **踩过的坑**:
  - 整支 nyxx-dev rebase→新 main = 冲突地狱(changelog/runner/CLAUDE/PROGRESS 两边都改)且与拆分意图相悖 → 放弃,改干净分支 + cherry-pick 思路。
  - git worktree **不共享 gitignored 文件** → ptrade worktree 无 `data/db` → 对账脚本跑不了;用 `mklink /J` junction 解决(**勿删 main worktree,否则 junction 悬空**)。
  - PTrade 归因里 510300 负贡献 ≠ qfq 污染:实测走原始价(分红盲),但分红只占 ~3%,主因是 money-weighted 度量口径(~66%)+ 执行差异(~31%)。
  - 引擎 `positions` 稀疏(执行日行只记当前持仓那只=1.0,其余 NaN):派生逐日持仓必须**先 `fillna(0)` 再 `reindex+ffill`**,顺序反了会按列前向填充旧持仓→多列同时 1.0→`idxmax` 取错。此 bug 曾使对账 held 一致率假性跌到 23%(实为 98%)。
- **最后更新**:2026-06-23

---

## PTrade 执行端 — 待办(细项)

### 1. 逐日对账 harness(迁移"完成"的硬指标)— 已完成 ✅(2026-06-21)
采用方案 B(committed CSV fixture):`tests/test_ptrade_reconciliation.py` + 导出脚本 `scripts/export_framework_reference.py` + 解析模块 `scripts/ptrade_recon/`。逻辑三维 bit-exact 硬门 + held 端到端容差对账门(≥97%,实测 98.2%)全绿。跑/刷新方法见 `CONTRACT.md` §3、`backtest/ptrade/reference/MANIFEST.md`、`docs/plans/2026-06-21_ptrade_reconciliation_harness.md`。

### 2. 交付 owner(PR #17 评论 #4)— 已闭环 ✅(PR 已关闭,2026-06-23)
- 交付文本:`backtest/ptrade/2026-06-21_owner_handoff.md`(对账完成 + 归因结论,回 PR #17 四点)。
- 已附:rd2/rd5 交易详情+持仓明细 4 个 CSV + 归因 memo(`2026-06-15_attribution_reconciliation.md`)。
- **PR #17 已关闭**(2026-06-23):owner 要求 deploy 不进 main、移对应流程(符合 overlay「永不回流」铁律);该 PR 本就不应 merge,关闭即预期结果。
- point #3(rd=2 去留):三闸门已跑完、框架回滚 rd=5;PTrade **rd=2 暂时保留观察**作外部对照,长期去留待定。

### 3. main-PR(已合并 ✅)
- [PR #23](https://github.com/pingdixiaoguai/QuantFramework/pull/23) — `transaction_cost_rate` 测试(default-0 + 全切换 turnover=2)。已于 2026-06-16 合并。
- [PR #26](https://github.com/pingdixiaoguai/QuantFramework/pull/26) — CI 防护(挡 ptrade 文件误合 main)。已于 2026-06-16 合并,防护已生效。

### 4. 收尾遗留 — 复核后基本 moot(2026-06-21)
- ~~修「PROGRESS 单一入口」漂移~~ **幻影**:全仓无任何 CLAUDE.md 含此声明;root CLAUDE.md(main 与 ptrade 逐字一致)Entry Points 是 DESIGN.md/specs/,不提 PROGRESS。无可修。
- ~~补 ptrade 分支模型~~ **已记于 `PTRADE.md`**(铁律/overlay 清单/工作流/CI 防护)。是否再在 main 的 CLAUDE.md 加一句「存在 overlay 分支」属 main 治理,与「main 保持 ptrade-agnostic」权衡,**留给 owner 定**(若要则走 main-PR)。
- main 侧 PROGRESS 归属:main 无 PROGRESS.md 是**正确**的(PROGRESS 为 ptrade 专属;框架进展在 DESIGN.md 决策日志/changelog)。无 action。
- ~~main worktree 未跟踪 `backtest/ptrade/` 归位~~ **moot**:该 worktree 下无 backtest/ptrade 未跟踪文件。
