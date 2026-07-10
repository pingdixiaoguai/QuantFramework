# PROGRESS — PTrade 执行端(`ptrade` 分支)

> 本文件是 `ptrade` overlay 分支的进展入口(PTrade 实盘执行端)。
> 框架研究进展在 `main` 侧,不在此处。分支模型见 [`PTRADE.md`](../PTRADE.md),迁移完成契约见 [`CONTRACT.md`](../CONTRACT.md)。

## 🔄 会话交接(Session Handoff)
> 由 /bye 覆盖式维护为最新。下一轮 PTrade 会话:开 ptrade worktree,先读此区块。

- **当前状态**:**首次模拟盘轮动失败已定位并完成本地修复,待重新部署验证**(2026-07-10)。`BatchLog.txt` 证实旧代码把回测同 bar 同步成交错误套到交易模式:卖单价格 0 被拒后仍立即买入,造成资金不足和 `g.held`/真实持仓分裂。现已改为显式 ETF 限价 + 卖出确认/资金到账后再买入 + 真实持仓确认后更新状态;执行状态机 8 项测试和原有对账 5 项均通过。
- **下一步**:
  1. 把修复后的 `deploy/ptrade_quality_momentum_top1.py` 全文重新粘贴到 PTrade 模拟盘并重启策略。
  2. 等待下一次换仓,按 checklist §4 核对“卖出提交→资金/持仓同步→买入提交→持仓确认”完整链路。
  3. 至少观察 1–2 个调仓周期;旧三类错误不复现后再走 checklist §7「模拟 → 实盘」gate。
- **悬而未决**:
  - 实盘 rd=2 **暂时保留观察**(框架已三闸门回滚 rd=5,PTrade 留 rd=2 作外部对照),长期去留待 owner。
  - 修复后的异步执行状态机尚未经过 PTrade 平台完整换仓复验;当前只有本地柜台替身测试通过。
- **决策与理由**:
  - PR #17 记为**已关闭**(非合并):repo 实为 CLOSED 且 owner 明确要求关掉;overlay 永不回流 main,本就不该 merge。(用户初称"已合并",经核对更正。)
  - 三条 `交易不支持...` WARNING 判为**预期**:set_commission/slippage/volume_ratio 均回测专用旋钮,交易模式走真实费率 + 真实盘口撮合,被拒属正常,非版本缺功能。
  - 同仓长期 `ptrade` overlay(= main + ptrade 文件),**单向 main→ptrade、永不回流**,main 保持框架单一真相源。
- **踩过的坑**:
  - PTrade 回测里的 `order_target*` 同 bar 同步成交不能外推到交易模式;交易柜台订单/持仓/资金异步更新,必须显式限价并管理 pending 状态。
  - 模拟盘初始化日志必有三条 `交易不支持xxx函数` —— 别误判为故障(已在 checklist §3 注明)。
  - Bash 工具是 Git Bash:commit message 用 heredoc(`-F - <<'EOF'`),**勿**用 PowerShell here-string `@'...'@`(上轮污染了 commit subject,已 amend 修掉)。
  - git worktree **不共享 gitignored 文件** → ptrade worktree 无 `data/db` → 对账脚本跑不了;用 `mklink /J` junction 解决(**勿删 main worktree,否则 junction 悬空**)。
  - 引擎 `positions` 稀疏:派生逐日持仓必须**先 `fillna(0)` 再 `reindex+ffill`**,顺序反了会多列同时 1.0→`idxmax` 取错(曾使 held 一致率假性跌到 23%,实为 98%)。
  - PTrade 归因 510300 负贡献 ≠ qfq 污染:主因 money-weighted 口径(~66%)+ 执行差异(~31%),分红仅 ~3%。
- **最后更新**:2026-07-10

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
