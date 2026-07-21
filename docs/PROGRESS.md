# PROGRESS — PTrade 执行端(`ptrade` 分支)

> 本文件是 `ptrade` overlay 分支的进展入口(PTrade 实盘执行端)。
> 框架研究进展在 `main` 侧,不在此处。分支模型见 [`PTRADE.md`](../PTRADE.md),迁移完成契约见 [`CONTRACT.md`](../CONTRACT.md)。

## 🔄 会话交接(Session Handoff)
> 由 /bye 覆盖式维护为最新。下一轮 PTrade 会话:开 ptrade worktree,先读此区块。

- **当前状态**:**拆单修复已上模拟盘,7/21 全现金建仓验证通过(买入半程)**。7/21 首次建仓因单笔 210 万股 > 交易所 100 万股上限被后端撤单;已在 `deploy/ptrade_quality_momentum_top1.py` 买卖两侧按 `MAX_ORDER_SHARES=100万` 拆成整百股子单(改走 `order(sec, amount, limit_price=)`,仅交易模式,回测路径不变)。重部署后同日建仓:510300 增量 207.4 万股拆 3 笔全额成交、无超限撤单、`买入持仓已确认`。已 commit `e4e0e84` + push origin/ptrade;ptrade 全套 15 测试通过。
- **下一步**:
  1. **首次真正换仓时**验证三项(有提醒 chip `task_f7e91912`):卖出拆单、159915 深市单笔上限、§6.1 完整异步 gate(卖→买→确认三段)。触发:rd=2 建仓满 2 个交易日后信号切换。
  2. 换仓结果回填 `PTRADE_MIGRATION.md §6.2` 的「仍待观察 ⏳」→「已验证 ✅」+ `checklist §4` 勾选。
  3. 三项全过后评估 `checklist §7`「模拟 → 实盘」gate。
- **悬而未决**:
  - 卖出拆单 / 159915 深市上限 / 完整异步换仓 gate — 本次全现金建仓未覆盖,待首次换仓。
  - `MAX_ORDER_SHARES=100万` 是沪市 510300 实测值;深市 159915 上限若更低,调小该常量重部署(只是子单变多)。
  - 实盘 rd=2 长期去留待 owner(框架已三闸门回滚 rd=5,PTrade 留 rd=2 作外部对照 + 加速换仓验证)。
- **决策与理由**:
  - 拆单用按股数的 `order()`(非 `order_value`):股数是上限单位,拆分边界最精确;回测路径保留 `order_target*` 维持 §7/对账口径。
  - 买入完成判定收紧为「所有在途子单终结后再收敛」:防多子单下某子单先失败就提前判完成(单笔行为不变)。
  - rd=2 维持(7/20 复核):换仓周期短,加速异步链路验证 + 外部对照。
- **踩过的坑**:
  - PTrade 单笔限价申报上限 100 万股/份:全仓 Top1 + 千万级资金买 4–9 元 ETF 天然超限,无配置绕法,必须拆单。(6/22 首次失败是价格 0 异步 bug,7/21 首次失败是超限——两个不同坑。)
  - 数据同步需 `TUSHARE_TOKEN`(在 `.env`,已可用);本地 `data/db` 曾滞后到 7/14,同步后到 7/17。
  - 从 6/1 起跑回测需前置预热:因子 min_history=21、`query` 只加载 [start,end],起点须提前 ~1 月让因子在 6/1 当天预热完毕再切片报告。
  - PTrade 回测里的 `order_target*` 同 bar 同步成交不能外推到交易模式;交易柜台订单/持仓/资金异步更新,必须显式限价并管理 pending 状态。
  - 模拟盘初始化日志必有三条 `交易不支持xxx函数` —— 别误判为故障(已在 checklist §3 注明)。
  - Bash 工具是 Git Bash:commit message 用 heredoc(`-F - <<'EOF'`),**勿**用 PowerShell here-string `@'...'@`。
  - git worktree **不共享 gitignored 文件** → ptrade worktree 无 `data/db` → 对账脚本跑不了;用 `mklink /J` junction 解决(**勿删 main worktree,否则 junction 悬空**)。
  - 引擎 `positions` 稀疏:派生逐日持仓必须**先 `fillna(0)` 再 `reindex+ffill`**,顺序反了会多列同时 1.0→`idxmax` 取错。
- **最后更新**:2026-07-21

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

### 5. 模拟盘部署验证(进行中)— 2026-07-21
- [x] 异步执行修复版部署(commit 7395370,§6.1)+ rd=2 维持(§5.1)。
- [x] 单笔超限拆单修复(commit e4e0e84,§6.2):买卖两侧按 MAX_ORDER_SHARES 拆单。
- [x] 7/21 买入拆单验证通过(510300 拆 3 笔全额成交)。
- [ ] 首次真正换仓验证:卖出拆单 / 159915 深市上限 / §6.1 完整异步 gate(chip task_f7e91912)。
- [ ] 三项全过 → checklist §7 模拟→实盘 gate。
细项跟踪见 `deploy/SIM_DEPLOYMENT_CHECKLIST.md` §4/§7。
