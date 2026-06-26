# 上模拟盘 Checklist — quality_momentum_top1（PTrade）

> 操作前读 `PTRADE_MIGRATION.md`(§6 部署 / §7 回测验证)。本表是可勾选的执行清单。
> 路径默认相对仓库根。策略主文件:`deploy/ptrade_quality_momentum_top1.py`。

## 0. 前置门(已满足,开跑前确认一眼)

- [x] **对账 harness 全绿**:`uv run pytest tests/test_ptrade_reconciliation.py -v` ——
      score 框架 vs 实盘函数 max diff = 0.0、Top1/min_hold 规则精确、逐日持仓 rd2 98.27%/rd5 98.20%。
- [x] **PTrade 回测量级已对齐框架**:年化波动 25.75% vs ~25.8%、回撤/换手吻合(MIGRATION §7)。
- [x] **归因已澄清**:510300 负贡献 = 度量口径+执行,非脏数据(`backtest/ptrade/2026-06-15_attribution_reconciliation.md`)。

> 即:逻辑等价性已闭环。模拟盘要验的是**平台实盘数据馈送 + 下单执行**这最后一段。

## 1. 粘贴策略 + 核对常量

- [ ] PTrade 新建策略,粘贴 `ptrade_quality_momentum_top1.py` **全文**。
- [ ] 逐条核对顶部常量与本次意图一致:
  - [ ] `SECURITIES` = 510300/159915/513100/518880,**后缀 `.SS`**(⚠️ `.SH` 会静默返回空 DataFrame)
  - [ ] `WINDOW = 20`、`REBALANCE_MODE = "min_hold"`
  - [ ] `REBALANCE_DAYS = 2` ←(**刻意保留**,见 §5;要对齐框架基准则改 5)
  - [ ] `INVEST_RATIO = 0.98`、`REBALANCE_TIME = "09:31"`、`BENCHMARK = "000300.SS"`
  - [ ] `COMMISSION_RATIO = 0.00005`(仅回测口径;模拟/实盘按券商真实费率)

## 2. 平台设置

- [ ] 频率 = **日线**(务必;分钟频慢一个数量级且无收益,MIGRATION §7)
- [ ] 交易时段允许 `run_daily(time="09:31")` 在开盘后触发
- [ ] 选**模拟盘**账户;起始资金 ≥ 能买整数手最贵 ETF 的数倍(`INVEST_RATIO=0.98` 已留 2% 缓冲)
- [ ] `set_volume_ratio` 仅回测有效;模拟/实盘会被平台拒绝(`交易不支持...`,**属预期**,见 §3),足额成交改由 §4「首次轮动」验证

## 3. 启动自检(看 initialize 日志)— 已确认 ✅(2026-06-26 模拟盘)

- [x] 出现 `initialized: universe=[...] rd=2 mode=min_hold`
- [x] **无** `initialize FAILED` traceback
- [x] `set_commission / set_fixed_slippage / set_volume_ratio` 三个**回测专用** API:
      模拟/实盘(交易)模式下平台会打出 `交易不支持xxx函数` WARNING —— **这是预期行为,不是版本缺功能**。
      这三个旋钮(佣金/滑点/成交比例)只作用于回测引擎;交易模式走券商真实费率 + 真实盘口撮合,
      noop 不影响策略。每次跑模拟/实盘都会出现,记录、不阻断。
      (回测里 `set_volume_ratio(1.0)` 解决的「部分成交」,在交易模式由真实流动性决定,改由 §4「首次轮动足额成交」验证。)

## 4. 观察 1–2 个调仓周期(MIGRATION §6 step 4)

- [ ] 每交易日 09:31 后有 `scores={...} -> best=X`
- [ ] 调仓日:`买入 X 到目标市值 ...(总资产 ... × 0.98)`;非调仓日:`Top1 未变` 或 `hold window active`
- [ ] 收盘 `after_trading_end`:`收盘持仓: [(X, 数量)] | g.held=X held_days=N`
- [ ] **首次轮动重点**:卖旧仓资金当日即时买入新仓 → 核对日志 buy **足额成交**(MIGRATION §6 注)
- [ ] 持仓始终收敛到**单一** ETF(无部分成交残留的杂仓长期挂着)
- [ ] 跨日/重启后 `g.held` 与券商真实持仓对账正常(代码 _rebalance_impl 第 0 步已处理)

## 5. 与框架信号对账(强校验,可选)

- [ ] 取框架实盘信号(`run_daily.py`)与模拟盘每日 `g.held` 比对,几日内一致
      (harness 已证逻辑等价;这步确认平台 `get_history(fq=post)` 数据馈送 = 框架 parquet)

## 6. 已知分歧 — 不要误判为 bug

- [ ] **rd=2 刻意保留**:框架基准已三闸门回滚 rd=5,实盘 rd=2 作外部对照(MIGRATION §5.1)。是有意分歧。
- [ ] **P&L 分红盲**:PTrade 回测走原始价;**实盘分红进现金 → 实盘 ≈ 框架 HFQ**,回测偏保守(~0.3pp,仅 510300)。
- [ ] **执行相位差**:近似平手日 PTrade 与框架可能持不同标的几天(~2% 的日子),是 T+1 开盘 vs 同 bar 的执行差异,非 bug。
- [ ] **基准不可比**:`set_benchmark` 单一 000300 vs 框架四资产等权 → 基准对比口径不同。

## 7. 模拟 → 实盘 gate

- [ ] 1–2 个调仓周期内,下单 / 对账 / 日志全部正常
- [ ] 首次轮动足额成交已确认
- [ ] rd 去留已定(维持 rd=2 作对照,还是等 owner 三闸门结果改 rd=5)
- [ ] 切实盘前最后确认:账户、资金规模、风控限制、券商真实费率

---

> 出问题先看 `initialize FAILED` / `rebalance FAILED` 的 traceback(代码已包裹打印)。
> API 行为基准见 MIGRATION §6 的已验证清单(2026-06-12 探针)。
