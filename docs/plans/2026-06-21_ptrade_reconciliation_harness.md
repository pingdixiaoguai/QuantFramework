# 实施计划:PTrade 逐日对账 harness(方案 B)

> 创建于 2026-06-21。对应 `docs/PROGRESS.md` 待办第二项、`CONTRACT.md` §3「待建」。

## 目标

把 score / Top1 / 持仓 三维逐日对账做成**可重复、CI 可跑**的 pytest 测试,容差内全绿即满足 `CONTRACT.md` 对迁移「完成」的定义:

> 接口契约 = 逐日信号/持仓对账测试,与框架引擎逐行对齐才算完成。

绿 = PTrade 移植版与框架研究版每日决策一致的客观凭证,也是日后改策略的回归安全网,是上模拟盘/实盘前的最后一道技术闸门。

## 方案选择(B):committed CSV fixture

- ptrade 是**同仓 overlay 分支**,框架代码(`backtest/runner.py`、`factors/`)与 `data/db` junction 都在同一工作树,`from backtest.runner import run` 本就能跑。
- 但 **CI 跑不了引擎**(`data/db/*.parquet` gitignored,仅本地 junction 可见)。故:本地跑一次引擎导出小体积 CSV fixture 提交进仓;pytest 只读 `fixture CSV + PTrade 导出 CSV` → 测试 hermetic、CI 可跑、确定性强。代价:换参数/数据要手动重导 fixture。

## 关键技术认知

- 框架因子 `compute()` 是**因果的**(`pct_change`/`rolling`,每日值只用过去),全序列一次算 = 逐日截断算,score 复算不必模拟引擎 future-info guard 循环。
- PTrade 的 `_quality_momentum_score` 与框架公式**代数等价**(已被 200 随机序列 0.000e+00 证实);本测试把它升级成「真实价格序列上的回归测试」。
- score 完全**尺度不变**(momentum 是比率、ER 是位移/路径之比),两侧后复权锚点差异在窗口内抵消 → 用同一份 parquet `raw_close*adj_factor` 复算两侧,差异趋近浮点 0。
- 边界:本 harness 证「数学/逻辑」对齐;PTrade 真实 `get_history(fq='post')` 是否 = parquet,是**数据馈送等价性**,由 `deploy/ptrade_research_probe.py` 在 PTrade 环境单独验,不在本测试范围。

## 参数与容差(锁定)

- **rd 集合**:`{2, 5}`,对齐已有 `backtest/ptrade/rd2/`、`rd5/` 两份导出。
- **区间**:`2014-01-01 ~ 导出末日`(~2026-06)。
- **成本率**:fixture 用 `transaction_cost_rate=0.0002`(万2 / c20,匹配导出口径)。成本率不影响 score/held/Top1 选择,仅为可复现而记录。
- **三条容差**(逐字落自 `CONTRACT.md` §1,对账时显式扣除、非失败):
  1. 执行模型:框架 T+1 开盘 vs PTrade 日线同 bar → 切换日归属可差 ±1 bar。
  2. 复权:框架 HFQ 含分红 vs PTrade 原始价 → 只影响 P&L,不影响 score/选股。
  3. 停牌/日历:个别标的停牌日缺 bar。

## 任务分解(6 任务 = 6 commit;依赖:1 → (2‖3) → 4 → (5‖6))

### 任务 1 — 锁定参数与目录骨架
- 新建 `backtest/ptrade/reference/rd2/`、`reference/rd5/`(放 committed fixture CSV)。
- 新建 `scripts/ptrade_recon/`(含 `__init__.py`)。
- 新建 `backtest/ptrade/reference/MANIFEST.md`:记录 fixture 生成参数 + 数据快照标识 + 生成 commit(data/db gitignored,必须记可复现依据)。
- 验收:目录就位、约定写清,`uv run pytest` 不报新错。

### 任务 2 — 框架参考导出脚本 `scripts/export_framework_reference.py`
- score:每资产读 `data/db/<code>.parquet`,`close = raw_close*adj_factor`,喂 `factors.quality_momentum.compute(df)` → 宽表 `scores.csv`(行=日期,列=资产)。
- held/positions:`backtest.runner.run(config)`,config = `strategy.top1.Top1` + 四资产 + `quality_momentum(window:20)` + `rebalance_days=rd` + `rebalance_mode=min_hold` + `start=2014-01-01` + **`transaction_cost_rate=0.0002`(新 key,非 `commission_ratio`)**。`result.positions` 先 `.reindex(daily_returns.index).ffill()`(positions 仅执行日有行),再 `idxmax` → `held.csv`;ffill 宽表 → `positions.csv`。
- 每 rd 一套 → `reference/rd{2,5}/{scores,held,positions}.csv`;生成参数写进 MANIFEST。
- 验收:两套 CSV 落地;held 与现有归因脚本一致。

### 任务 3 — PTrade 导出解析模块 `scripts/ptrade_recon/parse.py`
- 抽出统一:gbk 编码读取、`.SS→.SH` 映射、持仓 idxmax(多仓取市值最大)。
- `parse_holdings(rd) -> (held: Series, positions: DataFrame)`,schema 与任务 2 fixture 完全对齐(date 索引、资产列用 `.SH`)。
- 数据形状:`持仓明细` 若仅变动日有行,需 `reindex(交易日).ffill()` 补逐日。
- 验收:`parse_holdings(2/5)` 非空、schema 对齐。

### 任务 4 — 逐日对账测试 `tests/test_ptrade_reconciliation.py`(核心)

> **设计修订(基于任务 2 实测,见下「实施发现」)**:held 维度**不能用精确 / ±1bar 硬门** ——
> 框架(T+1 开盘成交)与 PTrade(同 bar)执行模型本就不同,叠加 min_hold 相位 + 近似平手日
> 的数据馈送微差,分歧会成多日小段(rd5 实测最长 ~5 个交易日的合法执行差)。故把测试分成
> **「逻辑精确硬门」+「持仓容差对账门」**两层。

参数化 `rd ∈ {2,5}`:

**逻辑精确硬门(用同一份 parquet,隔离策略逻辑,必须 bit-exact):**
1. **score**:fixture `scores.csv` vs 从 parquet 用 deploy 真实 `_quality_momentum_score`(`importlib` 加载)滚动复算 → 每日 `abs(diff)<1e-9`(仅两侧非 NaN 日)。
2. **Top1(调仓前)**:`scores.csv` 每日 argmax vs PTrade-port 复算 argmax → 精确相等(抓 tie-break / argmax 口径差)。
3. **min_hold 规则**:deploy `_should_hold` vs 框架 `should_hold_position`,在 (有无持仓 × holding_days × rd × mode) 输入网格上 → 精确相等(隔离调仓时序逻辑,不含执行噪声)。

**持仓容差对账门(端到端,含 PTrade 真实数据/执行,作回归 tripwire):**
4. **held(调仓后)**:fixture `held.csv` vs `持仓明细` 解析的每日持仓,在**索引交集**上(两侧均有值,不 reindex 引 NaN)比对 → **一致率 ≥ 97%**(实测 98.2%,留余量;我的 fillna bug 曾使其跌到 23%,正是此门要挡的回归);**所有分歧连续段(起止/长度/标的)全部 `log`** 入测试输出,供人工审阅,不静默吞。

- 验收:`uv run pytest tests/test_ptrade_reconciliation.py -v` 两 rd 全绿 + 打印 score 最大误差 / Top1 错配数 / held 一致率 + 分歧段清单。**此步绿 = 迁移完成。**

### 任务 5 — 修静默失效死 key + 旧脚本对齐
- `commission_ratio` → `transaction_cost_rate`(PR #23 合并后前者已被忽略)。改 `ptrade_vs_framework_attribution.py`、`ptrade_dividend_attribution_check.py`。
- 两脚本 CSV 解析切到 `scripts/ptrade_recon/parse.py`(去重)。
- 低风险确认:两脚本框架侧只取 `held`(成本无关)+ parquet 收益归因,不读 `daily_returns`,修 key 不改归因输出 —— 跑一遍确认数值不变。
- 验收:原结论不变;grep 确认仓内无残留 `commission_ratio`(历史文档引用除外)。

### 任务 6 — 文档回填
- `CONTRACT.md` §3:「待建」→「已建」,补「怎么跑测试 / 怎么刷新 fixture」。
- `backtest/ptrade/README.md`:新增一节说明 `reference/` fixture 与对账测试。
- `docs/PROGRESS.md`:待办第二项 → 完成;更新会话交接「下一步」+「最后更新」。
- 验收:三处文档与代码一致。

## 实施发现(2026-06-21,任务 2 期间)

- **导出脚本曾有 fillna 顺序 bug**:引擎 `positions` 稀疏(执行日行只记当前持仓那只=1.0,
  其余 NaN)。若先 `reindex+ffill` 再 `fillna(0)`,ffill 会按列前向填充旧持仓 → 多列同时 1.0
  → `idxmax` 取错。**正确顺序:先 `fillna(0)` 补完整 one-hot,再 reindex+ffill**(旧诊断脚本本就如此)。
  该 bug 使 held 一致率假性跌到 23%;修复后 rd2=98.27% / rd5=98.20%。
- **score 复算已验 == 引擎内部**(truncate<=t 取 last,误差 <1e-12)。`compute` 因果 → 全序列一次算即可。
- **held 残余分歧 = 执行模型差异**:实测残余段框架自身 argmax 与 PTrade 持仓一致,框架 held 因
  T+1 开盘 + min_hold 相位偏离自身 argmax 数日 —— 印证归因里的「执行差异 ~31%」,非 port bug。
- **覆盖**:框架 held 自 2014-02-07 起(START=2014-01-01 后 21bar 预热),PTrade 自 2014-01-02;
  交集 3002 天 / PTrade 3023 天 ≈ 99.3%,首月 ~20 天无框架对照(可接受)。

## 风险与备注

- 任务 4 ③ 若出现残余错配 = harness 发挥价值:实盘移植与框架真有偏差(很可能是 `_should_hold` 时序移植 bug),需定位修复后再绿。
- `持仓明细` 数据形状(逐日 vs 仅变动日)在任务 3 实现时确认,影响 reindex 策略。
- fixture 刷新依赖本地 `data/db` junction;**勿删 main worktree**(junction 会悬空)。
