# PTrade 迁移说明 — quality_momentum_top1

把框架内的实盘策略 `strategy/configs/quality_momentum_top1.yaml` 迁移到恒生 PTrade 平台。
交付物：[`deploy/ptrade_quality_momentum_top1.py`](ptrade_quality_momentum_top1.py)，单文件，粘贴进 PTrade 策略编辑器即可运行。

> 维护约定：PTrade 实盘是框架之外的执行端。本目录文档独立于 `strategy_changelog.md`，
> 但任何会改变信号/时序的改动仍应回到框架侧先回测。

---

## 1. 概念映射

| 框架（QuantFramework） | PTrade 实现 | 说明 |
|---|---|---|
| `factors/quality_momentum.py` | `_quality_momentum_score()`（numpy 移植） | 已逐位验证一致，见 §4 |
| `strategy/top1.py`（Top1 全仓） | 回测同步下单；交易模式卖出确认后再 `order_target_value` | 全仓得分最高者，见 §6 |
| `strategy/rebalance.py`（min_hold/fixed_cycle） | `_should_hold()`（直接移植） | 最短持有期 / 固定周期判定 |
| 信号 T 收盘 → 成交 T+1 开盘 | `run_daily(time='09:31')` + `get_history(include=False)` | 见 §2 时序 |
| 后复权 HFQ | `get_history(fq='post')` | 见 §3 |
| 数据层 `query()` | `get_history(count,'1d','close',s,fq='post')` | 逐标的取收盘价 |
| 等权基准 | `set_benchmark('000300.SS')` | ⚠️ 口径不同，见 §5 |
| 持仓状态 `state/*.json` | `g.held` / `g.held_days`（PTrade 序列化持久化） | 跨日/重启持久化 |
| 代码 `.SH` | `.SS` | PTrade 上交所后缀 |

资产池代码转换：
`510300.SH→510300.SS`、`513100.SH→513100.SS`、`518880.SH→518880.SS`、`159915.SZ`（不变）。

---

## 2. 调仓时序如何对齐

框架约定（`strategy_changelog.md §1.1`）：**T 日收盘后**用 T 及之前数据生成 score，**T+1 日开盘**成交。

PTrade 复现方式：
- `run_daily(context, rebalance, time='09:31')` 让 `rebalance` 在**每个交易日开盘**执行。
- `get_history(WINDOW+5, '1d', 'close', s, fq='post', include=False)` 中 `include=False`
  **排除当日 bar**，取到的最新收盘价即「昨日收盘」——等价于「用 T 日及之前数据」。
- 在 09:31 开盘开始执行换仓 ≈ 框架的「T+1 开盘成交」(用 09:31 而非 09:30 整点:部分版本拒绝整点开盘会致 initialize 抛错)。
- 回测模式仍同 bar 同步卖买；交易模式受柜台异步回报约束，实际买入会在旧仓和卖出资金均确认同步后发生，
  通常比 09:31 晚若干秒。这是可靠执行所需的受控时差。

这与框架**实盘** `run_daily.py` 的语义完全一致（早盘用昨日数据出信号、当日开盘成交）。

> 交易模式统一使用对手一档加 0.2% 保护的显式 ETF 限价；无有效正价格时不报单。
> `PRICE_PROTECTION_RATIO` 是最大成交保护范围，可按账户滑点要求调整。

### 最短持有期（rebalance_days）的状态机

`g.held_days` 表示「截至今日开盘，当前持仓已持有的交易日数」，与框架 `holding_days` 对齐：
- 建仓当日置 `0`，次日开盘 `+1` 变 `1`（= 框架建仓日 holding_days=1）。
- `min_hold`：`holding_days < REBALANCE_DAYS` 时压制信号、保持持仓。
- `REBALANCE_DAYS=2` 时：建仓后持有 2 个交易日，第 3 个交易日开盘起才评估切换。
- 重启时不覆盖 PTrade 已序列化的 `g.held_days`；若平台状态确实丢失，则从真实持仓恢复并按 `0`
  天处理。宁可额外持有一个窗口，也不因未知入场日提前换仓。

---

## 3. 复权口径

框架自 2026-05-25 起用**本地 HFQ 后复权**（修正了旧 qfq 未连续化 corporate action 的 bug）。

PTrade 用 `fq='post'`（后复权）。两者复权 baseline 不同（框架锁定最早一日 adj_factor，
PTrade 用各自规则），但**质量动量只依赖窗口内的比率（momentum）与差分（ER 路径长度）**，
绝对 baseline 不影响这些量；后复权都能正确连续化分红/拆分。因此因子值等价。

---

## 4. 已做的验证

**因子数学移植** — 用项目真实的 `factors/quality_momentum.py` 与本文件的 numpy 移植版，
在 200 组随机价格序列上对比「引擎实际使用的最新值」：

```
trials = 200, max |framework - port| = 0.000e+00   → PASS
framework Top1 == port Top1                          → 一致
```

逐位精确匹配，Top1 选择一致。验证脚本逻辑见 git 提交说明。

**执行状态机回归测试（2026-07-10）**：`tests/test_ptrade_execution_state.py` 直接加载部署文件，
覆盖显式 ETF 限价、卖出前禁止买入、卖出拒单、部分卖出、部分买入补仓、重复订单拦截、
柜台查询失败 fail-closed、重启状态恢复。平台复验步骤见 §6。

---

## 5. 与框架的已知差异（如实保留）

1. **rebalance_days = 2（刻意保留，与框架基准 rd=5 不一致——非配置漂移）**：
   框架已于 **2026-06-13** 经「三闸门 + 成本面板」正式评估，**否决 rd=2 转正、回滚 rd=5**
   （`strategy_changelog.md` §3.7 结论 + §3.8；生产 yaml 现为 `rebalance_days: 5`）。
   裁定要点：近年 rd 曲面 U 型非单调（rd=7 最慢档反而最优，"近年奖励快轮动"被证伪）、
   滚动基率 rd=2 非系统性占优（当前领先处 96–98 分位）、成本面板 rd2−rd5 优势在 5bp 翻负。
   **本 PTrade 实盘当前仍跑 `REBALANCE_DAYS = 2`，是经决策刻意保留**（2026-06-15），
   用作框架 rd=5 之外的**外部对照样本**，便于继续向框架侧供 Gate-1 逐年对账数据。
   这是一处**已知、已记录的有意分歧**，不是未记录的配置漂移。
   要与框架基准对齐，改 `REBALANCE_DAYS = 5`；任何切换都须在此处补记日期与理由。
2. **交易成本**：框架回测当前**未扣成本**（§3.2 最高优先级待办）。PTrade 实盘按券商真实费率
   成交；`set_commission`/`set_fixed_slippage` 仅影响 PTrade 回测，已设 ETF 万 2 / 0 滑点，按需调整。
3. **基准**：框架基准是「四资产等权」，PTrade `set_benchmark` 只能设单一标的，本文件用
   沪深300指数（`000300.SS`）。绩效对比口径与框架报告不同，仅供参考。
4. **仓位 98%**：`INVEST_RATIO=0.98` 留 2% 缓冲，避免费用/价格波动导致下单资金不足。
   框架假设 100% 满仓——这会带来轻微跟踪偏差。券商成交可靠时可调到 0.99~1.0。
5. **始终满仓**：Top1 永远全仓四资产中 score 最高者（即使 score 为负，选最不差的），
   无现金/防御档——这是框架的设计，不是迁移引入的。

---

## 6. 部署步骤与待确认项

**部署：**
1. 在 PTrade 新建策略，把 `ptrade_quality_momentum_top1.py` 全文粘贴进编辑器。
2. 策略频率选**日线**；交易时段设为开盘后能执行 `run_daily(time='09:31')`。
3. 先用**回测**跑一段（如 2014-01-01 至今），核对收益曲线与框架回测的量级是否吻合
   （注意成本/基准口径差异，量级接近即可）。
4. 再切**模拟盘**观察 1–2 个调仓周期，确认下单、对账、日志正常。
5. 最后接**实盘**。

**API 细节已用探针在平台上验证（2026-06-12，研究环境 + 5 日回测）：**
- [x] `get_history(..., fq='post', include=False)` 单标的返回带 `close` 列的 DataFrame，
      升序、`DatetimeIndex`、float64；`include=False` 取到「昨日及之前」。
- [x] **代码后缀必须 `.SS`**：`.SH` 静默返回空 DataFrame（len=0）。研究/回测环境一致。
- [x] `context.portfolio`：`total_value` / `cash` / `positions` 均存在；`positions` 为
      `PositionDict`，**键为 `.SS` 形式**；`Position` 有 `amount` / `enable_amount` /
      `cost_basis` / `last_sale_price` / `market_value`（无 `value` / `business_amount`）。
- [x] `order_target_value(security, value)` 可用，返回订单号，回测内同 bar 同步成交；
      内部把 `.SS` 规范化为 `.XSHG`。`order_target(security, 0)` 清仓接口可用。
- [x] `run_daily(context, func, time=...)` 回调签名 `func(context)`，注册并触发正常
      （日线回测中固定在开盘 09:31 触发；故策略取 `time='09:31'`，避免 09:30 整点被部分版本拒绝而 initialize 抛错）。
- [x] `set_benchmark` / `set_commission(type="ETF")` / `set_fixed_slippage` 均支持。

**探针额外暴露并已在策略中处理的两点：**
1. **A 股 T+1**：当日买入当日不可卖（`enable_amount=0`）。正常运行因有最短持有期不会触发；
   策略已加防御——切换前检查待换出标的 `enable_amount>0`，否则本日不切、次日重试。
2. **`get_history` 后复权价 ≠ 真实成交价**（基准不同）。策略用 `order_target_value`
   按市值下单（平台内部按真实价折算股数），不受复权基准差异影响。

### 6.1 交易模式异步执行修复（2026-07-10）

模拟盘 `BatchLog.txt` 记录了 2026-06-22 的首次轮动失败：旧仓卖单因委托价为 `0` 被拒，
代码却立即买入新仓，继而因可用资金仍只有 2180.82 元报资金不足；收盘真实持仓还是
`513100.SS`，但旧代码已把 `g.held` 写成 `159915.SZ`。

现行交易路径已改为：

1. 从实时盘口取对手一档，按 ETF 三位小数生成显式正限价；行情无效时不下单。
2. 换仓只先提交卖单，`run_interval(..., seconds=10)` 轮询真实持仓和未完成订单。
3. 只有旧仓数量归零且目标增量所需资金已可用，才提交买单；不会再同回调连续卖买。
4. 订单号只记为 `pending`，真实目标持仓出现后才更新 `g.held`。
5. 废单、撤单、部分成交、重启、未完成订单和查询异常均走 fail-closed；部分买入次日补足。
6. 收盘清理未完成链路，次日从真实持仓重新计算，不携带失真的乐观状态。

重新部署后仍须在模拟盘至少观察一次完整换仓，确认依次出现“卖出委托已提交”→
“买入委托已提交”→“买入持仓已确认”，且不再出现价格为 0、资金不足或 `g.held` 与持仓分裂。

---

## 7. 回测验证结果（PTrade，2014-01 ~ 2026-06，rd=2，含万2佣金）

在 PTrade 跑全周期回测（日线频率），指标由每日「总资产」序列独立计算，与框架回测对比：

| 指标 | PTrade rd=2（万2佣金） | 框架 rd=2（HFQ, 0.01%） | 解读 |
|------|------|------|------|
| 总收益 | **+2652.76%** | — | 10万 → 275万 |
| 年化 CAGR | **30.54%** | 33.25% | 差 ~2.7pp = 真实佣金拖累 |
| Sharpe | **1.20** | 1.24 | 接近 |
| 最大回撤 | **−27.14%** | −28.44% | 接近 |
| 年化波动 | **25.75%** | ~25.8% | 几乎一致 |
| 成交/换手 | 793 笔 ≈ 397 次切换 ≈ 32次/年 | 平均持有 8.68 日 | 平均持有 ~7.6 日 |

**结论（仅回测/信号层）**：年化波动几乎一致、最大回撤与换手率吻合 → 策略行为与框架逐项对齐；
唯一实质差异（年化低 ~2.7pp）正是 PTrade 计入真实佣金、框架未扣成本所致——**差异即成本，非 bug**。
交易执行层必须另过 §6.1 的模拟盘换仓 gate，不能再用回测通过代替交易模式验收。

> 副产物：这次 PTrade 回测天然就是「含真实成本」的，正是框架 §3.2 一直缺的东西。把
> `REBALANCE_DAYS` 改成 5 再跑，即可得到含成本的 rd=2 vs rd=5 直接对比，用于定夺实盘口径。

**关键修复（据回测日志定位）**：
- `set_volume_ratio(1.0)`：回测默认成交比例 0.25 会导致大单部分成交、资金闲置、旧仓清不掉；放开后足额成交。
- 每个评估日清理所有非 best 持仓：避免部分成交残留的杂仓长期挂着，组合始终收敛到单一标的。
- 回测频率务必设**日线**（分钟频率会慢一个数量级且无收益）。

数据存档：`backtest/ptrade/`（截图 + 交易/持仓明细 CSV；原始 Log.txt 不入版本库）。

---

## 8. 文件清单

- `deploy/ptrade_quality_momentum_top1.py` — PTrade 策略主文件（粘贴即用，已按平台探针校准 + 回测验证）
- `deploy/PTRADE_MIGRATION.md` — 本说明
- `deploy/ptrade_research_probe.py` — 研究环境数据 API 探针
- `deploy/ptrade_backtest_probe.py` — 回测环境交易 API 探针
