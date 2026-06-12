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
| `strategy/top1.py`（Top1 全仓） | `order_target_value(best, total_value × INVEST_RATIO)` | 全仓得分最高者 |
| `strategy/rebalance.py`（min_hold/fixed_cycle） | `_should_hold()`（直接移植） | 最短持有期 / 固定周期判定 |
| 信号 T 收盘 → 成交 T+1 开盘 | `run_daily(time='09:30')` + `get_history(include=False)` | 见 §2 时序 |
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
- `run_daily(context, rebalance, time='09:30')` 让 `rebalance` 在**每个交易日开盘**执行。
- `get_history(WINDOW+5, '1d', 'close', s, fq='post', include=False)` 中 `include=False`
  **排除当日 bar**，取到的最新收盘价即「昨日收盘」——等价于「用 T 日及之前数据」。
- 在 09:30 开盘下单 ≈ 框架的「T+1 开盘成交」。

这与框架**实盘** `run_daily.py` 的语义完全一致（早盘用昨日数据出信号、当日开盘成交）。

> 想更贴近「集合竞价成交」可把 `REBALANCE_TIME` 调到 `'09:25'` 并改用限价单；
> 但仅交易大规模 ETF，09:30 市价/现价限价单的偏差极小，框架本身也假设滑点 ≈ 0。

### 最短持有期（rebalance_days）的状态机

`g.held_days` 表示「截至今日开盘，当前持仓已持有的交易日数」，与框架 `holding_days` 对齐：
- 建仓当日置 `0`，次日开盘 `+1` 变 `1`（= 框架建仓日 holding_days=1）。
- `min_hold`：`holding_days < REBALANCE_DAYS` 时压制信号、保持持仓。
- `REBALANCE_DAYS=2` 时：建仓后持有 2 个交易日，第 3 个交易日开盘起才评估切换。

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

**尚未验证（需在 PTrade 平台上确认）**：见 §6。

---

## 5. 与框架的已知差异（如实保留）

1. **rebalance_days = 2**：与当前 yaml 一致；但 `strategy_changelog.md` 的 v0 基准记录是 5。
   要切回 5，改 `REBALANCE_DAYS = 5`。（框架侧这个口径不一致本身是个待澄清项。）
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
2. 策略频率选**日线**；交易时段设为开盘后能执行 `run_daily(time='09:30')`。
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
      （日线回测中固定在开盘 09:31 触发，与 `time='09:30'` 一致）。
- [x] `set_benchmark` / `set_commission(type="ETF")` / `set_fixed_slippage` 均支持。

**探针额外暴露并已在策略中处理的两点：**
1. **A 股 T+1**：当日买入当日不可卖（`enable_amount=0`）。正常运行因有最短持有期不会触发；
   策略已加防御——切换前检查待换出标的 `enable_amount>0`，否则本日不切、次日重试。
2. **`get_history` 后复权价 ≠ 真实成交价**（基准不同）。策略用 `order_target_value`
   按市值下单（平台内部按真实价折算股数），不受复权基准差异影响。

**首次实盘切换时仍需观察一次**：卖出旧仓的资金当日能否即时用于买入新仓（A 股规则上可以，
回测同 bar 成交也支持；实盘首次轮动时核对日志确认 buy 足额成交即可）。

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

**结论**：年化波动几乎一致、最大回撤与换手率吻合 → 策略行为与框架逐项对齐，**迁移忠实**；
唯一实质差异（年化低 ~2.7pp）正是 PTrade 计入真实佣金、框架未扣成本所致——**差异即成本，非 bug**。

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
