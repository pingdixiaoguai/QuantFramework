# PTrade 迁移交付 — 给 owner(回 PR #17)

> 2026-06-21 定稿的**可贴交付文本**。对应 PR #17 的四点拆分,尤其 owner comment #4
> 「backtest/ptrade/ 的 CSV 留着,这是我做逐笔对账的原料,先单独发我」。
>
> **随附 4 个 CSV**(逐笔对账原料,万2/c20 口径,2014-01~2026-06 日线回测导出):
> - `backtest/ptrade/rd2/交易详情20260612152933.csv`
> - `backtest/ptrade/rd2/持仓明细20260612152928.csv`
> - `backtest/ptrade/rd5/交易详情20260612210106.csv`
> - `backtest/ptrade/rd5/持仓明细20260612210056.csv`
>
> 以下分隔线内为发给 owner 的正文。

---

老板,按 PR #17 四点拆分,PTrade 这边进展 + 你要的原料:

## 1. 你要的 CSV(point #4)— 附上

逐笔对账原料 4 个文件(万2/c20 口径,2014-01~2026-06 日线回测导出):
- rd=2:`交易详情20260612152933.csv`、`持仓明细20260612152928.csv`
- rd=5:`交易详情20260612210106.csv`、`持仓明细20260612210056.csv`

## 2. 接口契约(point #1)已完成 ✅

按你定的「逐日信号/持仓对账,与框架引擎逐行对齐」,已建成自动化对账 harness 并全绿:

- **因子 score**:框架 `compute()` vs PTrade 实盘 `_quality_momentum_score`,同份后复权价上 12014 单元 **max abs diff = 0.0**(逐位一致)。
- **Top1 选股 + min_hold 调仓规则**:精确一致(规则函数逐输入对齐)。
- **逐日持仓**:引擎 held vs PTrade 持仓明细,rd2 **98.27%** / rd5 **98.20%** 一致;残余 ~2% 全是执行模型差异(框架 T+1 开盘 vs PTrade 同 bar,近平手日相位错 1 个调仓窗),非逻辑 bug —— 逻辑三门已 bit-exact 排除 port 问题,分歧段逐段记录。
- 方式:框架每日产出快照成 committed CSV fixture,对账只读 CSV → 可重复、CI 可跑。

## 3. 顺手做了你想做的逐笔归因对账(point #4 的目的)

你 review 时提的 510300 在 rd=2 负贡献、疑似 qfq 污染 —— 用「同份 parquet 价 + 同一 return-summed 口径,逐列只切一个变量」隔离来源:

**结论:不是脏数据,也不是 regime。** 510300 负贡献 ≈97% 来自度量口径(money-weighted)+ 执行差异,≈3% 才是分红口径。打分走 `fq=post`(干净);P&L 走原始价(分红盲,实测确认),代价仅在 510300、仅 ~0.3pp。

rd=2 510300 增量分解(money −5.45% → 框架 HFQ +4.60%,共 +10.05pp):

| 来源 | 增量 | 占比 |
|---|---|---|
| 度量口径 money→return-summed | +6.65pp | 66% |
| 执行/持仓窗口 | +3.13pp | 31% |
| 分红口径 原始→HFQ | +0.27pp | 3% |

价格核对:510300 持有日 PTrade `最新价` 与 parquet `raw_close` 逐位一致(差<0.005)→ 确认 PTrade 用原始价。
实盘含义:实盘分红进现金 → 实盘 ≈ 框架 HFQ;PTrade 回测是分红盲下界(偏保守),迁移忠实度不受影响。

全文 memo(方法 + 逐年 + 边界)随附:`backtest/ptrade/2026-06-15_attribution_reconciliation.md`。

## 4. 其余两点

- **point #2**(commission_ratio 撤回 → transaction_cost_rate):测试已改造成 PR #23 合并;诊断脚本也已切到新 key。
- **point #3**(rd=2 转正 / 天气预报员):等你框架侧三闸门结果。实盘当前**刻意保留 rd=2 作外部对照**(已显式标注为有意分歧、非配置漂移),听你闸门结论再定去留。

> harness 已为上模拟盘放行。需要我把 memo / 测试输出 / 哪些 CSV 单独整理打包,说一声。
