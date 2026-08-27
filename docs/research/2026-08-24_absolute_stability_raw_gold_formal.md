# 正式v3：绝对稳定性状态 + Raw Gold RAQM-W5

> 历史回滚记录：v3已于2026-08-24被无锁确认与Top1快速反转桥接v4替代，不再是默认
> 生产信号。本文保留v3当时的规则、检查点与晋升证据，不按v4结果改写。

## 正式决策

2026-08-24按用户明确指令，将
`momentum_defender_absolute_stability_raw_gold_v3`晋升为正式策略。

旧正式v2保留在
`strategy/configs/momentum_defender_c2_gold_raqm_w5_log_qm_v2.yaml`，其状态、治理记录和
前瞻账本不覆盖。v3使用新的策略ID和前瞻账本。

## 完整规则

### Momentum

四只ETF使用`quality_momentum 2.0.0`：20日对数收益乘对数路径Kaufman ER，Top1至少持有
5个交易日。

### Momentum/Defender状态

- 沪深300ETF的120日对数收益必须为正；
- Momentum上一收盘实际持有ETF的120日对数收益也必须为正；
- 两者同时为正才希望持有Momentum，否则希望Defender；
- Momentum最短持有20日，Defender最短持有40日；
- 紧急退出只看当前Momentum持仓：5日对数收益为负，且20日下行波动率超过该ETF自身
  严格滞后扩展历史q95时，允许打破Momentum持有锁。

### Defender

Defender内部月度红利/低波轮动、区间网格、Rogers–Satchell仓位上限、三因子满仓迟滞与
10年国债补位保持不变。

### Raw Gold RAQM-W5

Gold和Defender连续净值计算5日Raw RAQM：对数收益除以窗口波动率，再乘对数路径ER；不设
波动率地板、不做剪裁。

- 基础状态为Defender且`Gold−Defender > 2.0`：下一开盘进入Gold；
- Gold前5个完整交易日无条件持有；
- 第6个开盘起，基础状态恢复Momentum则交回Momentum；
- 否则差值`≤0.75`时退出Gold回到Defender。

## 正式检查点

区间2019-01-18至2026-08-21，共1,841个交易日，费用已计入。

|累计收益|年化收益|年化波动|Sharpe|最大回撤|
|---:|---:|---:|---:|---:|
|1,605.89%|47.45%|18.41%|2.203|-16.77%|

状态审计：

- 基础Defender入场18次；
- 基础Defender状态1,167日；
- 基础状态切换35次；
- Gold入场31次、持有180日；
- 最终候选切换157次；
- 逐日收益SHA-256：`312d4881eadfd45dbc8da9e6cde007129a984050cc68c086ebec3dc62a81c7dc`。

逐一删除任一Gold事件后，最低年化46.44%、最低Sharpe 2.171、最差MDD -16.77%。

## 证据边界

切换机制研究累计3,641条全局唯一路径；候选绝对Deflated Sharpe稳定，但相对旧正式基线的
增量没有通过年度Reality Check。Raw Gold专项测试258个参数ID、183条唯一路径：PBO
16.45%，Walk-forward收益/Sharpe胜率80%/60%，Bootstrap年化差95%区间为+0.20%至
+10.29%，但Reality Check `p=0.5864`不显著。

因此本次晋升是明确生产选择，不将回溯结果描述为独立样本外证明。严格前瞻期从新策略ID的
第一未观察执行日开始。

## 正式报告

以原Momentum为base的HTML：
[`formal_vs_original_momentum.html`](../../experiments/20260824_momentum_defender_absolute_stability_raw_gold_v3_formal/formal_vs_original_momentum.html)。

这里的“原Momentum”使用历史原版20日简单收益MOM乘价格路径ER，5日最短持有，不使用
Defender或Gold。全区间年化32.57%、Sharpe 1.159、MDD -25.51%。正式v3内部Momentum仍
使用双对数`quality_momentum 2.0.0`，两者通过独立因子和配置隔离。
