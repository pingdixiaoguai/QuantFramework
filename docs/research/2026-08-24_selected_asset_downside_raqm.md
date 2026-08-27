# 仅在510300/518880为Momentum Top-1时检查下行RAQM

## 结论

请求中的`510330`不在当前Momentum资产池，也没有本地历史数据；结合前文，本实验把它解释
为现有的`510300.SH`，没有新增ETF或改写Momentum基线。

在“只有当前开盘将持有510300或518880时才允许DRAQM触发Defender，创业板和纳指完全不
检查”的约束下，最终双启用研究候选为：

`momentum_defender_selected_asset_draqm_v1`

2019-01-18至2026-08-21：252日年化47.51%、日历CAGR 45.36%、Sharpe 1.618、最大回撤
-25.50%。它显著优于纯Log-QM Momentum，但弱于上一轮通用510300门控的50.88%/1.721，
因此不替换现有研究候选，更不自动修改生产策略。

## 精确机制

两个资产都沿用注册RAQM正则化和严格滞后504日历史分位：

```text
R_X    = log(close_t / close_{t-X})
ER_X   = |R_X| / sum(|daily_log_return|, X)
vol_X  = std(daily_log_return, X) * sqrt(X)
RAQM_X = clip(R_X / max(vol_X, 8% * sqrt(X/252)), -3, 3) * ER_X
D_X    = max(0, -RAQM_X)
P_X    = D_X相对此前504日（不含当日）的经验分位；D_X=0时P_X=0
```

上一收盘计算，下一开盘使用。

### 当前Momentum Top-1为510300

```text
score_510300 = 25% * P(DRAQM30) + 75% * P(DRAQM40)
```

- `score >= 0.35`：下一开盘候选进入Defender，确认1日；
- `score <= 0.25`：Defender候选恢复Momentum，确认1日。

### 当前Momentum Top-1为518880

```text
score_518880 = 25% * P(DRAQM20) + 75% * P(DRAQM40)
```

- `score >= 0.45`连续5日：下一开盘候选进入Defender；
- `score <= 0`：Defender候选恢复Momentum，确认1日。

黄金恢复线为0表示20/40日组合已经完全没有下行RAQM，而不是“历史0分位附近”。

### 状态语义

- Momentum最短持有20日，Defender最短持有23日；两个锁都不可绕过；
- 入场只看当前开盘的Momentum Top-1；若为159915或513100，不累计信号、不触发Defender；
- Defender期间采用`sticky_entry_asset`：持续监控最初触发Defender的那只资产，达到它自己的
  恢复线后才恢复；
- 没有5日桥接、Gold覆盖或紧急破锁。

## 绩效比较

|策略|年化|Sharpe|MDD|
|---|---:|---:|---:|
|纯Log-QM Momentum|33.92%|1.193|-25.50%|
|指定资产DRAQM|47.51%|1.618|-25.50%|
|通用510300 DRAQM|50.88%|1.721|-25.50%|

固定分段：

|时期|年化|Sharpe|
|---|---:|---:|
|Development 2019—2022|40.48%|1.645|
|Validation 2023—2024|49.51%|1.439|
|Recent 2025—2026-08|63.31%|1.888|

选中锁参数周围6个直接邻域全部达到45%年化；邻域年化Q25为46.87%，Sharpe Q25为1.601。
3倍费用下年化46.42%、Sharpe 1.590。不过逐次删除任一年后的最低年化只有41.55%，说明
“全历史超过45%”不能外推成所有时期都超过45%。

## 资产归因与关键风险

相对纯Momentum共有11段Defender事件：

- 510300触发9段，8段正贡献、1段负贡献，合计log excess +0.514；
- 518880触发仅2段，两段均为正，合计log excess +0.192；
- 关闭黄金政策后，年化从47.51%降至44.60%、Sharpe从1.618降至1.527；
- 但黄金单独启用只有33.40%年化、1.197 Sharpe，几乎没有独立优势。

因此黄金政策在联合状态机中确实贡献历史收益，但阈值只由两次实际事件支持，参数不确定性很高。
这也是不晋升的主要原因之一。

## 全局过拟合审计

三轮共测试25,101个候选ID，按逐日收益哈希全局去重为3,766条路径。

- CSCV-PBO：32.03%；
- 全局walk-forward相对Momentum收益/Sharpe胜率：100%/100%；
- 对Momentum的年度块Reality Check：`p=0.0864`；
- 对通用510300门控：`p=0.9882`，没有任何统计优势；
- 配对分块Bootstrap相对Momentum的年化差95%区间为+2.10至+25.88个百分点，Sharpe差
  为+0.132至+0.809；
- 相对通用510300门控的Bootstrap年化与Sharpe差区间均跨0，均值为负。

最终判断：该版本是用户指定机制下的最佳稳健研究候选，但没有证据支持替换通用510300门控。

版本化配置：
[`research/configs/momentum_defender_selected_asset_draqm_selected.yaml`](../../research/configs/momentum_defender_selected_asset_draqm_selected.yaml)。

完整机器证据：
[`experiments/20260824_momentum_defender_selected_asset_draqm_final_selection/`](../../experiments/20260824_momentum_defender_selected_asset_draqm_final_selection/)。
