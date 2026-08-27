# 510300下行RAQM：Momentum/Defender重设计与稳健寻参

## 最终结论

在固定20日双对数Momentum（对数收益 × 对数路径Kaufman ER）和既有Defender的前提下，
只使用`X>=20`的510300下行风险调整质量动量决定顶层袖套切换，并把Momentum与Defender
锁定期都限制在20—30个交易日。最终保留的研究候选为：

`momentum_defender_downside_raqm_weighted_v1`

它使用30/40日下行RAQM严格滞后分位的25%/75%加权。2019-01-18至2026-08-21历史年化
50.88%、Sharpe 1.721，达到年化45%的硬门槛。该候选不自动替换生产策略。

## 机制

对窗口`X`，先计算：

```text
R_X   = log(close_t / close_{t-X})
ER_X  = |R_X| / sum(|daily_log_return|, X)
vol_X = std(daily_log_return, X) * sqrt(X)
RAQM_X = clip(R_X / max(vol_X, 8% * sqrt(X/252)), -3, 3) * ER_X
D_X    = max(0, -RAQM_X)
```

`D_X`只在趋势为负时取正值。随后把当前`D_X`与此前504个有效观测比较，形成严格滞后历史
分位；当前收盘信号最早在下一开盘生效。最终分数为：

```text
score = 25% * percentile(D_30) + 75% * percentile(D_40)
```

状态规则：

1. `score >= 0.55`连续3日，且Momentum已持有满30日，下一开盘切Defender；
2. `score <= 0.20`，且Defender已持有满30日，下一开盘恢复Momentum；
3. 两个锁定期都不可被绕过；没有紧急破锁、5日桥接或Gold覆盖；
4. 切换日使用旧候选退出腿与新候选进入腿的复合收益，费用沿用已有精确交易接口。

## 完整历史与固定分段

|时期|候选年化|候选Sharpe|候选MDD|Log-QM Momentum年化|Momentum Sharpe|
|---|---:|---:|---:|---:|---:|
|Development 2019—2022|44.16%|1.710|-23.36%|26.52%|1.081|
|Validation 2023—2024|59.64%|1.665|-25.50%|37.79%|1.166|
|Recent 2025—2026-08|57.26%|1.891|-21.73%|48.44%|1.485|
|Full|50.88%|1.721|-25.50%|33.92%|1.193|

候选降低了年化波动并显著提高收益与Sharpe，但全历史最大回撤仍为-25.50%，没有改善
Momentum基线的最差单次回撤。因此该机制是收益/风险调整改善，不是尾部回撤彻底解决方案。

## 参数稳定性

三轮实验合计测试72,144个候选ID，按逐日收益哈希全局去重后为42,010条路径。

- 最终参数的同权重邻域共16组，年化达到45%的比例为100%；
- 邻域年化Q25/中位数为49.74%/49.87%；
- 邻域Sharpe Q25/中位数为1.690/1.695；
- 单窗口审计显示收益主要集中在30—40日区域，45—50日明显较弱；
- 30/40日加权候选相对最终单40日近邻候选具有更高年化、全样本Sharpe和最差分段Sharpe，
  因此在第三轮近似并列时选择加权版本。

上述最终选择是第三轮后的近似并列治理判断，不是独立样本外选择；审计文件明确保留这一状态。

全样本年化达到45%的候选中，最高表面Sharpe为1.801，但其参数邻域仅50%仍能达到45%年化，
因此按“稳健优先”约束拒绝。第一轮稳定单40日候选Sharpe为1.728，最终加权候选为1.721，
仅低0.006；加权候选年化更高，且固定留一年、删单事件和最差分段证据更强，因而接受这点
Sharpe差值来降低对单一窗口的依赖。

## 压力测试

- 固定候选逐次删除任一年：最低年化45.98%，最低Sharpe 1.601；
- 17段与Momentum不同的Defender事件中，14段正贡献、3段负贡献；
- 逐一删除任一事件：最低年化46.97%，最低Sharpe 1.620；
- 删除最大1个正事件：年化46.97%；删除最大2个后降至43.84%；
- 前两大正事件占全部正向log excess的36.94%，仍存在事件集中风险；
- 3倍费用：年化49.75%、Sharpe 1.692；
- 20日配对分块Bootstrap相对Momentum的年化差95%区间为+4.06至+31.07个百分点，
  Sharpe差区间为+0.187至+0.957。

## 多重试验与证据边界

- 全局CSCV-PBO：37.01%；
- CSCV赢家样本外击败Momentum比例：90.26%；
- 全局walk-forward收益/Sharpe胜率：60%/60%；
- 年度块Reality Check：`p=0.1742`。

Reality Check没有达到常用显著性标准，因此不能声称42,010条路径中选出的候选已被统计证明
优于Momentum。Bootstrap支持最终固定候选，但它不替代对整个搜索族的多重试验校正。

## 最新检查点

截至2026-08-21开盘状态为Defender，下行RAQM加权分位为0.9107，状态原因是继续持有，
并非当日新切换。

版本化配置：
[`research/configs/momentum_defender_downside_raqm_selected.yaml`](../../research/configs/momentum_defender_downside_raqm_selected.yaml)。

完整机器证据、逐日状态和HTML报告：
[`experiments/20260824_momentum_defender_downside_raqm_final_selection/`](../../experiments/20260824_momentum_defender_downside_raqm_final_selection/)。
