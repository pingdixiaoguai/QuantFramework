# 冻结通用510300下行DRAQM正式策略晋升

日期：2026-08-25  
正式策略：`momentum_defender_downside_raqm_weighted_v1`  
晋升权限：用户明确指令  
证据性质：回溯稳健候选，不是独立样本外

## 决策

将冻结通用510300门控晋升为当前正式策略。原
`momentum_defender_confirmation_bridge_raw_gold_v4`改为已取代的回滚检查点，保留原配置、
治理记录、回测和前瞻账本，不复用其状态机或策略ID。

正式策略固定：

- Momentum：四ETF、`quality_momentum 2.0.0`、20日双对数质量动量、5日内部持有；
- 门控资产：仅`510300.SH`；
- 因子：30/40日下行DRAQM，波动率年化地板8%，剪裁`[-3,3]`；
- 分位历史：严格滞后滚动504个观测，至少252个历史值；
- 组合分位：`25%×P30 + 75%×P40`；
- 进入Defender：组合分位不低于55%，连续3日；
- 恢复Momentum：组合分位不高于20%，1日；
- Momentum/Defender锁：30/30日，不可绕过；
- Gold、快速反转、紧急退出：全部关闭；
- 执行：上一收盘信号、下一开盘成交，切换复合退出腿和进入腿。

## 正式检查点

区间为2019-01-18至2026-08-21，共1,841个交易日。

|策略|年化收益|Sharpe|最大回撤|
|---|---:|---:|---:|
|新正式通用门控|50.88%|1.721|-25.50%|
|已取代v4|38.68%|1.700|-19.31%|
|双对数Momentum|33.93%|1.193|-25.50%|
|历史原版Momentum|32.57%|1.159|-25.51%|

新正式策略提高收益和Sharpe，但最大回撤比旧v4扩大约6.19个百分点。它不是低回撤版本，正式
选择接受这一取舍。路径包含17次Defender进入、739个Defender交易日、33次袖套切换、134次
实际候选切换。正式代码与研究选中路径逐日最大误差为0，收益SHA-256为：

```text
f83166623f8749c77404a35f6559839ed3c59f60f6438501ab5c28b59ae6e687
```

## 稳健性边界

- 同权重参数邻域16组，年化Q25为49.74%，Sharpe Q25为1.690；
- 固定leave-one-year最低年化45.98%；
- 删除任一Defender事件最低年化46.97%；
- 三倍费用年化49.75%；
- 全局Reality Check `p=0.1742`，没有达到统计显著。

因此，晋升依据是用户在已验证候选间的明确治理选择，不应描述成统计上证明未来优于所有版本。
参数自2026-08-25冻结，之后只能用新的未观察数据评价。

## Badcase台账

新路径共有17段实际Defender持仓，其中2段相对原Momentum跑输严格超过1个百分点：

- 2021-03-11至2021-05-10：差5.21个百分点，主要来自黄金与海外资产独立上涨；
- 2025-01-06至2025-02-24：差1.68个百分点，主要来自黄金20日上涨5.24%。

两段共同说明：单一510300风险锚不能直接表达黄金的独立趋势，30日Defender锁会延长机会
成本。这是已知并接受的机制边界，不据此重新加入Gold覆盖。

## 运维与回滚

- 正式配置：`strategy/configs/momentum_defender_downside_raqm.yaml`；
- 日跑入口：`run_daily_momentum_defender.py`；
- 正式报告：`experiments/20260825_momentum_defender_downside_raqm_weighted_v1_formal/`；
- 正式主HTML：`formal_backtest.html`，base固定为非对数历史原Momentum；
- 前瞻账本：`state/momentum_defender_downside_raqm_weighted_v1_forward.jsonl`；
- 回滚配置：`strategy/configs/momentum_defender_c2_gold_raqm_w5.yaml`。

新策略ID不继承旧v4持仓文件。首次生产执行前必须用`--dry-run`核对完整目标，并由运维人员
确认旧持仓到新目标的迁移；不得静默复制旧策略状态。
