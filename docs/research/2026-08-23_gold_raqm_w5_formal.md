# C2 + Gold RAQM-W5 正式策略晋升报告

## 决策

2026-08-23按用户明确指令，将`momentum_defender_c2_gold_raqm_w5_v1`晋升为正式策略。
基础C2和Defender实现保持不变；新增机制只允许黄金在C2处于Defender时触发覆盖。

本次晋升是生产决策，不会把回溯检验转化为独立样本外证据。严格前瞻期从2026-08-24开始。

## 策略机制

Gold与Defender整体连续持有净值均计算注册因子：

```text
risk_adjusted_quality_momentum(window=5, vol_floor_annual=0.08)
```

因子使用5日对数收益、5日波动、8%年化波动率地板、Kaufman路径效率和[-3,3]风险调整
动量裁剪。第t日收盘计算，最早第t+1交易日开盘执行。

- 基础C2为Defender且`Gold X - Defender X > 2.20`：下一开盘切入黄金。
- 黄金前5个完整交易日无条件持有，即使基础C2已经恢复Momentum。
- 第6个开盘起：基础C2为Momentum则切原Momentum Top1；否则差值`≤0.60`时回Defender。
- 黄金是唯一覆盖标的；不开杠杆、不做空。

## 历史检查点

区间：2019-01-18至2026-08-21，共1,841个交易日，费用已计入。

|策略|年化收益|年化波动|Sharpe|最大回撤|
|---|---:|---:|---:|---:|
|正式Gold RAQM-W5|56.87%|20.06%|2.343|-12.97%|
|基础C2|51.37%|19.72%|2.199|-12.79%|

黄金入场20次、持有111日。相对基础C2，年化提高5.49个百分点、Sharpe提高0.144、MDD
加深0.17个百分点。

## 稳健性证据

- 参数网格2,714组；852组提高年化、567组同时提高年化与Sharpe。
- PBO 14.7%，样本外排名中位96.4%。
- 扩展式walk-forward收益胜率60%、Sharpe胜率40%。
- 20日成对分块bootstrap 5,000次：年化差为正概率99.4%，95%区间+1.35%至+10.77%；
  Sharpe差为正概率98.2%，95%区间+0.011至+0.289。
- 最优点附近25组中96%提高年化，60%同时提高年化与Sharpe。
- 20次黄金事件14正6负，前两大正事件占正贡献26.2%。
- 删除任一事件后最低年化55.79%、最低Sharpe 2.316。

风险披露：年度块White式多重试验校正p=0.657，未达到统计显著；综合过拟合风险评为
MODERATE。正式晋升依据是用户明确决策，后续必须以冻结参数进行前瞻验证。

## 最新信号

截至2026-08-21收盘、用于2026-08-24开盘的正式5日注册风险调整动量为Gold 0.951、
Defender 0.496，差值0.455，
未达到2.20入场线，因此下一开盘仍为Defender。

## 运维

```bash
# 验证正式检查点
uv run python -m research.run_formal_gold_raqm_w5

# 本地演练，不发送、不写状态
uv run python run_daily_momentum_defender.py --dry-run

# 正式运行
uv run python run_daily_momentum_defender.py
```

回滚时显式使用基础配置：

```bash
uv run python run_daily_momentum_defender.py \
  --config strategy/configs/momentum_defender_c2_main.yaml
```
