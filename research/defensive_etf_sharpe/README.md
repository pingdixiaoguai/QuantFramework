# Defensive ETF Sharpe Research

独立研究目录：面向长期做多的防守型 ETF 动态配置策略。

## 已确认约束

- 回测起点：2013 年；终点默认使用数据可用的最新交易日。
- 初始资金：0 元。
- 资金流：每月第一个交易日投入 20,000 元。
- 标的：512890.SH（红利低波）、511260.SH（10年国债）、511360.SH（短融）和511880.SH（货币）四只ETF；不包含黄金、宽基、成长或科技ETF。
- 方向：仅做多，可持有现金；不使用杠杆或融券。
- 交易粒度：日级或更低频率。
- 交易成本：默认单边 0.05%，ETF 按 100 股整数手成交。
- 目标：在年化收益率不低于5%的候选策略中，优先提高年化Sharpe。

## 研究原则

每日收盘只使用当日及此前的后复权收盘价计算20日反转因子。四只ETF的固定基准权重为35%/40%/15%/10%，因子排名通过“全池50%倾斜”调整目标权重。每月首个交易日的新增资金按上月末目标权重买入；组合单边偏离达到10%后于下一交易日开盘再平衡，每月最多一次，再平衡单笔不足10,000元不执行。

## 当前研究产物

- `engine.py`：带月度入金、现金、整数手和交易成本的资本路径回测器。
- `factor_allocation.py`：20日反转因子、横截面排名和权重倾斜公式。
- `rebalance_timing.py`：每日目标权重、月初锚定目标序列、月初入金和最小成交门槛执行规则。
- `threshold_rebalance.py`：组合偏离触发、月初固定或偏离复合触发及每月再平衡次数约束。
- `research.py`：生成策略资本路径、交易与逐标的信号审计。
- `deliver.py`：生成当前10%偏离策略的标准CSV交付包和QuantStats HTML回测报告。
- `drift_vs_calendar_analysis.py`：10%日频锚策略弱于月初基线的归因分析。
- `confirm05d_research.py`：偏离连续5日确认变体的回测对比。
- `noise_mitigation_research.py`：日频目标噪声缓解实验（平滑、慢锚、慢因子、弱倾斜）。
- `score_mechanism_research.py`：得分-权重映射机制实验（z-score指数/sigmoid/加法倾斜、波动调整反转）。

## 初步结论

当前正式版本为 `defensive_etf_reversal20_global_tilt50_drift10_monthly_cap1`。最新完整结果见 `deliverable/core_metrics.csv` 和 `deliverable/backtest_report.html`；阈值与因子均经过全样本探索，仍应作为研究结果而非未来收益保证。
