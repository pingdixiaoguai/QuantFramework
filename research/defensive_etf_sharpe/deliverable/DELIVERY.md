# 防守型ETF策略标准交付包

## 当前正式策略

正式版本固定为 `defensive_etf_reversal20_global_tilt50_drift10_monthly_cap1`。四只ETF基准权重为512890.SH 35%、511260.SH 40%、511360.SH 15%、511880.SH 10%；每日使用20日反转因子做全池50%排名倾斜。每月首个交易日新增20,000元按上月末目标权重买入；组合单边偏离达到10.0%后，下一交易日开盘再平衡，每月最多一次，单笔不足10,000元不执行。

## 核心结果

全期时间加权年化收益6.38%，年化波动4.57%，Sharpe 1.378，最大回撤-5.06%，期末资产5,534,622元，估算交易成本89,358元。

完整策略公式、执行顺序、基线定义、图表和限制见 `backtest_report.html`。明细数据见 `core_metrics.csv`、`annual_metrics.csv`、`daily_performance.csv`、`daily_target_weights.csv`、`strategy_signals.csv` 与 `strategy_trades.csv`。
