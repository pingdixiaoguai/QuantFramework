# 双sleeve事后shadow中心：最终审计设计

中心为phase14的`P14_03`：初始资本70%配置风险延迟Top1 sleeve，30%配置`SAFE_POS(T=0.70,budget=20%)`防御sleeve；两条净值独立增长，不做日度sleeve再平衡。该中心在读取T后从5个跨段三目标候选中确定，只能作为shadow规则。

最终审计要求：phase14网格跨段三目标通过率≥50%；中心D/V/T/FULL严格改善Sharpe、年化和Top10平均回撤；5bp方向为正；最大回撤不恶化1pp；滚动36个月Sharpe领先≥60%；原策略最深十次窗口至少改善7次；官方基准精确复现。另报告年度胜率和20/60/120日区块bootstrap，不因bootstrap结果再调参数。
