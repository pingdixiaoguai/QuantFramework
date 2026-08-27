# Phase 3：条件低风险分仓的稳健性审计

## 候选来源与证据限制

Phase 2 只按 D/V 选出的 `SAFE_T80_W50` 提高 Sharpe 与 Top10 平均回撤，但两条严格尾部 Gate 略未通过。其相邻候选 `SAFE_T75_W50` 在 D、V、T、全期 Sharpe 均不降，并回溯通过全部尾部 Gate。由于它是在查看 Phase 2 全期结果后识别，本阶段将其明确标为结果后候选；不把 T 当作新鲜 OOS，也不修改生产配置。

## 冻结审计

不再搜索新机制，只审计 `SAFE_T75_W50`：

1. 阈值邻域：0.70 / 0.75 / 0.80 / 0.85，预算固定50%；
2. 预算邻域：40% / 50% / 60%，阈值固定0.75；
3. 风险字段消融，阈值0.75、预算50%：
   - `PRICE_ONLY`：下行LPM、CVaR、振幅、跳空；
   - `TUSHARE_ONLY`：Amihud、成交额冲击、份额申赎、NAV溢价；
   - `LIQUIDITY_ONLY`：Amihud与成交额冲击；
   - `FLOW_PREMIUM_ONLY`：份额申赎与NAV溢价；
   - `FULL`：八字段综合风险；
4. 单边1bp/5bp、D/V/T/全期、Top10、相同窗口、滚动36月；
5. 2,000次配对移动区块bootstrap，区块20/60/120日，同时记录 Sharpe 增量与Top10平均深度改善。

## 稳健性判断

- 核心候选必须通过 Phase 1 全部 Gate；
- 阈值和预算邻域中至少一半配置必须同时实现全期 Sharpe 不降与Top10平均深度改善至少1pp；
- `FULL` 必须不弱于 `PRICE_ONLY` 的 Sharpe/Top10二者之一，且 Tushare-only 因子必须产生非零持仓差异，否则“新增字段”没有提供可识别信息；
- bootstrap 仅评估不确定性，不以95%显著性作为硬部署出口；若95%区间跨零，候选最多进入 shadow。

