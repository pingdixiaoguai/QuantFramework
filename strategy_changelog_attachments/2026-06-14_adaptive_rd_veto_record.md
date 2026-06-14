---
# adaptive rebalance_days 研究否决记录

> 归档日期: 2026-06-14
> 来源: `2026-06-14_adaptive_rd_predictability.md`
>
> 否决对象: 基于近期表现/近期 P&L 在 `rebalance_days in {2,3,5,7}` 间自适应切换的研究方向。
>
> 否决原因: “过去最优 rd 预测未来最优 rd”的前置闸门未通过。Gate A 顶四分位领先后的前瞻中位差在全部 K/价差对为负；Gate B 近期最优重复命中率全部低于 25% 随机基线；Gate C 可实现选择器在 5bp 且含 meta 换手下全部弱于固定 rd=5。
>
> 代码/配置处置: 不修改生产配置、不修改引擎、不写入 `strategy_changelog.md`。本记录仅作为研究否决存证。

---

## 证据摘要

- 原生序列锚点校验通过: rd=2/3/5/7 均与已归档汇总口径一致。
- Gate A: 顶四分位前瞻中位差全部为负；正收益占比全部低于 50%。
- Gate B: 近期最优重复命中率为 9.52% 到 20.29%，全部低于四选一 25% 随机基线。
- Gate C: 可实现选择器在 @5bp 下相对固定 rd=5 的年化差为 -3.85pp 到 -7.14pp。

## 存档文件

- `2026-06-14_adaptive_rd_predictability.md`
- `2026-06-14_adaptive_rd_predictability_daily_series_validation.csv`
- `2026-06-14_adaptive_rd_predictability_intersection_calendar.csv`
- `2026-06-14_adaptive_rd_predictability_intersection_excluded_dates.csv`
- `2026-06-14_adaptive_rd_predictability_gate_a_summary.csv`
- `2026-06-14_adaptive_rd_predictability_gate_b_hit_rates.csv`
- `2026-06-14_adaptive_rd_predictability_gate_c_metrics.csv`
