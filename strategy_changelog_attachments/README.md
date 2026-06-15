# Strategy Changelog Attachments

本目录是策略研究的诊断/否决归档；不进入 `strategy_changelog.md` 正文的研究记录存于此。

## 索引

| 目录名 | 日期 | 主题一句话 | 对应 changelog 条目 | 结论 |
| --- | --- | --- | --- | --- |
| `2026-05-20_shape_signal_diagnostic/` | 2026-05-20 | DTW/路径形状信号(com)诊断 | 研究 | 部分调查 |
| `2026-05-21_drawdown_return_attribution/` | 2026-05-21 | 回撤与收益归因 | 研究 | 诊断 |
| `2026-05-25_hfq_rebuild/` | 2026-05-25 | HFQ 复权口径修复测量 | §3.4 | 已部署(Bug修复) |
| `2026-06-02_periodic_reeval/` | 2026-06-02 | 周期性重评/固定周期扫描 | 研究 | 否决 |
| `2026-06-02_fixed_cycle_research_archive.md` | 2026-06-02 | fixed_cycle 作废降级 | §3.5 | 否决 |
| `2026-06-04_cost_tau_scan/` | 2026-06-04 | 成本模型扫描 + τ 否决 | §3.6 / τ 归档 | 成本已兑现/τ否决 |
| `2026-06-05_absolute_momentum_diagnostic/` | 2026-06-05 | 绝对动量现金 overlay | 研究 | 否决 |
| `2026-06-13_rd2_vs_rd5_evaluation/` | 2026-06-13 | rd2 vs rd5 三闸门 + 成本面板 | §3.7/§3.8 | 否决 rd2/回滚 rd5 |
| `2026-06-14_adaptive_rd_predictability/` | 2026-06-14 | 自适应 rd/动量的动量可预测性 | 研究 | 否决(三 Gate 一致证伪) |
| `2026-06-15_close_execution_variant/` | 2026-06-15 | T+1 收盘成交 vs T+1 开盘成交变体诊断 | 研究 | 轻微提升但回撤恶化/2024-09 拖累 |
| `2026-06-15_ytd_attribution_rebalance_scan/` | 2026-06-15 | 2026 YTD 损失归因: rebalance_days 扫描 | 研究 | rd=7 优于 rd=5; 指定事件B未被消除 |

## 命名规范

新研究产物按 `{YYYY-MM-DD}_{event_slug}/` 建目录，目录内文件按 `{YYYY-MM-DD}_{event_slug}_{artifact}.{md|csv}` 命名。`event_slug` 使用小写下划线，并尽量与 changelog 条目或研究主题对应。

## 重生成说明

逐日序列和大型中间 CSV 不进 git，可由 `scripts/` 下对应脚本按同口径重生成。本目录 `.gitignore` 排除了这些可再生大件。

例外：`2026-05-21_momentum_strategy_daily_returns.csv` 是配置 Project 的 ERC 接口交付物，必须进 git、保留在顶层，路径不可变。高频复用的基准序列如 `momentum_strategy_daily_returns` / `quality_momentum_top1_daily_positions` 优先通过脚本重生成。
