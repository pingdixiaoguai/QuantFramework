# Strategy Research Attachments

本目录保存 `quality_momentum_top1` 的研究报告、诊断表与可复核的 CSV。它不是变更日志：是否部署、当前参数与历史变更，以根目录的 [`strategy_changelog.md`](../strategy_changelog.md) 和生产配置为准；这里记录的是形成或否决决策的证据。

目录名沿用历史名称 `strategy_changelog_attachments`，但内容既包括 changelog 的佐证，也包括未进入 changelog 的只读研究。除特别注明外，研究均未修改生产策略。

## 先读这里：研究结论地图

| 结论类别 | 已形成的结论 | 对生产策略的含义 |
| --- | --- | --- |
| 已采用的测量改进 | HFQ 数据口径与 `transaction_cost_rate` 成本模型已完成校验并入库。 | 后续比较应优先使用 HFQ、显式单边成本与实际执行换手。 |
| 保留当前设计 | `rebalance_days=5`、`momentum(20) × ER(20)`、点对点 20 日动量均经后续研究支持。 | 不因这些研究改动参数；当前配置仍是 20 日窗口、rd=5。 |
| 已关闭的方向 | 固定周期/时间维度抗换手、hysteresis τ、绝对动量转现金、自适应 rd、DTW/路径形状族、“单日跳变下的信号估计量 winsorize 稳健化”，以及回归 slope 动量线中的 A3/A1。 | 不应把历史表面优势或未经验证的机制重新当作候选部署依据，除非有新的预注册研究明确推翻相应关闭条件。 |
| 仅诊断、尚无动作 | 收益/回撤归因、T+1 收盘成交、2026 YTD rd 扫描、外部持仓差异。 | 它们说明现象或提出假设，不构成参数变更授权。 |
| 未预注册观察 | A2（`slope × R²(26)` 等权）在 2026-06-29 诊断中数字上优于 B，但不是本次预注册机制假设，且回撤更深、窗口本身脆弱。 | 不构成部署出口；若重开，必须作为独立预注册研究处理。 |

## 实验总览

`状态`描述的是该研究线的治理结论，而非某张表的单次样本排名。报告中的起止日、成交时点、成本和数据截面不同，不能直接把年化或 Sharpe 跨报告比较。

| 日期 | 研究线 | 核心问题与结论 | 状态 | 主要报告 / 数据 |
| --- | --- | --- | --- | --- |
| 2026-05-20 | 形状信号初诊 | 检查 ER、动量与 `conv`/`com` 的前瞻分层；后续发现当时 `conv` 误取回归常数项，报告仅保留为历史线索，不作为决策证据。 | 后续关闭 | [`报告`](2026-05-20_shape_signal_diagnostic/2026-05-20_shape_signal_diagnostic.md) |
| 2026-05-21 | 收益与回撤归因 | `159915.SZ` 是主要收益来源；深回撤不全是换手造成，短持有亏损换仓是重要但非唯一现象。 | 诊断 | [`报告`](2026-05-21_drawdown_return_attribution/2026-05-21_drawdown_return_attribution.md) |
| 2026-05-25 | HFQ 重建复测 | 修正复权口径后，rd=5 的“少量收益换更浅回撤/更低换手”治理方向未翻转；沪深 300 的历史低贡献部分来自旧 qfq 污染。 | 已采用（数据口径） | [`报告`](2026-05-25_hfq_rebuild/2026-05-25_hfq_phase3_measurement.md) |
| 2026-06-02 | 周期性重评 / fixed cycle | 固定日历重评的表面优势依赖相位；干净数据、成本和 OOS 后不成立。`fixed_cycle` 候选 YAML 已移除。 | 否决 / 归档 | [`扫描`](2026-06-02_periodic_reeval/2026-06-02_periodic_reeval_scan.md)、[`作废存证`](2026-06-02_fixed_cycle_research_archive.md) |
| 2026-06-04 | 成本模型与 hysteresis τ | 验证零成本闸门并把 `transaction_cost_rate` 作为引擎能力入库；低 τ 需不现实的极高成本才划算，高 τ 虽有表面优势但会错过趋势启动，故 τ 不部署。 | 成本模型已采用；τ 否决 | [`报告`](2026-06-04_cost_tau_scan/2026-06-04_cost_tau_report.md) |
| 2026-06-05 | 绝对动量现金 overlay | 各种窗口、阈值、两种接入顺序与现金利息假设均未稳健优于原 Top1；避险并未换来足够的风险调整收益。 | 否决 | [`报告`](2026-06-05_absolute_momentum_diagnostic/2026-06-05_absolute_momentum_diagnostic.md) |
| 2026-06-13 | rd=2 vs rd=5 三闸门 | rd=2 的近年表现是孤立/episode 集中的优势，成本与滚动窗口不支持转正；由 rd=2 回滚至 rd=5。 | rd=5 已恢复 | [`报告`](2026-06-13_rd2_vs_rd5_evaluation/2026-06-13_rd2_vs_rd5_evaluation.md) |
| 2026-06-14 | 自适应 rd 可预测性 | “过去表现最优的 rd 会继续最优”在前瞻排序、持续性和可实现选择器三道 Gate 中均失败。 | 否决 | [`报告`](2026-06-14_adaptive_rd_predictability/2026-06-14_adaptive_rd_predictability.md) |
| 2026-06-15 | T+1 收盘成交变体 | 全期收益/Sharpe 略升，但最大回撤更深、OOS Sharpe 变弱，且 2024-09 反弹窗口明显拖累；不据此改成交规则。 | 诊断 | [`报告`](2026-06-15_close_execution_variant/2026-06-15_close_execution_variant.md) |
| 2026-06-15 | 2026 YTD rd 扫描 | 在该短窗口中 rd=7 优于 rd=5/10，但指定的 3 月拥挤抖动并未消除；改善来自整体路径而非单一事件。 | 诊断，不改参数 | [`报告`](2026-06-15_ytd_attribution_rebalance_scan/2026-06-15_ytd_attribution_rebalance_scan.md) |
| 2026-06-15 | 与外部策略的持仓差异 | 两者没有持续的选标的差异，主要是调仓相位错位及两次 whipsaw；外部策略呈 5 日倍数的固定周期节奏。 | 方向假设 | [`报告`](2026-06-15_holdings_diff_vs_external/2026-06-15_holdings_diff_vs_external.md) |
| 2026-06-16 | ER 乘子消融 | `× ER` 相比纯动量在全期、OOS、5bp 和分歧 episode 中均有增量；ER 单独使用却不佳，说明它是互补交互项。维持 `k=1`；`k=1.5` 仅作观察。 | 保留现行 ER | [`判读`](2026-06-16_er_ablation/2026-06-16_er_ablation_diagnosis.md)、[`汇总`](2026-06-16_er_ablation/2026-06-16_er_ablation_summary.md) |
| 2026-06-16 | mom5 × COM 形状 Gate | 在 HFQ 与 every-5 非重叠抽样下，控制短动量后 COM 的增量排序不稳定；不转向 mom5。 | 关闭形状族 | [`报告`](2026-06-16_mom5_gate_hfq/2026-06-16_mom5_gate_hfq.md) |
| 2026-06-17 / 06-22 | 动量基消融（窗口 + 形式） | 独立检验窗口与计算形式：20 日优于 10/40/60/120，点对点收益优于 log、OLS slope、OLS t-stat。ER 在不同动量形式下仍有独立增量。 | 保留现行 `momentum(20) × ER(20)` | [`整合报告`](2026-06-22_momentum_base_ablation/2026-06-22_momentum_base_ablation_report.md)、[`窗口汇总`](2026-06-17_momentum_base_ablation/2026-06-17_momentum_base_ablation_summary.md)、[`形式汇总`](2026-06-17_momentum_base_ablation/form_scan/2026-06-22_momentum_base_ablation_form_summary.md) |
| 2026-06-20 | 跨资产池验证 | 4-ETF 大类池在训练/测试均优于自身等权基准；25 行业 ETF 池训练期 Sharpe 很弱，说明 edge 依赖低相关大类轮动，不能泛化为行业横截面策略。 | 诊断，界定适用域 | [`报告`](2026-06-20_cross_pool_validation/2026-06-20_cross_pool_validation.md) |
| 2026-06-20 | Top2 × 国债防御腿 | 仅 Top2 会恶化回撤；仅加债无效；Top2+债可压回撤但以约 13pp CAGR 换取约 1.7pp 全期最大回撤，Sharpe 仍低于基准。 | 弱成立，不部署 | [`报告`](2026-06-20_top2_bond_defensive_leg/2026-06-20_top2_bond_defensive_leg.md) |
| 2026-06-22 | 单日跳变 / score 稳健化 | 生产轮动序列复现通过；但 k=2.5 / 3.0 的崩盘相邻轮动事件仅为 17 / 9，严格阈值触发预注册规模闸门（N<10）。winsorize 后“离群驱动”桶必为该稀疏事件集的子集，反事实必然更小，不能支撑可解释推断。 | **关闭：不做信号稳健化** | [`报告`](2026-06-22_jump_robustness_diagnostic/2026-06-22_jump_robustness_diagnostic_report.md)、[`事件`](2026-06-22_jump_robustness_diagnostic/2026-06-22_jump_robustness_diagnostic_events.csv)、[`漏斗`](2026-06-22_jump_robustness_diagnostic/2026-06-22_jump_robustness_diagnostic_funnel.csv) |
| 2026-06-29 | 回归 slope 动量诊断 | 预注册的 slope 端点鲁棒性机制成立：A3 相比窗口匹配 B′ 降低换手、切换和 15 日 round-trip；但收益/Sharpe 未兑现，触发关闭条件。完整 swap A1 被 B 全面压制，A2 仅作为未预注册观察挂起。 | **A3/A1 关闭；A2 挂起，不部署** | [`报告`](2026-06-29_regression_momentum_diagnostic/2026-06-29_regression_momentum_diagnostic_report.md)、[`指标`](2026-06-29_regression_momentum_diagnostic/2026-06-29_regression_momentum_diagnostic_metrics_full.csv)、[`whipsaw`](2026-06-29_regression_momentum_diagnostic/2026-06-29_regression_momentum_diagnostic_whipsaw_panel.csv) |
| 2026-07-14 | 池内换腿(510300→红利低波/现金流) | 相关性假设成立(512890 与 159915 相关 0.35，远低于 510300 的 0.84)，但 ETF 真实窗口(2019+)是精确等效替换：累计超额≈0、滚动 36m 领先仅 24%；长窗 proxy 优势集中于 2015/2018 不可交易年代，判 regime 依赖。现金流 proxy 全面落后；近 16 个月 ETF 强势仅为观察。2022 熊市"伪防御"A股腿吸走黄金腿配置(−1.2% vs +22.8%)。 | **双线关闭，维持现行池** | [`报告`](2026-07-14_pool_leg_swap_dividend_cashflow/2026-07-14_pool_leg_swap_report.md)、[`设计`](2026-07-14_pool_leg_swap_dividend_cashflow/2026-07-14_pool_leg_swap_design.md) |

## 按研究主题阅读

### 信号与因子

- **已确认的核心信号**：ER 是对动量的有效交互增量；20 日点对点动量是当前四资产异质池中最稳健的动量基。两条结论分别由 ER 消融和动量基消融独立支持。
- **已关闭的形状扩展**：5 月初诊中的 `conv` 曾误取二次拟合的常数项；PR #32 已修复为真正的二次系数，旧 `conv` 表不作为证据。后续 HFQ 加 every-5 抽样的 mom5 × COM Gate 仍未复现稳定增量，因此 DTW/路径形状族维持关闭。
- **已关闭的单日跳变稳健化**：6 月 22 日的预注册诊断在完整轮动复现后，发现严格 k=3.0 下实际“暴跌相邻轮出”只有 9 次；winsorize 反事实的“离群驱动”桶是其子集，样本量只会进一步下降。该线因漏斗稀疏而关闭，**不是**等待参数扫描或实现的候选改动。除非新的预注册设计实质扩大事件宇宙，否则不得在后续 session 将它重新作为新想法提出。
- **已关闭的回归 slope 线**：6 月 29 日诊断确认 slope 相比窗口匹配 B′ 能降低 whipsaw，但 A3 没有把机制兑现为年化或 Sharpe；完整近端加权版本 A1 也被 B 在收益、Sharpe、回撤三轴压制。A2（等权 `slope × R²`）是未预注册观察，不是部署候选；重开必须另起预注册研究，先给出“为什么 R² 应优于 ER”的机制假设，再做窗口/干净度稳健性与回撤门槛检验。
- **资产池边界**：跨池验证显示有效性依赖低相关大类资产的轮动结构；它并未在行业 ETF 池中稳定复现。当前结论只适用于四资产、Top1、T+1 执行和报告锁定的样本，不外推至个股池或行业横截面。池内换腿诊断(2026-07-14)进一步确认：即使把 510300 换成相关性显著更低的红利低波腿(0.35 vs 0.84)，真实可交易窗口下也只是等效替换——"降低池内相关性"本身不自动兑现为收益；池内已有黄金真防御腿时，A股"伪防御"腿在熊市反而是负贡献。红利低波与现金流两条替换线均已关闭。

### 调仓、成本与执行

- **当前参数依据**：rd=5 相对 rd=2 的回滚由成本面板、滚动窗口和 episode 分解共同支持。2026 YTD 的 rd=7 结果是一个短样本诊断，不能覆盖这一全期结论。
- **已否决的抗换手方案**：fixed-cycle/periodic、τ hysteresis 与基于近期 P&L 的自适应 rd 都未通过稳健性检验；减少换手本身不等于增加可实现收益。
- **成本口径**：单边费率按实际执行的 `Σ|Δw|` 扣费，Top1 全仓换仓为两条腿。报告中“单边年化换手”和 `Σ|Δw|` 年化可能相差一倍，阅读时须先确认列定义。
- **成交时点**：T+1 收盘变体并非无效，但收益提升与更深回撤/OOS 走弱并存，当前仅保留为解释执行敏感性的基准实验。
- **分散化尝试**：Top2 与国债腿组合能降低单资产崩盘回撤，但牺牲的 CAGR 远大于改善的全期最大回撤；作为独立策略，仍不如 Top1 基准。它只有在上层资产配置明确追求低回撤 sleeve 时才值得另行评估。

### 归因与外部对照

- 归因报告用于区分资产贡献、深回撤与短持有 whipsaw，不能代替严格的归因模型或直接推导规则。
- 与外部策略的对照只比较持仓路径，不反推其信号、更不计算其收益；其“固定周期指纹”是观察，不是 fixed-cycle 重启的证据。

## 产物与复现约定

- 每项研究以 `{YYYY-MM-DD}_{event_slug}/` 建目录，主要报告与数据文件以同一前缀命名；`event_slug` 使用小写下划线。
- 研究报告应写明：数据口径（HFQ/qfq）、样本截点、暖机起点、信号/成交时点、成本定义、是否含 OOS，以及治理结论。
- 逐日序列和大型中间 CSV 可由 `scripts/` 的对应脚本再生成，通常不进 Git；目录内的 `.gitignore` 记录了这类规则。
- 顶层 [`2026-05-21_momentum_strategy_daily_returns.csv`](2026-05-21_momentum_strategy_daily_returns.csv) 是 Project 的 ERC 接口交付物，必须保留在此路径。高频复用的基准序列优先由脚本重建。

## 维护规则

新增实验时，先放入总览表，并标明它属于“已采用 / 保留 / 否决 / 诊断”哪一类；若它改变生产行为，还必须在 `strategy_changelog.md` 记录正式变更。只读诊断不应伪装成 changelog 条目。
