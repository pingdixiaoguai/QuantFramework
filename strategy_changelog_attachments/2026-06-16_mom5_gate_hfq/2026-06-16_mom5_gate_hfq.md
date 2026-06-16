# Mom5 x COM gate HFQ diagnostic

- Run date: 2026-06-16
- Assets: 510300.SH, 159915.SZ, 513100.SH, 518880.SH
- Signal dates: 2014-01-01 to 2026-06-04; rows needing `open[t+6]` are naturally truncated.
- Scope: read-only local Parquet diagnostic; no strategy, engine, YAML, or production entry point code is imported or modified.
- Prices: HFQ open/close from `data.store.query()`. This was verified against `read_storage()` raw prices as `raw_price * adj_factor / first_adj_factor`; this is not qfq.
- Features: ER, mom20, mom5, and COM use only `[t-20, t]` or shorter lookback data; forward return is `open[t+6] / open[t+1] - 1`.
- Standardization: ER, mom20, mom5, and COM are percentile-ranked within each asset before tercile grouping.
- Grouping: all reported tables first gate to the high ER tercile, then sort by COM or by mom5 x COM.
- Caveat: overlapping daily windows have strong autocorrelation; the every-5-trading-day sample is included as a second view. Cells with single-digit counts are not interpreted.
- Runtime: 0.92s

## HFQ口径确认

| asset     | date       | raw_open | raw_close | adj_factor | baseline_adj_factor | query_open | expected_hfq_open | open_abs_diff  | query_close | expected_hfq_close | close_abs_diff |
| --------- | ---------- | -------- | --------- | ---------- | ------------------- | ---------- | ----------------- | -------------- | ----------- | ------------------ | -------------- |
| 510300.SH | 2013-01-04 | 2.554000 | 2.530000  | 1.014000   | 1.014000            | 2.554000   | 2.554000          | 0.000000000000 | 2.530000    | 2.530000           | 0.000000000000 |
| 510300.SH | 2019-09-18 | 3.966000 | 3.968000  | 1.114000   | 1.014000            | 4.357124   | 4.357124          | 0.000000000000 | 4.359321    | 4.359321           | 0.000000000000 |
| 159915.SZ | 2013-01-04 | 0.720000 | 0.708000  | 1.000000   | 1.000000            | 0.720000   | 0.720000          | 0.000000000000 | 0.708000    | 0.708000           | 0.000000000000 |
| 159915.SZ | 2019-09-17 | 1.646000 | 1.618000  | 1.000000   | 1.000000            | 1.646000   | 1.646000          | 0.000000000000 | 1.618000    | 1.618000           | 0.000000000000 |
| 513100.SH | 2013-05-15 | 0.990000 | 0.997000  | 1.000000   | 1.000000            | 0.990000   | 0.990000          | 0.000000000000 | 0.997000    | 0.997000           | 0.000000000000 |
| 513100.SH | 2019-11-21 | 3.018000 | 3.017000  | 1.000000   | 1.000000            | 3.018000   | 3.018000          | 0.000000000000 | 3.017000    | 3.017000           | 0.000000000000 |
| 518880.SH | 2013-07-29 | 2.633000 | 2.626000  | 1.000000   | 1.000000            | 2.633000   | 2.633000          | 0.000000000000 | 2.626000    | 2.626000           | 0.000000000000 |
| 518880.SH | 2019-12-26 | 3.353000 | 3.353000  | 1.000000   | 1.000000            | 3.353000   | 3.353000          | 0.000000000000 | 3.353000    | 3.353000           | 0.000000000000 |

## 因子口径与对齐手验

- `mom20 * ER` matched `factors.quality_momentum.compute(window=20)` on the eligible panel within floating-point tolerance.
- The alignment rows show feature window end `t`, forward start `t+1`, and forward end `t+6`, so the signal window and forward return window do not overlap.

| asset     | date       | mom20    | er       | quality_momentum_from_parts | quality_momentum | qmom_abs_diff  |
| --------- | ---------- | -------- | -------- | --------------------------- | ---------------- | -------------- |
| 510300.SH | 2014-01-02 | -6.1807% | 50.3226% | -0.031102704361             | -0.031102704361  | 0.000000000000 |
| 510300.SH | 2014-03-20 | -8.8840% | 49.3917% | -0.043879740399             | -0.043879740399  | 0.000000000000 |
| 510300.SH | 2026-05-27 | 3.1368%  | 19.5312% | 0.006126489962              | 0.006126489962   | 0.000000000000 |
| 159915.SZ | 2014-01-02 | 7.0492%  | 26.2195% | 0.018482606957              | 0.018482606957   | 0.000000000000 |
| 159915.SZ | 2014-03-20 | -6.8105% | 23.6534% | -0.016109190656             | -0.016109190656  | 0.000000000000 |
| 159915.SZ | 2026-05-27 | 10.3974% | 35.7009% | 0.037119643466              | 0.037119643466   | 0.000000000000 |
| 513100.SH | 2014-01-02 | 3.2686%  | 26.6187% | 0.008700460127              | 0.008700460127   | 0.000000000000 |
| 513100.SH | 2014-03-20 | 2.0185%  | 11.0092% | 0.002222205076              | 0.002222205076   | 0.000000000000 |
| 513100.SH | 2026-05-27 | 16.5705% | 53.9249% | 0.089356439631              | 0.089356439631   | 0.000000000000 |
| 518880.SH | 2014-01-02 | -0.2076% | 1.3774%  | -0.000028600716             | -0.000028600716  | 0.000000000000 |
| 518880.SH | 2014-03-20 | 3.1567%  | 17.3448% | 0.005475156093              | 0.005475156093   | 0.000000000000 |
| 518880.SH | 2026-05-27 | -4.8952% | 31.5410% | -0.015439866797             | -0.015439866797  | 0.000000000000 |

| asset     | window_start | window_end | t_plus_1   | t_plus_6   | open_t_plus_1 | open_t_plus_6 | fwd      | mom20    | er       | quality_momentum_from_parts | quality_momentum |
| --------- | ------------ | ---------- | ---------- | ---------- | ------------- | ------------- | -------- | -------- | -------- | --------------------------- | ---------------- |
| 510300.SH | 2013-12-04   | 2014-01-02 | 2014-01-03 | 2014-01-10 | 2.358000      | 2.269000      | -3.7744% | -6.1807% | 50.3226% | -0.031102704361             | -0.031102704361  |
| 159915.SZ | 2014-02-20   | 2014-03-20 | 2014-03-21 | 2014-03-28 | 1.373000      | 1.326000      | -3.4232% | -6.8105% | 23.6534% | -0.016109190656             | -0.016109190656  |
| 513100.SH | 2026-04-24   | 2026-05-27 | 2026-05-28 | 2026-06-04 | 11.134452     | 11.554620     | 3.7736%  | 16.5705% | 53.9249% | 0.089356439631              | 0.089356439631   |

## 样本量

- Full overlapping high-ER rows: 4016; total eligible rows before ER gate: 12046.
- Every-5 high-ER rows: 802; total eligible rows before ER gate: 2412.
- (a) full COM cell count range: 1036-1719.
- (a) every-5 COM cell count range: 194-356.
- (b) full mom5 x COM cell count range: 266-824.
- (b) every-5 mom5 x COM cell count range: 53-175.

## (a) COM复现检查

- Full sample COM spread: high-low mean 0.3875%, median 0.3660%, min count 1036.
- Every-5 sample COM spread: high-low mean 0.1919%, median 0.2124%, min count 194.

Full sample:

| com_group | mean    | median  | count |
| --------- | ------- | ------- | ----- |
| low       | 0.3917% | 0.3020% | 1036  |
| mid       | 0.5929% | 0.4356% | 1261  |
| high      | 0.7792% | 0.6679% | 1719  |

Every-5 sample:

| com_group | mean    | median  | count |
| --------- | ------- | ------- | ----- |
| low       | 0.5671% | 0.4679% | 194   |
| mid       | 0.7349% | 0.4209% | 252   |
| high      | 0.7590% | 0.6803% | 356   |

## (b) mom5 x COM gate

- Full sample cell extremes: highest mean: mom5=low, com=high, mean 0.8790%, median 0.8443%, count 592; lowest mean: mom5=mid, com=low, mean 0.1483%, median 0.2174%, count 437.
- Every-5 sample cell extremes: highest mean: mom5=high, com=low, mean 1.1525%, median 0.7886%, count 53; lowest mean: mom5=mid, com=low, mean 0.2139%, median 0.2824%, count 81.
- COM high-minus-low spreads by mom5 tercile:

Full sample:

| mom5_group | com_high_minus_low_mean | com_high_minus_low_median | min_count |
| ---------- | ----------------------- | ------------------------- | --------- |
| low        | 0.1579%                 | 0.4013%                   | 266       |
| mid        | 0.3807%                 | 0.2966%                   | 303       |
| high       | 0.3513%                 | 0.1795%                   | 333       |

Every-5 sample:

| mom5_group | com_high_minus_low_mean | com_high_minus_low_median | min_count |
| ---------- | ----------------------- | ------------------------- | --------- |
| low        | 0.3898%                 | 0.4846%                   | 60        |
| mid        | 0.0360%                 | 0.1663%                   | 63        |
| high       | -0.3165%                | -0.1309%                  | 53        |

Full 3x3 table:

| mom5_group | com_group | mean    | median  | count |
| ---------- | --------- | ------- | ------- | ----- |
| low        | low       | 0.7210% | 0.4430% | 266   |
| low        | mid       | 0.5492% | 0.4873% | 326   |
| low        | high      | 0.8790% | 0.8443% | 592   |
| mid        | low       | 0.1483% | 0.2174% | 437   |
| mid        | mid       | 0.5833% | 0.4715% | 391   |
| mid        | high      | 0.5289% | 0.5140% | 303   |
| high       | low       | 0.4482% | 0.3960% | 333   |
| high       | mid       | 0.6260% | 0.3835% | 544   |
| high       | high      | 0.7996% | 0.5756% | 824   |

Every-5 3x3 table:

| mom5_group | com_group | mean    | median  | count |
| ---------- | --------- | ------- | ------- | ----- |
| low        | low       | 0.5268% | 0.4165% | 60    |
| low        | mid       | 0.9308% | 0.4217% | 70    |
| low        | high      | 0.9165% | 0.9011% | 118   |
| mid        | low       | 0.2139% | 0.2824% | 81    |
| mid        | mid       | 0.7097% | 0.4657% | 85    |
| mid        | high      | 0.2498% | 0.4487% | 63    |
| high       | low       | 1.1525% | 0.7886% | 53    |
| high       | mid       | 0.6156% | 0.0000% | 97    |
| high       | high      | 0.8360% | 0.6576% | 175   |

## CSV outputs

- `2026-06-16_mom5_gate_hfq_a_full.csv`
- `2026-06-16_mom5_gate_hfq_a_every_5.csv`
- `2026-06-16_mom5_gate_hfq_b_full.csv`
- `2026-06-16_mom5_gate_hfq_b_every_5.csv`

## 判读结论

- (a) COM 在干净 HFQ 口径上复现，但强度比重叠全样本显示得更弱。全样本 COM low/mid/high 的 mean 为 0.3917%/0.5929%/0.7792%，median 为 0.3020%/0.4356%/0.6679%，high-low spread 为 +0.3875%/+0.3660%。Every-5 非重叠样本仍为正，但 mean 0.5671%/0.7349%/0.7590% 中到高基本走平，median 0.4679%/0.4209%/0.6803% 非单调，high-low spread 缩到 +0.1919%/+0.2124%。
- (b) 控住 mom5 后，COM 的增量信号在全样本和 every-5 两版之间不一致。全样本三个 mom5 桶内 COM high-low mean spread 为 +0.1579%/+0.3807%/+0.3513%，但 every-5 为 +0.3898%/+0.0360%/-0.3165%；其中 mom5 高桶从全样本 +0.3513% 翻为 every-5 -0.3165%。按事先规则，重叠窗口自相关严重，两版冲突时信非重叠版。
- 判定：控住 mom5 后，COM 没有稳定的增量前向收益排序；DTW/路径形状这一族关闭。此前 conv 已在稳健抽样下失效，COM 是最后保留的形状候选，本次 mom5 gate 没有给更复杂形状工具留下触发条件。
- 不转向短窗口动量项。表格没有把 mom5 本身立成稳定前向预测变量；高 ER 组内，无论用 COM 还是 mom5 切 20 日路径的内部时序结构，都没有拿出抽样稳定的排序。Every-5 中局部活跃的单格不作为新方向依据。
- 治理归位：本次为 Mode C 只读诊断，无部署、无策略/引擎/YAML 修改；不进入 `strategy_changelog.md` 正文，不触发对现因子或历史认知条目的修正。
