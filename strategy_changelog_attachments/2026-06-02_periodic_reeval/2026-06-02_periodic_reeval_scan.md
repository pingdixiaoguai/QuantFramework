# Periodic Re-evaluation Scan

- Run date: 2026-06-02
- Config base: `strategy\configs\quality_momentum_top1.yaml` with start forced to `2014-01-01`.
- Data: local HFQ parquet via `data.store.query()`.
- Cost: one-side 0.01% baseline, deducted on actual executed `abs(delta_weight).sum()`.
- Execution: existing engine T+1 open, zero slippage.
- Split: train `2014-01-01` to `2021-12-31`; test `2022-01-01` to data end.

## Core Evidence - Phase Robustness (4.1)

| label | reeval_mode | rebalance_days | phase_offset | start | end | trading_days | annual_return | sharpe | max_drawdown | annual_turnover | avg_holding_days | switch_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| periodic_N5_k0 | periodic | 5 | 0 | 2014-02-07 | 2026-05-19 | 2985 | 28.12% | 1.15 | -30.37% | 1650.45% | 15.23 | 195 |
| periodic_N5_k1 | periodic | 5 | 1 | 2014-02-07 | 2026-05-19 | 2985 | 30.68% | 1.20 | -30.91% | 1498.49% | 16.77 | 177 |
| periodic_N5_k2 | periodic | 5 | 2 | 2014-02-07 | 2026-05-19 | 2984 | 27.92% | 1.13 | -33.95% | 1541.22% | 16.31 | 182 |
| periodic_N5_k3 | periodic | 5 | 3 | 2014-02-07 | 2026-05-19 | 2983 | 24.01% | 0.97 | -34.91% | 1507.95% | 16.66 | 178 |
| periodic_N5_k4 | periodic | 5 | 4 | 2014-02-07 | 2026-05-19 | 2983 | 33.87% | 1.27 | -31.31% | 1347.44% | 18.64 | 159 |

Readout: phase annual-return spread is 9.86%; best k=4, worst k=3. Max-drawdown range is 4.54%. Only 1/5 phases beat the min_hold N=5 annual return (32.26%), which points toward B.

## Core Evidence - Sample Split (4.2)

| sample | label | reeval_mode | rebalance_days | phase_offset | start | end | trading_days | annual_return | sharpe | max_drawdown | annual_turnover | avg_holding_days | switch_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train_2014_2021 | train_2014_2021_min_hold | min_hold | 5 | 0 | 2014-02-07 | 2021-12-31 | 1929 | 29.98% | 1.20 | -25.79% | 2384.14% | 10.54 | 182 |
| train_2014_2021 | train_2014_2021_periodic_k0 | periodic | 5 | 0 | 2014-02-07 | 2021-12-31 | 1929 | 30.05% | 1.19 | -30.37% | 1691.76% | 14.84 | 129 |
| test_2022_data_end | test_2022_data_end_min_hold | min_hold | 5 | 0 | 2022-02-09 | 2026-05-19 | 1035 | 37.29% | 1.25 | -25.51% | 2203.48% | 11.37 | 90 |
| test_2022_data_end | test_2022_data_end_periodic_k0 | periodic | 5 | 0 | 2022-02-09 | 2026-05-19 | 1035 | 26.34% | 1.12 | -24.08% | 1570.43% | 15.92 | 64 |

Readout: train_2014_2021: periodic annual return minus min_hold 0.08%, max drawdown delta -4.58%; test_2022_data_end: periodic annual return minus min_hold -10.95%, max drawdown delta 1.43%. The test-period reversal points toward B rather than a stable A mechanism.

## Baseline Gate (4.0)

Data gate:

| asset | first_valid_date | last_valid_date | rows | has_2014_01_02 |
| --- | --- | --- | --- | --- |
| 510300.SH | 2014-01-02 | 2026-05-19 | 3006 | yes |
| 159915.SZ | 2014-01-02 | 2026-05-19 | 3005 | yes |
| 513100.SH | 2014-01-02 | 2026-05-19 | 3005 | yes |
| 518880.SH | 2014-01-02 | 2026-05-19 | 3006 | yes |

Min-hold explicit vs current default `rebalance_days=5`:

| metric | explicit_min_hold | current_default | diff | exact_match |
| --- | --- | --- | --- | --- |
| annual_return | 0.3225502191 | 0.3225502191 | 0 | yes |
| sharpe | 1.211607944 | 1.211607944 | 0 | yes |
| max_drawdown | -0.2579467007 | -0.2579467007 | 0 | yes |
| annual_turnover | 23.00502513 | 23.00502513 | 0 | yes |
| avg_holding_days | 10.93406593 | 10.93406593 | 0 | yes |
| switch_count | 272 | 272 | 0 | yes |
| trading_days | 2985 | 2985 | 0 | yes |
| total_return | 26.42559653 | 26.42559653 | 0 | yes |
| net_returns_series | True | True | 0 | yes |
| positions_df | True | True | 0 | yes |

Gate result: passed exactly for metrics, net return series, and positions.

## Skipped Switch Attribution (4.3)

| event_count | old_win_rate | sum_old_minus_new | compounded_old_minus_new | same_old_asset_count |
| --- | --- | --- | --- | --- |
| 39 | 41.03% | -15.92% | -16.77% | 34 |

Readout: among qualifying non-grid min_hold switches (`holding_days` in {6,7,8,9}), periodic's held asset outperformed in 41.03% of event windows; sum old-minus-new is -15.92%. This does not support A's skipped-whipsaw mechanism.

Raw event rows are in `2026-06-02_periodic_reeval_skipped_switch_attribution.csv`.

## N Scan (4.4)

| label | reeval_mode | rebalance_days | phase_offset | start | end | trading_days | annual_return | sharpe | max_drawdown | annual_turnover | avg_holding_days | switch_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| N2_min_hold | min_hold | 2 | 0 | 2014-02-07 | 2026-05-19 | 2985 | 33.26% | 1.24 | -28.44% | 3330.45% | 7.56 | 394 |
| N2_periodic_k0 | periodic | 2 | 0 | 2014-02-07 | 2026-05-19 | 2985 | 31.46% | 1.19 | -27.58% | 2815.48% | 8.94 | 333 |
| N3_min_hold | min_hold | 3 | 0 | 2014-02-07 | 2026-05-19 | 2985 | 30.45% | 1.15 | -28.59% | 2942.11% | 8.55 | 348 |
| N3_periodic_k0 | periodic | 3 | 0 | 2014-02-07 | 2026-05-19 | 2985 | 37.86% | 1.37 | -31.54% | 2165.43% | 11.61 | 256 |
| N4_min_hold | min_hold | 4 | 0 | 2014-02-07 | 2026-05-19 | 2985 | 28.50% | 1.09 | -30.45% | 2629.75% | 9.57 | 311 |
| N4_periodic_k0 | periodic | 4 | 0 | 2014-02-07 | 2026-05-19 | 2985 | 28.12% | 1.08 | -29.38% | 1937.49% | 12.98 | 229 |
| N5_min_hold | min_hold | 5 | 0 | 2014-02-07 | 2026-05-19 | 2985 | 32.26% | 1.21 | -25.79% | 2300.50% | 10.93 | 272 |
| N5_periodic_k0 | periodic | 5 | 0 | 2014-02-07 | 2026-05-19 | 2985 | 28.12% | 1.15 | -30.37% | 1650.45% | 15.23 | 195 |
| N6_min_hold | min_hold | 6 | 0 | 2014-02-07 | 2026-05-19 | 2985 | 32.09% | 1.21 | -31.54% | 2055.68% | 12.23 | 243 |
| N6_periodic_k0 | periodic | 6 | 0 | 2014-02-07 | 2026-05-19 | 2985 | 40.40% | 1.43 | -30.78% | 1371.86% | 18.31 | 162 |
| N7_min_hold | min_hold | 7 | 0 | 2014-02-07 | 2026-05-19 | 2984 | 32.85% | 1.23 | -32.55% | 1971.92% | 12.75 | 233 |
| N7_periodic_k0 | periodic | 7 | 0 | 2014-02-07 | 2026-05-19 | 2985 | 37.74% | 1.43 | -29.81% | 1203.02% | 20.87 | 142 |
| N10_min_hold | min_hold | 10 | 0 | 2014-02-07 | 2026-05-19 | 2985 | 33.58% | 1.34 | -43.44% | 1591.36% | 15.79 | 188 |
| N10_periodic_k0 | periodic | 10 | 0 | 2014-02-07 | 2026-05-19 | 2983 | 24.99% | 1.06 | -33.97% | 1246.06% | 20.16 | 147 |

Readout: periodic beats min_hold on annual return in 3/7 tested N values. The advantage is not smooth across the N band, pointing toward B.

## Cost Sensitivity (4.5)

| label | reeval_mode | cost_bps_one_side | annual_return | sharpe | max_drawdown | annual_turnover | avg_holding_days | switch_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cost_0.0001_min_hold | min_hold | 1.0 | 32.26% | 1.21 | -25.79% | 2300.50% | 10.93 | 272 |
| cost_0.0001_periodic_k0 | periodic | 1.0 | 28.12% | 1.15 | -30.37% | 1650.45% | 15.23 | 195 |
| cost_0.0005_min_hold | min_hold | 5.0 | 29.85% | 1.14 | -26.22% | 2300.50% | 10.93 | 272 |
| cost_0.0005_periodic_k0 | periodic | 5.0 | 26.44% | 1.09 | -30.49% | 1650.45% | 15.23 | 195 |
| cost_0.0010_min_hold | min_hold | 10.0 | 26.89% | 1.05 | -27.11% | 2300.50% | 10.93 | 272 |
| cost_0.0010_periodic_k0 | periodic | 10.0 | 24.36% | 1.02 | -30.63% | 1650.45% | 15.23 | 195 |

Readout: periodic annual-return advantage by cost level = 1.0bp: -4.14%; 5.0bp: -3.41%; 10.0bp: -2.53%. Higher costs narrow the deficit but do not rescue k=0, so this is not evidence for A.

## Raw CSV

- `2026-06-02_periodic_reeval_data_gate.csv`
- `2026-06-02_periodic_reeval_gate.csv`
- `2026-06-02_periodic_reeval_phase_metrics.csv`
- `2026-06-02_periodic_reeval_split_metrics.csv`
- `2026-06-02_periodic_reeval_n_scan_metrics.csv`
- `2026-06-02_periodic_reeval_cost_metrics.csv`
- `2026-06-02_periodic_reeval_skipped_switch_attribution.csv`
- `2026-06-02_periodic_reeval_skipped_switch_summary.csv`
