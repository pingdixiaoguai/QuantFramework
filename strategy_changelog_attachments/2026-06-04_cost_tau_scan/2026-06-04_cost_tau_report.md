# Cost x Tau Scan

- Run date: 2026-06-04
- Config base: `strategy\configs\quality_momentum_top1.yaml`; in-memory overrides only: `start=2014-01-01`, `rebalance_days=5`, `transaction_cost_rate=0`, `hysteresis_threshold=tau`.
- Scope: Mode A research backtest only; no live/backfill, no production YAML, no changelog edits.
- Execution: existing T+1 open engine; cost applied as one-side fee times executed `Σ|Δw|`.
- Fee grid: 0.5, 1, 3, 5, 10 bps one-side. 10 bps is stress only; ETF no stamp duty, so this is not a real ETF cost assumption.

## Three Questions

Q1: The previously reported annual turnover around 2300% is the single-side/net-rotation convention: `0.5 * Σ|w_new - w_old| / years`. The 1bp deduction path in `scripts/periodic_reeval_scan.py` used `Σ|Δw| * fee`, so a Top1 full switch paid `2 * fee`. Deduction was already aligned with the requested cost formula; the reported turnover label needs the single-side qualifier.

Q2: Existing research cost was based on executed weight-delta magnitude from position changes, not on order count. `execution.diff()` does emit `hold` orders with `weight_delta=0`, but the cost path does not charge holds; hold days have zero turnover and zero cost.

Q3: Before this change, 0.01% was not a unified backtest-engine parameter. It was ad-hoc research post-processing in scan scripts. This patch formalizes `transaction_cost_rate` in the backtest engine while keeping the production YAML untouched.

## Tau=0 Gate

| check | passed |
| --- | --- |
| gross_daily_returns | yes |
| positions | yes |
| turnover | yes |
| net_daily_returns_zero_cost | yes |

## Main Metrics

Annual turnover is shown in both requested `Σ|Δw|` annualized form and the old single-side convention.

| tau | fee_bps_one_side | annual_return | sharpe | max_drawdown | annual_turnover_sum_abs | annual_turnover_single_side | avg_holding_days | switch_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.50 | 32.56% | 1.22 | -25.77% | 4601.01% | 2300.50% | 10.93 | 272 |
| 0 | 1.00 | 32.26% | 1.21 | -25.79% | 4601.01% | 2300.50% | 10.93 | 272 |
| 0 | 3.00 | 31.05% | 1.18 | -25.88% | 4601.01% | 2300.50% | 10.93 | 272 |
| 0 | 5.00 | 29.85% | 1.14 | -26.22% | 4601.01% | 2300.50% | 10.93 | 272 |
| 0 | 10.00 | 26.89% | 1.05 | -27.11% | 4601.01% | 2300.50% | 10.93 | 272 |
| 0.0005 | 0.50 | 28.18% | 1.08 | -32.91% | 4465.93% | 2232.96% | 11.26 | 264 |
| 0.0005 | 1.00 | 27.90% | 1.07 | -32.95% | 4465.93% | 2232.96% | 11.26 | 264 |
| 0.0005 | 3.00 | 26.76% | 1.04 | -33.11% | 4465.93% | 2232.96% | 11.26 | 264 |
| 0.0005 | 5.00 | 25.63% | 1.00 | -33.27% | 4465.93% | 2232.96% | 11.26 | 264 |
| 0.0005 | 10.00 | 22.86% | 0.92 | -33.68% | 4465.93% | 2232.96% | 11.26 | 264 |
| 0.001 | 0.50 | 26.09% | 1.02 | -32.91% | 4229.55% | 2114.77% | 11.89 | 250 |
| 0.001 | 1.00 | 25.83% | 1.01 | -32.95% | 4229.55% | 2114.77% | 11.89 | 250 |
| 0.001 | 3.00 | 24.77% | 0.97 | -33.11% | 4229.55% | 2114.77% | 11.89 | 250 |
| 0.001 | 5.00 | 23.72% | 0.94 | -33.27% | 4229.55% | 2114.77% | 11.89 | 250 |
| 0.001 | 10.00 | 21.12% | 0.86 | -33.68% | 4229.55% | 2114.77% | 11.89 | 250 |
| 0.0025 | 0.50 | 26.36% | 1.03 | -33.45% | 3858.09% | 1929.05% | 13.03 | 228 |
| 0.0025 | 1.00 | 26.12% | 1.02 | -33.55% | 3858.09% | 1929.05% | 13.03 | 228 |
| 0.0025 | 3.00 | 25.15% | 0.99 | -33.95% | 3858.09% | 1929.05% | 13.03 | 228 |
| 0.0025 | 5.00 | 24.19% | 0.96 | -34.35% | 3858.09% | 1929.05% | 13.03 | 228 |
| 0.0025 | 10.00 | 21.81% | 0.88 | -35.33% | 3858.09% | 1929.05% | 13.03 | 228 |
| 0.005 | 0.50 | 35.51% | 1.30 | -25.50% | 3300.90% | 1650.45% | 15.23 | 195 |
| 0.005 | 1.00 | 35.29% | 1.29 | -25.50% | 3300.90% | 1650.45% | 15.23 | 195 |
| 0.005 | 3.00 | 34.40% | 1.27 | -25.50% | 3300.90% | 1650.45% | 15.23 | 195 |
| 0.005 | 5.00 | 33.51% | 1.24 | -25.50% | 3300.90% | 1650.45% | 15.23 | 195 |
| 0.005 | 10.00 | 31.33% | 1.18 | -25.57% | 3300.90% | 1650.45% | 15.23 | 195 |
| 0.0075 | 0.50 | 33.89% | 1.25 | -27.59% | 3064.52% | 1532.26% | 16.40 | 181 |
| 0.0075 | 1.00 | 33.69% | 1.24 | -27.61% | 3064.52% | 1532.26% | 16.40 | 181 |
| 0.0075 | 3.00 | 32.87% | 1.22 | -27.67% | 3064.52% | 1532.26% | 16.40 | 181 |
| 0.0075 | 5.00 | 32.06% | 1.20 | -27.75% | 3064.52% | 1532.26% | 16.40 | 181 |
| 0.0075 | 10.00 | 30.05% | 1.14 | -27.97% | 3064.52% | 1532.26% | 16.40 | 181 |
| 0.01 | 0.50 | 37.06% | 1.34 | -27.59% | 2659.30% | 1329.65% | 18.89 | 157 |
| 0.01 | 1.00 | 36.87% | 1.33 | -27.61% | 2659.30% | 1329.65% | 18.89 | 157 |
| 0.01 | 3.00 | 36.15% | 1.31 | -27.67% | 2659.30% | 1329.65% | 18.89 | 157 |
| 0.01 | 5.00 | 35.43% | 1.29 | -27.75% | 2659.30% | 1329.65% | 18.89 | 157 |
| 0.01 | 10.00 | 33.64% | 1.24 | -27.97% | 2659.30% | 1329.65% | 18.89 | 157 |

## Break-Even

| tau | delta_gross_ann_return | turnover_reduction_sum_abs_ann | linear_break_even_bps_one_side | exact_break_even_bps_one_side | exact_break_even_status |
| --- | --- | --- | --- | --- | --- |
| 0.0005 | -4.39% | 135.08% | 325.30 | 124.13 | extrapolated_above_grid |
| 0.001 | -6.50% | 371.46% | 175.10 | 89.85 | extrapolated_above_grid |
| 0.0025 | -6.26% | 742.91% | 84.20 | 54.08 | extrapolated_above_grid |
| 0.005 | 2.87% | 1300.10% | -22.06 | -17.74 | extrapolated_below_grid |
| 0.0075 | 1.23% | 1536.48% | -8.03 | -6.20 | extrapolated_below_grid |
| 0.01 | 4.38% | 1941.71% | -22.53 | -17.91 | extrapolated_below_grid |

Readout material:
- tau=0.0005: economically preferable only when real one-side cost is above about 124.13 bps (extrapolated_above_grid).
- tau=0.001: economically preferable only when real one-side cost is above about 89.85 bps (extrapolated_above_grid).
- tau=0.0025: economically preferable only when real one-side cost is above about 54.08 bps (extrapolated_above_grid).
- tau=0.005: already ahead at nonnegative cost levels; the extrapolated crossing is -17.74 bps (extrapolated_below_grid).
- tau=0.0075: already ahead at nonnegative cost levels; the extrapolated crossing is -6.20 bps (extrapolated_below_grid).
- tau=0.01: already ahead at nonnegative cost levels; the extrapolated crossing is -17.91 bps (extrapolated_below_grid).

## Episode Decomposition

Switch counts are actual T+1 open executions inside the inclusive date window. Under the enforced rd=5/min-hold path, tau=0 has 4 executions inside `2024-11-04` to `2024-12-17`; the next execution is `2024-12-20`, outside the requested window.

| episode | tau | fee_bps_one_side | switch_count | cumulative_pnl |
| --- | --- | --- | --- | --- |
| whipsaw | 0 | 1.00 | 4 | 2.73% |
| whipsaw | 0 | 3.00 | 4 | 2.56% |
| whipsaw | 0 | 5.00 | 4 | 2.40% |
| single_asset_crash | 0 | 1.00 | 2 | -10.38% |
| single_asset_crash | 0 | 3.00 | 2 | -10.45% |
| single_asset_crash | 0 | 5.00 | 2 | -10.52% |
| whipsaw | 0.0005 | 1.00 | 4 | 2.73% |
| whipsaw | 0.0005 | 3.00 | 4 | 2.56% |
| whipsaw | 0.0005 | 5.00 | 4 | 2.40% |
| single_asset_crash | 0.0005 | 1.00 | 2 | -10.38% |
| single_asset_crash | 0.0005 | 3.00 | 2 | -10.45% |
| single_asset_crash | 0.0005 | 5.00 | 2 | -10.52% |
| whipsaw | 0.001 | 1.00 | 4 | 2.73% |
| whipsaw | 0.001 | 3.00 | 4 | 2.56% |
| whipsaw | 0.001 | 5.00 | 4 | 2.40% |
| single_asset_crash | 0.001 | 1.00 | 2 | -10.38% |
| single_asset_crash | 0.001 | 3.00 | 2 | -10.45% |
| single_asset_crash | 0.001 | 5.00 | 2 | -10.52% |
| whipsaw | 0.0025 | 1.00 | 4 | 2.73% |
| whipsaw | 0.0025 | 3.00 | 4 | 2.56% |
| whipsaw | 0.0025 | 5.00 | 4 | 2.40% |
| single_asset_crash | 0.0025 | 1.00 | 2 | -10.38% |
| single_asset_crash | 0.0025 | 3.00 | 2 | -10.45% |
| single_asset_crash | 0.0025 | 5.00 | 2 | -10.52% |
| whipsaw | 0.005 | 1.00 | 3 | 7.09% |
| whipsaw | 0.005 | 3.00 | 3 | 6.96% |
| whipsaw | 0.005 | 5.00 | 3 | 6.83% |
| single_asset_crash | 0.005 | 1.00 | 2 | -10.38% |
| single_asset_crash | 0.005 | 3.00 | 2 | -10.45% |
| single_asset_crash | 0.005 | 5.00 | 2 | -10.52% |
| whipsaw | 0.0075 | 1.00 | 3 | 1.08% |
| whipsaw | 0.0075 | 3.00 | 3 | 0.96% |
| whipsaw | 0.0075 | 5.00 | 3 | 0.84% |
| single_asset_crash | 0.0075 | 1.00 | 2 | -10.38% |
| single_asset_crash | 0.0075 | 3.00 | 2 | -10.45% |
| single_asset_crash | 0.0075 | 5.00 | 2 | -10.52% |
| whipsaw | 0.01 | 1.00 | 3 | 1.08% |
| whipsaw | 0.01 | 3.00 | 3 | 0.96% |
| whipsaw | 0.01 | 5.00 | 3 | 0.84% |
| single_asset_crash | 0.01 | 1.00 | 2 | -10.38% |
| single_asset_crash | 0.01 | 3.00 | 2 | -10.45% |
| single_asset_crash | 0.01 | 5.00 | 2 | -10.52% |

## Canary

| baseline_execution_date | wave_end | tau | status | actual_execution_date | delay_trading_days | baseline_capture | actual_capture |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2024-09-26 | 2024-11-01 | 0 | on_time | 2024-09-26 | 0 | 29.93% | 29.93% |
| 2024-09-26 | 2024-11-01 | 0.0005 | on_time | 2024-09-26 | 0 | 29.93% | 29.93% |
| 2024-09-26 | 2024-11-01 | 0.001 | on_time | 2024-09-26 | 0 | 29.93% | 29.93% |
| 2024-09-26 | 2024-11-01 | 0.0025 | on_time | 2024-09-26 | 0 | 29.93% | 29.93% |
| 2024-09-26 | 2024-11-01 | 0.005 | delayed | 2024-09-27 | 1 | 29.93% | 21.66% |
| 2024-09-26 | 2024-11-01 | 0.0075 | delayed | 2024-09-27 | 1 | 29.93% | 21.66% |
| 2024-09-26 | 2024-11-01 | 0.01 | delayed | 2024-09-27 | 1 | 29.93% | 21.66% |

## Other Delayed/Blocked Positive Baseline Switches

| baseline_execution_date | wave_end | tau | status | actual_execution_date | delay_trading_days | baseline_capture | actual_capture | new_asset |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2014-02-20 | 2014-03-25 | 0 | on_time | 2014-02-20 | 0 | 1.76% | 1.76% | 518880.SH |
| 2014-02-20 | 2014-03-25 | 0.0005 | on_time | 2014-02-20 | 0 | 1.76% | 1.76% | 518880.SH |
| 2014-02-20 | 2014-03-25 | 0.001 | on_time | 2014-02-20 | 0 | 1.76% | 1.76% | 518880.SH |
| 2014-02-20 | 2014-03-25 | 0.0025 | on_time | 2014-02-20 | 0 | 1.76% | 1.76% | 518880.SH |
| 2014-02-20 | 2014-03-25 | 0.005 | delayed | 2014-02-26 | 4 | 1.76% | -0.34% | 518880.SH |
| 2014-02-20 | 2014-03-25 | 0.0075 | delayed | 2014-02-26 | 4 | 1.76% | -0.34% | 518880.SH |
| 2014-02-20 | 2014-03-25 | 0.01 | delayed | 2014-02-26 | 4 | 1.76% | -0.34% | 518880.SH |
| 2014-05-09 | 2014-06-13 | 0 | on_time | 2014-05-09 | 0 | 6.89% | 6.89% | 513100.SH |
| 2014-05-09 | 2014-06-13 | 0.0005 | delayed | 2014-05-14 | 3 | 6.89% | 4.81% | 513100.SH |
| 2014-05-09 | 2014-06-13 | 0.001 | delayed | 2014-05-14 | 3 | 6.89% | 4.81% | 513100.SH |
| 2014-05-09 | 2014-06-13 | 0.0025 | delayed | 2014-05-14 | 3 | 6.89% | 4.81% | 513100.SH |
| 2014-05-09 | 2014-06-13 | 0.005 | delayed | 2014-05-14 | 3 | 6.89% | 4.81% | 513100.SH |
| 2014-05-09 | 2014-06-13 | 0.0075 | delayed | 2014-05-15 | 4 | 6.89% | 5.17% | 513100.SH |
| 2014-05-09 | 2014-06-13 | 0.01 | on_time | 2014-05-09 | 0 | 6.89% | 6.89% | 513100.SH |
| 2014-07-21 | 2014-08-20 | 0 | on_time | 2014-07-21 | 0 | 9.17% | 9.17% | 510300.SH |
| 2014-07-21 | 2014-08-20 | 0.0005 | on_time | 2014-07-21 | 0 | 9.17% | 9.17% | 510300.SH |
| 2014-07-21 | 2014-08-20 | 0.001 | on_time | 2014-07-21 | 0 | 9.17% | 9.17% | 510300.SH |
| 2014-07-21 | 2014-08-20 | 0.0025 | delayed | 2014-07-23 | 2 | 9.17% | 7.90% | 510300.SH |
| 2014-07-21 | 2014-08-20 | 0.005 | delayed | 2014-07-25 | 4 | 9.17% | 5.25% | 510300.SH |
| 2014-07-21 | 2014-08-20 | 0.0075 | delayed | 2014-07-25 | 4 | 9.17% | 5.25% | 510300.SH |
| 2014-07-21 | 2014-08-20 | 0.01 | delayed | 2014-07-25 | 4 | 9.17% | 5.25% | 510300.SH |
| 2014-09-22 | 2014-09-26 | 0 | on_time | 2014-09-22 | 0 | 0.65% | 0.65% | 510300.SH |
| 2014-09-22 | 2014-09-26 | 0.0005 | on_time | 2014-09-22 | 0 | 0.65% | 0.65% | 510300.SH |
| 2014-09-22 | 2014-09-26 | 0.001 | on_time | 2014-09-22 | 0 | 0.65% | 0.65% | 510300.SH |
| 2014-09-22 | 2014-09-26 | 0.0025 | blocked |  |  | 0.65% | 0.00% | 510300.SH |
| 2014-09-22 | 2014-09-26 | 0.005 | blocked |  |  | 0.65% | 0.00% | 510300.SH |
| 2014-09-22 | 2014-09-26 | 0.0075 | blocked |  |  | 0.65% | 0.00% | 510300.SH |
| 2014-09-22 | 2014-09-26 | 0.01 | blocked |  |  | 0.65% | 0.00% | 510300.SH |
| 2014-09-29 | 2014-10-16 | 0 | on_time | 2014-09-29 | 0 | 0.68% | 0.68% | 159915.SZ |
| 2014-09-29 | 2014-10-16 | 0.0005 | on_time | 2014-09-29 | 0 | 0.68% | 0.68% | 159915.SZ |

## Raw CSV

- `2026-06-04_cost_tau_tau0_gate.csv`
- `2026-06-04_cost_tau_main_metrics.csv`
- `2026-06-04_cost_tau_break_even.csv`
- `2026-06-04_cost_tau_episodes.csv`
- `2026-06-04_cost_tau_canary.csv`
- `2026-06-04_cost_tau_good_switches.csv`
