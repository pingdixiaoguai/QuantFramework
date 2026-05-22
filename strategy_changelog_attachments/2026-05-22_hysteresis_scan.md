# Hysteresis Threshold Scan

Generated on 2026-05-22. This is research evidence for a post-2026-06-02
decision. It does not deploy a threshold and it does not modify the production
`quality_momentum_top1.yaml`.

## Fixed Method

- Evaluation start: 2014-01-01 request, trimmed until the full asset pool and
  factor-produced strategy returns are available.
- Cost: 0.01% per one-way executed weight delta.
- `rebalance_days=5` for every run.
- Independent complete runs for `tau in {0, 0.0005, 0.001, 0.0025, 0.005, 0.0075, 0.01}`.
- Gate: `tau=0` raw returns and executed position rows exactly matched the
  same-mouth plain Top1 baseline before the scan.

## Standard Panel

| tau | evaluation_start | annualized_return | sharpe | max_drawdown | annualized_turnover | average_holding_days | switch_count |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 2016-02-02 | 0.293777 | 1.15539 | -0.255053 | 45.2954 | 11.1022 | 224 |
| 0.0005 | 2016-02-02 | 0.258327 | 1.04565 | -0.255041 | 44.4884 | 11.3032 | 220 |
| 0.001 | 2016-02-02 | 0.256373 | 1.03836 | -0.255867 | 41.462 | 12.1262 | 205 |
| 0.0025 | 2016-02-02 | 0.256004 | 1.0392 | -0.255041 | 37.8303 | 13.2872 | 187 |
| 0.005 | 2016-02-02 | 0.331697 | 1.27915 | -0.255041 | 32.1809 | 15.6125 | 159 |
| 0.0075 | 2016-02-02 | 0.314811 | 1.21775 | -0.276083 | 29.7598 | 16.8784 | 147 |
| 0.01 | 2016-02-02 | 0.356738 | 1.34372 | -0.276083 | 26.1281 | 19.2154 | 129 |

## Surface Observations

- Turnover falls from 45.3 at `tau=0` to 26.13 at `tau=0.01`.
- Whipsaw count falls from 108 to 46, but cumulative whipsaw P&L is not monotonic across the curve.
- Maximum drawdown first worsens at `tau=0.001`. Full-period return and Sharpe are also non-monotonic, so a high headline metric alone is not a deployment rule.
- The 2024-10 single-asset row is unchanged across the scan. The 2024-09-26 `159915.SZ` canary first delays at `tau=0.005`.

## Turnover-Side Evidence

Whipsaw rows use the reconstructed rule documented in the attribution archive:
leave an asset and return to it on the second executed switch; the intervening
holding P&L is the whipsaw P&L.

| tau | whipsaw_count | cumulative_whipsaw_pnl |
| --- | --- | --- |
| 0 | 108 | 0.437084 |
| 0.0005 | 104 | 0.281604 |
| 0.001 | 92 | 0.494067 |
| 0.0025 | 77 | 0.898746 |
| 0.005 | 63 | 0.82763 |
| 0.0075 | 50 | 0.623035 |
| 0.01 | 46 | 1.34854 |

The focus episodes separate switch-heavy drawdown months from the 2024-10
single-asset month so threshold gains do not get credited to the wrong
mechanism. The requested 2015-10 row remains in raw output, but this local
reconstruction has no 2015-10 values because the Parquet pool available here
starts in 2016.

| tau | episode | start | end | return | max_drawdown | switches_in_window |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 2015-10_switch_heavy | 2015-10-01 | 2015-10-31 |  |  | 0 |
| 0 | 2020-09_switch_heavy | 2020-09-01 | 2020-09-30 | -0.0995322 | -0.162111 | 1 |
| 0 | 2024-10_single_asset | 2024-10-01 | 2024-10-31 | -0.0492832 | -0.255041 | 0 |
| 0 | 2025-10_switch_heavy | 2025-10-01 | 2025-10-31 | -0.0307604 | -0.0884729 | 2 |
| 0.0005 | 2015-10_switch_heavy | 2015-10-01 | 2015-10-31 |  |  | 0 |
| 0.0005 | 2020-09_switch_heavy | 2020-09-01 | 2020-09-30 | -0.0995322 | -0.162111 | 1 |
| 0.0005 | 2024-10_single_asset | 2024-10-01 | 2024-10-31 | -0.0492832 | -0.255041 | 0 |
| 0.0005 | 2025-10_switch_heavy | 2025-10-01 | 2025-10-31 | -0.0307604 | -0.0884729 | 2 |
| 0.001 | 2015-10_switch_heavy | 2015-10-01 | 2015-10-31 |  |  | 0 |
| 0.001 | 2020-09_switch_heavy | 2020-09-01 | 2020-09-30 | -0.0995322 | -0.162111 | 1 |
| 0.001 | 2024-10_single_asset | 2024-10-01 | 2024-10-31 | -0.0492832 | -0.255041 | 0 |
| 0.001 | 2025-10_switch_heavy | 2025-10-01 | 2025-10-31 | -0.0307604 | -0.0884729 | 2 |
| 0.0025 | 2015-10_switch_heavy | 2015-10-01 | 2015-10-31 |  |  | 0 |
| 0.0025 | 2020-09_switch_heavy | 2020-09-01 | 2020-09-30 | -0.0995322 | -0.162111 | 1 |
| 0.0025 | 2024-10_single_asset | 2024-10-01 | 2024-10-31 | -0.0492832 | -0.255041 | 0 |
| 0.0025 | 2025-10_switch_heavy | 2025-10-01 | 2025-10-31 | -0.0307604 | -0.0884729 | 2 |
| 0.005 | 2015-10_switch_heavy | 2015-10-01 | 2015-10-31 |  |  | 0 |
| 0.005 | 2020-09_switch_heavy | 2020-09-01 | 2020-09-30 | -0.0995322 | -0.162111 | 1 |
| 0.005 | 2024-10_single_asset | 2024-10-01 | 2024-10-31 | -0.0492832 | -0.255041 | 0 |
| 0.005 | 2025-10_switch_heavy | 2025-10-01 | 2025-10-31 | -0.0307604 | -0.0884729 | 2 |
| 0.0075 | 2015-10_switch_heavy | 2015-10-01 | 2015-10-31 |  |  | 0 |
| 0.0075 | 2020-09_switch_heavy | 2020-09-01 | 2020-09-30 | -0.0994464 | -0.162111 | 0 |
| 0.0075 | 2024-10_single_asset | 2024-10-01 | 2024-10-31 | -0.0492832 | -0.255041 | 0 |
| 0.0075 | 2025-10_switch_heavy | 2025-10-01 | 2025-10-31 | -0.0307604 | -0.0884729 | 2 |
| 0.01 | 2015-10_switch_heavy | 2015-10-01 | 2015-10-31 |  |  | 0 |
| 0.01 | 2020-09_switch_heavy | 2020-09-01 | 2020-09-30 | -0.0994464 | -0.162111 | 0 |
| 0.01 | 2024-10_single_asset | 2024-10-01 | 2024-10-31 | -0.0492832 | -0.255041 | 0 |
| 0.01 | 2025-10_switch_heavy | 2025-10-01 | 2025-10-31 | 0.0570678 | -0.0884729 | 1 |

## Stickiness Cost

Suppressed or delayed plain-baseline switches are judged over the baseline
target holding window. `wrong_missed_trend` means the baseline target window
outperformed the threshold path over that same window; `correct_avoided_loss`
means the sticky path did not lose that comparison.

| tau | verdict | count |
| --- | --- | --- |
| 0.0005 | correct_avoided_loss | 17 |
| 0.0005 | wrong_missed_trend | 26 |
| 0.001 | correct_avoided_loss | 32 |
| 0.001 | wrong_missed_trend | 39 |
| 0.0025 | correct_avoided_loss | 59 |
| 0.0025 | wrong_missed_trend | 61 |
| 0.005 | correct_avoided_loss | 81 |
| 0.005 | wrong_missed_trend | 85 |
| 0.0075 | correct_avoided_loss | 91 |
| 0.0075 | wrong_missed_trend | 96 |
| 0.01 | correct_avoided_loss | 104 |
| 0.01 | wrong_missed_trend | 93 |

### 2024-09-26 Canary

| tau | baseline_entry_date | baseline_next_entry_date | from_asset | target_asset | tau_entry_date | status | delay_trading_days | baseline_window_pnl | tau_window_pnl | missed_pnl | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.005 | 2024-09-26 | 2024-11-04 | 518880.SH | 159915.SZ | 2024-09-27 | delayed | 1 | 0.303532 | 0.218686 | 0.0848457 | wrong_missed_trend |
| 0.0075 | 2024-09-26 | 2024-11-04 | 518880.SH | 159915.SZ | 2024-09-27 | delayed | 1 | 0.303532 | 0.218686 | 0.0848457 | wrong_missed_trend |
| 0.01 | 2024-09-26 | 2024-11-04 | 518880.SH | 159915.SZ | 2024-09-27 | delayed | 1 | 0.303532 | 0.218686 | 0.0848457 | wrong_missed_trend |

## Score Scale Samples

`tau` is in the same score units as `quality_momentum = momentum * ER`.
These regime summaries show the observed local score scale.

| regime | rows | min_score | median_score | max_score | median_abs_score | p90_abs_score |
| --- | --- | --- | --- | --- | --- | --- |
| 2015-10_switch_heavy | 0 |  |  |  |  |  |
| 2020-09_switch_heavy | 88 | -0.0278152 | -0.00166288 | 0.263912 | 0.00656012 | 0.025498 |
| 2024-10_single_asset | 72 | -0.000467299 | 0.0451272 | 0.601569 | 0.0451272 | 0.13017 |
| 2025-10_switch_heavy | 68 | -0.00389105 | 0.0168896 | 0.143928 | 0.0168896 | 0.0773381 |
| 2024-09_canary | 76 | -0.0271335 | -0.000128945 | 0.359284 | 0.00605192 | 0.0269122 |

## Raw Files

CSV companions store the full panel, trade ledgers, position periods, whipsaw
rows, focus episodes, delayed-switch comparisons, canary rows, and score regime
samples. The surface should be interpreted as a trade-off curve, not as a
deployment selector.
