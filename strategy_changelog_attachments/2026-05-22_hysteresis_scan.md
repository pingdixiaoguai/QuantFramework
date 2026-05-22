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
| 0 | 2014-02-07 | 0.322122 | 1.20475 | -0.257947 | 45.1356 | 11.1455 | 267 |
| 0.0005 | 2014-02-07 | 0.277067 | 1.06485 | -0.32953 | 44.4607 | 11.3144 | 263 |
| 0.001 | 2014-02-07 | 0.269283 | 1.04006 | -0.32953 | 41.761 | 12.0444 | 247 |
| 0.0025 | 2014-02-07 | 0.273728 | 1.05539 | -0.335518 | 38.2176 | 13.1586 | 226 |
| 0.005 | 2014-02-07 | 0.364721 | 1.32738 | -0.255041 | 32.8182 | 15.3179 | 194 |
| 0.0075 | 2014-02-07 | 0.344016 | 1.26125 | -0.276083 | 30.6247 | 16.4121 | 181 |
| 0.01 | 2014-02-07 | 0.37285 | 1.34751 | -0.276083 | 26.7439 | 18.7862 | 158 |

## Surface Observations

- Turnover falls from 45.14 at `tau=0` to 26.74 at `tau=0.01`.
- Whipsaw count falls from 126 to 58, but cumulative whipsaw P&L is not monotonic across the curve.
- Maximum drawdown first worsens at `tau=0.0005`. Full-period return and Sharpe are also non-monotonic, so a high headline metric alone is not a deployment rule.
- The 2024-10 single-asset row is unchanged across the scan. The 2024-09-26 `159915.SZ` canary first delays at `tau=0.005`.

## Turnover-Side Evidence

Whipsaw rows below are provisional until they are checked against the original
2026-05-21 attribution definition, which was not found in this repository. The
reconstructed rule documented in the attribution archive is: leave an asset and
return to it on the second executed switch; the intervening holding P&L is the
whipsaw P&L.

| tau | whipsaw_count | cumulative_whipsaw_pnl |
| --- | --- | --- |
| 0 | 126 | 0.969126 |
| 0.0005 | 123 | 0.799822 |
| 0.001 | 109 | 1.02216 |
| 0.0025 | 93 | 1.42261 |
| 0.005 | 78 | 1.35758 |
| 0.0075 | 65 | 1.11427 |
| 0.01 | 58 | 1.77401 |

The focus episodes separate switch-heavy drawdown months from the 2024-10
single-asset month so threshold gains do not get credited to the wrong
mechanism.

| tau | episode | start | end | return | max_drawdown | switches_in_window |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 2015-10_switch_heavy | 2015-10-01 | 2015-10-31 | 0.213953 | -0.0715182 | 0 |
| 0 | 2020-09_switch_heavy | 2020-09-01 | 2020-09-30 | -0.0995322 | -0.162111 | 1 |
| 0 | 2024-10_single_asset | 2024-10-01 | 2024-10-31 | -0.0492832 | -0.255041 | 0 |
| 0 | 2025-10_switch_heavy | 2025-10-01 | 2025-10-31 | -0.0307604 | -0.0884729 | 2 |
| 0.0005 | 2015-10_switch_heavy | 2015-10-01 | 2015-10-31 | 0.213953 | -0.0715182 | 0 |
| 0.0005 | 2020-09_switch_heavy | 2020-09-01 | 2020-09-30 | -0.0995322 | -0.162111 | 1 |
| 0.0005 | 2024-10_single_asset | 2024-10-01 | 2024-10-31 | -0.0492832 | -0.255041 | 0 |
| 0.0005 | 2025-10_switch_heavy | 2025-10-01 | 2025-10-31 | -0.0307604 | -0.0884729 | 2 |
| 0.001 | 2015-10_switch_heavy | 2015-10-01 | 2015-10-31 | 0.213953 | -0.0715182 | 0 |
| 0.001 | 2020-09_switch_heavy | 2020-09-01 | 2020-09-30 | -0.0995322 | -0.162111 | 1 |
| 0.001 | 2024-10_single_asset | 2024-10-01 | 2024-10-31 | -0.0492832 | -0.255041 | 0 |
| 0.001 | 2025-10_switch_heavy | 2025-10-01 | 2025-10-31 | -0.0307604 | -0.0884729 | 2 |
| 0.0025 | 2015-10_switch_heavy | 2015-10-01 | 2015-10-31 | 0.213953 | -0.0715182 | 0 |
| 0.0025 | 2020-09_switch_heavy | 2020-09-01 | 2020-09-30 | -0.0995322 | -0.162111 | 1 |
| 0.0025 | 2024-10_single_asset | 2024-10-01 | 2024-10-31 | -0.0492832 | -0.255041 | 0 |
| 0.0025 | 2025-10_switch_heavy | 2025-10-01 | 2025-10-31 | -0.0307604 | -0.0884729 | 2 |
| 0.005 | 2015-10_switch_heavy | 2015-10-01 | 2015-10-31 | 0.201273 | -0.0715182 | 1 |
| 0.005 | 2020-09_switch_heavy | 2020-09-01 | 2020-09-30 | -0.0995322 | -0.162111 | 1 |
| 0.005 | 2024-10_single_asset | 2024-10-01 | 2024-10-31 | -0.0492832 | -0.255041 | 0 |
| 0.005 | 2025-10_switch_heavy | 2025-10-01 | 2025-10-31 | -0.0307604 | -0.0884729 | 2 |
| 0.0075 | 2015-10_switch_heavy | 2015-10-01 | 2015-10-31 | 0.201273 | -0.0715182 | 1 |
| 0.0075 | 2020-09_switch_heavy | 2020-09-01 | 2020-09-30 | -0.0994464 | -0.162111 | 0 |
| 0.0075 | 2024-10_single_asset | 2024-10-01 | 2024-10-31 | -0.0492832 | -0.255041 | 0 |
| 0.0075 | 2025-10_switch_heavy | 2025-10-01 | 2025-10-31 | -0.0307604 | -0.0884729 | 2 |
| 0.01 | 2015-10_switch_heavy | 2015-10-01 | 2015-10-31 | 0.201273 | -0.0715182 | 1 |
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
| 0.0005 | correct_avoided_loss | 18 |
| 0.0005 | wrong_missed_trend | 30 |
| 0.001 | correct_avoided_loss | 33 |
| 0.001 | wrong_missed_trend | 46 |
| 0.0025 | correct_avoided_loss | 63 |
| 0.0025 | wrong_missed_trend | 70 |
| 0.005 | correct_avoided_loss | 95 |
| 0.005 | wrong_missed_trend | 97 |
| 0.0075 | correct_avoided_loss | 108 |
| 0.0075 | wrong_missed_trend | 107 |
| 0.01 | correct_avoided_loss | 120 |
| 0.01 | wrong_missed_trend | 106 |

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
| 2015-10_switch_heavy | 68 | -0.00088864 | 0.0183235 | 0.261401 | 0.0183235 | 0.0746332 |
| 2020-09_switch_heavy | 88 | -0.0278152 | -0.00166288 | 0.263912 | 0.00656012 | 0.025498 |
| 2024-10_single_asset | 72 | -0.000467299 | 0.0451272 | 0.601569 | 0.0451272 | 0.13017 |
| 2025-10_switch_heavy | 68 | -0.00389105 | 0.0168896 | 0.143928 | 0.0168896 | 0.0773381 |
| 2024-09_canary | 76 | -0.0271335 | -0.000128945 | 0.359284 | 0.00605192 | 0.0269122 |

## Raw Files

CSV companions store the full panel, trade ledgers, position periods, whipsaw
rows, focus episodes, delayed-switch comparisons, canary rows, and score regime
samples. The surface should be interpreted as a trade-off curve, not as a
deployment selector.
