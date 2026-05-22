# 2026-05-21 Attribution Reconstruction

This is a reconstructed artifact generated on 2026-05-22 from the local data
available in this repository worktree. The original 2026-05-21 Markdown and
raw CSV attribution outputs were not found in the repository, so these files
preserve the rerunnable evidence used by the hysteresis scan instead of claiming
to be the original files.

## Scope

- Plain `quality_momentum_top1` path.
- `rebalance_days=5`.
- Evaluation begins after the configured asset pool is complete.
- Cost-adjusted returns charge 0.01% per one-way executed weight delta.
- Whipsaw rows use an executable reconstruction rule: an asset is left and
  returned to on the second executed switch, and the intervening holding period
  P&L is the whipsaw P&L.

## Metrics

| evaluation_start | annualized_return | sharpe | max_drawdown | annualized_turnover | average_holding_days | switch_count |
| --- | --- | --- | --- | --- | --- | --- |
| 2014-02-07 | 0.322122 | 1.20475 | -0.257947 | 45.1356 | 11.1455 | 267 |

## Whipsaw Summary

| whipsaw_count | cumulative_whipsaw_pnl |
| --- | --- |
| 126 | 0.969126 |

## Largest Drawdown Episodes

| start | trough | recovery | max_drawdown |
| --- | --- | --- | --- |
| 2015-06-04 | 2015-07-08 | 2015-10-23 | -0.257947 |
| 2025-10-20 | 2026-04-03 |  | -0.255053 |
| 2024-10-09 | 2024-10-17 | 2025-07-22 | -0.255041 |
| 2015-10-28 | 2016-01-05 | 2017-02-22 | -0.244451 |
| 2020-09-04 | 2021-03-09 | 2022-07-22 | -0.242097 |
| 2020-02-14 | 2020-03-17 | 2020-07-09 | -0.2174 |
| 2019-03-13 | 2019-05-14 | 2019-08-26 | -0.150458 |
| 2023-07-20 | 2023-12-05 | 2024-03-18 | -0.137087 |
| 2020-07-14 | 2020-08-12 | 2020-08-27 | -0.130974 |
| 2020-01-23 | 2020-02-03 | 2020-02-11 | -0.128703 |

Raw CSV companions store metrics, trade ledger, position periods, drawdown
episodes, and whipsaw rows under `strategy_changelog_attachments/`.
