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
| 2016-02-02 | 0.293777 | 1.15539 | -0.255053 | 45.2954 | 11.1022 | 224 |

## Whipsaw Summary

| whipsaw_count | cumulative_whipsaw_pnl |
| --- | --- |
| 108 | 0.437084 |

## Largest Drawdown Episodes

| start | trough | recovery | max_drawdown |
| --- | --- | --- | --- |
| 2025-10-20 | 2026-04-03 |  | -0.255053 |
| 2024-10-09 | 2024-10-17 | 2025-07-22 | -0.255041 |
| 2020-09-04 | 2021-03-09 | 2022-07-22 | -0.242097 |
| 2020-02-14 | 2020-03-17 | 2020-07-09 | -0.2174 |
| 2019-03-13 | 2019-05-14 | 2019-08-26 | -0.150458 |
| 2023-07-20 | 2023-12-05 | 2024-03-18 | -0.137087 |
| 2020-07-14 | 2020-08-12 | 2020-08-27 | -0.130974 |
| 2020-01-23 | 2020-02-03 | 2020-02-11 | -0.128703 |
| 2018-01-25 | 2018-02-06 | 2018-03-09 | -0.120533 |
| 2024-07-10 | 2024-09-09 | 2024-09-27 | -0.113864 |

Raw CSV companions store metrics, trade ledger, position periods, drawdown
episodes, and whipsaw rows under `strategy_changelog_attachments/`.
