# Regression-Slope Momentum Diagnostic

Mode C read-only diagnostic. Artifacts are written only in this attachment directory; production configs, registry, run_daily, changelog, summary, and state files are not modified.

## Controls

- Asset pool: 510300.SH, 159915.SZ, 513100.SH, 518880.SH
- Strategy: Top1 full allocation; rebalance_days=5
- Execution/cost semantics: existing backtest engine T+1 open execution, HFQ data, future-info truncation, transaction_cost_rate x sum(abs(delta weights)).
- Full asset-union calendar in requested window: 2014-01-02 to 2026-06-04, 3018 rows.
- Paired evaluation index used for all arms: 2014-01-02 to 2026-06-04, 3016 rows.
- IS/OOS split: train_ratio=0.7, train_end=2022-09-05.
- Cost grid: one-side 0.5/1/3/5 bp.

## Calendar Audit

| scope                     | arm     |   transaction_cost_rate |   full_calendar_days |   common_eval_days |   missing_days | missing_dates         |
|:--------------------------|:--------|------------------------:|---------------------:|-------------------:|---------------:|:----------------------|
| all_arms_common_index     |         |                         |                 3018 |               3016 |              2 | 2021-02-08;2021-02-09 |
| arm_cost_vs_full_calendar | A1      |                  5e-05  |                 3018 |                    |              0 |                       |
| arm_cost_vs_full_calendar | A1      |                  0.0001 |                 3018 |                    |              0 |                       |
| arm_cost_vs_full_calendar | A1      |                  0.0003 |                 3018 |                    |              0 |                       |
| arm_cost_vs_full_calendar | A1      |                  0.0005 |                 3018 |                    |              0 |                       |
| arm_cost_vs_full_calendar | A2      |                  5e-05  |                 3018 |                    |              1 | 2021-02-08            |
| arm_cost_vs_full_calendar | A2      |                  0.0001 |                 3018 |                    |              1 | 2021-02-08            |
| arm_cost_vs_full_calendar | A2      |                  0.0003 |                 3018 |                    |              1 | 2021-02-08            |
| arm_cost_vs_full_calendar | A2      |                  0.0005 |                 3018 |                    |              1 | 2021-02-08            |
| arm_cost_vs_full_calendar | A3      |                  5e-05  |                 3018 |                    |              2 | 2021-02-08;2021-02-09 |
| arm_cost_vs_full_calendar | A3      |                  0.0001 |                 3018 |                    |              2 | 2021-02-08;2021-02-09 |
| arm_cost_vs_full_calendar | A3      |                  0.0003 |                 3018 |                    |              2 | 2021-02-08;2021-02-09 |
| arm_cost_vs_full_calendar | A3      |                  0.0005 |                 3018 |                    |              2 | 2021-02-08;2021-02-09 |
| arm_cost_vs_full_calendar | B       |                  5e-05  |                 3018 |                    |              0 |                       |
| arm_cost_vs_full_calendar | B       |                  0.0001 |                 3018 |                    |              0 |                       |
| arm_cost_vs_full_calendar | B       |                  0.0003 |                 3018 |                    |              0 |                       |
| arm_cost_vs_full_calendar | B       |                  0.0005 |                 3018 |                    |              0 |                       |
| arm_cost_vs_full_calendar | B_prime |                  5e-05  |                 3018 |                    |              2 | 2021-02-08;2021-02-09 |
| arm_cost_vs_full_calendar | B_prime |                  0.0001 |                 3018 |                    |              2 | 2021-02-08;2021-02-09 |
| arm_cost_vs_full_calendar | B_prime |                  0.0003 |                 3018 |                    |              2 | 2021-02-08;2021-02-09 |
| arm_cost_vs_full_calendar | B_prime |                  0.0005 |                 3018 |                    |              2 | 2021-02-08;2021-02-09 |

## Anchor And Warmup Gates

| asset     |   comparable_points |   max_abs_diff | passed   |
|:----------|--------------------:|---------------:|:---------|
| 510300.SH |                3236 |              0 | True     |
| 159915.SZ |                3235 |              0 | True     |
| 513100.SH |                3152 |              0 | True     |
| 518880.SH |                3103 |              0 | True     |

| asset     | data_start   | data_end   |   rows_before_2014_01_01 | first_eval_trading_day   | w26_min_history_27_eligible_date   |
|:----------|:-------------|:-----------|-------------------------:|:-------------------------|:-----------------------------------|
| 510300.SH | 2013-01-04   | 2026-06-04 |                      238 | 2014-01-02               | 2013-02-18                         |
| 159915.SZ | 2013-01-04   | 2026-06-04 |                      238 | 2014-01-02               | 2013-02-18                         |
| 513100.SH | 2013-05-15   | 2026-06-04 |                      155 | 2014-01-02               | 2013-06-25                         |
| 518880.SH | 2013-07-29   | 2026-06-04 |                      105 | 2014-01-02               | 2013-09-03                         |

## Full Metrics

| arm     |   transaction_cost_bps_one_side | annual_return   |   sharpe | max_drawdown   | annual_turnover_single_side   |   avg_holding_days | is_annual_return   |   is_sharpe | oos_annual_return   |   oos_sharpe |
|:--------|--------------------------------:|:----------------|---------:|:---------------|:------------------------------|-------------------:|:-------------------|------------:|:--------------------|-------------:|
| A1      |                             0.5 | 32.99%          |     1.23 | -35.89%        | 1387.00%                      |              17.95 | 32.64%             |        1.24 | 33.80%              |         1.21 |
| A2      |                             0.5 | 36.88%          |     1.33 | -30.58%        | 1336.87%                      |              18.62 | 34.01%             |        1.28 | 43.83%              |         1.46 |
| A3      |                             0.5 | 26.76%          |     1.04 | -35.16%        | 1595.89%                      |              15.71 | 22.77%             |        0.93 | 36.62%              |         1.27 |
| B       |                             0.5 | 34.26%          |     1.27 | -25.77%        | 2281.03%                      |              10.97 | 31.04%             |        1.24 | 42.09%              |         1.34 |
| B_prime |                             0.5 | 31.66%          |     1.19 | -39.28%        | 1938.46%                      |              12.94 | 19.74%             |        0.86 | 64.34%              |         1.86 |
| A1      |                             1   | 32.80%          |     1.23 | -35.93%        | 1387.00%                      |              17.95 | 32.46%             |        1.24 | 33.61%              |         1.21 |
| A2      |                             1   | 36.70%          |     1.33 | -30.59%        | 1336.87%                      |              18.62 | 33.83%             |        1.27 | 43.64%              |         1.46 |
| A3      |                             1   | 26.56%          |     1.03 | -35.28%        | 1595.89%                      |              15.71 | 22.56%             |        0.92 | 36.42%              |         1.26 |
| B       |                             1   | 33.95%          |     1.26 | -25.79%        | 2281.03%                      |              10.97 | 30.74%             |        1.23 | 41.77%              |         1.33 |
| B_prime |                             1   | 31.40%          |     1.19 | -39.32%        | 1938.46%                      |              12.94 | 19.50%             |        0.85 | 64.06%              |         1.85 |
| A1      |                             3   | 32.07%          |     1.2  | -36.12%        | 1387.00%                      |              17.95 | 31.73%             |        1.21 | 32.87%              |         1.18 |
| A2      |                             3   | 35.97%          |     1.31 | -30.65%        | 1336.87%                      |              18.62 | 33.11%             |        1.25 | 42.90%              |         1.44 |
| A3      |                             3   | 25.76%          |     1    | -35.74%        | 1595.89%                      |              15.71 | 21.77%             |        0.9  | 35.60%              |         1.24 |
| B       |                             3   | 32.74%          |     1.22 | -25.88%        | 2281.03%                      |              10.97 | 29.56%             |        1.19 | 40.48%              |         1.3  |
| B_prime |                             3   | 30.39%          |     1.16 | -39.49%        | 1938.46%                      |              12.94 | 18.53%             |        0.82 | 62.93%              |         1.83 |
| A1      |                             5   | 31.34%          |     1.18 | -36.30%        | 1387.00%                      |              17.95 | 31.01%             |        1.19 | 32.13%              |         1.16 |
| A2      |                             5   | 35.24%          |     1.29 | -30.70%        | 1336.87%                      |              18.62 | 32.39%             |        1.23 | 42.15%              |         1.42 |
| A3      |                             5   | 24.96%          |     0.98 | -36.21%        | 1595.89%                      |              15.71 | 20.97%             |        0.87 | 34.78%              |         1.22 |
| B       |                             5   | 31.53%          |     1.19 | -26.22%        | 2281.03%                      |              10.97 | 28.38%             |        1.15 | 39.20%              |         1.27 |
| B_prime |                             5   | 29.38%          |     1.13 | -39.66%        | 1938.46%                      |              12.94 | 17.57%             |        0.79 | 61.80%              |         1.81 |

## IS/OOS Metrics

| arm     |   transaction_cost_bps_one_side | split   | start      | end        |   days | annual_return   |   sharpe |
|:--------|--------------------------------:|:--------|:-----------|:-----------|-------:|:----------------|---------:|
| A1      |                             0.5 | IS      | 2014-01-02 | 2022-09-05 |   2112 | 32.64%          |     1.24 |
| A1      |                             0.5 | OOS     | 2022-09-06 | 2026-06-04 |    904 | 33.80%          |     1.21 |
| A2      |                             0.5 | IS      | 2014-01-02 | 2022-09-05 |   2112 | 34.01%          |     1.28 |
| A2      |                             0.5 | OOS     | 2022-09-06 | 2026-06-04 |    904 | 43.83%          |     1.46 |
| A3      |                             0.5 | IS      | 2014-01-02 | 2022-09-05 |   2112 | 22.77%          |     0.93 |
| A3      |                             0.5 | OOS     | 2022-09-06 | 2026-06-04 |    904 | 36.62%          |     1.27 |
| B       |                             0.5 | IS      | 2014-01-02 | 2022-09-05 |   2112 | 31.04%          |     1.24 |
| B       |                             0.5 | OOS     | 2022-09-06 | 2026-06-04 |    904 | 42.09%          |     1.34 |
| B_prime |                             0.5 | IS      | 2014-01-02 | 2022-09-05 |   2112 | 19.74%          |     0.86 |
| B_prime |                             0.5 | OOS     | 2022-09-06 | 2026-06-04 |    904 | 64.34%          |     1.86 |
| A1      |                             1   | IS      | 2014-01-02 | 2022-09-05 |   2112 | 32.46%          |     1.24 |
| A1      |                             1   | OOS     | 2022-09-06 | 2026-06-04 |    904 | 33.61%          |     1.21 |
| A2      |                             1   | IS      | 2014-01-02 | 2022-09-05 |   2112 | 33.83%          |     1.27 |
| A2      |                             1   | OOS     | 2022-09-06 | 2026-06-04 |    904 | 43.64%          |     1.46 |
| A3      |                             1   | IS      | 2014-01-02 | 2022-09-05 |   2112 | 22.56%          |     0.92 |
| A3      |                             1   | OOS     | 2022-09-06 | 2026-06-04 |    904 | 36.42%          |     1.26 |
| B       |                             1   | IS      | 2014-01-02 | 2022-09-05 |   2112 | 30.74%          |     1.23 |
| B       |                             1   | OOS     | 2022-09-06 | 2026-06-04 |    904 | 41.77%          |     1.33 |
| B_prime |                             1   | IS      | 2014-01-02 | 2022-09-05 |   2112 | 19.50%          |     0.85 |
| B_prime |                             1   | OOS     | 2022-09-06 | 2026-06-04 |    904 | 64.06%          |     1.85 |
| A1      |                             3   | IS      | 2014-01-02 | 2022-09-05 |   2112 | 31.73%          |     1.21 |
| A1      |                             3   | OOS     | 2022-09-06 | 2026-06-04 |    904 | 32.87%          |     1.18 |
| A2      |                             3   | IS      | 2014-01-02 | 2022-09-05 |   2112 | 33.11%          |     1.25 |
| A2      |                             3   | OOS     | 2022-09-06 | 2026-06-04 |    904 | 42.90%          |     1.44 |
| A3      |                             3   | IS      | 2014-01-02 | 2022-09-05 |   2112 | 21.77%          |     0.9  |
| A3      |                             3   | OOS     | 2022-09-06 | 2026-06-04 |    904 | 35.60%          |     1.24 |
| B       |                             3   | IS      | 2014-01-02 | 2022-09-05 |   2112 | 29.56%          |     1.19 |
| B       |                             3   | OOS     | 2022-09-06 | 2026-06-04 |    904 | 40.48%          |     1.3  |
| B_prime |                             3   | IS      | 2014-01-02 | 2022-09-05 |   2112 | 18.53%          |     0.82 |
| B_prime |                             3   | OOS     | 2022-09-06 | 2026-06-04 |    904 | 62.93%          |     1.83 |
| A1      |                             5   | IS      | 2014-01-02 | 2022-09-05 |   2112 | 31.01%          |     1.19 |
| A1      |                             5   | OOS     | 2022-09-06 | 2026-06-04 |    904 | 32.13%          |     1.16 |
| A2      |                             5   | IS      | 2014-01-02 | 2022-09-05 |   2112 | 32.39%          |     1.23 |
| A2      |                             5   | OOS     | 2022-09-06 | 2026-06-04 |    904 | 42.15%          |     1.42 |
| A3      |                             5   | IS      | 2014-01-02 | 2022-09-05 |   2112 | 20.97%          |     0.87 |
| A3      |                             5   | OOS     | 2022-09-06 | 2026-06-04 |    904 | 34.78%          |     1.22 |
| B       |                             5   | IS      | 2014-01-02 | 2022-09-05 |   2112 | 28.38%          |     1.15 |
| B       |                             5   | OOS     | 2022-09-06 | 2026-06-04 |    904 | 39.20%          |     1.27 |
| B_prime |                             5   | IS      | 2014-01-02 | 2022-09-05 |   2112 | 17.57%          |     0.79 |
| B_prime |                             5   | OOS     | 2022-09-06 | 2026-06-04 |    904 | 61.80%          |     1.81 |

## Whipsaw Panel

| arm     |   switch_count |   round_trip_count_15d | first_possible_switch_episode_share   |   avg_holding_days |   episode_count |
|:--------|---------------:|-----------------------:|:--------------------------------------|-------------------:|----------------:|
| A1      |            167 |                     25 | 20.83%                                |              17.95 |             168 |
| A2      |            161 |                     24 | 21.60%                                |              18.62 |             162 |
| A3      |            191 |                     49 | 34.38%                                |              15.71 |             192 |
| B       |            274 |                    107 | 46.18%                                |              10.97 |             275 |
| B_prime |            232 |                     98 | 43.78%                                |              12.94 |             233 |

## Pre-Registered Questions: Facts Only

Q1 mechanism facts (A3 vs B_prime, no deployment conclusion):
- single-side annual turnover: A3 lower than B_prime (15.96 vs 19.38)
- 15d round trips: A3 lower than B_prime (49 vs 98)
- avg holding days: A3 higher than B_prime (15.71 vs 12.94)
- switches: A3 lower than B_prime (191 vs 232)

Q2 realization facts (regression arms vs B, no deployment conclusion):
- 0.5bp annual-return+Sharpe winners vs B: A2
- 1bp annual-return+Sharpe winners vs B: A2
- 3bp annual-return+Sharpe winners vs B: A2
- 5bp annual-return+Sharpe winners vs B: A2

Notes: round-trip count is consecutive A->B->A episode triples where the second switch occurs within 15 trading days of the first switch. Episode lengths are computed from forward-filled daily holdings, not sparse execution-only position rows.

## Post-Read Decision Addendum

This addendum records the governance read after the full table was reviewed.
It does not change the raw measurement artifacts above.

One-line conclusion: the pre-registered positive-prior mechanism is real but
does not pay. Slope reduces whipsaw, but A3 does not convert that into return
or Sharpe; A1 is worse than B; A2 is an unregistered observation, not a
deployment exit.

### Q1 Mechanism

A3 versus B_prime is the clean window-matched `pct_change -> slope` isolation.
The mechanism is confirmed:

| comparison | B_prime | A3 | read |
| --- | ---: | ---: | --- |
| single-side annual turnover | 1938% | 1596% | lower |
| 15-day round trips | 98 | 49 | lower, roughly halved |
| switches | 232 | 191 | lower |
| average holding days | 12.94 | 15.71 | higher |

Endpoint robustness exists at the mechanism layer.

### Q2 Realization

The mechanism does not pay. At the 1bp anchor, A3 trails B_prime on annual
return and Sharpe:

| arm | annual return | Sharpe | maxDD |
| --- | ---: | ---: | ---: |
| B_prime | 31.40% | 1.19 | -39.32% |
| A3 | 26.56% | 1.03 | -35.28% |

A3 improves whipsaw and max drawdown versus B_prime, but loses on return and
Sharpe. This triggers the pre-registered close condition: whipsaw improvement
without material return or risk-adjusted improvement. A3 is closed.

### A1 Read

A1 is the full swap that originally motivated the diagnostic. At the 1bp
anchor it is worse than B on annual return, Sharpe, and max drawdown:

| arm | annual return | Sharpe | maxDD |
| --- | ---: | ---: | ---: |
| B | 33.95% | 1.26 | -25.79% |
| A1 | 32.80% | 1.23 | -35.93% |

The A2 -> A1 step isolates the only changed variable as recency-squared
weighting. That step worsens annual return from 36.70% to 32.80% and max
drawdown from -30.59% to -35.93%. The recency weighting is a net negative in
this diagnostic. A1 is closed.

### A2 Read

A2 is the only regression arm that beats B on annual return and Sharpe across
0.5/1/3/5bp costs. At the 1bp anchor it posts 36.70% annual return and 1.33
Sharpe versus B's 33.95% and 1.26. It also survives the 5bp cost row on those
two metrics.

This is not a deployment signal in this diagnostic. A2 is not the
pre-registered slope-whipsaw hypothesis; it is the `ER -> R2` cleanliness-form
swap. The B -> B_prime window step is itself fragile, with max drawdown moving
to -39.32% and IS Sharpe falling to 0.85. A2 also has a deeper max drawdown
than B (-30.59% versus -25.79%), trading modest Sharpe improvement for a
drawdown direction that is opposite the strategy's recent governance
preference.

A2 is parked as an unregistered observation. Reopening it requires a separate
pre-registered study with an explicit mechanism for why R2 should dominate ER,
a window/cleanliness robustness grid, and an ex ante drawdown acceptance
threshold.

### Audit Notes

- B in this report is internally comparable to all arms because all metrics use
  the paired 3016-row all-arm index. It is not bit-for-bit the prior 3018-row
  anchor because 2021-02-08 and 2021-02-09 are excluded after long-window arms
  produced missing returns there.
- The B anchor gate against production `quality_momentum(window=20)` passed
  with zero max absolute difference for all four assets, so the discrepancy is
  calendar pairing, not factor or engine drift.
