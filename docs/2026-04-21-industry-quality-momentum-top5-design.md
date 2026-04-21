# Industry Quality Momentum Top5 — Design

**Date**: 2026-04-21
**Status**: Proposed
**Relates to**: `strategy/configs/quality_momentum_top1.yaml` (reference strategy)

## Goal

Create a second live strategy, `industry_quality_momentum_top5`, that runs in
parallel with the existing `quality_momentum_top1`. It reuses the same factor
(`quality_momentum`, window=20) but applies it to a pool of ~26 single-sector
ETFs selected for Top-N=5 equal-weight allocation, rebalanced daily.

The two strategies must coexist without clobbering each other's state or
notification flow.

## Non-goals

- No weekly / monthly rebalance support — framework stays `daily`-only. If
  turnover proves unacceptable in production, revisit.
- No market-regime risk filter (e.g., MA200 gate, cash sleeve).
- No DingTalk push for the new strategy in the first version — console output
  and state-file writes only. Revisit after 1–3 months of observed signal
  quality.
- No dynamic pool pruning (liquidity screening, survivorship handling) beyond
  what happens naturally: an asset with no data on a given day is silently
  dropped from candidates by the existing factor-validate pipeline.
- No changes to `Top1` or `MomentumRotation` classes.
- No shared-state multi-strategy orchestration; each strategy is invoked by its
  own `run_daily.py --config` command.

## Decisions (locked)

| Area | Decision |
|---|---|
| Relationship to current strategy | Independent, parallel; lives alongside |
| Pool composition | Single-sector (Shenwan-aligned Level-2 industry) ETFs only |
| Selection rule | Top-N equal weight, `N = 5` |
| Rebalance frequency | Daily |
| Factor | `quality_momentum`, unchanged from current, `window=20` |
| Backtest start | `2020-01-01` (pool grows over time as ETFs list) |
| Benchmark | Pool-equal-weight (existing backtest default) |
| DingTalk | Disabled for new strategy (`enable_dingtalk: false` in YAML) |
| State isolation | Per-strategy file: `state/{strategy_name}_position.json` |
| Old state file | Migrated once via manual `git mv` during rollout |

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  run_daily.py --config <yaml>                               │
│    reads config.strategy_name                               │
│    passes strategy_name to execution.position.*             │
│    reads config.enable_dingtalk (default True)              │
└──────────────────────────┬──────────────────────────────────┘
                           │
          ┌────────────────┼──────────────────┐
          ▼                ▼                  ▼
  strategy.topn.TopN   execution.position  notification.dingtalk
  (new class)          (stateful, now      (skipped when
                       per-strategy)        enable_dingtalk=false)
```

### New components

**`strategy/topn.py`** — `TopN` class
- Reads `top_n` from config (default 5)
- Reads `factors[0]` name and `direction_flip`
- Sorts candidates, takes top `K = min(top_n, len(scored))`, equal-weights
- Edge cases: empty input → `{}`; no factors configured → `{}`;
  `top_n > candidates` → equal-weight to all candidates (avoid empty position)

**`strategy/configs/industry_quality_momentum_top5.yaml`** — new config
- `strategy_class: strategy.topn.TopN`
- `strategy_name: industry_quality_momentum_top5`
- `top_n: 5`
- `enable_dingtalk: false`
- `asset_pool`: ~26 industry ETFs (final list fixed during Phase 4)
- `factors: [{name: quality_momentum, weight: 1.0, params: {window: 20}}]`
- `start: 2020-01-01`, `end: 2026-04-21` (or current date)
- `rebalance_rule: daily`, `train_ratio: 0.7`

**`scripts/refresh_asset_names.py`** — one-shot helper
- Queries `pro.fund_basic(market="E")`
- Intersects with union of all strategy YAMLs' `asset_pool`
- Prints suggested `ASSET_NAMES` entries with shortened Chinese names
  (stripping fund-company prefix, index prefix, "ETF" suffix)
- User manually pastes into `notification/formatter.py`

### Modified components

**`execution/position.py`**
- Delete module-level `STATE_FILE` constant
- Add `_state_file(strategy_name: str) -> Path` helper
- Add `strategy_name: str` parameter to `read_position`, `write_position`,
  `save_position`
- No backward-compat shim — callers that pass no argument will `TypeError`

**`run_daily.py`**
- `run(config)` reads `strategy_name = config["strategy_name"]`
- Passes `strategy_name` to all position-layer calls and to
  `_backfill_open_prices`
- Wraps DingTalk call: `if config.get("enable_dingtalk", True): notifier.send(...)`

**`notification/formatter.py`**
- Extend `ASSET_NAMES` dict with new industry-ETF entries (one per final-pool
  code, ~23–26 depending on Phase 4 verification drops)
- No schema change; no logic change

### Unchanged components

- `backtest/runner.py` — already config-driven via `load_strategy(config)`; the
  per-day benchmark is `np.mean(bench_assets)` across assets that have data
  that day, which correctly handles the pool growing as ETFs list.
- `backtest/experiment_log.py`, `backtest/report.py`
- `data/sync.py` / `data/store.py` — `sync_all(asset_pool)` already handles the
  "first-time asset → full history from 2016-01-01" case and its Tushare
  rate-limit retries.
- `factors/quality_momentum.py` — reused as-is.
- `strategy/loader.py` — dispatches via dotted `strategy_class`; no registry
  change needed.
- `strategy/top1.py`, `strategy/momentum_rotation.py` — untouched.

## Asset pool (initial draft)

| Group | Code | Name | Note |
|---|---|---|---|
| Financials / Real Estate | 512880.SH | 证券 | |
| | 512800.SH | 银行 | |
| | 512200.SH | 房地产 | |
| Consumer / Home Appliance | 512690.SH | 酒 | |
| | 159928.SZ | 消费 | |
| | 159996.SZ | 家电 | |
| Healthcare | 512010.SH | 医药 | |
| | 512170.SH | 医疗 | |
| TMT | 512480.SH | 半导体 | |
| | 515880.SH | 通信 | *verify in Phase 4* |
| | 512720.SH | 计算机 | *verify in Phase 4* |
| | 159939.SZ | 信息技术 | |
| | 512980.SH | 传媒 | |
| Advanced Manufacturing | 512660.SH | 军工 | |
| | 515030.SH | 新能源车 | |
| | 516110.SH | 汽车 | |
| | 515790.SH | 光伏 | |
| | 562800.SH | 风电 | *verify in Phase 4* |
| Cyclicals | 515220.SH | 煤炭 | |
| | 512400.SH | 有色金属 | |
| | 515210.SH | 钢铁 | |
| | 159870.SZ | 化工 | |
| Utilities / Agri / Infra | 159611.SZ | 电力 | |
| | 159825.SZ | 农业 | |
| | 516970.SH | 基建 | *verify in Phase 4* |

Codes marked *verify in Phase 4* must be cross-checked against
`pro.fund_basic(market="E")`. If a code is invalid or illiquid, replace with
the largest same-industry ETF or drop from the pool. Final size expected
23–26.

## Implementation phases

Each phase is self-contained and commit-able.

### Phase 1 — `TopN` strategy class
- Create `strategy/topn.py`
- Add `strategy/tests/test_topn.py` covering: empty input, Top-N selection,
  `direction_flip`, `N > candidates`, missing factor config
- Gate: `uv run pytest strategy/tests/` green

### Phase 2 — State isolation refactor
- Refactor `execution/position.py` (add `strategy_name` parameter)
- Update `execution/tests/` call sites
- Manually rename `state/current_position.json` →
  `state/quality_momentum_top1_position.json` (`git mv`)
- Gate: `uv run pytest execution/tests/` green; renamed state file exists

### Phase 3 — `run_daily.py` adaptation
- Thread `strategy_name` through position calls
- Add `enable_dingtalk` config gate
- Gate: running `run_daily.py --config strategy/configs/quality_momentum_top1.yaml`
  produces identical output to pre-refactor (state written to the renamed
  file; DingTalk fired as before)

### Phase 4 — Asset pool preparation
- Write and run `scripts/refresh_asset_names.py`
- Verify *verify in Phase 4* codes against `fund_basic`; substitute or drop as
  needed
- Append new `ASSET_NAMES` entries to `notification/formatter.py`
- Run `sync_all(new_pool)` to populate `data/db/` (first-time full history;
  expect 10–15 min and Tushare quota use)
- Gate: `data/db/` has parquet files for every final-pool code with latest
  date ≤ 5 days behind today

### Phase 5 — New strategy YAML + validation
- Write `strategy/configs/industry_quality_momentum_top5.yaml` with final pool
- Run `run_backtest.py --config <yaml>` → inspect train/test/benchmark metrics
- Dry-run `run_daily.py --config <yaml>` → verify Top-5 picks printed, state
  file created at `state/industry_quality_momentum_top5_position.json`,
  DingTalk not fired
- Gate: backtest report generated, single dry-run day successful

## Risks / open questions

- **Turnover**: daily Top-5 on 26 industries with a smoothed momentum factor
  may or may not churn heavily. Monitor after 1 month; if monthly turnover
  exceeds ~60%, add a `weekly` rebalance-rule extension in a follow-up.
- **Pool drift**: new industry ETFs list over time. Manually curating the YAML
  is acceptable short-term; a periodic `refresh_asset_names.py` review every
  6 months is the maintenance plan.
- **Survivorship / delisting**: current backtest ignores delisted ETFs
  entirely. This matches live behavior (can only trade what exists) but
  inflates historical returns. Accepted for now as a known bias.
- **Tushare quota**: first-time sync of the new ETFs (~23–26) consumes part
  of the daily `pro_bar` quota. If the sync fails partway, re-running is safe
  (incremental).

## File inventory

New:
- `strategy/topn.py`
- `strategy/tests/test_topn.py`
- `strategy/configs/industry_quality_momentum_top5.yaml`
- `scripts/refresh_asset_names.py`

Modified:
- `execution/position.py`
- `execution/tests/*` (call sites)
- `run_daily.py`
- `notification/formatter.py` (`ASSET_NAMES` extension)
- `strategy/CLAUDE.md` (note TopN class alongside Top1 / MomentumRotation)
- `execution/CLAUDE.md` (update STATE_FILE contract)

Runtime-only (no code):
- `state/current_position.json` → `state/quality_momentum_top1_position.json`
  (manual git mv)
- `data/db/*.parquet` (one new file per final-pool code from sync)
