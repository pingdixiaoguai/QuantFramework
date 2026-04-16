# Backtest Engine

## Contract
`run(config: dict | None) -> BacktestResult` where `BacktestResult = {daily_returns, benchmark_returns, positions, train_end, config}`
`report.generate(result, output_path)` → HTML via quantstats
`experiment_log.save(result, output_dir)` → YAML snapshot under `experiments/`

## Implementation Notes
- `runner.py` — day-by-day traversal over the union of all assets' trading days
  - **Future-info guard**: each day `t` feeds factors only `df[df["date"] <= t]` before calling `compute()`. Enforced by the engine, not trusted to factor code.
  - Each factor output is validated via `factors.validator.validate()` before its last value is used; validation failures emit `warnings.warn` and skip that asset/day
  - Skips assets whose truncated history is shorter than `max(min_history)` across registered factors
  - Returns calculation: weights decided at `t` are applied to the `close[t] / close[t-1] - 1` per-asset return realized at `t+1` (`strat_ret = Σ prev_weights[a] * asset_ret[a]`)
  - Benchmark is equal-weight across `asset_pool` (not `1/len(asset_pool)` weighted by availability — it's `np.mean` of the per-asset returns on that day)
  - Train/test split by day index at `train_ratio` (default 0.7); overfit warning fires when `train_sharpe > 2 × test_sharpe` and both windows have ≥ 20 days
- `report.py` — lazy-imports `quantstats` and calls `qs.reports.html(...)`
- `experiment_log.py` — generates ID `YYYYMMDD-NNN` (sequence = count of existing logs today + 1), serializes config (ISO-formatting `start`/`end` dates), and writes metrics for train/test/full/benchmark slices

### Known deviations from DESIGN.md
- The engine bypasses the standardization layer. `strategy.generate_weights()` receives raw factor values; cross-sectional ranking is done inside the strategy (see `strategy/CLAUDE.md`). DESIGN.md §2.3 implies standardization is a pipeline step.
- The engine skips the execution and notification layers — it only produces a returns series, never emits `Order`s (by design: backtest vs. live split at this boundary).
- Factor computation failures are logged as warnings, not hard errors, so a bad day on one asset won't abort the whole run. DESIGN.md doesn't specify the fault policy.

## Pitfalls
- `max_min_history` is taken across **all** registered factors (in `factor_configs`), so adding a factor with a large `min_history` pushes back the first usable day for every asset
- `strategy_returns` are NOT saved on days where `prev_weights` is `None` (the first valid weighted day) — the daily_returns series is shorter than the trading calendar by the warm-up + first rebalance day
- `positions_df` rows are only appended when the strategy returns non-empty weights; a day with insufficient history produces no row (not a zero row)
- `experiment_log._next_id` is not atomic — concurrent runs can collide on the same ID
- `config["start"]` / `config["end"]` must be `datetime.date` for the ISO serialization path; passing strings bypasses the `isoformat()` branch and is written as-is
- Changing `factors/registry.yaml` between a run and its replay breaks `--from-log` reproducibility; the experiment YAML records `params` but not the registry version
