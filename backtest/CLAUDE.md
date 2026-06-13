# Backtest Engine

## Contract
`run(config: dict | None) -> BacktestResult` where `BacktestResult = {daily_returns, benchmark_returns, positions, train_end, config, baseline_strategy_name?, gross_daily_returns?, turnover?, costs?}`
`report.generate(result, output_path, benchmark_title=None)` → HTML via quantstats; `benchmark_title` labels the benchmark series when set
`experiment_log.save(result, output_dir)` → YAML snapshot under `experiments/`

## Implementation Notes
- `runner.py` — day-by-day traversal over the union of all assets' trading days
  - **Future-info guard**: each day `t` feeds factors only `df[df["date"] <= t]` before calling `compute()`. Enforced by the engine, not trusted to factor code.
  - Each factor output is validated via `factors.validator.validate()` before its last value is used; validation failures emit `warnings.warn` and skip that asset/day
  - Skips assets whose truncated history is shorter than `max(min_history)` across registered factors
  - Returns calculation uses live open-execution semantics. A signal generated at `t` close is executed at the next trading day's open. On an execution day, the outgoing holding earns the overnight `close[t] -> open[t+1]` return, then the incoming holding earns the intraday `open[t+1] -> close[t+1]` return. On non-execution days, the carried position earns ordinary close-to-close return.
  - Benchmark is equal-weight across `asset_pool` (not `1/len(asset_pool)` weighted by availability — it's `np.mean` of the per-asset returns on that day)
  - Transaction costs are optional via `config["transaction_cost_rate"]` (one-side decimal fee). Costs are deducted from strategy returns on actual open execution days as `rate * Σ_assets |w_new - w_old|`. `daily_returns` is net of cost; `gross_daily_returns`, `turnover`, and `costs` retain the decomposition.
  - Train/test split by day index at `train_ratio` (default 0.7); overfit warning fires when `train_sharpe > 2 × test_sharpe` and both windows have ≥ 20 days
  - **Rebalance timing** (`config["rebalance_mode"]`, default `min_hold`; `config["rebalance_days"]`, int, default 1): the engine tracks actual open entry dates. `min_hold` carries the current position while `holding_days < rebalance_days`, then re-evaluates daily until a different non-empty target is scheduled for the next open. `fixed_cycle` only calls `strategy.generate_weights()` on held-day multiples of `rebalance_days` (N, 2N, 3N...).
- `report.py` — lazy-imports `quantstats` and calls `qs.reports.html(...)`
- `experiment_log.py` — generates ID `YYYYMMDD-NNN` (sequence = count of existing logs today + 1), serializes config (ISO-formatting `start`/`end` dates), and writes metrics for train/test/full/benchmark slices. When `result.baseline_strategy_name` is set, an optional `baseline_config: <strategy_name>` field is appended at the end of the `experiment` block; absent otherwise to keep historical YAMLs unchanged.

### Known deviations from DESIGN.md
- The engine bypasses the standardization layer. `strategy.generate_weights()` receives raw factor values; cross-sectional ranking is done inside the strategy (see `strategy/CLAUDE.md`). DESIGN.md §2.3 implies standardization is a pipeline step.
- The engine skips the execution and notification layers — it only produces a returns series, never emits `Order`s (by design: backtest vs. live split at this boundary).
- Factor computation failures are logged as warnings, not hard errors, so a bad day on one asset won't abort the whole run. DESIGN.md doesn't specify the fault policy.

## Pitfalls
- `max_min_history` is taken across **all** registered factors (in `factor_configs`), so adding a factor with a large `min_history` pushes back the first usable day for every asset
- `strategy_returns` are NOT saved before the first position has actually opened. The first saved return is the entry day's open-to-close return.
- `positions_df` rows are appended on actual open execution days, not signal days. With `rebalance_days > 1`, intermediate days have no row even though the position is still held (forward-fill the DataFrame downstream if daily position rows are needed).
- `experiment_log._next_id` is not atomic — concurrent runs can collide on the same ID
- `backtest.runner.run()` expects `config["start"]` / `config["end"]` as `datetime.date`. `run_backtest.py` normalizes YAML strings first, including dynamic `end: "today"` / omitted `end`; direct callers must do their own normalization.
- Changing `factors/registry.yaml` between a run and its replay breaks `--from-log` reproducibility; the experiment YAML records `params` but not the registry version
