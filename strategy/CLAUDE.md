# Strategy Layer

## Contract
Input: `dict[str, dict[str, float]]` (asset → factor → value, cross-sectional snapshot per trading day)
Output: `dict[str, float]` (asset → weight, sum = 1.0; empty dict if no input)

## Implementation Notes
- Base class: `base.py` defines `BaseStrategy.generate_weights()` interface; `__init__(config: dict)` stores `self.config`
- `momentum_rotation.py` (`MomentumRotation`): rank-weighted combiner. For each factor config, sorts assets by that factor's value and awards rank (1..n), flipping if `direction_flip: true`. Weights = `factor_weight * rank` summed across factors, then normalized.
- `top1.py` (`Top1`): all-in on the single asset with the highest score on `factors[0]` (or lowest if `direction_flip: true`). Ignores factors beyond index 0.
- `topn.py` (`TopN`): equal-weight on the top `top_n` assets by `factors[0]` (lowest `top_n` if `direction_flip: true`). Reads `top_n` from config (default 5). When `top_n > len(scored)` falls back to equal-weight on all candidates rather than emitting an empty position.
- `loader.py` (`load_strategy(config)`): imports class from `config["strategy_class"]` dotted path, defaults to `strategy.momentum_rotation.MomentumRotation`
- `rolling_ohlc_er.py` is a notification-only shadow strategy for the 4ETF production config. It keeps quarterly OHLC path weights, rolls them forward using the prior 1008 union-calendar trading dates, searches ±0.05 around the previous quarter at 0.01 steps, and averages the training-Sharpe Top10. Its output never enters production target weights, execution diff, or position persistence.
- Configs live in `strategy/configs/*.yaml` — factor weights, asset pool, rebalance rule are config, not code. Backtest configs may set `end: "today"` (or omit `end`) to run through the current date; `run_backtest.py` normalizes that before calling the engine.
- **Rebalance timing** is set at config level via `rebalance_mode` + `rebalance_days: N` (int, default 1). `rebalance_mode` defaults to `min_hold`: suppress changes while `holding_days < N`, then re-evaluate daily once the hold window has elapsed. `fixed_cycle` only evaluates on held-day multiples of `N` (N, 2N, 3N...). The backtest engine, `run_daily.py`, and `backfill_ytd.py` share `strategy.rebalance.should_hold_position()`. The legacy `rebalance_rule: daily` field is ignored — kept in old configs for back-compat only.

### Known deviations from DESIGN.md
- Input format: actual is `dict[str, dict[str, float]]` (snapshot at time t), not `dict[str, pd.Series]` (full series). Aligns with how the backtest engine feeds per-day data.
- Cross-sectional ranking happens inside the strategy, not in the standardization layer. Different strategies may rank differently, and `standardization.cross_sectional_rank` is currently not implemented.
- `MomentumRotation` with a single asset shortcuts to `{asset: 1.0}` regardless of factor values (avoids divide-by-zero in rank normalization).

## Pitfalls
- Changing a config YAML schema without updating the strategy's parsing → silent wrong behavior
- Weights must sum to 1.0 — `MomentumRotation` guarantees this via normalization, but new strategies are responsible for their own sums
- Adding a new asset to `asset_pool` in config requires that asset exists in the data layer Parquet store (`data/db/`)
- `direction_flip` on a factor only affects ranking inside the strategy; `METADATA["direction"]` on the factor itself is informational — the strategy decides how to use it
- Registering a new strategy class means adding the dotted path to `config["strategy_class"]`; there is no registry file
- The rolling OHLC ER seed in `quality_momentum_top1.yaml` is a reproducible checkpoint, not a production position. Quarter updates are cached in ignored `state/*_rolling_ohlc_er.json`; deleting the cache causes deterministic replay from the seed.
