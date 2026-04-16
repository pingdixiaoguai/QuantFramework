# Strategy Layer

## Contract
Input: `dict[str, dict[str, float]]` (asset → factor → value, cross-sectional snapshot per trading day)
Output: `dict[str, float]` (asset → weight, sum = 1.0; empty dict if no input)

## Implementation Notes
- Base class: `base.py` defines `BaseStrategy.generate_weights()` interface; `__init__(config: dict)` stores `self.config`
- `momentum_rotation.py` (`MomentumRotation`): rank-weighted combiner. For each factor config, sorts assets by that factor's value and awards rank (1..n), flipping if `direction_flip: true`. Weights = `factor_weight * rank` summed across factors, then normalized.
- `top1.py` (`Top1`): all-in on the single asset with the highest score on `factors[0]` (or lowest if `direction_flip: true`). Ignores factors beyond index 0.
- `loader.py` (`load_strategy(config)`): imports class from `config["strategy_class"]` dotted path, defaults to `strategy.momentum_rotation.MomentumRotation`
- Configs live in `strategy/configs/*.yaml` — factor weights, asset pool, rebalance rule are config, not code

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
