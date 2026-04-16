# Standardization Layer

## Contract
Input: `dict[str, pd.Series]` (factor name → raw factor series)
Output: `dict[str, pd.Series]` (factor name → standardized series, same keys/index as input)
Dispatch: `standardize(raw, method="cross_sectional_rank", **kwargs)` in `methods.py`

## Implementation Notes
- Available methods: `z_score(raw, window=60)`, `percentile(raw, window=60)` — both use rolling windows
- `cross_sectional_rank` is registered but **raises `NotImplementedError`** — it requires multi-asset input, which this layer's single-asset `dict[str, pd.Series]` shape cannot express
- Unknown method name → `ValueError` listing valid methods
- Operates per-factor independently; no cross-factor normalization

### Known deviations from DESIGN.md
- DESIGN.md §2.3 names `cross_sectional_rank` as the default. It is **not functional** — calling the default raises. Callers must pass `method="z_score"` or `method="percentile"` explicitly.
- In practice the strategy layer does its own cross-sectional ranking over `dict[str, dict[str, float]]` snapshots (see `strategy/CLAUDE.md`), so this layer is currently **bypassed** by the live/backtest pipeline. `backtest/runner.py` feeds raw factor values directly to `strategy.generate_weights()` without calling `standardize()`.

## Pitfalls
- Do not call `standardize()` with the default method in new code — it will raise. Pick `z_score` or `percentile` explicitly.
- `z_score` / `percentile` emit NaN for the first `window - 1` rows on each series. Downstream consumers must handle these or increase the backtest warm-up.
- If you wire standardization into the backtest pipeline later, remember the strategy layer already ranks internally — pick one place to do it or you will double-rank.
- Adding a new method requires registering it in the `methods` dict inside `standardize()` — there is no auto-discovery.
