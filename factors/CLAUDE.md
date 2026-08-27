# Factor Layer

## Contract
Input: `pd.DataFrame` (date, open, high, low, close, volume) — single asset
Output: `pd.Series` (date index, float) — raw factor values

## Implementation Notes
- Each factor is a standalone `.py` with `METADATA` dict + `compute(df, params=None)` function
- Registration: must add entry to `registry.yaml` to be loaded; unregistered files are ignored
- `registry.py` enforces required METADATA fields (`name, author, version, params, min_history, direction, description`) and validates `direction ∈ {higher_better, lower_better}`; violations raise `RuntimeError` at load time
- Output validation (`validator.py`): Series length == input rows, index == `df["date"]`, dtype is float, no NaN from position `min_history - 1` onward
- Factor receives `df.copy()` — must not mutate input
- Template: `_template.py`. Currently registered: `momentum`, `volatility`, `quality_momentum`, `legacy_quality_momentum`, `three_factor_trend`, `ohlc_quality_momentum`, `risk_adjusted_quality_momentum`, `rsi`, `drawdown_percentile`, `rebound_percentile`, `volume_percentile`

### Known deviations from DESIGN.md
- `momentum.py` uses `min_history=21` (not 20): `pct_change(20)` produces 20 NaN rows, need 21 points for first valid output
- DESIGN.md §2.2 output validation corrected: Series length = input DataFrame rows (not minus min_history-1)

## Pitfalls
- Adding a factor without updating `registry.yaml` → silently ignored, no error
- `params` override: `compute()` merges caller params over METADATA defaults — test both paths
- `direction` field ("higher_better"/"lower_better") is consumed by strategy layer, not factor layer — don't apply direction logic inside factor
- Setting `series.index = df["date"]` is mandatory for `validator.validate()` to pass — by default `pct_change` keeps the RangeIndex
- `ohlc_quality_momentum` consumes the post-adjusted `open/high/low/close` columns and keeps its four path weights in the strategy YAML; do not silently substitute raw OHLC data.
- Return convention is per-factor, not uniform: `momentum` and `volatility` use simple returns; `quality_momentum` version 2 uses log-return momentum and a log-path Kaufman ER; `risk_adjusted_quality_momentum` uses log returns throughout (numerator R, vol denominator, Kaufman path). When refactoring shared helpers, don't assume one convention.
- `three_factor_trend` keeps signed 20-day simple-return momentum and simple-return volatility; log prices are used only for the short-window R-squared path-linearity term. Its default parameters are a shadow-research checkpoint, not production defaults.
- `legacy_quality_momentum` is report-only compatibility code for the genuine pre-v2 baseline (simple MOM × price-path ER); never substitute it into the formal v3 signal.
