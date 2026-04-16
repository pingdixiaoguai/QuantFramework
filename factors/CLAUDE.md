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
- Template: `_template.py`. Currently registered: `momentum`, `volatility`, `quality_momentum`

### Known deviations from DESIGN.md
- `momentum.py` uses `min_history=21` (not 20): `pct_change(20)` produces 20 NaN rows, need 21 points for first valid output
- DESIGN.md §2.2 output validation corrected: Series length = input DataFrame rows (not minus min_history-1)

## Pitfalls
- Adding a factor without updating `registry.yaml` → silently ignored, no error
- `params` override: `compute()` merges caller params over METADATA defaults — test both paths
- `direction` field ("higher_better"/"lower_better") is consumed by strategy layer, not factor layer — don't apply direction logic inside factor
- Setting `series.index = df["date"]` is mandatory for `validator.validate()` to pass — by default `pct_change` keeps the RangeIndex
