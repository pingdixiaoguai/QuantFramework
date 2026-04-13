# QuantFramework

## Status

| Phase | Deliverable | Status |
|-------|------------|--------|
| 0 | Project scaffold + interface stubs | done |
| 1 | Data layer (Tushare sync, Parquet store, query) | done |
| 2 | Factor layer + Standardization layer | done |
| 3 | Backtest engine | done |
| 4 | Strategy layer | done |
| 5 | Execution layer | done |
| 6 | Notification layer | done |

## Entry Points

- `docs/DESIGN.md` — authoritative architecture source (six-layer design, interface contracts, decision log)
- `specs/` — module spec cards (historical records of requirements per phase)
- `run_daily.py` — live daily run entry point
- `run_backtest.py` — backtest entry point

## Conventions

- **Factor contributor workflow** (DESIGN.md §4.2):
  1. Copy `factors/_template.py`, rename, implement `compute()`
  2. Fill in METADATA (name, params, min_history, direction)
  3. Run `uv run pytest factors/tests/`
  4. Register in `factors/registry.yaml`
  5. PR scope: 1 factor file + 1 registry line
- **Dependencies**: managed via `uv add` / `uv add --dev`, never hand-edit `pyproject.toml` dependencies
- **Testing**: `uv run pytest` from project root
- **Python version**: 3.12 (locked in `.python-version`)

## Deviations

- `standardization.cross_sectional_rank` is a `NotImplementedError` placeholder in Phase 2 — requires multi-asset input, will be implemented in Phase 4.
- `factors/momentum.py` uses `min_history=21` (not 20 as in the spec) because `pct_change(20)` produces 20 NaN rows, requiring 21 data points for the first valid output.
- DESIGN.md §2.2 output validation corrected: Series length = input DataFrame rows (not minus min_history-1), and NaN allowed for first min_history-1 rows (not min_history).
- Strategy layer input differs from DESIGN.md §2.4: actual input is `dict[str, dict[str, float]]` (asset → factor → value, a cross-sectional snapshot) instead of `dict[str, pd.Series]` (standardized factor series). This aligns with how the backtest engine already feeds data to the strategy — one snapshot per trading day.
- The strategy layer performs its own cross-sectional ranking internally (in MomentumRotation), rather than consuming pre-ranked values from the standardization layer. This is a deliberate choice: ranking logic is part of the strategy, and different strategies may rank differently.
