# QuantFramework

## Status

| Phase | Deliverable | Status |
|-------|------------|--------|
| 0 | Project scaffold + interface stubs | done |
| 1 | Data layer (Tushare sync, Parquet store, query) | done |
| 2 | Factor layer + Standardization layer | done |
| 3 | Backtest engine | done |
| 4 | Strategy layer | pending |
| 5 | Execution layer | pending |
| 6 | Notification layer | pending |

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
- Backtest engine (Phase 3) uses an inline factor-weighted ranking strategy since the strategy layer (Phase 4) is not yet built. This will be replaced with the pluggable strategy interface in Phase 4.
