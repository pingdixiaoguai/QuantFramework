# QuantFramework

## Status

| Phase | Deliverable | Status |
|-------|------------|--------|
| 0 | Project scaffold + interface stubs | done |
| 1 | Data layer (Tushare sync, Parquet store, query) | pending |
| 2 | Factor layer + Standardization layer | pending |
| 3 | Backtest engine | pending |
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

(None yet — record any deviations from DESIGN.md here as phases are completed.)
