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

## Subdirectory CLAUDE.md

Each module directory (`data/`, `factors/`, `standardization/`, `strategy/`, `execution/`, `notification/`, `backtest/`) has its own `CLAUDE.md` containing that module's contract, implementation notes, known deviations, and pitfalls. **These are the authoritative source for module-level context — do not look for deviation notes in this root file.**

**IMPORTANT**: When modifying a module's code, always check if the same directory's `CLAUDE.md` needs updating. Specifically:
- Interface change (input/output format) → update Contract section
- Behavior differs from DESIGN.md → update Known deviations
- Discovered a new gotcha → add to Pitfalls section
If unsure whether an update is needed, read the subdirectory `CLAUDE.md` and verify each statement still holds against the code you just changed.

## Writing Plans 规则
当使用 superpowers:writing-plans 生成实施计划时：
1. 先输出计划的骨架结构（仅标题、任务编号、一句话摘要）
2. 确认骨架无误后，再逐个任务段落填充详细内容
3. 严禁一次性输出完整的长篇计划文档