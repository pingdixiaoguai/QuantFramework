# Execution Layer

## Contract
Input: `diff(target: dict[str, float], current: dict[str, float])`
Output: `list[Order]` where `Order = {asset: str, action: "buy"|"sell"|"hold", weight_delta: float}`

## Implementation Notes
- `interfaces.py` — the `Order` dataclass and pure-functional `diff()` entry point
- `diff()` iterates the union of target and current assets in sorted order; one `Order` per asset (including `hold` when `delta == 0`)
- Sign convention: `weight_delta = target - current` — positive means buy, negative means sell. Assets only in `current` produce a sell with `weight_delta = -current_weight`
- `position.py` — position state persistence at `state/current_position.json` (outside DESIGN.md §2.5 scope; added for live runs)
  - `PositionState` tracks `weights`, `entry_date`, `entry_prices`, and a `ytd_history` list of closed `PositionPeriod`s
  - `save_position(target, entry_date, entry_prices)` archives the outgoing position into `ytd_history` with `exit_prices=None` (backfilled by the next run)
  - `read_position()` auto-migrates the legacy flat `{asset: weight}` format to the current schema

### Known deviations from DESIGN.md
- DESIGN.md §2.5 states "execution layer only does diff; it does not know why we're rebalancing." That still holds for `diff()`, but the module **also** owns position-state persistence (`position.py`), which is not in the design doc. Notification/daily-run code depends on it.
- `ytd_history` entries carry `exit_prices=None` until the next daily run backfills them — consumers must tolerate the null.

## Pitfalls
- `hold` orders are emitted with `weight_delta=0` and are included in the list — filter them out if you only want actionable instructions (see `notification/formatter.py`)
- `save_position()` unconditionally archives the current position whenever called, even with identical weights. Only call it on an actual rebalance, or `ytd_history` will fill with duplicates (see root `CLAUDE.md` recent commit `993dba9`)
- `STATE_FILE` path is computed relative to `position.py` — don't move the file without updating the `resolve().parent.parent` chain
- Legacy flat-dict JSON files have no `entry_date`/`entry_prices`; the first live run after migration will have `None` for these, which cascades into notification's "—" placeholders
