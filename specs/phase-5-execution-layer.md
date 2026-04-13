## Phase 5 — Execution Layer
Module: execution/   |   DESIGN.md §2.5

Interface (from DESIGN.md):
  Input:  (target: dict[str, float], current: dict[str, float])
  Output: list[Order]  where Order = {asset, action, weight_delta}

Requirements:
  1. diff() compares target weights vs current weights, returns list[Order]
  2. action = "buy" if weight increases, "sell" if decreases, "hold" if unchanged
  3. Assets in target but not in current → buy (new position)
  4. Assets in current but not in target → sell (full exit, weight_delta = -current_weight)
  5. weight_delta is always the signed difference: target - current
  6. Orders with weight_delta == 0 should use action "hold" (or be omitted — decide during impl)
  7. state/current_position.json: persist current positions between daily runs
     - read_position() → dict[str, float] (empty dict if file missing)
     - save_position(weights: dict[str, float]) → writes JSON

Acceptance:
  - uv run pytest execution/tests/ all pass
  - diff({A: 0.6, B: 0.4}, {A: 0.5, C: 0.5}) → buy A +0.1, buy B +0.4, sell C -0.5
  - diff({A: 1.0}, {A: 1.0}) → hold A 0.0 (or empty list)
  - diff({A: 1.0}, {}) → buy A +1.0

Non-goals:
  - No actual broker integration (this is weight diff only)
  - No lot-size or slippage calculation
  - No run_daily.py integration (Phase 6 will wire notification + daily run)
