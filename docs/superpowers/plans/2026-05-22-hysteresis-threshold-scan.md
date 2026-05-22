# Hysteresis Threshold Scan Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add research-only hysteresis support to the Top1 backtest path and produce archived attribution and threshold-scan evidence for the post-2026-06-02 decision.

**Architecture:** Extend the strategy call contract with optional executed holdings context, pass that context only from `backtest.runner`, and keep `Top1` stateless. A focused research script will rerun the rebalance-days=5 Top1 path, derive cost-adjusted trade ledgers from backtest returns and executed position rows, archive reconstructed 2026-05-21 attribution evidence when original artifacts are unavailable, then emit raw CSV plus Markdown evidence for the hysteresis scan.

**Tech Stack:** Python 3.12, pandas, numpy, pytest, PyYAML, local Parquet data, uv.

**Spec:** [docs/superpowers/specs/2026-05-22-hysteresis-threshold-scan-design.md](../specs/2026-05-22-hysteresis-threshold-scan-design.md)

---

## File Map

- Modify `strategy/base.py`: strategy method signature and contract.
- Modify `strategy/top1.py`: stateless Top1 hysteresis selection.
- Modify `strategy/topn.py` and `strategy/momentum_rotation.py`: accept optional context while preserving behavior.
- Modify `strategy/tests/test_top1.py`: threshold unit coverage.
- Modify `backtest/runner.py`: pass executed current weights to strategy.
- Modify `backtest/tests/test_runner.py`: runner context and post-hold-floor reevaluation coverage.
- Create `scripts/hysteresis_threshold_scan.py`: research-only metrics, evidence CSVs, Markdown attachment generation.
- Create `tests/test_hysteresis_threshold_scan.py`: metric and ledger helper tests.
- Create `strategy_changelog_attachments/`: archived Markdown and CSV evidence.

### Task 1: Extend Top1 With Stateless Hysteresis

**Files:**
- Modify: `strategy/base.py`
- Modify: `strategy/top1.py`
- Modify: `strategy/topn.py`
- Modify: `strategy/momentum_rotation.py`
- Modify: `strategy/tests/test_top1.py`

- [ ] **Step 1: Write failing Top1 tests**

Add these tests to `strategy/tests/test_top1.py`:

```python
    def test_hysteresis_keeps_incumbent_below_threshold(self):
        s = Top1({**_config(), "hysteresis_threshold": 0.05})
        result = s.generate_weights(
            {"A.SH": {"qmom": 0.50}, "B.SH": {"qmom": 0.53}},
            current_weights={"A.SH": 1.0},
        )
        assert result == {"A.SH": 1.0}

    def test_hysteresis_switches_when_challenger_clears_threshold(self):
        s = Top1({**_config(), "hysteresis_threshold": 0.05})
        result = s.generate_weights(
            {"A.SH": {"qmom": 0.50}, "B.SH": {"qmom": 0.56}},
            current_weights={"A.SH": 1.0},
        )
        assert result == {"B.SH": 1.0}

    def test_hysteresis_falls_back_when_incumbent_score_missing(self):
        s = Top1({**_config(), "hysteresis_threshold": 0.05})
        result = s.generate_weights(
            {"B.SH": {"qmom": 0.53}},
            current_weights={"A.SH": 1.0},
        )
        assert result == {"B.SH": 1.0}

    def test_direction_flip_hysteresis_requires_lower_challenger(self):
        config = _config([{"name": "vol", "weight": 1.0, "direction_flip": True}])
        s = Top1({**config, "hysteresis_threshold": 0.05})
        result = s.generate_weights(
            {"A.SH": {"vol": 0.50}, "B.SH": {"vol": 0.46}},
            current_weights={"A.SH": 1.0},
        )
        assert result == {"A.SH": 1.0}
```

- [ ] **Step 2: Run the tests and confirm the contract fails**

Run:

```powershell
uv run python -m pytest strategy/tests/test_top1.py -q
```

Expected: new tests fail because `Top1.generate_weights()` does not accept `current_weights`.

- [ ] **Step 3: Extend signatures and implement Top1 threshold behavior**

Use this contract shape in `strategy/base.py`, `strategy/topn.py`, and `strategy/momentum_rotation.py`:

```python
def generate_weights(
    self,
    factor_values: dict[str, dict[str, float]],
    current_weights: dict[str, float] | None = None,
) -> dict[str, float]:
```

Implement Top1 with these helpers:

```python
    def _incumbent_asset(self, current_weights: dict[str, float] | None) -> str | None:
        if not current_weights or len(current_weights) != 1:
            return None
        incumbent, weight = next(iter(current_weights.items()))
        return incumbent if weight > 0 else None
```

Then after building `scored`, keep current ranking when `tau <= 0`, incumbent is absent, or incumbent is missing from `scored`; otherwise choose the challenger only when it strictly clears the higher-better or lower-better threshold.

- [ ] **Step 4: Run Top1 tests**

Run:

```powershell
uv run python -m pytest strategy/tests/test_top1.py strategy/tests/test_topn.py strategy/tests/test_momentum_rotation.py -q
```

Expected: strategy tests pass.

### Task 2: Pass Executed Holdings From The Backtest Runner

**Files:**
- Modify: `backtest/runner.py`
- Modify: `backtest/tests/test_runner.py`

- [ ] **Step 1: Add runner tests for current holdings context**

Add a synthetic strategy spy to `backtest/tests/test_runner.py`:

```python
class TestStrategyCurrentWeights:
    def test_runner_passes_executed_current_weights(self, monkeypatch):
        seen = []

        class SpyTop1:
            def __init__(self, config):
                self.config = config

            def generate_weights(self, factor_values, current_weights=None):
                seen.append(dict(current_weights or {}))
                return {"A.SH": 1.0}
```

Patch `backtest.runner.load_strategy` to return `SpyTop1`, use the existing synthetic one-factor fixtures, and assert the first valid signal sees `{}` while a later post-entry signal sees `{"A.SH": 1.0}`.

- [ ] **Step 2: Run the runner test and confirm it fails**

Run:

```powershell
uv run python -m pytest backtest/tests/test_runner.py::TestStrategyCurrentWeights -q
```

Expected: the later signal never receives current holdings because runner still calls `generate_weights(asset_factor_values)`.

- [ ] **Step 3: Pass `current_weights` into runner strategy calls**

Change the runner call to:

```python
new_weights = strategy.generate_weights(
    asset_factor_values,
    current_weights=current_weights,
)
```

Update any test spies with an optional `current_weights` parameter so existing runner coverage keeps measuring the same behavior.

- [ ] **Step 4: Add threshold reevaluation regression**

Use a synthetic Top1 score path where the hold floor elapses, the challenger first remains inside `tau`, then clears it on the following day. Assert positions switch only on the next open after the clearing day, proving hysteresis suppression does not reset the five-day floor.

- [ ] **Step 5: Run runner tests**

Run:

```powershell
uv run python -m pytest backtest/tests/test_runner.py -q
```

Expected: runner tests pass.

### Task 3: Build Research Metrics Helpers

**Files:**
- Create: `scripts/hysteresis_threshold_scan.py`
- Create: `tests/test_hysteresis_threshold_scan.py`

- [ ] **Step 1: Write helper tests**

Add tests that cover:

```python
def test_cost_adjusted_returns_charge_entry_and_full_switch():
    raw = pd.Series([0.01, 0.02], index=pd.to_datetime(["2024-01-02", "2024-01-03"]))
    positions = pd.DataFrame(
        [{"date": pd.Timestamp("2024-01-02"), "A.SH": 1.0},
         {"date": pd.Timestamp("2024-01-03"), "B.SH": 1.0}]
    ).set_index("date")
    adjusted, trades = apply_transaction_costs(raw, positions, cost_rate=0.0001)
    assert trades["traded_weight"].tolist() == [1.0, 2.0]
    assert adjusted.iloc[0] == pytest.approx(0.01 - 0.0001)
    assert adjusted.iloc[1] == pytest.approx(0.02 - 0.0002)
```

Cover annualized turnover, position period extraction, period P&L compounding, and Markdown/CSV output paths with `tmp_path`.

- [ ] **Step 2: Run helper tests and confirm they fail**

Run:

```powershell
uv run python -m pytest tests/test_hysteresis_threshold_scan.py -q
```

Expected: import failure because the research script does not exist.

- [ ] **Step 3: Implement reusable helpers**

Implement functions named:

```python
def forward_filled_positions(result: BacktestResult) -> pd.DataFrame: ...
def apply_transaction_costs(raw_returns: pd.Series, positions: pd.DataFrame, cost_rate: float) -> tuple[pd.Series, pd.DataFrame]: ...
def extract_position_periods(positions: pd.DataFrame, returns: pd.Series) -> pd.DataFrame: ...
def summarize_metrics(returns: pd.Series, trade_ledger: pd.DataFrame, periods: pd.DataFrame) -> dict[str, float]: ...
def write_csv(df: pd.DataFrame, path: Path) -> None: ...
```

`apply_transaction_costs` must charge `cost_rate * sum(abs(new_weights - old_weights))` on executed position dates so a first full buy costs one side and a full asset switch costs two sides.

- [ ] **Step 4: Run helper tests**

Run:

```powershell
uv run python -m pytest tests/test_hysteresis_threshold_scan.py -q
```

Expected: helper tests pass.

### Task 4: Emit Attribution And Hysteresis Evidence

**Files:**
- Modify: `scripts/hysteresis_threshold_scan.py`
- Create: `strategy_changelog_attachments/2026-05-21_attribution_reconstruction.md`
- Create: `strategy_changelog_attachments/2026-05-21_attribution_*.csv`
- Create: `strategy_changelog_attachments/YYYY-MM-DD_hysteresis_scan.md`
- Create: `strategy_changelog_attachments/YYYY-MM-DD_hysteresis_*.csv`

- [ ] **Step 1: Implement deterministic scan inputs**

Load `strategy/configs/quality_momentum_top1.yaml`, override:

```python
config["start"] = date(2014, 1, 1)
config["rebalance_days"] = 5
config["hysteresis_threshold"] = tau
```

Run the asset-pool completeness guard after backtest results are available by trimming the evaluation start to the first date on which every configured asset has local data and factor history has produced strategy returns.

- [ ] **Step 2: Add baseline gate**

Run a plain baseline config without `hysteresis_threshold`, run `tau=0`, and fail with `RuntimeError` unless daily return indexes, raw daily returns, and executed positions are exactly equal before cost adjustment.

- [ ] **Step 3: Archive reconstructed 2026-05-21 evidence**

When no original 2026-05-21 Markdown/CSV exists, generate a reconstruction Markdown whose first paragraph says it is reconstructed on the current run date from local data because original raw artifacts were absent from the repository. Save the raw metric table, trade ledger, position periods, drawdown episodes, and whipsaw rows as `2026-05-21_attribution_*.csv`.

- [ ] **Step 4: Define threshold trade-off diagnostics in code and output**

Emit raw rows for:

- standard panel by `tau`
- trade ledger and score samples by `tau`
- drawdown episode summaries for `2015-10`, `2020-09`, `2024-10`, `2025-10`
- suppressed or delayed plain-baseline switches compared with each `tau` path
- `2024-09-26` `159915.SZ` canary rows

Define whipsaw rows from position periods that switch away and then return to the prior asset within the next two executed switches; report the compounded P&L of the intervening non-incumbent period as whipsaw P&L, and state this executable reconstruction definition in the Markdown.

- [ ] **Step 5: Run the research scan**

Run:

```powershell
uv run python scripts/hysteresis_threshold_scan.py
```

Expected: attachments directory contains the reconstructed attribution archive plus the actual-date hysteresis Markdown and CSVs.

### Task 5: Verify The Research Boundary

**Files:**
- Inspect modified code and generated attachments.

- [ ] **Step 1: Run focused tests**

Run:

```powershell
uv run python -m pytest strategy/tests/test_top1.py strategy/tests/test_topn.py strategy/tests/test_momentum_rotation.py backtest/tests/test_runner.py tests/test_hysteresis_threshold_scan.py -q
```

Expected: all focused tests pass.

- [ ] **Step 2: Check forbidden paths**

Run:

```powershell
git diff --name-only
```

Expected: no diffs in `run_daily.py`, `backfill_ytd.py`, or `strategy/configs/quality_momentum_top1.yaml`.

- [ ] **Step 3: Review attachment claims**

Read the generated Markdown and confirm it states:

- no deployment decision is made
- `tau=0` baseline gate result
- any 2026-05-21 output is labeled original or reconstructed
- `2024-10` single-asset crash evidence is separated from switch-heavy drawdown segments

