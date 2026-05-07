# Risk-Adjusted Quality Momentum Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a new factor `risk_adjusted_quality_momentum` alongside the existing `quality_momentum`, fully tested and registered, so cross-asset rotation pools (equity ETFs + bonds + gold + dividend-low-vol) can be ranked on risk-adjusted rather than raw returns.

**Architecture:** New factor file follows the standard factor contract from `factors/CLAUDE.md` (single `compute(df, params)` returning a `pd.Series` indexed by `df["date"]`). The formula is `score = clip(R_N / max(vol_N, floor_N), -3, +3) × Kaufman_ER`, with all return/vol arithmetic in log space for self-consistency. Existing `quality_momentum` is left untouched for A/B comparison.

**Tech Stack:** Python 3.12, pandas, numpy, pytest, uv (package manager).

**Spec:** [docs/superpowers/specs/2026-05-06-risk-adjusted-quality-momentum-design.md](../specs/2026-05-06-risk-adjusted-quality-momentum-design.md)

---

### Task 1: Scaffold factor file with METADATA only

**Files:**
- Create: `factors/risk_adjusted_quality_momentum.py`

**Goal of this task:** Get the module loadable with valid METADATA. No formula yet — `compute()` returns a structurally valid placeholder so the registry's metadata validation can pass before the real formula lands. This isolates "registry can find me" from "formula is right".

- [ ] **Step 1.1: Create the file with full METADATA and a placeholder `compute()`**

Create `factors/risk_adjusted_quality_momentum.py`:

```python
"""风险调整质量动量因子 (Risk-Adjusted Quality Momentum)

公式：
    R_N        = ln(close_t / close_{t-N})
    path_N     = sum(|ln(close_j / close_{j-1})|) over last N
    ER_N       = |R_N| / path_N
    vol_N      = std(daily log return, N) * sqrt(N)
    floor_N    = vol_floor_annual * sqrt(N / 252)
    adj_vol_N  = max(vol_N, floor_N)
    ram        = clip(R_N / adj_vol_N, -3, +3)
    score      = ram * ER_N

相对 quality_momentum 的改动：动量项从原始涨幅替换为风险调整动量，
解决跨资产（股票 / 债券 / 黄金 / 红利低波）轮动时高波动资产被天然偏好的问题。
"""

import numpy as np
import pandas as pd

METADATA = {
    "name": "risk_adjusted_quality_momentum",
    "author": "quantframework",
    "version": "1.0.0",
    "params": {"window": 60, "vol_floor_annual": 0.08},
    "min_history": 61,
    "direction": "higher_better",
    "description": "风险调整动量 × Kaufman 效率比率（对数收益、波动率地板、winsorize），跨资产可比",
}

_WINSOR_LIMIT = 3.0


def compute(df: pd.DataFrame, params: dict | None = None) -> pd.Series:
    p = {**METADATA["params"], **(params or {})}
    n = p["window"]
    # Placeholder: returns a Series of NaN with correct shape/index/dtype.
    # Real formula lands in tasks 2-5.
    series = pd.Series(np.nan, index=range(len(df)), dtype=float)
    series.index = df["date"]
    return series
```

- [ ] **Step 1.2: Verify the module is importable and METADATA is well-formed**

Run:
```bash
uv run python -c "from factors.risk_adjusted_quality_momentum import METADATA, compute; print(METADATA['name'], METADATA['min_history'])"
```
Expected output: `risk_adjusted_quality_momentum 61`

- [ ] **Step 1.3: Commit**

```bash
git add factors/risk_adjusted_quality_momentum.py
git commit -m "feat(factors): scaffold risk_adjusted_quality_momentum module

Placeholder compute() returns NaN; METADATA fully populated.
Real formula lands in subsequent tasks."
```

---

### Task 2: Output shape contract + log-return baseline

**Files:**
- Create: `factors/tests/test_risk_adjusted_quality_momentum.py`
- Modify: `factors/risk_adjusted_quality_momentum.py`

**Goal of this task:** Lock in the contract from `factors/CLAUDE.md` (length matches input, index is `df["date"]`, first `min_history-1 = 60` rows are NaN, rows 60+ are finite floats). Implement just enough of the formula — log return `R_N`, path length, naive `R_N/vol_N` without floor or winsorize — to produce real finite numbers in the tail. The placeholder from Task 1 is replaced.

- [ ] **Step 2.1: Create the test file with shape tests**

Create `factors/tests/test_risk_adjusted_quality_momentum.py`:

```python
"""Tests for factors.risk_adjusted_quality_momentum."""

import numpy as np
import pandas as pd

from factors.risk_adjusted_quality_momentum import METADATA, compute


def _make_df(prices: list[float]) -> pd.DataFrame:
    n = len(prices)
    dates = pd.bdate_range("2024-01-01", periods=n)
    return pd.DataFrame({
        "date": dates,
        "open": prices,
        "high": [p * 1.01 for p in prices],
        "low": [p * 0.99 for p in prices],
        "close": prices,
        "volume": [1000.0] * n,
    })


class TestOutputShape:
    def test_length_matches_input(self):
        df = _make_df([100.0 + i for i in range(80)])
        result = compute(df)
        assert len(result) == len(df)

    def test_index_is_date(self):
        df = _make_df([100.0 + i for i in range(80)])
        result = compute(df)
        assert (result.index == df["date"]).all()

    def test_dtype_is_float(self):
        df = _make_df([100.0 + i for i in range(80)])
        result = compute(df)
        assert result.dtype == float

    def test_first_60_rows_are_nan(self):
        df = _make_df([100.0 + i for i in range(80)])
        result = compute(df)
        # min_history - 1 = 60
        assert result.iloc[:60].isna().all()

    def test_rows_60_and_after_are_finite(self):
        df = _make_df([100.0 + i for i in range(80)])
        result = compute(df)
        tail = result.iloc[60:]
        assert tail.notna().all()
        assert np.isfinite(tail).all()

    def test_input_df_not_mutated(self):
        df = _make_df([100.0 + i for i in range(80)])
        before = df.copy(deep=True)
        compute(df)
        pd.testing.assert_frame_equal(df, before)
```

- [ ] **Step 2.2: Run tests to verify they fail on the placeholder**

Run:
```bash
uv run pytest factors/tests/test_risk_adjusted_quality_momentum.py -v
```
Expected: `test_rows_60_and_after_are_finite` FAILS (placeholder returns all-NaN). Other shape tests pass against the placeholder.

- [ ] **Step 2.3: Replace placeholder with the minimal real formula (log-return / vol, no floor, no winsor)**

Edit `factors/risk_adjusted_quality_momentum.py` — replace the body of `compute()`:

```python
def compute(df: pd.DataFrame, params: dict | None = None) -> pd.Series:
    p = {**METADATA["params"], **(params or {})}
    n = p["window"]
    close = df["close"].astype(float)

    # Daily log return r_t = ln(close_t / close_{t-1})
    log_ret = np.log(close).diff()

    # N-period log return R_N = ln(close_t / close_{t-N})
    R = np.log(close).diff(n)

    # Path length: rolling sum of |r_t| over last N
    path = log_ret.abs().rolling(window=n).sum()

    # N-period log vol = std(r_t, N) * sqrt(N), ddof=1 (sample std)
    vol = log_ret.rolling(window=n).std(ddof=1) * np.sqrt(n)

    # Naive risk-adjusted momentum (floor + winsor land in later tasks)
    ram = R / vol

    series = ram.astype(float)
    series.index = df["date"]
    return series
```

- [ ] **Step 2.4: Run tests to verify shape tests pass**

Run:
```bash
uv run pytest factors/tests/test_risk_adjusted_quality_momentum.py -v
```
Expected: all 6 shape tests PASS. Note: `test_rows_60_and_after_are_finite` passes because monotonically increasing prices produce nonzero `vol` and `R`.

- [ ] **Step 2.5: Run validator-driven check via the registry path**

Run:
```bash
uv run python -c "
import pandas as pd
from factors.risk_adjusted_quality_momentum import compute, METADATA
from factors.validator import validate
dates = pd.bdate_range('2024-01-01', periods=80)
prices = [100.0 + i for i in range(80)]
df = pd.DataFrame({'date': dates, 'open': prices, 'high': prices, 'low': prices, 'close': prices, 'volume': [1000.0]*80})
out = compute(df)
validate(out, df, METADATA['min_history'], METADATA['name'])
print('validator OK')
"
```
Expected output: `validator OK` (no exception).

If `validate` does not exist or has a different signature, run instead:
```bash
uv run pytest factors/tests/test_validator.py -v
```
to confirm validator behavior, then mirror the existing factor's call site.

- [ ] **Step 2.6: Commit**

```bash
git add factors/risk_adjusted_quality_momentum.py factors/tests/test_risk_adjusted_quality_momentum.py
git commit -m "feat(factors): risk_adjusted_quality_momentum shape contract

Implements baseline log-return / vol formula. Floor, ER multiplier
and winsorization land in subsequent commits."
```

---

### Task 3: Efficiency Ratio multiplier

**Files:**
- Modify: `factors/tests/test_risk_adjusted_quality_momentum.py` (append new test class)
- Modify: `factors/risk_adjusted_quality_momentum.py` (extend `compute()`)

**Goal of this task:** Add the Kaufman ER term so that `score = (R / vol) × ER`. Adds two behavioral tests: (a) on a strictly monotonic series ER must equal 1, so the score equals the bare `R/vol` baseline from Task 2; (b) on two paths with identical endpoints, the smoother path must score higher.

- [ ] **Step 3.1: Append ER tests to the test file**

Append to `factors/tests/test_risk_adjusted_quality_momentum.py`:

```python
class TestEfficiencyRatio:
    def test_straight_line_er_equals_one(self):
        """Strictly monotonic prices => ER = 1.0 exactly => score == R/vol."""
        prices = [100.0 + i for i in range(80)]
        df = _make_df(prices)
        result = compute(df)

        # Manual R/vol for the last row
        log_close = np.log(np.array(prices))
        log_ret = np.diff(log_close)
        n = METADATA["params"]["window"]
        R = log_close[-1] - log_close[-1 - n]
        vol = np.std(log_ret[-n:], ddof=1) * np.sqrt(n)
        expected_score = R / vol  # ER = 1, no floor / no winsor at this scale

        assert abs(result.iloc[-1] - expected_score) < 1e-9

    def test_smooth_path_beats_choppy_same_endpoints(self):
        """Two paths, same start and end after >N steps; smoother wins."""
        n_pts = 80
        smooth = [100.0 + i * 0.5 for i in range(n_pts)]

        # Zigzag oscillating but ending at the same point as smooth[-1]
        zigzag = [smooth[0]]
        for i in range(1, n_pts - 1):
            step = 3.0 if i % 2 == 1 else -2.5
            zigzag.append(zigzag[-1] + step)
        zigzag.append(smooth[-1])  # force matching endpoint

        score_smooth = compute(_make_df(smooth)).iloc[-1]
        score_zigzag = compute(_make_df(zigzag)).iloc[-1]
        assert score_smooth > score_zigzag
```

- [ ] **Step 3.2: Run the new tests to verify behavior before adding ER**

Run:
```bash
uv run pytest factors/tests/test_risk_adjusted_quality_momentum.py::TestEfficiencyRatio -v
```
Expected: `test_straight_line_er_equals_one` PASSES on the bare baseline (ER would be 1.0 anyway on a monotonic series — this is the regression guard for after we multiply). `test_smooth_path_beats_choppy_same_endpoints` may already pass on the baseline because the choppy series has higher `vol`; that's fine — multiplying by ER must keep it green and make the gap larger.

If neither test fails on the baseline, that's expected. Proceed with the implementation; Step 3.4 confirms ER is engaged.

- [ ] **Step 3.3: Add the ER term to `compute()`**

Edit `factors/risk_adjusted_quality_momentum.py` — replace the body of `compute()`:

```python
def compute(df: pd.DataFrame, params: dict | None = None) -> pd.Series:
    p = {**METADATA["params"], **(params or {})}
    n = p["window"]
    close = df["close"].astype(float)

    log_ret = np.log(close).diff()
    R = np.log(close).diff(n)
    path = log_ret.abs().rolling(window=n).sum()
    vol = log_ret.rolling(window=n).std(ddof=1) * np.sqrt(n)

    # Kaufman efficiency ratio in [0, 1]
    er = R.abs() / path.replace(0, np.nan)

    ram = R / vol
    score = ram * er

    series = score.astype(float)
    series.index = df["date"]
    return series
```

- [ ] **Step 3.4: Run all tests for this factor to verify**

Run:
```bash
uv run pytest factors/tests/test_risk_adjusted_quality_momentum.py -v
```
Expected: all `TestOutputShape` tests still pass; `TestEfficiencyRatio::test_straight_line_er_equals_one` PASS (because ER==1 exactly on monotonic series); `TestEfficiencyRatio::test_smooth_path_beats_choppy_same_endpoints` PASS.

- [ ] **Step 3.5: Sanity-check ER value on the choppy path**

Run:
```bash
uv run python -c "
import numpy as np, pandas as pd
n_pts = 80
zigzag = [100.0]
for i in range(1, n_pts - 1):
    zigzag.append(zigzag[-1] + (3.0 if i % 2 == 1 else -2.5))
zigzag.append(100.0 + (n_pts - 1) * 0.5)
df = pd.DataFrame({'close': zigzag})
log_ret = np.log(df['close']).diff()
R = np.log(df['close']).diff(60).iloc[-1]
path = log_ret.abs().rolling(60).sum().iloc[-1]
print('ER =', abs(R)/path, 'expected < 0.5')
"
```
Expected output: `ER = <some value> expected < 0.5` (zigzag's |R|/path is small relative to 1).

- [ ] **Step 3.6: Commit**

```bash
git add factors/risk_adjusted_quality_momentum.py factors/tests/test_risk_adjusted_quality_momentum.py
git commit -m "feat(factors): risk_adjusted_quality_momentum adds ER multiplier

Score = (R/vol) * Kaufman_ER. Vol floor and winsorization land next."
```

---

### Task 4: Vol floor

**Files:**
- Modify: `factors/tests/test_risk_adjusted_quality_momentum.py` (append new test class)
- Modify: `factors/risk_adjusted_quality_momentum.py` (extend `compute()`)

**Goal of this task:** Add the volatility floor `floor_N = vol_floor_annual * sqrt(N/252)` so that ultra-low-vol assets (short bonds) can't blow up the denominator. Test the floor-binds case (constant-growth log-price → vol=0 → without the floor `R/vol` is `inf`; with the floor, score is finite and matches the manually computed value). Also test that overriding `vol_floor_annual` via `params` works.

- [ ] **Step 4.1: Append vol-floor tests to the test file**

Append to `factors/tests/test_risk_adjusted_quality_momentum.py`:

```python
class TestVolFloor:
    def test_floor_binds_when_vol_is_zero(self):
        """Constant-rate log growth => vol=0; floor must rescue the divisor."""
        # Tiny constant daily log return; std == 0 exactly
        daily_log = 0.0001
        prices = [100.0 * np.exp(daily_log * i) for i in range(80)]
        df = _make_df(prices)
        result = compute(df)
        last = result.iloc[-1]

        # Manual expected value
        n = METADATA["params"]["window"]                      # 60
        vol_floor_annual = METADATA["params"]["vol_floor_annual"]  # 0.08
        floor_n = vol_floor_annual * np.sqrt(n / 252.0)
        R = daily_log * n
        # ER == 1 (constant positive log return: |R| == path)
        expected = (R / floor_n) * 1.0

        assert np.isfinite(last)
        assert abs(last - expected) < 1e-9

    def test_floor_does_not_bind_for_normal_vol(self):
        """A noisy ~1% daily-vol series: vol_60 ≈ 7.7% > floor ≈ 3.9%, floor irrelevant."""
        rng = np.random.default_rng(seed=42)
        log_rets = rng.normal(loc=0.0005, scale=0.01, size=79)
        log_close = np.concatenate([[np.log(100.0)], np.log(100.0) + np.cumsum(log_rets)])
        prices = list(np.exp(log_close))
        df = _make_df(prices)
        result = compute(df).iloc[-1]

        # Recompute the no-floor score and assert they match exactly
        n = METADATA["params"]["window"]
        log_ret = np.diff(log_close)
        R = log_close[-1] - log_close[-1 - n]
        path = np.sum(np.abs(log_ret[-n:]))
        vol = np.std(log_ret[-n:], ddof=1) * np.sqrt(n)
        er = abs(R) / path
        expected = (R / vol) * er  # vol > floor so adj_vol == vol

        assert abs(result - expected) < 1e-9

    def test_vol_floor_annual_param_override(self):
        """Caller-supplied vol_floor_annual must override METADATA default."""
        daily_log = 0.0001
        prices = [100.0 * np.exp(daily_log * i) for i in range(80)]
        df = _make_df(prices)

        n = METADATA["params"]["window"]
        custom_floor_annual = 0.20  # much larger; floor binds harder
        result = compute(df, params={"vol_floor_annual": custom_floor_annual}).iloc[-1]

        floor_n = custom_floor_annual * np.sqrt(n / 252.0)
        R = daily_log * n
        expected = (R / floor_n) * 1.0
        assert abs(result - expected) < 1e-9
```

- [ ] **Step 4.2: Run the new tests to verify they fail**

Run:
```bash
uv run pytest factors/tests/test_risk_adjusted_quality_momentum.py::TestVolFloor -v
```
Expected: `test_floor_binds_when_vol_is_zero` and `test_vol_floor_annual_param_override` FAIL — Task 3's code divides by `vol == 0`, producing `inf`/`NaN`. `test_floor_does_not_bind_for_normal_vol` PASSES on Task 3's code (because `vol > 0`).

- [ ] **Step 4.3: Add the floor to `compute()`**

Edit `factors/risk_adjusted_quality_momentum.py` — replace the body of `compute()`:

```python
def compute(df: pd.DataFrame, params: dict | None = None) -> pd.Series:
    p = {**METADATA["params"], **(params or {})}
    n = p["window"]
    vol_floor_annual = p["vol_floor_annual"]
    close = df["close"].astype(float)

    log_ret = np.log(close).diff()
    R = np.log(close).diff(n)
    path = log_ret.abs().rolling(window=n).sum()
    vol = log_ret.rolling(window=n).std(ddof=1) * np.sqrt(n)

    floor_n = vol_floor_annual * np.sqrt(n / 252.0)
    adj_vol = vol.where(vol > floor_n, floor_n)

    er = R.abs() / path.replace(0, np.nan)
    ram = R / adj_vol
    score = ram * er

    series = score.astype(float)
    series.index = df["date"]
    return series
```

Note: `vol.where(vol > floor_n, floor_n)` is the pandas equivalent of `max(vol, floor_n)` element-wise; it keeps NaN where `vol` is NaN (positions before `min_history`), preserving the prefix-NaN contract.

- [ ] **Step 4.4: Run the full factor test suite**

Run:
```bash
uv run pytest factors/tests/test_risk_adjusted_quality_momentum.py -v
```
Expected: all tests PASS — shape (6) + ER (2) + vol-floor (3) = 11 tests.

- [ ] **Step 4.5: Commit**

```bash
git add factors/risk_adjusted_quality_momentum.py factors/tests/test_risk_adjusted_quality_momentum.py
git commit -m "feat(factors): risk_adjusted_quality_momentum adds vol floor

adj_vol = max(vol_N, vol_floor_annual * sqrt(N/252)). Prevents
ultra-low-vol assets (short-duration bonds) from being over-rewarded
by a near-zero denominator."
```

---

### Task 5: Winsorization

**Files:**
- Modify: `factors/tests/test_risk_adjusted_quality_momentum.py` (append new test class)
- Modify: `factors/risk_adjusted_quality_momentum.py` (extend `compute()`)

**Goal of this task:** Clip `R / adj_vol` to `[-3, +3]` before multiplying by ER. Force the clip to engage (constant high log return + tiny vol → raw `R/adj_vol ≈ 7.7`) and assert the score equals `3 × ER`. Also test the symmetric negative side.

- [ ] **Step 5.1: Append winsorize tests to the test file**

Append to `factors/tests/test_risk_adjusted_quality_momentum.py`:

```python
class TestWinsorize:
    def test_positive_extreme_is_clipped_to_3(self):
        """High constant log return + bound floor => raw R/adj_vol > 3 => clipped."""
        # 0.5% per day compounding => R_60 = 0.30; vol = 0; floor binds at ~0.039
        daily_log = 0.005
        prices = [100.0 * np.exp(daily_log * i) for i in range(80)]
        df = _make_df(prices)
        result = compute(df).iloc[-1]

        n = METADATA["params"]["window"]
        floor_n = METADATA["params"]["vol_floor_annual"] * np.sqrt(n / 252.0)
        R = daily_log * n
        # Sanity-check the clip engages
        assert R / floor_n > 3.0
        # ER == 1 on a strictly monotonic positive series; expected score = 3 * 1
        assert abs(result - 3.0) < 1e-9

    def test_negative_extreme_is_clipped_to_minus_3(self):
        """Symmetric: -0.5% per day => raw R/adj_vol < -3 => clipped."""
        daily_log = -0.005
        prices = [100.0 * np.exp(daily_log * i) for i in range(80)]
        df = _make_df(prices)
        result = compute(df).iloc[-1]

        n = METADATA["params"]["window"]
        floor_n = METADATA["params"]["vol_floor_annual"] * np.sqrt(n / 252.0)
        R = daily_log * n
        assert R / floor_n < -3.0
        # ER on a strictly monotonic negative series is also 1 (|R| == path)
        assert abs(result - (-3.0)) < 1e-9

    def test_in_range_is_not_clipped(self):
        """Within ±3: score must equal (R/adj_vol)*ER exactly, no clipping."""
        rng = np.random.default_rng(seed=7)
        log_rets = rng.normal(loc=0.0005, scale=0.01, size=79)
        log_close = np.concatenate([[np.log(100.0)], np.log(100.0) + np.cumsum(log_rets)])
        prices = list(np.exp(log_close))
        df = _make_df(prices)
        result = compute(df).iloc[-1]

        n = METADATA["params"]["window"]
        log_ret = np.diff(log_close)
        R = log_close[-1] - log_close[-1 - n]
        path = np.sum(np.abs(log_ret[-n:]))
        vol = np.std(log_ret[-n:], ddof=1) * np.sqrt(n)
        floor_n = METADATA["params"]["vol_floor_annual"] * np.sqrt(n / 252.0)
        adj_vol = max(vol, floor_n)
        raw = R / adj_vol
        # Confirm we're inside the winsor band so this test is actually testing "no clip"
        assert -3.0 < raw < 3.0
        expected = raw * (abs(R) / path)
        assert abs(result - expected) < 1e-9
```

- [ ] **Step 5.2: Run the new tests to verify they fail**

Run:
```bash
uv run pytest factors/tests/test_risk_adjusted_quality_momentum.py::TestWinsorize -v
```
Expected: `test_positive_extreme_is_clipped_to_3` and `test_negative_extreme_is_clipped_to_minus_3` FAIL — Task 4's code returns ~7.69 / -7.69, not 3 / -3. `test_in_range_is_not_clipped` PASSES (no clip needed in that range).

- [ ] **Step 5.3: Add the clip to `compute()`**

Edit `factors/risk_adjusted_quality_momentum.py` — replace the body of `compute()`:

```python
def compute(df: pd.DataFrame, params: dict | None = None) -> pd.Series:
    p = {**METADATA["params"], **(params or {})}
    n = p["window"]
    vol_floor_annual = p["vol_floor_annual"]
    close = df["close"].astype(float)

    log_ret = np.log(close).diff()
    R = np.log(close).diff(n)
    path = log_ret.abs().rolling(window=n).sum()
    vol = log_ret.rolling(window=n).std(ddof=1) * np.sqrt(n)

    floor_n = vol_floor_annual * np.sqrt(n / 252.0)
    adj_vol = vol.where(vol > floor_n, floor_n)

    er = R.abs() / path.replace(0, np.nan)
    ram = (R / adj_vol).clip(lower=-_WINSOR_LIMIT, upper=_WINSOR_LIMIT)
    score = ram * er

    series = score.astype(float)
    series.index = df["date"]
    return series
```

(`_WINSOR_LIMIT = 3.0` is already defined at the module level from Task 1.)

- [ ] **Step 5.4: Run the full factor test suite**

Run:
```bash
uv run pytest factors/tests/test_risk_adjusted_quality_momentum.py -v
```
Expected: all tests PASS — shape (6) + ER (2) + vol-floor (3) + winsorize (3) = 14 tests.

- [ ] **Step 5.5: Commit**

```bash
git add factors/risk_adjusted_quality_momentum.py factors/tests/test_risk_adjusted_quality_momentum.py
git commit -m "feat(factors): risk_adjusted_quality_momentum winsorizes R/adj_vol

Clip risk-adjusted momentum to [-3, +3] before multiplying by ER,
preventing low-vol or sample anomalies from dominating the ranking."
```

---

### Task 6: Cross-asset comparability (integration test)

**Files:**
- Modify: `factors/tests/test_risk_adjusted_quality_momentum.py` (append new test class)

**Goal of this task:** Verify the entire point of the factor — when a high-vol asset has a bigger absolute move but a low-vol asset has a smaller, smoother trend, the new factor must rank the low-vol asset higher. Also assert that the *old-style raw return* would have ranked them in the opposite order, proving the upgrade actually fixes the bias. **No production code changes** in this task; if anything fails here, it points to a real bug in tasks 2–5.

- [ ] **Step 6.1: Append the cross-asset test class**

Append to `factors/tests/test_risk_adjusted_quality_momentum.py`:

```python
class TestCrossAssetComparability:
    """The point of the factor: low-vol clean trend should beat high-vol bigger-but-noisier trend."""

    def _series_with(self, *, mu: float, sigma: float, n_days: int, seed: int) -> list[float]:
        """Geometric Brownian path with given daily log-return mean/std."""
        rng = np.random.default_rng(seed=seed)
        log_rets = rng.normal(loc=mu, scale=sigma, size=n_days - 1)
        log_close = np.concatenate([[np.log(100.0)],
                                    np.log(100.0) + np.cumsum(log_rets)])
        return list(np.exp(log_close))

    def test_low_vol_clean_trend_beats_high_vol_big_move(self):
        n_days = 80
        # High-vol asset: ~+12% over 60d, daily sigma ~1.57% (annualized ~25%)
        prices_high = self._series_with(mu=0.00189, sigma=0.0157, n_days=n_days, seed=11)
        # Low-vol asset: ~+5% over 60d, daily sigma ~0.5% (annualized ~8%)
        prices_low = self._series_with(mu=0.000813, sigma=0.005, n_days=n_days, seed=22)

        score_high = compute(_make_df(prices_high)).iloc[-1]
        score_low = compute(_make_df(prices_low)).iloc[-1]

        # The low-vol clean trend must score higher
        assert score_low > score_high, (
            f"Expected low-vol clean trend to win; got "
            f"score_low={score_low:.4f} vs score_high={score_high:.4f}"
        )

    def test_raw_momentum_would_have_ranked_them_oppositely(self):
        """Sanity check: under the OLD raw-return logic, high-vol asset wins.
        Demonstrates the new factor genuinely fixes the bias."""
        n_days = 80
        prices_high = self._series_with(mu=0.00189, sigma=0.0157, n_days=n_days, seed=11)
        prices_low = self._series_with(mu=0.000813, sigma=0.005, n_days=n_days, seed=22)

        n = METADATA["params"]["window"]
        raw_mom_high = prices_high[-1] / prices_high[-1 - n] - 1
        raw_mom_low = prices_low[-1] / prices_low[-1 - n] - 1
        assert raw_mom_high > raw_mom_low, (
            "Test setup invalid: raw momentum should favor the high-vol asset "
            "for this comparison to be meaningful."
        )
```

- [ ] **Step 6.2: Run the new tests**

Run:
```bash
uv run pytest factors/tests/test_risk_adjusted_quality_momentum.py::TestCrossAssetComparability -v
```
Expected: both tests PASS.

If `test_low_vol_clean_trend_beats_high_vol_big_move` fails, that's a real signal — likely one of:
- vol is being computed in the wrong space (simple vs log returns)
- the floor isn't binding when it should
- ER is canceling out the risk adjustment

Investigate before patching the test. The seeds are fixed, so behavior is reproducible.

- [ ] **Step 6.3: Run the full factor test file once more for the green sweep**

Run:
```bash
uv run pytest factors/tests/test_risk_adjusted_quality_momentum.py -v
```
Expected: 16 tests PASS (shape 6 + ER 2 + floor 3 + winsorize 3 + cross-asset 2).

- [ ] **Step 6.4: Commit**

```bash
git add factors/tests/test_risk_adjusted_quality_momentum.py
git commit -m "test(factors): risk_adjusted_quality_momentum cross-asset ranking

Verifies the design intent: low-vol clean trend beats high-vol
bigger-but-noisier trend, while old-style raw-return ranking would
have done the opposite."
```

---

### Task 7: Register the factor and update the test asserting registry contents

**Files:**
- Modify: `factors/registry.yaml` (append 1 line)
- Modify: `factors/tests/test_registry.py` (update hardcoded count + name list)
- Modify: `factors/CLAUDE.md` (update "Currently registered" line)

**Goal of this task:** Make the new factor reachable via `load_registered_factors()`, fix the registry test that hardcodes `len(facs) == 3`, update the module docs, and run the full factor test suite end-to-end.

- [ ] **Step 7.1: Append to `factors/registry.yaml`**

Edit `factors/registry.yaml` to read:

```yaml
factors:
  - module: factors.momentum
  - module: factors.volatility
  - module: factors.quality_momentum
  - module: factors.risk_adjusted_quality_momentum
```

- [ ] **Step 7.2: Update `factors/tests/test_registry.py`**

Edit `factors/tests/test_registry.py` — replace the body of `TestLoadBothFactors.test_returns_momentum_and_volatility`:

```python
class TestLoadBothFactors:
    def test_returns_momentum_and_volatility(self):
        facs = load_registered_factors()
        assert "momentum" in facs
        assert "volatility" in facs
        assert "quality_momentum" in facs
        assert "risk_adjusted_quality_momentum" in facs
        assert len(facs) == 4
        for name, fac in facs.items():
            assert "METADATA" in fac
            assert "compute" in fac
            assert callable(fac["compute"])
```

(The class name says "BothFactors" — that name was already stale before this PR; not in scope to rename here. Leave it.)

- [ ] **Step 7.3: Run the registry test to verify the new factor loads**

Run:
```bash
uv run pytest factors/tests/test_registry.py -v
```
Expected: both tests PASS. If `test_missing_direction_raises` flakes, it's unrelated to this task — investigate before proceeding.

- [ ] **Step 7.4: End-to-end registry sanity check**

Run:
```bash
uv run python -c "
from factors.registry import load_registered_factors
facs = load_registered_factors()
fac = facs['risk_adjusted_quality_momentum']
print('name:', fac['METADATA']['name'])
print('window:', fac['METADATA']['params']['window'])
print('vol_floor_annual:', fac['METADATA']['params']['vol_floor_annual'])
print('min_history:', fac['METADATA']['min_history'])
print('direction:', fac['METADATA']['direction'])
print('callable:', callable(fac['compute']))
"
```
Expected output:
```
name: risk_adjusted_quality_momentum
window: 60
vol_floor_annual: 0.08
min_history: 61
direction: higher_better
callable: True
```

- [ ] **Step 7.5: Update `factors/CLAUDE.md`**

Edit `factors/CLAUDE.md` — find the line:

```
- Template: `_template.py`. Currently registered: `momentum`, `volatility`, `quality_momentum`
```

Replace with:

```
- Template: `_template.py`. Currently registered: `momentum`, `volatility`, `quality_momentum`, `risk_adjusted_quality_momentum`
```

- [ ] **Step 7.6: Run the entire factor test suite**

Run:
```bash
uv run pytest factors/tests/ -v
```
Expected: full green sweep — every existing factor test still passes, plus the 16 new tests for `risk_adjusted_quality_momentum`.

If any pre-existing test broke, that points to either (a) a stray import side effect from the new module or (b) a contract violation in the new factor — investigate before committing.

- [ ] **Step 7.7: Commit**

```bash
git add factors/registry.yaml factors/tests/test_registry.py factors/CLAUDE.md
git commit -m "feat(factors): register risk_adjusted_quality_momentum

- Add to registry.yaml
- Update test_registry expected count (3 -> 4) and name set
- Update factors/CLAUDE.md Currently registered list"
```

---

## Acceptance criteria

- `uv run pytest factors/tests/test_risk_adjusted_quality_momentum.py` — 16 tests pass
- `uv run pytest factors/tests/` — full factor suite passes (registry + 4 factors)
- `factors/registry.yaml` lists `factors.risk_adjusted_quality_momentum`
- `factors/CLAUDE.md` "Currently registered" line includes the new factor
- Existing `quality_momentum.py` and `quality_momentum_top1.yaml` are NOT modified

