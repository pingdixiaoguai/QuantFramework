# Phase 2 — 因子层 + 标准化层

Module: `factors/`, `standardization/`  |  DESIGN.md §2.2、§2.3

> 规格卡是**一次性任务输入**。实现完成后由代码和测试承担文档职责，本文件不再维护。

---

## 目标

让"新增一个因子"的改动收敛到一个文件：建立因子模板 + 显式注册机制 + 输出校验器，交付两个示例因子（`momentum`、`volatility`），并实现标准化层的三种方法。为 Phase 3（回测引擎）和 Phase 4（策略层）提供可消费的因子产线。

## Interface（from DESIGN.md §2.2、§2.3）

**因子单元**（每个因子文件导出两个符号）：
```python
METADATA: dict      # name, author, version, params, min_history, direction, description
def compute(df: pd.DataFrame, params: dict | None = None) -> pd.Series
```

**因子输入**：单个标的的标准化行情 `DataFrame[date, open, high, low, close, volume]`（数据层 `query()` 输出）。

**因子输出（采纳 DESIGN.md §2.2 矛盾中的"方案 X"）**：
- 返回 `pd.Series`
- **长度等于输入 `len(df)`**
- **索引严格等于 `df["date"]`**（或 `df.index`，视 Phase 1 实际产出而定——实现者先读一次 Phase 1 产出再决定）
- 前 `min_history - 1` 行允许为 NaN
- 第 `min_history - 1` 行之后不允许 NaN
- dtype 为 `float64`

**注册接口**（`factors/registry.py` 新增函数）：
```python
load_registered_factors() -> dict[str, FactorModule]   # 名字 → 可调用模块句柄
```
从 `factors/registry.yaml` 读取因子清单并 import 对应模块，返回 `{metadata.name: module}`。

**标准化接口**（from DESIGN.md §2.3）：
```python
standardize(raw: dict[str, pd.Series], method: str = "cross_sectional_rank") -> dict[str, pd.Series]
```
入参 `raw` 的每个 Series 都应有相同索引；出参保持相同键和索引。

## Requirements

### 因子层

1. **因子模板**：创建 `factors/_template.py`，内容严格复刻 DESIGN.md §4.3，但加一行注释 `# 复制此文件后请改文件名、改 METADATA["name"]、改 compute 逻辑`。模板不在 `registry.yaml` 中注册。
2. **注册表**：创建 `factors/registry.yaml`，结构：
   ```yaml
   factors:
     - module: factors.momentum
     - module: factors.volatility
   ```
   只登记模块路径，不登记参数——参数由策略配置覆盖。
3. **注册加载器**：`factors/registry.py`，`load_registered_factors()` 按 yaml 顺序 `importlib.import_module()`，提取每个模块的 `METADATA` 和 `compute`，断言元数据必填字段（`name`, `author`, `version`, `params`, `min_history`, `direction`, `description`）完整，`direction ∈ {"higher_better", "lower_better"}`。缺字段或模块路径无效 → 抛 `RuntimeError("registry load failed: ...")`。
4. **校验器**：`factors/validator.py`，`validate(series, df, metadata)`：
   - 长度等于 `len(df)`
   - 索引 `.equals(df.index)`（若 Phase 1 的 DataFrame 以 date 为列而非 index，则校验 `series.index.equals(pd.Index(df["date"]))`）
   - dtype 为 float 兼容
   - `series.iloc[:metadata["min_history"]-1]` 允许含 NaN
   - `series.iloc[metadata["min_history"]-1:]` 不允许含 NaN
   - 违反项汇总抛 `ValueError("factor <name> validation failed: ...")`
5. **动量因子** `factors/momentum.py`：
   - `METADATA`：`name="momentum"`, `params={"window": 20}`, `min_history=20`, `direction="higher_better"`, `description="N 日收益率"`
   - `compute(df, params)`：`df["close"].pct_change(periods=p["window"])`，返回 Series，索引同 `df`
6. **波动率因子** `factors/volatility.py`：
   - `METADATA`：`name="volatility"`, `params={"window": 20}`, `min_history=21`（需要 20 日收益率再算标准差，多一天），`direction="lower_better"`, `description="N 日收益率的滚动标准差"`
   - `compute(df, params)`：`df["close"].pct_change().rolling(p["window"]).std()`
7. **沙盒约束**：两个因子文件内部不得 `import` 除 `pandas`, `numpy`, 标准库 之外的任何第三方库（通过 grep 人工检查即可，不做运行时拦截）。
8. **隔离性**：`load_registered_factors()` 调用 `compute()` 时必须传 `df.copy()`，防止因子修改原 DataFrame。

### 标准化层

9. **三种方法** `standardization/methods.py`：
   - `cross_sectional_rank(raw)`：**注意，此方法需要多标的横截面数据**，而 Phase 2 的 `raw` 字典是"单标的多因子"，横截面排名无从谈起。**因此 Phase 2 的 `cross_sectional_rank` 暂以 `NotImplementedError("cross-sectional rank needs multi-asset input; will be wired in Phase 4")` 占位**，同时在函数 docstring 说明原因并指向 Phase 4 规格卡。—— 这是本 Phase 的一个计划性缺口，刻意保留。
   - `z_score(raw, window=60)`：对每个 Series 做 60 日滚动 z-score，`(x - rolling_mean) / rolling_std`
   - `percentile(raw, window=60)`：每个 Series 做 60 日滚动百分位，`rolling.rank(pct=True)`
   - 三个函数签名统一为 `(raw: dict[str, pd.Series], **kwargs) -> dict[str, pd.Series]`
10. **方法分发** `standardization/methods.py` 暴露一个顶层 `standardize(raw, method, **kwargs)`，根据 `method` 字符串派发到具体实现，未知 method 抛 `ValueError`。

## Acceptance

**自动化测试**（`uv run pytest factors/tests/ standardization/tests/` 通过）：

1. `factors/tests/test_registry.py::test_load_both_factors`：`load_registered_factors()` 返回含 `momentum` 和 `volatility` 两个键的 dict。
2. `factors/tests/test_registry.py::test_missing_metadata_field_raises`：把一个因子的 METADATA 临时去掉 `direction`（用 monkeypatch 或 tmp 模块），加载时报错。
3. `factors/tests/test_validator.py::test_length_mismatch`：构造一个长度和输入不符的 Series，校验器抛 ValueError。
4. `factors/tests/test_validator.py::test_tail_nan_rejected`：构造一个在 `min_history` 之后仍含 NaN 的 Series，校验器抛 ValueError。
5. `factors/tests/test_validator.py::test_prefix_nan_allowed`：前 `min_history-1` 行为 NaN、之后无 NaN 的 Series 通过校验。
6. `factors/tests/test_momentum.py::test_compute_shape_and_dtype`：在一个 50 行合成 DataFrame 上跑 `compute`，返回 Series 长度 50、dtype float、前 19 行 NaN、第 20 行起无 NaN。
7. `factors/tests/test_volatility.py::test_compute_shape_and_dtype`：同上，注意 `min_history=21`。
8. `standardization/tests/test_methods.py::test_zscore_basic`：对一个线性上升的 Series 做 z-score，结果中位值接近 0。
9. `standardization/tests/test_methods.py::test_percentile_range`：百分位结果全部落在 `[0, 1]`。
10. `standardization/tests/test_methods.py::test_cross_sectional_rank_raises`：显式检查 `standardize(..., method="cross_sectional_rank")` 抛 `NotImplementedError`，消息含 `"Phase 4"`。

**手动烟雾测试**（必须跑一次）：

`uv run python -c "from data.store import query; from factors.registry import load_registered_factors; from factors.validator import validate; from datetime import date; df = query('510300.SH', date(2024,1,1), date(2024,12,31)); facs = load_registered_factors(); [validate(m['compute'](df.copy()), df, m['METADATA']) for m in facs.values()]; print('ok')"`

→ 输出 `ok`，说明两个因子都能在真实数据上算出结果并通过校验。

## Non-goals

- 不写策略层（那是 Phase 4）
- 不实现 `cross_sectional_rank` 的真实逻辑（计划性缺口，见 Requirement 9）
- 不支持多标的横截面因子（Phase 2 的因子只消费单标的 DataFrame）
- 不做因子性能回测（Phase 3）
- 不写 `docs/FACTOR_GUIDE.md`（如有需要 Phase 2 末尾再加，不是验收项）
- 不做因子缓存、持久化因子值到磁盘
- 不实现因子间依赖（因子只依赖行情，不依赖其他因子）
- 不做第三方依赖声明的运行时检查

## 收尾动作

1. 更新项目根 `CLAUDE.md` Status 表：Phase 2 = `done`
2. **回写 DESIGN.md**：修正 §2.2 "输出校验" 第 1、3 两条的自相矛盾：
   - 删除 `Series 长度 = 输入 DataFrame 行数 - (min_history - 1)`
   - 改为 `Series 长度 = 输入 DataFrame 行数`
   - 保留"索引与输入 DataFrame 的日期索引对齐"
   - 保留"允许前 min_history-1 行为 NaN，之后不允许"（并把"min_history"改为"min_history-1"以保持一致）
   - 这个改动应作为 Phase 2 最后一个 commit 提交
3. 在 CLAUDE.md 的 `Deviations` 节记录：`standardization.cross_sectional_rank` 在 Phase 2 是 NotImplementedError 占位，等到 Phase 4 接入多标的输入后实现
4. 提交 git commit
