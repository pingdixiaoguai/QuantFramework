# Risk-Adjusted Quality Momentum — 因子设计

## 背景

现有 `factors/quality_momentum.py` 用 `(close_t / close_{t-N}) - 1` 作为动量项，乘以 Kaufman 效率比率 (ER)。在跨资产轮动池（同时含股票 ETF、债券、黄金、红利低波等异质资产）中，原始涨幅天然偏向高波动资产；牛市偏向高 beta，熊市/震荡市低估趋势稳健的低波资产。

新增一个**独立的**因子 `risk_adjusted_quality_momentum`，把动量项升级为风险调整动量（除以 N 期波动率），其余维持质量动量的核心思想。

不替换 `quality_momentum`：保留旧因子用于跨期 A/B 对比，新旧策略可并行回测。

## 设计决策

| 决策 | 结论 | 理由 |
|------|------|------|
| 文件归属 | 新增 `factors/risk_adjusted_quality_momentum.py` | 与 `quality_momentum` 并存，便于 A/B 对比；符合"PR scope: 1 factor file + 1 registry line" |
| 默认窗口 | `window=60` | 跨资产轮动 60 日是常见信号-噪声平衡点（GPT 推荐） |
| 收益形式 | 全程使用对数收益 | R、path、vol 同空间，数学上自洽；和 GPT 给出的公式一致 |
| 波动率地板 | `vol_floor_annual=0.08`，按 √(N/252) 折算到 N 期 | 防止超低波资产（短债）因分母过小被无限抬高 |
| 负收益处理 | 不分支，直接 `Score = (R/AdjVol) × ER` | 保持因子表达式无条件分支；防御逻辑由策略层池子组成承担（已含 bond/gold） |
| 极端值截断 | `clip(R/AdjVol, -3, +3)` | 防止低波或样本异常被无限放大 |
| 是否做波动率平滑（Vol_60 + Vol_120） | 暂不做 | YAGNI；后续可作为 `params["vol_blend"]` 加入 |

## 公式

```text
r_t        = ln(close_t / close_{t-1})           # 日对数收益
R_N        = ln(close_t / close_{t-N})           # N 期对数收益
path_N     = sum(|r_t|) over last N              # N 期路径长度
ER_N       = |R_N| / path_N                      # 效率比率（path=0 时取 NaN）
vol_N      = std(r_t, N) * sqrt(N)               # N 期对数波动率（ddof=1）
floor_N    = vol_floor_annual * sqrt(N/252)
adj_vol_N  = max(vol_N, floor_N)
ram        = clip(R_N / adj_vol_N, -3, +3)       # 风险调整动量，截断
score      = ram * ER_N
```

## 接口契约

遵守 `factors/CLAUDE.md` 既定契约：

- 输入：`pd.DataFrame(date, open, high, low, close, volume)`，单资产
- 输出：`pd.Series`，索引为 `df["date"]`，`float` 类型，长度等于输入
- 不修改输入 df
- 第 `min_history - 1 = 60` 行起为有限值

```python
METADATA = {
    "name": "risk_adjusted_quality_momentum",
    "author": "quantframework",
    "version": "1.0.0",
    "params": {"window": 60, "vol_floor_annual": 0.08},
    "min_history": 61,
    "direction": "higher_better",
    "description": "风险调整动量 × Kaufman 效率比率（对数收益、波动率地板、winsorize），跨资产可比",
}
```

参数约束：调用方可覆盖 `window` 和 `vol_floor_annual`，但 METADATA 默认值不变。

## 测试矩阵

测试文件：`factors/tests/test_risk_adjusted_quality_momentum.py`，沿用 `test_quality_momentum.py` 的 `_make_df` helper。

| 测试类 | 用例 | 期望 |
|--------|------|------|
| `TestOutputShape` | 长度等于输入 | `len(result) == len(df)` |
|  | 前 60 行为 NaN | `result.iloc[:60].isna().all()` |
|  | 60 行起有限 | `result.iloc[60:].notna().all() and isfinite(...)` |
| `TestEfficiencyRatio` | 直线行情 ER ≈ 1 | `score ≈ R/adj_vol`（在容差内） |
|  | 同净涨幅下，锯齿 < 直线 | `score_straight > score_zigzag` |
| `TestVolFloor` | 超低波资产触发地板 | 构造一条波动率远小于 floor 的合成序列，断言 `score` 等于按 floor 计算的值（手算对照） |
| `TestWinsorize` | `R/AdjVol` 超 +3 时被截断 | 构造合成序列让 raw `R/vol` 远大于 3，断言 `score / ER == 3`（在浮点容差内） |
| `TestCrossAssetComparability` | 高波大涨幅 vs 低波小涨幅且路径更平滑 | 低波资产 score 更高（GPT 示例方向） |
| `TestMetadata` | 必填字段存在 | `name, params, min_history, direction` 在 METADATA |
|  | direction 正确 | `direction == "higher_better"` |

## 注册

在 `factors/registry.yaml` 末尾追加：

```yaml
  - module: factors.risk_adjusted_quality_momentum
```

## 范围之外（不在本次实现）

- **波动率平滑**（Vol_60 与 Vol_120 的 max/blend）：GPT §6 提到的可选增强，本期不做。
- **策略 YAML 切换**：`quality_momentum_top1.yaml`、`industry_quality_momentum_top5.yaml` 暂不切换到新因子。
- **回测对比**：新旧因子在同一池子上的对比回测属于后续动作，不在本次 PR。
- **历史 z-score 主因子**：GPT §5 明确不推荐作为主版本，不实现。

## 验收标准

- `uv run pytest factors/tests/test_risk_adjusted_quality_momentum.py` 全部通过
- `uv run pytest factors/tests/` 全套通过（包含 registry 加载新因子的测试）
- 新因子可从 `factors.registry.load_factors()` 中按名拿到
- `factors/CLAUDE.md` 中 "Currently registered" 一行更新为包含新因子
