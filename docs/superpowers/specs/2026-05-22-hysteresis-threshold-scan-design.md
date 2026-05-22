# Hysteresis 阈值扫描 - 研究设计

## 背景

本轮研究评估 `quality_momentum_top1` 的 Top1 切换阈值。当前策略在挑战者 score 反超在任资产时立即切换；研究版本允许在任资产获得 `tau` 的 score 边际，只有挑战者 score 严格超过 `incumbent_score + tau` 才切换。

这是研究回测，不是部署。任何 `tau > 0` 的实盘落地都锁在 2026-06-02 之后的独立 changelog 决策中。本轮不修改生产配置 `strategy/configs/quality_momentum_top1.yaml`，产出是证据附件与研究代码。

## 已确认决策

| 决策 | 结论 | 理由 |
|------|------|------|
| incumbent 来源 | 回测 runner 显式传递真实 `current_weights` | hysteresis 依赖真实执行轨迹，不能让策略沿原始每日 signal 自持状态 |
| 契约扩展 | `generate_weights(..., current_weights=None)` 向后兼容扩展 | 只让需要持仓上下文的研究策略读取上下文，现有策略调用仍可工作 |
| 研究范围 | 只接 `backtest.runner` | 先验证 1D `tau` 曲线是否值得继续，避免提前铺 live/backfill |
| 排除范围 | 不改 `run_daily.py`、`backfill_ytd.py`、生产 YAML | 本轮不是部署，避免研究行为进入实盘路径 |
| hold floor | 被阈值拦下后不重置 `rebalance_days` | `rebalance_days` 只约束新仓最短持有；runner 现有逐日重评语义已匹配 |
| 阈值量纲 | `tau` 与 `quality_momentum` score 同尺度 | score 为 20 日简单收益乘 ER，ER 在 `[0, 1]` |

## 架构

### Strategy 契约

`BaseStrategy.generate_weights` 新增可选 `current_weights` 参数：

```python
def generate_weights(
    self,
    factor_values: dict[str, dict[str, float]],
    current_weights: dict[str, float] | None = None,
) -> dict[str, float]:
    ...
```

`MomentumRotation`、`TopN` 与已有调用者保持兼容；现有策略无需读取该参数。`backtest.runner` 在策略求 signal 时传入当前实际持仓。

### Top1 hysteresis

`Top1` 读取研究配置中的 `hysteresis_threshold`，默认 `0.0`。

- `tau == 0` 时维持现有 Top1 排名行为。
- 没有 incumbent 时维持现有 Top1 排名行为。
- incumbent 不是当日 scored 资产时维持现有 Top1 排名行为，避免在 score 缺失时凭旧仓硬持有。
- 只有单资产 Top1 incumbent 才参与阈值比较；挑战者必须严格超过 `incumbent_score + tau`。
- 若挑战者没有清过阈值，`Top1` 返回 incumbent 权重；runner 视为没有不同 target，保持仓位并在 hold floor 过后继续逐日重评。

`direction_flip` 仍保持现有语义。研究目标是 `quality_momentum` 的 higher-better score；若为 lower-better 因子，则镜像为挑战者 score 必须严格低于 `incumbent_score - tau`，并用单测锁住该行为。

## 证据链与附件

先建立 `strategy_changelog_attachments/`，补存 2026-05-21 归因诊断的证据链，再写本轮 hysteresis 扫描附件。

仓库当前未发现 2026-05-21 诊断脚本、原始 Markdown 附件或 raw CSV。补档按以下规则处理：

- 若后续在 workspace 中找到原始产物，保留其原始内容与来源说明。
- 若只能从现有数据与同口径逻辑重跑，则附件明确标注为“重建产物”，不把重建 CSV 冒充为 2026-05-21 原始落盘。
- hysteresis 扫描可复用该诊断的口径，但必须先通过 `tau=0` 与同口径 `rebalance_days=5` baseline 的内部一致性校验。

## 研究扫描

### 固定口径

- 起始日期：`2014-01-01`，剔除资产池不完整期。
- 成本：单边 `0.01%`，按实际买卖成交权重扣减。
- `rebalance_days=5` 固定，不与阈值联调。
- `tau` 扫描：`0`, `0.0005`, `0.001`, `0.0025`, `0.005`, `0.0075`, `0.01`。
- 每个 `tau` 运行独立完整回测，不从静态 score gap 分布外推交易轨迹。

### Baseline gate

先运行 plain Top1 `rebalance_days=5` 基准，再运行 `tau=0` hysteresis 版本。两者在相同回测区间、成本口径与返回计算下必须精确一致。若未对齐，停止扫描并修正实现。

### 输出指标

标准面板对每个 `tau` 输出：

- 年化收益
- Sharpe
- 最大回撤
- 年化换手率
- 平均持有期
- 切换次数

trade-off 诊断输出：

- whipsaw 次数与累计 whipsaw P&L。
- episode 层回撤拆分，重点区分多切换回撤段 `2015-10`、`2020-09`、`2025-10` 与零切换单资产崩盘段 `2024-10`。
- 被阈值拦下或推迟但事后有利的切换损失。
- 被拦切换中事后正确与错误的数量。
- `2024-09-26` 切进 `159915.SZ` 金丝雀：逐 `tau` 记录是否被拦、延迟天数与少赚收益。
- 若干 regime 下的 score 量级样本，验证扫描阈值相对 score 的大小。

本轮结论写 trade-off 与曲面观察，不直接给出部署决策。解释优先关注阈值是否减少换手型回撤，以及是否开始黏住输家、错过趋势。

## 测试与验证

### 回归测试

- `Top1` 测试默认 `tau=0` 与现有行为一致。
- `Top1` 测试 incumbent 保持、挑战者越阈值切换、incumbent score 缺失 fallback。
- runner 测试策略能收到执行中的 `current_weights`，且阈值拦截后在 hold floor 过后继续逐日重评。
- 现有 strategy 与 runner 测试保持通过。

### 研究验证

- `tau=0` baseline gate 通过后才扫完整参数集。
- 成本扣减、换手率、whipsaw、episode 与 delayed-entry 指标由研究诊断脚本产出 raw CSV 和 Markdown 摘要。
- 最终附件按实际跑日命名为 `strategy_changelog_attachments/YYYY-MM-DD_hysteresis_scan.md`。

## 范围之外

- 不部署 `tau > 0`。
- 不修改 live daily 信号路径。
- 不重构 `backfill_ytd.py` 的 signal/replay 边界。
- 不做 `rebalance_days` 与 `tau` 的二维联合扫描。
- 不修改 2026-06-02 changelog 决策条目；附件只提供后续决策依据。
