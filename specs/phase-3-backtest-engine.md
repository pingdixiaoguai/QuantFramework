# Phase 3 — 回测引擎

Module: `backtest/`  |  DESIGN.md §三

> 规格卡是**一次性任务输入**。实现完成后由代码和测试承担文档职责，本文件不再维护。

---

## 目标

实现回测引擎核心：逐日遍历交易日、未来信息截断、因子计算、等权组合基准、train/test 分割、实验日志、quantstats 报告集成。策略层（Phase 4）尚未实现，因此本阶段的"策略"暂用一个内置的简单排名策略（基于因子加权排名选择持仓权重），为 Phase 4 提供可替换的策略接口。

## Interface

**回测入口**：
```python
backtest.runner.run(config: dict) -> BacktestResult
```

`config` 结构（对应 DESIGN.md §3.3 实验日志中的参数）：
```python
{
    "strategy_name": str,              # 策略名称标识
    "asset_pool": list[str],           # 标的池
    "start": date,                     # 回测开始日期
    "end": date,                       # 回测结束日期
    "factors": [                       # 因子配置
        {"name": "momentum", "weight": 0.7, "params": {"window": 20}},
        {"name": "volatility", "weight": 0.3, "direction_flip": True, "params": {"window": 20}},
    ],
    "train_ratio": float,              # 默认 0.7
    "rebalance_rule": str,             # "daily"（本阶段只支持 daily）
}
```

**BacktestResult**：
```python
@dataclass
class BacktestResult:
    daily_returns: pd.Series           # 策略日收益率（全期）
    benchmark_returns: pd.Series       # 等权基准日收益率
    positions: pd.DataFrame            # 每日持仓权重 (date × asset)
    train_end: date                    # 训练集截止日期
    config: dict                       # 原始配置快照
```

**实验日志**：
```python
backtest.experiment_log.save(result: BacktestResult, output_dir: Path) -> Path
```
写入 `experiments/{id}.yaml`，返回文件路径。ID 格式 `YYYYMMDD-NNN`。

**报告**：
```python
backtest.report.generate(result: BacktestResult, output_path: Path) -> Path
```
调用 quantstats 生成 HTML 报告。

## Requirements

### 回测引擎核心（`backtest/runner.py`）

1. **逐日遍历**：获取回测区间内所有交易日（取各标的数据的日期并集），按日期升序遍历。
2. **未来信息截断**：在每个交易日 t，对每个标的只传入 `data[data["date"] <= t]` 给因子。
3. **因子计算**：对每个标的、每个因子调用 `compute(truncated_df.copy(), params)`，通过 validator 校验。校验失败的因子在该交易日跳过（打印警告），不中断回测。
4. **因子加权排名**：对每个交易日的因子值做简单加权排名：
   - 对每个因子，在资产间排名（rank），`direction_flip=True` 的因子翻转排名
   - 各因子排名按权重加权求和
   - 归一化为权重（合计 1.0）
5. **收益率计算**：持仓权重在 t 日收盘确定，收益在 t+1 实现。`strategy_return[t+1] = sum(weight[asset] * asset_return[t+1])`。资产日收益率 = `close[t+1] / close[t] - 1`。
6. **等权基准**：同一标的池等权买入持有，每日再平衡。`benchmark_return[t] = mean(asset_return[t])`。
7. **train/test 分割**：按 `train_ratio` 将交易日序列分为训练集和测试集。`train_end = trading_days[int(len * train_ratio)]`。分割点记录在 `BacktestResult` 中。
8. **过拟合警告**：若训练集 Sharpe > 2× 测试集 Sharpe，打印警告。

### 实验日志（`backtest/experiment_log.py`）

9. **日志格式**：YAML，结构严格按 DESIGN.md §3.3。
10. **日志 ID**：`YYYYMMDD-NNN`，NNN 为当天序号（从 `experiments/` 目录中已有文件推断）。
11. **绩效指标**：每个子集（train/test/全期）记录 `total_return`, `sharpe`, `max_drawdown`。
12. **基准指标**：同样记录 `total_return`, `sharpe`, `max_drawdown`。
13. **目录**：`experiments/` 不存在则自动创建。

### 报告（`backtest/report.py`）

14. **quantstats 集成**：`generate()` 调用 `quantstats.reports.html(returns, benchmark, output=path)`。
15. **输出路径**：默认 `experiments/{experiment_id}_report.html`。

### 入口（`run_backtest.py`）

16. **命令行**：`uv run python run_backtest.py [--config path/to/config.yaml] [--from-log path/to/experiment.yaml]`。
17. **默认配置**：不传 `--config` 时使用内置默认配置（4 只 ETF、momentum+volatility、2016-01-01 至今、train_ratio=0.7）。
18. **from-log 复现**：`--from-log` 从已有实验日志中读取配置并重新运行。

## Acceptance

**自动化测试**（`uv run pytest backtest/tests/`）：

1. `test_runner.py::test_future_info_truncation`：构造合成数据，验证因子在日期 t 只接收到 <= t 的数据。用一个假因子，返回 `len(df)` 作为 Series 值，验证值逐日递增。
2. `test_runner.py::test_returns_calculation`：用 2 个标的、已知价格序列（如线性增长），验证收益率计算正确。
3. `test_runner.py::test_benchmark_equal_weight`：验证基准收益率 = 各资产收益率的等权平均。
4. `test_runner.py::test_train_test_split`：验证 `train_end` 日期在正确位置。
5. `test_experiment_log.py::test_save_and_load`：保存一个 BacktestResult，读回 YAML，检查关键字段完整。
6. `test_experiment_log.py::test_id_auto_increment`：同一天保存两次，ID 序号递增。

**手动烟雾测试**（必须跑一次）：

`uv run python run_backtest.py` → 输出实验 ID、训练集/测试集绩效指标，`experiments/` 下生成 YAML 日志文件。（quantstats HTML 报告如果环境允许则一并生成，否则 graceful skip。）

## Non-goals

- 不实现策略层的通用策略接口（Phase 4）——本阶段用内置的因子加权排名逻辑
- 不支持非 daily 再平衡
- 不支持交易成本 / 滑点
- 不支持做空
- 不支持现金仓位（始终满仓）
- 不做跨资产池批量回测（单次只跑一个配置）
- 不做可视化（quantstats HTML 报告承担）
- 不做增量回测（每次全量重新计算）

## 收尾动作

1. 更新 CLAUDE.md Status 表：Phase 3 = `done`
2. 记录偏差到 CLAUDE.md Deviations（如有）
3. 提交 git commit
