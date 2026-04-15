# QuantFramework

可扩展的量化策略研究与执行框架。六层架构，回测与实盘共享同一套因子代码，架构层面强制隔离未来信息。

## 架构

```
数据层 → 因子层 → 标准化层 → 策略层 → 执行层 → 通知层
                                ↑
                            回测引擎（横切所有层的运行模式）
```

| 层 | 职责 | 关键文件 |
|----|------|----------|
| 数据层 | Tushare 同步 + Parquet 本地存储 | `data/sync.py`, `data/store.py` |
| 因子层 | 接收行情，输出原始因子值 | `factors/momentum.py`, `factors/volatility.py` |
| 标准化层 | 因子值映射到可比较空间 | `standardization/methods.py` |
| 策略层 | 消费因子输出，生成目标持仓权重 | `strategy/momentum_rotation.py` |
| 执行层 | 对比目标与当前持仓，生成调仓指令 | `execution/interfaces.py` |
| 通知层 | 调仓指令推送到钉钉 | `notification/dingtalk.py` |

## 快速开始

### 1. 环境准备

需要 Python 3.12 和 [uv](https://docs.astral.sh/uv/)。

```bash
git clone <repo-url> && cd QuantFramework
uv sync
```

### 2. 配置密钥

复制 `.env.example` 为 `.env`，填入真实值：

```bash
cp .env.example .env
```

```dotenv
# .env
TUSHARE_TOKEN=你的tushare_token          # 必填，https://tushare.pro 注册获取
DINGTALK_WEBHOOK=https://oapi.dingtalk.com/robot/send?access_token=xxx  # 实盘通知用
DINGTALK_SECRET=SEC...                    # 钉钉机器人加签密钥（可选）
```

### 3. 同步数据

首次运行会拉取 2016 年至今的全量历史数据，后续只增量同步：

```bash
# 按策略配置批量同步（推荐，从 asset_pool 读取标的列表）
uv run python -m data --config strategy/configs/momentum_rotation.yaml

# 或同步单个标的
uv run python -m data 510300.SH
```

数据存储在 `data/db/` 下，每个标的一个 Parquet 文件。

> **注意：** `run_daily.py` 运行时会自动同步数据并检查新鲜度，无需手动执行此步骤。首次使用时仍需手动同步以拉取全量历史。

### 4. 运行回测

```bash
# 使用默认配置回测
uv run python run_backtest.py

# 使用自定义配置
uv run python run_backtest.py --config strategy/configs/momentum_rotation.yaml

# 从实验日志复现
uv run python run_backtest.py --from-log experiments/20260413-001.yaml
```

输出示例：

```
Running backtest...

Backtest complete: 2400 trading days
Train/test split at: 2023-01-15
  Train: return=85.20%  sharpe=1.20  max_dd=-18.30%
  Test : return=32.10%  sharpe=0.80  max_dd=-22.50%
  Bench: return=45.00%  sharpe=0.60  max_dd=-30.10%

Experiment log: experiments/20260413-001.yaml
HTML report: experiments/20260413-001.html
```

### 5. 每日实盘运行

```bash
uv run python run_daily.py --config strategy/configs/momentum_rotation.yaml
```

执行流程：计算因子 → 策略生成目标权重 → 对比当前持仓 → 钉钉推送调仓信号 → 保存新持仓。

当前持仓保存在 `state/current_position.json`。

## 完整范例：从零到第一次回测

以下演示从安装到拿到回测报告的完整流程。

**第一步：安装并配置**

```bash
cd QuantFramework
uv sync
cp .env.example .env
# 编辑 .env，填入 TUSHARE_TOKEN
```

**第二步：同步 4 只 ETF 的历史数据**

```bash
uv run python -m data --config strategy/configs/momentum_rotation.yaml
```

**第三步：编写策略配置**

创建 `strategy/configs/my_strategy.yaml`：

```yaml
strategy_name: momentum_rotation
strategy_class: strategy.momentum_rotation.MomentumRotation
asset_pool:
  - 510300.SH   # 沪深300 ETF
  - 159915.SZ   # 创业板 ETF
  - 513100.SH   # 纳斯达克 ETF
  - 518880.SH   # 黄金 ETF
start: "2018-01-01"
end: "2026-04-13"
factors:
  - name: momentum
    weight: 0.7
    params: {window: 20}
  - name: volatility
    weight: 0.3
    direction_flip: true    # 低波动更好，翻转排序方向
    params: {window: 20}
train_ratio: 0.7
rebalance_rule: daily
```

配置说明：

| 字段 | 含义 |
|------|------|
| `strategy_class` | 策略类的完整路径，省略则默认 MomentumRotation |
| `asset_pool` | 标的池，Tushare 代码 |
| `start` / `end` | 回测时间范围 |
| `factors` | 使用的因子列表，`weight` 为组合权重，`direction_flip` 翻转排序 |
| `train_ratio` | 训练集占比，用于过拟合检测 |

**第四步：运行回测**

```bash
uv run python run_backtest.py --config strategy/configs/my_strategy.yaml
```

结果自动保存到 `experiments/` 目录：
- `*.yaml` — 完整实验日志（参数、指标快照，可一键复现）
- `*.html` — quantstats 可视化报告

**第五步：配置每日实盘推送**

```bash
# 编辑 .env，添加钉钉配置
# DINGTALK_WEBHOOK=https://oapi.dingtalk.com/robot/send?access_token=xxx
# DINGTALK_SECRET=SECxxx

uv run python run_daily.py --config strategy/configs/my_strategy.yaml
```

钉钉收到的消息格式：

```
2026-04-13 调仓信号

买入: 2 | 卖出: 1 | 持有: 1

| 标的       | 操作 | 变动    | 目标权重 |
|-----------|------|---------|---------|
| 159915.SZ | 买入 | +15.00% | 30.00%  |
| 510300.SH | 持有 | +0.00%  | 35.00%  |
| 513100.SH | 买入 | +10.00% | 25.00%  |
| 518880.SH | 卖出 | -25.00% | 10.00%  |
```

## 新增因子

新增一个因子只需要改动两个文件：

**1. 复制模板并实现计算逻辑**

```bash
cp factors/_template.py factors/my_factor.py
```

编辑 `factors/my_factor.py`：

```python
"""我的因子 — 简短描述。"""

import pandas as pd

METADATA = {
    "name": "my_factor",
    "author": "your_name",
    "version": "1.0.0",
    "params": {"window": 10},
    "min_history": 11,
    "direction": "higher_better",  # 或 "lower_better"
    "description": "一句话描述",
}

def compute(df: pd.DataFrame, params: dict | None = None) -> pd.Series:
    p = {**METADATA["params"], **(params or {})}
    series = df["close"].rolling(p["window"]).mean() / df["close"] - 1
    series.index = df["date"]
    return series
```

**2. 注册因子**

在 `factors/registry.yaml` 添加一行：

```yaml
factors:
  - module: factors.momentum
  - module: factors.volatility
  - module: factors.my_factor      # ← 新增
```

**3. 验证**

```bash
uv run pytest factors/tests/
```

然后在策略配置的 `factors` 列表中引用 `my_factor` 即可。

## 新增策略

继承 `BaseStrategy`，实现 `generate_weights` 方法：

```python
# strategy/equal_weight.py
from strategy.base import BaseStrategy

class EqualWeight(BaseStrategy):
    def generate_weights(
        self, factor_values: dict[str, dict[str, float]]
    ) -> dict[str, float]:
        if not factor_values:
            return {}
        n = len(factor_values)
        return {asset: 1.0 / n for asset in factor_values}
```

在配置中指定 `strategy_class`：

```yaml
strategy_class: strategy.equal_weight.EqualWeight
```

## 测试

```bash
uv run pytest              # 运行全部测试
uv run pytest factors/     # 只测因子层
uv run pytest -v           # 详细输出
```

## 目录结构

```
QuantFramework/
├── data/                    # 数据层
│   ├── sync.py              #   Tushare 增量同步
│   ├── store.py             #   Parquet 读写 + 查询接口
│   ├── config.py            #   Token 配置
│   └── db/                  #   Parquet 数据文件
├── factors/                 # 因子层
│   ├── _template.py         #   因子模板
│   ├── momentum.py          #   动量因子（20日收益率）
│   ├── volatility.py        #   波动率因子（20日滚动标准差）
│   ├── registry.yaml        #   因子注册表
│   └── validator.py         #   因子输出校验器
├── standardization/         # 标准化层
│   └── methods.py           #   z-score, percentile
├── strategy/                # 策略层
│   ├── base.py              #   策略基类 (BaseStrategy)
│   ├── momentum_rotation.py #   动量轮动策略
│   ├── loader.py            #   策略动态加载器
│   └── configs/             #   策略配置 YAML
├── execution/               # 执行层
│   ├── interfaces.py        #   Order + diff()
│   └── position.py          #   当前持仓读写
├── notification/            # 通知层
│   ├── interfaces.py        #   Notifier 基类
│   ├── dingtalk.py          #   钉钉适配器
│   └── formatter.py         #   调仓信号格式化
├── backtest/                # 回测引擎
│   ├── runner.py            #   时间序列遍历 + 未来信息截断
│   ├── report.py            #   quantstats HTML 报告
│   └── experiment_log.py    #   实验日志
├── experiments/             # 实验日志存储
├── state/                   # 运行状态（当前持仓）
├── run_backtest.py          # 回测入口
├── run_daily.py             # 每日实盘入口
└── docs/DESIGN.md           # 架构设计文档
```
