# Phase 4 — Strategy Layer Design

## 概述

实现可插拔的策略层，消费因子值，生成目标持仓权重。替换回测引擎中内联的 `_compute_weights()` 临时逻辑。

## 设计决策

| 决策 | 结论 | 理由 |
|------|------|------|
| 策略接口风格 | 可插拔基类（BaseStrategy + 子类） | 支持不同策略逻辑，新增策略 = 新文件 + 新配置 |
| 输入数据形状 | 截面快照 `dict[str, dict[str, float]]` | 当前只需每日最新因子值，和回测引擎已有数据流对齐 |
| 截面排序归属 | 策略层内部完成 | 排序是策略逻辑的一部分，不同策略可能用不同排序方式 |
| 再平衡频率 | 仅 daily | YAGNI，后续按需扩展 |
| 向后兼容 | config 无 strategy.class 时默认 MomentumRotation | 不破坏现有回测入口和实验日志复现 |

## 组件

### 1. BaseStrategy（strategy/base.py）

策略抽象基类：

```python
class BaseStrategy(ABC):
    def __init__(self, config: dict):
        self.config = config

    @abstractmethod
    def generate_weights(
        self, factor_values: dict[str, dict[str, float]]
    ) -> dict[str, float]:
        """
        输入: asset_code → factor_name → 当日因子值（截面快照）
        输出: asset_code → 权重（合计 = 1.0）
        """
        ...
```

约束：
- 输出权重必须合计为 1.0
- 输入为空时返回空 dict
- 不得访问未来信息（引擎保证截断，策略无需关心）

### 2. MomentumRotation（strategy/momentum_rotation.py）

第一个策略实现，逻辑来自 `backtest/runner.py` 的 `_compute_weights()`：

1. 对每个因子，将资产按因子值截面排名（1-based）
2. 根据 `direction_flip` 翻转排名方向
3. 按因子权重加权汇总得分
4. 归一化为权重（合计 1.0）

边界情况：
- 只有 1 个资产时，该资产权重 = 1.0
- 输入为空时，返回空 dict

### 3. 策略加载器（strategy/loader.py）

动态加载策略类：

```python
def load_strategy(config: dict) -> BaseStrategy:
    class_path = config.get("strategy_class",
        "strategy.momentum_rotation.MomentumRotation")
    module_path, class_name = class_path.rsplit(".", 1)
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)
    return cls(config)
```

### 4. 策略配置（strategy/configs/momentum_rotation.yaml）

```yaml
strategy_name: momentum_rotation
strategy_class: strategy.momentum_rotation.MomentumRotation
asset_pool:
  - 510300.SH
  - 159915.SZ
  - 513100.SH
  - 518880.SH
start: "2016-01-01"
end: "2026-04-13"
factors:
  - name: momentum
    weight: 0.7
    params: {window: 20}
  - name: volatility
    weight: 0.3
    direction_flip: true
    params: {window: 20}
train_ratio: 0.7
rebalance_rule: daily
```

配置与现有 backtest config 结构兼容，新增 `strategy_class` 字段。

### 5. 回测引擎改造（backtest/runner.py）

改动点：
1. `run()` 开头调用 `load_strategy(config)` 获取策略实例
2. 日循环中 `_compute_weights(...)` → `strategy.generate_weights(...)`
3. 删除 `_compute_weights()` 函数
4. 无 `strategy_class` 字段时默认使用 MomentumRotation（向后兼容）

### 6. run_backtest.py 更新

支持 `--config` 加载包含 `strategy_class` 的 YAML 配置文件。无需其他改动。

## 测试计划

- `strategy/tests/test_base.py` — 验证基类约束（权重合计 1.0、空输入）
- `strategy/tests/test_momentum_rotation.py` — 搬运并扩展 backtest 中排名逻辑的测试
  - 多资产正常排名
  - direction_flip 翻转
  - 单资产边界情况
  - 空输入边界情况
- `strategy/tests/test_loader.py` — 验证动态加载和默认回退
- `backtest/tests/test_runner.py` — 确认现有回测测试仍然通过（行为不变）

## Non-goals

- 不实现 weekly/monthly 再平衡频率
- 不改动标准化层接口
- 不实现 `cross_sectional_rank` 标准化方法
- 不实现执行层和通知层（Phase 5、6）
- 不实现 run_daily.py 的策略集成（依赖执行层）
