# 量化策略框架系统设计文档

> 本文档是系统架构的执行依据，不涉及具体代码实现。
> 所有模块的边界定义、接口契约、依赖决策均以本文档为准。
> 
> 设计哲学来源：《Vibe Coding 真解》中的核心原则——机制与策略分离、接口先于实现、复杂性预算。

---

## 一、系统目标

构建一个可扩展的量化策略研究与执行框架，核心设计目标：

1. **让"新增一个因子"的改动范围收敛到一个文件**
2. **回测与实盘共享同一套因子代码**
3. **架构层面强制隔离未来信息，防止回测中的前视偏差**
4. **支持多人协作：贡献者只需理解因子接口，无需了解系统全貌**

---

## 二、六层架构

系统分为六个边界清晰的层，每层只依赖上一层的输出接口，禁止跨层调用。

### 2.1 数据层（Data Layer）

**职责**：从 Tushare 同步原始行情数据到本地存储，提供统一的查询接口。

**关键约束**：
- 这一层不知道"因子"是什么
- 只回答"给我某个标的从 X 到 Y 的日线数据"

**接口契约**：

| 方向 | 格式 | 说明 |
|------|------|------|
| 输入 | `(asset_code: str, start: date, end: date)` | 资产代码 + 时间范围 |
| 输出 | `pd.DataFrame` | 标准化列名：`date, open, high, low, close, volume` |

**设计决策**：
- 本地存储采用轻量方案（SQLite 或 Parquet），不引入数据库服务
- 增量同步：首次全量拉取，后续只补充新数据
- 标的池不硬编码在数据层——哪些标的需要同步由上层配置决定

### 2.2 因子层（Factor Layer）

**职责**：接收标准化行情数据，输出原始因子值。每个因子是一个独立模块。

**接口契约**：

| 方向 | 格式 | 说明 |
|------|------|------|
| 输入 | `pd.DataFrame`（数据层输出格式） | 单个标的的历史行情 |
| 输出 | `pd.Series`（日期索引，float 值） | 原始因子值，量纲由因子自身决定 |

**因子元数据结构**（每个因子必须声明）：

```
name:           因子名称（唯一标识）
author:         作者
version:        版本号
params:         可调参数及默认值（如 {"window": 20}）
min_history:    最小历史窗口需求（如 20 个交易日）
direction:      因子方向（"higher_better" / "lower_better"）
description:    一句话描述
```

**注册机制**：
- 因子文件放在约定目录 `factors/` 下
- 必须在 `factors/registry.yaml` 中显式注册才会被系统加载
- 未注册的因子文件不会被执行

**输出校验**（由系统自动执行，因子作者无需关心）：
- Series 长度 = 输入 DataFrame 行数 - (min_history - 1)
- 索引与输入 DataFrame 的日期索引对齐
- NaN 处理：允许前 min_history 行为 NaN，之后不允许
- 校验失败 → 拒绝加载，报告具体违反项

**隔离性保证**：
- 因子接收的 DataFrame 是只读副本（`df.copy()`）
- 因子内部不得访问全局状态或其他因子的输出
- 因子不得引入系统依赖清单之外的第三方库（需在元数据中声明额外依赖）

### 2.3 标准化层（Standardization Layer）

**职责**：将不同因子的原始值映射到同一可比较空间，使策略层可以直接组合。

**为什么独立成层**：
- 标准化方法本身是可实验、可替换的（z-score、横截面排名、百分位）
- 如果写死在因子内部，换一种标准化方法要改所有因子文件
- 独立后：因子只管输出原始值，标准化层负责统一量纲

**接口契约**：

| 方向 | 格式 | 说明 |
|------|------|------|
| 输入 | `dict[str, pd.Series]`（因子名 → 原始值） | 多个因子在同一标的池上的原始输出 |
| 输出 | `dict[str, pd.Series]`（因子名 → 标准化值） | 所有因子映射到可比较空间（如 z-score） |

**内置方法**：
- `cross_sectional_rank`：同一时间点，各标的在该因子上的排名（除以标的数，映射到 0-1）
- `z_score`：滚动窗口 z-score
- `percentile`：滚动窗口百分位

**设计决策**：默认使用 `cross_sectional_rank`，因为它对异常值最鲁棒，且天然在 0-1 范围内。

### 2.4 策略层（Strategy Layer）

**职责**：消费标准化因子输出，生成目标持仓。这是系统中唯一包含"策略逻辑"的层。

**接口契约**：

| 方向 | 格式 | 说明 |
|------|------|------|
| 输入 | `dict[str, pd.Series]`（标准化因子值） | 标准化层的输出 |
| 输出 | `dict[str, float]`（资产代码 → 权重） | 目标持仓权重，合计 = 1.0 |

**策略配置（与策略代码分离）**：

```yaml
strategy:
  name: momentum_rotation
  factors:
    - name: momentum_20d
      weight: 0.7
    - name: volatility_20d
      weight: 0.3
      direction_flip: true   # 低波动更好，翻转方向
  asset_pool:
    - 510300.SH   # 沪深300
    - 159915.SZ   # 创业板
    - 513100.SH   # 纳斯达克
    - 518880.SH   # 黄金
  rebalance_rule: daily
```

**关键设计**：因子权重、标的池、再平衡频率都是配置，不是代码。改配置不需要改任何代码文件。

### 2.5 执行层（Execution Layer）

**职责**：对比目标持仓与当前持仓，生成持仓变动指令。

**接口契约**：

| 方向 | 格式 | 说明 |
|------|------|------|
| 输入 | `(target: dict, current: dict)` | 目标权重 + 当前权重 |
| 输出 | `list[Order]` | 变动指令：`{asset, action, weight_delta}` |

**设计决策**：执行层不关心为什么要调仓。它只做 diff。

### 2.6 通知层（Notification Layer）

**职责**：纯输出适配器。将执行层的指令格式化后发送到外部渠道。

**接口契约**：

| 方向 | 格式 | 说明 |
|------|------|------|
| 输入 | `list[Order]` + 组合快照 | 执行层输出 + 当前状态 |
| 输出 | 外部消息 | 钉钉 / 微信 / 日志文件 |

**设计决策**：每个渠道是一个独立适配器。新增渠道 = 新增一个文件，实现统一的 `send(message)` 接口。

---

## 三、回测引擎

回测引擎不是第七层，它是一个**横切所有层的运行模式**。实盘和回测共享数据层、因子层、标准化层、策略层的全部代码，区别仅在：

| | 实盘 | 回测 |
|--|------|------|
| 数据来源 | 数据层实时查询 | 数据层历史查询 |
| 时间推进 | 自然日历 | 引擎逐日遍历 |
| 未来信息 | 物理上不可能 | 引擎强制截断 |
| 输出 | 执行层 → 通知层 | 日收益率 Series → 分析报告 |

### 3.1 核心护栏：未来信息隔离

回测引擎在每个交易日 t 调用因子时，**只传入 t 日及之前的数据**。这个截断由引擎执行，不依赖因子作者自觉。

```
for t in trading_days:
    truncated_data = data[data.index <= t]   # 引擎强制截断
    factor_values = factor.compute(truncated_data)   # 因子只能看到过去
    ...
```

### 3.2 防过拟合护栏

**参数显式声明**：
- 所有可调参数集中在策略配置文件中
- 回测报告头部自动列出全部参数及其值
- 一眼可见系统的自由度数量

**样本外验证**：
- 回测引擎原生支持数据切分：`train_ratio` 参数（默认 0.7）
- 输出报告并排显示训练集和测试集的绩效指标
- 训练集 SR > 2× 测试集 SR 时自动标记警告

**跨资产池验证**：
- 标的池是配置输入，不是系统常量
- 引擎支持"同一策略 × 多个资产池"的批量回测
- 批量报告并排显示各资产池的绩效，验证策略泛化性

**基准对比**：
- 每次回测自动生成等权持有基准（买入并持有标的池中所有资产）
- 报告并排显示策略 vs 基准的收益、回撤、SR
- 策略的复杂度必须用超额收益来"支付"

### 3.3 实验日志

每次回测自动记录完整快照，不允许只保留最好的结果：

```yaml
experiment:
  id: "20260411-001"
  timestamp: "2026-04-11T14:30:00"
  strategy: momentum_rotation
  params:
    momentum_window: 20
    volatility_window: 20
    factor_weights: {momentum: 0.7, volatility: 0.3}
  asset_pool: [510300.SH, 159915.SZ, 513100.SH, 518880.SH]
  data_range: "2016-01-01 ~ 2026-03-31"
  train_test_split: 0.7
  results:
    train:
      total_return: 0.85
      sharpe: 1.2
      max_drawdown: -0.18
    test:
      total_return: 0.32
      sharpe: 0.8
      max_drawdown: -0.22
    benchmark:
      total_return: 0.45
      sharpe: 0.6
```

**设计原则**：任何人拿到这个快照都能一键复现同一结果。"我试了 20 组参数，只有 2 组好看"——这本身就是过拟合的证据，实验日志让这个事实可见。

### 3.4 外部依赖：quantstats

**引入理由**：纯分析报告工具，不侵入架构，退出成本接近零。

**集成方式**：回测引擎输出日收益率 Series → 直接传入 quantstats → 生成 HTML 报告。

**接口**：`quantstats.reports.html(returns_series, benchmark_series, output="report.html")`

---

## 四、团队协作规约

### 4.1 核心原则

**贡献者只需要在因子沙盒内工作，因子以外的一切由系统保证。**

### 4.2 贡献者工作流

1. 复制因子模板文件 `factors/_template.py`
2. 实现 `compute(df: pd.DataFrame) -> pd.Series` 方法
3. 填写元数据（name, params, min_history, direction）
4. 运行本地测试：`python -m pytest factors/tests/test_my_factor.py`
5. 在 `factors/registry.yaml` 中添加一行注册
6. 提交 PR（改动范围：1 个因子文件 + registry 的 1 行新增）

**目标**：新人在 30 分钟内提交第一个因子。如果做不到，说明因子接口设计得还不够简单。

### 4.3 因子模板

```python
"""
因子模板 — 复制此文件，重命名，实现 compute 方法。
"""

METADATA = {
    "name": "my_factor",
    "author": "your_name",
    "version": "1.0.0",
    "params": {"window": 20},
    "min_history": 20,
    "direction": "higher_better",  # or "lower_better"
    "description": "一句话描述因子逻辑",
}

def compute(df, params=None):
    """
    输入: df — 标准化行情 DataFrame (date, open, high, low, close, volume)
    输出: pd.Series — 因子值，日期索引，float 类型
    
    约束:
    - 不得修改输入 df
    - 不得访问全局状态
    - 不得引入 registry 之外的第三方库
    """
    p = {**METADATA["params"], **(params or {})}
    # 你的因子计算逻辑
    return df["close"].pct_change(p["window"])
```

### 4.4 PR 审查清单

因子 PR 只需检查以下项目：

- [ ] 元数据完整（所有必填字段）
- [ ] `compute` 方法签名正确
- [ ] 本地测试通过
- [ ] 未引入未声明的第三方依赖
- [ ] 未修改 `factors/` 目录之外的任何文件
- [ ] registry.yaml 只增加了一行

### 4.5 回测实验的可复现性

- 每次回测自动记录实验日志（见 3.3）
- 实验日志入版本管理
- 任何团队成员可通过日志一键复现：`python run_backtest.py --from-log experiments/20260411-001.yaml`
- 讨论回测结果时必须引用实验 ID，不接受"我这边跑出来不一样"

---

## 五、分模块构建流程

### 5.1 构建策略

采用**水平切片 + 迭代修正**的构建方式：按依赖顺序逐层实现，每个 Phase 对应一次 Claude Code 会话，交付一个可独立运行和测试的模块。

选择水平切片而非垂直切片的前提：开发者已有一个能跑的旧系统，对领域足够熟悉，在 Phase 0 定义的接口出错概率较低。如果是从零开始的全新领域，应改用垂直切片（先跑通最小端到端管线，再重构分层）。

**修正原则**：Phase 0 定义的接口是"最小可行"接口，不追求完美。每个 Phase 结束时，若发现前面的接口定义需要调整，允许回头修正。

### 5.2 Phase 序列

| Phase | 交付模块 | 依赖 | 说明 |
|-------|---------|------|------|
| 0 | 项目骨架 + 接口桩 | 无 | 目录结构、所有层的接口定义（只有签名和类型注解）、CLAUDE.md、uv 环境 |
| 1 | 数据层 | Phase 0 | Tushare 增量同步、本地存储、查询接口 |
| 2 | 因子层 + 标准化层 | Phase 1 | 因子模板、注册机制、输出校验、标准化方法 |
| 3 | 回测引擎 | Phase 2 | 时间序列遍历、未来信息截断、quantstats 集成、实验日志 |
| 4 | 策略层 | Phase 2 | 因子组合、信号生成、策略配置 |
| 5 | 执行层 | Phase 4 | 目标 vs 当前持仓 diff |
| 6 | 通知层 | Phase 5 | 钉钉适配器 |

**每个 Phase 的交付标准**：
1. 模块代码实现完成
2. 该模块的测试全部通过（`uv run pytest {module}/tests/`）
3. CLAUDE.md 更新（记录实际实现细节和与原始设计的偏差）

### 5.3 模块规格卡（Module Spec Card）

每个 Phase 开始前，编写一份模块规格卡作为 Claude Code 的任务输入。规格卡在实现完成后不需要持续维护——代码和测试本身就是最准确的文档。

**模板**：

```
## Phase N — [模块名]
Module: [目录路径]   |   DESIGN.md §[章节号]

Interface (from DESIGN.md):
  Input:  [接口契约中的输入格式]
  Output: [接口契约中的输出格式]

Requirements:
  1. [具体行为需求]
  2. [具体行为需求]
  3. ...

Acceptance:
  - [可执行的验收标准]
  - [可执行的验收标准]

Non-goals:
  - [本阶段明确不做的事]
  - [本阶段明确不做的事]
```

**Non-goals 的作用**：防止 Claude Code 在一次会话里做太多事。明确告诉它"这些事现在不做"，控制每次会话的复杂性预算。

**示例（Phase 1 — 数据层）**：

```
## Phase 1 — Data Layer
Module: data/   |   DESIGN.md §2.1

Interface (from DESIGN.md):
  Input:  (asset_code: str, start: date, end: date)
  Output: pd.DataFrame [date, open, high, low, close, volume]

Requirements:
  1. Tushare 增量同步：本地已有数据则只拉 max(local_date)+1 到今天
  2. 存储：每个标的一个 Parquet 文件，按日期升序
  3. 去重：按日期去重，保留最新记录
  4. 限流：Tushare 返回限流错误时，等待 60s 后重试，最多 3 次
  5. 查询接口返回的 DataFrame 列名严格匹配接口契约

Acceptance:
  - uv run pytest data/tests/ 全部通过
  - 对 510300.SH 连续 sync 两次，第二次日志显示"增量拉取 N 条"
  - query("510300.SH", "2025-01-01", "2025-12-31") 返回正确格式

Non-goals:
  - 本阶段不处理停牌日补全
  - 不实现分钟级数据
  - 不实现数据校验（后续 Phase 按需追加）
```

### 5.4 模块规格卡的生命周期

```
编写 spec → 作为 Claude Code 任务输入 → 实现完成 → 代码+测试成为 living doc → 更新 DESIGN.md（若接口有变）
```

规格卡存放在 `specs/` 目录下，文件名约定 `phase-N-module-name.md`。它们是历史记录（记录了当时的需求），不是需要持续维护的文档。

---

## 六、开发环境与依赖管理

### 6.1 Python 环境：uv

项目使用 [uv](https://docs.astral.sh/uv/) 管理 Python 版本和依赖。

**初始化**：

```bash
uv init quant-framework
cd quant-framework
uv python install 3.12
```

**依赖管理**：

```bash
# 添加核心依赖
uv add pandas numpy tushare pyarrow

# 添加开发依赖
uv add --dev pytest

# 添加可选依赖（回测报告）
uv add quantstats matplotlib

# 运行
uv run python run_daily.py
uv run pytest
```

**关键文件**：
- `pyproject.toml`：项目元数据和依赖声明（uv 自动管理）
- `uv.lock`：锁定文件，确保团队成员环境一致
- `.python-version`：锁定 Python 版本

**团队协作**：新成员 clone 项目后，只需运行 `uv sync` 即可还原完整环境。

### 6.2 依赖决策

| 依赖 | 决定 | 理由 |
|------|------|------|
| pandas / numpy | 引入 | 核心数值计算，无可替代 |
| tushare | 引入 | 数据源，已在使用 |
| pyarrow | 引入 | Parquet 读写引擎 |
| quantstats | 引入 | 纯报告工具，不侵入架构，退出成本零 |
| matplotlib | 引入 | 基础可视化 |
| pytest | 引入（dev） | 测试框架 |
| scipy | 按需 | ERC 优化等场景才引入 |
| Backtrader / Zipline / QLib | **不引入** | 重型框架，强制架构约束，学习成本高，大部分功能用不到 |
| 任何 ORM | **不引入** | Parquet + pandas 足够 |

**依赖引入原则**：每个新依赖必须回答三个问题：
1. 自己写需要多少行？（< 200 行就自己写）
2. 退出成本是什么？（能否在一天内替换掉？）
3. 它会不会替我做架构决策？（如果会，不引入）

| 依赖 | 决定 | 理由 |
|------|------|------|
| pandas / numpy | 引入 | 核心数值计算，无可替代 |
| tushare | 引入 | 数据源，已在使用 |
| quantstats | 引入 | 纯报告工具，不侵入架构，退出成本零 |
| matplotlib | 引入 | 基础可视化 |
| scipy | 按需 | ERC 优化等场景才引入 |
| Backtrader / Zipline / QLib | **不引入** | 重型框架，强制架构约束，学习成本高，大部分功能用不到 |
| 任何 ORM | **不引入** | SQLite + pandas 足够 |

**依赖引入原则**：每个新依赖必须回答三个问题：
1. 自己写需要多少行？（< 200 行就自己写）
2. 退出成本是什么？（能否在一天内替换掉？）
3. 它会不会替我做架构决策？（如果会，不引入）

---

## 七、目录结构

```
quant-framework/
├── pyproject.toml              # uv 项目配置 + 依赖声明
├── uv.lock                     # 依赖锁定文件（入版本管理）
├── .python-version             # Python 版本锁定
├── CLAUDE.md                   # Claude Code 执行依据（每个 Phase 结束时更新）
│
├── data/                       # 数据层
│   ├── sync.py                 # Tushare 增量同步逻辑
│   ├── store.py                # Parquet 本地存储读写
│   ├── db/                     # Parquet 文件目录
│   └── tests/
│
├── factors/                    # 因子层（贡献者的沙盒）
│   ├── _template.py            # 因子模板
│   ├── registry.yaml           # 因子注册表
│   ├── validator.py            # 因子输出校验器
│   ├── momentum.py             # 动量因子
│   ├── volatility.py           # 波动率因子
│   └── tests/
│
├── standardization/            # 标准化层
│   ├── methods.py              # z-score, rank, percentile
│   └── tests/
│
├── strategy/                   # 策略层
│   ├── base.py                 # 策略基类
│   ├── momentum_rotation.py    # 动量轮动策略
│   ├── configs/                # 策略配置 YAML
│   └── tests/
│
├── execution/                  # 执行层
│   ├── engine.py               # 目标 vs 当前 diff
│   └── tests/
│
├── notification/               # 通知层
│   ├── base.py                 # 通知接口
│   ├── dingtalk.py             # 钉钉适配器
│   └── tests/
│
├── backtest/                   # 回测引擎
│   ├── runner.py               # 时间序列遍历 + 未来信息截断
│   ├── report.py               # quantstats 集成
│   ├── experiment_log.py       # 实验日志记录
│   └── tests/
│
├── experiments/                # 实验日志存储
│   └── 20260411-001.yaml
│
├── state/                      # 运行状态持久化
│   └── current_position.json
│
├── specs/                      # 模块规格卡（历史记录）
│   ├── phase-0-scaffold.md
│   ├── phase-1-data-layer.md
│   ├── phase-2-factor-layer.md
│   └── ...
│
├── docs/                       # 文档
│   ├── DESIGN.md               # 本文档
│   └── FACTOR_GUIDE.md         # 因子开发者指南
│
├── run_daily.py                # 实盘每日运行入口
└── run_backtest.py             # 回测运行入口
```

---

## 八、设计决策日志

| 决策点 | 结论 | 依据 |
|--------|------|------|
| 标准化独立成层 vs 放在因子内部 | 独立成层 | 标准化方法是可替换的实验对象，耦合在因子内会导致换方法时改所有因子 |
| 通用框架 vs 自建 | 自建 | 策略复杂度低（日线、少量标的），通用框架的学习成本和约束远超收益 |
| quantstats 引入 | 引入 | 纯工具，不侵入架构，退出成本零 |
| 未来信息隔离 | 引擎强制截断 | 不依赖因子作者自觉，架构层面防止前视偏差 |
| 因子注册机制 | 显式注册（registry.yaml） | 自动扫描会加载未经审查的代码，手动注册太容易遗漏，显式注册兼顾安全和便利 |
| 参数集中声明 | 策略配置文件 | 散落在代码中的参数是过拟合的温床，集中后一眼可见系统自由度 |
| 实验日志 | 每次回测必须完整记录 | "只保留最好的结果"是过拟合的帮凶 |
| 跨资产池验证 | 内置能力 | 4 个标的上调参的过拟合风险极高，跨资产验证是最有效的泛化性检验 |
| 回测全量计算 vs 增量 | 先全量 | 复杂性预算原则——当前数据量不构成瓶颈，增量计算的复杂度不值得 |
| 回测 train/test 分割 | 默认 0.7 | 原生支持，输出并排显示，SR 差异过大自动警告 |
| 水平切片 vs 垂直切片 | 水平切片 + 迭代修正 | 开发者已有旧系统，领域理解成熟，接口定义出错概率低；新项目应选垂直切片 |
| 模块级设计文档 vs 规格卡 | 规格卡（一次性任务输入） | 正式文档需要持续维护，六个模块六份文档的维护成本超过价值；规格卡被消费后由代码和测试承担文档职责 |
| 规格卡中的 Non-goals | 必填字段 | 控制 Claude Code 每次会话的复杂性预算，防止自主扩展实现范围 |
| 会话间状态传递 | 每个 Phase 结束更新 CLAUDE.md | 确保下一次会话基于最新系统状态，而非过时设计 |
| 环境管理 | uv | 依赖锁定（uv.lock）确保团队环境一致，新成员 `uv sync` 一条命令还原 |
| 数据存储 | Parquet（非 SQLite） | pyarrow 读写性能好，文件级存储与 pandas 原生兼容，无需 ORM |
