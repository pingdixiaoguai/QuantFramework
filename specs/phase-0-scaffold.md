# Phase 0 — 项目骨架与接口桩

Module: `/`（项目根）  |  DESIGN.md §二、§5.2、§6、§七

> 规格卡是**一次性任务输入**。实现完成后，代码与测试本身即是最准确的文档；本文件不再维护。

---

## 目标

建立 DESIGN.md §七 完整目录树，为六层架构各写一份仅含签名与类型注解的接口桩，准备好 uv 依赖与 `.env` 机制。本阶段不写任何真实业务逻辑，只搭出让后续 Phase 能"填空"的骨架。

## Interface（本 Phase 的产出是"接口本身"）

为六层各创建 `interfaces.py`，函数体统一 `raise NotImplementedError`。六层的签名按 DESIGN.md §二 接口契约逐字落地：

| 层 | 函数签名 |
|---|---|
| 数据层 | `query(asset_code: str, start: date, end: date) -> pd.DataFrame` |
| 因子层 | `compute(df: pd.DataFrame, params: dict \| None = None) -> pd.Series` |
| 标准化层 | `standardize(raw: dict[str, pd.Series], method: str = "cross_sectional_rank") -> dict[str, pd.Series]` |
| 策略层 | `generate_weights(standardized: dict[str, pd.Series], config: dict) -> dict[str, float]` |
| 执行层 | `diff(target: dict[str, float], current: dict[str, float]) -> list[Order]` |
| 通知层 | `send(message: str) -> None`（抽象基类 `Notifier.send`） |

`Order` 在 `execution/interfaces.py` 以 `@dataclass` 定义：`asset: str`, `action: Literal["buy", "sell", "hold"]`, `weight_delta: float`。

## Requirements

1. **目录结构**：按 DESIGN.md §七 创建全部子目录（`data/`, `factors/`, `standardization/`, `strategy/`, `execution/`, `notification/`, `backtest/`, `experiments/`, `state/`, `specs/`, `docs/`）。每个 Python 包必须含 `__init__.py`；每个带 `tests/` 的模块再建一个 `tests/__init__.py`。
2. **接口桩**：为数据层、因子层、标准化层、策略层、执行层、通知层分别创建 `{layer}/interfaces.py`，签名严格按上表，函数体 `raise NotImplementedError`，附一行 docstring 指向 DESIGN.md 对应章节。
3. **pyproject.toml**：在现有文件基础上补齐依赖：
   - runtime：`pandas`, `numpy`, `tushare`, `pyarrow`, `python-dotenv`, `pyyaml`
   - dev：`pytest`
   - 通过 `uv add <pkg>` 和 `uv add --dev pytest` 添加，让 uv 自动写入；不要手改 `[project.dependencies]`。
4. **删除占位**：删 `main.py`；新建 `run_daily.py`、`run_backtest.py` 作为入口桩，`if __name__ == "__main__": print("not implemented yet")`。
5. **环境变量机制**：
   - 新建 `.env.example`，仅含 `TUSHARE_TOKEN=your_token_here`
   - 更新 `.gitignore`，追加 `.env`、`data/db/`、`experiments/*.yaml`、`state/*.json`、`__pycache__/`、`.pytest_cache/`
6. **CLAUDE.md（项目根）**：创建种子内容，包含三节：
   - `# Status`：表格，行 = Phase 0-6，列 = 交付物 / 完成状态，Phase 0 一项填 `in-progress`，其余 `pending`
   - `# Entry Points`：列出本项目的导航起点（`docs/DESIGN.md` 为权威架构源，`specs/` 为规格卡历史）
   - `# Conventions`：复述 DESIGN.md §4.2 贡献者工作流要点
7. **接口桩的最小可加载性**：从每个接口桩 `from {layer}.interfaces import <symbol>` 必须可成功 import；调用才报 `NotImplementedError`。

## Acceptance

- `uv sync` 成功，`uv.lock` 产生或更新，入 git
- `uv run python -c "import data.interfaces, factors.interfaces, standardization.interfaces, strategy.interfaces, execution.interfaces, notification.interfaces; print('ok')"` 输出 `ok`
- `uv run python run_daily.py` 与 `uv run python run_backtest.py` 都输出 `not implemented yet`
- `uv run pytest` 成功退出（无测试也允许，退出码 0 或 5 均可接受——5 表示"no tests collected"，这是 Phase 0 的预期）
- 项目根不再有 `main.py`
- `git status` 下新增 / 修改的文件清单符合本规格卡的 Requirements，不含 `__pycache__` 或 `.env`
- `CLAUDE.md` 存在，Status 表格 Phase 0 列改为 `done`（由实现者在收尾时更新）

## Non-goals

- 不写任何真实业务逻辑：禁止在 Phase 0 动手实现任何接口桩的函数体
- 不引入 `quantstats` / `matplotlib` / `scipy`（留给 Phase 3）
- 不创建因子模板 `factors/_template.py`（留给 Phase 2，以免与接口桩语义混淆）
- 不创建 `factors/registry.yaml`、`strategy/configs/*.yaml`（留给对应 Phase）
- 不配置 CI、pre-commit、格式化工具
- 不写 `docs/FACTOR_GUIDE.md`
- 不预先创建 `experiments/` 下的示例日志

## 收尾动作

实现者在 Phase 0 结束时：

1. 更新 `CLAUDE.md` Status 表：Phase 0 = `done`
2. 提交 git commit，信息类似 `Phase 0: scaffold + interface stubs`
3. 不要修改本规格卡文件
