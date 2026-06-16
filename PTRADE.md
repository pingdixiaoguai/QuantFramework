# `ptrade` 分支 — PTrade 实盘执行端(overlay 分支,**永不合并到 main**）

本分支 = `main` 的全部框架代码 **+** 一层 PTrade-only overlay。是恒生 PTrade 平台的实盘执行端。

> **铁律**:本分支 **永不 merge 到 `main`**。`main` 保持框架单一真相源,不含任何 PTrade 文件。
> 合并方向是**单向**的:`main` → 定期合进 `ptrade`(拿框架更新);`ptrade` 绝不回流 `main`。

## PTrade-only overlay(只在本分支存在的文件）

| 路径 | 内容 |
|------|------|
| `deploy/` | PTrade 策略主文件 + `PTRADE_MIGRATION.md` + 探针 |
| `backtest/ptrade/` | rd2/rd5 回测 CSV + 截图 + `README.md` + 归因 memo（`Log.txt` 不入库） |
| `scripts/ptrade_*.py` | 对账诊断脚本 |
| `CONTRACT.md` | 迁移"完成"的接口契约 + 待建逐日对账测试 |
| `PTRADE.md` | 本文件 |

其余一切（`factors/`、`strategy/`、`backtest/runner.py`、`data/`、`changelog` …）都来自 `main`,**不要在本分支改框架代码**。

## 工作流

```
框架研究：  从 main 切 feat/*  →  PR 合 main  →  回到 ptrade: git merge main
PTrade 工作：直接在 ptrade 上提交（或 ptrade/* 小分支合进 ptrade）
```

- **框架改动必须在 main 侧分支写**,绝不在 ptrade 上写后再想弄回 main(那要 cherry-pick,易乱)。
- **PTrade-only 文件绝不出现在任何对着 main 的 PR 上**。main 上有 CI 防护(`.github/workflows/no-ptrade-on-main.yml`)做物理拦截。
- 任何会改变信号/时序/选股的 PTrade 改动,**先回 main 侧框架回测**,再同步过来。

## 拿框架最新代码

```bash
git checkout ptrade
git merge main          # 把 main 的框架更新合进来(单向)
```

## 出处

PTrade overlay 内容迁出自原 `nyxx-dev` 分支(已退役),截至 commit `2755db9`(2026-06-15)。
背景见 QuantFramework PR #17 review:PTrade 迁移不进 main,框架保持单一真相源。
