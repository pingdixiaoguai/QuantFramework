# Phase 1 — 数据层

Module: `data/`  |  DESIGN.md §2.1

> 规格卡是**一次性任务输入**。实现完成后由代码和测试承担文档职责，本文件不再维护。

---

## 目标

填实 Phase 0 建立的 `data/interfaces.py` 桩：从 Tushare 增量同步日线行情到本地 Parquet 存储，提供统一查询接口，为 Phase 2 的因子层喂标准化 DataFrame。

## Interface（from DESIGN.md §2.1）

**查询**：
```
query(asset_code: str, start: date, end: date) -> pd.DataFrame
```
返回列严格为 `[date, open, high, low, close, volume]`，`date` 为 `datetime64[ns]`，按日期升序，不含重复日期。

**同步（新增，不在 DESIGN.md §2.1 的对外契约内，为实现细节）**：
```
sync(asset_code: str) -> int   # 返回本次新增行数
```
增量策略：读取本地文件最新日期 `max_local`，若存在则请求 Tushare `(max_local + 1, today)`；若不存在则全量拉取 `(2016-01-01, today)`。

## Requirements

1. **存储布局**：一标的一文件，路径 `data/db/{asset_code}.parquet`（示例：`data/db/510300.SH.parquet`）。首次同步时 `data/db/` 不存在则自动创建。
2. **Tushare 对接**：
   - 使用 `tushare.pro_api().pro_bar(ts_code, asset, start_date, end_date, adj="qfq")` 或等价接口拉日线
   - token 通过 `data/config.py` 的 `get_tushare_token()` 加载：先查 `os.environ["TUSHARE_TOKEN"]`，无则用 `python-dotenv` 加载项目根 `.env` 后再查环境变量
   - 若 token 仍缺失，抛 `RuntimeError("TUSHARE_TOKEN missing; set env var or put it in .env")`
3. **列名标准化**：Tushare 原生列（如 `trade_date`, `vol`）必须映射为契约规定的 `[date, open, high, low, close, volume]`。映射逻辑放在 `data/store.py` 的一个小函数里，不要散落。
4. **增量合并去重**：新拉取的数据与本地已有合并后按 `date` 去重（保留新）、升序排序，再覆写 Parquet 文件。
5. **限流重试**：捕获 Tushare 返回的限流异常（`Exception` 消息含 `"rate"` 或 `"40203"` 等——以实际 Tushare 行为为准，允许实现者宽松匹配）；等待 60 秒后重试，最多 3 次；仍失败则抛出。
6. **命令行入口**：`python -m data.sync <asset_code>` 能直接运行，调用 `sync()` 并打印 `"synced N rows for {asset_code}"`。
7. **ETF 标的适配**：初始标的池 `510300.SH / 159915.SZ / 513100.SH / 518880.SH` 都是 ETF，需要在 `pro_bar` 中传 `asset="FD"`（基金）。`sync()` 内部可以写死 `asset="FD"`——本阶段只支持 ETF，后续扩展到股票再抽象。

## Acceptance

**手动验收（必须跑一次）**：

1. 在 `.env` 配好 `TUSHARE_TOKEN`
2. `uv run python -m data.sync 510300.SH` → 输出类似 `synced 2400 rows for 510300.SH`，`data/db/510300.SH.parquet` 存在
3. 立即再跑一次 `uv run python -m data.sync 510300.SH` → 输出 `synced 0 rows` 或 `synced 1 rows`（取决于是否跨交易日），**不得重新拉取全部历史**
4. `uv run python -c "from datetime import date; from data.store import query; df = query('510300.SH', date(2024,1,1), date(2024,12,31)); print(df.columns.tolist()); print(df.shape); print(df.head(3))"` → 列名严格为 `['date','open','high','low','close','volume']`，行数 > 200
5. 批量跑一遍 4 只 ETF：`for c in 510300.SH 159915.SZ 513100.SH 518880.SH; do uv run python -m data.sync $c; done` 全部成功

**自动化单元测试（最小集）**：放 `data/tests/`，通过 `uv run pytest data/tests/`：

1. `test_merge_incremental.py::test_dedup_keeps_latest`：构造两个有日期重叠的 DataFrame，合并后重叠日期只保留新值且行数正确。
2. `test_merge_incremental.py::test_sort_ascending`：合并后日期严格升序。
3. `test_config.py::test_token_from_env_var`：用 `monkeypatch.setenv("TUSHARE_TOKEN", "fake")`，`get_tushare_token()` 返回 `"fake"`。
4. `test_config.py::test_token_missing_raises`：同时清掉 env 和 `.env`，调用抛 `RuntimeError`。

**非要求（显式豁免）**：不要求 mock Tushare 去测 `sync()` 主流程——手动验收承担这一角色。

## Non-goals

- 不处理停牌日补全（停牌日 Tushare 本就不返回，`query()` 自然缺这天，不管）
- 不做分钟线、日内行情、龙虎榜
- 不做字段合法性校验（比如 `close > 0`）——Phase 1 之外按需追加
- 不支持 A 股个股 / 港股 / 美股（`asset="FD"` 写死）
- 不做缓存层、连接池、并发同步（一次一只，串行）
- 不处理 Tushare 积分不足（让异常直接抛出即可）
- 不写数据层的 CLI 子命令聚合器（只有 `python -m data.sync` 一个入口）

## 收尾动作

1. 更新项目根 `CLAUDE.md` Status 表：Phase 1 = `done`，附一行"实际实现与 DESIGN.md 的偏差"（如有）
2. 如果 DESIGN.md §2.1 的接口契约与实际实现有冲突，记录在 CLAUDE.md 的 `Deviations` 节
3. 提交 git commit
