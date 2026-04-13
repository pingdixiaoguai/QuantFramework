# Notification Layer Redesign

**Date:** 2026-04-13  
**Status:** Approved  

## Overview

Redesign the DingTalk notification to show a rich signal summary: current holding with days held and P&L, same-period benchmark comparison for all pool assets, YTD return, and @all mention.

---

## 1. State Structure

`state/current_position.json` is extended to carry entry metadata and YTD history.

### New schema

```json
{
  "weights": { "513100.SH": 1.0 },
  "entry_date": "2026-04-08",
  "entry_prices": { "513100.SH": 1.234 },
  "ytd_history": [
    {
      "weights": { "159915.SZ": 1.0 },
      "entry_date": "2026-01-02",
      "exit_date": "2026-04-07",
      "entry_prices": { "159915.SZ": 3.012 },
      "exit_prices":  { "159915.SZ": 2.988 }
    }
  ]
}
```

### Field semantics

| Field | Type | Meaning |
|-------|------|---------|
| `weights` | `dict[str, float]` | Current target weights |
| `entry_date` | `str \| null` | Actual entry date (signal_date + 1 trading day). `null` when tomorrow's open not yet known |
| `entry_prices` | `dict[str, float] \| null` | Open prices on `entry_date`. `null` until the next run fills them in |
| `ytd_history` | `list[PositionPeriod]` | Closed position periods since Jan 1 |

### PositionPeriod fields

| Field | Meaning |
|-------|---------|
| `weights` | Weights held during this period |
| `entry_date` | Actual entry date |
| `exit_date` | Actual exit date |
| `entry_prices` | Open prices on entry day |
| `exit_prices` | Open prices on exit day (next trading day after signal) |

### Migration

Existing state files in the old format `{ "ASSET": weight }` are auto-migrated on first read:
- Keys become `weights`
- `entry_date`, `entry_prices` set to `null`
- `ytd_history` set to `[]`

### Entry price lifecycle (delayed write)

Because signal is generated at today's close and the actual open price is unavailable until the next trading day:

1. **On rebalance day T:** write `entry_date = T+1`, `entry_prices = null`
2. **On next run T+1:** if `entry_prices` is `null`, read open prices for `entry_date` from the data store and backfill them before proceeding

---

## 2. Message Format

The formatter produces a DingTalk markdown string from a `NotificationContext` dataclass.

### Hold (no rebalance)

```
## 📊 Quality_Momentum 信号

**信号日期：** 2026-04-09
基于当日收盘价，建议次日开盘调仓

---

**当前持仓**
• 513100 纳指ETF　100%　|　已持有 2 交易日　|　收益 **+0.89%**

---

**同期对比**（自 2026-04-08 开盘起）
• 159915 创业板　+2.44%　|　超出持仓 +1.54% ↑
• 510300 沪深300　+1.17%　|　超出持仓 +0.28% ↑
• 518880 黄金ETF　-1.93%　|　落后持仓 -2.82% ↓

---

**年初至今：** -2.09%
```

### Rebalance day

The "当前持仓" section is replaced by a rebalance instruction block:

```
**调仓指令**
• 卖出：159915 创业板　100% → 0%
• 买入：513100 纳指ETF　0% → 100%
```

Benchmark comparison and YTD are still shown below.

### Fallback states

| Condition | Display |
|-----------|---------|
| `entry_prices` is `null` | 收益显示"待更新" |
| `entry_date` is `null` | 持有天数和收益均显示"—" |
| `ytd_history` empty and no closed periods | YTD 显示"数据不足" |

### Asset name table

Maintained as a static dict in `formatter.py`:

```python
ASSET_NAMES = {
    "510300.SH": "沪深300",
    "159915.SZ": "创业板",
    "513100.SH": "纳指ETF",
    "518880.SH": "黄金ETF",
}
```

Assets not in the table fall back to showing the raw ticker code.

### NotificationContext interface

```python
@dataclass
class NotificationContext:
    strategy_name: str
    signal_date: date
    orders: list[Order]
    target_weights: dict[str, float]
    current_weights: dict[str, float]
    entry_date: date | None           # actual entry day (signal+1 trading day)
    holding_days: int | None          # trading days held
    position_return: float | None     # current position weighted return
    benchmark_returns: dict[str, float]  # asset → same-period return
    ytd_return: float | None          # YTD cumulative return
    asset_names: dict[str, str]       # asset → Chinese name
```

---

## 3. Data Computation

### execution/position.py

`save_position` signature is extended:

```python
def save_position(
    target_weights: dict[str, float],
    entry_date: date,
    entry_prices: dict[str, float] | None,
) -> None: ...
```

On save, the outgoing position (with `exit_date` and `exit_prices`) is appended to `ytd_history` before the new position is written.

`read_position` returns a typed `PositionState` dataclass (or dict with the new schema) instead of bare weights, and handles old-format migration.

### run_daily.py orchestration order

1. Data sync + freshness check
2. Read state → `current_state` (auto-migrates old format if needed)
3. **Backfill open prices** if `current_state.entry_prices is null`: read open prices for `entry_date` from data store, write back to state. Also backfill `exit_prices` on the last `ytd_history` entry (same date, same open prices).
4. Strategy → `target_weights`
5. `diff()` → `orders`
6. Query price data → compute:
   - `holding_days`: count rows in data store with date in `[entry_date, today]` for any pool asset
   - `position_return`: `sum(weight * (close_today / open_entry - 1) for asset, weight in weights)`
   - `benchmark_returns`: same formula for every asset in `asset_pool`
   - `ytd_return`: chain-multiply closed periods from `ytd_history` plus current period: `(1+r1)*(1+r2)*...*(1+r_current) - 1`
7. Assemble `NotificationContext` — populate `asset_names` from `formatter.ASSET_NAMES` — format → send
8. Compute `next_entry_date` (next date after today that has data in the store) and call `save_position(target_weights, next_entry_date, entry_prices=None)`

---

## 4. DingTalk @all

`DingTalkNotifier.send()` adds `at.isAtAll: true` to the payload:

```python
payload = {
    "msgtype": "markdown",
    "markdown": {"title": "调仓信号", "text": message},
    "at": {"isAtAll": True},
}
```

The `Notifier` base interface (`send(message: str)`) is unchanged. `@all` is an internal DingTalk adapter detail.

---

## 5. Files Changed

| File | Change |
|------|--------|
| `state/current_position.json` | Schema extended (auto-migrated on read) |
| `execution/position.py` | `read_position` → returns `PositionState`; `save_position` → accepts entry metadata |
| `notification/formatter.py` | Replace `format_orders()` with `format_notification(ctx: NotificationContext)` |
| `notification/dingtalk.py` | Add `at.isAtAll: true` to payload |
| `run_daily.py` | Add price queries, assemble `NotificationContext`, update `save_position` call |
| `notification/tests/test_formatter.py` | Update tests for new interface |

---

## 6. Out of Scope

- No changes to backtest engine
- No changes to strategy or factor layers
- No new external dependencies (trading-day calendar uses existing data store)
