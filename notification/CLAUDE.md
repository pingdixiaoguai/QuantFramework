# Notification Layer

## Contract
Adapter interface (`interfaces.py`): `class Notifier(ABC)` with `send(message: str) -> None`
Message builder (`formatter.py`): `format_notification(ctx: NotificationContext) -> str` → markdown string for DingTalk

## Implementation Notes
- `dingtalk.py` — `DingTalkNotifier(webhook_url, secret)`; reads `DINGTALK_WEBHOOK` / `DINGTALK_SECRET` env vars as fallback; HMAC-SHA256 signing appended as `timestamp` + `sign` query params when `secret` is set
- `send()` performs **two POSTs**: (1) the `markdown` card with the formatted message, (2) a plain `text` message `"请查看今日调仓信号，及时操作！"` with `isAtAll: True` to trigger @所有人 (DingTalk only reliably fires the group-wide alert for `text` type — see commit `68a3fc7`)
- `formatter.py` renders different layouts based on whether orders contain `buy`/`sell` (rebalance) or only `hold`
  - Sections: header → (rebalance instructions OR current position) → alpha comparison (rebalance only) → benchmark comparison → YTD return
  - `NotificationContext` aggregates: orders, target/current weights, entry date, holding days, position return, benchmark returns per asset, YTD return, optional per-asset factor values
  - `ASSET_NAMES` dict maps `510300.SH → 沪深300` etc.; unknown tickers fall back to the raw code

### Known deviations from DESIGN.md
- DESIGN.md §2.6 describes a single `send(message)` call per notification. Actual DingTalk flow sends **two** messages per notification to work around the platform's @所有人 limitation.
- Input to the layer is richer than "`list[Order]` + snapshot" — it's a full `NotificationContext` including benchmark and YTD data. The extra fields come from the daily runner, not the execution layer.

## Pitfalls
- A raw `markdown` payload with `@all` inside the text does NOT trigger the notification sound on DingTalk — the separate `text` POST is required. Do not fold them back into one call.
- HMAC signing timestamp is milliseconds (`int(time.time() * 1000)`), not seconds — DingTalk rejects second-precision timestamps with a signature error
- Adding a new asset to `ASSET_NAMES` must be coordinated with `strategy/configs/*.yaml`'s `asset_pool`; mismatched codes render as raw tickers
- `_build_alpha_section` assumes all `asset_factor_values` entries share the same factor names — it reads them from the first asset only
- `benchmark_returns` and `target_weights` keys must overlap for the alpha/superiority calculation to work; missing keys are silently skipped
