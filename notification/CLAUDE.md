# Notification Layer

## Contract
Adapter interface (`interfaces.py`): `class Notifier(ABC)` with `send(message: str) -> None`
Message builder (`formatter.py`): `format_notification(ctx: NotificationContext) -> str` → markdown string for DingTalk

## Implementation Notes
- `dingtalk.py` — `DingTalkNotifier(webhook_url, secret)`; reads `DINGTALK_WEBHOOK` / `DINGTALK_SECRET` env vars as fallback; HMAC-SHA256 signing appended as `timestamp` + `sign` query params when `secret` is set
- `send()` performs **two POSTs**: (1) the `markdown` card with the formatted message, (2) a plain `text` message `"请查看今日调仓信号，及时操作！"` with `isAtAll: True` to trigger @所有人 (DingTalk only reliably fires the group-wide alert for `text` type — see commit `68a3fc7`). Optional keyword-only `title` and `alert_text` override those labels for safe test sends while preserving production defaults.
- `run_daily.py --notification-only` sends a clearly labelled test notification and returns before any position backfill or persistence. It still syncs market data and computes the full production/shadow comparison.
- `formatter.py` renders different layouts based on whether orders contain `buy`/`sell` (rebalance) or only `hold`
  - Sections: header → (rebalance instructions OR current position) → alpha comparison (rebalance only) → benchmark comparison → YTD return
  - `NotificationContext` aggregates: orders, target/current weights, entry date, holding days, position return, benchmark returns per asset, YTD return, optional per-asset factor values, and optional production signal confidence
  - The notification appends the production signal's scale-free cross-sectional softmax confidence to each ETF comparison line, after the excess-return item, and shows the old/new Top1 targets in a compact summary. Raw old/new ER diagnostics are not sent to DingTalk.
  - `ASSET_NAMES` dict maps `510300.SH → 沪深300` etc.; unknown tickers fall back to the raw code

### Known deviations from DESIGN.md
- DESIGN.md §2.6 describes a single `send(message)` call per notification. Actual DingTalk flow sends **two** messages per notification to work around the platform's @所有人 limitation.
- Input to the layer is richer than "`list[Order]` + snapshot" — it's a full `NotificationContext` including benchmark and YTD data. The extra fields come from the daily runner, not the execution layer.

## Pitfalls
- A raw `markdown` payload with `@all` inside the text does NOT trigger the notification sound on DingTalk — the separate `text` POST is required. Do not fold them back into one call.
- HMAC signing timestamp is milliseconds (`int(time.time() * 1000)`), not seconds — DingTalk rejects second-precision timestamps with a signature error
- `.env.example` contains non-working `your_token_here` / `your_secret_here` values. `DingTalkNotifier` rejects those placeholders with a clear `ValueError`; replace both with the custom robot's actual credentials before testing.
- Adding a new asset to `ASSET_NAMES` must be coordinated with `strategy/configs/*.yaml`'s `asset_pool`; mismatched codes render as raw tickers
- `_build_alpha_section` assumes all `asset_factor_values` entries share the same factor names — it reads them from the first asset only
- `benchmark_returns` and `target_weights` keys must overlap for the alpha/superiority calculation to work; missing keys are silently skipped
- Signal confidence is `softmax(score / population_std(score))` across the candidate ETFs. The standard-deviation temperature is required because raw return-like scores are small and direct softmax would remain close to equal probabilities.
