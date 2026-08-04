# Notification Layer

## Contract
Adapter interface (`interfaces.py`): `class Notifier(ABC)` with `send(message: str) -> None`
Message builder (`formatter.py`): `format_notification(ctx: NotificationContext) -> str` → markdown string for DingTalk

## Implementation Notes
- `dingtalk.py` — `DingTalkNotifier(webhook_url, secret)`; loads `.env` and reads `DINGTALK_WEBHOOK` / `DINGTALK_SECRET` as fallback; HMAC-SHA256 signing appended as `timestamp` + `sign` query params when `secret` is set
- `send()` performs **two POSTs**: (1) the `markdown` card with the formatted message, (2) a plain `text` message `"请查看今日调仓信号，及时操作！"` with `isAtAll: True` to trigger @所有人 (DingTalk only reliably fires the group-wide alert for `text` type — see commit `68a3fc7`). Optional keyword-only `title` and `alert_text` override those labels for safe test sends while preserving production defaults.
- `send_alert(message)` performs one plain-text `isAtAll: True` POST and is used by unattended-job failure reporting
- `formatter.py` renders different layouts based on whether orders contain `buy`/`sell` (rebalance) or only `hold`
  - Sections: header → actual execution → symmetric production/shadow signal comparison → benchmark comparison → YTD return
  - `NotificationContext` aggregates execution state and optional `StrategySignalView` objects containing each strategy's target, primary-factor scores, relative strength, and expected universe
  - Each strategy renders Top2 scores, scale-free cross-sectional softmax relative strength, Top1 lead, data completeness, and cross-strategy rank changes. Relative strength is explicitly labelled as non-probabilistic.
  - Shadow configs are ordinary strategy YAMLs loaded read-only through the factor registry and strategy loader. They do not affect production orders or position state.
  - `ASSET_NAMES` dict maps `510300.SH → 沪深300` etc.; unknown tickers fall back to the raw code

### Known deviations from DESIGN.md
- DESIGN.md §2.6 describes a single `send(message)` call per notification. Actual DingTalk flow sends **two** messages per notification to work around the platform's @所有人 limitation.
- Input to the layer is richer than "`list[Order]` + snapshot" — it's a full `NotificationContext` including benchmark and YTD data. The extra fields come from the daily runner, not the execution layer.

## Pitfalls
- A raw `markdown` payload with `@all` inside the text does NOT trigger the notification sound on DingTalk — the separate `text` POST is required. Do not fold them back into one call.
- HMAC signing timestamp is milliseconds (`int(time.time() * 1000)`), not seconds — DingTalk rejects second-precision timestamps with a signature error
- `.env.example` contains non-working `your_token_here` / `your_secret_here` values. `DingTalkNotifier` rejects those placeholders with a clear `ValueError`; replace both with the custom robot's actual credentials before testing.
- Adding a new asset to `ASSET_NAMES` must be coordinated with `strategy/configs/*.yaml`'s `asset_pool`; mismatched codes render as raw tickers
- Signal confidence is `softmax(score / population_std(score))` across the candidate ETFs. The standard-deviation temperature is required because raw return-like scores are small and direct softmax would remain close to equal probabilities.
- Shadow diagnostics compare raw daily targets. They do not maintain a separate shadow position ledger or apply the shadow config's hold window.
