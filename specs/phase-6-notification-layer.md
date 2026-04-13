## Phase 6 — Notification Layer
Module: notification/   |   DESIGN.md §2.6

Interface (from DESIGN.md):
  Input:  list[Order] + portfolio snapshot
  Output: external message (DingTalk)

Requirements:
  1. Notifier base class already exists (notification/interfaces.py)
  2. DingTalkNotifier(webhook_url, secret=None) — send via DingTalk robot webhook
     - POST JSON to webhook URL with msgtype=markdown
     - Support optional HMAC-SHA256 signing (secret) for security
     - webhook_url from env var DINGTALK_WEBHOOK
     - secret from env var DINGTALK_SECRET (optional)
  3. format_orders(orders, target_weights) → markdown string
     - Table of orders: asset, action, weight_delta, target_weight
     - Summary line: date, total buy/sell count
  4. run_daily.py wiring: data → factors → strategy → execution → notification
     - Load config from YAML
     - Run strategy for today
     - Diff against current position
     - Send notification
     - Save new position

Acceptance:
  - uv run pytest notification/tests/ all pass
  - DingTalkNotifier.send() calls correct URL with correct payload
  - format_orders produces readable markdown
  - run_daily.py runs end-to-end (with mocked external calls)

Non-goals:
  - No WeChat or other channel adapters
  - No retry logic for webhook failures
  - No backtest integration (notification is live-only)
