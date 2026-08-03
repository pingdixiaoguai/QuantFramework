# Notification module contract

`format_notification(NotificationContext)` returns markdown. The daily runner
provides production execution data plus optional read-only shadow diagnostics.

Confidence is a cross-sectional softmax of the production primary-factor
scores and is rendered on every daily comparison line, including hold days.
Old/new signal targets are informational only; the message must state that the
shadow signal does not change production trading.

`--notification-only` may run on weekends and holidays using the latest common
priced date. It must not backfill or write production position state.
