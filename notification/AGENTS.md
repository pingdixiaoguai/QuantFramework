# Notification module contract

`format_notification(NotificationContext)` returns markdown. The daily runner
provides production execution data plus optional read-only shadow diagnostics
as `StrategySignalView` objects.

Each strategy view renders its own Top2 scores, cross-sectional softmax relative
strength, Top1 lead, universe completeness, and rank changes. Relative strength
is not a forecast probability. The execution block must distinguish the
production raw target from the hold-filtered trading target.

Shadow scores and targets are informational only. They do not apply a separate
shadow hold window, and the message must state that the shadow signal does not
change production trading or position persistence.

`--notification-only` may run on weekends and holidays using the latest common
priced date. It must not backfill or write production position state.
