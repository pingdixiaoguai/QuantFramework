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

`run_daily_momentum_defender.py` formats the composite strategy's sleeve state,
slow gate, emergency cap, both inner sleeve targets, exact orders, and implied
cash. Its `--dry-run` mode must call neither DingTalk nor position persistence;
`--notification-only` may send a labelled test but must not write state.
The same runner is the formal Gold RAQM-W5 entry point. Formal notifications
must expose the base-C2 target, Gold and Defender factor values, their
difference, frozen thresholds, Gold hard-hold count, and final executable
allocation.
