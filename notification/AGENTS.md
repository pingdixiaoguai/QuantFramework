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

Formal W40 notifications also include a read-only peak-warning progress block
for the daily Momentum Top1, including while the formal sleeve is Defender.
Every condition shows pass/fail, its current value, threshold, and remaining
gap. Triggering requires a strict prior-200-session close breakout, a
20-session return of at least 15%, and signal-day volume at least 1.5 times the
strict prior-20-session median. For 159915.SZ only, the rule requires an
additional strictly positive 20-session fund-share change; flat, falling,
stale, or unavailable shares suppress the alert.
The block is diagnostic only and must never alter targets, orders, locks,
position persistence, or the prospective strategy ledger.

The current formal W40/QM40/Gold notification uses exactly three Chinese sleeve
labels (动量、防守、黄金逃生) and the fixed Chinese reason mapping documented
in README. It does not display workstation position state. Its instruction is
derived from the formal signal's previous and target candidates; identical
model allocations render as "继续持有" even when a local position JSON is
absent. W40 evidence streaks render as elapsed consecutive days, without the
redundant `/1` confirmation denominator.
It also exposes the strict-lag 756-session W40 percentile, 60%/35% lines,
base-Defender pre-decision count, 510300 R40/ER40/QM40, the strict
`QM40 > 0.0075` 10-session recovery streak, and whether early or day-30 fallback recovery
triggered. Base-Defender count must not be labelled as contiguous actual
Defender or bottom-ETF holding time.

The current formal Gold-escape message ends with two deterministic performance
blocks. Same-period performance starts at the latest formal target-weight
change open and compares the current holding with a freshly entered continuous
Momentum sleeve, freshly entered continuous Defender sleeve, and every
non-held Momentum ETF. Calendar-period performance compounds month-to-date,
quarter-to-date, and year-to-date daily net returns for the complete formal
strategy, legacy simple-price Momentum, current pure Momentum, and pure
Defender. The legacy Momentum line is explicitly a clean model replay and must
not be presented as another deployment's persisted live YTD ledger. Failure of
this auxiliary snapshot renders an unavailable line and must not block the
formal signal.

Production notifications are deterministic templates. Runtime code must not
call an LLM, agent, model API, search service, or natural-language reasoning
service. All prose variants, pass/fail labels, thresholds, and gap calculations
come from fixed Python branches and numeric formulas.
