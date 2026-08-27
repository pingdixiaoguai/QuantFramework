# Defender module contract

## Contract

This package contains the promoted local full-equity dividend sleeve and the
prior implementation vendored from `Castle47/Defender` main. The rollback
source commit is recorded in `UPSTREAM.md`. QuantFramework's local HFQ data
layer is the only price source; the package must not read another repository
or pre-generated handoff CSV at runtime.

The current formal composite sleeve is
`dividend_w40_qm_reversal_full_equity_v3`: at each month boundary it selects the
listed and executable dividend ETF with the lowest prior-close signed log/log
QM40 from 512890, 513530, 515080, 510880, 515450, and 513630 and holds it at
100%. It never targets 511260 or cash. The prior v2 pure-return formal
implementation remains the listing-aware 2013 rotation rollback strategy. It uses
510880.SH as the signal bridge before 512890.SH exists, switches to 512890.SH
from the next open after its first close, rotates the primary sleeve monthly
among listed and ranking-eligible equity ETFs, and assigns the remainder to
511260.SH or cash when the defensive ETF is unavailable.

The current top-level production composite is
`momentum_defender_w40_qm40_threshold_v5`. Its Gold escape and immediate entry
veto are strategy-layer
overlays and do not modify this Defender sleeve: the continuous Defender NAV
is still computed from the same 100% monthly dividend policy, including while
the executable composite temporarily holds Gold.

Formal reports configure history from 2013-01-01. Before any candidate owns a
40-session score, 510880.SH is the only warmup holding; later ETFs participate
only after their actual listing and score history are available.

## Integration rules

- Preserve previous-close to next-open signal timing.
- Preserve per-asset one-way transaction costs from the upstream strategy.
- Keep policy target, executable target, and cash weight distinct.
- The Momentum/Defender composite must consume the in-memory switch interface;
  production and backtest paths must not depend on external deliverable files.
- Upstream source changes require a new commit pin and parity validation.
- The promoted full-equity sleeve has no persistent grid or champion state;
  live targets replay monthly selection from local history and must not read
  the rollback Defender's state or target weights.
- Gold escape state belongs to `strategy/`; do not add Gold, X/Y thresholds, or
  five-day hold state to the Defender package.

## Pitfalls

- The Defender calendar is the union of required assets, not the calendar of a
  single anchor ETF.
- An unavailable or suspended asset can make a policy target non-executable;
  the remainder is cash and must not be silently renormalized.
- Historical promotion evidence is retrospective, not independent out-of-sample
  evidence.
