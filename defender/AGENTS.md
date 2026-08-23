# Defender module contract

## Contract

This package contains the production implementation vendored from
`Castle47/Defender` main. The exact upstream source commit is recorded in
`UPSTREAM.md`. QuantFramework's local HFQ data layer is the only price source;
the package must not read another repository or pre-generated handoff CSV at
runtime.

The formal implementation is the listing-aware 2013 rotation strategy. It uses
510880.SH as the signal bridge before 512890.SH exists, switches to 512890.SH
from the next open after its first close, rotates the primary sleeve monthly
among listed and ranking-eligible equity ETFs, and assigns the remainder to
511260.SH or cash when the defensive ETF is unavailable.

## Integration rules

- Preserve previous-close to next-open signal timing.
- Preserve per-asset one-way transaction costs from the upstream strategy.
- Keep policy target, executable target, and cash weight distinct.
- The Momentum/Defender composite must consume the in-memory switch interface;
  production and backtest paths must not depend on external deliverable files.
- Upstream source changes require a new commit pin and parity validation.

## Pitfalls

- The Defender calendar is the union of required assets, not the calendar of a
  single anchor ETF.
- An unavailable or suspended asset can make a policy target non-executable;
  the remainder is cash and must not be silently renormalized.
- Historical promotion evidence is retrospective, not independent out-of-sample
  evidence.
