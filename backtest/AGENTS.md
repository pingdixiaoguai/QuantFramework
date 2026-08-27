# Backtest module contract

The backtest runner consumes the same strategy YAML, registered factors, and
`strategy.loader.load_strategy()` path as daily execution. It must preserve
the configured `rebalance_days`, execution timing, and post-adjusted OHLC
data basis when replaying a strategy.

Tests use mocked `data.store.query` data and must not require Tushare,
credentials, or network access.

Formal Momentum/Defender reports configure the sample from 2013-01-01 through
the latest complete common close. Factor warmup may make the first executable
return later than the configured start; reports must disclose both dates and
must never backfill pre-listing ETF returns.

QuantStats 0.0.81 emits an undefined `save()` body onload handler. The standard
report adapter removes that dead handler after generation so browser QA stays
console-error free; do not remove the post-processing without upgrading and
rechecking QuantStats output.
