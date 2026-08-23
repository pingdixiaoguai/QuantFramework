# Backtest module contract

The backtest runner consumes the same strategy YAML, registered factors, and
`strategy.loader.load_strategy()` path as daily execution. It must preserve
the configured `rebalance_days`, execution timing, and post-adjusted OHLC
data basis when replaying a strategy.

Tests use mocked `data.store.query` data and must not require Tushare,
credentials, or network access.

QuantStats 0.0.81 emits an undefined `save()` body onload handler. The standard
report adapter removes that dead handler after generation so browser QA stays
console-error free; do not remove the post-processing without upgrading and
rechecking QuantStats output.
