# Data module contract

The data layer owns Tushare synchronization and local Parquet storage. Price
queries expose the configured post-adjusted OHLC fields to both factors and
backtests. Offline tests must mock the Tushare client and never use local
production credentials or external network state.

`fund_share.py` provides an on-demand, point-in-time Tushare `fund_share`
reader for the read-only DingTalk peak warning. It returns observations by
their actual `trade_date` and never backdates or persists them. The warning
fetches 159915.SZ shares whenever it is the daily Momentum Top1 so the message
can show condition progress even before price and volume both pass. Missing or
stale signal-date shares suppress the warning and must not block the formal
daily signal.

## Pitfalls

- Tushare `fund_share.fund_type` can be null for valid Shanghai ETFs. When ETF
  identity is already established by `etf_basic`, do not filter point-in-time
  share snapshots with `fund_type == "ETF"`; that silently drops valid funds.
- Do not prepend older rows to a production raw-storage Parquet merely for
  research. Fixed-baseline HFQ projection would choose a new first adjustment
  factor and can change formal float64 return hashes through rounding. Keep
  pre-production history in an explicit research-only market override.
