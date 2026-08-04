# Data module contract

The data layer owns Tushare synchronization and local Parquet storage. Price
queries expose the configured post-adjusted OHLC fields to both factors and
backtests. Offline tests must mock the Tushare client and never use local
production credentials or external network state.
