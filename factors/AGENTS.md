# Factor module contract

Each factor exposes `METADATA` and `compute(df, params=None)`, is registered in
`registry.yaml`, and returns a float Series indexed by `df["date"]`.

`ohlc_quality_momentum` implements the weighted OHLC ER formula using the
post-adjusted OHLC columns. Its parameter weights live in the independent
strategy YAML checkpoint and are validated before calculation.

`rsi` implements Wilder's recursively smoothed Relative Strength Index. The
default `window` is 14, so its first finite value requires 15 close prices;
flat windows return the neutral value 50.

`drawdown_percentile`, `rebound_percentile`, and `volume_percentile` compare
the current X-day range/volume state with a trailing historical window. Their
default `window=60, history=504` requires 563 rows and uses rolling ranks that
include the current observation without accessing future rows.
