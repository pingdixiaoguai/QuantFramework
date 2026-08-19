# Factor module contract

Each factor exposes `METADATA` and `compute(df, params=None)`, is registered in
`registry.yaml`, and returns a float Series indexed by `df["date"]`.

`ohlc_quality_momentum` implements the weighted OHLC ER formula using the
post-adjusted OHLC columns. Its parameter weights live in the independent
strategy YAML checkpoint and are validated before calculation.

`rsi` implements Wilder's recursively smoothed Relative Strength Index. The
default `window` is 14, so its first finite value requires 15 close prices;
flat windows return the neutral value 50.
