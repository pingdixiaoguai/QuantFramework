# Factor module contract

Each factor exposes `METADATA` and `compute(df, params=None)`, is registered in
`registry.yaml`, and returns a float Series indexed by `df["date"]`.

`ohlc_quality_momentum` implements the weighted OHLC ER formula using the
post-adjusted OHLC columns. Its parameter weights live in the independent
strategy YAML checkpoint and are validated before calculation.
