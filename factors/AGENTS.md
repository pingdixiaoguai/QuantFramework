# Factor module contract

Each factor exposes `METADATA` and `compute(df, params=None)`, is registered in
`registry.yaml`, and returns a float Series indexed by `df["date"]`.

`three_factor_trend` is a research-only candidate with signed 20-day momentum
fixed as its base. It softly combines ER path efficiency, short-window log
price linearity, and a strictly lagged relative-volatility multiplier. Its
defaults are the train-only plateau representative from the broad search;
it remains separate from the production `quality_momentum` factor.

`quality_momentum` version 2 fixes both its momentum term and Kaufman
efficiency ratio in log-price space. The factor's calculation convention is a
frozen input to subsequent Momentum/Defender switching research and must not
be included in switching-parameter searches. `risk_adjusted_quality_momentum`
also remains log-return based throughout.

`legacy_quality_momentum` freezes the pre-v2 baseline formula (simple-return
momentum times price-path Kaufman ER). It exists only so formal reports can
compare against the genuine historical Momentum strategy; production signals
must continue to use `quality_momentum` version 2.

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

## Known deviations and pitfalls

- The generic validator currently applies the static `METADATA.min_history`
  for a factor's default parameters. Increasing a YAML lookback beyond that
  default can therefore fail validation even when `compute()` is causal and
  correct. Do not treat a larger configured window as supported by the generic
  runner until dynamic minimum-history resolution is implemented. The current
  production `quality_momentum(window=20)` is unaffected; robustness research
  that varies this window must prove exact parity at 20 and manage warmup
  explicitly.
