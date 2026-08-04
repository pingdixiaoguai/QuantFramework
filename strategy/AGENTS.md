# Strategy module contract

## Contract

Strategy YAML is the shared source of truth for live and backtest runs. The
loader reads `strategy_class`; strategies consume the factor snapshot
`asset -> factor -> value` and return target weights.

## Current implementation

- `Top1` is the production decision rule for the 4ETF quality-momentum pool.
- `quality_momentum_top1_ohlc_er.yaml` is an independent, read-only shadow
  configuration using the registered `ohlc_quality_momentum` factor.
- Its `parameter_checkpoint` records effective date, training range, history,
  and the post-adjusted OHLC data basis. Daily execution reads this checkpoint
  only when the file is explicitly passed as `--shadow-config`; quarterly
  search belongs to research tooling.
- `rolling_ohlc_er.py` is research/training code and must not be imported by
  the daily execution path.

## Known deviations and pitfalls

- Factor values are raw snapshots; cross-sectional ranking remains inside the
  strategy implementation.
- The OHLC factor uses the data layer's post-adjusted O/H/L/C columns. Do not
  mix raw and adjusted fields in one calculation.
- Shadow targets must never enter execution diff, hold filtering, position
  persistence, or production state caches.
