# Strategy module contract

## Contract

Strategy YAML is the shared source of truth for live and backtest runs. The
loader reads `strategy_class`; strategies consume the factor snapshot
`asset -> factor -> value` and return target weights.

## Current implementation

- `Top1` is the production decision rule for the 4ETF quality-momentum pool.
- `CompositeTop1` is an opt-in Top-1 rule that combines configured
  cross-sectional factor ranks. `direction_flip` reverses a factor rank before
  its configured weight is applied. A factor may declare per-asset
  `asset_weights`; use `center_rank: true` with unequal weights so the weights
  change factor sensitivity without adding an asset-specific score intercept.
  `score_mode: centered_value` adds a normalized raw factor value instead of a
  cross-sectional rank and requires explicit `center` and `scale` in YAML.
- `quality_momentum_top1_ohlc_er.yaml` is an independent, read-only shadow
  configuration using the registered `ohlc_quality_momentum` factor.
- Its `parameter_checkpoint` records effective date, training range, history,
  and the post-adjusted OHLC data basis. Daily execution reads this checkpoint
  only when the file is explicitly passed as `--shadow-config`; quarterly
  search belongs to research tooling.
- `rolling_ohlc_er.py` is research/training code and must not be imported by
  the daily execution path.
- `momentum_defender_c2_main.yaml` is the integrated full-history composite.
  It uses the dedicated `run_daily_momentum_defender.py` entry point because
  the C2 state machine consumes time-series sleeve state and cannot be reduced
  to the ordinary single-snapshot `generate_weights()` interface.
- `momentum_defender_c2_gold_raqm_w5.yaml` is the current formal production
  signal. It layers the frozen five-day registered risk-adjusted-quality-
  momentum Gold override on the integrated C2 replay. The generic daily runner
  defaults to this config; the base C2 config remains an explicit rollback.

## Known deviations and pitfalls

- Factor values are raw snapshots; cross-sectional ranking remains inside the
  strategy implementation.
- The OHLC factor uses the data layer's post-adjusted O/H/L/C columns. Do not
  mix raw and adjusted fields in one calculation.
- Shadow targets must never enter execution diff, hold filtering, position
  persistence, or production state caches.
- The Momentum/Defender runner deterministically replays its 30-day sleeve
  state and the Momentum sleeve's 5-day state through the latest close before
  advancing exactly one next-open step. It must not read external Defender CSVs.
- The formal Gold override must hard-hold Gold for five complete sessions.
  During that interval it overrides a base-C2 Momentum recovery; from the sixth
  open onward base Momentum takes precedence. Entry and exit thresholds are
  immutable production parameters (2.20 and 0.60).
- Every promoted research strategy must complete and preserve the evidence in
  `research/DEVELOPMENT_VALIDATION.md` before its formal config is changed.
- A research configuration must be passed explicitly; its presence under
  `strategy/configs/` does not authorize changing the production config.
