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
- `three_factor_trend_top1_research.yaml` is an independent research checkpoint
  using fixed 20-day momentum plus soft path-quality and relative-volatility
  adjustments. It is not part of the production daily run.
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
- `momentum_defender_w40_gold_escape.yaml` is the current formal production
  signal, versioned as `momentum_defender_w40_qm40_threshold_v5`. It uses
  510300 W40 downside loss with strict-lag 756-session history and 0.60/0.35
  hysteresis. Momentum retains its 30-session lock. Defender selects the
  lowest signed log/log QM40 dividend ETF monthly at 100%; after five base-W40
  Defender sessions, ten consecutive 510300 QM40 observations strictly above
  0.0075 can restore Momentum early, while day 30 and W40 <=0.35 remain the fallback.
  Base Defender state and counts continue under Gold. Gold retains v3's
  0.005/-0.020 thresholds, five-session hold, and immediate entry veto.
- The archived v3 configuration in its formal experiment is the direct
  rollback. It is versioned as `momentum_defender_w40_gold_qm20_escape_v3` and keeps
  the W40 0.55/0.40, 30/30-lock state and the monthly 100%
  lowest-40-session-return dividend sleeve. The v2 dividend universe is fixed
  to 512890, 513530, 515080, 510880, 515450, and 513630. After five actual Defender
  sessions, only a current Gold Top1 may break the Defender lock when
  `QM20(Gold)-QM20(Defender)>0.005`. Gold is then hard-held for five sessions;
  while base W40 remains Defender it returns when the Gold gap is below -0.020
  or the current Top1 is no longer Gold; otherwise it continues Gold. The
  v3 module remains explicitly dispatchable for rollback. Its next-open signal exposes both the
  previous executed candidate and the new target candidate so notifications
  can say "continue holding" or show a model transition without relying on a
  workstation position file. It also carries an optional deterministic
  notification performance snapshot built from the already replayed formal
  path; snapshot failure is isolated from target generation.
- The archived v4 configuration in
  `experiments/20260826_momentum_defender_w40_qm40_signed_exit_v4_formal/`
  is the direct rollback. It differs from v5 only by using the natural strict
  `QM40 > 0` recovery threshold.
- Formal composite research and reports always configure the interval from
  2013-01-01 through the latest complete common close. The first executable
  return follows factor warmup; ETFs enter the cross-section only after listing.
- On the exact open where base W40 enters Defender, v5 retains v3's immediate Gold
  instead when Gold is Top1 and the same QM20 entry gap is already above 0.005.
  Base W40 still enters Defender and starts its lock; only the executable target
  is vetoed. The ordinary five-session Defender eligibility remains unchanged
  after returning from Gold.
- `momentum_defender_w40_full_equity.yaml` is the direct rollback checkpoint,
  versioned as `momentum_defender_w40_reversal_full_equity_v2`. It has the same
  W40 and 100% dividend sleeve but no Gold escape overlay.
- `momentum_defender_w40_loss.yaml` is the superseded direct rollback
  checkpoint, versioned as `momentum_defender_w40_loss_excluding_extremes_v1`.
  It fixes
  log/log `quality_momentum` v2, the listing-aware Defender, and one 510300
  W40 downside-log-loss percentile state. Rolling-504 strict-lag percentiles
  use 0.55/0.40 entry/recovery, 1/1 confirmations, and immutable 30/30 locks.
  Path efficiency, volatility adjustment, floor, clip, Gold, emergency, and
  every lock bypass are disabled.
- `momentum_defender_downside_raqm.yaml` is the superseded weighted-DRAQM
  rollback checkpoint and is never selected implicitly.
- `momentum_defender_c2_gold_raqm_w5.yaml` is the superseded v4 rollback
  checkpoint and is never selected implicitly. Its rollback Momentum fixes
  `quality_momentum` version 2 to log-return momentum times
  log-path ER. The base state observes both CSI300 and the held Momentum ETF's
  120-day log trends but has no Momentum or Defender minimum holding period.
  Risk-off evidence requires 20 consecutive observations; recovery requires
  10, and opposite evidence resets the streak. A Top1-versus-Defender raw
  RAQM5 bridge uses 2.0/0.75 entry/exit hysteresis with no holding lock and does
  not mutate base confirmation state. A held-asset emergency exits immediately
  when five-day log momentum is negative and 20-day downside volatility exceeds
  its strict-lag expanding q95. Gold uses raw five-day RAQM with no volatility
  floor or winsorization, entry 2.0, exit 0.75, and a five-day hard hold. The
  v4 and v3 governance remain rollback evidence; the old config must be passed
  explicitly to the composite runner.
- Formal `vs original Momentum` reports use the independent
  `quality_momentum_top1_legacy_simple_price.yaml` baseline. That report-only
  strategy freezes simple-return MOM times price-path ER; it must not alter the
  current formal sleeve, which continues to use log/log `quality_momentum` v2.

## Known deviations and pitfalls

- Factor values are raw snapshots; cross-sectional ranking remains inside the
  strategy implementation.
- The OHLC factor uses the data layer's post-adjusted O/H/L/C columns. Do not
  mix raw and adjusted fields in one calculation.
- Shadow targets must never enter execution diff, hold filtering, position
  persistence, or production state caches.
- The default Momentum/Defender runner deterministically replays the single
  W40 loss-percentile state, monthly full-equity dividend selection, and
  Gold-only QM20 escape through the latest close before advancing exactly one
  next-open step. It must not read external Defender CSVs, legacy grid state,
  or old v4 state.
- The current formal strategy has the frozen base-Defender QM40 recovery and
  the Gold-only QM20 escape above. It has no rapid bridge, emergency state,
  raw RAQM Gold state, or Gold-style bypass for the other three Momentum ETFs.
- `research.momentum_defender_w40_asset_specific_escape` exposes the immediate
  Gold veto as an explicit switch. Production v3 enables it; the archived v2
  formal report preserves the disabled-switch rollback hash.
- Every promoted research strategy must complete and preserve the evidence in
  `research/DEVELOPMENT_VALIDATION.md` before its formal config is changed.
- The formal v5 W40 parameters, QM40 full-equity Defender, signed-QM40 recovery,
  and Gold X/Y escape were
  selected retrospectively and promoted by explicit user decisions. Do not
  re-fit the 756-session history, 0.60/0.35 lines, 0.0075 threshold,
  5/10/30 recovery rule,
  monthly QM40 window, 100% equity weight, Gold-only asset scope,
  0.005/-0.020 thresholds, or five-session Gold eligibility/hold on the same
  history. Future evaluation belongs to the new strategy's prospective ledger.
- The W40 recovery score has a point mass at zero: every non-negative 40-day
  anchor return is clipped to zero downside loss. It therefore cannot rank a
  weak recovery against a strong recovery. The 2019-start exit audit found
  154 recovery observations blocked inside the Defender lock and confirmed
  that 30 sessions is a historical local peak, while no short-lock,
  confirmation, signed-return, or relative-QM early exit improved both return
  and Sharpe. The user subsequently promoted the frozen QM40 5/10/30 rule as
  part of v4; do not tune it further on the same history.
- A later 2019-start absolute-QM40 threshold scan found a retrospective plateau
  at 0.005-0.010 and selected 0.0075 as its robust center. Bootstrap and
  Reality Check did not support automatic promotion, but the user explicitly
  promoted 0.0075 as v5. Do not continue tuning inside the same plateau.
- A research configuration must be passed explicitly; its presence under
  `strategy/configs/` does not authorize changing the production config.
