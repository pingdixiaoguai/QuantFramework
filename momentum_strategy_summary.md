# Momentum Strategy Summary

## Current Baseline

- Strategy: `quality_momentum_top1`
- Config: `strategy/configs/quality_momentum_top1.yaml`
- Score: 20-day momentum multiplied by Kaufman efficiency ratio.
- Decision rule: Top1 all-in selection with `rebalance_days=5`; no hysteresis threshold is deployed.
- Effective diagnostic start: 2014-01-01 onward, after the 2013 asset-pool completion issue recorded in `strategy_changelog.md`.

## ERC Interface Inputs

| Input | Current value | Basis | Status |
|-------|---------------|-------|--------|
| Annualized volatility | 25.98% | Current strategy daily returns from 2014-01-02 through 2026-05-19, `daily std * sqrt(252)` | Ready |
| Correlation with other sleeves | 待补 | Requires daily return series from the asset-allocation Project sleeves / functional layers | Blocked on cross-Project inputs; do not substitute correlations among the four underlying ETFs |

The clean strategy daily-return export for the ERC handoff is `strategy_changelog_attachments/2026-05-21_momentum_strategy_daily_returns.csv`.

## Diagnostic Attachments

`strategy_changelog_attachments/` is the raw diagnostic archive for strategy notes that do not belong in `strategy_changelog.md`.

- Return / drawdown attribution report: `2026-05-21_drawdown_return_attribution.md`
- Return attribution raw table: `2026-05-21_drawdown_return_attribution_returns_by_asset.csv`
- Drawdown episode raw table: `2026-05-21_drawdown_return_attribution_drawdown_episodes.csv`
- Switch holding-period P&L raw table: `2026-05-21_drawdown_return_attribution_switch_pnl.csv`
- Daily position alignment archive: `2026-05-21_quality_momentum_top1_daily_positions.csv`
- Shape-signal diagnostic report: `2026-05-20_shape_signal_diagnostic.md`
- Shape-signal raw grids: `2026-05-20_shape_signal_diagnostic_*.csv`
