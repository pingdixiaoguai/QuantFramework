"""Research helpers for the Top1 hysteresis threshold scan."""

from __future__ import annotations

import copy
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.runner import BacktestResult
from backtest.runner import run
from data.store import query
from factors.registry import load_registered_factors
from run_backtest import _load_config_from_yaml

ROOT = Path(__file__).resolve().parents[1]
ATTACHMENTS_DIR = ROOT / "strategy_changelog_attachments"
CONFIG_PATH = ROOT / "strategy" / "configs" / "quality_momentum_top1.yaml"
COST_RATE = 0.0001
TAUS = (0.0, 0.0005, 0.001, 0.0025, 0.005, 0.0075, 0.01)
FOCUS_WINDOWS = {
    "2015-10_switch_heavy": ("2015-10-01", "2015-10-31"),
    "2020-09_switch_heavy": ("2020-09-01", "2020-09-30"),
    "2024-10_single_asset": ("2024-10-01", "2024-10-31"),
    "2025-10_switch_heavy": ("2025-10-01", "2025-10-31"),
}
CANARY_ENTRY_DATE = pd.Timestamp("2024-09-26")
CANARY_ASSET = "159915.SZ"


def forward_filled_positions(result: BacktestResult) -> pd.DataFrame:
    """Expand executed position rows onto the backtest return calendar."""
    dates = result.daily_returns.index
    if len(dates) == 0:
        return pd.DataFrame(index=dates)
    if result.positions.empty:
        return pd.DataFrame(index=dates)

    positions = result.positions.sort_index().copy().fillna(0.0)
    positions = positions.reindex(dates).ffill().fillna(0.0)
    return positions.astype(float)


def apply_transaction_costs(
    raw_returns: pd.Series,
    positions: pd.DataFrame,
    cost_rate: float,
) -> tuple[pd.Series, pd.DataFrame]:
    """Charge one-way costs on executed weight deltas."""
    adjusted = raw_returns.copy().astype(float)
    if positions.empty:
        return adjusted, pd.DataFrame(
            columns=["date", "traded_weight", "cost"]
        )

    events = positions.sort_index().fillna(0.0).astype(float)
    assets = list(events.columns)
    prior = pd.Series(0.0, index=assets, dtype=float)
    ledger_rows: list[dict[str, object]] = []

    for executed_at, row in events.iterrows():
        target = row.reindex(assets, fill_value=0.0).astype(float)
        traded_weight = float((target - prior).abs().sum())
        cost = traded_weight * cost_rate
        if executed_at in adjusted.index:
            adjusted.loc[executed_at] -= cost
        ledger_rows.append({
            "date": executed_at,
            "from_asset": _held_asset(prior),
            "to_asset": _held_asset(target),
            "traded_weight": traded_weight,
            "cost": cost,
        })
        prior = target

    return adjusted, pd.DataFrame(ledger_rows)


def extract_position_periods(
    positions: pd.DataFrame,
    returns: pd.Series,
) -> pd.DataFrame:
    """Summarize executed Top1 position rows into return periods."""
    if positions.empty or returns.empty:
        return pd.DataFrame(
            columns=[
                "asset",
                "entry_date",
                "exit_date",
                "next_entry_date",
                "holding_days",
                "pnl",
            ]
        )

    events = positions.sort_index().fillna(0.0).astype(float)
    rows: list[dict[str, object]] = []
    for idx, (entry_date, row) in enumerate(events.iterrows()):
        next_entry_date = (
            events.index[idx + 1]
            if idx + 1 < len(events.index)
            else None
        )
        segment = returns[returns.index >= entry_date]
        if next_entry_date is not None:
            segment = segment[segment.index < next_entry_date]

        rows.append({
            "asset": row.idxmax() if len(row) and row.max() > 0 else None,
            "entry_date": entry_date,
            "exit_date": segment.index[-1] if len(segment) else entry_date,
            "next_entry_date": next_entry_date,
            "holding_days": int(len(segment)),
            "pnl": float((1 + segment).prod() - 1) if len(segment) else 0.0,
        })

    return pd.DataFrame(rows)


def summarize_metrics(
    returns: pd.Series,
    trade_ledger: pd.DataFrame,
    periods: pd.DataFrame,
) -> dict[str, float | int]:
    """Return the scan panel metrics for a cost-adjusted path."""
    if returns.empty:
        return {
            "annualized_return": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "annualized_turnover": 0.0,
            "average_holding_days": 0.0,
            "switch_count": 0,
        }

    cumulative = (1 + returns).cumprod()
    years = len(returns) / 252
    annualized_return = float(cumulative.iloc[-1] ** (1 / years) - 1)
    sharpe = (
        float(returns.mean() / returns.std() * np.sqrt(252))
        if returns.std() > 0
        else 0.0
    )
    drawdown = cumulative / cumulative.cummax() - 1
    traded_weight = (
        float(trade_ledger["traded_weight"].sum())
        if "traded_weight" in trade_ledger
        else 0.0
    )
    average_holding = (
        float(periods["holding_days"].mean())
        if "holding_days" in periods and len(periods)
        else 0.0
    )

    return {
        "annualized_return": annualized_return,
        "sharpe": sharpe,
        "max_drawdown": float(drawdown.min()),
        "annualized_turnover": traded_weight / years,
        "average_holding_days": average_holding,
        "switch_count": max(int(len(periods)) - 1, 0),
    }


def write_csv(df: pd.DataFrame, path: Path) -> None:
    """Write a CSV artifact below a created parent directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _held_asset(weights: pd.Series) -> str | None:
    return str(weights.idxmax()) if len(weights) and weights.max() > 0 else None


def _copy_config(tau: float | None) -> dict:
    config = copy.deepcopy(_load_config_from_yaml(CONFIG_PATH))
    config["start"] = date(2014, 1, 1)
    config["rebalance_days"] = 5
    if tau is None:
        config.pop("hysteresis_threshold", None)
    else:
        config["hysteresis_threshold"] = tau
    return config


def _complete_asset_start(config: dict) -> pd.Timestamp:
    first_dates = []
    for asset in config["asset_pool"]:
        frame = query(asset, config["start"], config["end"])
        if frame.empty:
            raise RuntimeError(f"no local data found for {asset}")
        first_dates.append(pd.Timestamp(frame["date"].min()))
    return max(first_dates)


def _assert_zero_tau_matches_baseline(
    baseline: BacktestResult,
    zero_tau: BacktestResult,
) -> None:
    try:
        pd.testing.assert_series_equal(
            baseline.daily_returns,
            zero_tau.daily_returns,
            check_names=False,
            check_exact=True,
        )
        pd.testing.assert_frame_equal(
            baseline.positions,
            zero_tau.positions,
            check_names=False,
            check_exact=True,
        )
    except AssertionError as exc:
        raise RuntimeError("tau=0 failed plain Top1 baseline gate") from exc


def _path_tables(
    result: BacktestResult,
    evaluation_start: pd.Timestamp,
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    adjusted_all, trade_ledger = apply_transaction_costs(
        result.daily_returns,
        result.positions,
        cost_rate=COST_RATE,
    )
    adjusted = adjusted_all[adjusted_all.index >= evaluation_start]
    trades = trade_ledger[trade_ledger["date"] >= evaluation_start].reset_index(
        drop=True
    )
    periods = extract_position_periods(result.positions, adjusted)
    periods = periods[periods["holding_days"] > 0].reset_index(drop=True)
    return adjusted, trades, periods


def _attach_tau(df: pd.DataFrame, tau: float) -> pd.DataFrame:
    out = df.copy()
    out.insert(0, "tau", tau)
    return out


def _drawdown_episodes(returns: pd.Series) -> pd.DataFrame:
    if returns.empty:
        return pd.DataFrame(
            columns=["start", "trough", "recovery", "max_drawdown"]
        )

    cumulative = (1 + returns).cumprod()
    peaks = cumulative.cummax()
    drawdowns = cumulative / peaks - 1
    rows: list[dict[str, object]] = []
    active_start = None
    trough_date = None
    trough_drawdown = 0.0

    for current_date, drawdown in drawdowns.items():
        if drawdown < 0 and active_start is None:
            active_start = current_date
            trough_date = current_date
            trough_drawdown = float(drawdown)
            continue
        if active_start is None:
            continue
        if drawdown < trough_drawdown:
            trough_date = current_date
            trough_drawdown = float(drawdown)
        if drawdown >= 0:
            rows.append({
                "start": active_start,
                "trough": trough_date,
                "recovery": current_date,
                "max_drawdown": trough_drawdown,
            })
            active_start = None
            trough_date = None
            trough_drawdown = 0.0

    if active_start is not None:
        rows.append({
            "start": active_start,
            "trough": trough_date,
            "recovery": pd.NaT,
            "max_drawdown": trough_drawdown,
        })
    return pd.DataFrame(rows)


def _whipsaw_rows(periods: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for idx in range(max(len(periods) - 2, 0)):
        prior = periods.iloc[idx]
        away = periods.iloc[idx + 1]
        returned = periods.iloc[idx + 2]
        if prior["asset"] != returned["asset"]:
            continue
        rows.append({
            "from_asset": prior["asset"],
            "away_asset": away["asset"],
            "away_entry_date": away["entry_date"],
            "return_entry_date": returned["entry_date"],
            "away_holding_days": away["holding_days"],
            "whipsaw_pnl": away["pnl"],
        })
    return pd.DataFrame(rows)


def _focus_episode_rows(
    tau: float,
    returns: pd.Series,
    periods: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    entries = pd.to_datetime(periods["entry_date"]) if len(periods) else pd.Series()
    for label, (start_text, end_text) in FOCUS_WINDOWS.items():
        start = pd.Timestamp(start_text)
        end = pd.Timestamp(end_text)
        segment = returns[(returns.index >= start) & (returns.index <= end)]
        segment_entries = entries[(entries >= start) & (entries <= end)]
        cumulative = (1 + segment).cumprod()
        max_drawdown = (
            float((cumulative / cumulative.cummax() - 1).min())
            if len(segment)
            else np.nan
        )
        rows.append({
            "tau": tau,
            "episode": label,
            "start": start,
            "end": end,
            "return": float((1 + segment).prod() - 1) if len(segment) else np.nan,
            "max_drawdown": max_drawdown,
            "switches_in_window": max(int(len(segment_entries)), 0),
        })
    return pd.DataFrame(rows)


def _compounded_between(
    returns: pd.Series,
    start: pd.Timestamp,
    end: pd.Timestamp | None,
) -> float:
    segment = returns[returns.index >= start]
    if end is not None and not pd.isna(end):
        segment = segment[segment.index < end]
    return float((1 + segment).prod() - 1) if len(segment) else 0.0


def _trading_delay(
    calendar: pd.DatetimeIndex,
    start: pd.Timestamp,
    delayed_entry: pd.Timestamp | None,
    end: pd.Timestamp | None,
) -> int:
    window = calendar[calendar >= start]
    if delayed_entry is not None and not pd.isna(delayed_entry):
        return int((window < delayed_entry).sum())
    if end is not None and not pd.isna(end):
        return int((window < end).sum())
    return int(len(window))


def _delayed_switch_rows(
    tau: float,
    baseline_periods: pd.DataFrame,
    baseline_returns: pd.Series,
    tau_periods: pd.DataFrame,
    tau_returns: pd.Series,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if len(baseline_periods) < 2:
        return pd.DataFrame()

    tau_entries = tau_periods.copy()
    tau_entries["entry_date"] = pd.to_datetime(tau_entries["entry_date"])
    for idx in range(1, len(baseline_periods)):
        baseline_period = baseline_periods.iloc[idx]
        start = pd.Timestamp(baseline_period["entry_date"])
        next_entry = baseline_period["next_entry_date"]
        end = pd.Timestamp(next_entry) if not pd.isna(next_entry) else None
        target = baseline_period["asset"]
        candidates = tau_entries[
            (tau_entries["asset"] == target)
            & (tau_entries["entry_date"] >= start)
        ]
        if end is not None:
            candidates = candidates[candidates["entry_date"] < end]
        delayed_entry = (
            pd.Timestamp(candidates.iloc[0]["entry_date"])
            if len(candidates)
            else None
        )
        delay_days = _trading_delay(
            baseline_returns.index,
            start,
            delayed_entry,
            end,
        )
        if delay_days == 0:
            continue

        baseline_pnl = _compounded_between(baseline_returns, start, end)
        tau_pnl = _compounded_between(tau_returns, start, end)
        rows.append({
            "tau": tau,
            "baseline_entry_date": start,
            "baseline_next_entry_date": end,
            "from_asset": baseline_periods.iloc[idx - 1]["asset"],
            "target_asset": target,
            "tau_entry_date": delayed_entry,
            "status": "delayed" if delayed_entry is not None else "blocked",
            "delay_trading_days": delay_days,
            "baseline_window_pnl": baseline_pnl,
            "tau_window_pnl": tau_pnl,
            "missed_pnl": baseline_pnl - tau_pnl,
            "verdict": (
                "wrong_missed_trend"
                if baseline_pnl > tau_pnl
                else "correct_avoided_loss"
            ),
        })
    return pd.DataFrame(rows)


def _score_regime_rows(config: dict) -> pd.DataFrame:
    factor_config = config["factors"][0]
    factor_name = factor_config["name"]
    factor = load_registered_factors()[factor_name]
    rows: list[dict[str, object]] = []
    for asset in config["asset_pool"]:
        frame = query(asset, config["start"], config["end"])
        scores = factor["compute"](frame.copy(), factor_config.get("params"))
        score_frame = pd.DataFrame({
            "date": pd.to_datetime(scores.index),
            "asset": asset,
            "score": scores.values,
        }).dropna()
        rows.extend(score_frame.to_dict("records"))

    all_scores = pd.DataFrame(rows)
    stats: list[dict[str, object]] = []
    for label, (start_text, end_text) in {
        **FOCUS_WINDOWS,
        "2024-09_canary": ("2024-09-01", "2024-09-30"),
    }.items():
        segment = all_scores[
            (all_scores["date"] >= pd.Timestamp(start_text))
            & (all_scores["date"] <= pd.Timestamp(end_text))
        ]
        abs_scores = segment["score"].abs()
        stats.append({
            "regime": label,
            "rows": int(len(segment)),
            "min_score": float(segment["score"].min()) if len(segment) else np.nan,
            "median_score": (
                float(segment["score"].median()) if len(segment) else np.nan
            ),
            "max_score": float(segment["score"].max()) if len(segment) else np.nan,
            "median_abs_score": float(abs_scores.median()) if len(segment) else np.nan,
            "p90_abs_score": (
                float(abs_scores.quantile(0.9)) if len(segment) else np.nan
            ),
        })
    return pd.DataFrame(stats)


def _markdown_value(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (float, np.floating)):
        return f"{value:.6g}"
    if isinstance(value, (pd.Timestamp, date)):
        return pd.Timestamp(value).date().isoformat()
    return str(value)


def _markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    columns = [str(column) for column in df.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in df.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(_markdown_value(value) for value in row) + " |")
    return "\n".join(lines)


def _write_attribution_markdown(
    run_date: str,
    metrics: pd.DataFrame,
    drawdowns: pd.DataFrame,
    whipsaws: pd.DataFrame,
) -> None:
    top_drawdowns = drawdowns.nsmallest(10, "max_drawdown")
    body = f"""# 2026-05-21 Attribution Reconstruction

This is a reconstructed artifact generated on {run_date} from the local data
available in this repository worktree. The original 2026-05-21 Markdown and
raw CSV attribution outputs were not found in the repository, so these files
preserve the rerunnable evidence used by the hysteresis scan instead of claiming
to be the original files.

## Scope

- Plain `quality_momentum_top1` path.
- `rebalance_days=5`.
- Evaluation begins after the configured asset pool is complete.
- Cost-adjusted returns charge 0.01% per one-way executed weight delta.
- Whipsaw rows use an executable reconstruction rule: an asset is left and
  returned to on the second executed switch, and the intervening holding period
  P&L is the whipsaw P&L.

## Metrics

{_markdown_table(metrics)}

## Whipsaw Summary

| whipsaw_count | cumulative_whipsaw_pnl |
| --- | --- |
| {len(whipsaws)} | {whipsaws["whipsaw_pnl"].sum() if len(whipsaws) else 0.0:.6g} |

## Largest Drawdown Episodes

{_markdown_table(top_drawdowns)}

Raw CSV companions store metrics, trade ledger, position periods, drawdown
episodes, and whipsaw rows under `strategy_changelog_attachments/`.
"""
    ATTACHMENTS_DIR.mkdir(parents=True, exist_ok=True)
    (ATTACHMENTS_DIR / "2026-05-21_attribution_reconstruction.md").write_text(
        body,
        encoding="utf-8",
    )


def _write_scan_markdown(
    run_date: str,
    panel: pd.DataFrame,
    whipsaw_summary: pd.DataFrame,
    focus_rows: pd.DataFrame,
    delayed_rows: pd.DataFrame,
    canary_rows: pd.DataFrame,
    score_rows: pd.DataFrame,
) -> Path:
    verdict_counts = (
        delayed_rows.groupby(["tau", "verdict"], dropna=False)
        .size()
        .reset_index(name="count")
        if len(delayed_rows)
        else pd.DataFrame(columns=["tau", "verdict", "count"])
    )
    body = f"""# Hysteresis Threshold Scan

Generated on {run_date}. This is research evidence for a post-2026-06-02
decision. It does not deploy a threshold and it does not modify the production
`quality_momentum_top1.yaml`.

## Fixed Method

- Evaluation start: 2014-01-01 request, trimmed until the full asset pool and
  factor-produced strategy returns are available.
- Cost: 0.01% per one-way executed weight delta.
- `rebalance_days=5` for every run.
- Independent complete runs for `tau in {{0, 0.0005, 0.001, 0.0025, 0.005, 0.0075, 0.01}}`.
- Gate: `tau=0` raw returns and executed position rows exactly matched the
  same-mouth plain Top1 baseline before the scan.

## Standard Panel

{_markdown_table(panel)}

## Surface Observations

{_surface_observations(panel, whipsaw_summary, focus_rows, canary_rows)}

## Turnover-Side Evidence

Whipsaw rows use the reconstructed rule documented in the attribution archive:
leave an asset and return to it on the second executed switch; the intervening
holding P&L is the whipsaw P&L.

{_markdown_table(whipsaw_summary)}

The focus episodes separate switch-heavy drawdown months from the 2024-10
single-asset month so threshold gains do not get credited to the wrong
mechanism. The requested 2015-10 row remains in raw output, but this local
reconstruction has no 2015-10 values because the Parquet pool available here
starts in 2016.

{_markdown_table(focus_rows)}

## Stickiness Cost

Suppressed or delayed plain-baseline switches are judged over the baseline
target holding window. `wrong_missed_trend` means the baseline target window
outperformed the threshold path over that same window; `correct_avoided_loss`
means the sticky path did not lose that comparison.

{_markdown_table(verdict_counts)}

### 2024-09-26 Canary

{_markdown_table(canary_rows)}

## Score Scale Samples

`tau` is in the same score units as `quality_momentum = momentum * ER`.
These regime summaries show the observed local score scale.

{_markdown_table(score_rows)}

## Raw Files

CSV companions store the full panel, trade ledgers, position periods, whipsaw
rows, focus episodes, delayed-switch comparisons, canary rows, and score regime
samples. The surface should be interpreted as a trade-off curve, not as a
deployment selector.
"""
    path = ATTACHMENTS_DIR / f"{run_date}_hysteresis_scan.md"
    path.write_text(body, encoding="utf-8")
    return path


def _surface_observations(
    panel: pd.DataFrame,
    whipsaw_summary: pd.DataFrame,
    focus_rows: pd.DataFrame,
    canary_rows: pd.DataFrame,
) -> str:
    baseline = panel.loc[panel["tau"] == 0].iloc[0]
    highest_tau = panel.loc[panel["tau"].idxmax()]
    baseline_whipsaw = whipsaw_summary.loc[whipsaw_summary["tau"] == 0].iloc[0]
    highest_whipsaw = whipsaw_summary.loc[
        whipsaw_summary["tau"] == highest_tau["tau"]
    ].iloc[0]
    dd_worse = panel[
        panel["max_drawdown"] < baseline["max_drawdown"] - 1e-12
    ]
    dd_message = (
        f"Maximum drawdown first worsens at `tau={dd_worse.iloc[0]['tau']:.4g}`."
        if len(dd_worse)
        else "Maximum drawdown does not worsen inside the scanned thresholds."
    )
    canary_message = (
        f"The 2024-09-26 `159915.SZ` canary first delays at "
        f"`tau={canary_rows['tau'].min():.4g}`."
        if len(canary_rows)
        else "No scanned threshold delayed the 2024-09-26 `159915.SZ` canary."
    )
    crash_rows = focus_rows[focus_rows["episode"] == "2024-10_single_asset"]
    crash_unchanged = (
        crash_rows["return"].nunique(dropna=True) == 1
        and crash_rows["max_drawdown"].nunique(dropna=True) == 1
    )
    crash_message = (
        "The 2024-10 single-asset row is unchanged across the scan."
        if crash_unchanged
        else "The 2024-10 single-asset row changes across the scan."
    )
    return "\n".join([
        f"- Turnover falls from {baseline['annualized_turnover']:.4g} at `tau=0` "
        f"to {highest_tau['annualized_turnover']:.4g} at `tau={highest_tau['tau']:.4g}`.",
        f"- Whipsaw count falls from {int(baseline_whipsaw['whipsaw_count'])} "
        f"to {int(highest_whipsaw['whipsaw_count'])}, but cumulative whipsaw "
        "P&L is not monotonic across the curve.",
        f"- {dd_message} Full-period return and Sharpe are also non-monotonic, "
        "so a high headline metric alone is not a deployment rule.",
        f"- {crash_message} {canary_message}",
    ])


def main() -> None:
    run_date = date.today().isoformat()
    baseline_config = _copy_config(None)
    baseline_result = run(baseline_config)
    zero_result = run(_copy_config(0.0))
    _assert_zero_tau_matches_baseline(baseline_result, zero_result)

    evaluation_start = max(
        _complete_asset_start(baseline_config),
        pd.Timestamp(baseline_result.daily_returns.index.min()),
    )
    baseline_returns, baseline_trades, baseline_periods = _path_tables(
        baseline_result,
        evaluation_start,
    )
    attribution_metrics = pd.DataFrame([
        {
            "evaluation_start": evaluation_start,
            **summarize_metrics(
                baseline_returns,
                baseline_trades,
                baseline_periods,
            ),
        }
    ])
    attribution_drawdowns = _drawdown_episodes(baseline_returns)
    attribution_whipsaws = _whipsaw_rows(baseline_periods)

    write_csv(
        attribution_metrics,
        ATTACHMENTS_DIR / "2026-05-21_attribution_metrics.csv",
    )
    write_csv(
        baseline_trades,
        ATTACHMENTS_DIR / "2026-05-21_attribution_trade_ledger.csv",
    )
    write_csv(
        baseline_periods,
        ATTACHMENTS_DIR / "2026-05-21_attribution_position_periods.csv",
    )
    write_csv(
        attribution_drawdowns,
        ATTACHMENTS_DIR / "2026-05-21_attribution_drawdown_episodes.csv",
    )
    write_csv(
        attribution_whipsaws,
        ATTACHMENTS_DIR / "2026-05-21_attribution_whipsaws.csv",
    )
    _write_attribution_markdown(
        run_date,
        attribution_metrics,
        attribution_drawdowns,
        attribution_whipsaws,
    )

    panel_rows: list[dict[str, object]] = []
    all_trades = []
    all_periods = []
    all_whipsaws = []
    all_focus = []
    all_delayed = []

    for tau in TAUS:
        result = zero_result if tau == 0 else run(_copy_config(tau))
        returns, trades, periods = _path_tables(result, evaluation_start)
        panel_rows.append({
            "tau": tau,
            "evaluation_start": evaluation_start,
            **summarize_metrics(returns, trades, periods),
        })
        all_trades.append(_attach_tau(trades, tau))
        all_periods.append(_attach_tau(periods, tau))
        whipsaws = _attach_tau(_whipsaw_rows(periods), tau)
        all_whipsaws.append(whipsaws)
        all_focus.append(_focus_episode_rows(tau, returns, periods))
        all_delayed.append(_delayed_switch_rows(
            tau,
            baseline_periods,
            baseline_returns,
            periods,
            returns,
        ))

    panel = pd.DataFrame(panel_rows)
    trades = pd.concat(all_trades, ignore_index=True)
    periods = pd.concat(all_periods, ignore_index=True)
    whipsaws = pd.concat(all_whipsaws, ignore_index=True)
    focus_rows = pd.concat(all_focus, ignore_index=True)
    delayed_rows = pd.concat(all_delayed, ignore_index=True)
    score_rows = _score_regime_rows(baseline_config)
    whipsaw_summary = (
        whipsaws.groupby("tau", dropna=False)
        .agg(
            whipsaw_count=("whipsaw_pnl", "size"),
            cumulative_whipsaw_pnl=("whipsaw_pnl", "sum"),
        )
        .reset_index()
    )
    canary_rows = delayed_rows[
        (delayed_rows["baseline_entry_date"] == CANARY_ENTRY_DATE)
        & (delayed_rows["target_asset"] == CANARY_ASSET)
    ].copy()

    prefix = ATTACHMENTS_DIR / f"{run_date}_hysteresis"
    write_csv(panel, prefix.with_name(f"{prefix.name}_panel.csv"))
    write_csv(trades, prefix.with_name(f"{prefix.name}_trade_ledgers.csv"))
    write_csv(periods, prefix.with_name(f"{prefix.name}_position_periods.csv"))
    write_csv(whipsaws, prefix.with_name(f"{prefix.name}_whipsaws.csv"))
    write_csv(focus_rows, prefix.with_name(f"{prefix.name}_focus_episodes.csv"))
    write_csv(delayed_rows, prefix.with_name(f"{prefix.name}_delayed_switches.csv"))
    write_csv(canary_rows, prefix.with_name(f"{prefix.name}_canary.csv"))
    write_csv(score_rows, prefix.with_name(f"{prefix.name}_score_regimes.csv"))
    markdown_path = _write_scan_markdown(
        run_date,
        panel,
        whipsaw_summary,
        focus_rows,
        delayed_rows,
        canary_rows,
        score_rows,
    )
    print(f"tau=0 baseline gate passed at evaluation start {evaluation_start.date()}")
    print(f"wrote {markdown_path}")


if __name__ == "__main__":
    main()
