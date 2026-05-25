"""Diagnostics for Top1 return, drawdown, and switch attribution."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import date
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backtest.runner import BacktestResult, run
from run_backtest import _load_config_from_yaml


CONFIG_PATH = REPO_ROOT / "strategy" / "configs" / "quality_momentum_top1.yaml"
OUTPUT_DIR = REPO_ROOT / "strategy_changelog_attachments"
RUN_DATE = "2026-05-21"
START = pd.Timestamp(date(2014, 1, 1))
DRAW_DOWN_THRESHOLD = -0.15

REPORT_PATH = OUTPUT_DIR / f"{RUN_DATE}_drawdown_return_attribution.md"
RETURN_ATTRIBUTION_PATH = OUTPUT_DIR / f"{RUN_DATE}_drawdown_return_attribution_returns_by_asset.csv"
DRAWDOWN_PATH = OUTPUT_DIR / f"{RUN_DATE}_drawdown_return_attribution_drawdown_episodes.csv"
SWITCH_PATH = OUTPUT_DIR / f"{RUN_DATE}_drawdown_return_attribution_switch_pnl.csv"
DAILY_RETURN_PATH = OUTPUT_DIR / f"{RUN_DATE}_momentum_strategy_daily_returns.csv"
DAILY_POSITION_PATH = OUTPUT_DIR / f"{RUN_DATE}_quality_momentum_top1_daily_positions.csv"


@dataclass(frozen=True)
class DiagnosticBundle:
    result: BacktestResult
    returns: pd.Series
    daily_positions: pd.DataFrame
    held_assets: pd.Series
    attribution: pd.DataFrame
    drawdowns: pd.DataFrame
    switches: pd.DataFrame
    annualized_volatility: float
    elapsed_seconds: float


def _fmt_date(value: object) -> str:
    if pd.isna(value):
        return ""
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _fmt_pct(value: object) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.2%}"


def _load_result() -> BacktestResult:
    config = _load_config_from_yaml(CONFIG_PATH)
    return run(config)


def _daily_positions(result: BacktestResult) -> pd.DataFrame:
    if result.positions.empty:
        raise RuntimeError("Backtest result has no positions to attribute.")

    sparse = result.positions.sort_index().fillna(0.0)
    daily = sparse.reindex(result.daily_returns.index).ffill()
    daily = daily.dropna(how="all")
    if daily.empty:
        raise RuntimeError("Forward-filled positions are empty after return alignment.")
    return daily


def _held_assets(daily_positions: pd.DataFrame) -> pd.Series:
    asset = daily_positions.fillna(0.0).idxmax(axis=1)
    asset.name = "held_asset"
    return asset


def _return_attribution(returns: pd.Series, held_assets: pd.Series) -> pd.DataFrame:
    aligned = pd.DataFrame({"daily_return": returns, "asset": held_assets}).dropna()
    grouped = (
        aligned.groupby("asset", observed=False)["daily_return"]
        .agg(trading_days="size", summed_daily_return="sum")
        .reset_index()
    )
    total = grouped["summed_daily_return"].sum()
    grouped["share_of_summed_return"] = (
        grouped["summed_daily_return"] / total if total != 0 else np.nan
    )
    return grouped.sort_values("summed_daily_return", ascending=False).reset_index(drop=True)


def _switch_rows(result: BacktestResult, returns: pd.Series) -> pd.DataFrame:
    positions = result.positions.sort_index().fillna(0.0)
    assets = _held_assets(positions)
    rows: list[dict[str, object]] = []

    for idx in range(1, len(assets)):
        entry_date = assets.index[idx]
        next_date = assets.index[idx + 1] if idx + 1 < len(assets) else pd.NaT
        hold_returns = returns.loc[entry_date:]
        if pd.notna(next_date):
            hold_returns = hold_returns.loc[hold_returns.index < next_date]

        if hold_returns.empty:
            cumulative = np.nan
            hold_days = 0
        else:
            cumulative = float((1 + hold_returns).prod() - 1)
            hold_days = int(len(hold_returns))

        rows.append(
            {
                "switch_date": entry_date,
                "from_asset": assets.iloc[idx - 1],
                "to_asset": assets.iloc[idx],
                "next_switch_date": next_date,
                "holding_days": hold_days,
                "holding_period_return": cumulative,
                "whipsaw_le_5d_loss": bool(hold_days <= 5 and pd.notna(cumulative) and cumulative < 0),
            }
        )

    return pd.DataFrame(rows)


def _drawdown_episodes(
    returns: pd.Series,
    held_assets: pd.Series,
    switches: pd.DataFrame,
) -> pd.DataFrame:
    equity = (1 + returns).cumprod()
    running_peak = equity.cummax()
    drawdown = equity / running_peak - 1
    peak_dates = pd.Series(index=equity.index, dtype="datetime64[ns]")
    current_peak = equity.index[0]

    for ts, value in equity.items():
        if value >= running_peak.loc[ts]:
            current_peak = ts
        peak_dates.loc[ts] = current_peak

    episodes: list[dict[str, object]] = []
    episode_start: int | None = None
    index = list(equity.index)

    for pos, ts in enumerate(index):
        if drawdown.loc[ts] < 0 and episode_start is None:
            episode_start = pos

        is_last = pos == len(index) - 1
        recovered = drawdown.loc[ts] >= 0
        if episode_start is None or (not recovered and not is_last):
            continue

        finish = pos - 1 if recovered else pos
        underwater_index = equity.index[episode_start : finish + 1]
        trough_date = drawdown.loc[underwater_index].idxmin()
        depth = float(drawdown.loc[trough_date])
        if depth <= DRAW_DOWN_THRESHOLD:
            peak_date = pd.Timestamp(peak_dates.loc[trough_date])
            recovery_date = ts if recovered else pd.NaT
            if pd.notna(recovery_date):
                recovery_days = int(equity.index.get_loc(recovery_date) - equity.index.get_loc(trough_date))
            else:
                recovery_days = np.nan

            interval_assets = held_assets.loc[peak_date:trough_date].dropna().unique().tolist()
            interval_switches = switches.loc[
                switches["switch_date"].between(peak_date, trough_date, inclusive="both")
            ]
            episodes.append(
                {
                    "peak_date": peak_date,
                    "trough_date": trough_date,
                    "drawdown_depth": depth,
                    "recovery_date": recovery_date,
                    "recovery_trading_days_from_trough": recovery_days,
                    "assets_held_peak_to_trough": ", ".join(interval_assets),
                    "switches_peak_to_trough": _switch_summary(interval_switches),
                }
            )

        episode_start = None

    return pd.DataFrame(episodes)


def _switch_summary(switches: pd.DataFrame) -> str:
    if switches.empty:
        return ""
    return "; ".join(
        f"{_fmt_date(row.switch_date)} {row.from_asset}->{row.to_asset}"
        for row in switches.itertuples(index=False)
    )


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"

    view = frame.copy()
    for column in view.columns:
        if "date" in str(column):
            view[column] = view[column].map(_fmt_date)
    for column in ["summed_daily_return", "share_of_summed_return", "drawdown_depth", "holding_period_return"]:
        if column in view.columns:
            view[column] = view[column].map(_fmt_pct)
    view = view.fillna("")

    headers = [str(column) for column in view.columns]
    rows = [[str(value) for value in row] for row in view.to_numpy()]
    widths = [
        max(len(headers[pos]), *(len(row[pos]) for row in rows))
        for pos in range(len(headers))
    ]

    def render(values: list[str]) -> str:
        cells = [values[pos].ljust(widths[pos]) for pos in range(len(values))]
        return "| " + " | ".join(cells) + " |"

    separator = "| " + " | ".join("-" * width for width in widths) + " |"
    return "\n".join([render(headers), separator, *(render(row) for row in rows)])


def _drawdown_transition_mode(drawdowns: pd.DataFrame) -> str:
    counter: Counter[str] = Counter()
    for summary in drawdowns.get("switches_peak_to_trough", pd.Series(dtype=str)).dropna():
        for item in str(summary).split("; "):
            if " " not in item:
                continue
            counter[item.split(" ", maxsplit=1)[1]] += 1
    if not counter:
        return "No switch was recorded inside the qualifying peak-to-trough intervals."
    top_count = max(counter.values())
    top_modes = sorted(mode for mode, count in counter.items() if count == top_count)
    return f"{', '.join(top_modes)} ({top_count} occurrence{'s' if top_count != 1 else ''})"


def _write_outputs(bundle: DiagnosticBundle) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    bundle.attribution.to_csv(RETURN_ATTRIBUTION_PATH, index=False)
    bundle.drawdowns.to_csv(DRAWDOWN_PATH, index=False)
    bundle.switches.to_csv(SWITCH_PATH, index=False)
    bundle.returns.rename("daily_return").rename_axis("date").reset_index().to_csv(
        DAILY_RETURN_PATH,
        index=False,
    )
    bundle.daily_positions.rename_axis("date").reset_index().to_csv(
        DAILY_POSITION_PATH,
        index=False,
    )

    top_asset = bundle.attribution.iloc[0]
    worst_drawdown = bundle.drawdowns.sort_values("drawdown_depth").iloc[0]
    whipsaws = bundle.switches.loc[bundle.switches["whipsaw_le_5d_loss"]]
    no_switch_drawdowns = bundle.drawdowns.loc[
        bundle.drawdowns["switches_peak_to_trough"].fillna("").eq("")
    ]
    transition_mode = _drawdown_transition_mode(bundle.drawdowns)
    daily_start = _fmt_date(bundle.returns.index.min())
    daily_end = _fmt_date(bundle.returns.index.max())

    lines = [
        "# Drawdown and return attribution diagnostic",
        "",
        f"- Run date: {RUN_DATE}",
        f"- Config: `{CONFIG_PATH.relative_to(REPO_ROOT)}`",
        f"- Evaluation window: {daily_start} to {daily_end}",
        "- Return series: current backtest result from `quality_momentum_top1.yaml` with `rebalance_days=5`.",
        "- Position alignment: sparse execution-day `result.positions` reindexed to strategy return dates and forward-filled before attribution.",
        f"- Annualized volatility for the ERC input export: {bundle.annualized_volatility:.2%} (`daily std * sqrt(252)`).",
        f"- Runtime: {bundle.elapsed_seconds:.2f}s",
        "",
        "## Return attribution by held asset",
        "",
        _markdown_table(bundle.attribution),
        "",
        "## Drawdown episodes deeper than 15%",
        "",
        _markdown_table(bundle.drawdowns),
        "",
        "## Switch holding-period P&L",
        "",
        _markdown_table(bundle.switches),
        "",
        "## Conclusion",
        "",
        (
            f"- The summed daily-return attribution points to `{top_asset['asset']}` as the main return engine: "
            f"{top_asset['summed_daily_return']:.2%} summed daily-return points, "
            f"{top_asset['share_of_summed_return']:.2%} of the cross-asset sum."
        ),
        (
            f"- The worst qualifying peak-to-trough interval ran from {_fmt_date(worst_drawdown['peak_date'])} "
            f"to {_fmt_date(worst_drawdown['trough_date'])} at {worst_drawdown['drawdown_depth']:.2%}."
        ),
        (
            f"- The most common switch pattern inside qualifying peak-to-trough intervals is {transition_mode}."
        ),
        (
            f"- {len(no_switch_drawdowns)} qualifying peak-to-trough interval"
            f"{'s' if len(no_switch_drawdowns) != 1 else ''} had no switch at all, so deep drawdown evidence is not only a churn story."
        ),
        (
            f"- The all-sample whipsaw flag catches {len(whipsaws)} switch holding periods with "
            "holding days <= 5 and negative cumulative return."
        ),
        "",
        "## Caveats and raw outputs",
        "",
        "- On an execution day the backtest return mixes the outgoing asset overnight move and the incoming asset intraday move. This report groups that mixed return under the post-open held asset after the forward-filled position step.",
        "- Return attribution is a sum of daily returns by held asset, not an exact compounded Brinson-style decomposition.",
        f"- `{RETURN_ATTRIBUTION_PATH.name}`",
        f"- `{DRAWDOWN_PATH.name}`",
        f"- `{SWITCH_PATH.name}`",
        f"- `{DAILY_RETURN_PATH.name}`",
        f"- `{DAILY_POSITION_PATH.name}`",
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_diagnostic() -> DiagnosticBundle:
    started = perf_counter()
    result = _load_result()
    daily_positions = _daily_positions(result)
    all_held_assets = _held_assets(daily_positions)
    returns = result.daily_returns.loc[result.daily_returns.index >= START].copy()
    daily_positions = daily_positions.reindex(returns.index).ffill()
    held_assets = all_held_assets.reindex(returns.index).ffill()
    attribution = _return_attribution(returns, held_assets)
    switches = _switch_rows(result, returns)
    switches = switches.loc[switches["switch_date"] >= START].reset_index(drop=True)
    drawdowns = _drawdown_episodes(returns, held_assets, switches)
    annualized_volatility = float(returns.std() * np.sqrt(252))
    elapsed = perf_counter() - started

    bundle = DiagnosticBundle(
        result=result,
        returns=returns,
        daily_positions=daily_positions,
        held_assets=held_assets,
        attribution=attribution,
        drawdowns=drawdowns,
        switches=switches,
        annualized_volatility=annualized_volatility,
        elapsed_seconds=elapsed,
    )
    _write_outputs(bundle)
    return bundle


def main() -> None:
    bundle = run_diagnostic()
    print(f"Wrote report: {REPORT_PATH}")
    print(f"Wrote returns CSV: {DAILY_RETURN_PATH}")
    print(f"Annualized volatility: {bundle.annualized_volatility:.6f}")
    print(f"Return attribution rows: {len(bundle.attribution)}")
    print(f"Drawdown rows: {len(bundle.drawdowns)}")
    print(f"Switch rows: {len(bundle.switches)}")


if __name__ == "__main__":
    main()
