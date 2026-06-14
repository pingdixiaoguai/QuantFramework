"""Research-only cost x tau scan for quality_momentum_top1.

This script does not edit production YAMLs and does not call live/backfill.
It runs each tau once at zero cost, then applies fee levels by arithmetic
from the gross return and executed turnover series.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from backtest.runner import BacktestResult, run
from data.store import query


REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "strategy" / "configs" / "quality_momentum_top1.yaml"
OUT_DIR = REPO_ROOT / "strategy_changelog_attachments"
PREFIX = "2026-06-04_cost_tau"

START = date(2014, 1, 1)
ASSET_POOL = ["510300.SH", "159915.SZ", "513100.SH", "518880.SH"]
TAUS = [0.0, 0.0005, 0.001, 0.0025, 0.005, 0.0075, 0.01]
FEES = [0.00005, 0.0001, 0.0003, 0.0005, 0.001]
EPISODE_FEES = [0.0001, 0.0003, 0.0005]
CANARY_DATE = pd.Timestamp("2024-09-26")
CANARY_ASSET = "159915.SZ"


@dataclass
class TauRun:
    tau: float
    result: BacktestResult
    gross: pd.Series
    turnover: pd.Series
    positions: pd.DataFrame


def _load_config(tau: float | None = None) -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config["asset_pool"] = list(ASSET_POOL)
    config["start"] = START
    config["end"] = date.today()
    config["rebalance_days"] = 5
    config["transaction_cost_rate"] = 0.0
    config.pop("rebalance_mode", None)
    if tau is not None:
        config["hysteresis_threshold"] = tau
    return config


def _returns_for(run_: TauRun, fee: float) -> pd.Series:
    returns = run_.gross.copy()
    if returns.empty or fee == 0:
        return returns
    costs = run_.turnover.reindex(returns.index, fill_value=0.0) * fee
    return returns - costs


def _metrics(returns: pd.Series, turnover: pd.Series, positions: pd.DataFrame) -> dict[str, object]:
    n_days = int(len(returns))
    if n_days == 0:
        return {
            "start": "",
            "end": "",
            "trading_days": 0,
            "total_return": 0.0,
            "annual_return": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "annual_turnover_sum_abs": 0.0,
            "annual_turnover_single_side": 0.0,
            "avg_holding_days": 0.0,
            "switch_count": 0,
        }

    cumulative = (1.0 + returns).cumprod()
    total_return = float(cumulative.iloc[-1] - 1.0)
    annual_return = float((1.0 + total_return) ** (252.0 / n_days) - 1.0)
    std = returns.std()
    sharpe = float(returns.mean() / std * math.sqrt(252.0)) if std > 0 else 0.0
    max_drawdown = float((cumulative / cumulative.cummax() - 1.0).min())
    years = n_days / 252.0
    turnover_sum_abs = float(turnover.sum() / years) if years else 0.0
    switch_count = max(int(len(positions) - 1), 0)
    avg_holding_days = float(n_days / len(positions)) if len(positions) else 0.0

    return {
        "start": returns.index.min().date().isoformat(),
        "end": returns.index.max().date().isoformat(),
        "trading_days": n_days,
        "total_return": total_return,
        "annual_return": annual_return,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "annual_turnover_sum_abs": turnover_sum_abs,
        "annual_turnover_single_side": turnover_sum_abs / 2.0,
        "avg_holding_days": avg_holding_days,
        "switch_count": switch_count,
    }


def _weights_from_row(row: pd.Series) -> dict[str, float]:
    return {
        str(asset): float(value)
        for asset, value in row.items()
        if pd.notna(value) and float(value) != 0.0
    }


def _top_asset(row: pd.Series) -> str | None:
    weights = _weights_from_row(row)
    return max(weights, key=weights.get) if weights else None


def _switch_table(run_: TauRun) -> pd.DataFrame:
    if run_.positions.empty:
        return pd.DataFrame(columns=["execution_date", "old_asset", "new_asset", "turnover"])

    rows = []
    old_asset = None
    for dt, row in run_.positions.fillna(0.0).iterrows():
        new_asset = _top_asset(row)
        turnover = float(run_.turnover.get(pd.Timestamp(dt), 0.0))
        rows.append(
            {
                "execution_date": pd.Timestamp(dt),
                "old_asset": old_asset,
                "new_asset": new_asset,
                "turnover": turnover,
            }
        )
        old_asset = new_asset
    return pd.DataFrame(rows)


def _price_maps(end: date) -> tuple[dict[str, pd.Series], dict[str, pd.Series]]:
    open_prices: dict[str, pd.Series] = {}
    close_prices: dict[str, pd.Series] = {}
    for asset in ASSET_POOL:
        df = query(asset, START, end)
        open_prices[asset] = pd.Series(df["open"].values, index=pd.DatetimeIndex(df["date"]))
        close_prices[asset] = pd.Series(df["close"].values, index=pd.DatetimeIndex(df["date"]))
    return open_prices, close_prices


def _asset_return(
    asset: str | None,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    open_prices: dict[str, pd.Series],
    close_prices: dict[str, pd.Series],
) -> float | None:
    if asset is None:
        return None
    opens = open_prices.get(asset)
    closes = close_prices.get(asset)
    if opens is None or closes is None:
        return None
    if start_date not in opens.index or end_date not in closes.index:
        return None
    return float(closes.loc[end_date] / opens.loc[start_date] - 1.0)


def _previous_trading_day(days: pd.DatetimeIndex, dt: pd.Timestamp) -> pd.Timestamp:
    before = days[days < dt]
    if len(before) == 0:
        return dt
    return pd.Timestamp(before[-1])


def _main_table(runs: dict[float, TauRun]) -> pd.DataFrame:
    rows = []
    for tau, tau_run in runs.items():
        for fee in FEES:
            returns = _returns_for(tau_run, fee)
            rows.append(
                {
                    "tau": tau,
                    "fee_one_side": fee,
                    "fee_bps_one_side": fee * 10000.0,
                    "fee_note": "stress only; ETF no stamp duty" if fee == 0.001 else "",
                    **_metrics(returns, tau_run.turnover, tau_run.positions),
                }
            )
    return pd.DataFrame(rows)


def _break_even_table(runs: dict[float, TauRun], main_df: pd.DataFrame) -> pd.DataFrame:
    baseline = runs[0.0]
    baseline_gross_ann = _metrics(baseline.gross, baseline.turnover, baseline.positions)[
        "annual_return"
    ]
    baseline_turnover = _metrics(baseline.gross, baseline.turnover, baseline.positions)[
        "annual_turnover_sum_abs"
    ]

    rows = []
    for tau, tau_run in runs.items():
        if tau == 0.0:
            continue
        tau_gross_metrics = _metrics(tau_run.gross, tau_run.turnover, tau_run.positions)
        delta_gross = float(tau_gross_metrics["annual_return"]) - float(baseline_gross_ann)
        turnover_reduction = float(baseline_turnover) - float(
            tau_gross_metrics["annual_turnover_sum_abs"]
        )
        linear_fee = (
            -delta_gross / turnover_reduction
            if turnover_reduction != 0
            else float("nan")
        )

        tau_panel = main_df[main_df["tau"] == tau].set_index("fee_one_side")
        base_panel = main_df[main_df["tau"] == 0.0].set_index("fee_one_side")
        fees = sorted(set(tau_panel.index) & set(base_panel.index))
        advantages = [
            float(tau_panel.loc[fee, "annual_return"] - base_panel.loc[fee, "annual_return"])
            for fee in fees
        ]
        exact_fee, status = _piecewise_root(fees, advantages)

        rows.append(
            {
                "tau": tau,
                "delta_gross_ann_return": delta_gross,
                "turnover_reduction_sum_abs_ann": turnover_reduction,
                "linear_break_even_fee": linear_fee,
                "linear_break_even_bps_one_side": linear_fee * 10000.0,
                "exact_break_even_fee": exact_fee,
                "exact_break_even_bps_one_side": exact_fee * 10000.0,
                "exact_break_even_status": status,
            }
        )
    return pd.DataFrame(rows)


def _piecewise_root(xs: list[float], ys: list[float]) -> tuple[float, str]:
    for i in range(len(xs) - 1):
        y0 = ys[i]
        y1 = ys[i + 1]
        if y0 == 0:
            return xs[i], "grid_point"
        if y0 * y1 <= 0:
            if y1 == y0:
                return xs[i], "flat_segment"
            root = xs[i] - y0 * (xs[i + 1] - xs[i]) / (y1 - y0)
            return float(root), "interpolated"

    if len(xs) < 2:
        return float("nan"), "not_enough_points"
    if ys[0] > 0 and ys[-1] > 0:
        i0, i1 = 0, 1
        status = "extrapolated_below_grid"
    else:
        i0, i1 = len(xs) - 2, len(xs) - 1
        status = "extrapolated_above_grid"
    if ys[i1] == ys[i0]:
        return float("nan"), "no_crossing_flat"
    root = xs[i0] - ys[i0] * (xs[i1] - xs[i0]) / (ys[i1] - ys[i0])
    return float(root), status


def _episode_table(runs: dict[float, TauRun]) -> pd.DataFrame:
    episodes = [
        ("whipsaw", pd.Timestamp("2024-11-04"), pd.Timestamp("2024-12-17")),
        ("single_asset_crash", pd.Timestamp("2024-10-08"), pd.Timestamp("2024-11-15")),
    ]
    rows = []
    for tau, tau_run in runs.items():
        switches = _switch_table(tau_run)
        for episode, start, end in episodes:
            mask = (tau_run.gross.index >= start) & (tau_run.gross.index <= end)
            switch_count = int(
                (
                    (switches["execution_date"] >= start)
                    & (switches["execution_date"] <= end)
                    & (switches["turnover"] > 0)
                ).sum()
            )
            for fee in EPISODE_FEES:
                returns = _returns_for(tau_run, fee).loc[mask]
                pnl = float((1.0 + returns).prod() - 1.0) if len(returns) else 0.0
                rows.append(
                    {
                        "episode": episode,
                        "start": start.date().isoformat(),
                        "end": end.date().isoformat(),
                        "tau": tau,
                        "fee_bps_one_side": fee * 10000.0,
                        "switch_count": switch_count,
                        "cumulative_pnl": pnl,
                    }
                )
    return pd.DataFrame(rows)


def _canary_and_good_switches(
    runs: dict[float, TauRun],
    open_prices: dict[str, pd.Series],
    close_prices: dict[str, pd.Series],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline_switches = _switch_table(runs[0.0])
    all_days = pd.DatetimeIndex(runs[0.0].gross.index)
    baseline_events = baseline_switches[baseline_switches["old_asset"].notna()].reset_index(drop=True)
    rows = []
    good_rows = []

    canary = baseline_events[
        (baseline_events["execution_date"] == CANARY_DATE)
        & (baseline_events["new_asset"] == CANARY_ASSET)
    ]
    if canary.empty:
        canary_event = None
    else:
        canary_event = canary.iloc[0]

    for idx, event in baseline_events.iterrows():
        execution_date = pd.Timestamp(event["execution_date"])
        next_execution = (
            pd.Timestamp(baseline_events.iloc[idx + 1]["execution_date"])
            if idx + 1 < len(baseline_events)
            else all_days[-1]
        )
        wave_end = _previous_trading_day(all_days, next_execution)
        baseline_capture = _asset_return(
            event["new_asset"],
            execution_date,
            wave_end,
            open_prices,
            close_prices,
        )
        if baseline_capture is None:
            continue

        event_rows = []
        for tau, tau_run in runs.items():
            switches = _switch_table(tau_run)
            same_asset = switches[
                (switches["new_asset"] == event["new_asset"])
                & (switches["execution_date"] >= execution_date)
                & (switches["execution_date"] <= wave_end)
            ]
            if same_asset.empty:
                status = "blocked"
                actual_date = pd.NaT
                delay_days = None
                capture = 0.0
            else:
                actual_date = pd.Timestamp(same_asset.iloc[0]["execution_date"])
                delay_days = int((all_days >= execution_date).sum() - (all_days > actual_date).sum() - 1)
                status = "on_time" if actual_date == execution_date else "delayed"
                capture = _asset_return(
                    event["new_asset"],
                    actual_date,
                    wave_end,
                    open_prices,
                    close_prices,
                )
                capture = 0.0 if capture is None else float(capture)

            row = {
                "baseline_execution_date": execution_date.date().isoformat(),
                "wave_end": wave_end.date().isoformat(),
                "old_asset": event["old_asset"],
                "new_asset": event["new_asset"],
                "tau": tau,
                "status": status,
                "actual_execution_date": (
                    "" if pd.isna(actual_date) else actual_date.date().isoformat()
                ),
                "delay_trading_days": delay_days,
                "baseline_capture": baseline_capture,
                "actual_capture": capture,
                "capture_loss": baseline_capture - capture,
            }
            event_rows.append(row)
            if canary_event is not None and execution_date == CANARY_DATE and event["new_asset"] == CANARY_ASSET:
                rows.append(row)

        if baseline_capture > 0 and any(r["status"] != "on_time" for r in event_rows):
            good_rows.extend(event_rows)

    return pd.DataFrame(rows), pd.DataFrame(good_rows)


def _run_all() -> dict[float, TauRun]:
    runs: dict[float, TauRun] = {}
    for tau in TAUS:
        config = _load_config(tau)
        result = run(config)
        if result.gross_daily_returns is None or result.turnover is None:
            raise RuntimeError("engine did not return gross/turnover decomposition")
        runs[tau] = TauRun(
            tau=tau,
            result=result,
            gross=result.gross_daily_returns,
            turnover=result.turnover,
            positions=result.positions,
        )
    return runs


def _gate(runs: dict[float, TauRun]) -> pd.DataFrame:
    baseline = run(_load_config(None))
    tau0 = runs[0.0].result
    rows = [
        {
            "check": "gross_daily_returns",
            "passed": bool(baseline.gross_daily_returns.equals(tau0.gross_daily_returns)),
        },
        {"check": "positions", "passed": bool(baseline.positions.equals(tau0.positions))},
        {"check": "turnover", "passed": bool(baseline.turnover.equals(tau0.turnover))},
        {"check": "net_daily_returns_zero_cost", "passed": bool(baseline.daily_returns.equals(tau0.daily_returns))},
    ]
    gate = pd.DataFrame(rows)
    if not bool(gate["passed"].all()):
        raise RuntimeError("tau=0 gate failed; stop before scanning conclusions")
    return gate


def _fmt_pct(value: float) -> str:
    return f"{value:.2%}"


def _fmt(value: object, column: str) -> str:
    if isinstance(value, str):
        return value
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    if column in {
        "annual_return",
        "max_drawdown",
        "annual_turnover_sum_abs",
        "annual_turnover_single_side",
        "delta_gross_ann_return",
        "turnover_reduction_sum_abs_ann",
        "cumulative_pnl",
        "baseline_capture",
        "actual_capture",
        "capture_loss",
    }:
        return _fmt_pct(float(value))
    if column in {"fee_bps_one_side", "linear_break_even_bps_one_side", "exact_break_even_bps_one_side"}:
        return f"{float(value):.2f}"
    if column == "sharpe":
        return f"{float(value):.2f}"
    if column == "avg_holding_days":
        return f"{float(value):.2f}"
    if column in {"trading_days", "switch_count", "delay_trading_days"}:
        return "" if pd.isna(value) else str(int(value))
    if isinstance(value, (bool, np.bool_)):
        return "yes" if bool(value) else "no"
    if isinstance(value, float):
        return f"{value:.10g}"
    return str(value)


def _markdown_table(df: pd.DataFrame, columns: list[str], max_rows: int | None = None) -> str:
    show = df.loc[:, columns]
    if max_rows is not None:
        show = show.head(max_rows)
    lines = ["| " + " | ".join(columns) + " |"]
    lines.append("| " + " | ".join("---" for _ in columns) + " |")
    for _, row in show.iterrows():
        lines.append("| " + " | ".join(_fmt(row[col], col) for col in columns) + " |")
    return "\n".join(lines)


def _write_outputs(
    gate: pd.DataFrame,
    main_df: pd.DataFrame,
    break_even_df: pd.DataFrame,
    episode_df: pd.DataFrame,
    canary_df: pd.DataFrame,
    good_switch_df: pd.DataFrame,
) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    gate.to_csv(OUT_DIR / f"{PREFIX}_tau0_gate.csv", index=False, encoding="utf-8-sig")
    main_df.to_csv(OUT_DIR / f"{PREFIX}_main_metrics.csv", index=False, encoding="utf-8-sig")
    break_even_df.to_csv(OUT_DIR / f"{PREFIX}_break_even.csv", index=False, encoding="utf-8-sig")
    episode_df.to_csv(OUT_DIR / f"{PREFIX}_episodes.csv", index=False, encoding="utf-8-sig")
    canary_df.to_csv(OUT_DIR / f"{PREFIX}_canary.csv", index=False, encoding="utf-8-sig")
    good_switch_df.to_csv(OUT_DIR / f"{PREFIX}_good_switches.csv", index=False, encoding="utf-8-sig")

    main_cols = [
        "tau",
        "fee_bps_one_side",
        "annual_return",
        "sharpe",
        "max_drawdown",
        "annual_turnover_sum_abs",
        "annual_turnover_single_side",
        "avg_holding_days",
        "switch_count",
    ]
    be_cols = [
        "tau",
        "delta_gross_ann_return",
        "turnover_reduction_sum_abs_ann",
        "linear_break_even_bps_one_side",
        "exact_break_even_bps_one_side",
        "exact_break_even_status",
    ]
    episode_cols = [
        "episode",
        "tau",
        "fee_bps_one_side",
        "switch_count",
        "cumulative_pnl",
    ]
    canary_cols = [
        "baseline_execution_date",
        "wave_end",
        "tau",
        "status",
        "actual_execution_date",
        "delay_trading_days",
        "baseline_capture",
        "actual_capture",
    ]

    lines = [
        "# Cost x Tau Scan",
        "",
        f"- Run date: {date.today().isoformat()}",
        f"- Config base: `{CONFIG_PATH.relative_to(REPO_ROOT)}`; in-memory overrides only: `start=2014-01-01`, `rebalance_days=5`, `transaction_cost_rate=0`, `hysteresis_threshold=tau`.",
        "- Scope: Mode A research backtest only; no live/backfill, no production YAML, no changelog edits.",
        "- Execution: existing T+1 open engine; cost applied as one-side fee times executed `Σ|Δw|`.",
        "- Fee grid: 0.5, 1, 3, 5, 10 bps one-side. 10 bps is stress only; ETF no stamp duty, so this is not a real ETF cost assumption.",
        "",
        "## Three Questions",
        "",
        "Q1: The previously reported annual turnover around 2300% is the single-side/net-rotation convention: `0.5 * Σ|w_new - w_old| / years`. The 1bp deduction path in `scripts/periodic_reeval_scan.py` used `Σ|Δw| * fee`, so a Top1 full switch paid `2 * fee`. Deduction was already aligned with the requested cost formula; the reported turnover label needs the single-side qualifier.",
        "",
        "Q2: Existing research cost was based on executed weight-delta magnitude from position changes, not on order count. `execution.diff()` does emit `hold` orders with `weight_delta=0`, but the cost path does not charge holds; hold days have zero turnover and zero cost.",
        "",
        "Q3: Before this change, 0.01% was not a unified backtest-engine parameter. It was ad-hoc research post-processing in scan scripts. This patch formalizes `transaction_cost_rate` in the backtest engine while keeping the production YAML untouched.",
        "",
        "## Tau=0 Gate",
        "",
        _markdown_table(gate, list(gate.columns)),
        "",
        "## Main Metrics",
        "",
        "Annual turnover is shown in both requested `Σ|Δw|` annualized form and the old single-side convention.",
        "",
        _markdown_table(main_df, main_cols),
        "",
        "## Break-Even",
        "",
        _markdown_table(break_even_df, be_cols),
        "",
        "Readout material:",
    ]
    for _, row in break_even_df.iterrows():
        if float(row["exact_break_even_bps_one_side"]) <= 0:
            lines.append(
                f"- tau={row['tau']}: already ahead at nonnegative cost levels; "
                f"the extrapolated crossing is {row['exact_break_even_bps_one_side']:.2f} bps "
                f"({row['exact_break_even_status']})."
            )
        else:
            lines.append(
                f"- tau={row['tau']}: economically preferable only when real one-side cost is above about "
                f"{row['exact_break_even_bps_one_side']:.2f} bps ({row['exact_break_even_status']})."
            )
    lines.extend(
        [
            "",
            "## Episode Decomposition",
            "",
            "Switch counts are actual T+1 open executions inside the inclusive date window. Under the enforced rd=5/min-hold path, tau=0 has 4 executions inside `2024-11-04` to `2024-12-17`; the next execution is `2024-12-20`, outside the requested window.",
            "",
            _markdown_table(episode_df, episode_cols),
            "",
            "## Canary",
            "",
            _markdown_table(canary_df, canary_cols),
            "",
            "## Other Delayed/Blocked Positive Baseline Switches",
            "",
            _markdown_table(good_switch_df, canary_cols + ["new_asset"], max_rows=30)
            if not good_switch_df.empty
            else "No positive baseline switches were delayed or blocked.",
            "",
            "## Raw CSV",
            "",
            f"- `{PREFIX}_tau0_gate.csv`",
            f"- `{PREFIX}_main_metrics.csv`",
            f"- `{PREFIX}_break_even.csv`",
            f"- `{PREFIX}_episodes.csv`",
            f"- `{PREFIX}_canary.csv`",
            f"- `{PREFIX}_good_switches.csv`",
            "",
        ]
    )
    report = OUT_DIR / f"{PREFIX}_report.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate-only", action="store_true")
    args = parser.parse_args()

    runs = _run_all()
    gate = _gate(runs)
    if args.gate_only:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        gate.to_csv(OUT_DIR / f"{PREFIX}_tau0_gate.csv", index=False, encoding="utf-8-sig")
        print(gate.to_string(index=False))
        return

    main_df = _main_table(runs)
    break_even_df = _break_even_table(runs, main_df)
    episode_df = _episode_table(runs)
    open_prices, close_prices = _price_maps(date.today())
    canary_df, good_switch_df = _canary_and_good_switches(runs, open_prices, close_prices)
    report = _write_outputs(
        gate,
        main_df,
        break_even_df,
        episode_df,
        canary_df,
        good_switch_df,
    )
    print(f"REPORT {report}")


if __name__ == "__main__":
    main()
