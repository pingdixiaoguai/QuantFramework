"""Attribute the performance gap: 10% drift trigger vs fixed monthly check.

Compares ``portfolio_drift_10_monthly_cap1`` (current strategy) against
``calendar_monthly_reference`` (baseline) on identical market data, deposits,
costs, and execution rules, and decomposes the gap into measurable parts:

1. Tracking error to the fresh daily reversal target (signal staleness).
2. What actually moves the drift trigger: target change vs price drift.
3. Rebalance timing inside the month and per-rebalance tilt quality.
4. Whipsaw: how quickly the executed target becomes stale afterwards.
5. Daily return attribution by asset.
6. Cost and turnover per rebalance.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .rebalance_timing import daily_reversal_targets
from .reduced_pool_research import REDUCED_TARGET
from .strategy import CASH_ASSET, load_confirmed_market
from .threshold_rebalance import TRIGGER_SPECS, simulate_threshold_rebalance


ROOT = Path(__file__).parent
OUTPUT = ROOT / "threshold_rebalance_experiments"

CALENDAR = "calendar_monthly_reference"
DRIFT = "portfolio_drift_10_monthly_cap1"


def _positions_frame(daily: pd.DataFrame) -> pd.DataFrame:
    frame = pd.DataFrame(daily["positions"].tolist(), index=daily.index).fillna(0.0)
    return frame.reindex(columns=list(REDUCED_TARGET), fill_value=0.0)


def _rebalance_events(result, daily_targets, dates) -> pd.DataFrame:
    """One row per executed rebalance with signal date and executed target."""
    trades = result.trades.loc[result.trades["reason"] == "threshold_rebalance"]
    if trades.empty:
        return pd.DataFrame()
    date_positions = {timestamp: index for index, timestamp in enumerate(dates)}
    day_of_month = pd.Series(dates, index=dates).groupby([dates.year, dates.month]).cumcount() + 1
    day_of_month.index = dates
    rows = []
    for execution_date, group in trades.groupby("date"):
        execution_date = pd.Timestamp(execution_date)
        index = date_positions[execution_date]
        signal_date = dates[index - 1]
        target = dict(daily_targets[signal_date])
        rows.append({
            "execution_date": execution_date,
            "signal_date": signal_date,
            "trading_day_of_month": int(day_of_month.loc[execution_date]),
            "turnover": float(group["notional"].sum()),
            "orders": int(len(group)),
            **{f"target_{asset}": target.get(asset, 0.0) for asset in REDUCED_TARGET},
        })
    return pd.DataFrame(rows)


def _forward_returns(market, dates, horizon: int) -> pd.DataFrame:
    closes = pd.DataFrame({asset: market.closes[asset] for asset in REDUCED_TARGET})
    closes = closes.reindex(dates).ffill()
    return closes.shift(-horizon) / closes - 1.0


def run() -> dict[str, pd.DataFrame]:
    _, market = load_confirmed_market()
    daily_targets = daily_reversal_targets(market, REDUCED_TARGET)
    specs = {spec.name: spec for spec in TRIGGER_SPECS}
    results = {
        name: simulate_threshold_rebalance(
            market,
            daily_targets,
            specs[name],
            initial_target=REDUCED_TARGET,
            cash_asset=CASH_ASSET,
            min_rebalance_notional=10_000.0,
        )
        for name in (CALENDAR, DRIFT)
    }
    dates = pd.DatetimeIndex(market.dates)
    target_frame = pd.DataFrame(daily_targets).T.reindex(dates).astype(float)

    weights = {name: _positions_frame(result.daily) for name, result in results.items()}
    cash_weight = {name: result.daily["cash_weight"].astype(float) for name, result in results.items()}

    # 1. Tracking error to the fresh daily target (single-sided, ex-cash-target ~0).
    tracking = pd.DataFrame({
        name: 0.5 * (frame - target_frame).abs().sum(axis=1) + cash_weight[name].clip(lower=0.0)
        for name, frame in weights.items()
    })

    # 2. Trigger decomposition for the drift strategy: how much of the daily
    # drift value comes from the target moving vs prices moving.
    drift_result = results[DRIFT]
    executed_targets: list[dict[str, float]] = []
    price_only: dict[pd.Timestamp, float] = {}
    target_only: dict[pd.Timestamp, float] = {}
    last_exec = dict(REDUCED_TARGET)
    rebalance_exec_dates = set(
        pd.to_datetime(drift_result.trades.loc[
            drift_result.trades["reason"] == "threshold_rebalance", "date"
        ])
    )
    signal_index = {pd.Timestamp(row["date"]): idx for idx, row in drift_result.signals.iterrows()} \
        if not drift_result.signals.empty else {}
    del signal_index
    for timestamp in dates:
        actual = weights[DRIFT].loc[timestamp]
        fresh = target_frame.loc[timestamp]
        last = pd.Series(last_exec)
        price_only[timestamp] = float(0.5 * (actual - last).abs().sum())
        target_only[timestamp] = float(0.5 * (fresh - last).abs().sum())
        if timestamp in rebalance_exec_dates:
            # executed target was fixed at the previous close signal
            signal_date = dates[dates.get_loc(timestamp) - 1]
            last_exec = dict(daily_targets[signal_date])
            executed_targets.append(last_exec)
    trigger_days = pd.to_datetime(drift_result.signals["date"]) if not drift_result.signals.empty else []
    decomposition = pd.DataFrame({
        "price_drift_component": pd.Series(price_only),
        "target_change_component": pd.Series(target_only),
    })

    # 3. Rebalance events and per-rebalance tilt quality.
    events = {name: _rebalance_events(result, daily_targets, dates) for name, result in results.items()}
    fwd20 = _forward_returns(market, dates, 20)
    base = pd.Series(REDUCED_TARGET)
    for name, frame in events.items():
        if frame.empty:
            continue
        tilt_cols = [f"target_{asset}" for asset in REDUCED_TARGET]
        tilts = frame[tilt_cols].copy()
        tilts.columns = list(REDUCED_TARGET)
        forward = fwd20.reindex(pd.DatetimeIndex(frame["signal_date"])).to_numpy()
        tilt_values = (tilts - base).to_numpy()
        frame["tilt_forward_20d"] = np.nansum(tilt_values * forward, axis=1)
        later20 = target_frame.shift(-20).reindex(pd.DatetimeIndex(frame["signal_date"])).to_numpy()
        frame["whipsaw_20d"] = 0.5 * np.abs(tilts.to_numpy() - later20).sum(axis=1)

    # 4. Annual returns.
    annual = pd.DataFrame({
        name: (1.0 + result.daily["return"].astype(float).fillna(0.0)).groupby(
            result.daily.index.year
        ).prod() - 1.0
        for name, result in results.items()
    })
    annual["gap_calendar_minus_drift"] = annual[CALENDAR] - annual[DRIFT]

    # 5. Daily return attribution by asset (close-to-close approximation).
    asset_returns = pd.DataFrame({
        asset: market.closes[asset] for asset in REDUCED_TARGET
    }).reindex(dates).ffill().pct_change().fillna(0.0)
    weight_gap = (weights[DRIFT] - weights[CALENDAR]).shift(1).fillna(0.0)
    attribution_daily = weight_gap * asset_returns
    attribution_total = attribution_daily.sum()
    attribution_cumulative = float(attribution_daily.sum(axis=1).sum())

    # 6. Costs.
    cost_summary = {}
    for name, result in results.items():
        trades = result.trades
        rebalances = trades.loc[trades["reason"] == "threshold_rebalance"]
        cost_summary[name] = {
            "total_cost": float((trades["notional"] * 0.0005).sum()),
            "rebalance_count": int(rebalances["date"].nunique()) if not rebalances.empty else 0,
            "avg_turnover_per_rebalance": float(rebalances.groupby("date")["notional"].sum().mean())
            if not rebalances.empty else 0.0,
            "avg_orders_per_rebalance": float(rebalances.groupby("date")["asset"].count().mean())
            if not rebalances.empty else 0.0,
        }

    summary_rows = []
    for name in (CALENDAR, DRIFT):
        frame = events[name]
        summary_rows.append({
            "strategy": name,
            "avg_equity_512890_weight": float(weights[name]["512890.SH"].mean()),
            "avg_tracking_error": float(tracking[name].mean()),
            "median_tracking_error": float(tracking[name].median()),
            "p90_tracking_error": float(tracking[name].quantile(0.9)),
            "rebalances": int(len(frame)),
            "avg_tilt_forward_20d_bp": float(frame["tilt_forward_20d"].mean() * 1e4) if not frame.empty else np.nan,
            "median_tilt_forward_20d_bp": float(frame["tilt_forward_20d"].median() * 1e4) if not frame.empty else np.nan,
            "avg_whipsaw_20d": float(frame["whipsaw_20d"].mean()) if not frame.empty else np.nan,
            "avg_trading_day_of_month": float(frame["trading_day_of_month"].mean()) if not frame.empty else np.nan,
            **cost_summary[name],
        })
    summary = pd.DataFrame(summary_rows).set_index("strategy")

    trigger_decomp_on_hits = decomposition.reindex(pd.DatetimeIndex(trigger_days)).mean()

    OUTPUT.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT / "drift_vs_calendar_attribution_summary.csv")
    annual.to_csv(OUTPUT / "drift_vs_calendar_annual_returns.csv")
    events[DRIFT].to_csv(OUTPUT / "drift_vs_calendar_drift_rebalance_events.csv", index=False)
    events[CALENDAR].to_csv(OUTPUT / "drift_vs_calendar_calendar_rebalance_events.csv", index=False)
    attribution_total.to_csv(OUTPUT / "drift_vs_calendar_asset_attribution.csv")

    print("== summary ==")
    print(summary.to_string())
    print("\n== drift trigger decomposition on trigger days (mean single-sided) ==")
    print(trigger_decomp_on_hits.to_string())
    print("\n== annual returns ==")
    print(annual.map(lambda value: f"{value:.2%}").to_string())
    print("\n== asset attribution of (drift - calendar) daily returns ==")
    print((attribution_total * 100).map(lambda value: f"{value:+.3f}%").to_string())
    print(f"cumulative attribution gap: {attribution_cumulative * 100:+.2f}%")
    return {"summary": summary, "annual": annual, "events": events, "tracking": tracking}


if __name__ == "__main__":
    run()
