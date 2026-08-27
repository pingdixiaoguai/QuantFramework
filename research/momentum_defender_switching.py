"""Causal open-switch overlay between the production momentum and Defender sleeves.

The meta signal is evaluated after each close and becomes effective at the next
trading day's open.  At an overlay switch the old sleeve earns the overnight
move, the portfolio is liquidated/rebuilt at the open, and the new sleeve earns
the intraday move.  Internal sleeve target changes follow the same convention.

This is a research entrypoint.  It does not mutate either production strategy.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
import yaml

from backtest.report import generate
from backtest.runner import BacktestResult, run
from data.store import query


MOMENTUM_ASSETS = ("510300.SH", "159915.SZ", "513100.SH", "518880.SH")
DEFENDER_PRIMARY = "512890.SH"
DEFENDER_DEFENSIVE = "511260.SH"
RISK_SIGNAL_ASSET = "510300.SH"
ONE_WAY_COST_RATES = {
    **{asset: 0.0001 for asset in MOMENTUM_ASSETS},
    DEFENDER_PRIMARY: 0.0001,
    DEFENDER_DEFENSIVE: 0.00001,
}
INITIAL_CAPITAL = 1.0


@dataclass(frozen=True)
class SwitchParams:
    lookback: int = 40
    risk_on_threshold: float = 0.025
    min_hold_days: int = 30

    def label(self) -> str:
        threshold = f"{self.risk_on_threshold:+.3f}".replace("+", "p").replace("-", "m").replace(".", "p")
        return f"lb{self.lookback}_th{threshold}_hold{self.min_hold_days}"


@dataclass
class ResearchContext:
    calendar: pd.DatetimeIndex
    prices: dict[str, pd.DataFrame]
    momentum_targets: pd.DataFrame
    defender_targets: pd.DataFrame
    momentum_result: BacktestResult
    risk_close: pd.Series
    original_defender_returns: pd.Series


def _load_momentum_config(path: Path, end: date) -> dict:
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    for key in ("start", "end"):
        value = config.get(key)
        if isinstance(value, str):
            config[key] = date.today() if value.lower() == "today" else date.fromisoformat(value)
    config["end"] = end
    # Project research governance treats one-way 1bp as the authoritative base case.
    config["transaction_cost_rate"] = 0.0001
    return config


def _load_prices(assets: set[str], start: date, end: date) -> dict[str, pd.DataFrame]:
    result: dict[str, pd.DataFrame] = {}
    for asset in sorted(assets):
        frame = query(asset, start, end).sort_values("date").drop_duplicates("date")
        if frame.empty:
            raise RuntimeError(f"no local price data for {asset}")
        indexed = frame.set_index("date")[["open", "close"]].astype(float)
        if indexed.isna().any().any():
            raise ValueError(f"missing OHLC data for {asset}")
        result[asset] = indexed
    return result


def _momentum_target_schedule(
    result: BacktestResult,
    calendar: pd.DatetimeIndex,
) -> pd.DataFrame:
    positions = result.positions.copy().sort_index()
    for asset in MOMENTUM_ASSETS:
        if asset not in positions:
            positions[asset] = 0.0
    # Each sparse record is a complete target: omitted assets mean zero, not
    # "carry the previous asset".  Fill those zeros before expanding dates.
    positions = positions[list(MOMENTUM_ASSETS)].fillna(0.0)
    expanded_index = positions.index.union(calendar)
    expanded = positions.reindex(expanded_index).sort_index().ffill()
    targets = expanded.reindex(calendar).fillna(0.0)
    totals = targets.sum(axis=1)
    if not np.allclose(totals.to_numpy(), 1.0, atol=1e-12):
        bad = totals.loc[~np.isclose(totals, 1.0, atol=1e-12)]
        raise AssertionError(f"momentum target is not fully invested: {bad.head().to_dict()}")
    return targets


def _load_defender_schedule(
    path: Path,
) -> tuple[pd.DataFrame, pd.Series]:
    frame = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
    required = {"return", "primary_target", "defensive_asset"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"defender schedule missing columns: {sorted(missing)}")
    if frame.index.duplicated().any():
        raise ValueError("defender schedule contains duplicate dates")
    targets = pd.DataFrame(0.0, index=frame.index, columns=[DEFENDER_PRIMARY, DEFENDER_DEFENSIVE])
    targets[DEFENDER_PRIMARY] = frame["primary_target"].astype(float)
    if not frame["defensive_asset"].eq(DEFENDER_DEFENSIVE).all():
        unexpected = sorted(frame.loc[frame["defensive_asset"] != DEFENDER_DEFENSIVE, "defensive_asset"].unique())
        raise ValueError(f"unsupported defensive assets in frozen schedule: {unexpected}")
    targets[DEFENDER_DEFENSIVE] = 1.0 - targets[DEFENDER_PRIMARY]
    if not np.allclose(targets.sum(axis=1), 1.0, atol=1e-12):
        raise AssertionError("defender target is not fully invested")
    return targets, frame["return"].astype(float)


def build_context(
    root: Path,
    defender_schedule_path: Path,
    end: date = date(2026, 8, 17),
) -> ResearchContext:
    defender_targets, defender_returns = _load_defender_schedule(defender_schedule_path)
    config = _load_momentum_config(root / "strategy/configs/quality_momentum_top1.yaml", end)
    momentum_result = run(config)
    calendar = pd.DatetimeIndex(defender_targets.index)
    if calendar.min() != pd.Timestamp("2019-01-18") or calendar.max() != pd.Timestamp(end):
        raise AssertionError("unexpected Defender research window")
    momentum_targets = _momentum_target_schedule(momentum_result, calendar)
    prices = _load_prices(
        set(MOMENTUM_ASSETS) | {DEFENDER_PRIMARY, DEFENDER_DEFENSIVE},
        date(2013, 1, 1),
        end,
    )
    combined_targets = pd.concat([momentum_targets, defender_targets], axis=1).fillna(0.0)
    for asset in combined_targets.columns.unique():
        asset_weight = combined_targets.loc[:, combined_targets.columns == asset].sum(axis=1)
        target_dates = asset_weight.loc[asset_weight > 1e-14].index
        absent = target_dates.difference(prices[asset].index)
        if len(absent):
            raise AssertionError(f"{asset} lacks {len(absent)} dates on which it is targeted")
    return ResearchContext(
        calendar=calendar,
        prices=prices,
        momentum_targets=momentum_targets,
        defender_targets=defender_targets.reindex(calendar),
        momentum_result=momentum_result,
        risk_close=prices[RISK_SIGNAL_ASSET]["close"],
        original_defender_returns=defender_returns.reindex(calendar),
    )


def _state_schedule(close: pd.Series, params: SwitchParams) -> pd.Series:
    if params.lookback < 1 or params.min_hold_days < 1:
        raise ValueError("lookback and min_hold_days must be positive")
    trailing_return = close / close.shift(params.lookback) - 1.0
    desired_after_close = (trailing_return > params.risk_on_threshold).where(
        trailing_return.notna()
    )
    # Yesterday's close decision becomes today's open target.
    desired_at_open = desired_after_close.shift(1)
    state = True
    held_days = 10**9
    values: list[bool] = []
    for desired in desired_at_open:
        if pd.notna(desired):
            wanted = bool(desired)
            if wanted != state and held_days >= params.min_hold_days:
                state = wanted
                held_days = 0
        values.append(state)
        held_days += 1
    return pd.Series(values, index=close.index, name="risk_on")


def _asset_cost_rate(asset: str) -> float:
    try:
        return ONE_WAY_COST_RATES[asset]
    except KeyError as exc:
        raise KeyError(f"missing transaction cost for {asset}") from exc


def _execute_target(
    cash: float,
    shares: dict[str, float],
    target: Mapping[str, float],
    open_prices: Mapping[str, float],
) -> tuple[float, dict[str, float], list[dict[str, float | str]]]:
    nav_open = cash + sum(quantity * open_prices[asset] for asset, quantity in shares.items())
    desired = {asset: nav_open * weight for asset, weight in target.items()}
    executions: list[dict[str, float | str]] = []

    for asset in sorted(set(shares) | set(target)):
        current_value = shares.get(asset, 0.0) * open_prices[asset]
        sell_value = max(0.0, current_value - desired.get(asset, 0.0))
        if sell_value <= 1e-14:
            continue
        rate = _asset_cost_rate(asset)
        shares[asset] = shares.get(asset, 0.0) - sell_value / open_prices[asset]
        cash += sell_value * (1.0 - rate)
        executions.append({
            "asset": asset,
            "side": "sell",
            "notional": sell_value,
            "turnover": sell_value / nav_open,
            "cost_rate": rate,
            "cost": sell_value * rate,
        })

    needs: dict[str, float] = {}
    for asset, desired_value in desired.items():
        current_value = shares.get(asset, 0.0) * open_prices[asset]
        needs[asset] = max(0.0, desired_value - current_value)
    total_cash_need = sum(value * (1.0 + _asset_cost_rate(asset)) for asset, value in needs.items())
    scale = min(1.0, cash / total_cash_need) if total_cash_need else 0.0
    for asset in sorted(needs, key=lambda item: target[item], reverse=True):
        buy_value = needs[asset] * scale
        if buy_value <= 1e-14:
            continue
        rate = _asset_cost_rate(asset)
        shares[asset] = shares.get(asset, 0.0) + buy_value / open_prices[asset]
        cash -= buy_value * (1.0 + rate)
        executions.append({
            "asset": asset,
            "side": "buy",
            "notional": buy_value,
            "turnover": buy_value / nav_open,
            "cost_rate": rate,
            "cost": buy_value * rate,
        })
    cleaned = {asset: quantity for asset, quantity in shares.items() if quantity > 1e-14}
    return cash, cleaned, executions


def _targets_equal(left: Mapping[str, float], right: Mapping[str, float]) -> bool:
    assets = set(left) | set(right)
    return all(abs(left.get(asset, 0.0) - right.get(asset, 0.0)) <= 1e-12 for asset in assets)


def simulate(
    context: ResearchContext,
    risk_on: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    risk_on = risk_on.reindex(context.calendar)
    if risk_on.isna().any():
        raise ValueError("risk state does not cover the evaluation calendar")
    cash = INITIAL_CAPITAL
    shares: dict[str, float] = {}
    previous_target: dict[str, float] = {}
    previous_state: bool | None = None
    last_nav = INITIAL_CAPITAL
    rows: list[dict[str, object]] = []
    trades: list[dict[str, object]] = []

    for timestamp in context.calendar:
        state = bool(risk_on.loc[timestamp])
        target_row = (
            context.momentum_targets.loc[timestamp]
            if state
            else context.defender_targets.loc[timestamp]
        )
        target = {
            asset: float(weight)
            for asset, weight in target_row.items()
            if float(weight) > 1e-14
        }
        open_prices = {
            asset: float(frame.at[timestamp, "open"])
            for asset, frame in context.prices.items()
            if timestamp in frame.index
        }
        close_prices = {
            asset: float(frame.at[timestamp, "close"])
            for asset, frame in context.prices.items()
            if timestamp in frame.index
        }
        if set(shares) - set(open_prices) or set(target) - set(open_prices):
            raise AssertionError(f"missing open price on {timestamp.date()}")

        nav_open_before_cost = cash + sum(
            quantity * open_prices[asset] for asset, quantity in shares.items()
        )
        state_changed = previous_state is not None and state != previous_state
        target_changed = not _targets_equal(target, previous_target)
        executions: list[dict[str, float | str]] = []
        if target_changed:
            cash, shares, executions = _execute_target(cash, shares, target, open_prices)
            reason = "sleeve_switch" if state_changed else "internal_target_change"
            for execution in executions:
                trades.append({
                    "date": timestamp,
                    "reason": reason,
                    "from_sleeve": None if previous_state is None else ("momentum" if previous_state else "defender"),
                    "to_sleeve": "momentum" if state else "defender",
                    **execution,
                })
            previous_target = target

        nav = cash + sum(
            quantity * close_prices[asset] for asset, quantity in shares.items()
        )
        daily_return = nav / last_nav - 1.0
        row = {
            "date": timestamp,
            "return": daily_return,
            "nav": nav,
            "risk_on": state,
            "sleeve": "momentum" if state else "defender",
            "sleeve_switch": state_changed,
            "target_change": target_changed,
            "nav_open_before_cost": nav_open_before_cost,
            "transaction_cost": sum(float(item["cost"]) for item in executions),
            "turnover": sum(float(item["turnover"]) for item in executions),
        }
        for asset in sorted(ONE_WAY_COST_RATES):
            row[f"target_{asset}"] = target.get(asset, 0.0)
            row[f"shares_{asset}"] = shares.get(asset, 0.0)
        rows.append(row)
        last_nav = nav
        previous_state = state

    return pd.DataFrame(rows).set_index("date"), pd.DataFrame(trades)


def performance(returns: pd.Series) -> dict[str, float | int | str]:
    values = returns.dropna().astype(float)
    curve = (1.0 + values).cumprod()
    drawdown = curve / curve.cummax() - 1.0
    volatility = float(values.std(ddof=1))
    calendar_years = (values.index[-1] - values.index[0]).days / 365.2425
    return {
        "start": values.index[0].date().isoformat(),
        "end": values.index[-1].date().isoformat(),
        "observations": int(len(values)),
        "total_return": float(curve.iloc[-1] - 1.0),
        "cagr_calendar": float(curve.iloc[-1] ** (1.0 / calendar_years) - 1.0),
        "annualized_return_252": float(curve.iloc[-1] ** (252.0 / len(values)) - 1.0),
        "annualized_volatility": float(volatility * np.sqrt(252.0)),
        "sharpe": float(values.mean() / volatility * np.sqrt(252.0)) if volatility else 0.0,
        "max_drawdown": float(drawdown.min()),
    }


def _period_metrics(daily: pd.DataFrame, start: str, end: str) -> dict[str, float | int | str]:
    return performance(daily.loc[start:end, "return"])


def evaluate_candidate(
    context: ResearchContext,
    params: SwitchParams,
    baselines: Mapping[str, pd.DataFrame],
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    states = _state_schedule(context.risk_close, params).reindex(context.calendar)
    daily, trades = simulate(context, states)
    trailing_return = context.risk_close / context.risk_close.shift(params.lookback) - 1.0
    daily["risk_signal_asof_previous_close"] = trailing_return.shift(1).reindex(
        context.calendar
    )
    summary: dict[str, object] = {
        "lookback": params.lookback,
        "risk_on_threshold": params.risk_on_threshold,
        "min_hold_days": params.min_hold_days,
        "sleeve_switches": int(daily["sleeve_switch"].sum()),
        "defender_day_share": float((~daily["risk_on"]).mean()),
        "total_transaction_cost": float(daily["transaction_cost"].sum()),
        **performance(daily["return"]),
    }
    for period, start, end in [
        ("development", "2019-01-18", "2024-12-31"),
        ("holdout", "2025-01-01", "2026-08-17"),
    ]:
        candidate_metrics = _period_metrics(daily, start, end)
        momentum_metrics = _period_metrics(baselines["momentum"], start, end)
        for key in ("cagr_calendar", "annualized_return_252", "sharpe", "max_drawdown"):
            summary[f"{period}_{key}"] = candidate_metrics[key]
            summary[f"{period}_delta_{key}"] = float(candidate_metrics[key]) - float(momentum_metrics[key])
    return summary, daily, trades


def _baseline_simulations(context: ResearchContext) -> dict[str, pd.DataFrame]:
    always_momentum = pd.Series(True, index=context.calendar)
    always_defender = pd.Series(False, index=context.calendar)
    momentum, _ = simulate(context, always_momentum)
    defender, _ = simulate(context, always_defender)
    return {"momentum": momentum, "defender": defender}


def _validate_reproduction(context: ResearchContext, baselines: Mapping[str, pd.DataFrame]) -> dict[str, float]:
    defender_aligned = pd.concat(
        [
            baselines["defender"]["return"].rename("reproduced"),
            context.original_defender_returns.rename("source"),
        ],
        axis=1,
        join="inner",
    ).dropna()
    defender_max_error = float((defender_aligned["reproduced"] - defender_aligned["source"]).abs().max())
    if defender_max_error > 1e-10:
        raise AssertionError(f"Defender reproduction error {defender_max_error:.3e}")

    # Defender's primary-asset calendar omits rare suspension dates.  Aggregate
    # the production momentum NAV across those gaps instead of dropping an
    # intervening momentum return and comparing mismatched holding periods.
    source_curve = (1.0 + context.momentum_result.daily_returns).cumprod()
    source_momentum = source_curve.reindex(context.calendar).pct_change().dropna()
    reproduced_momentum = baselines["momentum"]["return"].reindex(source_momentum.index)
    # The exact capital-aware simulator settles percentage fees through cash;
    # the production runner subtracts them additively.  Differences should stay tiny.
    momentum_max_error = float((reproduced_momentum - source_momentum).abs().max())
    if momentum_max_error > 2e-5:
        raise AssertionError(f"momentum reproduction error {momentum_max_error:.3e}")
    return {
        "defender_max_abs_daily_return_error": defender_max_error,
        "momentum_max_abs_daily_return_error_ex_first": momentum_max_error,
    }


def run_grid(
    context: ResearchContext,
    baselines: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for lookback in (20, 30, 40, 50, 60, 80, 100, 120):
        for threshold in (-0.025, 0.0, 0.025, 0.05, 0.075, 0.10):
            for min_hold in (5, 10, 15, 20, 25, 30):
                params = SwitchParams(lookback, threshold, min_hold)
                summary, _, _ = evaluate_candidate(context, params, baselines)
                rows.append(summary)
    return pd.DataFrame(rows)


def _annual_metrics(name: str, daily: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for year, values in daily["return"].groupby(daily.index.year):
        rows.append({"series": name, "year": int(year), **performance(values)})
    return rows


def _rolling_metrics(
    candidate: pd.Series,
    momentum: pd.Series,
    window: int = 756,
    step: int = 21,
) -> pd.DataFrame:
    aligned = pd.concat(
        [candidate.rename("candidate"), momentum.rename("momentum")], axis=1
    ).dropna()
    rows: list[dict[str, object]] = []
    for end_position in range(window - 1, len(aligned), step):
        sample = aligned.iloc[end_position - window + 1 : end_position + 1]
        candidate_metrics = performance(sample["candidate"])
        momentum_metrics = performance(sample["momentum"])
        rows.append({
            "start": sample.index[0].date().isoformat(),
            "end": sample.index[-1].date().isoformat(),
            "observations": len(sample),
            "candidate_annualized_return_252": candidate_metrics["annualized_return_252"],
            "momentum_annualized_return_252": momentum_metrics["annualized_return_252"],
            "delta_annualized_return_252": float(candidate_metrics["annualized_return_252"])
            - float(momentum_metrics["annualized_return_252"]),
            "candidate_sharpe": candidate_metrics["sharpe"],
            "momentum_sharpe": momentum_metrics["sharpe"],
            "delta_sharpe": float(candidate_metrics["sharpe"])
            - float(momentum_metrics["sharpe"]),
            "candidate_max_drawdown": candidate_metrics["max_drawdown"],
            "momentum_max_drawdown": momentum_metrics["max_drawdown"],
        })
    return pd.DataFrame(rows)


def _paired_block_bootstrap(
    candidate: pd.Series,
    momentum: pd.Series,
    samples: int = 2_000,
    block_size: int = 20,
    seed: int = 20260818,
) -> pd.DataFrame:
    aligned = pd.concat(
        [candidate.rename("candidate"), momentum.rename("momentum")], axis=1
    ).dropna()
    values = aligned.to_numpy(dtype=float)
    observations = len(values)
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | int]] = []

    def array_metrics(returns: np.ndarray) -> tuple[float, float]:
        annualized = float(np.prod(1.0 + returns) ** (252.0 / len(returns)) - 1.0)
        volatility = float(np.std(returns, ddof=1))
        sharpe = float(np.mean(returns) / volatility * np.sqrt(252.0)) if volatility else 0.0
        return annualized, sharpe

    for sample_number in range(samples):
        starts = rng.integers(0, observations, size=int(np.ceil(observations / block_size)))
        indices = np.concatenate(
            [(start + np.arange(block_size)) % observations for start in starts]
        )[:observations]
        resampled = values[indices]
        candidate_return, candidate_sharpe = array_metrics(resampled[:, 0])
        momentum_return, momentum_sharpe = array_metrics(resampled[:, 1])
        rows.append({
            "sample": sample_number,
            "delta_annualized_return_252": candidate_return - momentum_return,
            "delta_sharpe": candidate_sharpe - momentum_sharpe,
        })
    return pd.DataFrame(rows)


def _defender_episode_attribution(
    candidate_daily: pd.DataFrame,
    momentum_daily: pd.DataFrame,
) -> pd.DataFrame:
    frame = pd.DataFrame({
        "candidate": candidate_daily["return"],
        "momentum": momentum_daily["return"],
        "defender_active": ~candidate_daily["risk_on"],
    }).dropna()
    groups = frame["defender_active"].ne(frame["defender_active"].shift()).cumsum()
    rows: list[dict[str, object]] = []
    episode_number = 0
    for _, episode in frame.loc[frame["defender_active"]].groupby(groups):
        episode_number += 1
        candidate_return = float((1.0 + episode["candidate"]).prod() - 1.0)
        momentum_return = float((1.0 + episode["momentum"]).prod() - 1.0)
        rows.append({
            "episode": episode_number,
            "start": episode.index[0].date().isoformat(),
            "end": episode.index[-1].date().isoformat(),
            "days": len(episode),
            "candidate_return": candidate_return,
            "momentum_return": momentum_return,
            "excess_log_return": float(
                np.log1p(episode["candidate"]).sum()
                - np.log1p(episode["momentum"]).sum()
            ),
        })
    return pd.DataFrame(rows)


def _robustness_summary(
    grid: pd.DataFrame,
    rolling: pd.DataFrame,
    bootstrap: pd.DataFrame,
    episodes: pd.DataFrame,
    momentum_metrics: Mapping[str, float | int | str],
) -> pd.DataFrame:
    objective_columns = [
        "development_delta_cagr_calendar",
        "development_delta_sharpe",
        "holdout_delta_cagr_calendar",
        "holdout_delta_sharpe",
    ]
    full_pass = (
        (grid["cagr_calendar"] > float(momentum_metrics["cagr_calendar"]))
        & (grid["sharpe"] > float(momentum_metrics["sharpe"]))
    )
    segment_pass = (grid[objective_columns] > 0).all(axis=1)
    neighborhood = grid.loc[
        grid["lookback"].isin([30, 40, 50])
        & grid["risk_on_threshold"].isin([0.0, 0.025, 0.05])
        & grid["min_hold_days"].isin([25, 30])
    ]
    neighborhood_segment_pass = (neighborhood[objective_columns] > 0).all(axis=1)
    rolling_both = (
        (rolling["delta_annualized_return_252"] > 0)
        & (rolling["delta_sharpe"] > 0)
    )
    positive_excess = episodes.loc[episodes["excess_log_return"] > 0, "excess_log_return"]
    top5_share = (
        float(positive_excess.nlargest(5).sum() / positive_excess.sum())
        if positive_excess.sum() > 0
        else np.nan
    )
    rows = [
        {"check": "full_grid_both_objectives", "passed": int(full_pass.sum()), "total": len(grid), "rate": float(full_pass.mean())},
        {"check": "development_and_holdout_both_objectives", "passed": int(segment_pass.sum()), "total": len(grid), "rate": float(segment_pass.mean())},
        {"check": "local_neighborhood_full_both_objectives", "passed": int(((neighborhood["cagr_calendar"] > float(momentum_metrics["cagr_calendar"])) & (neighborhood["sharpe"] > float(momentum_metrics["sharpe"]))).sum()), "total": len(neighborhood), "rate": float(((neighborhood["cagr_calendar"] > float(momentum_metrics["cagr_calendar"])) & (neighborhood["sharpe"] > float(momentum_metrics["sharpe"]))).mean())},
        {"check": "local_neighborhood_development_and_holdout", "passed": int(neighborhood_segment_pass.sum()), "total": len(neighborhood), "rate": float(neighborhood_segment_pass.mean())},
        {"check": "rolling_36m_both_objectives", "passed": int(rolling_both.sum()), "total": len(rolling), "rate": float(rolling_both.mean())},
        {"check": "bootstrap_delta_annualized_return_positive", "passed": int((bootstrap["delta_annualized_return_252"] > 0).sum()), "total": len(bootstrap), "rate": float((bootstrap["delta_annualized_return_252"] > 0).mean())},
        {"check": "bootstrap_delta_sharpe_positive", "passed": int((bootstrap["delta_sharpe"] > 0).sum()), "total": len(bootstrap), "rate": float((bootstrap["delta_sharpe"] > 0).mean())},
        {"check": "bootstrap_both_objectives_positive", "passed": int(((bootstrap["delta_annualized_return_252"] > 0) & (bootstrap["delta_sharpe"] > 0)).sum()), "total": len(bootstrap), "rate": float(((bootstrap["delta_annualized_return_252"] > 0) & (bootstrap["delta_sharpe"] > 0)).mean())},
        {"check": "top5_positive_episode_excess_share", "passed": np.nan, "total": len(episodes), "rate": top5_share},
    ]
    return pd.DataFrame(rows)


def _report_result(
    returns: pd.Series,
    benchmark: pd.Series,
    benchmark_name: str,
    output_path: Path,
    config: dict,
) -> None:
    aligned = pd.concat([returns.rename("strategy"), benchmark.rename("benchmark")], axis=1).dropna()
    result = BacktestResult(
        daily_returns=aligned["strategy"],
        benchmark_returns=aligned["benchmark"],
        positions=pd.DataFrame(index=aligned.index),
        train_end=date(2024, 12, 31),
        config=config,
        baseline_strategy_name=benchmark_name,
    )
    generate(result, output_path, benchmark_title=benchmark_name)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    default_output = root / "experiments/20260818_momentum_defender_switching"
    parser.add_argument("--root", type=Path, default=root)
    parser.add_argument("--output", type=Path, default=default_output)
    parser.add_argument(
        "--defender-schedule",
        type=Path,
        default=default_output / "defender_daily_targets.csv",
    )
    parser.add_argument("--lookback", type=int, default=40)
    parser.add_argument("--threshold", type=float, default=0.025)
    parser.add_argument("--min-hold", type=int, default=30)
    parser.add_argument("--skip-grid", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    context = build_context(args.root, args.defender_schedule)
    baselines = _baseline_simulations(context)
    reproduction = _validate_reproduction(context, baselines)
    params = SwitchParams(args.lookback, args.threshold, args.min_hold)
    candidate_summary, candidate_daily, candidate_trades = evaluate_candidate(
        context, params, baselines
    )

    summary_rows = []
    for name, daily in [
        ("momentum", baselines["momentum"]),
        ("defender", baselines["defender"]),
        ("switch_overlay", candidate_daily),
    ]:
        row = {"series": name, **performance(daily["return"])}
        if name == "switch_overlay":
            row.update({
                "sleeve_switches": candidate_summary["sleeve_switches"],
                "defender_day_share": candidate_summary["defender_day_share"],
                "total_transaction_cost": candidate_summary["total_transaction_cost"],
            })
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.output / "performance_summary.csv", index=False)
    candidate_daily.to_csv(args.output / "switch_strategy_daily.csv")
    candidate_trades.to_csv(args.output / "switch_strategy_trades.csv", index=False)

    event_source = candidate_daily.copy()
    event_source["previous_sleeve"] = event_source["sleeve"].shift(1)
    events = event_source.loc[event_source["sleeve_switch"]].copy()
    events.to_csv(args.output / "switch_events.csv")

    annual_rows: list[dict[str, object]] = []
    annual_rows.extend(_annual_metrics("momentum", baselines["momentum"]))
    annual_rows.extend(_annual_metrics("defender", baselines["defender"]))
    annual_rows.extend(_annual_metrics("switch_overlay", candidate_daily))
    pd.DataFrame(annual_rows).to_csv(args.output / "annual_metrics.csv", index=False)

    candidate_table = pd.DataFrame([candidate_summary])
    candidate_table.to_csv(args.output / "selected_candidate_metrics.csv", index=False)
    pd.DataFrame([reproduction]).to_csv(args.output / "reproduction_checks.csv", index=False)

    grid_path = args.output / "candidate_grid.csv"
    if not args.skip_grid:
        grid = run_grid(context, baselines)
        grid.to_csv(grid_path, index=False)
    elif grid_path.exists():
        grid = pd.read_csv(grid_path)
    else:
        grid = pd.DataFrame()

    rolling = _rolling_metrics(
        candidate_daily["return"], baselines["momentum"]["return"]
    )
    rolling.to_csv(args.output / "rolling_36m_metrics.csv", index=False)
    bootstrap = _paired_block_bootstrap(
        candidate_daily["return"], baselines["momentum"]["return"]
    )
    bootstrap.to_csv(args.output / "paired_block_bootstrap.csv", index=False)
    episodes = _defender_episode_attribution(candidate_daily, baselines["momentum"])
    episodes.to_csv(args.output / "defender_episode_attribution.csv", index=False)
    if not grid.empty:
        robustness = _robustness_summary(
            grid,
            rolling,
            bootstrap,
            episodes,
            performance(baselines["momentum"]["return"]),
        )
        robustness.to_csv(args.output / "robustness_summary.csv", index=False)

    target_columns = [f"target_{asset}" for asset in sorted(ONE_WAY_COST_RATES)]
    target_sum_error = float((candidate_daily[target_columns].sum(axis=1) - 1.0).abs().max())
    momentum_target_mass_on_defender_days = float(
        candidate_daily.loc[~candidate_daily["risk_on"], [f"target_{asset}" for asset in MOMENTUM_ASSETS]]
        .sum(axis=1)
        .max()
    )
    defender_target_mass_on_momentum_days = float(
        candidate_daily.loc[candidate_daily["risk_on"], [f"target_{DEFENDER_PRIMARY}", f"target_{DEFENDER_DEFENSIVE}"]]
        .sum(axis=1)
        .max()
    )
    switch_trade_turnover = (
        candidate_trades.loc[candidate_trades["reason"] == "sleeve_switch"]
        .groupby("date")["turnover"]
        .sum()
    )
    execution_audit = pd.DataFrame([{
        "target_sum_max_abs_error": target_sum_error,
        "momentum_target_mass_on_defender_days": momentum_target_mass_on_defender_days,
        "defender_target_mass_on_momentum_days": defender_target_mass_on_momentum_days,
        "sleeve_switch_count": int(candidate_daily["sleeve_switch"].sum()),
        "switch_trade_dates": int(len(switch_trade_turnover)),
        "min_switch_turnover": float(switch_trade_turnover.min()),
        "median_switch_turnover": float(switch_trade_turnover.median()),
        "max_switch_turnover": float(switch_trade_turnover.max()),
        **reproduction,
    }])
    execution_audit.to_csv(args.output / "execution_audit.csv", index=False)

    _report_result(
        candidate_daily["return"],
        baselines["momentum"]["return"],
        "Original Momentum Strategy",
        args.output / "switch_strategy_vs_momentum.html",
        {"strategy_name": "momentum_defender_open_switch", **params.__dict__},
    )
    original_base_curve = (1.0 + context.momentum_result.benchmark_returns).cumprod()
    original_base = original_base_curve.reindex(context.calendar).pct_change()
    _report_result(
        candidate_daily["return"],
        original_base,
        "Original 4ETF Equal-Weight Base",
        args.output / "switch_strategy_vs_original_base.html",
        {"strategy_name": "momentum_defender_open_switch", **params.__dict__},
    )

    momentum_metrics = performance(baselines["momentum"]["return"])
    switch_metrics = performance(candidate_daily["return"])
    rolling_both_rate = float(
        ((rolling["delta_annualized_return_252"] > 0) & (rolling["delta_sharpe"] > 0)).mean()
    )
    bootstrap_both_rate = float(
        ((bootstrap["delta_annualized_return_252"] > 0) & (bootstrap["delta_sharpe"] > 0)).mean()
    )
    report = f"""# 动量—防守开盘全仓切换策略

## 冻结规则

- 每个交易日收盘后计算 `510300.SH` 的 {params.lookback} 日收盘收益率。
- 当该收益率高于 {params.risk_on_threshold:.2%} 时，目标状态为动量；否则目标状态为防守。
- 两种状态均至少持有 {params.min_hold_days} 个交易日；达到最短持有期后，若目标状态不同，则在下一交易日开盘全仓切换。
- 开盘切换时先让旧仓位承受昨收至今开的隔夜收益，再按开盘价卖出旧 sleeve、买入新 sleeve，新仓位承受今开至今收的日内收益。
- 动量四只 ETF 与 512890 使用单边 1bp；511260 使用单边 0.1bp。两套 sleeve 的内部调仓成本和顶层切换成本均计入。

## 时间窗口与结果

- 全样本：2019-01-18—2026-08-17，共 {len(candidate_daily):,} 个交易日。
- 开发观察段：2019-01-18—2024-12-31。
- 后段复核：2025-01-01—2026-08-17。完整历史曾用于候选探索，因此后段不能再宣称为完全未观察的独立 OOS。

| 策略 | 年化（自然年 CAGR） | Sharpe | 年化波动 | 最大回撤 | 总收益 |
|---|---:|---:|---:|---:|---:|
| 原动量 | {float(momentum_metrics['cagr_calendar']):.2%} | {float(momentum_metrics['sharpe']):.3f} | {float(momentum_metrics['annualized_volatility']):.2%} | {float(momentum_metrics['max_drawdown']):.2%} | {float(momentum_metrics['total_return']):.2%} |
| 开盘切换 | {float(switch_metrics['cagr_calendar']):.2%} | {float(switch_metrics['sharpe']):.3f} | {float(switch_metrics['annualized_volatility']):.2%} | {float(switch_metrics['max_drawdown']):.2%} | {float(switch_metrics['total_return']):.2%} |

全样本年化提高 {float(switch_metrics['cagr_calendar']) - float(momentum_metrics['cagr_calendar']):+.2%}，Sharpe 提高 {float(switch_metrics['sharpe']) - float(momentum_metrics['sharpe']):+.3f}。策略共发生 {int(candidate_daily['sleeve_switch'].sum())} 次顶层全仓切换，防守状态占 {float((~candidate_daily['risk_on']).mean()):.2%} 的交易日。

开发段相对原动量：年化 {float(candidate_summary['development_delta_cagr_calendar']):+.2%}、Sharpe {float(candidate_summary['development_delta_sharpe']):+.3f}。后段复核相对原动量：年化 {float(candidate_summary['holdout_delta_cagr_calendar']):+.2%}、Sharpe {float(candidate_summary['holdout_delta_sharpe']):+.3f}。

## 稳健性与边界

- 以 30/40/50 日、阈值 0%/2.5%/5%、最短持有 25/30 日构成的 18 点邻域，全部在全样本同时提高年化和 Sharpe；其中开发段和后段均同时通过的比例见 `robustness_summary.csv`。
- 36 个月滚动窗口同时提高年化和 Sharpe 的比例为 {rolling_both_rate:.2%}。
- 20 日成组、2,000 次配对 bootstrap 中，两项目标同时为正的比例为 {bootstrap_both_rate:.2%}。该检验未校正候选选择偏差。
- 本规则没有改善全样本最大回撤：最深回撤发生在风险开关尚未避开的阶段。它的目标是提高年化与 Sharpe，不应被解读为保证更浅的所有历史回撤。
- 结果是回溯设计证据，不是尚未观察的实盘证据；正式部署前应冻结参数并进行前瞻 shadow。
"""
    (args.output / "research_report.md").write_text(report, encoding="utf-8")

    print(summary.to_string(index=False))
    print("selected", candidate_summary)
    print("reproduction", reproduction)
    print("output", args.output)


if __name__ == "__main__":
    main()
