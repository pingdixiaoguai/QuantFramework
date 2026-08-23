"""Causal listing-aware 2013 extension of the formal ETF rotation.

The formal strategy cannot literally exist before 512890.SH was listed.  This
formal history extension therefore applies the frozen champion parameters to
510880.SH until the first 512890 close is observable, then uses 512890.SH from
the following open onward.  Monthly selection only considers already listed,
tradeable ETFs with enough history for the requested ranking signal.
"""

from __future__ import annotations

import io
import json
from dataclasses import asdict
from datetime import date
from html import escape
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from .defender_opt_v2 import (
    CostRateSpec,
    _asof_price,
    _asset_cost_rate,
    _execute_portfolio_target,
    _indexed_market,
)
from .grid_reproduction import INITIAL_CAPITAL, TRADING_DAYS
from .relative_defender_champion import (
    champion_params,
    target_schedule as champion_target_schedule,
)
from .relative_defender_rotation import (
    BASE_PRIMARY_ASSET,
    DEFENSIVE_ASSET,
    ROTATION_ASSETS,
    ROTATION_COST_RATES,
    RelativeDefenderRotationParams,
    _stable_extreme,
    load_rotation_market,
    rotation_params,
)


START_DATE = date(2013, 7, 1)
BRIDGE_SIGNAL_ASSET = "510880.SH"
STRATEGY_ID = "relative_defender_rotation_2013_listing_aware"
PROMOTION_DATE = "2026-08-22"
OUTPUT = (
    Path(__file__).parent
    / "deliverable"
    / "relative_defender_rotation_2013_report.html"
)
PREFIX = "relative_defender_rotation_2013"


def _clean_market(
    market: Mapping[str, pd.DataFrame] | None,
    end: date | None,
    params: RelativeDefenderRotationParams,
) -> dict[str, pd.DataFrame]:
    prices = (
        load_rotation_market(start=date(1900, 1, 1), end=end, params=params)
        if market is None
        else {asset: frame.copy() for asset, frame in market.items()}
    )
    required = {*params.assets, params.defensive_asset}
    missing = sorted(required - set(prices))
    if missing:
        raise RuntimeError(f"missing local data for: {', '.join(missing)}")
    result: dict[str, pd.DataFrame] = {}
    for asset, frame in prices.items():
        cleaned = frame.copy()
        cleaned["date"] = pd.to_datetime(cleaned["date"])
        cleaned = cleaned.sort_values("date").drop_duplicates("date")
        if end is not None:
            cleaned = cleaned.loc[cleaned["date"] <= pd.Timestamp(end)]
        required_columns = ["date", "open", "high", "low", "close"]
        if cleaned.empty or cleaned[required_columns].isna().any().any():
            raise ValueError(f"price data for {asset} is empty or incomplete")
        if (cleaned[["open", "high", "low", "close"]] <= 0.0).any().any():
            raise ValueError(f"price data for {asset} contains non-positive OHLC")
        result[asset] = cleaned.reset_index(drop=True)
    return result


def _union_calendar(
    market: Mapping[str, pd.DataFrame],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DatetimeIndex:
    values: set[pd.Timestamp] = set()
    for frame in market.values():
        dates = pd.to_datetime(frame["date"])
        values.update(pd.Timestamp(item) for item in dates[(dates >= start) & (dates <= end)])
    calendar = pd.DatetimeIndex(sorted(values), name="date")
    if calendar.empty:
        raise ValueError("no trading dates in requested interval")
    return calendar


def _schedule_row_before(
    schedule: pd.DataFrame,
    timestamp: pd.Timestamp,
) -> tuple[pd.Timestamp, pd.Series] | None:
    position = int(schedule.index.searchsorted(timestamp, side="left")) - 1
    if position < 0:
        return None
    return pd.Timestamp(schedule.index[position]), schedule.iloc[position]


def _aligned_signal_frame(
    schedule: pd.DataFrame,
    calendar: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, pd.Series]:
    aligned = schedule.reindex(calendar).ffill()
    observations = pd.Series(schedule.index, index=schedule.index, dtype="datetime64[ns]")
    observation_dates = observations.reindex(calendar).ffill()
    return aligned, observation_dates


def hybrid_champion_schedule(
    market: Mapping[str, pd.DataFrame],
    calendar: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Use 510880 signals before 512890 is observable, then switch causally."""
    bridge = champion_target_schedule(
        market[BRIDGE_SIGNAL_ASSET],
        champion_params(BRIDGE_SIGNAL_ASSET),
    )
    anchor = champion_target_schedule(
        market[BASE_PRIMARY_ASSET],
        champion_params(BASE_PRIMARY_ASSET),
    )
    bridge_aligned, bridge_observation = _aligned_signal_frame(bridge, calendar)
    anchor_aligned, anchor_observation = _aligned_signal_frame(anchor, calendar)
    anchor_first_close = pd.Timestamp(anchor.index.min())
    close_uses_anchor = calendar >= anchor_first_close

    schedule = bridge_aligned.copy()
    common = schedule.columns.intersection(anchor_aligned.columns)
    schedule.loc[close_uses_anchor, common] = anchor_aligned.loc[
        close_uses_anchor, common
    ]
    schedule["signal_anchor_asset"] = np.where(
        close_uses_anchor,
        BASE_PRIMARY_ASSET,
        BRIDGE_SIGNAL_ASSET,
    )
    schedule["signal_observation_date"] = bridge_observation
    schedule.loc[close_uses_anchor, "signal_observation_date"] = (
        anchor_observation.loc[close_uses_anchor]
    )

    primary_targets: list[float] = []
    execution_sources: list[str] = []
    execution_observations: list[pd.Timestamp] = []
    execution_active: list[bool] = []
    execution_base_reasons: list[str] = []
    for timestamp in calendar:
        source = anchor if timestamp > anchor_first_close else bridge
        source_asset = (
            BASE_PRIMARY_ASSET
            if timestamp > anchor_first_close
            else BRIDGE_SIGNAL_ASSET
        )
        prior = _schedule_row_before(source, pd.Timestamp(timestamp))
        if prior is None:
            primary_targets.append(1.0)
            execution_sources.append(source_asset)
            execution_observations.append(pd.NaT)
            execution_active.append(False)
            execution_base_reasons.append("initial_buy")
            continue
        observation_date, row = prior
        primary_targets.append(float(row["signal_primary_target"]))
        execution_sources.append(source_asset)
        execution_observations.append(observation_date)
        execution_active.append(bool(row["signal_full_override_active"]))
        execution_base_reasons.append(str(row["signal_base_reason"]))

    active = pd.Series(execution_active, index=calendar, dtype=bool)
    was_active = active.shift(1, fill_value=False)
    schedule["primary_target"] = np.asarray(primary_targets, dtype=float)
    schedule["defensive_target"] = 1.0 - schedule["primary_target"]
    schedule["execution_signal_source_asset"] = execution_sources
    schedule["execution_signal_observation_date"] = execution_observations
    schedule["execution_full_override_active"] = active
    schedule["execution_reason"] = np.select(
        [active, was_active & ~active],
        ["champion_full_override", "champion_full_override_exit"],
        default=execution_base_reasons,
    )
    schedule.index.name = "date"
    return schedule


def _trailing_return_panel(
    market: Mapping[str, pd.DataFrame],
    assets: tuple[str, ...],
    calendar: pd.DatetimeIndex,
    lookback: int,
) -> pd.DataFrame:
    panel = pd.DataFrame(index=calendar, columns=assets, dtype=float)
    for asset in assets:
        frame = market[asset].set_index("date").sort_index()
        close = frame["close"].astype(float)
        values = close / close.shift(lookback) - 1.0
        panel[asset] = values.reindex(calendar).ffill()
    return panel


def _open_assets_on(
    market: Mapping[str, pd.DataFrame],
    assets: tuple[str, ...],
    timestamp: pd.Timestamp,
) -> list[str]:
    result: list[str] = []
    for asset in assets:
        frame = market[asset]
        row = frame.loc[frame["date"].eq(timestamp), "open"]
        if not row.empty and pd.notna(row.iloc[-1]) and float(row.iloc[-1]) > 0.0:
            result.append(asset)
    return result


def listing_aware_rotation_schedule(
    market: Mapping[str, pd.DataFrame],
    champion_schedule: pd.DataFrame,
    full_calendar: pd.DatetimeIndex,
    backtest_calendar: pd.DatetimeIndex,
    params: RelativeDefenderRotationParams,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select monthly only among listed/tradeable ETFs with causal history."""
    reversal = _trailing_return_panel(
        market, params.assets, full_calendar, params.reversal_lookback_days
    )
    trend = _trailing_return_panel(
        market, params.assets, full_calendar, params.trend_lookback_days
    )
    regime = _trailing_return_panel(
        market,
        (BRIDGE_SIGNAL_ASSET, BASE_PRIMARY_ASSET),
        full_calendar,
        params.regime_lookback_days,
    )

    selected_asset: str | None = None
    selected_reason = ""
    rows: list[dict[str, object]] = []
    events: list[dict[str, object]] = []
    for position, execution_date in enumerate(backtest_calendar):
        execution_date = pd.Timestamp(execution_date)
        previous_date = (
            pd.Timestamp(full_calendar[full_calendar.get_loc(execution_date) - 1])
            if full_calendar.get_loc(execution_date) > 0
            else execution_date
        )
        month_changed = (
            position == 0
            or execution_date.to_period("M")
            != pd.Timestamp(backtest_calendar[position - 1]).to_period("M")
        )
        if month_changed:
            listed = _open_assets_on(market, params.assets, execution_date)
            if not listed:
                raise RuntimeError(f"no listed rotation ETF at {execution_date.date()}")
            signal_row = champion_schedule.loc[previous_date]
            range_location = float(signal_row["range_location"])
            signal_anchor = str(signal_row["signal_anchor_asset"])
            regime_return = float(regime.at[previous_date, signal_anchor])
            low_scene = (
                np.isfinite(range_location)
                and range_location <= params.range_threshold
            )
            weak_regime = (
                np.isfinite(regime_return)
                and regime_return <= params.regime_return_ceiling
            )
            warmup_reversal = (
                low_scene
                and params.warmup_low_scene_reversal
                and not np.isfinite(regime_return)
            )
            use_reversal = low_scene and (weak_regime or warmup_reversal)
            source = reversal if use_reversal else trend
            values = source.loc[previous_date, listed]
            next_asset = _stable_extreme(
                values,
                find_maximum=not use_reversal,
            )
            next_reason = "low_scene_reversal" if use_reversal else "long_term_trend"
            if next_asset is None:
                next_asset = listed[0]
                next_reason = "insufficient_history_first_listed"
            events.append({
                "signal_date": previous_date,
                "execution_date": execution_date,
                "old_selected_asset": selected_asset,
                "new_selected_asset": next_asset,
                "selection_reason": next_reason,
                "signal_anchor_asset": signal_anchor,
                "range_location": range_location,
                "anchor_regime_return_180": regime_return,
                "listed_assets": "|".join(listed),
                "ranking_eligible_assets": "|".join(
                    str(asset) for asset in values.index[np.isfinite(values.to_numpy(float))]
                ),
                "selected_reversal_return": float(reversal.at[previous_date, next_asset]),
                "selected_trend_return": float(trend.at[previous_date, next_asset]),
            })
            selected_asset = next_asset
            selected_reason = next_reason

        if selected_asset is None:
            raise RuntimeError("rotation selection was not initialized")
        fractions = {asset: 0.0 for asset in params.assets}
        fractions[selected_asset] = 1.0
        rows.append({
            "date": execution_date,
            "selected_asset": selected_asset,
            "selection_reason": selected_reason,
            "signal_source_asset": champion_schedule.at[
                execution_date, "execution_signal_source_asset"
            ],
            "signal_observation_date": champion_schedule.at[
                execution_date, "execution_signal_observation_date"
            ],
            **{
                f"primary_fraction_{asset}": weight
                for asset, weight in fractions.items()
            },
        })
    return pd.DataFrame(rows).set_index("date"), pd.DataFrame(events)


def _performance_metrics(daily: pd.DataFrame) -> dict[str, float | int]:
    returns = daily["nav"].pct_change().astype(float)
    returns.iloc[0] = float(daily["nav"].iloc[0] / INITIAL_CAPITAL - 1.0)
    curve = (1.0 + returns).cumprod()
    stdev = float(returns.std(ddof=1)) if len(returns) > 1 else 0.0
    drawdown = curve / curve.cummax() - 1.0
    return {
        "observations": int(len(returns)),
        "final_nav": float(curve.iloc[-1]),
        "total_return": float(curve.iloc[-1] - 1.0),
        "annualized_return": float(
            curve.iloc[-1] ** (TRADING_DAYS / len(returns)) - 1.0
        ),
        "annualized_volatility": float(stdev * np.sqrt(TRADING_DAYS)),
        "sharpe": (
            float(returns.mean() / stdev * np.sqrt(TRADING_DAYS))
            if stdev
            else 0.0
        ),
        "max_drawdown": float(drawdown.min()),
    }


def _simulate(
    market: Mapping[str, pd.DataFrame],
    schedule: pd.DataFrame,
    selection: pd.DataFrame,
    params: RelativeDefenderRotationParams,
    cost_rates: CostRateSpec,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float | int]]:
    indexed = _indexed_market(market)
    calendar = pd.DatetimeIndex(schedule.index)
    cash = INITIAL_CAPITAL
    shares: dict[str, float] = {}
    previous_target: dict[str, float] = {}
    unavailable_targets: set[str] = set()
    previous_closes: dict[str, float] = {}
    previous_close_nav = INITIAL_CAPITAL
    previous_close_weights = {
        asset: 0.0 for asset in (*params.assets, params.defensive_asset)
    }
    previous_close_cash_weight = 1.0
    last_nav = 0.0
    rows: list[dict[str, object]] = []
    trades: list[dict[str, object]] = []
    total_cost = 0.0
    gross_pnl = {asset: 0.0 for asset in (*params.assets, params.defensive_asset)}

    for position, timestamp in enumerate(calendar):
        timestamp = pd.Timestamp(timestamp)
        primary_weight = float(schedule.at[timestamp, "primary_target"])
        selected_asset = str(selection.at[timestamp, "selected_asset"])
        target: dict[str, float] = {}
        if primary_weight > 1e-14:
            target[selected_asset] = primary_weight
        if 1.0 - primary_weight > 1e-14:
            target[params.defensive_asset] = 1.0 - primary_weight

        open_prices = {
            asset: float(frame.at[timestamp, "open"])
            for asset, frame in indexed.items()
            if timestamp in frame.index
            and pd.notna(frame.at[timestamp, "open"])
            and float(frame.at[timestamp, "open"]) > 0.0
        }
        mark_open = {
            asset: (_asof_price(frame, timestamp, "close") or 0.0)
            for asset, frame in indexed.items()
        }
        mark_open.update(open_prices)
        day_gross: dict[str, float] = {}
        day_cost: dict[str, float] = {}
        for asset, quantity in shares.items():
            if asset in previous_closes:
                pnl = quantity * (
                    mark_open.get(asset, previous_closes[asset])
                    - previous_closes[asset]
                )
                gross_pnl[asset] = gross_pnl.get(asset, 0.0) + pnl
                day_gross[asset] = day_gross.get(asset, 0.0) + pnl

        open_nav_before_trade = cash + sum(
            quantity * mark_open.get(asset, 0.0)
            for asset, quantity in shares.items()
        )
        overnight_gross_return = (
            open_nav_before_trade / previous_close_nav - 1.0
            if position > 0
            else 0.0
        )

        retry_available = any(asset in open_prices for asset in unavailable_targets)
        executions: list[dict[str, float | str]] = []
        if target != previous_target or retry_available:
            cash, shares, executions = _execute_portfolio_target(
                cash,
                shares,
                target,
                open_prices,
                mark_open,
                cost_rates,
            )
            for execution in executions:
                asset = str(execution["asset"])
                cost = float(execution["cost"])
                total_cost += cost
                day_cost[asset] = day_cost.get(asset, 0.0) + cost
                trades.append({
                    "date": timestamp,
                    "reason": (
                        "initial_buy"
                        if position == 0
                        else "rotation_or_primary_target_change"
                    ),
                    "selected_asset": selected_asset,
                    "signal_source_asset": selection.at[
                        timestamp, "signal_source_asset"
                    ],
                    "selection_reason": selection.at[
                        timestamp, "selection_reason"
                    ],
                    "primary_target": primary_weight,
                    "signal_execution_reason": schedule.at[
                        timestamp, "execution_reason"
                    ],
                    **execution,
                })
            previous_target = target
            unavailable_targets = {
                asset for asset in target if asset not in open_prices
            }

        internal_cost = sum(float(execution["cost"]) for execution in executions)
        internal_turnover = sum(
            float(execution["turnover"]) for execution in executions
        )
        internal_cost_rate = (
            internal_cost / open_nav_before_trade
            if open_nav_before_trade > 0.0
            else 0.0
        )
        post_open_nav = cash + sum(
            quantity * mark_open.get(asset, 0.0)
            for asset, quantity in shares.items()
        )
        post_open_weights = (
            {
                asset: shares.get(asset, 0.0)
                * mark_open.get(asset, 0.0)
                / post_open_nav
                for asset in (*params.assets, params.defensive_asset)
            }
            if post_open_nav > 0.0
            else {
                asset: 0.0
                for asset in (*params.assets, params.defensive_asset)
            }
        )
        post_open_cash_weight = (
            cash / post_open_nav if post_open_nav > 0.0 else 0.0
        )

        close_prices = {
            asset: (_asof_price(frame, timestamp, "close") or 0.0)
            for asset, frame in indexed.items()
        }
        for asset, quantity in shares.items():
            if asset in open_prices:
                pnl = quantity * (
                    close_prices.get(asset, open_prices[asset])
                    - open_prices[asset]
                )
                gross_pnl[asset] = gross_pnl.get(asset, 0.0) + pnl
                day_gross[asset] = day_gross.get(asset, 0.0) + pnl

        nav = cash + sum(
            quantity * close_prices.get(asset, 0.0)
            for asset, quantity in shares.items()
        )
        daily_return = nav / last_nav - 1.0 if position > 0 else np.nan
        actual_weights = (
            {
                asset: quantity * close_prices.get(asset, 0.0) / nav
                for asset, quantity in shares.items()
            }
            if nav > 0.0
            else {}
        )
        intraday_gross_return = (
            nav / post_open_nav - 1.0 if post_open_nav > 0.0 else 0.0
        )
        daily_gross_return = (
            (1.0 + overnight_gross_return)
            * (1.0 + intraday_gross_return)
            - 1.0
        )
        daily_net_return_if_held = nav / previous_close_nav - 1.0
        reconstructed_return = (
            (1.0 + overnight_gross_return)
            * (1.0 - internal_cost_rate)
            * (1.0 + intraday_gross_return)
            - 1.0
        )
        row: dict[str, object] = {
            "date": timestamp,
            "nav": nav,
            "return": daily_return,
            "cash": cash,
            "cash_weight": cash / nav if nav > 0.0 else 0.0,
            "has_previous_close": position > 0,
            "overnight_gross_return": overnight_gross_return,
            "intraday_gross_return_if_held": intraday_gross_return,
            "daily_gross_return_if_held": daily_gross_return,
            "internal_turnover": internal_turnover,
            "internal_cost_rate_at_open": internal_cost_rate,
            "daily_net_return_if_held": daily_net_return_if_held,
            "daily_net_return_reconstructed": reconstructed_return,
            "previous_closing_cash_weight": previous_close_cash_weight,
            "post_open_cash_weight": post_open_cash_weight,
            "primary_target": primary_weight,
            "defensive_target": 1.0 - primary_weight,
            "selected_asset": selected_asset,
            "signal_source_asset": selection.at[
                timestamp, "signal_source_asset"
            ],
            "signal_observation_date": selection.at[
                timestamp, "signal_observation_date"
            ],
            "selection_reason": selection.at[timestamp, "selection_reason"],
            "signal_execution_reason": schedule.at[timestamp, "execution_reason"],
        }
        for asset in (*params.assets, params.defensive_asset):
            row[f"target_{asset}"] = target.get(asset, 0.0)
            row[f"previous_closing_weight_{asset}"] = (
                previous_close_weights[asset]
            )
            row[f"post_open_weight_{asset}"] = post_open_weights[asset]
            row[f"weight_{asset}"] = actual_weights.get(asset, 0.0)
            gross = day_gross.get(asset, 0.0)
            cost = day_cost.get(asset, 0.0)
            row[f"gross_pnl_{asset}"] = gross
            row[f"transaction_cost_{asset}"] = cost
            row[f"net_pnl_{asset}"] = gross - cost
        rows.append(row)
        last_nav = nav
        previous_closes = close_prices
        previous_close_nav = nav
        previous_close_weights = {
            asset: actual_weights.get(asset, 0.0)
            for asset in (*params.assets, params.defensive_asset)
        }
        previous_close_cash_weight = cash / nav if nav > 0.0 else 0.0

    daily = pd.DataFrame(rows).set_index("date")
    trades_frame = pd.DataFrame(trades)
    metrics = _performance_metrics(daily)
    metrics.update({
        "execution_count": int(len(trades_frame)),
        "total_turnover": (
            float(trades_frame["turnover"].sum())
            if not trades_frame.empty
            else 0.0
        ),
        "total_cost": total_cost,
        "rotation_switch_count": int(
            selection["selected_asset"].ne(
                selection["selected_asset"].shift()
            ).sum() - 1
        ),
    })
    for asset, pnl in gross_pnl.items():
        metrics[f"gross_pnl_{asset.split('.', maxsplit=1)[0]}"] = pnl
    return daily, trades_frame, metrics


def run_backtest(
    market: Mapping[str, pd.DataFrame] | None = None,
    start: date = START_DATE,
    end: date | None = None,
    params: RelativeDefenderRotationParams | None = None,
    cost_rate: CostRateSpec | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object], pd.DataFrame, pd.DataFrame]:
    """Run the listing-aware 2013 research extension."""
    selected = rotation_params() if params is None else params
    if selected.selected_asset_weight != 1.0:
        raise ValueError("2013 extension requires 100% primary-sleeve rotation")
    prices = _clean_market(market, end, selected)
    latest = min(pd.Timestamp(frame["date"].max()) for frame in prices.values())
    full_start = pd.Timestamp(prices[BRIDGE_SIGNAL_ASSET]["date"].min())
    full_calendar = _union_calendar(prices, full_start, latest)
    backtest_calendar = full_calendar[full_calendar >= pd.Timestamp(start)]
    if backtest_calendar.empty:
        raise ValueError("requested start is after available market data")
    full_schedule = hybrid_champion_schedule(prices, full_calendar)
    selection, events = listing_aware_rotation_schedule(
        prices,
        full_schedule,
        full_calendar,
        backtest_calendar,
        selected,
    )
    schedule = full_schedule.reindex(backtest_calendar)
    applied_costs = ROTATION_COST_RATES if cost_rate is None else cost_rate
    daily, trades, metrics = _simulate(
        prices,
        schedule,
        selection,
        selected,
        applied_costs,
    )
    first_anchor_close = pd.Timestamp(
        prices[BASE_PRIMARY_ASSET]["date"].min()
    )
    first_anchor_execution = backtest_calendar[
        backtest_calendar > first_anchor_close
    ][0]
    metrics = {
        **metrics,
        "strategy": STRATEGY_ID,
        "research_status": "retrospective_history_extension_not_oos",
        "formal_status": "production_signal_frozen",
        "formal_promotion_date": PROMOTION_DATE,
        "start": str(daily.index.min().date()),
        "end": str(daily.index.max().date()),
        "requested_start": str(start),
        "calendar_method": "union_of_required_asset_trading_dates",
        "pre_anchor_signal_asset": BRIDGE_SIGNAL_ASSET,
        "formal_anchor_asset": BASE_PRIMARY_ASSET,
        "anchor_first_close": str(first_anchor_close.date()),
        "anchor_first_execution": str(pd.Timestamp(first_anchor_execution).date()),
        "defensive_prelisting_treatment": "cash",
        "listing_rule": (
            "listed_at_execution; finite_history_for_ranking; "
            "first_listed_fallback"
        ),
        "core_parameters_changed": False,
        **asdict(selected),
    }
    return daily, trades, metrics, selection, events


def _returns_from_nav(daily: pd.DataFrame) -> pd.Series:
    values = daily["nav"].pct_change().astype(float)
    values.iloc[0] = float(daily["nav"].iloc[0] / INITIAL_CAPITAL - 1.0)
    return values.rename("return")


def _buy_hold_returns(
    prices: pd.DataFrame,
    calendar: pd.DatetimeIndex,
) -> pd.Series:
    frame = prices.set_index("date").sort_index()
    close = frame["close"].astype(float).reindex(calendar).ffill()
    result = close.pct_change(fill_method=None)
    first_date = pd.Timestamp(calendar[0])
    result.iloc[0] = float(
        close.iloc[0] / float(frame.at[first_date, "open"]) - 1.0
    )
    return result.rename("return")


def _annual_returns(returns: pd.Series) -> pd.DataFrame:
    annual = ((1.0 + returns).groupby(returns.index.year).prod() - 1.0).rename(
        "strategy_return"
    ).to_frame()
    annual.index.name = "year"
    return annual


def _selection_timeline_svg(selection: pd.DataFrame) -> str:
    import matplotlib.pyplot as plt

    positions = {asset: index for index, asset in enumerate(ROTATION_ASSETS)}
    selected = selection["selected_asset"].map(positions).astype(float)
    changes = selection["selected_asset"].ne(selection["selected_asset"].shift())
    figure, axis = plt.subplots(figsize=(12, 3.4), dpi=110)
    axis.step(selection.index, selected, where="post", color="#2563eb", linewidth=1.5)
    axis.scatter(
        selection.index[changes],
        selected.loc[changes],
        color="#f59e0b",
        edgecolor="white",
        linewidth=0.5,
        s=24,
        zorder=3,
    )
    axis.set_yticks(np.arange(len(ROTATION_ASSETS), dtype=float))
    axis.set_yticklabels(ROTATION_ASSETS, fontsize=9)
    axis.set_ylim(-0.5, len(ROTATION_ASSETS) - 0.5)
    axis.set_title("Monthly Selected ETF — Only Listed and Ranking-Eligible Assets")
    axis.set_xlabel("Date")
    axis.grid(axis="x", color="#e5e7eb", linewidth=0.6)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    figure.subplots_adjust(left=0.12, right=0.985, top=0.88, bottom=0.17)
    buffer = io.BytesIO()
    figure.savefig(buffer, format="svg", bbox_inches="tight")
    plt.close(figure)
    svg = buffer.getvalue().decode("utf-8")
    return svg[svg.find("<svg"):]


def _append_research_section(
    report: Path,
    market: Mapping[str, pd.DataFrame],
    daily: pd.DataFrame,
    metrics: Mapping[str, object],
    selection: pd.DataFrame,
    events: pd.DataFrame,
    annual: pd.DataFrame,
) -> None:
    listing_rows = []
    for asset in (*ROTATION_ASSETS, DEFENSIVE_ASSET):
        first_date = pd.Timestamp(market[asset]["date"].min())
        if asset == DEFENSIVE_ASSET:
            selected_dates = daily.index[daily[f"weight_{asset}"].gt(1e-14)]
        else:
            selected_dates = selection.index[selection["selected_asset"].eq(asset)]
        listing_rows.append({
            "ETF": asset,
            "本地首日": str(first_date.date()),
            "首次入选/持有": (
                str(pd.Timestamp(selected_dates.min()).date())
                if len(selected_dates)
                else "未入选"
            ),
            "角色": "防守资产" if asset == DEFENSIVE_ASSET else "股票ETF候选",
        })
    listing_html = pd.DataFrame(listing_rows).to_html(
        index=False, border=0, classes="history-table", escape=True
    )
    annual_display = annual.reset_index().copy()
    for column in (
        "strategy_return",
        "benchmark_510880_return",
        "excess_vs_510880",
    ):
        annual_display[column] = annual_display[column].map(
            lambda value: f"{float(value):.2%}"
        )
    annual_html = annual_display.rename(
        columns={
            "year": "年份",
            "strategy_return": "策略收益",
            "benchmark_510880_return": "510880买入持有",
            "excess_vs_510880": "超额收益",
        }
    ).to_html(index=False, border=0, classes="history-table", escape=True)
    summary = (
        selection.assign(month=selection.index.to_period("M"))
        .groupby("month", as_index=False)
        .first()
        .groupby("selected_asset")
        .size()
        .rename("入选月份")
        .reset_index()
        .rename(columns={"selected_asset": "ETF"})
    )
    summary_html = summary.to_html(
        index=False, border=0, classes="history-table", escape=True
    )
    event_columns = [
        "signal_date",
        "execution_date",
        "new_selected_asset",
        "selection_reason",
        "signal_anchor_asset",
        "listed_assets",
        "ranking_eligible_assets",
    ]
    event_html = events[event_columns].to_html(
        index=False, border=0, classes="history-table", escape=True
    )
    timeline = _selection_timeline_svg(selection)
    section = f"""
    <section id="listing_aware_2013_appendix" class="history-appendix">
      <div class="history-callout"><strong>正式版本与证据边界：</strong>
      本版本于{PROMOTION_DATE}按用户明确指令晋升为主策略；但2013历史段不是
      512890策略的直接实盘复刻，因为512890当时尚未上市。
      2013-07-01至512890首个收盘之前，使用510880和完全相同的冻结参数产生仓位信号；
      自{escape(str(metrics['anchor_first_execution']))}开盘起切回512890信号。</div>

      <h2>因果规则与上市约束</h2>
      <ol>
        <li>每月首个交易日开盘选标，只使用上一交易日及更早信息。</li>
        <li>未上市或执行日没有可成交开盘价的ETF不进入候选池。</li>
        <li>已经上市但没有40日/150日所需历史的ETF不参与对应排名；若全部历史不足，
        选择配置顺序中的第一只已上市ETF。</li>
        <li>2017-08-24之前511260尚未上市，未分配给股票ETF的仓位保留现金，不伪造债券收益。</li>
        <li>股票ETF单边费率0.01%，511260单边费率0.001%；后复权OHLC、连续份额、开盘成交。</li>
      </ol>

      <h2>ETF上市与首次入选</h2>
      <div class="history-scroll">{listing_html}</div>

      <h2>回测结果摘要</h2>
      <table class="history-table"><tbody>
        <tr><th>区间</th><td>{escape(str(metrics['start']))} 至 {escape(str(metrics['end']))}</td></tr>
        <tr><th>期末净值</th><td>{float(metrics['final_nav']):.4f}</td></tr>
        <tr><th>年化收益</th><td>{float(metrics['annualized_return']):.2%}</td></tr>
        <tr><th>Sharpe</th><td>{float(metrics['sharpe']):.3f}</td></tr>
        <tr><th>最大回撤</th><td>{float(metrics['max_drawdown']):.2%}</td></tr>
        <tr><th>主标的切换次数</th><td>{int(metrics['rotation_switch_count'])}</td></tr>
      </tbody></table>
      <div class="history-scroll">{annual_html}</div>

      <h2>月度选择轨迹</h2>
      <div class="history-timeline">{timeline}</div>
      <div class="history-scroll">{summary_html}</div>

      <h2>完整月度选择记录</h2>
      <div class="history-events">{event_html}</div>

      <h2>解释限制</h2>
      <ul>
        <li>2013–2018段依赖510880信号桥接假设，不能与2019年起的正式512890锚定样本等同解释。</li>
        <li>后上市ETF只有积累足够排名历史后才可能入选，避免使用上市前价格或未来数据。</li>
        <li>报告未模拟滑点、冲击成本、涨跌停和申赎限制；停牌日按最近收盘价估值，无法成交的目标等待恢复交易。</li>
        <li>策略已正式晋升，但全部历史结果仍属于回溯证据，不是独立样本外证据。</li>
      </ul>
    </section>
    <style>
      .history-appendix{{clear:both;max-width:960px;margin:42px auto 24px;padding:0 18px 36px;
        font-family:Arial,"PingFang SC","Microsoft YaHei",sans-serif;color:#1f2937;line-height:1.65}}
      .history-appendix h2{{font-size:21px;margin:34px 0 14px;padding-bottom:7px;border-bottom:2px solid #dbeafe;color:#0f172a}}
      .history-appendix p,.history-appendix li{{font-size:13px}}
      .history-callout{{background:#fff7ed;border-left:5px solid #f59e0b;padding:13px 16px;border-radius:4px;font-size:13px}}
      .history-table{{width:100%;border-collapse:collapse;font-size:12px;margin:8px 0 16px}}
      .history-table th,.history-table td{{border:1px solid #dbe2ea;padding:7px 8px;text-align:right;white-space:nowrap}}
      .history-table th{{background:#eff6ff;font-weight:600;color:#1e3a5f}}
      .history-table th:first-child,.history-table td:first-child{{text-align:left}}
      .history-scroll{{width:100%;overflow-x:auto}}
      .history-events{{overflow:auto;max-height:560px;border:1px solid #e5e7eb}}
      .history-events .history-table{{margin:0}}
      .history-events thead th{{position:sticky;top:0;z-index:1}}
      .history-timeline{{width:100%;overflow:hidden;border:1px solid #e5e7eb;border-radius:6px;padding:8px;box-sizing:border-box}}
      .history-timeline svg{{display:block;width:100%;height:auto}}
      @media screen and (max-width:760px){{.history-table{{font-size:11px}}.history-table th,.history-table td{{white-space:normal}}}}
    </style>
    """
    document = report.read_text(encoding="utf-8")
    if "</body>" not in document:
        raise RuntimeError("could not locate HTML body end")
    report.write_text(
        document.replace("</body>", section + "</body>", 1),
        encoding="utf-8",
    )


def build_html(output: Path = OUTPUT) -> Path:
    """Build the listing-aware 2013 HTML report and audit artifacts."""
    from .deliver import asset_pnl_from_daily, write_standard_html_report

    params = rotation_params()
    market = _clean_market(None, None, params)
    daily, trades, metrics, selection, events = run_backtest(
        market=market,
        start=START_DATE,
        params=params,
    )
    strategy_returns = _returns_from_nav(daily)
    benchmark_returns = _buy_hold_returns(
        market[BRIDGE_SIGNAL_ASSET],
        pd.DatetimeIndex(daily.index),
    )
    annual = _annual_returns(strategy_returns)
    benchmark_annual = _annual_returns(benchmark_returns).rename(
        columns={"strategy_return": "benchmark_510880_return"}
    )
    annual = annual.join(benchmark_annual)
    annual["excess_vs_510880"] = (
        annual["strategy_return"] - annual["benchmark_510880_return"]
    )

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(output.parent / f"{PREFIX}_daily.csv")
    trades.to_csv(output.parent / f"{PREFIX}_trades.csv", index=False)
    selection.to_csv(output.parent / f"{PREFIX}_selection.csv")
    events.to_csv(output.parent / f"{PREFIX}_events.csv", index=False)
    annual.to_csv(output.parent / f"{PREFIX}_annual.csv")
    (output.parent / f"{PREFIX}_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report = write_standard_html_report(
        strategy_returns,
        benchmark_returns,
        str(metrics["start"]),
        str(metrics["end"]),
        output=output,
        report_title="Relative Defender 2013上市感知正式策略回测报告",
        report_heading="Defender正式策略：2013起仅在已上市ETF中轮动",
        report_subtitle=(
            "2013-07-01起 • 510880信号桥接至512890上市 • "
            "未上市ETF忽略 • 月末收盘信号、次月首开执行 • 固定冻结参数"
        ),
        benchmark_title="510880 Buy-and-Hold",
        trades=trades,
        asset_pnl=asset_pnl_from_daily(daily),
    )
    _append_research_section(
        report,
        market,
        daily,
        metrics,
        selection,
        events,
        annual,
    )
    return report


def main() -> None:
    print(build_html())


if __name__ == "__main__":
    main()
