"""Low-dimensional Defender sleeves for Momentum/Defender research.

The module deliberately separates two concerns:

* a causal monthly selector that uses one trailing score; and
* an exact next-open portfolio interface for any target-weight schedule.

This lets research replace only the Defender sleeve while keeping the formal
Momentum factor, top-level downside-RAQM state, transaction costs, and switch
execution unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from defender.defender_opt_v2 import (
    CostRateSpec,
    _asof_price,
    _asset_cost_rate,
    _execute_portfolio_target,
    _indexed_market,
)
from research.momentum_defender_occam import (
    ENTER_RETURN,
    ENTRY_COST,
    EXIT_COST,
    EXIT_RETURN,
    HELD_RETURN,
    INTERNAL_COST,
)
from research.momentum_volatility import asof_previous_close


SCORE_RETURN = "return"
SCORE_QUALITY = "quality"
SCORE_RISK_ADJUSTED = "risk_adjusted"
SCORE_METHODS = (SCORE_RETURN, SCORE_QUALITY, SCORE_RISK_ADJUSTED)
SELECT_HIGHEST = "highest"
SELECT_LOWEST = "lowest"
SELECTION_DIRECTIONS = (SELECT_HIGHEST, SELECT_LOWEST)


@dataclass(frozen=True)
class MonthlySelectionSpec:
    """One-score, one-window monthly Top-1 selection rule."""

    window: int
    score_method: str = SCORE_RETURN
    direction: str = SELECT_LOWEST

    def __post_init__(self) -> None:
        if self.window < 2:
            raise ValueError("window must be at least 2")
        if self.score_method not in SCORE_METHODS:
            raise ValueError(f"unsupported score method: {self.score_method}")
        if self.direction not in SELECTION_DIRECTIONS:
            raise ValueError(f"unsupported selection direction: {self.direction}")

    @property
    def candidate_id(self) -> str:
        return f"{self.score_method}_{self.direction}_w{self.window}"


def _clean_close(frame: pd.DataFrame) -> pd.Series:
    prices = frame.copy()
    prices["date"] = pd.to_datetime(prices["date"])
    prices = prices.sort_values("date").drop_duplicates("date").set_index("date")
    close = prices["close"].astype(float)
    if close.empty or close.isna().any() or close.le(0.0).any():
        raise ValueError("score input must contain finite positive closes")
    return close


def score_at_open(
    market: Mapping[str, pd.DataFrame],
    assets: tuple[str, ...],
    calendar: pd.DatetimeIndex,
    spec: MonthlySelectionSpec,
) -> pd.DataFrame:
    """Return scores known strictly before every execution open."""
    result: dict[str, pd.Series] = {}
    for asset in assets:
        if asset not in market:
            raise RuntimeError(f"missing market data for: {asset}")
        log_close = np.log(_clean_close(market[asset]))
        momentum = log_close.diff(spec.window)
        if spec.score_method == SCORE_RETURN:
            score = momentum
        elif spec.score_method == SCORE_QUALITY:
            path = log_close.diff().abs().rolling(spec.window).sum()
            efficiency = momentum.abs() / path.replace(0.0, np.nan)
            score = momentum * efficiency
        else:
            volatility = (
                log_close.diff().rolling(spec.window).std(ddof=1)
                * np.sqrt(spec.window)
            )
            score = momentum / volatility.replace(0.0, np.nan)
        result[asset] = asof_previous_close(score, calendar)
    frame = pd.DataFrame(result, index=calendar)
    frame.index.name = "date"
    return frame


def _open_dates(frame: pd.DataFrame) -> set[pd.Timestamp]:
    valid = frame.loc[
        frame["open"].notna() & frame["open"].astype(float).gt(0.0),
        "date",
    ]
    return set(pd.to_datetime(valid))


def monthly_top1_selection(
    market: Mapping[str, pd.DataFrame],
    assets: tuple[str, ...],
    calendar: pd.DatetimeIndex,
    scores_at_open: pd.DataFrame,
    spec: MonthlySelectionSpec,
) -> pd.DataFrame:
    """Select one tradable asset at each first union-calendar open of a month.

    A target that cannot be bought, or a current holding that cannot be sold,
    leaves the previous selection unchanged. Ties preserve ``assets`` order.
    """
    if calendar.empty or not calendar.is_monotonic_increasing or not calendar.is_unique:
        raise ValueError("calendar must be non-empty, ordered, and unique")
    if tuple(scores_at_open.columns) != assets:
        raise ValueError("score columns must match the configured asset order")
    traded = {asset: _open_dates(market[asset]) for asset in assets}
    current: str | None = None
    rows: list[dict[str, object]] = []
    for position, execution_date in enumerate(calendar):
        execution_date = pd.Timestamp(execution_date)
        month_changed = position == 0 or (
            execution_date.to_period("M")
            != pd.Timestamp(calendar[position - 1]).to_period("M")
        )
        previous = current
        reason = "hold"
        proposed: str | None = None
        if month_changed:
            values = scores_at_open.loc[execution_date]
            eligible = [
                asset
                for asset in assets
                if np.isfinite(values.get(asset, np.nan))
                and execution_date in traded[asset]
            ]
            if eligible:
                ranked = values.loc[eligible]
                extreme = (
                    float(ranked.max())
                    if spec.direction == SELECT_HIGHEST
                    else float(ranked.min())
                )
                proposed = next(
                    asset
                    for asset in assets
                    if asset in eligible
                    and np.isclose(float(ranked[asset]), extreme, atol=1e-14)
                )
            if current is None:
                current = proposed
                reason = "initial_monthly_selection"
            elif proposed is None:
                reason = "monthly_no_eligible_target"
            elif proposed == current:
                reason = "monthly_same_target"
            elif execution_date not in traded[current]:
                reason = "monthly_switch_blocked_untradable_exit"
            else:
                current = proposed
                reason = "monthly_reselection"
        if current is None:
            raise RuntimeError(
                f"no eligible initial Defender asset on {execution_date.date()}"
            )
        rows.append(
            {
                "date": execution_date,
                "selected_asset": current,
                "previous_selected_asset": previous,
                "proposed_asset": proposed,
                "selection_reason": reason,
                "selection_changed": current != previous and previous is not None,
            }
        )
    return pd.DataFrame(rows).set_index("date")


def selected_asset_targets(
    selection: pd.Series,
    assets: tuple[str, ...],
    *,
    selected_weight: float = 1.0,
    residual_asset: str | None = None,
) -> pd.DataFrame:
    """Map a selected equity asset to a fixed equity/bond target schedule."""
    if not 0.0 <= selected_weight <= 1.0:
        raise ValueError("selected_weight must lie in [0, 1]")
    if residual_asset is None and not np.isclose(selected_weight, 1.0):
        raise ValueError("a residual asset is required below 100% selected weight")
    columns = list(assets)
    if residual_asset is not None and residual_asset not in columns:
        columns.append(residual_asset)
    targets = pd.DataFrame(0.0, index=selection.index, columns=columns)
    for timestamp, asset_value in selection.items():
        asset = str(asset_value)
        if asset not in assets:
            raise ValueError(f"selection contains an unknown asset: {asset}")
        targets.at[timestamp, asset] = selected_weight
        if residual_asset is not None:
            targets.at[timestamp, residual_asset] += 1.0 - selected_weight
    targets.index.name = "date"
    return targets


def _normalized_target(row: pd.Series) -> dict[str, float]:
    return {
        str(asset): float(weight)
        for asset, weight in row.items()
        if float(weight) > 1e-14
    }


def _market_prices_at(
    indexed: Mapping[str, pd.DataFrame],
    timestamp: pd.Timestamp,
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
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
    close_prices = {
        asset: (_asof_price(frame, timestamp, "close") or 0.0)
        for asset, frame in indexed.items()
    }
    return open_prices, mark_open, close_prices


def build_portfolio_switch_interface(
    market: Mapping[str, pd.DataFrame],
    targets: pd.DataFrame,
    cost_rates: CostRateSpec,
) -> pd.DataFrame:
    """Build exact hold, fresh-entry, and fresh-exit legs for one sleeve.

    The continuous-hold path rebalances only when the policy target changes.
    Fresh entry buys the current policy at the open; fresh exit liquidates the
    prior close's continuous-hold positions at the open. Unavailable policy
    mass remains cash, matching the formal Defender interface.
    """
    calendar = pd.DatetimeIndex(targets.index)
    if calendar.empty or not calendar.is_monotonic_increasing or not calendar.is_unique:
        raise ValueError("targets must have a non-empty ordered unique index")
    if targets.isna().any().any() or targets.lt(-1e-14).any().any():
        raise ValueError("targets must be finite and non-negative")
    totals = targets.sum(axis=1)
    if totals.gt(1.0 + 1e-12).any():
        raise ValueError("target weights cannot exceed one")
    missing = set(targets.columns) - set(market)
    if missing:
        raise RuntimeError(f"missing target market data: {sorted(missing)}")

    normalized_market: dict[str, pd.DataFrame] = {}
    for asset in targets.columns:
        frame = market[asset].copy()
        frame["date"] = pd.to_datetime(frame["date"])
        normalized_market[asset] = (
            frame.sort_values("date").drop_duplicates("date").reset_index(drop=True)
        )
    indexed = _indexed_market(normalized_market)
    cash = 1.0
    shares: dict[str, float] = {}
    previous_target: dict[str, float] = {}
    previous_nav = 1.0
    nav = 1.0
    rows: list[dict[str, object]] = []

    for position, timestamp_value in enumerate(calendar):
        timestamp = pd.Timestamp(timestamp_value)
        policy_target = _normalized_target(targets.loc[timestamp])
        open_prices, mark_open, close_prices = _market_prices_at(indexed, timestamp)

        if position == 0:
            exit_return = np.nan
            exit_cost_rate = np.nan
            exit_fully_executable: bool | float = np.nan
        else:
            normalized_cash = cash / previous_nav
            normalized_shares = {
                asset: quantity / previous_nav for asset, quantity in shares.items()
            }
            held_assets = set(normalized_shares)
            exit_fully_executable = held_assets.issubset(open_prices)
            open_nav = normalized_cash + sum(
                quantity * mark_open.get(asset, 0.0)
                for asset, quantity in normalized_shares.items()
            )
            exit_cash, exit_shares, exit_executions = _execute_portfolio_target(
                normalized_cash,
                normalized_shares,
                {},
                open_prices,
                mark_open,
                cost_rates,
            )
            exit_cost = sum(float(item["cost"]) for item in exit_executions)
            exit_cost_rate = exit_cost / open_nav if open_nav > 0.0 else 0.0
            exit_nav = exit_cash + sum(
                quantity * mark_open.get(asset, 0.0)
                for asset, quantity in exit_shares.items()
            )
            exit_return = (
                exit_nav - 1.0 if bool(exit_fully_executable) else np.nan
            )

        executable_target = {
            asset: weight
            for asset, weight in policy_target.items()
            if asset in open_prices
        }
        entry_cash, entry_shares, entry_executions = _execute_portfolio_target(
            1.0,
            {},
            executable_target,
            open_prices,
            mark_open,
            cost_rates,
        )
        entry_cost_rate = sum(
            float(item["cost"]) for item in entry_executions
        )
        entry_close_nav = entry_cash + sum(
            quantity * close_prices.get(asset, 0.0)
            for asset, quantity in entry_shares.items()
        )
        entry_return = entry_close_nav - 1.0

        internal_executions: list[dict[str, float | str]] = []
        if policy_target != previous_target:
            cash, shares, internal_executions = _execute_portfolio_target(
                cash,
                shares,
                policy_target,
                open_prices,
                mark_open,
                cost_rates,
            )
            previous_target = policy_target
        nav = cash + sum(
            quantity * close_prices.get(asset, 0.0)
            for asset, quantity in shares.items()
        )
        held_return = nav / previous_nav - 1.0
        internal_cost_rate = sum(
            float(item["cost"]) for item in internal_executions
        ) / previous_nav
        if not np.isfinite(held_return) or held_return <= -1.0:
            raise ValueError(f"invalid sleeve return on {timestamp.date()}")

        row: dict[str, object] = {
            HELD_RETURN: held_return,
            ENTER_RETURN: entry_return,
            EXIT_RETURN: exit_return,
            INTERNAL_COST: internal_cost_rate,
            ENTRY_COST: entry_cost_rate,
            EXIT_COST: exit_cost_rate,
            "nav_if_held": nav,
            "policy_target_cash_weight": 1.0 - sum(policy_target.values()),
            "target_cash_weight": 1.0 - sum(executable_target.values()),
            "fresh_entry_policy_fully_executable": set(policy_target).issubset(
                open_prices
            ),
            "fresh_exit_fully_executable": exit_fully_executable,
            "internal_rebalanced": bool(internal_executions),
        }
        for asset in targets.columns:
            row[f"policy_target_{asset}"] = policy_target.get(asset, 0.0)
            row[f"target_{asset}"] = executable_target.get(asset, 0.0)
        rows.append(row)
        previous_nav = nav

    frame = pd.DataFrame(rows, index=calendar)
    frame.index.name = "date"
    reconstructed = (1.0 + frame[HELD_RETURN].astype(float)).cumprod()
    error = float((reconstructed - frame["nav_if_held"].astype(float)).abs().max())
    if error > 1e-12:
        raise AssertionError(f"sleeve NAV reconstruction failed: {error:.3e}")
    return frame
