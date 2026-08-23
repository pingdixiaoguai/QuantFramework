"""Direct quality-momentum comparison of four ETFs and the whole Defender NAV.

Defender is treated as one synthetic candidate whose close series is its exact
continuous-hold net NAV.  The same registered ``quality_momentum`` function is
applied to all five candidates, and the prior close's Top-1 score selects the
next-open holding.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from data.store import query
from defender.relative_defender_rotation_2013_export import build_switch_return_frame
from factors.quality_momentum import compute as quality_momentum
from research.momentum_defender_occam import (
    ENTER_RETURN,
    ENTRY_COST,
    EXIT_COST,
    EXIT_RETURN,
    HELD_RETURN,
    INTERNAL_COST,
    MOMENTUM_ASSETS,
    performance,
)


DEFENDER_CANDIDATE = "DEFENDER"
ALL_CANDIDATES = (*MOMENTUM_ASSETS, DEFENDER_CANDIDATE)
ETF_COST_RATE = 0.0001


@dataclass(frozen=True)
class CurveMomentumParams:
    window: int = 20
    rebalance_days: int = 1
    start: date = date(2019, 1, 18)

    def __post_init__(self) -> None:
        if self.window < 1 or self.rebalance_days < 1:
            raise ValueError("window and rebalance_days must be positive")


@dataclass(frozen=True)
class CurveMomentumBacktest:
    params: CurveMomentumParams
    calendar: pd.DatetimeIndex
    close_curves: pd.DataFrame
    scores_at_close: pd.DataFrame
    desired_at_open: pd.Series
    interfaces: Mapping[str, pd.DataFrame]
    daily: pd.DataFrame
    audit: dict[str, object]


def _chain(left: float, right: float) -> float:
    return (1.0 + float(left)) * (1.0 + float(right)) - 1.0


def _single_etf_interface(
    asset: str,
    calendar: pd.DatetimeIndex,
    end: date,
    cost_rate: float = ETF_COST_RATE,
) -> tuple[pd.DataFrame, pd.Series]:
    """Build exact hold/entry/exit legs for one ETF on the master calendar."""
    prices = query(asset, date(2013, 1, 1), end).copy()
    if prices.empty:
        raise RuntimeError(f"no local prices for {asset}")
    prices = prices.sort_values("date").drop_duplicates("date").set_index("date")
    indexed = prices[["open", "close"]].astype(float)
    close_curve = indexed["close"].reindex(calendar).ffill()
    rows: list[dict[str, object]] = []
    nav = 1.0
    for timestamp in calendar:
        traded = timestamp in indexed.index
        history = indexed.loc[indexed.index < timestamp, "close"].dropna()
        previous_close = float(history.iloc[-1]) if not history.empty else np.nan
        if traded:
            open_price = float(indexed.at[timestamp, "open"])
            close_price = float(indexed.at[timestamp, "close"])
        else:
            open_price = previous_close
            close_price = previous_close

        if np.isfinite(previous_close):
            held = close_price / previous_close - 1.0
            if traded:
                entered = (1.0 - cost_rate) * (close_price / open_price) - 1.0
                exited = (open_price / previous_close) * (1.0 - cost_rate) - 1.0
                entry_cost = cost_rate
                exit_cost = cost_rate
            else:
                entered = np.nan
                exited = np.nan
                entry_cost = np.nan
                exit_cost = np.nan
        elif traded:
            held = close_price / open_price - 1.0
            entered = (1.0 - cost_rate) * (close_price / open_price) - 1.0
            exited = np.nan
            entry_cost = cost_rate
            exit_cost = np.nan
        else:
            held = 0.0
            entered = np.nan
            exited = np.nan
            entry_cost = np.nan
            exit_cost = np.nan
        nav *= 1.0 + float(held)
        rows.append(
            {
                "date": timestamp,
                HELD_RETURN: held,
                ENTER_RETURN: entered,
                EXIT_RETURN: exited,
                INTERNAL_COST: 0.0,
                ENTRY_COST: entry_cost,
                EXIT_COST: exit_cost,
                "nav_if_held": nav,
                "traded": traded,
            }
        )
    return pd.DataFrame(rows).set_index("date"), close_curve.rename(asset)


def build_candidate_bundle(
    *,
    end: date | None = None,
    window: int = 20,
) -> tuple[
    pd.DatetimeIndex,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, pd.DataFrame],
]:
    """Build common close curves, scores, and exact switch interfaces."""
    defender, _ = build_switch_return_frame()
    cutoff = defender.index.max().date() if end is None else end
    defender = defender.loc[: pd.Timestamp(cutoff)].copy()
    calendar = pd.DatetimeIndex(defender.index)
    interfaces: dict[str, pd.DataFrame] = {
        DEFENDER_CANDIDATE: defender,
    }
    close_curves: dict[str, pd.Series] = {
        DEFENDER_CANDIDATE: defender["nav_if_held"].astype(float),
    }
    for asset in MOMENTUM_ASSETS:
        interface, close = _single_etf_interface(asset, calendar, cutoff)
        interfaces[asset] = interface
        close_curves[asset] = close

    curves = pd.DataFrame(close_curves, index=calendar)
    scores = score_close_curves(curves, window)
    curves.index.name = "date"
    scores.index.name = "date"
    return calendar, curves, scores, interfaces


def score_close_curves(curves: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Apply the exact same quality-momentum implementation to every curve."""
    if window < 1:
        raise ValueError("window must be positive")
    scores = pd.DataFrame(index=curves.index)
    for candidate in ALL_CANDIDATES:
        frame = pd.DataFrame(
            {
                "date": curves.index,
                "close": curves[candidate].to_numpy(float),
            }
        )
        scores[candidate] = quality_momentum(frame, {"window": window}).reindex(
            curves.index
        )
    scores.index.name = "date"
    return scores


def _simulate(
    scores: pd.DataFrame,
    interfaces: Mapping[str, pd.DataFrame],
    params: CurveMomentumParams,
) -> tuple[pd.Series, pd.DataFrame]:
    prior_scores = scores.shift(1)
    available = prior_scores.notna().all(axis=1)
    calendar = scores.index[
        available & (scores.index >= pd.Timestamp(params.start))
    ]
    if calendar.empty:
        raise ValueError("no common scored dates in the requested sample")
    desired = prior_scores.loc[calendar].idxmax(axis=1).rename("desired_at_open")

    current: str | None = None
    held_days = 10**9
    nav = 1.0
    rows: list[dict[str, object]] = []
    for timestamp in calendar:
        raw_target = str(desired.loc[timestamp])
        may_rebalance = current is None or held_days >= params.rebalance_days
        target = raw_target if may_rebalance else current
        switched = target != current
        blocked = False
        exit_leg = np.nan
        enter_leg = np.nan
        held_leg = np.nan

        if switched:
            exit_ok = current is None or pd.notna(
                interfaces[current].at[timestamp, EXIT_RETURN]
            )
            enter_ok = pd.notna(interfaces[target].at[timestamp, ENTER_RETURN])
            if not exit_ok or not enter_ok:
                blocked = True
                target = current
                switched = False

        if current is None and target is None:
            daily_return = 0.0
            cost_rate = 0.0
            transition = "cash_hold"
        elif current is None and target is not None:
            enter_leg = float(interfaces[target].at[timestamp, ENTER_RETURN])
            daily_return = enter_leg
            cost_rate = float(interfaces[target].at[timestamp, ENTRY_COST])
            transition = f"cash_to_{target}"
            current = target
            held_days = 0
        elif switched and target is not None:
            exit_leg = float(interfaces[current].at[timestamp, EXIT_RETURN])
            enter_leg = float(interfaces[target].at[timestamp, ENTER_RETURN])
            daily_return = _chain(exit_leg, enter_leg)
            exit_cost = float(interfaces[current].at[timestamp, EXIT_COST])
            entry_cost = float(interfaces[target].at[timestamp, ENTRY_COST])
            cost_rate = 1.0 - (1.0 - exit_cost) * (1.0 - entry_cost)
            transition = f"{current}_to_{target}"
            current = target
            held_days = 0
        else:
            assert current is not None
            held_leg = float(interfaces[current].at[timestamp, HELD_RETURN])
            daily_return = held_leg
            cost_rate = float(interfaces[current].at[timestamp, INTERNAL_COST])
            transition = f"{current}_hold"

        if not np.isfinite(daily_return) or daily_return <= -1.0:
            raise ValueError(f"invalid return on {timestamp}: {daily_return}")
        nav *= 1.0 + daily_return
        score_row = prior_scores.loc[timestamp]
        rows.append(
            {
                "date": timestamp,
                "return": daily_return,
                "nav": nav,
                "candidate": current,
                "desired_candidate": raw_target,
                "transition": transition,
                "switched": switched,
                "switch_blocked_untradable": blocked,
                "held_days_at_open": held_days,
                "cost_rate_at_open": cost_rate,
                "held_return_leg_used": held_leg,
                "exit_return_leg_used": exit_leg,
                "enter_return_leg_used": enter_leg,
                **{
                    f"score_{candidate}": float(score_row[candidate])
                    for candidate in ALL_CANDIDATES
                },
            }
        )
        held_days += 1
    return desired, pd.DataFrame(rows).set_index("date")


def validate_curve_momentum(
    backtest: CurveMomentumBacktest,
) -> dict[str, object]:
    daily = backtest.daily
    reconstructed = (1.0 + daily["return"].astype(float)).cumprod()
    nav_error = float((reconstructed - daily["nav"].astype(float)).abs().max())
    prior_scores = backtest.scores_at_close.shift(1).loc[daily.index]
    desired = prior_scores.idxmax(axis=1)
    desired_matches = bool(desired.equals(daily["desired_candidate"]))
    defender_input_error = float(
        (
            backtest.close_curves.loc[daily.index, DEFENDER_CANDIDATE]
            - backtest.interfaces[DEFENDER_CANDIDATE].loc[
                daily.index, "nav_if_held"
            ].astype(float)
        )
        .abs()
        .max()
    )
    switch_rows = daily["switched"].astype(bool)
    switch_legs_finite = bool(
        daily.loc[switch_rows, "enter_return_leg_used"].notna().all()
        and daily.loc[switch_rows & daily.index.to_series().ne(daily.index[0]),
                      "exit_return_leg_used"].notna().all()
    )
    if (
        nav_error > 1e-12
        or not desired_matches
        or defender_input_error > 1e-12
        or not switch_legs_finite
    ):
        raise AssertionError(
            "curve momentum audit failed: "
            f"nav={nav_error:.3e}, desired={desired_matches}, "
            f"defender={defender_input_error:.3e}, legs={switch_legs_finite}"
        )
    return {
        "status": "passed",
        "nav_reconstruction_max_abs_error": nav_error,
        "desired_is_prior_close_top1": desired_matches,
        "defender_curve_equals_continuous_nav_max_abs_error": defender_input_error,
        "all_executed_switch_legs_finite": switch_legs_finite,
        "observations": int(len(daily)),
        "start": daily.index.min().date().isoformat(),
        "end": daily.index.max().date().isoformat(),
        "switches": int(daily["switched"].sum()),
        "blocked_switches": int(daily["switch_blocked_untradable"].sum()),
        "defender_days": int(daily["candidate"].eq(DEFENDER_CANDIDATE).sum()),
        "performance": performance(daily["return"]),
    }


def run_curve_momentum(
    params: CurveMomentumParams,
    *,
    end: date | None = None,
) -> CurveMomentumBacktest:
    calendar, curves, scores, interfaces = build_candidate_bundle(
        end=end,
        window=params.window,
    )
    return run_curve_momentum_from_bundle(params, curves, scores, interfaces)


def run_curve_momentum_from_bundle(
    params: CurveMomentumParams,
    curves: pd.DataFrame,
    scores: pd.DataFrame,
    interfaces: Mapping[str, pd.DataFrame],
) -> CurveMomentumBacktest:
    """Run one rebalance variant from a shared, immutable candidate bundle."""
    desired, daily = _simulate(scores, interfaces, params)
    provisional = CurveMomentumBacktest(
        params=params,
        calendar=pd.DatetimeIndex(daily.index),
        close_curves=curves,
        scores_at_close=scores,
        desired_at_open=desired,
        interfaces=interfaces,
        daily=daily,
        audit={},
    )
    audit = validate_curve_momentum(provisional)
    return CurveMomentumBacktest(
        params=params,
        calendar=provisional.calendar,
        close_curves=curves,
        scores_at_close=scores,
        desired_at_open=desired,
        interfaces=interfaces,
        daily=daily,
        audit=audit,
    )
