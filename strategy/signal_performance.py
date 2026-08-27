"""Deterministic performance snapshot for formal DingTalk signals."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from backtest.runner import run as run_backtest
from data.store import query
from research.defender_curve_momentum import DEFENDER_CANDIDATE
from research.formal_strategy_holdings import build_formal_target_schedule
from research.momentum_defender_occam import (
    ENTER_RETURN,
    HELD_RETURN,
    MOMENTUM_ASSETS,
    _load_momentum_config,
)


LEGACY_MOMENTUM_CONFIG = Path(
    "strategy/configs/quality_momentum_top1_legacy_simple_price.yaml"
)


@dataclass(frozen=True)
class PeriodPerformance:
    month: float
    quarter: float
    year: float


@dataclass(frozen=True)
class SignalPerformanceSnapshot:
    since_date: date
    current_holding_label: str
    current_holding_return: float
    concurrent_returns: Mapping[str, float]
    period_returns: Mapping[str, PeriodPerformance]


def _compound(values: pd.Series) -> float:
    numeric = values.astype(float)
    if numeric.empty or numeric.isna().any():
        raise RuntimeError("performance return series is incomplete")
    return float((1.0 + numeric).prod() - 1.0)


def _open_to_close_return(
    weights: Mapping[str, float],
    start: date,
    end: date,
) -> float:
    total = 0.0
    for asset, weight in weights.items():
        frame = (
            query(asset, start, end)
            .sort_values("date")
            .drop_duplicates("date")
        )
        if frame.empty:
            raise RuntimeError(f"performance price history missing for {asset}")
        first = frame.loc[frame["date"].eq(pd.Timestamp(start))]
        if first.empty:
            raise RuntimeError(f"performance entry open missing for {asset}")
        open_start = float(first.iloc[0]["open"])
        close_end = float(frame.iloc[-1]["close"])
        total += float(weight) * (close_end / open_start - 1.0)
    return total


def _entered_sleeve_return(
    interface: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> float:
    sample = interface.loc[start:end]
    if sample.empty:
        raise RuntimeError("performance sleeve interval is empty")
    returns = sample[HELD_RETURN].astype(float).copy()
    returns.iloc[0] = float(sample.iloc[0][ENTER_RETURN])
    return _compound(returns)


def _last_target_change(targets: pd.DataFrame) -> pd.Timestamp:
    if targets.empty:
        raise RuntimeError("formal target schedule is empty")
    values = targets.to_numpy(float)
    last = len(targets) - 1
    start = last
    while start > 0 and np.allclose(
        values[start - 1], values[last], atol=1e-12
    ):
        start -= 1
    return pd.Timestamp(targets.index[start])


def _period_start(timestamp: pd.Timestamp, period: str) -> pd.Timestamp:
    if period == "month":
        return pd.Timestamp(timestamp.year, timestamp.month, 1)
    if period == "quarter":
        month = ((timestamp.month - 1) // 3) * 3 + 1
        return pd.Timestamp(timestamp.year, month, 1)
    if period == "year":
        return pd.Timestamp(timestamp.year, 1, 1)
    raise ValueError(f"unknown period: {period}")


def _period_performance(
    returns: pd.Series,
    signal_timestamp: pd.Timestamp,
) -> PeriodPerformance:
    return PeriodPerformance(
        month=_compound(
            returns.loc[_period_start(signal_timestamp, "month") : signal_timestamp]
        ),
        quarter=_compound(
            returns.loc[
                _period_start(signal_timestamp, "quarter") : signal_timestamp
            ]
        ),
        year=_compound(
            returns.loc[_period_start(signal_timestamp, "year") : signal_timestamp]
        ),
    )


def build_signal_performance(
    root: Path,
    historical,
    signal_date: date,
) -> SignalPerformanceSnapshot:
    """Build current-holding and calendar-period performance comparisons."""

    signal_timestamp = pd.Timestamp(signal_date)
    if historical.daily.index.max() != signal_timestamp:
        raise AssertionError("performance snapshot cutoff differs from signal date")
    targets = build_formal_target_schedule(historical)
    since_timestamp = _last_target_change(targets)
    current_row = targets.loc[since_timestamp]
    current_weights = {
        asset: float(current_row[asset])
        for asset in MOMENTUM_ASSETS
        if float(current_row.get(asset, 0.0)) > 1e-12
    }
    defender_assets = [
        asset
        for asset in targets.columns
        if asset not in {*MOMENTUM_ASSETS, "target_cash_weight"}
        and float(current_row.get(asset, 0.0)) > 1e-12
    ]
    for asset in defender_assets:
        current_weights[asset] = float(current_row[asset])
    if not current_weights:
        raise RuntimeError("performance snapshot has no current invested asset")
    current_holding_return = _open_to_close_return(
        current_weights, since_timestamp.date(), signal_date
    )
    if len(current_weights) == 1:
        current_holding_label = next(iter(current_weights))
    else:
        current_holding_label = "CURRENT_PORTFOLIO"

    context = historical.context
    momentum_interface = context.integrated.result.inputs.momentum
    defender_interface = context.interfaces[DEFENDER_CANDIDATE]
    concurrent: dict[str, float] = {
        "MOMENTUM": _entered_sleeve_return(
            momentum_interface, since_timestamp, signal_timestamp
        ),
        "DEFENDER": _entered_sleeve_return(
            defender_interface, since_timestamp, signal_timestamp
        ),
    }
    for asset in MOMENTUM_ASSETS:
        if asset not in current_weights:
            concurrent[asset] = _open_to_close_return(
                {asset: 1.0}, since_timestamp.date(), signal_date
            )

    calendar = historical.daily.index
    formal_returns = historical.daily["return"].astype(float)
    pure_momentum_returns = momentum_interface[HELD_RETURN].astype(float).reindex(
        calendar
    )
    pure_defender_returns = defender_interface[HELD_RETURN].astype(float).reindex(
        calendar
    )
    legacy_config = _load_momentum_config(
        root / LEGACY_MOMENTUM_CONFIG, signal_date
    )
    legacy_returns = run_backtest(legacy_config).daily_returns.reindex(calendar)
    if any(
        values.isna().any()
        for values in (
            pure_momentum_returns,
            pure_defender_returns,
            legacy_returns,
        )
    ):
        raise RuntimeError("period performance calendar is incomplete")
    period_returns = {
        "FORMAL": _period_performance(formal_returns, signal_timestamp),
        "LEGACY_MOMENTUM": _period_performance(
            legacy_returns, signal_timestamp
        ),
        "PURE_MOMENTUM": _period_performance(
            pure_momentum_returns, signal_timestamp
        ),
        "PURE_DEFENDER": _period_performance(
            pure_defender_returns, signal_timestamp
        ),
    }
    return SignalPerformanceSnapshot(
        since_date=since_timestamp.date(),
        current_holding_label=current_holding_label,
        current_holding_return=current_holding_return,
        concurrent_returns=concurrent,
        period_returns=period_returns,
    )
