"""Frozen implementation of the current full-data Relative Defender champion.

The production-facing implementation is intentionally independent from the
large research candidate catalog. Every threshold is an expanding quantile
of the primary ETF's own prior observations. Close signals execute at the
next open, and the sleeve not allocated to the primary ETF is assigned to the
fixed defensive asset, 511260.SH.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from .grid_reproduction import GridParams, _realized_volatility
from .relative_defender import (
    DEFENSIVE_ASSET,
    PRIMARY_ASSET,
    RELATIVE_DEFENDER_COST_RATES,
    RelativeDefenderParams,
    _simulate,
    load_relative_defender_market,
)


CHAMPION_CANDIDATE_ID = 9906
CHAMPION_SELECTION_DATE = "2026-08-20"


@dataclass(frozen=True)
class RelativeDefenderChampionParams:
    """Fixed parameters selected by the all-history six-ETF audit."""

    primary_asset: str = PRIMARY_ASSET
    defensive_asset: str = DEFENSIVE_ASSET
    range_window: int = 40
    lower_range_threshold: float = 0.10
    upper_range_threshold: float = 0.95
    position_step: float = 0.20
    cap_volatility_window: int = 20
    cap_quantile: float = 0.80
    cap_multiplier: float = 1.00
    factor_return_window: int = 15
    path_efficiency_window: int = 15
    low_volatility_window: int = 5
    return_weight: float = 0.25
    path_efficiency_weight: float = 0.25
    low_volatility_weight: float = 0.50
    regime_return_window: int = 60
    entry_quantile: float = 0.90
    exit_quantile: float = 0.60
    score_multiplier: float = 1.25
    minimum_history: int = 20

    def __post_init__(self) -> None:
        if self.primary_asset == self.defensive_asset:
            raise ValueError("primary and defensive assets must differ")
        if self.range_window < 2:
            raise ValueError("range_window must be at least 2")
        if not (
            0.0
            <= self.lower_range_threshold
            < self.upper_range_threshold
            <= 1.0
        ):
            raise ValueError("range thresholds must satisfy 0 <= lower < upper <= 1")
        if not 0.0 < self.position_step <= 1.0:
            raise ValueError("position_step must lie in (0, 1]")
        level_count = round(1.0 / self.position_step)
        if not np.isclose(level_count * self.position_step, 1.0, atol=1e-12):
            raise ValueError("position_step must divide the full position exactly")
        for name in (
            "cap_volatility_window",
            "factor_return_window",
            "path_efficiency_window",
            "low_volatility_window",
            "regime_return_window",
        ):
            if getattr(self, name) < 2:
                raise ValueError(f"{name} must be at least 2")
        if self.minimum_history < 1:
            raise ValueError("minimum_history must be positive")
        if not 0.0 < self.cap_quantile < 1.0:
            raise ValueError("cap_quantile must lie in (0, 1)")
        if not 0.0 < self.exit_quantile < self.entry_quantile < 1.0:
            raise ValueError("score quantiles must satisfy 0 < exit < entry < 1")
        if self.cap_multiplier <= 0.0 or self.score_multiplier <= 0.0:
            raise ValueError("quantile multipliers must be positive")
        weights = (
            self.return_weight,
            self.path_efficiency_weight,
            self.low_volatility_weight,
        )
        if any(weight <= 0.0 for weight in weights):
            raise ValueError("score weights must be positive")
        if not np.isclose(sum(weights), 1.0, atol=1e-12):
            raise ValueError("score weights must sum to one")


def champion_params(
    primary_asset: str = PRIMARY_ASSET,
) -> RelativeDefenderChampionParams:
    """Return the same frozen rule for a requested portability-test ETF."""
    return RelativeDefenderChampionParams(primary_asset=primary_asset)


def causal_expanding_quantile(
    values: pd.Series,
    quantile: float,
    minimum_history: int,
) -> pd.Series:
    """Quantile of finite observations strictly before the current close."""
    return values.shift(1).expanding(
        min_periods=minimum_history,
    ).quantile(quantile)


def _clean_prices(prices: pd.DataFrame) -> pd.DataFrame:
    frame = prices.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    required = ["date", "open", "high", "low", "close"]
    if frame.empty or frame[required].isna().any().any():
        raise ValueError("primary price data is empty or incomplete")
    if (frame[["open", "high", "low", "close"]] <= 0.0).any().any():
        raise ValueError("primary price data contains non-positive OHLC")
    return frame


def _rolling_range_location(close: pd.Series, window: int) -> pd.Series:
    rolling_low = close.rolling(window).min()
    rolling_high = close.rolling(window).max()
    width = rolling_high - rolling_low
    location = (close - rolling_low) / width.replace(0.0, np.nan)
    return location.where(width != 0.0, 0.5)


def _volatility_series(
    frame: pd.DataFrame,
    window: int,
    index: pd.DatetimeIndex,
) -> pd.Series:
    values = _realized_volatility(
        frame,
        GridParams(
            volatility_method="rogers_satchell",
            volatility_window=window,
        ),
    )
    return pd.Series(values, index=index, dtype=float)


def _base_signal_frame(
    location: pd.Series,
    volatility: pd.Series,
    cap_threshold: pd.Series,
    params: RelativeDefenderChampionParams,
) -> pd.DataFrame:
    level_count = int(round(1.0 / params.position_step))
    grid_level = level_count
    rows: list[dict[str, object]] = []
    for range_location, realized_volatility, threshold in zip(
        location,
        volatility,
        cap_threshold,
    ):
        reason = "hold"
        if np.isfinite(range_location):
            if range_location <= params.lower_range_threshold:
                grid_level = min(level_count, grid_level + 1)
                reason = "relative_low_add"
            elif range_location >= params.upper_range_threshold:
                grid_level = max(0, grid_level - 1)
                reason = "relative_high_reduce"
        else:
            reason = "warmup"

        grid_target = grid_level / level_count
        if (
            np.isfinite(realized_volatility)
            and realized_volatility > 0.0
            and np.isfinite(threshold)
            and threshold > 0.0
        ):
            raw_cap = min(1.0, float(threshold) / float(realized_volatility))
            cap_level = int(np.floor(raw_cap * level_count + 1e-12))
            volatility_cap = cap_level / level_count
        else:
            volatility_cap = 1.0
        base_target = min(grid_target, volatility_cap)
        if base_target < grid_target:
            reason = "adaptive_volatility_cap"
        rows.append({
            "signal_grid_target": grid_target,
            "signal_volatility_cap": volatility_cap,
            "signal_base_target": base_target,
            "signal_base_reason": reason,
        })
    return pd.DataFrame(rows, index=location.index)


def _hysteresis_state(
    score: pd.Series,
    regime_return: pd.Series,
    entry_line: pd.Series,
    exit_line: pd.Series,
) -> pd.Series:
    active = False
    states: list[bool] = []
    for value, regime, entry, exit_ in zip(
        score,
        regime_return,
        entry_line,
        exit_line,
    ):
        valid = all(np.isfinite(item) for item in (value, regime, entry, exit_))
        if not valid or regime <= 0.0:
            active = False
        elif not active and value > max(float(entry), 0.0):
            active = True
        elif active and value <= max(float(exit_), 0.0):
            active = False
        states.append(active)
    return pd.Series(states, index=score.index, dtype=bool)


def target_schedule(
    prices: pd.DataFrame,
    params: RelativeDefenderChampionParams = RelativeDefenderChampionParams(),
) -> pd.DataFrame:
    """Build all causal close signals and next-open portfolio targets."""
    frame = _clean_prices(prices)
    index = pd.DatetimeIndex(frame["date"])
    close = pd.Series(frame["close"].to_numpy(float), index=index)
    range_location = _rolling_range_location(close, params.range_window)

    volatility_20 = _volatility_series(
        frame,
        params.cap_volatility_window,
        index,
    )
    cap_threshold = (
        causal_expanding_quantile(
            volatility_20,
            params.cap_quantile,
            params.minimum_history,
        )
        * params.cap_multiplier
    )
    base = _base_signal_frame(
        range_location,
        volatility_20,
        cap_threshold,
        params,
    )

    factor_return = close / close.shift(params.factor_return_window) - 1.0
    path_length = close.diff().abs().rolling(params.path_efficiency_window).sum()
    path_efficiency = (
        (close - close.shift(params.path_efficiency_window))
        / path_length.replace(0.0, np.nan)
    )
    volatility_5 = _volatility_series(
        frame,
        params.low_volatility_window,
        index,
    )
    low_volatility_anchor = (
        causal_expanding_quantile(
            volatility_5,
            params.cap_quantile,
            params.minimum_history,
        )
        * params.cap_multiplier
    )
    low_volatility_score = (
        1.0 - volatility_5 / low_volatility_anchor
    ).clip(0.0, 1.0)
    score = (
        factor_return.clip(lower=0.0).pow(params.return_weight)
        * path_efficiency.clip(lower=0.0).pow(params.path_efficiency_weight)
        * low_volatility_score.clip(lower=0.0).pow(
            params.low_volatility_weight
        )
    )
    regime_return = close / close.shift(params.regime_return_window) - 1.0
    entry_line = (
        causal_expanding_quantile(
            score,
            params.entry_quantile,
            params.minimum_history,
        )
        * params.score_multiplier
    )
    exit_line = (
        causal_expanding_quantile(
            score,
            params.exit_quantile,
            params.minimum_history,
        )
        * params.score_multiplier
    )
    signal_active = _hysteresis_state(
        score,
        regime_return,
        entry_line,
        exit_line,
    )

    schedule = pd.DataFrame({
        "range_location": range_location,
        "realized_volatility_20": volatility_20,
        "cap_volatility_threshold": cap_threshold,
        **{column: base[column] for column in base.columns},
        "factor_return_15": factor_return,
        "path_efficiency_15": path_efficiency,
        "realized_volatility_5": volatility_5,
        "low_volatility_anchor": low_volatility_anchor,
        "low_volatility_score": low_volatility_score,
        "champion_score": score,
        "entry_score_threshold": entry_line,
        "exit_score_threshold": exit_line,
        "regime_return_60": regime_return,
        "signal_full_override_active": signal_active,
    }, index=index)
    schedule.index.name = "date"
    schedule["signal_primary_target"] = np.where(
        signal_active,
        1.0,
        schedule["signal_base_target"],
    )
    schedule["execution_full_override_active"] = signal_active.shift(
        1,
        fill_value=False,
    )
    schedule["primary_target"] = schedule["signal_primary_target"].shift(1)
    schedule.iloc[0, schedule.columns.get_loc("primary_target")] = 1.0
    schedule["primary_target"] = schedule["primary_target"].astype(float)
    schedule["defensive_target"] = 1.0 - schedule["primary_target"]

    base_reason = schedule["signal_base_reason"].shift(1).fillna("initial_buy")
    active = schedule["execution_full_override_active"].astype(bool)
    was_active = active.shift(1, fill_value=False)
    schedule["execution_reason"] = np.select(
        [active, was_active & ~active],
        ["champion_full_override", "champion_full_override_exit"],
        default=base_reason,
    )
    return schedule


def _portfolio_params(
    params: RelativeDefenderChampionParams,
) -> RelativeDefenderParams:
    """Return the minimal asset/execution carrier required by the simulator."""
    return RelativeDefenderParams(
        range_window=params.range_window,
        lower_percentile=params.lower_range_threshold,
        upper_percentile=params.upper_range_threshold,
        exposure_step=params.position_step,
        volatility_window=params.cap_volatility_window,
        primary_asset=params.primary_asset,
        defensive_asset=params.defensive_asset,
    )


def run_backtest(
    market: Mapping[str, pd.DataFrame] | None = None,
    params: RelativeDefenderChampionParams | None = None,
    cost_rate: Mapping[str, float] | float | None = None,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict[str, float | int | str | bool],
    pd.DataFrame,
]:
    """Run the frozen champion on one primary ETF and fixed 511260 filling."""
    selected = champion_params() if params is None else params
    execution_params = _portfolio_params(selected)
    prices = (
        load_relative_defender_market(
            execution_params,
            start=date(1900, 1, 1),
        )
        if market is None
        else {asset: frame.copy() for asset, frame in market.items()}
    )
    missing = {
        selected.primary_asset,
        selected.defensive_asset,
    } - set(prices)
    if missing:
        raise RuntimeError(f"missing local data for: {', '.join(sorted(missing))}")
    applied_costs = (
        RELATIVE_DEFENDER_COST_RATES if cost_rate is None else cost_rate
    )
    schedule = target_schedule(prices[selected.primary_asset], selected)
    simulation = _simulate(
        prices,
        schedule,
        execution_params,
        applied_costs,
    )
    daily = simulation.daily
    metrics: dict[str, float | int | str | bool] = {
        **simulation.metrics,
        "strategy": "relative_defender_champion",
        "research_status": "full_data_champion_not_independent_oos",
        "candidate_id": CHAMPION_CANDIDATE_ID,
        "selection_date": CHAMPION_SELECTION_DATE,
        "start": str(daily.index.min().date()),
        "end": str(daily.index.max().date()),
        **asdict(selected),
        "signal_timing": "close_signal_next_open",
        "rsi_enabled": False,
        "csi300_enabled": False,
    }
    return simulation.daily, simulation.trades, metrics, schedule


def main() -> None:
    daily, trades, metrics, signals = run_backtest()
    output = Path(__file__).parent / "deliverable"
    output.mkdir(parents=True, exist_ok=True)
    prefix = "relative_defender_champion"
    daily.to_csv(output / f"{prefix}_daily.csv")
    trades.to_csv(output / f"{prefix}_trades.csv", index=False)
    signals.to_csv(output / f"{prefix}_signals.csv")
    for key, value in metrics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
