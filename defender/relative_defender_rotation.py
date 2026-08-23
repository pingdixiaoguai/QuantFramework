"""Six-ETF rotation layered on the frozen Relative Defender champion.

The default version anchors the target schedule to 512890.SH.  The experimental
selected-asset-signal version applies the identical frozen champion parameters
to each ETF independently and uses the selected ETF's own causal target.  The
monthly selection rule is shared, and execution remains at the first trading
day's open using information available through the previous close.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from data.store import read_local

from .defender_opt_v2 import (
    CostRateSpec,
    _asof_price,
    _execute_portfolio_target,
    _indexed_market,
)
from .grid_reproduction import INITIAL_CAPITAL, TRADING_DAYS
from .relative_defender import PortfolioSimulation
from .relative_defender_champion import (
    RelativeDefenderChampionParams,
    champion_params,
    target_schedule as champion_target_schedule,
)


BASE_PRIMARY_ASSET = "512890.SH"
ROTATION_ASSETS = (
    BASE_PRIMARY_ASSET,
    "159545.SZ",
    "513530.SH",
    "515080.SH",
    "510880.SH",
    "563020.SH",
)
DEFENSIVE_ASSET = "511260.SH"
BASE_ANCHOR_SIGNAL = "base_anchor"
SELECTED_ASSET_SIGNAL = "selected_asset"
ROTATION_COST_RATES: dict[str, float] = {
    **{asset: 0.0001 for asset in ROTATION_ASSETS},
    DEFENSIVE_ASSET: 0.00001,
}


@dataclass(frozen=True)
class RelativeDefenderRotationParams:
    """Low-dimensional monthly selection overlay for the primary sleeve."""

    assets: tuple[str, ...] = ROTATION_ASSETS
    base_primary_asset: str = BASE_PRIMARY_ASSET
    defensive_asset: str = DEFENSIVE_ASSET
    range_threshold: float = 0.20
    reversal_lookback_days: int = 40
    trend_lookback_days: int = 150
    regime_lookback_days: int = 180
    regime_return_ceiling: float = 0.05
    selected_asset_weight: float = 1.00
    primary_signal_source: str = BASE_ANCHOR_SIGNAL
    rebalance_frequency: str = "monthly"
    warmup_low_scene_reversal: bool = True

    def __post_init__(self) -> None:
        if not self.assets:
            raise ValueError("assets must not be empty")
        if self.assets[0] != self.base_primary_asset:
            raise ValueError("base_primary_asset must be the first rotation asset")
        if len(set(self.assets)) != len(self.assets):
            raise ValueError("rotation assets must be unique")
        if self.defensive_asset in self.assets:
            raise ValueError("defensive asset must not be a rotation asset")
        if not 0.0 <= self.range_threshold <= 1.0:
            raise ValueError("range_threshold must lie in [0, 1]")
        for field_name in (
            "reversal_lookback_days",
            "trend_lookback_days",
            "regime_lookback_days",
        ):
            if getattr(self, field_name) < 2:
                raise ValueError(f"{field_name} must be at least 2")
        if not 0.0 <= self.selected_asset_weight <= 1.0:
            raise ValueError("selected_asset_weight must lie in [0, 1]")
        if self.primary_signal_source not in {
            BASE_ANCHOR_SIGNAL,
            SELECTED_ASSET_SIGNAL,
        }:
            raise ValueError(
                "primary_signal_source must be base_anchor or selected_asset"
            )
        if self.rebalance_frequency != "monthly":
            raise ValueError("only monthly rotation is supported")


def rotation_params() -> RelativeDefenderRotationParams:
    """Return a fresh copy of the frozen formal anchored rotation."""
    return RelativeDefenderRotationParams()


def bounded_rotation_params() -> RelativeDefenderRotationParams:
    """Return the conservative variant that rotates 25% of the primary sleeve."""
    return RelativeDefenderRotationParams(selected_asset_weight=0.25)


def selected_asset_signal_params() -> RelativeDefenderRotationParams:
    """Use the selected ETF's own frozen champion signal for its sleeve."""
    return RelativeDefenderRotationParams(
        primary_signal_source=SELECTED_ASSET_SIGNAL,
    )


def load_rotation_market(
    start: date = date(1900, 1, 1),
    end: date | None = None,
    params: RelativeDefenderRotationParams = RelativeDefenderRotationParams(),
) -> dict[str, pd.DataFrame]:
    """Load all locally available ETFs used by the rotation candidate."""
    result: dict[str, pd.DataFrame] = {}
    end_timestamp = pd.Timestamp(end or date.today())
    for asset in (*params.assets, params.defensive_asset):
        frame = read_local(asset)
        if frame is None or frame.empty:
            raise RuntimeError(f"missing local data for: {asset}")
        dates = pd.to_datetime(frame["date"])
        mask = (dates >= pd.Timestamp(start)) & (dates <= end_timestamp)
        selected = frame.loc[mask].copy()
        selected["date"] = pd.to_datetime(selected["date"])
        result[asset] = selected.reset_index(drop=True)
    return result


def _close_panel(
    market: Mapping[str, pd.DataFrame],
    assets: tuple[str, ...],
    calendar: pd.DatetimeIndex,
) -> pd.DataFrame:
    panel = pd.DataFrame(index=calendar, columns=assets, dtype=float)
    for asset in assets:
        if asset not in market:
            raise RuntimeError(f"missing market data for: {asset}")
        frame = market[asset].copy()
        frame["date"] = pd.to_datetime(frame["date"])
        close = (
            frame.sort_values("date")
            .drop_duplicates("date")
            .set_index("date")["close"]
            .astype(float)
        )
        panel[asset] = close.reindex(calendar)
    return panel


def _stable_extreme(values: pd.Series, find_maximum: bool) -> str | None:
    """Select an extreme while preserving the configured asset tie order."""
    finite = values.loc[np.isfinite(values.to_numpy(float))]
    if finite.empty:
        return None
    extreme = finite.max() if find_maximum else finite.min()
    for asset, value in finite.items():
        if np.isclose(float(value), float(extreme), atol=1e-14):
            return str(asset)
    raise RuntimeError("failed to select a finite extreme")


def rotation_schedule(
    market: Mapping[str, pd.DataFrame],
    champion_schedule: pd.DataFrame,
    params: RelativeDefenderRotationParams = RelativeDefenderRotationParams(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build causal monthly ETF selection and primary-sleeve allocations."""
    calendar = pd.DatetimeIndex(champion_schedule.index)
    if calendar.empty or not calendar.is_monotonic_increasing or not calendar.is_unique:
        raise ValueError("champion schedule calendar must be non-empty and ordered")
    if "range_location" not in champion_schedule:
        raise ValueError("champion schedule is missing range_location")

    close = _close_panel(market, params.assets, calendar)
    reversal_return = close / close.shift(params.reversal_lookback_days) - 1.0
    trend_return = close / close.shift(params.trend_lookback_days) - 1.0
    base_regime_return = (
        close[params.base_primary_asset]
        / close[params.base_primary_asset].shift(params.regime_lookback_days)
        - 1.0
    )

    selected_asset = params.base_primary_asset
    selected_reason = "warmup_base"
    rows: list[dict[str, object]] = []
    events: list[dict[str, object]] = []
    for position, execution_date in enumerate(calendar):
        execution_date = pd.Timestamp(execution_date)
        if position > 0:
            signal_date = pd.Timestamp(calendar[position - 1])
            month_changed = (
                execution_date.to_period("M") != signal_date.to_period("M")
            )
            if month_changed:
                range_location = float(
                    champion_schedule.iloc[position - 1]["range_location"]
                )
                regime_return = float(base_regime_return.iloc[position - 1])
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
                signal_values = (
                    reversal_return.iloc[position - 1]
                    if use_reversal
                    else trend_return.iloc[position - 1]
                )
                next_asset = _stable_extreme(
                    signal_values,
                    find_maximum=not use_reversal,
                )
                next_reason = (
                    "low_scene_reversal"
                    if use_reversal
                    else "long_term_trend"
                )
                if next_asset is None:
                    next_asset = params.base_primary_asset
                    next_reason = "insufficient_history_base"
                if next_asset != selected_asset or next_reason != selected_reason:
                    events.append({
                        "signal_date": signal_date,
                        "execution_date": execution_date,
                        "old_selected_asset": selected_asset,
                        "new_selected_asset": next_asset,
                        "selection_reason": next_reason,
                        "range_location": range_location,
                        "base_regime_return": regime_return,
                        "selected_reversal_return": float(
                            reversal_return.iloc[position - 1].get(
                                next_asset, np.nan
                            )
                        ),
                        "selected_trend_return": float(
                            trend_return.iloc[position - 1].get(
                                next_asset, np.nan
                            )
                        ),
                    })
                selected_asset = next_asset
                selected_reason = next_reason

        fractions = {asset: 0.0 for asset in params.assets}
        if selected_asset == params.base_primary_asset:
            fractions[params.base_primary_asset] = 1.0
        else:
            fractions[params.base_primary_asset] = 1.0 - params.selected_asset_weight
            fractions[selected_asset] = params.selected_asset_weight
        rows.append({
            "date": execution_date,
            "selected_asset": selected_asset,
            "selection_reason": selected_reason,
            "base_regime_return": (
                float(base_regime_return.iloc[position - 1])
                if position > 0
                else np.nan
            ),
            **{f"primary_fraction_{asset}": weight for asset, weight in fractions.items()},
        })

    return pd.DataFrame(rows).set_index("date"), pd.DataFrame(events)


def _effective_champion_schedule(
    market: Mapping[str, pd.DataFrame],
    base_schedule: pd.DataFrame,
    selection: pd.DataFrame,
    params: RelativeDefenderRotationParams,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Choose the causal champion signal source without changing its rule."""
    calendar = pd.DatetimeIndex(base_schedule.index)
    effective = base_schedule.copy()
    source_asset = pd.Series(
        params.base_primary_asset,
        index=calendar,
        dtype=object,
    )
    fell_back = pd.Series(False, index=calendar, dtype=bool)

    if params.primary_signal_source == SELECTED_ASSET_SIGNAL:
        for asset in params.assets:
            chosen = selection["selected_asset"].eq(asset)
            if asset == params.base_primary_asset or not bool(chosen.any()):
                continue
            asset_schedule = champion_target_schedule(
                market[asset],
                champion_params(asset),
            )
            aligned = asset_schedule.reindex(calendar).ffill()
            available = aligned["primary_target"].notna()
            use_asset = chosen & available
            unavailable = chosen & ~available
            common_columns = effective.columns.intersection(aligned.columns)
            for column in common_columns:
                effective.loc[use_asset, column] = aligned.loc[
                    use_asset,
                    column,
                ]
            source_asset.loc[use_asset] = asset
            fell_back.loc[unavailable] = True

    effective["signal_source_asset"] = source_asset
    effective["signal_fallback_to_base"] = fell_back
    audited_selection = selection.copy()
    audited_selection["signal_source_asset"] = source_asset
    audited_selection["signal_primary_target"] = effective[
        "primary_target"
    ].astype(float)
    audited_selection["base_anchor_primary_target"] = base_schedule[
        "primary_target"
    ].astype(float)
    audited_selection["signal_fallback_to_base"] = fell_back
    return effective, audited_selection


def _performance_metrics(daily: pd.DataFrame) -> dict[str, float | int]:
    returns = daily["return"].dropna().astype(float)
    curve = daily["nav"].astype(float) / INITIAL_CAPITAL
    drawdown = curve / curve.cummax() - 1.0
    stdev = float(returns.std(ddof=1)) if len(returns) > 1 else 0.0
    years = len(returns) / TRADING_DAYS
    return {
        "observations": int(len(returns)),
        "final_nav": float(daily["nav"].iloc[-1]),
        "total_return": float(curve.iloc[-1] - 1.0),
        "annualized_return": float(curve.iloc[-1] ** (1.0 / years) - 1.0),
        "annualized_volatility": float(stdev * np.sqrt(TRADING_DAYS)),
        "sharpe": (
            float(returns.mean() / stdev * np.sqrt(TRADING_DAYS))
            if stdev
            else 0.0
        ),
        "max_drawdown": float(drawdown.min()),
    }


def _simulate_rotation(
    market: Mapping[str, pd.DataFrame],
    champion_schedule: pd.DataFrame,
    selection: pd.DataFrame,
    params: RelativeDefenderRotationParams,
    cost_rates: CostRateSpec,
) -> PortfolioSimulation:
    """Execute the anchored rotation sleeve and fixed defensive sleeve."""
    indexed = _indexed_market(market)
    calendar = pd.DatetimeIndex(champion_schedule.index)
    selection = selection.reindex(calendar)
    cash = INITIAL_CAPITAL
    shares: dict[str, float] = {}
    previous_target: dict[str, float] = {}
    previous_closes: dict[str, float] = {}
    last_nav = 0.0
    rows: list[dict[str, object]] = []
    trades: list[dict[str, object]] = []
    total_cost = 0.0
    gross_pnl = {asset: 0.0 for asset in (*params.assets, params.defensive_asset)}

    for position, timestamp in enumerate(calendar):
        timestamp = pd.Timestamp(timestamp)
        primary_weight = float(champion_schedule.at[timestamp, "primary_target"])
        fractions = {
            asset: float(selection.at[timestamp, f"primary_fraction_{asset}"])
            for asset in params.assets
        }
        target = {
            asset: primary_weight * fraction
            for asset, fraction in fractions.items()
            if primary_weight * fraction > 1e-14
        }
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

        if target != previous_target:
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
                    "selected_asset": selection.at[timestamp, "selected_asset"],
                    "signal_source_asset": selection.at[
                        timestamp,
                        "signal_source_asset",
                    ],
                    "selection_reason": selection.at[timestamp, "selection_reason"],
                    "primary_target": primary_weight,
                    "signal_execution_reason": champion_schedule.at[
                        timestamp,
                        "execution_reason",
                    ],
                    **execution,
                })
            previous_target = target

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
        row: dict[str, object] = {
            "date": timestamp,
            "nav": nav,
            "return": daily_return,
            "cash": cash,
            "primary_target": primary_weight,
            "defensive_target": 1.0 - primary_weight,
            "selected_asset": selection.at[timestamp, "selected_asset"],
            "signal_source_asset": selection.at[
                timestamp,
                "signal_source_asset",
            ],
            "selection_reason": selection.at[timestamp, "selection_reason"],
            "signal_execution_reason": champion_schedule.at[
                timestamp,
                "execution_reason",
            ],
            "base_anchor_primary_target": float(
                selection.at[timestamp, "base_anchor_primary_target"]
            ),
            "signal_fallback_to_base": bool(
                selection.at[timestamp, "signal_fallback_to_base"]
            ),
        }
        for asset in (*params.assets, params.defensive_asset):
            row[f"target_{asset}"] = target.get(asset, 0.0)
            row[f"weight_{asset}"] = actual_weights.get(asset, 0.0)
            gross = day_gross.get(asset, 0.0)
            cost = day_cost.get(asset, 0.0)
            row[f"gross_pnl_{asset}"] = gross
            row[f"transaction_cost_{asset}"] = cost
            row[f"net_pnl_{asset}"] = gross - cost
        rows.append(row)
        last_nav = nav
        previous_closes = close_prices

    daily = pd.DataFrame(rows).set_index("date")
    trade_frame = pd.DataFrame(trades)
    metrics = _performance_metrics(daily)
    metrics.update({
        "execution_count": int(len(trade_frame)),
        "total_turnover": (
            float(trade_frame["turnover"].sum()) if not trade_frame.empty else 0.0
        ),
        "total_cost": total_cost,
        "rotation_switch_count": int(
            selection["selected_asset"].ne(selection["selected_asset"].shift()).sum()
            - 1
        ),
    })
    for asset, pnl in gross_pnl.items():
        metrics[f"gross_pnl_{asset.split('.', maxsplit=1)[0]}"] = pnl
    return PortfolioSimulation(daily, trade_frame, metrics)


def run_backtest(
    market: Mapping[str, pd.DataFrame] | None = None,
    params: RelativeDefenderRotationParams | None = None,
    cost_rate: CostRateSpec | None = None,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict[str, float | int | str | bool | tuple[str, ...]],
    pd.DataFrame,
    pd.DataFrame,
]:
    """Run the bounded rotation candidate over the frozen champion schedule."""
    selected = rotation_params() if params is None else params
    prices = (
        load_rotation_market(params=selected)
        if market is None
        else {asset: frame.copy() for asset, frame in market.items()}
    )
    required = {*selected.assets, selected.defensive_asset}
    missing = sorted(required - set(prices))
    if missing:
        raise RuntimeError(f"missing local data for: {', '.join(missing)}")

    core_params: RelativeDefenderChampionParams = champion_params(
        selected.base_primary_asset
    )
    base_schedule = champion_target_schedule(
        prices[selected.base_primary_asset],
        core_params,
    )
    selection, events = rotation_schedule(prices, base_schedule, selected)
    champion_schedule, selection = _effective_champion_schedule(
        prices,
        base_schedule,
        selection,
        selected,
    )
    applied_costs = ROTATION_COST_RATES if cost_rate is None else cost_rate
    simulation = _simulate_rotation(
        prices,
        champion_schedule,
        selection,
        selected,
        applied_costs,
    )
    is_formal_anchor = (
        selected.primary_signal_source == BASE_ANCHOR_SIGNAL
        and selected.selected_asset_weight == 1.0
        and selected.assets == ROTATION_ASSETS
        and selected.base_primary_asset == BASE_PRIMARY_ASSET
        and selected.defensive_asset == DEFENSIVE_ASSET
    )
    metrics: dict[str, float | int | str | bool | tuple[str, ...]] = {
        **simulation.metrics,
        "strategy": (
            "relative_defender_rotation_selected_asset_signal"
            if selected.primary_signal_source == SELECTED_ASSET_SIGNAL
            else "relative_defender_rotation"
        ),
        "research_status": (
            "formal_production_signal_retrospective_evidence"
            if is_formal_anchor
            else "retrospective_shadow_candidate_not_oos"
        ),
        "formal_status": (
            "production_signal_frozen" if is_formal_anchor else "not_formal"
        ),
        "formal_promotion_date": "2026-08-20" if is_formal_anchor else None,
        "start": str(simulation.daily.index.min().date()),
        "end": str(simulation.daily.index.max().date()),
        **asdict(selected),
        "core_strategy": "relative_defender_champion",
        "core_parameters_changed": False,
        "indicator_input_switches_with_selected_asset": (
            selected.primary_signal_source == SELECTED_ASSET_SIGNAL
        ),
        "signal_fallback_observations": int(
            selection["signal_fallback_to_base"].sum()
        ),
        "signal_timing": "prior_close_month_start_open",
    }
    return simulation.daily, simulation.trades, metrics, selection, events


def main() -> None:
    daily, trades, metrics, selection, events = run_backtest()
    output = Path(__file__).parent / "deliverable"
    output.mkdir(parents=True, exist_ok=True)
    prefix = "relative_defender_rotation"
    daily.to_csv(output / f"{prefix}_daily.csv")
    trades.to_csv(output / f"{prefix}_trades.csv", index=False)
    selection.to_csv(output / f"{prefix}_selection.csv")
    events.to_csv(output / f"{prefix}_events.csv", index=False)
    for key, value in metrics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
