"""Ablation experiments around the calendar-monthly defensive baseline.

Every variant changes exactly one dimension of the confirmed baseline:
pool, score calculation, or rebalance mechanism.

Baseline = 4-ETF pool (512890/511260/511360/511880 at 35/40/15/10), 20-day
reversal ranks, global 50% tilt, month-start calendar trigger, next-open
execution, 100-share lots, 0.05% one-way cost, 10,000 CNY minimum order,
20,000 CNY monthly deposit on the first trading day.

Run with: python -m research.defensive_etf_sharpe.calendar_ablation
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from .engine import MarketData, load_market_data
from .factor_allocation import (
    FactorSpec,
    MechanismSpec,
    adjusted_weights,
    factor_ranks,
    factor_zscores,
)
from .rebalance_timing import daily_reversal_targets
from .strategy import CASH_ASSET, STATIC_BENCHMARK_TARGET, metrics_for_daily
from .threshold_rebalance import TriggerSpec, simulate_threshold_rebalance


ROOT = Path(__file__).parent
OUTPUT = ROOT / "calendar_ablation_experiments"

COST_RATE = 0.0005
LOT_SIZE = 100
MONTHLY_DEPOSIT = 20_000.0
MIN_REBALANCE_NOTIONAL = 10_000.0
RETURN_FLOOR = 0.05

BASELINE_TARGET = {
    "512890.SH": 0.35,
    "511260.SH": 0.40,
    "511360.SH": 0.15,
    "511880.SH": 0.10,
}

# Extra defensive candidate not present in universe.yaml: money-market ETF
# listed 2013-01-28, well past the three-year establishment requirement.
EXTRA_POOL_ASSETS = ("511990.SH",)

REVERSAL_20 = FactorSpec("reversal_20", "reversal", 20, "negative 20-day simple return")
GLOBAL_TILT_050 = MechanismSpec("global_tilt_050", "global_tilt", 0.50, "baseline * (1 + 0.5 * rank)")

CALENDAR_TRIGGER = TriggerSpec(
    "calendar_monthly_reference",
    "calendar_monthly",
    0.5,
    description="rebalance after every month-start close",
)


def _split_sleeve(weight: float, assets: list[str]) -> dict[str, float]:
    return {asset: weight / len(assets) for asset in assets}


def _pool_target(
    dividend: list[str] | None = None,
    sovereign: list[str] | None = None,
    credit: list[str] | None = None,
    money: list[str] | None = None,
) -> dict[str, float]:
    target: dict[str, float] = {}
    target.update(_split_sleeve(0.35, dividend or ["512890.SH"]))
    target.update(_split_sleeve(0.40, sovereign or ["511260.SH"]))
    target.update(_split_sleeve(0.15, credit or ["511360.SH"]))
    target.update(_split_sleeve(0.10, money or ["511880.SH"]))
    return target


def _history(data: MarketData, asset: str, timestamp: pd.Timestamp) -> pd.Series:
    close = data.closes[asset]
    end = int(close.index.searchsorted(timestamp, side="right"))
    return close.iloc[:end].dropna().astype(float)


def _annualized_vol(history: pd.Series, window: int) -> float:
    if len(history) < window + 1:
        return float("nan")
    returns = history.pct_change().dropna().tail(window)
    return float(returns.std(ddof=1) * np.sqrt(252.0))


def _zscore_targets(
    data: MarketData,
    baseline: Mapping[str, float],
    factor: FactorSpec,
    mechanism: MechanismSpec,
) -> dict[pd.Timestamp, dict[str, float]]:
    assets = tuple(baseline)
    return {
        timestamp: adjusted_weights(
            factor_zscores(data, timestamp, factor, assets),
            mechanism,
            dict(baseline),
        )
        for timestamp in data.dates
    }


def _dynamic_inverse_vol_targets(
    data: MarketData,
    baseline: Mapping[str, float],
    factor: FactorSpec,
    mechanism: MechanismSpec,
    vol_window: int = 60,
) -> dict[pd.Timestamp, dict[str, float]]:
    """Replace the fixed bi with global inverse-volatility bi, keep the tilt."""
    assets = tuple(baseline)
    schedule: dict[pd.Timestamp, dict[str, float]] = {}
    for timestamp in data.dates:
        vols = {
            asset: _annualized_vol(_history(data, asset, timestamp), vol_window)
            for asset in assets
        }
        raw = {
            asset: baseline[asset] / vols[asset]
            for asset in assets
            if np.isfinite(vols[asset]) and vols[asset] > 0
        }
        if len(raw) < len(assets):
            dynamic_base = dict(baseline)
        else:
            total = sum(raw.values())
            dynamic_base = {asset: value / total for asset, value in raw.items()}
        schedule[timestamp] = adjusted_weights(
            factor_ranks(data, timestamp, factor, assets),
            mechanism,
            dynamic_base,
        )
    return schedule


def _extended_metrics(result) -> dict[str, float | int | bool]:
    returns = result.daily["return"].dropna().astype(float)
    basic = metrics_for_daily(result.daily)
    downside = returns.loc[returns < 0].std(ddof=1)
    sortino = float(returns.mean() / downside * np.sqrt(252.0)) if downside > 0 else np.nan
    calmar = (
        basic["annualized_return"] / abs(basic["max_drawdown"])
        if basic["max_drawdown"] < 0
        else np.nan
    )
    annual = returns.groupby(returns.index.year).apply(lambda values: (1.0 + values).prod() - 1.0)
    notional = float(result.trades["notional"].sum()) if not result.trades.empty else 0.0
    rebalance_trades = (
        result.trades.loc[result.trades["reason"] == "threshold_rebalance"]
        if "reason" in result.trades.columns
        else result.trades
    )
    return {
        **basic,
        "sortino": sortino,
        "calmar": calmar,
        "worst_calendar_return": float(annual.min()),
        "best_calendar_return": float(annual.max()),
        "final_nav": result.final_nav,
        "total_deposits": result.total_deposits,
        "trades": len(result.trades),
        "rebalance_dates": (
            int(pd.to_datetime(rebalance_trades["date"]).nunique())
            if not rebalance_trades.empty
            else 0
        ),
        "traded_notional": notional,
        "estimated_transaction_cost": notional * COST_RATE,
        "average_cash_weight": float(result.daily["cash_weight"].mean()),
        "meets_5pct_return_floor": basic["annualized_return"] >= RETURN_FLOOR,
    }


TargetBuilder = Callable[[MarketData], dict[pd.Timestamp, dict[str, float]]]


def _rank_targets(
    baseline: Mapping[str, float],
    factor: FactorSpec = REVERSAL_20,
    mechanism: MechanismSpec = GLOBAL_TILT_050,
) -> TargetBuilder:
    def build(data: MarketData) -> dict[pd.Timestamp, dict[str, float]]:
        return daily_reversal_targets(data, dict(baseline), factor, mechanism)

    return build


def _zscore_target_builder(
    baseline: Mapping[str, float],
    factor: FactorSpec,
    mechanism: MechanismSpec,
) -> TargetBuilder:
    def build(data: MarketData) -> dict[pd.Timestamp, dict[str, float]]:
        return _zscore_targets(data, baseline, factor, mechanism)

    return build


def _dynamic_bi_builder(
    baseline: Mapping[str, float],
    factor: FactorSpec = REVERSAL_20,
    mechanism: MechanismSpec = GLOBAL_TILT_050,
    vol_window: int = 60,
) -> TargetBuilder:
    def build(data: MarketData) -> dict[pd.Timestamp, dict[str, float]]:
        return _dynamic_inverse_vol_targets(data, baseline, factor, mechanism, vol_window)

    return build


def _risk_only_tilt_builder(
    baseline: Mapping[str, float],
    strength: float,
    money_asset: str = "511880.SH",
    factor: FactorSpec = REVERSAL_20,
) -> TargetBuilder:
    """Tilt only the risk assets; the money sleeve keeps its exact budget."""
    def build(data: MarketData) -> dict[pd.Timestamp, dict[str, float]]:
        risk_assets = tuple(asset for asset in baseline if asset != money_asset)
        risk_budget = 1.0 - baseline[money_asset]
        schedule: dict[pd.Timestamp, dict[str, float]] = {}
        for timestamp in data.dates:
            ranks = factor_ranks(data, timestamp, factor, risk_assets)
            raw = {
                asset: max(0.0, baseline[asset] * (1.0 + strength * ranks.get(asset, 0.0)))
                for asset in risk_assets
            }
            total = sum(raw.values())
            weights = (
                {asset: risk_budget * value / total for asset, value in raw.items()}
                if total > 0
                else {asset: risk_budget * baseline[asset] / sum(baseline[a] for a in risk_assets) for asset in risk_assets}
            )
            weights[money_asset] = baseline[money_asset]
            schedule[timestamp] = weights
        return schedule

    return build


def _risk_only_zsigmoid_builder(
    baseline: Mapping[str, float],
    strength: float,
    money_asset: str = "511880.SH",
    factor: FactorSpec = REVERSAL_20,
) -> TargetBuilder:
    """Z-score + sigmoid tilt among risk assets only; money sleeve fixed."""
    def build(data: MarketData) -> dict[pd.Timestamp, dict[str, float]]:
        risk_assets = tuple(asset for asset in baseline if asset != money_asset)
        risk_budget = 1.0 - baseline[money_asset]
        risk_base_total = sum(baseline[asset] for asset in risk_assets)
        schedule: dict[pd.Timestamp, dict[str, float]] = {}
        for timestamp in data.dates:
            zscores = factor_zscores(data, timestamp, factor, risk_assets)
            raw = {
                asset: max(
                    0.0,
                    baseline[asset] * (1.0 + strength * float(np.tanh(zscores.get(asset, 0.0)))),
                )
                for asset in risk_assets
            }
            total = sum(raw.values())
            weights = (
                {asset: risk_budget * value / total for asset, value in raw.items()}
                if total > 0
                else {asset: risk_budget * baseline[asset] / risk_base_total for asset in risk_assets}
            )
            weights[money_asset] = baseline[money_asset]
            schedule[timestamp] = weights
        return schedule

    return build


def _pool_initial_target(name: str) -> dict[str, float]:
    if name.startswith("pool_add_510880"):
        return _pool_target(dividend=["512890.SH", "510880.SH"])
    if name.startswith("pool_add_515450"):
        return _pool_target(dividend=["512890.SH", "515450.SH"])
    if name.startswith("pool_add_511010"):
        return _pool_target(sovereign=["511260.SH", "511010.SH"])
    if name.startswith("pool_add_511090"):
        return _pool_target(sovereign=["511260.SH", "511090.SH"])
    if name.startswith("pool_add_511990"):
        return _pool_target(money=["511880.SH", "511990.SH"])
    if name.startswith("pool_full_8etf"):
        return dict(STATIC_BENCHMARK_TARGET)
    return dict(BASELINE_TARGET)


def _experiments() -> list[dict[str, object]]:
    """One change per experiment relative to the calendar baseline."""
    base = BASELINE_TARGET
    specs: list[dict[str, object]] = []

    def add(
        name: str,
        direction: str,
        change: str,
        targets: TargetBuilder,
        trigger: TriggerSpec = CALENDAR_TRIGGER,
        min_notional: float = MIN_REBALANCE_NOTIONAL,
    ) -> None:
        specs.append({
            "name": name,
            "direction": direction,
            "change": change,
            "targets": targets,
            "trigger": trigger,
            "min_notional": min_notional,
            "initial_target": _pool_initial_target(name),
        })

    add(
        "baseline_calendar",
        "baseline",
        "confirmed baseline: 4-ETF pool, reversal_20 ranks, 50% global tilt, month-start calendar rebalance",
        _rank_targets(base),
    )

    # direction 1: defensive pool expansion
    add(
        "pool_add_510880",
        "pool",
        "add 510880.SH (SSE dividend, 2013); dividend sleeve split 17.5/17.5",
        _rank_targets(_pool_target(dividend=["512890.SH", "510880.SH"])),
    )
    add(
        "pool_add_515450",
        "pool",
        "add 515450.SH (S&P China large dividend low-vol 50, 2020); dividend sleeve split 17.5/17.5",
        _rank_targets(_pool_target(dividend=["512890.SH", "515450.SH"])),
    )
    add(
        "pool_add_511010",
        "pool",
        "add 511010.SH (5y sovereign, 2013); sovereign sleeve split 20/20",
        _rank_targets(_pool_target(sovereign=["511260.SH", "511010.SH"])),
    )
    add(
        "pool_add_511090",
        "pool",
        "add 511090.SH (30y sovereign, listed 2023-06, just past 3y); sovereign sleeve split 20/20",
        _rank_targets(_pool_target(sovereign=["511260.SH", "511090.SH"])),
    )
    add(
        "pool_add_511990",
        "pool",
        "add 511990.SH (Huabao money market, 2013); money sleeve split 5/5",
        _rank_targets(_pool_target(money=["511880.SH", "511990.SH"])),
    )
    add(
        "pool_full_8etf",
        "pool",
        "restore the confirmed 8-ETF defensive universe (3 dividend / 3 sovereign / 1 credit / 1 money)",
        _rank_targets(dict(STATIC_BENCHMARK_TARGET)),
    )

    # direction 2: score calculation
    for name, window in (
        ("score_reversal_05", 5),
        ("score_reversal_10", 10),
        ("score_reversal_40", 40),
        ("score_reversal_60", 60),
    ):
        add(
            name,
            "score",
            f"reversal window 20 -> {window}",
            _rank_targets(base, FactorSpec(name, "reversal", window, f"negative {window}-day return")),
        )

    add(
        "score_momentum_20",
        "score",
        "factor direction flip: positive 20-day momentum instead of reversal",
        _rank_targets(base, FactorSpec("momentum_20", "momentum", 20, "20-day simple return")),
    )
    add(
        "score_reversal_voladj_20",
        "score",
        "reversal divided by window volatility (vol-adjusted reversal)",
        _rank_targets(
            base,
            FactorSpec("reversal_voladj_20", "reversal_vol_adjusted", 20, "reversal / window vol"),
        ),
    )
    add(
        "score_low_vol_20",
        "score",
        "replace reversal with 20-day low-volatility rank",
        _rank_targets(base, FactorSpec("low_vol_20", "low_vol", 20, "negative 20-day vol")),
    )
    add(
        "score_quality_momentum_20",
        "score",
        "replace reversal with 20-day momentum x Kaufman ER",
        _rank_targets(
            base,
            FactorSpec("quality_momentum_20", "quality_momentum", 20, "momentum x ER"),
        ),
    )
    add(
        "score_momentum_low_vol_60",
        "score",
        "composite: 60-day momentum and 60-day low-vol ranks averaged",
        _rank_targets(
            base,
            FactorSpec("momentum_low_vol_60", "momentum_low_vol", 60, "mom+lowvol composite"),
        ),
    )

    zscore_factor = FactorSpec("reversal_20", "reversal", 20, "negative 20-day simple return")
    add(
        "score_zscore_exp05",
        "score",
        "z-score normalization + exponential tilt exp(0.5 * z)",
        _zscore_target_builder(base, zscore_factor, MechanismSpec("exp_tilt_05", "exp_tilt", 0.50, "exp tilt")),
    )
    add(
        "score_zscore_exp03",
        "score",
        "z-score normalization + exponential tilt exp(0.3 * z)",
        _zscore_target_builder(base, zscore_factor, MechanismSpec("exp_tilt_03", "exp_tilt", 0.30, "exp tilt")),
    )
    add(
        "score_zscore_sigmoid05",
        "score",
        "z-score normalization + sigmoid tilt (1 + 0.5 * tanh(z))",
        _zscore_target_builder(base, zscore_factor, MechanismSpec("sigmoid_tilt_05", "sigmoid_tilt", 0.50, "sigmoid tilt")),
    )
    add(
        "score_zscore_additive",
        "score",
        "z-score normalization + additive tilt bi + 0.05 * z",
        _zscore_target_builder(base, zscore_factor, MechanismSpec("additive_tilt_005", "additive_tilt", 0.05, "additive tilt")),
    )

    add(
        "score_tilt_025",
        "score",
        "tilt strength 0.50 -> 0.25",
        _rank_targets(base, REVERSAL_20, MechanismSpec("global_tilt_025", "global_tilt", 0.25, "tilt 25%")),
    )
    add(
        "score_tilt_075",
        "score",
        "tilt strength 0.50 -> 0.75",
        _rank_targets(base, REVERSAL_20, MechanismSpec("global_tilt_075", "global_tilt", 0.75, "tilt 75%")),
    )
    add(
        "score_tilt_100",
        "score",
        "tilt strength 0.50 -> 1.00",
        _rank_targets(base, REVERSAL_20, MechanismSpec("global_tilt_100", "global_tilt", 1.00, "tilt 100%")),
    )
    add(
        "score_blend_050",
        "score",
        "50/50 blend of baseline and rank-tilted portfolio",
        _rank_targets(base, REVERSAL_20, MechanismSpec("global_blend_050", "global_blend", 0.50, "blend 50%")),
    )
    add(
        "score_dynamic_bi_invvol60",
        "score",
        "dynamic bi: fixed weights rescaled by inverse 60-day volatility",
        _dynamic_bi_builder(base, vol_window=60),
    )

    # direction 3: rebalance mechanism
    add(
        "mech_calendar_or_drift_05",
        "mechanism",
        "add 5% portfolio one-sided drift trigger on top of calendar",
        _rank_targets(base),
        trigger=TriggerSpec("calendar_or_drift_05", "calendar_or_portfolio_drift", 0.05),
    )
    add(
        "mech_calendar_or_drift_10",
        "mechanism",
        "add 10% portfolio one-sided drift trigger on top of calendar",
        _rank_targets(base),
        trigger=TriggerSpec("calendar_or_drift_10", "calendar_or_portfolio_drift", 0.10),
    )
    add(
        "mech_calendar_or_drift_15",
        "mechanism",
        "add 15% portfolio one-sided drift trigger on top of calendar",
        _rank_targets(base),
        trigger=TriggerSpec("calendar_or_drift_15", "calendar_or_portfolio_drift", 0.15),
    )
    add(
        "mech_min_notional_5k",
        "mechanism",
        "minimum rebalance order notional 10,000 -> 5,000",
        _rank_targets(base),
        min_notional=5_000.0,
    )
    add(
        "mech_min_notional_20k",
        "mechanism",
        "minimum rebalance order notional 10,000 -> 20,000",
        _rank_targets(base),
        min_notional=20_000.0,
    )
    add(
        "mech_min_notional_0",
        "mechanism",
        "remove the minimum rebalance order notional floor",
        _rank_targets(base),
        min_notional=0.0,
    )

    # round 2: follow the round-1 winners and densify the grids --------------
    add(
        "score_tilt_125",
        "score",
        "tilt strength 0.50 -> 1.25",
        _rank_targets(base, REVERSAL_20, MechanismSpec("global_tilt_125", "global_tilt", 1.25, "tilt 125%")),
    )
    add(
        "score_tilt_150",
        "score",
        "tilt strength 0.50 -> 1.50",
        _rank_targets(base, REVERSAL_20, MechanismSpec("global_tilt_150", "global_tilt", 1.50, "tilt 150%")),
    )
    add(
        "score_tilt_200",
        "score",
        "tilt strength 0.50 -> 2.00",
        _rank_targets(base, REVERSAL_20, MechanismSpec("global_tilt_200", "global_tilt", 2.00, "tilt 200%")),
    )
    add(
        "score_zscore_exp075",
        "score",
        "z-score normalization + exponential tilt exp(0.75 * z)",
        _zscore_target_builder(base, zscore_factor, MechanismSpec("exp_tilt_075", "exp_tilt", 0.75, "exp tilt")),
    )
    add(
        "score_zscore_exp100",
        "score",
        "z-score normalization + exponential tilt exp(1.0 * z)",
        _zscore_target_builder(base, zscore_factor, MechanismSpec("exp_tilt_100", "exp_tilt", 1.00, "exp tilt")),
    )
    add(
        "score_zscore_sigmoid075",
        "score",
        "z-score normalization + sigmoid tilt (1 + 0.75 * tanh(z))",
        _zscore_target_builder(base, zscore_factor, MechanismSpec("sigmoid_tilt_075", "sigmoid_tilt", 0.75, "sigmoid tilt")),
    )
    add(
        "score_zscore_sigmoid100",
        "score",
        "z-score normalization + sigmoid tilt (1 + 1.0 * tanh(z))",
        _zscore_target_builder(base, zscore_factor, MechanismSpec("sigmoid_tilt_100", "sigmoid_tilt", 1.00, "sigmoid tilt")),
    )
    add(
        "score_zscore_additive_010",
        "score",
        "z-score normalization + additive tilt bi + 0.10 * z",
        _zscore_target_builder(base, zscore_factor, MechanismSpec("additive_tilt_010", "additive_tilt", 0.10, "additive tilt")),
    )
    for name, window in (
        ("score_reversal_15", 15),
        ("score_reversal_25", 25),
        ("score_reversal_30", 30),
    ):
        add(
            name,
            "score",
            f"reversal window 20 -> {window}",
            _rank_targets(base, FactorSpec(name, "reversal", window, f"negative {window}-day return")),
        )
    add(
        "score_blend_025",
        "score",
        "75/25 blend of baseline and rank-tilted portfolio",
        _rank_targets(base, REVERSAL_20, MechanismSpec("global_blend_025", "global_blend", 0.25, "blend 25%")),
    )
    add(
        "score_blend_075",
        "score",
        "25/75 blend of baseline and rank-tilted portfolio",
        _rank_targets(base, REVERSAL_20, MechanismSpec("global_blend_075", "global_blend", 0.75, "blend 75%")),
    )
    add(
        "score_low_vol_60",
        "score",
        "replace reversal with 60-day low-volatility rank",
        _rank_targets(base, FactorSpec("low_vol_60", "low_vol", 60, "negative 60-day vol")),
    )

    # round 2 mechanism: alternative drift formulas on top of calendar
    add(
        "mech_calendar_or_maxdrift_05",
        "mechanism",
        "add 5% max single-asset drift trigger on top of calendar",
        _rank_targets(base),
        trigger=TriggerSpec("calendar_or_maxdrift_05", "calendar_or_max_asset_drift", 0.05),
    )
    add(
        "mech_calendar_or_maxdrift_10",
        "mechanism",
        "add 10% max single-asset drift trigger on top of calendar",
        _rank_targets(base),
        trigger=TriggerSpec("calendar_or_maxdrift_10", "calendar_or_max_asset_drift", 0.10),
    )
    add(
        "mech_calendar_or_targetchange_05",
        "mechanism",
        "add 5% target-change trigger on top of calendar",
        _rank_targets(base),
        trigger=TriggerSpec("calendar_or_targetchange_05", "calendar_or_target_change", 0.05),
    )
    add(
        "mech_calendar_or_targetchange_10",
        "mechanism",
        "add 10% target-change trigger on top of calendar",
        _rank_targets(base),
        trigger=TriggerSpec("calendar_or_targetchange_10", "calendar_or_target_change", 0.10),
    )

    # round 2 pool: 8-ETF universe with budget-preserving sleeve tilt
    add(
        "pool_full_8etf_sleeve_tilt",
        "pool",
        "8-ETF universe + 50% rank tilt only inside dividend and sovereign sleeves",
        _rank_targets(
            dict(STATIC_BENCHMARK_TARGET),
            REVERSAL_20,
            MechanismSpec("sleeve_tilt_050", "sleeve_tilt", 0.50, "sleeve tilt 50%"),
        ),
    )

    # round 3: complete the blend gradient, refine the tilt peak, risk-only tilt
    add(
        "score_blend_100",
        "score",
        "pure rank-tilted portfolio without baseline anchoring",
        _rank_targets(base, REVERSAL_20, MechanismSpec("global_blend_100", "global_blend", 1.00, "blend 100%")),
    )
    add(
        "score_tilt_090",
        "score",
        "tilt strength 0.50 -> 0.90 (refine the 0.75/1.00 peak)",
        _rank_targets(base, REVERSAL_20, MechanismSpec("global_tilt_090", "global_tilt", 0.90, "tilt 90%")),
    )
    add(
        "score_risk_only_tilt_050",
        "score",
        "50% rank tilt among the 3 risk assets only; money sleeve fixed at 10%",
        _risk_only_tilt_builder(base, 0.50),
    )
    add(
        "score_risk_only_tilt_100",
        "score",
        "100% rank tilt among the 3 risk assets only; money sleeve fixed at 10%",
        _risk_only_tilt_builder(base, 1.00),
    )

    # combinations of the round-1..3 winners (no longer single-variable)
    add(
        "combo_risk_only_tilt_075",
        "combo",
        "stack: risk-only ranking universe + rank linear tilt strength 0.75; money fixed 10%",
        _risk_only_tilt_builder(base, 0.75),
    )
    add(
        "combo_risk_only_zsigmoid_075",
        "combo",
        "full stack: risk-only universe + z-score normalization + sigmoid tilt 0.75; money fixed 10%",
        _risk_only_zsigmoid_builder(base, 0.75),
    )

    # combo strength grids: 0.75 was inherited from the single-variable runs
    for strength in (0.25, 0.50, 1.00, 1.25, 1.50, 2.00):
        label = f"{int(round(strength * 100)):03d}"
        add(
            f"combo_risk_only_tilt_{label}",
            "combo",
            f"stack A: risk-only universe + rank linear tilt strength {strength:.2f}; money fixed 10%",
            _risk_only_tilt_builder(base, strength),
        )
        add(
            f"combo_risk_only_zsigmoid_{label}",
            "combo",
            f"stack B: risk-only universe + z-score sigmoid tilt strength {strength:.2f}; money fixed 10%",
            _risk_only_zsigmoid_builder(base, strength),
        )

    return specs


def run() -> pd.DataFrame:
    from .strategy import load_confirmed_market

    universe, _ = load_confirmed_market()
    all_assets = sorted(set(universe) | set(EXTRA_POOL_ASSETS))
    market = load_market_data(
        all_assets,
        pd.Timestamp("2013-01-01").date(),
        pd.Timestamp.today().date(),
    )

    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for spec in _experiments():
        daily_targets = spec["targets"](market)
        result = simulate_threshold_rebalance(
            market,
            daily_targets,
            spec["trigger"],
            initial_target=spec["initial_target"],
            cash_asset=CASH_ASSET,
            monthly_deposit=MONTHLY_DEPOSIT,
            cost_rate=COST_RATE,
            min_rebalance_notional=spec["min_notional"],
        )
        metrics = _extended_metrics(result)
        rows.append({
            "experiment": spec["name"],
            "direction": spec["direction"],
            "change": spec["change"],
            **metrics,
        })
        print(f"[done] {spec['name']}: sharpe={metrics['sharpe']:.3f} ann={metrics['annualized_return']:.2%}")

    frame = pd.DataFrame(rows)
    frame.to_csv(OUTPUT / "calendar_ablation_metrics.csv", index=False)
    (OUTPUT / "SUMMARY.md").write_text(_summary(frame), encoding="utf-8")
    return frame


def _summary(frame: pd.DataFrame) -> str:
    baseline = frame.loc[frame["experiment"] == "baseline_calendar"].iloc[0]
    candidates = frame.loc[
        (frame["experiment"] != "baseline_calendar")
        & (frame["sharpe"] > baseline["sharpe"])
        & (frame["annualized_return"] > baseline["annualized_return"])
    ].sort_values(["sharpe", "annualized_return"], ascending=False)

    ranked = frame.sort_values(["sharpe", "annualized_return"], ascending=False)

    def table(sub: pd.DataFrame) -> str:
        cols = [
            "experiment", "direction", "annualized_return", "volatility", "sharpe",
            "max_drawdown", "sortino", "calmar", "trades", "estimated_transaction_cost",
        ]
        display = sub.loc[:, cols].copy()
        for col in ("annualized_return", "volatility", "max_drawdown"):
            display[col] = display[col].map(lambda value: f"{value:.2%}")
        for col in ("sharpe", "sortino", "calmar"):
            display[col] = display[col].map(lambda value: f"{value:.2f}")
        display["estimated_transaction_cost"] = display["estimated_transaction_cost"].map(
            lambda value: f"{value:,.0f}"
        )
        return display.to_markdown(index=False)

    return f"""# 月初固定检查基线消融实验

基线：4 只防守 ETF（512890 35% / 511260 40% / 511360 15% / 511880 10%），20 日反转 + 全池 50% 排名倾斜，每月首个交易日收盘触发、次日开盘再平衡；入金 20,000 元按上月末目标只买；单边成本 0.05%；100 股整数手；再平衡单笔门槛 10,000 元。

基线结果：年化 {baseline['annualized_return']:.2%}，Sharpe {baseline['sharpe']:.3f}，最大回撤 {baseline['max_drawdown']:.2%}。

每个实验只改一个变量。全样本回测，存在参数选择与样本内过拟合风险，不构成样本外结论。

## Sharpe 与年化收益同时高于基线的方案

{table(candidates) if not candidates.empty else '无。'}

## 全部实验（按 Sharpe 排名）

{table(ranked)}
"""


if __name__ == "__main__":
    run()
