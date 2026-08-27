"""Exploratory factor tilts around the fixed defensive allocation baseline.

All factor values use only data available at the monthly signal close. The
module deliberately uses cross-sectional ranks because raw equity, bond, and
money-market factor magnitudes are not directly comparable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .engine import BacktestResult, MarketData, simulate_static_allocation
from .strategy import CASH_ASSET, STATIC_BENCHMARK_TARGET


EQUITY_ASSETS = ("510880.SH", "512890.SH", "515450.SH")
SOVEREIGN_ASSETS = ("511010.SH", "511260.SH", "511090.SH")
SLEEVE_GROUPS = (EQUITY_ASSETS, SOVEREIGN_ASSETS, ("511360.SH",), (CASH_ASSET,))


@dataclass(frozen=True)
class FactorSpec:
    name: str
    kind: str
    window: int
    description: str


@dataclass(frozen=True)
class MechanismSpec:
    name: str
    kind: str
    strength: float
    description: str


FACTOR_SPECS = (
    FactorSpec("momentum_20", "momentum", 20, "20日简单收益率"),
    FactorSpec("momentum_60", "momentum", 60, "60日简单收益率"),
    FactorSpec("momentum_120", "momentum", 120, "120日简单收益率"),
    FactorSpec("reversal_5", "reversal", 5, "负的5日简单收益率"),
    FactorSpec("reversal_20", "reversal", 20, "负的20日简单收益率"),
    FactorSpec("reversal_voladj_20", "reversal_vol_adjusted", 20, "负的20日收益率除以同期窗口波动"),
    FactorSpec("low_vol_20", "low_vol", 20, "负的20日日收益年化波动率"),
    FactorSpec("low_vol_60", "low_vol", 60, "负的60日日收益年化波动率"),
    FactorSpec("quality_momentum_20", "quality_momentum", 20, "20日动量×Kaufman ER"),
    FactorSpec("quality_momentum_60", "quality_momentum", 60, "60日动量×Kaufman ER"),
    FactorSpec("momentum_low_vol_60", "momentum_low_vol", 60, "60日动量排名与低波排名等权复合"),
)


MECHANISM_SPECS = (
    MechanismSpec("sleeve_tilt_050", "sleeve_tilt", 0.50, "保持四类预算，仅在红利和国债类内做50%排名倾斜"),
    MechanismSpec("sleeve_tilt_100", "sleeve_tilt", 1.00, "保持四类预算，仅在红利和国债类内做100%排名倾斜"),
    MechanismSpec("global_tilt_025", "global_tilt", 0.25, "基线权重乘以全池25%排名倾斜后归一化"),
    MechanismSpec("global_tilt_050", "global_tilt", 0.50, "基线权重乘以全池50%排名倾斜后归一化"),
    MechanismSpec("global_tilt_075", "global_tilt", 0.75, "基线权重乘以全池75%排名倾斜后归一化"),
    MechanismSpec("global_blend_025", "global_blend", 0.25, "基线与全池排名组合按75%/25%混合"),
    MechanismSpec("global_blend_050", "global_blend", 0.50, "基线与全池排名组合按50%/50%混合"),
)


def _history(data: MarketData, asset: str, timestamp: pd.Timestamp) -> pd.Series:
    close = data.closes[asset]
    end = int(close.index.searchsorted(timestamp, side="right"))
    return close.iloc[:end].dropna().astype(float)


def _raw_factor(history: pd.Series, kind: str, window: int) -> float:
    if len(history) < window + 1:
        return np.nan
    latest = float(history.iloc[-1])
    lagged = float(history.iloc[-1 - window])
    momentum = latest / lagged - 1.0
    if kind == "momentum":
        return momentum
    if kind == "reversal":
        return -momentum
    returns = history.pct_change().dropna().tail(window)
    if kind == "low_vol":
        return -float(returns.std(ddof=1) * np.sqrt(252.0))
    if kind == "reversal_vol_adjusted":
        volatility = float(returns.std(ddof=1) * np.sqrt(252.0))
        window_vol = volatility * float(np.sqrt(window / 252.0))
        return -momentum / window_vol if window_vol > 0 else np.nan
    path = float(history.diff().abs().tail(window).sum())
    er = abs(latest - lagged) / path if path > 0 else np.nan
    if kind == "quality_momentum":
        return momentum * er
    raise ValueError(f"unsupported factor kind: {kind}")


def _rank_component(values: dict[str, float], assets: tuple[str, ...]) -> dict[str, float]:
    valid = {asset: value for asset, value in values.items() if np.isfinite(value)}
    if len(valid) < 2:
        return {asset: 0.0 for asset in assets}
    ranks = pd.Series(valid).rank(method="average")
    centered = ranks - ranks.mean()
    scale = float(centered.abs().max())
    normalized = centered / scale if scale > 0 else centered * 0.0
    return {asset: float(normalized.get(asset, 0.0)) for asset in assets}


def factor_ranks(
    data: MarketData,
    timestamp: pd.Timestamp,
    factor: FactorSpec,
    assets: tuple[str, ...],
) -> dict[str, float]:
    if factor.kind == "momentum_low_vol":
        momentum = {
            asset: _raw_factor(_history(data, asset, timestamp), "momentum", factor.window)
            for asset in assets
        }
        low_vol = {
            asset: _raw_factor(_history(data, asset, timestamp), "low_vol", factor.window)
            for asset in assets
        }
        mom_rank = _rank_component(momentum, assets)
        vol_rank = _rank_component(low_vol, assets)
        return {asset: (mom_rank[asset] + vol_rank[asset]) / 2.0 for asset in assets}

    raw = {
        asset: _raw_factor(_history(data, asset, timestamp), factor.kind, factor.window)
        for asset in assets
    }
    return _rank_component(raw, assets)


def factor_zscores(
    data: MarketData,
    timestamp: pd.Timestamp,
    factor: FactorSpec,
    assets: tuple[str, ...],
) -> dict[str, float]:
    """Cross-sectionally standardized factor values (mean 0, std 1 per day).

    Preserves magnitude differences between assets, unlike factor_ranks.
    Composite rank-based factors (e.g. momentum_low_vol) are unsupported.
    """
    raw = {
        asset: _raw_factor(_history(data, asset, timestamp), factor.kind, factor.window)
        for asset in assets
    }
    valid = {asset: value for asset, value in raw.items() if np.isfinite(value)}
    if len(valid) < 2:
        return {asset: 0.0 for asset in assets}
    series = pd.Series(valid)
    mean = float(series.mean())
    std = float(series.std(ddof=0))
    if std <= 0:
        return {asset: 0.0 for asset in assets}
    return {
        asset: float((raw[asset] - mean) / std) if np.isfinite(raw[asset]) else 0.0
        for asset in assets
    }


def adjusted_weights(
    scores: dict[str, float],
    mechanism: MechanismSpec,
    baseline: dict[str, float] | None = None,
) -> dict[str, float]:
    base = dict(baseline or STATIC_BENCHMARK_TARGET)
    ranks = scores
    if mechanism.kind == "global_tilt":
        weights = {
            asset: weight * (1.0 + mechanism.strength * ranks.get(asset, 0.0))
            for asset, weight in base.items()
        }
    elif mechanism.kind == "exp_tilt":
        weights = {
            asset: weight * float(np.exp(mechanism.strength * scores.get(asset, 0.0)))
            for asset, weight in base.items()
        }
    elif mechanism.kind == "sigmoid_tilt":
        weights = {
            asset: weight * (1.0 + mechanism.strength * float(np.tanh(scores.get(asset, 0.0))))
            for asset, weight in base.items()
        }
    elif mechanism.kind == "additive_tilt":
        weights = {
            asset: max(0.0, weight + mechanism.strength * scores.get(asset, 0.0))
            for asset, weight in base.items()
        }
    elif mechanism.kind == "global_blend":
        rank_raw = {asset: max(0.0, 1.0 + ranks.get(asset, 0.0)) for asset in base}
        rank_total = sum(rank_raw.values())
        rank_portfolio = {asset: value / rank_total for asset, value in rank_raw.items()}
        weights = {
            asset: (1.0 - mechanism.strength) * base[asset]
            + mechanism.strength * rank_portfolio[asset]
            for asset in base
        }
    elif mechanism.kind == "sleeve_tilt":
        weights = dict(base)
        for group in SLEEVE_GROUPS:
            budget = sum(base[asset] for asset in group)
            if len(group) == 1:
                weights[group[0]] = budget
                continue
            raw = {
                asset: base[asset] * (1.0 + mechanism.strength * ranks.get(asset, 0.0))
                for asset in group
            }
            raw_total = sum(raw.values())
            for asset in group:
                weights[asset] = budget * raw[asset] / raw_total
    else:
        raise ValueError(f"unsupported mechanism kind: {mechanism.kind}")

    # Clip negative masses before normalizing; strong tilts (strength > 1) can
    # turn a low-ranked asset's raw weight negative, and the strategy is
    # long-only. Clipping after normalization would break the sum-to-one check.
    clipped = {asset: max(0.0, value) for asset, value in weights.items()}
    total = sum(clipped.values())
    if total <= 0:
        raise RuntimeError("adjusted weights have no positive mass")
    normalized = {asset: value / total for asset, value in clipped.items()}
    if not np.isclose(sum(normalized.values()), 1.0):
        raise RuntimeError("adjusted weights do not sum to one")
    return normalized


def monthly_signal_dates(data: MarketData) -> list[pd.Timestamp]:
    dates: dict[tuple[int, int], pd.Timestamp] = {}
    for timestamp in data.dates:
        dates.setdefault((timestamp.year, timestamp.month), timestamp)
    return list(dates.values())


def build_target_schedule(
    data: MarketData,
    factor: FactorSpec,
    mechanism: MechanismSpec,
) -> dict[pd.Timestamp, dict[str, float]]:
    assets = tuple(STATIC_BENCHMARK_TARGET)
    schedule: dict[pd.Timestamp, dict[str, float]] = {}
    for timestamp in monthly_signal_dates(data):
        ranks = factor_ranks(data, timestamp, factor, assets)
        schedule[timestamp] = adjusted_weights(ranks, mechanism)
    return schedule


def simulate_factor_allocation(
    data: MarketData,
    factor: FactorSpec,
    mechanism: MechanismSpec,
) -> tuple[BacktestResult, dict[pd.Timestamp, dict[str, float]]]:
    schedule = build_target_schedule(data, factor, mechanism)
    result = simulate_static_allocation(
        data,
        STATIC_BENCHMARK_TARGET,
        cash_asset=CASH_ASSET,
        target_schedule=schedule,
    )
    return result, schedule
