"""Occam peak-escape overlay for the formal W40 Momentum/Defender path."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from data.store import query, read_storage
from research.defender_curve_momentum import DEFENDER_CANDIDATE
from research.generate_strategy_drawdown_badcases import (
    distinct_drawdown_episodes,
)
from research.momentum_defender_gold_override import (
    GoldOverrideContext,
    simulate_candidate_schedule,
)
from research.momentum_defender_occam import MOMENTUM_ASSETS, performance
from research.momentum_volatility import asof_previous_close


FUND_SHARE_CACHE = Path("data/db/four_etf_tushare_fields.parquet")
SUPPORTED_POLICIES = {"price_volume", "price_crowding"}


@dataclass(frozen=True)
class PeakEscapeParams:
    policy: str
    price_return_threshold: float
    volume_ratio_threshold: float
    fund_share_flow_threshold: float
    min_escape_hold_days: int

    def __post_init__(self) -> None:
        if self.policy not in SUPPORTED_POLICIES:
            raise ValueError(f"unsupported peak escape policy: {self.policy}")
        if not 0.0 <= self.price_return_threshold <= 1.0:
            raise ValueError("price return threshold must lie in [0, 1]")
        if self.volume_ratio_threshold <= 0.0:
            raise ValueError("volume ratio threshold must be positive")
        if not 0.0 <= self.fund_share_flow_threshold <= 1.0:
            raise ValueError("fund share flow threshold must lie in [0, 1]")
        if self.min_escape_hold_days < 1:
            raise ValueError("minimum escape hold must be positive")

    def candidate_id(self) -> str:
        policy = "pv" if self.policy == "price_volume" else "pc"
        return (
            f"peak_escape_{policy}_r{self.price_return_threshold:.2f}_"
            f"v{self.volume_ratio_threshold:.2f}_"
            f"s{self.fund_share_flow_threshold:.2f}_"
            f"h{self.min_escape_hold_days}"
        )


@dataclass(frozen=True)
class PeakEscapeFeatures:
    calendar: pd.DatetimeIndex
    price_breakout_at_open: pd.DataFrame
    price_return20_at_open: pd.DataFrame
    volume_ratio20_at_open: pd.DataFrame
    adjusted_share_flow20_at_open: pd.DataFrame
    coverage: pd.DataFrame


@dataclass(frozen=True)
class PeakEscapeBacktest:
    params: PeakEscapeParams
    state: pd.DataFrame
    daily: pd.DataFrame
    audit: Mapping[str, object]


def _load_adjusted_fund_share(
    root: Path,
    asset: str,
    price_index: pd.DatetimeIndex,
    *,
    cache_path: Path,
) -> pd.Series:
    path = cache_path if cache_path.is_absolute() else root / cache_path
    if not path.exists():
        raise RuntimeError(f"missing point-in-time fund-share cache: {path}")
    extra = pd.read_parquet(path)
    required = {"ts_code", "date", "fd_share"}
    missing = required - set(extra.columns)
    if missing:
        raise ValueError(f"fund-share cache missing columns: {sorted(missing)}")
    extra = extra.loc[extra["ts_code"].astype(str).eq(asset)].copy()
    extra["date"] = pd.to_datetime(extra["date"])
    raw_share = (
        pd.to_numeric(extra["fd_share"], errors="coerce")
        .set_axis(extra["date"])
        .sort_index()
    )
    raw_share = raw_share.loc[~raw_share.index.duplicated(keep="last")]
    raw_share = raw_share.reindex(price_index).ffill()

    storage = read_storage(asset)
    if storage is None or storage.empty or "adj_factor" not in storage.columns:
        raise RuntimeError(f"missing adjustment factors for {asset}")
    factors = storage[["date", "adj_factor"]].copy()
    factors["date"] = pd.to_datetime(factors["date"])
    adjustment = (
        pd.to_numeric(factors["adj_factor"], errors="coerce")
        .set_axis(factors["date"])
        .sort_index()
    )
    adjustment = adjustment.loc[~adjustment.index.duplicated(keep="last")]
    adjustment = adjustment.reindex(price_index).ffill()
    adjusted = raw_share / adjustment.replace(0.0, np.nan)
    adjusted.name = asset
    return adjusted


def build_peak_escape_features(
    root: Path,
    calendar: pd.DatetimeIndex,
    *,
    end: date,
    cache_path: Path = FUND_SHARE_CACHE,
) -> PeakEscapeFeatures:
    """Build asset-local close signals and expose them at the next open."""

    panels = {
        name: pd.DataFrame(index=calendar, columns=MOMENTUM_ASSETS, dtype=float)
        for name in ("breakout", "return20", "volume_ratio20", "share_flow20")
    }
    coverage_rows = []
    for asset in MOMENTUM_ASSETS:
        price = (
            query(asset, date(2013, 1, 1), end)
            .sort_values("date")
            .drop_duplicates("date")
            .set_index("date")
        )
        if price.empty:
            raise RuntimeError(f"missing price history for {asset}")
        close = price["close"].astype(float)
        volume = price["volume"].astype(float)
        prior_high200 = close.shift(1).rolling(200, min_periods=200).max()
        breakout = close / prior_high200 - 1.0
        return20 = close.pct_change(20, fill_method=None)
        volume_ratio20 = volume / (
            volume.shift(1).rolling(20, min_periods=20).median()
        )
        adjusted_share = _load_adjusted_fund_share(
            root,
            asset,
            pd.DatetimeIndex(price.index),
            cache_path=cache_path,
        )
        share_flow20 = adjusted_share.pct_change(20, fill_method=None)
        close_values = {
            "breakout": breakout,
            "return20": return20,
            "volume_ratio20": volume_ratio20,
            "share_flow20": share_flow20,
        }
        for name, values in close_values.items():
            panels[name][asset] = asof_previous_close(values, calendar)
        evaluation = pd.DataFrame(close_values).loc[
            pd.Timestamp(calendar.min()) : pd.Timestamp(calendar.max())
        ]
        coverage_rows.append(
            {
                "asset": asset,
                "observations": int(len(evaluation)),
                **{
                    f"{name}_coverage": float(values.notna().mean())
                    for name, values in evaluation.items()
                },
                "share_first_valid": share_flow20.first_valid_index(),
                "share_last_valid": share_flow20.last_valid_index(),
                "largest_abs_adjusted_share_change": float(
                    adjusted_share.pct_change(fill_method=None).abs().max()
                ),
            }
        )
    return PeakEscapeFeatures(
        calendar=calendar,
        price_breakout_at_open=panels["breakout"],
        price_return20_at_open=panels["return20"],
        volume_ratio20_at_open=panels["volume_ratio20"],
        adjusted_share_flow20_at_open=panels["share_flow20"],
        coverage=pd.DataFrame(coverage_rows).set_index("asset"),
    )


def _entry_evidence(
    features: PeakEscapeFeatures,
    timestamp: pd.Timestamp,
    asset: str,
    params: PeakEscapeParams,
) -> dict[str, object]:
    breakout = features.price_breakout_at_open.at[timestamp, asset]
    return20 = features.price_return20_at_open.at[timestamp, asset]
    volume_ratio = features.volume_ratio20_at_open.at[timestamp, asset]
    share_flow = features.adjusted_share_flow20_at_open.at[timestamp, asset]
    price_flag = bool(
        pd.notna(breakout)
        and pd.notna(return20)
        and float(breakout) > 0.0
        and float(return20) >= params.price_return_threshold
    )
    volume_flag = bool(
        pd.notna(volume_ratio)
        and float(volume_ratio) >= params.volume_ratio_threshold
    )
    scale_flag = bool(
        pd.notna(share_flow)
        and float(share_flow) >= params.fund_share_flow_threshold
    )
    qualified = (
        price_flag and volume_flag
        if params.policy == "price_volume"
        else price_flag and (volume_flag or scale_flag)
    )
    return {
        "price_breakout_at_open": breakout,
        "price_return20_at_open": return20,
        "volume_ratio20_at_open": volume_ratio,
        "adjusted_share_flow20_at_open": share_flow,
        "price_flag": price_flag,
        "volume_flag": volume_flag,
        "scale_flag": scale_flag,
        "entry_qualified": qualified,
    }


def peak_escape_schedule(
    baseline_requested: pd.Series,
    features: PeakEscapeFeatures,
    params: PeakEscapeParams,
) -> pd.DataFrame:
    """Replace an overheated formal Momentum target with Defender."""

    active = False
    held_days = 10**9
    rows = []
    for timestamp in features.calendar:
        baseline = str(baseline_requested.at[timestamp])
        previous_active = active
        reason = "hold"
        if baseline == DEFENDER_CANDIDATE:
            evidence = {
                "price_breakout_at_open": np.nan,
                "price_return20_at_open": np.nan,
                "volume_ratio20_at_open": np.nan,
                "adjusted_share_flow20_at_open": np.nan,
                "price_flag": False,
                "volume_flag": False,
                "scale_flag": False,
                "entry_qualified": False,
            }
            if active:
                reason = "formal_returned_to_defender"
            active = False
            held_days = 0
            target = DEFENDER_CANDIDATE
        else:
            if baseline not in MOMENTUM_ASSETS:
                raise AssertionError(f"unknown formal baseline candidate: {baseline}")
            evidence = _entry_evidence(features, timestamp, baseline, params)
            qualified = bool(evidence["entry_qualified"])
            if not active and qualified:
                active = True
                held_days = 0
                reason = "peak_escape_entry"
            elif active and held_days >= params.min_escape_hold_days and not qualified:
                active = False
                held_days = 0
                reason = "peak_escape_exit"
            target = DEFENDER_CANDIDATE if active else baseline
        rows.append(
            {
                "date": timestamp,
                "formal_requested_candidate": baseline,
                "peak_escape_active": active,
                "peak_escape_changed": active != previous_active,
                "escape_held_days_at_open": held_days,
                "state_reason": reason,
                "target_candidate": target,
                **evidence,
            }
        )
        if active:
            held_days += 1
    return pd.DataFrame(rows).set_index("date")


def run_peak_escape(
    context: GoldOverrideContext,
    baseline_requested: pd.Series,
    features: PeakEscapeFeatures,
    params: PeakEscapeParams,
) -> PeakEscapeBacktest:
    state = peak_escape_schedule(baseline_requested, features, params)
    daily = simulate_candidate_schedule(
        state["target_candidate"],
        context.interfaces,
        context.initial_previous_candidate,
    )
    entries = state["state_reason"].eq("peak_escape_entry")
    invalid_entries = int((~state.loc[entries, "entry_qualified"].astype(bool)).sum())
    nav_error = float(
        ((1.0 + daily["return"]).cumprod() - daily["nav"]).abs().max()
    )
    if invalid_entries or nav_error > 1e-12:
        raise AssertionError(
            f"peak escape audit failed: entries={invalid_entries}, nav={nav_error:.3e}"
        )
    audit = {
        "status": "passed",
        "candidate_id": params.candidate_id(),
        "params": asdict(params),
        "invalid_entries": invalid_entries,
        "nav_reconstruction_max_abs_error": nav_error,
        "escape_entries": int(entries.sum()),
        "escape_days": int(state["peak_escape_active"].sum()),
        "candidate_switches": int(daily["switched"].sum()),
        "entry_price_and_volume_count": int(
            (entries & state["price_flag"] & state["volume_flag"]).sum()
        ),
        "entry_scale_without_volume_count": int(
            (entries & state["scale_flag"] & ~state["volume_flag"]).sum()
        ),
        "performance": performance(daily["return"].astype(float)),
    }
    return PeakEscapeBacktest(params=params, state=state, daily=daily, audit=audit)


def collect_peak_escape_returns(
    context: GoldOverrideContext,
    baseline_requested: pd.Series,
    features: PeakEscapeFeatures,
    grid: Mapping[str, Iterable[object]],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, PeakEscapeBacktest]]:
    records = []
    returns = {}
    runs = {}
    for policy in grid["policy"]:
        for price_threshold in grid["price_return_threshold"]:
            for volume_threshold in grid["volume_ratio_threshold"]:
                for share_threshold in grid["fund_share_flow_threshold"]:
                    for hold_days in grid["min_escape_hold_days"]:
                        params = PeakEscapeParams(
                            policy=str(policy),
                            price_return_threshold=float(price_threshold),
                            volume_ratio_threshold=float(volume_threshold),
                            fund_share_flow_threshold=float(share_threshold),
                            min_escape_hold_days=int(hold_days),
                        )
                        run = run_peak_escape(
                            context, baseline_requested, features, params
                        )
                        candidate_id = params.candidate_id()
                        records.append(
                            {
                                "candidate_id": candidate_id,
                                **asdict(params),
                                **{
                                    key: value
                                    for key, value in run.audit.items()
                                    if key
                                    in {
                                        "escape_entries",
                                        "escape_days",
                                        "candidate_switches",
                                        "entry_price_and_volume_count",
                                        "entry_scale_without_volume_count",
                                    }
                                },
                            }
                        )
                        returns[candidate_id] = run.daily["return"].astype(float)
                        runs[candidate_id] = run
    metadata = pd.DataFrame(records).set_index("candidate_id")
    matrix = pd.DataFrame(returns, index=features.calendar)
    return metadata, matrix, runs


def top_drawdown_summary(
    returns: pd.Series,
    *,
    top_n: int,
) -> tuple[dict[str, float | int], pd.DataFrame]:
    values = returns.astype(float)
    daily = pd.DataFrame(
        {
            "return": values,
            "nav": (1.0 + values).cumprod(),
        },
        index=values.index,
    )
    episodes = distinct_drawdown_episodes(daily, top_n=top_n)
    if episodes.empty:
        return (
            {
                "top_drawdown_count": 0,
                "top_mean_drawdown": 0.0,
                "top_worst_drawdown": 0.0,
                "top_mean_underwater_sessions": 0.0,
            },
            episodes,
        )
    return (
        {
            "top_drawdown_count": int(len(episodes)),
            "top_mean_drawdown": float(episodes["max_drawdown"].mean()),
            "top_worst_drawdown": float(episodes["max_drawdown"].min()),
            "top_mean_underwater_sessions": float(
                episodes["underwater_sessions"].mean()
            ),
        },
        episodes,
    )


def same_window_top_drawdowns(
    candidate: pd.Series,
    baseline_episodes: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for _, episode in baseline_episodes.iterrows():
        start = pd.Timestamp(episode["decline_start"])
        trough = pd.Timestamp(episode["trough_date"])
        candidate_return = float(
            (1.0 + candidate.loc[start:trough].astype(float)).prod() - 1.0
        )
        baseline_return = float(episode["max_drawdown"])
        rows.append(
            {
                "baseline_rank": int(episode["rank"]),
                "start": start,
                "trough": trough,
                "baseline_return": baseline_return,
                "candidate_return": candidate_return,
                "improvement": candidate_return - baseline_return,
                "candidate_improved": candidate_return > baseline_return,
            }
        )
    return pd.DataFrame(rows)
