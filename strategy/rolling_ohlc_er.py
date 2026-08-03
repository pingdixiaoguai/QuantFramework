"""Research-only quarterly rolling four-path OHLC efficiency-ratio search.

This module deliberately stays outside the production order path.  The
research command uses it to:

1. rolls the OHLC path weights forward at calendar-quarter boundaries;
2. computes old close-only ER and new rolling OHLC ER diagnostics;
3. provide reproducible diagnostics for a reviewed YAML checkpoint.

The production and notification paths do not import this module. They consume
the registered ``ohlc_quality_momentum`` factor through a normal strategy YAML.
"""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

_DEFAULT_STATE_DIR = Path(__file__).resolve().parent.parent / "state"
_WEIGHT_KEYS = ("close", "gap", "body", "range")


@dataclass(frozen=True)
class RollingWeights:
    """Effective OHLC path weights and the data window used to select them."""

    effective_date: date
    training_start: date
    training_end: date
    values: tuple[float, float, float, float]


@dataclass(frozen=True)
class SignalComparison:
    """Old/new per-asset diagnostics for one signal date."""

    weights: RollingWeights
    old_target: str
    new_target: str
    assets: dict[str, dict[str, float]]


def _weights_from_mapping(raw: dict) -> tuple[float, float, float, float]:
    values = tuple(float(raw[key]) for key in _WEIGHT_KEYS)
    if len(values) != 4:
        raise ValueError("OHLC ER requires exactly four path weights")
    if min(values) < 0:
        raise ValueError(f"OHLC ER weights must be non-negative, got {values}")
    if values[0] + values[1] < 1:
        raise ValueError("close + gap weights must be >= 1")
    if values[0] + values[2] + values[3] < 1:
        raise ValueError("close + body + range weights must be >= 1")
    return values


def _state_path(strategy_name: str, state_dir: Path | None = None) -> Path:
    root = state_dir or _DEFAULT_STATE_DIR
    return root / f"{strategy_name}_rolling_ohlc_er.json"


def _quarter_id(value: date | pd.Timestamp) -> tuple[int, int]:
    stamp = pd.Timestamp(value)
    return stamp.year, stamp.quarter


def _read_cached_weights(
    strategy_name: str,
    state_dir: Path | None = None,
) -> RollingWeights | None:
    path = _state_path(strategy_name, state_dir)
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as handle:
        raw = json.load(handle)
    return RollingWeights(
        effective_date=date.fromisoformat(raw["effective_date"]),
        training_start=date.fromisoformat(raw["training_start"]),
        training_end=date.fromisoformat(raw["training_end"]),
        values=_weights_from_mapping(raw["weights"]),
    )


def _write_cached_weights(
    strategy_name: str,
    state: RollingWeights,
    state_dir: Path | None = None,
) -> None:
    path = _state_path(strategy_name, state_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "effective_date": state.effective_date.isoformat(),
        "training_start": state.training_start.isoformat(),
        "training_end": state.training_end.isoformat(),
        "weights": dict(zip(_WEIGHT_KEYS, state.values, strict=True)),
    }
    temporary = path.with_suffix(".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    temporary.replace(path)


def _seed_weights(config: dict) -> RollingWeights:
    seed = config["seed"]
    effective_date = date.fromisoformat(str(seed["effective_date"]))
    training_start = date.fromisoformat(str(seed["training_start"]))
    training_end = date.fromisoformat(str(seed["training_end"]))
    return RollingWeights(
        effective_date=effective_date,
        training_start=training_start,
        training_end=training_end,
        values=_weights_from_mapping(seed["weights"]),
    )


def _prepare_arrays(
    asset_data: dict[str, pd.DataFrame],
    assets: list[str],
    window: int,
) -> tuple[
    pd.DatetimeIndex,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Align assets on the union calendar, matching the research backtest."""
    missing = [asset for asset in assets if asset not in asset_data]
    if missing:
        raise ValueError("missing rolling OHLC ER data for: " + ", ".join(missing))

    frames = {
        asset: asset_data[asset].sort_values("date").set_index("date")
        for asset in assets
    }
    index = pd.DatetimeIndex(
        sorted(set().union(*(set(frame.index) for frame in frames.values())))
    )
    total_days = len(index)
    asset_count = len(assets)
    opens = np.full((total_days, asset_count), np.nan)
    closes = np.full((total_days, asset_count), np.nan)
    momentum = np.full((total_days, asset_count), np.nan)
    displacement = np.full((total_days, asset_count), np.nan)
    path_sums = np.full((4, total_days, asset_count), np.nan)

    for asset_index, asset in enumerate(assets):
        frame = frames[asset]
        close = frame["close"].astype(float)
        open_price = frame["open"].astype(float)
        high = frame["high"].astype(float)
        low = frame["low"].astype(float)
        components = (
            close.diff().abs(),
            (open_price - close.shift(1)).abs(),
            (close - open_price).abs(),
            high - low,
        )
        opens[:, asset_index] = open_price.reindex(index).to_numpy(float)
        closes[:, asset_index] = close.reindex(index).to_numpy(float)
        momentum[:, asset_index] = (
            close.pct_change(window, fill_method=None).reindex(index).to_numpy(float)
        )
        displacement[:, asset_index] = (
            (close - close.shift(window)).abs().reindex(index).to_numpy(float)
        )
        for path_index, component in enumerate(components):
            path_sums[path_index, :, asset_index] = (
                component.rolling(window).sum().reindex(index).to_numpy(float)
            )

    return index, opens, closes, momentum, displacement, path_sums


def _candidate_weights(
    center: np.ndarray,
    radius: float,
    step: float,
) -> np.ndarray:
    offsets = np.round(
        np.arange(-radius, radius + step / 2, step),
        6,
    )
    offset_grid = np.asarray(
        list(itertools.product(offsets, repeat=4)),
        dtype=float,
    )
    weights = center[None, :] + offset_grid
    valid = (
        (weights.min(axis=1) >= 0)
        & (weights[:, 0] + weights[:, 1] >= 1)
        & (weights[:, 0] + weights[:, 2] + weights[:, 3] >= 1)
    )
    return weights[valid]


def _simulate_candidate_returns(
    scores: np.ndarray,
    opens: np.ndarray,
    closes: np.ndarray,
    min_hold_days: int,
    transaction_cost_rate: float,
) -> np.ndarray:
    """Vectorized Top1/open-execution simulation for a candidate batch."""
    candidate_count, day_count, asset_count = scores.shape
    returns = np.full((candidate_count, day_count), np.nan)
    current = np.full(candidate_count, -1, np.int16)
    pending = np.full(candidate_count, -1, np.int16)
    pending_day = np.full(candidate_count, -1, np.int32)
    entry_day = np.full(candidate_count, -1, np.int32)

    for day_index in range(day_count):
        if day_index:
            opened = (pending >= 0) & (pending_day == day_index)
            if opened.any():
                old = current.copy()
                new = pending.copy()
                has_old = old >= 0
                old_index = np.clip(old, 0, asset_count - 1)
                new_index = np.clip(new, 0, asset_count - 1)
                growth = np.ones(candidate_count)
                if has_old.any():
                    growth[has_old] = (
                        opens[day_index, old_index[has_old]]
                        / closes[day_index - 1, old_index[has_old]]
                    )
                growth[opened] *= (
                    closes[day_index, new_index][opened]
                    / opens[day_index, new_index][opened]
                )
                costs = np.where(
                    has_old,
                    2 * transaction_cost_rate,
                    transaction_cost_rate,
                )
                returns[opened, day_index] = growth[opened] - 1 - costs[opened]
                current[opened] = new[opened]
                entry_day[opened] = day_index
                pending[opened] = -1
                pending_day[opened] = -1

            carry = (current >= 0) & ~opened
            if carry.any():
                current_index = np.clip(current, 0, asset_count - 1)
                returns[carry, day_index] = (
                    closes[day_index, current_index[carry]]
                    / closes[day_index - 1, current_index[carry]]
                    - 1
                )

        holding_days = np.where(
            current >= 0,
            day_index - entry_day + 1,
            0,
        )
        can_signal = (pending < 0) & (
            (current < 0) | (holding_days >= min_hold_days)
        )
        if can_signal.any() and day_index + 1 < day_count:
            row = scores[:, day_index, :].copy()
            finite = np.isfinite(row)
            row[~finite] = -np.inf
            target = np.argmax(row, axis=1)
            change = can_signal & finite.any(axis=1) & (target != current)
            pending[change] = target[change].astype(np.int16)
            pending_day[change] = day_index + 1

    return returns


def _annualized_sharpe(returns: np.ndarray) -> np.ndarray:
    finite = np.isfinite(returns)
    counts = finite.sum(axis=1)
    safe = np.where(finite, returns, 0.0)
    means = safe.sum(axis=1) / np.maximum(counts, 1)
    centered = np.where(finite, returns - means[:, None], np.nan)
    standard_deviations = np.nanstd(centered, axis=1, ddof=1)
    return np.divide(
        means,
        standard_deviations,
        out=np.full(len(returns), -np.inf),
        where=standard_deviations > 0,
    ) * np.sqrt(252)


def _select_top_k_mean(
    center: tuple[float, float, float, float],
    update_index: int,
    opens: np.ndarray,
    closes: np.ndarray,
    momentum: np.ndarray,
    displacement: np.ndarray,
    path_sums: np.ndarray,
    config: dict,
) -> tuple[tuple[float, float, float, float], float]:
    history_days = int(config["history_days"])
    start_index = update_index - history_days
    if start_index < 0:
        raise ValueError("not enough history for rolling OHLC ER update")

    weights = _candidate_weights(
        np.asarray(center, dtype=float),
        float(config["search_radius"]),
        float(config["search_step"]),
    )
    sharpes = np.full(len(weights), -np.inf)
    batch_size = int(config.get("optimizer_batch_size", 512))

    for batch_start in range(0, len(weights), batch_size):
        batch_end = min(batch_start + batch_size, len(weights))
        candidate_batch = weights[batch_start:batch_end]
        denominators = np.einsum(
            "cf,fta->cta",
            candidate_batch,
            path_sums[:, start_index:update_index],
            optimize=True,
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            scores = momentum[start_index:update_index][None, :, :] * (
                displacement[start_index:update_index][None, :, :]
                / denominators
            )
        candidate_returns = _simulate_candidate_returns(
            scores,
            opens[start_index:update_index],
            closes[start_index:update_index],
            int(config["min_hold_days"]),
            float(config["transaction_cost_rate"]),
        )
        sharpes[batch_start:batch_end] = _annualized_sharpe(candidate_returns)

    top_k = int(config["top_k"])
    if top_k < 1 or top_k > len(weights):
        raise ValueError(f"invalid top_k={top_k} for {len(weights)} candidates")
    order = np.argsort(-sharpes)
    selected = weights[order[:top_k]]
    mean_weights = tuple(
        float(value) for value in np.round(selected.mean(axis=0), 12)
    )
    return mean_weights, float(sharpes[order[0]])


def resolve_current_weights(
    asset_data: dict[str, pd.DataFrame],
    assets: list[str],
    signal_date: date,
    strategy_name: str,
    config: dict,
    state_dir: Path | None = None,
) -> RollingWeights:
    """Load the cached quarter or roll it forward without future information."""
    seed = _seed_weights(config)
    cached = _read_cached_weights(strategy_name, state_dir)
    if _quarter_id(signal_date) < _quarter_id(seed.effective_date):
        raise ValueError(
            f"signal date {signal_date} predates rolling OHLC ER seed "
            f"{seed.effective_date}"
        )
    state = (
        cached
        if cached is not None
        and _quarter_id(cached.effective_date) >= _quarter_id(seed.effective_date)
        and _quarter_id(cached.effective_date) <= _quarter_id(signal_date)
        else seed
    )
    if _quarter_id(signal_date) <= _quarter_id(state.effective_date):
        return state

    window = int(config["window"])
    (
        index,
        opens,
        closes,
        momentum,
        displacement,
        path_sums,
    ) = _prepare_arrays(asset_data, assets, window)
    periods = index.to_period("Q")
    first_of_quarter = np.r_[True, periods[1:] != periods[:-1]]

    for update_index in np.where(first_of_quarter)[0]:
        update_date = index[update_index].date()
        if _quarter_id(update_date) <= _quarter_id(state.effective_date):
            continue
        if _quarter_id(update_date) > _quarter_id(signal_date):
            break
        new_values, _ = _select_top_k_mean(
            state.values,
            update_index,
            opens,
            closes,
            momentum,
            displacement,
            path_sums,
            config,
        )
        history_days = int(config["history_days"])
        state = RollingWeights(
            effective_date=update_date,
            training_start=index[update_index - history_days].date(),
            training_end=index[update_index - 1].date(),
            values=new_values,
        )

    _write_cached_weights(strategy_name, state, state_dir)
    return state


def _latest_components(
    frame: pd.DataFrame,
    window: int,
    new_weights: tuple[float, float, float, float],
) -> dict[str, float]:
    ordered = frame.sort_values("date")
    close = ordered["close"].astype(float)
    open_price = ordered["open"].astype(float)
    high = ordered["high"].astype(float)
    low = ordered["low"].astype(float)
    momentum = close.pct_change(window, fill_method=None)
    displacement = (close - close.shift(window)).abs()
    old_path = close.diff().abs().rolling(window).sum()
    daily_new_path = (
        new_weights[0] * close.diff().abs()
        + new_weights[1] * (open_price - close.shift(1)).abs()
        + new_weights[2] * (close - open_price).abs()
        + new_weights[3] * (high - low)
    )
    new_path = daily_new_path.rolling(window).sum()
    old_er = displacement / old_path.replace(0, np.nan)
    new_er = displacement / new_path.replace(0, np.nan)
    return {
        "momentum": float(momentum.iloc[-1]),
        "old_er": float(old_er.iloc[-1]),
        "old_score": float((momentum * old_er).iloc[-1]),
        "new_er": float(new_er.iloc[-1]),
        "new_score": float((momentum * new_er).iloc[-1]),
    }


def _softmax_confidence(scores: dict[str, float]) -> dict[str, float]:
    """Scale-free cross-sectional softmax probabilities.

    The cross-sectional population standard deviation is used as temperature.
    This preserves ranking while avoiding the near-25% output produced by
    applying softmax directly to small return-like factor scores.
    """
    assets = list(scores)
    values = np.asarray([scores[asset] for asset in assets], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("softmax scores must all be finite")
    scale = float(np.std(values, ddof=0))
    if scale <= 1e-12:
        probabilities = np.full(len(values), 1.0 / len(values))
    else:
        logits = values / scale
        exponentials = np.exp(logits - logits.max())
        probabilities = exponentials / exponentials.sum()
    return dict(zip(assets, probabilities, strict=True))


def build_signal_comparison(
    asset_data: dict[str, pd.DataFrame],
    assets: list[str],
    signal_date: date,
    strategy_name: str,
    config: dict,
    state_dir: Path | None = None,
) -> SignalComparison:
    """Build old/new signal diagnostics without changing production weights."""
    current = resolve_current_weights(
        asset_data,
        assets,
        signal_date,
        strategy_name,
        config,
        state_dir,
    )
    diagnostics = {
        asset: _latest_components(
            asset_data[asset].loc[
                asset_data[asset]["date"] <= pd.Timestamp(signal_date)
            ],
            int(config["window"]),
            current.values,
        )
        for asset in assets
    }
    old_confidence = _softmax_confidence(
        {asset: values["old_score"] for asset, values in diagnostics.items()}
    )
    new_confidence = _softmax_confidence(
        {asset: values["new_score"] for asset, values in diagnostics.items()}
    )
    for asset in assets:
        diagnostics[asset]["old_confidence"] = old_confidence[asset]
        diagnostics[asset]["new_confidence"] = new_confidence[asset]

    old_target = max(assets, key=lambda asset: diagnostics[asset]["old_score"])
    new_target = max(assets, key=lambda asset: diagnostics[asset]["new_score"])
    return SignalComparison(
        weights=current,
        old_target=old_target,
        new_target=new_target,
        assets=diagnostics,
    )
