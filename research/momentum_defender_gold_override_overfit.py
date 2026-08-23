"""Overfitting diagnostics for the searched C2 Gold override family."""

from __future__ import annotations

from dataclasses import asdict
from itertools import combinations
from typing import Mapping

import numpy as np
import pandas as pd

from research.momentum_defender_gold_override import (
    GoldOverrideContext,
    GoldOverrideParams,
    gold_override_schedule,
    metric_at_open,
    simulate_candidate_schedule,
)


ANNUALIZATION = 252.0


def collect_candidate_returns(
    context: GoldOverrideContext,
    grids: list[Mapping[str, Mapping[str, list[float | int]]]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rebuild every unique searched candidate and retain its daily returns."""
    records: list[dict[str, object]] = []
    returns: dict[str, np.ndarray] = {}
    metric_cache: dict[tuple[str, int], pd.DataFrame] = {}
    for grid in grids:
        for metric, settings in grid.items():
            for window_value in settings["windows"]:
                window = int(window_value)
                cache_key = (str(metric), window)
                if cache_key not in metric_cache:
                    metric_cache[cache_key] = metric_at_open(
                        context.curves, str(metric), window
                    )
                metrics = metric_cache[cache_key]
                for entry_value in settings["entry_thresholds"]:
                    for exit_value in settings["exit_thresholds"]:
                        entry = float(entry_value)
                        exit_ = float(exit_value)
                        if exit_ > entry:
                            continue
                        for hold_value in settings["min_gold_hold_days"]:
                            params = GoldOverrideParams(
                                metric=str(metric),
                                window=window,
                                entry_threshold=entry,
                                exit_threshold=exit_,
                                min_gold_hold_days=int(hold_value),
                            )
                            candidate_id = params.candidate_id()
                            if candidate_id in returns:
                                continue
                            state = gold_override_schedule(
                                context, metrics, params
                            )
                            daily = simulate_candidate_schedule(
                                state["target_candidate"],
                                context.interfaces,
                                context.initial_previous_candidate,
                            )
                            returns[candidate_id] = daily["return"].to_numpy(float)
                            records.append(
                                {
                                    "candidate_id": candidate_id,
                                    **asdict(params),
                                    "gold_override_entries": int(
                                        (
                                            state["gold_override_changed"].astype(bool)
                                            & state["gold_override_active"].astype(bool)
                                        ).sum()
                                    ),
                                    "gold_override_days": int(
                                        state["gold_override_active"].sum()
                                    ),
                                }
                            )
    metadata = pd.DataFrame(records).set_index("candidate_id")
    matrix = pd.DataFrame(returns, index=context.calendar)
    return metadata, matrix


def _stats(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    means = values.mean(axis=0)
    std = values.std(axis=0, ddof=1)
    sharpe = np.divide(
        means * np.sqrt(ANNUALIZATION),
        std,
        out=np.zeros_like(means),
        where=std > 0.0,
    )
    total = np.prod(1.0 + values, axis=0) - 1.0
    annual = np.power(1.0 + total, ANNUALIZATION / len(values)) - 1.0
    nav = np.cumprod(1.0 + values, axis=0)
    drawdown = nav / np.maximum.accumulate(nav, axis=0) - 1.0
    mdd = drawdown.min(axis=0)
    return annual, sharpe, mdd


def full_metrics(
    returns: pd.DataFrame,
    baseline: pd.Series,
) -> pd.DataFrame:
    annual, sharpe, mdd = _stats(returns.to_numpy(float))
    baseline_annual, baseline_sharpe, baseline_mdd = _stats(
        baseline.to_numpy(float)[:, None]
    )
    return pd.DataFrame(
        {
            "annualized_return_252": annual,
            "sharpe": sharpe,
            "max_drawdown": mdd,
            "delta_annualized_return_252": annual - baseline_annual[0],
            "delta_sharpe": sharpe - baseline_sharpe[0],
            "delta_max_drawdown": mdd - baseline_mdd[0],
        },
        index=returns.columns,
    )


def _block_sufficient_statistics(
    values: np.ndarray,
    blocks: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sums = np.vstack([values[index].sum(axis=0) for index in blocks])
    sums_sq = np.vstack([(values[index] ** 2).sum(axis=0) for index in blocks])
    counts = np.asarray([len(index) for index in blocks], dtype=float)
    return sums, sums_sq, counts


def _sharpe_from_sufficient(
    sums: np.ndarray,
    sums_sq: np.ndarray,
    count: float,
) -> np.ndarray:
    mean = sums / count
    variance = np.maximum((sums_sq - count * mean**2) / max(count - 1.0, 1.0), 0.0)
    std = np.sqrt(variance)
    return np.divide(
        mean * np.sqrt(ANNUALIZATION),
        std,
        out=np.zeros_like(mean),
        where=std > 0.0,
    )


def cscv_pbo(
    returns: pd.DataFrame,
    baseline: pd.Series,
    *,
    block_count: int = 16,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    """Run symmetric combinatorially-symmetric CV using sequential blocks."""
    if block_count % 2:
        raise ValueError("block_count must be even")
    blocks = [np.asarray(index) for index in np.array_split(np.arange(len(returns)), block_count)]
    values = returns.to_numpy(float)
    base = baseline.to_numpy(float)[:, None]
    sums, sums_sq, counts = _block_sufficient_statistics(values, blocks)
    base_sums, base_sums_sq, _ = _block_sufficient_statistics(base, blocks)
    all_blocks = set(range(block_count))
    half = block_count // 2
    rows: list[dict[str, object]] = []
    # Requiring block zero in train removes duplicate complementary partitions.
    for split_id, rest in enumerate(combinations(range(1, block_count), half - 1), start=1):
        train_blocks = (0, *rest)
        test_blocks = tuple(sorted(all_blocks - set(train_blocks)))
        train_count = float(counts[list(train_blocks)].sum())
        test_count = float(counts[list(test_blocks)].sum())
        train_sharpe = _sharpe_from_sufficient(
            sums[list(train_blocks)].sum(axis=0),
            sums_sq[list(train_blocks)].sum(axis=0),
            train_count,
        )
        winner = int(np.argmax(train_sharpe))
        test_sharpe = _sharpe_from_sufficient(
            sums[list(test_blocks)].sum(axis=0),
            sums_sq[list(test_blocks)].sum(axis=0),
            test_count,
        )
        selected_test = float(test_sharpe[winner])
        percentile = float((test_sharpe <= selected_test).mean())
        clipped = min(max(percentile, 1e-12), 1.0 - 1e-12)
        base_test = float(
            _sharpe_from_sufficient(
                base_sums[list(test_blocks)].sum(axis=0),
                base_sums_sq[list(test_blocks)].sum(axis=0),
                test_count,
            )[0]
        )
        rows.append(
            {
                "split_id": split_id,
                "train_blocks": ",".join(map(str, train_blocks)),
                "test_blocks": ",".join(map(str, test_blocks)),
                "selected_candidate": returns.columns[winner],
                "train_sharpe": float(train_sharpe[winner]),
                "test_sharpe": selected_test,
                "test_rank_percentile": percentile,
                "logit_rank": float(np.log(clipped / (1.0 - clipped))),
                "baseline_test_sharpe": base_test,
                "selected_beats_baseline_test": selected_test > base_test,
            }
        )
    frame = pd.DataFrame(rows)
    summary = {
        "block_count": block_count,
        "split_count": int(len(frame)),
        "pbo": float(frame["test_rank_percentile"].le(0.5).mean()),
        "median_test_rank_percentile": float(frame["test_rank_percentile"].median()),
        "selected_beats_baseline_rate": float(
            frame["selected_beats_baseline_test"].mean()
        ),
    }
    return frame, summary


def expanding_walk_forward(
    returns: pd.DataFrame,
    baseline: pd.Series,
) -> pd.DataFrame:
    years = sorted(returns.index.year.unique())
    rows: list[dict[str, object]] = []
    for test_position in range(3, len(years)):
        train_years = years[:test_position]
        test_year = years[test_position]
        train = returns.loc[returns.index.year.isin(train_years)]
        _, train_sharpe, _ = _stats(train.to_numpy(float))
        winner = int(np.argmax(train_sharpe))
        test = returns.loc[returns.index.year == test_year].iloc[:, winner]
        base_test = baseline.loc[baseline.index.year == test_year]
        candidate_metrics = _stats(test.to_numpy(float)[:, None])
        baseline_metrics = _stats(base_test.to_numpy(float)[:, None])
        rows.append(
            {
                "test_year": int(test_year),
                "train_years": ",".join(map(str, train_years)),
                "selected_candidate": returns.columns[winner],
                "train_sharpe": float(train_sharpe[winner]),
                "test_annualized_return_252": float(candidate_metrics[0][0]),
                "test_sharpe": float(candidate_metrics[1][0]),
                "test_max_drawdown": float(candidate_metrics[2][0]),
                "baseline_annualized_return_252": float(baseline_metrics[0][0]),
                "baseline_sharpe": float(baseline_metrics[1][0]),
                "baseline_max_drawdown": float(baseline_metrics[2][0]),
                "test_return_delta": float(candidate_metrics[0][0] - baseline_metrics[0][0]),
                "test_sharpe_delta": float(candidate_metrics[1][0] - baseline_metrics[1][0]),
            }
        )
    return pd.DataFrame(rows)


def leave_one_year_selection(
    returns: pd.DataFrame,
    baseline: pd.Series,
) -> pd.DataFrame:
    years = sorted(returns.index.year.unique())
    rows: list[dict[str, object]] = []
    for held_year in years:
        train = returns.loc[returns.index.year != held_year]
        _, train_sharpe, _ = _stats(train.to_numpy(float))
        winner = int(np.argmax(train_sharpe))
        test = returns.loc[returns.index.year == held_year].iloc[:, winner]
        base_test = baseline.loc[baseline.index.year == held_year]
        candidate_metrics = _stats(test.to_numpy(float)[:, None])
        baseline_metrics = _stats(base_test.to_numpy(float)[:, None])
        rows.append(
            {
                "held_year": int(held_year),
                "selected_candidate": returns.columns[winner],
                "train_sharpe": float(train_sharpe[winner]),
                "test_annualized_return_252": float(candidate_metrics[0][0]),
                "test_sharpe": float(candidate_metrics[1][0]),
                "baseline_annualized_return_252": float(baseline_metrics[0][0]),
                "baseline_sharpe": float(baseline_metrics[1][0]),
                "test_return_delta": float(candidate_metrics[0][0] - baseline_metrics[0][0]),
                "test_sharpe_delta": float(candidate_metrics[1][0] - baseline_metrics[1][0]),
            }
        )
    return pd.DataFrame(rows)


def paired_block_bootstrap(
    candidate: pd.Series,
    baseline: pd.Series,
    *,
    block_size: int = 20,
    repetitions: int = 5000,
    seed: int = 20260823,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    rng = np.random.default_rng(seed)
    candidate_values = candidate.to_numpy(float)
    baseline_values = baseline.to_numpy(float)
    observations = len(candidate_values)
    block_count = int(np.ceil(observations / block_size))
    rows = []
    for repetition in range(repetitions):
        starts = rng.integers(0, observations, size=block_count)
        indices = np.concatenate(
            [
                (np.arange(start, start + block_size) % observations)
                for start in starts
            ]
        )[:observations]
        candidate_sample = candidate_values[indices]
        baseline_sample = baseline_values[indices]
        candidate_metrics = _stats(candidate_sample[:, None])
        baseline_metrics = _stats(baseline_sample[:, None])
        rows.append(
            {
                "repetition": repetition + 1,
                "annualized_return_delta": float(
                    candidate_metrics[0][0] - baseline_metrics[0][0]
                ),
                "sharpe_delta": float(
                    candidate_metrics[1][0] - baseline_metrics[1][0]
                ),
                "max_drawdown_delta": float(
                    candidate_metrics[2][0] - baseline_metrics[2][0]
                ),
            }
        )
    frame = pd.DataFrame(rows)
    summary: dict[str, float | int] = {
        "block_size": block_size,
        "repetitions": repetitions,
        "seed": seed,
    }
    for field in (
        "annualized_return_delta",
        "sharpe_delta",
        "max_drawdown_delta",
    ):
        summary[f"{field}_mean"] = float(frame[field].mean())
        summary[f"{field}_ci_lower"] = float(frame[field].quantile(0.025))
        summary[f"{field}_ci_upper"] = float(frame[field].quantile(0.975))
        summary[f"{field}_positive_probability"] = float(frame[field].gt(0).mean())
    return frame, summary


def yearly_reality_check(
    returns: pd.DataFrame,
    baseline: pd.Series,
    *,
    repetitions: int = 5000,
    seed: int = 20260823,
) -> dict[str, float | int | str]:
    """White-style max-mean test using calendar years as dependence blocks."""
    years = sorted(returns.index.year.unique())
    candidate_log = np.log1p(returns.to_numpy(float))
    baseline_log = np.log1p(baseline.to_numpy(float))[:, None]
    difference = candidate_log - baseline_log
    blocks = np.vstack(
        [difference[returns.index.year == year].sum(axis=0) for year in years]
    )
    observed_by_candidate = blocks.mean(axis=0)
    observed_max = float(observed_by_candidate.max())
    winner = int(np.argmax(observed_by_candidate))
    centered = blocks - blocks.mean(axis=0, keepdims=True)
    rng = np.random.default_rng(seed)
    maxima = np.empty(repetitions, dtype=float)
    batch_size = 100
    for start in range(0, repetitions, batch_size):
        size = min(batch_size, repetitions - start)
        selections = rng.integers(0, len(years), size=(size, len(years)))
        samples = centered[selections].mean(axis=1)
        maxima[start : start + size] = samples.max(axis=1)
    return {
        "year_blocks": int(len(years)),
        "repetitions": repetitions,
        "seed": seed,
        "observed_best_candidate": returns.columns[winner],
        "observed_max_mean_annual_log_excess": observed_max,
        "p_value": float((maxima >= observed_max).mean()),
        "bootstrap_95pct_critical_value": float(np.quantile(maxima, 0.95)),
    }
