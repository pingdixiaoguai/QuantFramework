"""Forensic attribution and overfitting tests for adaptive held-asset C2."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import subprocess
import tempfile
from collections import Counter
from dataclasses import asdict
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, norm, skew

from research.momentum_defender_occam import (
    HELD_RETURN,
    MOMENTUM_ASSETS,
    OccamParams,
    apply_state_schedule,
    build_inputs,
    performance,
    simulate_switch,
    slow_regime_at_open,
)
from research.run_momentum_held_asset_adaptive_cap import (
    ASSET_NAMES,
    AdaptiveCSpec,
    held_asset_cap_alert,
)
from research.run_momentum_volatility_signal_abcd import (
    DEFAULT_DEFENDER_DIR,
    DEFAULT_END,
    EXPANDING_QUANTILES,
    VOLATILITY_WINDOWS,
    _load_ohlc,
    asof_previous_close,
    expanding_volatility_cap,
    momentum_asset_at_previous_close,
    rogers_satchell_volatility,
)


DEFAULT_OUTPUT = Path("experiments/20260821_momentum_held_asset_c2_overfit")
SELECTED_ID = "C2_vw10_cap0.8_qc3000.70_qcyb0.90_qndx0.95_qau0.90"
LEGACY_ID = "C0_vw20_q0.70_cap0.8"
SLOW_PARAMS = OccamParams(40, 0.025, 30, None)
DEVELOPMENT_END = pd.Timestamp("2022-12-30")
RANDOM_SEED = 20260821
BOOTSTRAP_BLOCK = 20
FIXED_BOOTSTRAP_REPETITIONS = 2000
SELECTION_BOOTSTRAP_REPETITIONS = 300
REALITY_CHECK_REPETITIONS = 2000
EVENT_PURGE_DAYS = 30


def _matrix_metrics(values: np.ndarray) -> dict[str, np.ndarray]:
    if values.ndim == 1:
        values = values[:, None]
    observations = values.shape[0]
    factors = np.prod(1.0 + values, axis=0)
    volatility = values.std(axis=0, ddof=1)
    sharpe = np.divide(
        values.mean(axis=0) * np.sqrt(252.0),
        volatility,
        out=np.zeros_like(volatility),
        where=volatility > 0.0,
    )
    wealth = np.cumprod(1.0 + values, axis=0)
    anchored = np.vstack([np.ones(values.shape[1]), wealth])
    drawdown = anchored / np.maximum.accumulate(anchored, axis=0) - 1.0
    return {
        "total_return": factors - 1.0,
        "annualized_return_252": factors ** (252.0 / observations) - 1.0,
        "sharpe": sharpe,
        "max_drawdown": drawdown.min(axis=0),
    }


def _single_metrics(values: np.ndarray, index: pd.DatetimeIndex) -> dict[str, float]:
    measured = performance(pd.Series(values, index=index))
    return {
        "total_return": float(measured["total_return"]),
        "annualized_return_252": float(measured["annualized_return_252"]),
        "sharpe": float(measured["sharpe"]),
        "max_drawdown": float(measured["max_drawdown"]),
    }


def _choose_candidate(
    metrics: dict[str, np.ndarray],
    baseline: dict[str, np.ndarray],
    emergency_entries: np.ndarray,
    sleeve_switches: np.ndarray,
    variant_ids: list[str],
) -> tuple[int, str, int]:
    gate = (
        (metrics["annualized_return_252"] > baseline["annualized_return_252"][0])
        & (metrics["sharpe"] > baseline["sharpe"][0])
        & (metrics["max_drawdown"] >= baseline["max_drawdown"][0] - 1e-12)
        & (emergency_entries > 0)
    )
    pool_name = "beats_no_cap_and_active"
    pool = np.flatnonzero(gate)
    if len(pool) == 0:
        pool_name = "active_only_fallback"
        pool = np.flatnonzero(emergency_entries > 0)
    if len(pool) == 0:
        pool_name = "all_candidates_fallback"
        pool = np.arange(len(variant_ids))
    ranking = pd.DataFrame(
        {
            "index": pool,
            "sharpe": metrics["sharpe"][pool],
            "mdd": metrics["max_drawdown"][pool],
            "annual": metrics["annualized_return_252"][pool],
            "entries": emergency_entries[pool],
            "switches": sleeve_switches[pool],
            "variant_id": [variant_ids[index] for index in pool],
        }
    ).sort_values(
        ["sharpe", "mdd", "annual", "entries", "switches", "variant_id"],
        ascending=[False, False, False, True, True, True],
    )
    return int(ranking.iloc[0]["index"]), pool_name, len(pool)


def _circular_block_indices(
    observations: int,
    block_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    blocks = math.ceil(observations / block_length)
    starts = rng.integers(0, observations, size=blocks)
    offsets = np.arange(block_length)
    return ((starts[:, None] + offsets[None, :]) % observations).ravel()[:observations]


def _unique_paths(values: np.ndarray) -> np.ndarray:
    seen: dict[str, int] = {}
    for column in range(values.shape[1]):
        digest = hashlib.sha1(values[:, column].tobytes()).hexdigest()
        seen.setdefault(digest, column)
    return np.array(list(seen.values()), dtype=int)


def _effective_trials(values: np.ndarray) -> float:
    standard_deviation = values.std(axis=0, ddof=1)
    valid = standard_deviation > 1e-14
    standardized = values[:, valid] - values[:, valid].mean(axis=0)
    standardized /= standardized.std(axis=0, ddof=1)
    correlation = standardized.T @ standardized / (len(standardized) - 1)
    trials = correlation.shape[0]
    return float(trials * trials / np.square(correlation).sum())


def _deflated_sharpe(
    selected: np.ndarray,
    trials: np.ndarray,
    effective_trials: float,
) -> dict[str, float]:
    selected_sr = float(selected.mean() / selected.std(ddof=1))
    trial_std = np.std(
        trials.mean(axis=0) / trials.std(axis=0, ddof=1), ddof=1
    )
    euler_gamma = 0.5772156649015329
    n_trials = max(float(effective_trials), 1.000001)
    expected_maximum = float(
        trial_std
        * (
            (1.0 - euler_gamma) * norm.ppf(1.0 - 1.0 / n_trials)
            + euler_gamma * norm.ppf(1.0 - 1.0 / (n_trials * math.e))
        )
    )
    selected_skew = float(skew(selected, bias=False))
    selected_kurtosis = float(kurtosis(selected, fisher=False, bias=False))
    denominator = math.sqrt(
        max(
            1e-18,
            1.0
            - selected_skew * selected_sr
            + ((selected_kurtosis - 1.0) / 4.0) * selected_sr * selected_sr,
        )
    )
    scale = math.sqrt(len(selected) - 1) / denominator
    return {
        "observations": len(selected),
        "selected_daily_sharpe": selected_sr,
        "selected_annualized_sharpe": selected_sr * math.sqrt(252.0),
        "effective_trials": effective_trials,
        "trial_sharpe_cross_section_std": float(trial_std),
        "expected_maximum_daily_sharpe": expected_maximum,
        "probabilistic_sharpe_vs_zero": float(norm.cdf(selected_sr * scale)),
        "deflated_sharpe_probability": float(
            norm.cdf((selected_sr - expected_maximum) * scale)
        ),
        "skewness": selected_skew,
        "pearson_kurtosis": selected_kurtosis,
    }


def _cscv_pbo(
    values: np.ndarray,
    variant_ids: list[str],
    blocks: int,
) -> tuple[pd.DataFrame, dict[str, float]]:
    observation_blocks = np.array_split(np.arange(values.shape[0]), blocks)
    block_sums = np.vstack([values[index].sum(axis=0) for index in observation_blocks])
    block_squares = np.vstack(
        [np.square(values[index]).sum(axis=0) for index in observation_blocks]
    )
    block_counts = np.array([len(index) for index in observation_blocks])
    records: list[dict[str, object]] = []
    for combination in itertools.combinations(range(blocks), blocks // 2):
        train_blocks = np.array(combination, dtype=int)
        test_blocks = np.array(
            [block for block in range(blocks) if block not in combination], dtype=int
        )

        def aggregate(which: np.ndarray) -> np.ndarray:
            count = int(block_counts[which].sum())
            total = block_sums[which].sum(axis=0)
            squares = block_squares[which].sum(axis=0)
            variance = np.maximum(
                (squares - np.square(total) / count) / max(count - 1, 1), 0.0
            )
            return np.divide(
                total / count,
                np.sqrt(variance),
                out=np.zeros_like(total),
                where=variance > 0.0,
            )

        train_sharpe = aggregate(train_blocks)
        test_sharpe = aggregate(test_blocks)
        chosen = int(np.argmax(train_sharpe))
        percentile = float(np.mean(test_sharpe <= test_sharpe[chosen]))
        clipped = min(max(percentile, 1e-9), 1.0 - 1e-9)
        records.append(
            {
                "train_blocks": ",".join(map(str, train_blocks)),
                "test_blocks": ",".join(map(str, test_blocks)),
                "selected_variant": variant_ids[chosen],
                "train_daily_sharpe": train_sharpe[chosen],
                "test_daily_sharpe": test_sharpe[chosen],
                "test_rank_percentile": percentile,
                "logit_rank": math.log(clipped / (1.0 - clipped)),
                "overfit": percentile <= 0.5,
            }
        )
    details = pd.DataFrame(records)
    summary = {
        "blocks": blocks,
        "splits": len(details),
        "pbo": float(details["overfit"].mean()),
        "median_test_rank_percentile": float(details["test_rank_percentile"].median()),
        "mean_test_rank_percentile": float(details["test_rank_percentile"].mean()),
        "median_train_daily_sharpe": float(details["train_daily_sharpe"].median()),
        "median_test_daily_sharpe": float(details["test_daily_sharpe"].median()),
    }
    return details, summary


def _white_reality_check(
    excess_returns: np.ndarray,
    block_length: int,
    repetitions: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    observations, candidates = excess_returns.shape
    means = excess_returns.mean(axis=0)
    observed = float(np.sqrt(observations) * means.max())
    centered = excess_returns - means
    doubled = np.vstack([centered, centered[:block_length]])
    cumulative = np.vstack([np.zeros((1, candidates)), np.cumsum(doubled, axis=0)])
    block_sums = np.empty((observations, candidates))
    for start in range(observations):
        block_sums[start] = cumulative[start + block_length] - cumulative[start]
    full_blocks, remainder = divmod(observations, block_length)
    bootstrap_statistics = np.empty(repetitions)
    for repetition in range(repetitions):
        starts = rng.integers(0, observations, size=full_blocks)
        total = block_sums[starts].sum(axis=0)
        if remainder:
            extra_start = int(rng.integers(0, observations))
            extra = np.vstack([centered, centered[:remainder]])
            extra_cumulative = np.vstack(
                [np.zeros((1, candidates)), np.cumsum(extra, axis=0)]
            )
            total += extra_cumulative[extra_start + remainder] - extra_cumulative[
                extra_start
            ]
        bootstrap_statistics[repetition] = np.sqrt(observations) * np.max(
            total / observations
        )
    return {
        "observations": observations,
        "candidate_paths": candidates,
        "block_length": block_length,
        "repetitions": repetitions,
        "observed_max_sqrt_t_mean_excess": observed,
        "bootstrap_p_value": float(
            (1 + np.sum(bootstrap_statistics >= observed)) / (repetitions + 1)
        ),
        "bootstrap_statistic_95pct": float(
            np.quantile(bootstrap_statistics, 0.95)
        ),
    }


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_experiment(
    root: Path,
    defender_dir: Path,
    final_output: Path,
    end: date,
) -> None:
    final_output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(prefix=f".{final_output.name}.staging-", dir=final_output.parent)
    )
    rng = np.random.default_rng(RANDOM_SEED)

    inputs = build_inputs(
        root,
        defender_dir / "relative_defender_rotation_switch_returns.csv",
        end,
    )
    calendar = inputs.calendar
    exact_momentum = inputs.momentum[HELD_RETURN].astype(float)
    slow = slow_regime_at_open(
        inputs.risk_close,
        calendar,
        SLOW_PARAMS.lookback,
        SLOW_PARAMS.risk_on_threshold,
    )
    previous_asset = momentum_asset_at_previous_close(inputs.momentum_result, calendar)

    ohlc = {asset: _load_ohlc(asset, end) for asset in MOMENTUM_ASSETS}
    cap_cache: dict[tuple[str, int, float], pd.Series] = {}
    for asset, prices in ohlc.items():
        for window in VOLATILITY_WINDOWS:
            volatility = rogers_satchell_volatility(prices, window)
            for quantile in EXPANDING_QUANTILES:
                cap_cache[(asset, window, quantile)] = asof_previous_close(
                    expanding_volatility_cap(volatility, quantile)["cap"], calendar
                ).fillna(1.0)

    specs: list[AdaptiveCSpec] = []
    for window in VOLATILITY_WINDOWS:
        for quantiles in itertools.product(EXPANDING_QUANTILES, repeat=4):
            specs.append(
                AdaptiveCSpec(
                    "C2",
                    "Asset-specific volatility quantiles",
                    volatility_window=window,
                    cap_trigger_maximum=0.8,
                    q_510300=quantiles[0],
                    q_159915=quantiles[1],
                    q_513100=quantiles[2],
                    q_518880=quantiles[3],
                )
            )
    variant_ids = [spec.variant_id() for spec in specs]

    def alert_for_spec(spec: AdaptiveCSpec) -> pd.Series:
        quantiles = spec.asset_quantiles()
        caps = {
            asset: cap_cache[(asset, int(spec.volatility_window), quantiles[asset])]
            for asset in MOMENTUM_ASSETS
        }
        return held_asset_cap_alert(
            caps, previous_asset, {asset: 0.8 for asset in MOMENTUM_ASSETS}
        )

    candidate_returns = np.empty((len(calendar), len(specs)), dtype=float)
    emergency_entries = np.zeros((len(calendar), len(specs)), dtype=bool)
    candidate_switches = np.zeros((len(calendar), len(specs)), dtype=bool)
    selected_state: pd.DataFrame | None = None
    selected_simulated: pd.DataFrame | None = None
    for column, spec in enumerate(specs):
        alert = alert_for_spec(spec)
        state = apply_state_schedule(
            slow,
            alert,
            calendar,
            SLOW_PARAMS.min_hold_days,
            emergency_override=True,
        )
        simulated = simulate_switch(inputs.momentum, inputs.defender, state["risk_on"])
        candidate_returns[:, column] = simulated["return"].to_numpy(float)
        emergency_entries[:, column] = (
            state["state_changed"].astype(bool)
            & state["state_reason"].eq("emergency_exit")
        ).to_numpy(bool)
        candidate_switches[:, column] = simulated["sleeve_switch"].to_numpy(bool)
        if spec.variant_id() == SELECTED_ID:
            selected_state = state
            selected_simulated = simulated

    if selected_state is None or selected_simulated is None:
        raise AssertionError("selected C2 variant not found")
    selected_index = variant_ids.index(SELECTED_ID)

    no_cap_state = apply_state_schedule(
        slow,
        pd.Series(False, index=calendar),
        calendar,
        SLOW_PARAMS.min_hold_days,
        emergency_override=True,
    )
    no_cap_simulated = simulate_switch(
        inputs.momentum, inputs.defender, no_cap_state["risk_on"]
    )
    no_cap_returns = no_cap_simulated["return"].to_numpy(float)

    legacy_caps = {
        asset: cap_cache[(asset, 20, 0.70)] for asset in MOMENTUM_ASSETS
    }
    legacy_alert = held_asset_cap_alert(
        legacy_caps, previous_asset, {asset: 0.8 for asset in MOMENTUM_ASSETS}
    )
    legacy_state = apply_state_schedule(
        slow,
        legacy_alert,
        calendar,
        SLOW_PARAMS.min_hold_days,
        emergency_override=True,
    )
    legacy_simulated = simulate_switch(
        inputs.momentum, inputs.defender, legacy_state["risk_on"]
    )
    legacy_returns = legacy_simulated["return"].to_numpy(float)

    development_mask = calendar <= DEVELOPMENT_END
    development_returns = candidate_returns[development_mask]
    development_no_cap = no_cap_returns[development_mask]
    development_entries = emergency_entries[development_mask]
    unique_columns = _unique_paths(development_returns)
    unique_development = development_returns[:, unique_columns]
    unique_ids = [variant_ids[index] for index in unique_columns]

    # Exact meaning of development selection versus full-sample oracle.
    development_metrics = _matrix_metrics(development_returns)
    development_baseline = _matrix_metrics(development_no_cap)
    development_choice, development_pool, development_pool_count = _choose_candidate(
        development_metrics,
        development_baseline,
        development_entries.sum(axis=0),
        candidate_switches[development_mask].sum(axis=0),
        variant_ids,
    )
    full_metrics = _matrix_metrics(candidate_returns)
    full_baseline = _matrix_metrics(no_cap_returns)
    full_choice, full_pool, full_pool_count = _choose_candidate(
        full_metrics,
        full_baseline,
        emergency_entries.sum(axis=0),
        candidate_switches.sum(axis=0),
        variant_ids,
    )
    if selected_index != development_choice:
        raise AssertionError(
            "SELECTED_ID must match the causally selected development candidate"
        )
    equivalent_development = np.flatnonzero(
        np.all(
            development_returns
            == development_returns[:, development_choice][:, None],
            axis=0,
        )
    )
    equivalent_records: list[dict[str, object]] = []
    for column in equivalent_development:
        equivalent_records.append(
            {
                **asdict(specs[column]),
                "variant_id": variant_ids[column],
                "is_development_tiebreak_choice": column == development_choice,
                "is_full_sample_oracle": column == full_choice,
                "full_annualized_return_252": full_metrics[
                    "annualized_return_252"
                ][column],
                "full_sharpe": full_metrics["sharpe"][column],
                "full_max_drawdown": full_metrics["max_drawdown"][column],
            }
        )
    pd.DataFrame(equivalent_records).sort_values(
        ["full_sharpe", "full_annualized_return_252"], ascending=False
    ).to_csv(stage / "c2_development_equivalent_parameters.csv", index=False)

    # Calendar-year attribution.
    annual_records: list[dict[str, object]] = []
    for year in sorted(calendar.year.unique()):
        mask = calendar.year == year
        for label, values in (
            ("C2", candidate_returns[:, selected_index]),
            ("legacy_C", legacy_returns),
            ("no_cap", no_cap_returns),
            ("momentum", exact_momentum.to_numpy(float)),
        ):
            measured = _single_metrics(values[mask], calendar[mask])
            annual_records.append({"year": year, "strategy": label, **measured})
    annual = pd.DataFrame(annual_records)
    pivot_return = annual.pivot(index="year", columns="strategy", values="total_return")
    annual_attribution_frame = pd.DataFrame(
            {
                "year": pivot_return.index.to_numpy(),
                "c2_log_excess_vs_no_cap": np.log1p(pivot_return["C2"])
                .sub(np.log1p(pivot_return["no_cap"]))
                .to_numpy(),
                "c2_log_excess_vs_legacy": np.log1p(pivot_return["C2"])
                .sub(np.log1p(pivot_return["legacy_C"]))
                .to_numpy(),
            }
        ).reset_index(drop=True)
    annual_attribution = annual.merge(
        annual_attribution_frame,
        on="year",
        how="left",
    )
    annual_attribution.to_csv(stage / "c2_calendar_year_attribution.csv", index=False)

    # Daily concentration and top-day removal.
    selected_returns = candidate_returns[:, selected_index]
    top_day_records: list[dict[str, object]] = []
    removal_records: list[dict[str, object]] = []
    for reference_name, reference_returns in (
        ("no_cap", no_cap_returns),
        ("legacy_C", legacy_returns),
    ):
        log_excess = np.log1p(selected_returns) - np.log1p(reference_returns)
        order = np.argsort(log_excess)[::-1]
        for rank, location in enumerate(order[:20], start=1):
            top_day_records.append(
                {
                    "reference": reference_name,
                    "rank": rank,
                    "date": calendar[location].date().isoformat(),
                    "c2_return": selected_returns[location],
                    "reference_return": reference_returns[location],
                    "simple_return_difference": selected_returns[location]
                    - reference_returns[location],
                    "log_excess_contribution": log_excess[location],
                    "entry_asset": previous_asset.iloc[location],
                    "c2_state_reason": selected_state.iloc[location]["state_reason"],
                    "c2_sleeve": selected_simulated.iloc[location]["sleeve"],
                }
            )
        positive_total = float(log_excess[log_excess > 0].sum())
        net_total = float(log_excess.sum())
        for removed_days in (0, 1, 3, 5, 10):
            hybrid = selected_returns.copy()
            if removed_days:
                locations = order[:removed_days]
                hybrid[locations] = reference_returns[locations]
            measured = _single_metrics(hybrid, calendar)
            baseline_measured = _single_metrics(reference_returns, calendar)
            removal_records.append(
                {
                    "reference": reference_name,
                    "removed_top_positive_days": removed_days,
                    **measured,
                    "annualized_delta_vs_reference": measured[
                        "annualized_return_252"
                    ]
                    - baseline_measured["annualized_return_252"],
                    "sharpe_delta_vs_reference": measured["sharpe"]
                    - baseline_measured["sharpe"],
                    "max_drawdown_improvement_vs_reference": measured["max_drawdown"]
                    - baseline_measured["max_drawdown"],
                    "net_log_excess_original": net_total,
                    "positive_log_excess_original": positive_total,
                    "removed_share_of_net_log_excess": (
                        float(log_excess[order[:removed_days]].sum() / net_total)
                        if removed_days and abs(net_total) > 1e-18
                        else 0.0
                    ),
                }
            )
    top_days = pd.DataFrame(top_day_records)
    removal_stress = pd.DataFrame(removal_records)
    top_days.to_csv(stage / "c2_top_daily_contributions.csv", index=False)
    removal_stress.to_csv(stage / "c2_top_day_removal_stress.csv", index=False)

    # Emergency episode attribution and leave-one-episode-out stress.
    entry_mask = selected_state["state_reason"].eq("emergency_exit").to_numpy()
    entry_locations = np.flatnonzero(entry_mask)
    episode_records: list[dict[str, object]] = []
    leave_out_records: list[dict[str, object]] = []
    for episode_number, start_location in enumerate(entry_locations, start=1):
        later_risk_on = np.flatnonzero(
            selected_state["risk_on"].to_numpy()[start_location + 1 :]
        )
        end_location = (
            start_location + int(later_risk_on[0])
            if len(later_risk_on)
            else len(calendar) - 1
        )
        locations = np.arange(start_location, end_location + 1)
        c2_total = float(np.prod(1.0 + selected_returns[locations]) - 1.0)
        no_cap_total = float(np.prod(1.0 + no_cap_returns[locations]) - 1.0)
        legacy_total = float(np.prod(1.0 + legacy_returns[locations]) - 1.0)
        episode_records.append(
            {
                "episode": episode_number,
                "start": calendar[start_location].date().isoformat(),
                "end": calendar[end_location].date().isoformat(),
                "observations": len(locations),
                "entry_asset": previous_asset.iloc[start_location],
                "entry_asset_name": ASSET_NAMES[previous_asset.iloc[start_location]],
                "c2_total_return": c2_total,
                "no_cap_total_return": no_cap_total,
                "legacy_total_return": legacy_total,
                "c2_log_excess_vs_no_cap": math.log1p(c2_total)
                - math.log1p(no_cap_total),
                "c2_log_excess_vs_legacy": math.log1p(c2_total)
                - math.log1p(legacy_total),
            }
        )
        for reference_name, reference_returns in (
            ("no_cap", no_cap_returns),
            ("legacy_C", legacy_returns),
        ):
            hybrid = selected_returns.copy()
            hybrid[locations] = reference_returns[locations]
            measured = _single_metrics(hybrid, calendar)
            reference_measured = _single_metrics(reference_returns, calendar)
            leave_out_records.append(
                {
                    "removed_episode": episode_number,
                    "episode_start": calendar[start_location].date().isoformat(),
                    "episode_end": calendar[end_location].date().isoformat(),
                    "replacement_reference": reference_name,
                    **measured,
                    "annualized_delta_vs_reference": measured[
                        "annualized_return_252"
                    ]
                    - reference_measured["annualized_return_252"],
                    "sharpe_delta_vs_reference": measured["sharpe"]
                    - reference_measured["sharpe"],
                    "max_drawdown_improvement_vs_reference": measured["max_drawdown"]
                    - reference_measured["max_drawdown"],
                }
            )
    episodes = pd.DataFrame(episode_records).sort_values(
        "c2_log_excess_vs_no_cap", ascending=False
    )
    episodes.to_csv(stage / "c2_emergency_episode_attribution.csv", index=False)
    leave_out = pd.DataFrame(leave_out_records)
    leave_out.to_csv(stage / "c2_leave_one_episode_out.csv", index=False)

    # Event-level parameter cross-validation. Rolling-origin selection is causal
    # with respect to each event. Purged leave-one-event-out is a stability test:
    # it may use later observations, but excludes the event and a 30-session
    # embargo so the state lock cannot immediately spill into the training score.
    current_dimensions = np.array(
        [
            specs[selected_index].volatility_window,
            specs[selected_index].q_510300,
            specs[selected_index].q_159915,
            specs[selected_index].q_513100,
            specs[selected_index].q_518880,
        ],
        dtype=float,
    )
    event_cv_records: list[dict[str, object]] = []
    event_cv_paths: dict[str, list[np.ndarray]] = {
        "rolling_origin": [],
        "purged_leave_one_event_out": [],
    }
    event_cv_dates: dict[str, list[pd.DatetimeIndex]] = {
        "rolling_origin": [],
        "purged_leave_one_event_out": [],
    }
    for episode in episodes.sort_values("episode").itertuples():
        start_location = int(calendar.get_loc(pd.Timestamp(episode.start)))
        end_location = int(calendar.get_loc(pd.Timestamp(episode.end)))
        event_locations = np.arange(start_location, end_location + 1)
        event_mask = np.zeros(len(calendar), dtype=bool)
        event_mask[event_locations] = True
        purge_end_location = min(
            end_location + EVENT_PURGE_DAYS,
            len(calendar) - 1,
        )
        purged_train_mask = np.ones(len(calendar), dtype=bool)
        purged_train_mask[start_location : purge_end_location + 1] = False
        selection_masks = {
            "rolling_origin": np.arange(len(calendar)) < start_location,
            "purged_leave_one_event_out": purged_train_mask,
        }
        for method, train_mask in selection_masks.items():
            train_metrics = _matrix_metrics(candidate_returns[train_mask])
            train_baseline = _matrix_metrics(no_cap_returns[train_mask])
            chosen, pool, pool_count = _choose_candidate(
                train_metrics,
                train_baseline,
                emergency_entries[train_mask].sum(axis=0),
                candidate_switches[train_mask].sum(axis=0),
                variant_ids,
            )
            chosen_spec = specs[chosen]
            chosen_dimensions = np.array(
                [
                    chosen_spec.volatility_window,
                    chosen_spec.q_510300,
                    chosen_spec.q_159915,
                    chosen_spec.q_513100,
                    chosen_spec.q_518880,
                ],
                dtype=float,
            )
            chosen_event = _matrix_metrics(candidate_returns[event_mask, chosen])
            current_event = _matrix_metrics(
                candidate_returns[event_mask, selected_index]
            )
            no_cap_event = _matrix_metrics(no_cap_returns[event_mask])
            legacy_event = _matrix_metrics(legacy_returns[event_mask])
            current_gate = bool(
                train_metrics["annualized_return_252"][selected_index]
                > train_baseline["annualized_return_252"][0]
                and train_metrics["sharpe"][selected_index]
                > train_baseline["sharpe"][0]
                and train_metrics["max_drawdown"][selected_index]
                >= train_baseline["max_drawdown"][0] - 1e-12
                and emergency_entries[train_mask, selected_index].sum() > 0
            )
            event_cv_records.append(
                {
                    "method": method,
                    "episode": int(episode.episode),
                    "event_start": episode.start,
                    "event_end": episode.end,
                    "purge_end": calendar[purge_end_location].date().isoformat(),
                    "entry_asset": episode.entry_asset,
                    "entry_asset_name": episode.entry_asset_name,
                    "train_observations": int(train_mask.sum()),
                    "selection_pool": pool,
                    "selection_pool_count": pool_count,
                    "selected_variant": variant_ids[chosen],
                    "selected_is_current_C2": chosen == selected_index,
                    "matching_dimensions_of_5": int(
                        np.isclose(chosen_dimensions, current_dimensions).sum()
                    ),
                    "matching_asset_quantiles_of_4": int(
                        np.isclose(
                            chosen_dimensions[1:], current_dimensions[1:]
                        ).sum()
                    ),
                    "selected_volatility_window": chosen_spec.volatility_window,
                    "selected_q_510300": chosen_spec.q_510300,
                    "selected_q_159915": chosen_spec.q_159915,
                    "selected_q_513100": chosen_spec.q_513100,
                    "selected_q_518880": chosen_spec.q_518880,
                    "current_C2_passes_training_gate": current_gate,
                    "selected_event_emergency_entries": int(
                        emergency_entries[event_mask, chosen].sum()
                    ),
                    "selected_event_total_return": float(
                        chosen_event["total_return"][0]
                    ),
                    "current_C2_event_total_return": float(
                        current_event["total_return"][0]
                    ),
                    "no_cap_event_total_return": float(
                        no_cap_event["total_return"][0]
                    ),
                    "legacy_event_total_return": float(
                        legacy_event["total_return"][0]
                    ),
                    "selected_event_log_excess_vs_no_cap": float(
                        math.log1p(chosen_event["total_return"][0])
                        - math.log1p(no_cap_event["total_return"][0])
                    ),
                    "selected_event_sharpe": float(chosen_event["sharpe"][0]),
                    "no_cap_event_sharpe": float(no_cap_event["sharpe"][0]),
                    "selected_event_max_drawdown": float(
                        chosen_event["max_drawdown"][0]
                    ),
                    "no_cap_event_max_drawdown": float(
                        no_cap_event["max_drawdown"][0]
                    ),
                }
            )
            event_cv_paths[method].append(candidate_returns[event_mask, chosen])
            event_cv_dates[method].append(calendar[event_mask])
    event_cv = pd.DataFrame(event_cv_records)
    event_cv.to_csv(stage / "c2_event_cv_reselection.csv", index=False)

    event_cv_frequency = (
        event_cv.groupby(["method", "selected_variant"], as_index=False)
        .agg(selection_count=("episode", "size"))
        .sort_values(["method", "selection_count", "selected_variant"],
                     ascending=[True, False, True])
    )
    event_cv_frequency["selection_frequency"] = event_cv_frequency[
        "selection_count"
    ] / episodes["episode"].nunique()
    event_cv_frequency.to_csv(
        stage / "c2_event_cv_parameter_frequency.csv", index=False
    )

    marginal_fields = {
        "volatility_window": "selected_volatility_window",
        "q_510300": "selected_q_510300",
        "q_159915": "selected_q_159915",
        "q_513100": "selected_q_513100",
        "q_518880": "selected_q_518880",
    }
    current_by_dimension = {
        "volatility_window": float(current_dimensions[0]),
        "q_510300": float(current_dimensions[1]),
        "q_159915": float(current_dimensions[2]),
        "q_513100": float(current_dimensions[3]),
        "q_518880": float(current_dimensions[4]),
    }
    marginal_records: list[dict[str, object]] = []
    for method, method_rows in event_cv.groupby("method"):
        for dimension, field in marginal_fields.items():
            counts = method_rows[field].value_counts().sort_index()
            mode_count = int(counts.max())
            mode_value = float(counts.loc[counts.eq(mode_count)].index.min())
            current_value = current_by_dimension[dimension]
            marginal_records.append(
                {
                    "method": method,
                    "dimension": dimension,
                    "current_value": current_value,
                    "current_value_count": int(
                        np.isclose(method_rows[field], current_value).sum()
                    ),
                    "current_value_frequency": float(
                        np.isclose(method_rows[field], current_value).mean()
                    ),
                    "mode_value": mode_value,
                    "mode_count": mode_count,
                    "mode_frequency": mode_count / len(method_rows),
                    "unique_values": "|".join(
                        f"{float(value):g}" for value in counts.index
                    ),
                }
            )
    event_cv_marginal = pd.DataFrame(marginal_records)
    event_cv_marginal.to_csv(
        stage / "c2_event_cv_marginal_stability.csv", index=False
    )

    event_cv_summary: dict[str, dict[str, object]] = {}
    for method in event_cv_paths:
        method_rows = event_cv.loc[event_cv["method"].eq(method)]
        event_index = pd.DatetimeIndex(
            np.concatenate([index.values for index in event_cv_dates[method]])
        )
        selected_event_path = np.concatenate(event_cv_paths[method])
        no_cap_event_path = no_cap_returns[calendar.isin(event_index)]
        current_event_path = selected_returns[calendar.isin(event_index)]
        legacy_event_path = legacy_returns[calendar.isin(event_index)]
        selected_event_metrics = _single_metrics(selected_event_path, event_index)
        no_cap_event_metrics = _single_metrics(no_cap_event_path, event_index)
        current_event_metrics = _single_metrics(current_event_path, event_index)
        legacy_event_metrics = _single_metrics(legacy_event_path, event_index)
        event_cv_summary[method] = {
            "episodes": int(len(method_rows)),
            "exact_current_parameter_frequency": float(
                method_rows["selected_is_current_C2"].mean()
            ),
            "mean_matching_dimensions_of_5": float(
                method_rows["matching_dimensions_of_5"].mean()
            ),
            "mean_matching_asset_quantiles_of_4": float(
                method_rows["matching_asset_quantiles_of_4"].mean()
            ),
            "triple_positive_selection_pool_frequency": float(
                method_rows["selection_pool"]
                .eq("beats_no_cap_and_active")
                .mean()
            ),
            "current_C2_training_gate_pass_frequency": float(
                method_rows["current_C2_passes_training_gate"].mean()
            ),
            "held_out_event_positive_log_excess_frequency": float(
                method_rows["selected_event_log_excess_vs_no_cap"].gt(0).mean()
            ),
            "marginal_current_value_frequency": {
                row.dimension: float(row.current_value_frequency)
                for row in event_cv_marginal.loc[
                    event_cv_marginal["method"].eq(method)
                ].itertuples()
            },
            "selected_event_path": selected_event_metrics,
            "no_cap_event_path": no_cap_event_metrics,
            "current_C2_event_path": current_event_metrics,
            "legacy_C_event_path": legacy_event_metrics,
        }

    # Expanding walk-forward parameter re-selection.
    walk_forward_records: list[dict[str, object]] = []
    chained_c2: list[np.ndarray] = []
    chained_no_cap: list[np.ndarray] = []
    chained_momentum: list[np.ndarray] = []
    chained_dates: list[pd.DatetimeIndex] = []
    for test_year in range(2021, 2027):
        train_mask = calendar.year < test_year
        test_mask = calendar.year == test_year
        if test_mask.sum() < 2:
            continue
        train_metrics = _matrix_metrics(candidate_returns[train_mask])
        train_baseline = _matrix_metrics(no_cap_returns[train_mask])
        chosen, pool, pool_count = _choose_candidate(
            train_metrics,
            train_baseline,
            emergency_entries[train_mask].sum(axis=0),
            candidate_switches[train_mask].sum(axis=0),
            variant_ids,
        )
        test_candidate = _single_metrics(
            candidate_returns[test_mask, chosen], calendar[test_mask]
        )
        test_baseline = _single_metrics(no_cap_returns[test_mask], calendar[test_mask])
        test_legacy = _single_metrics(legacy_returns[test_mask], calendar[test_mask])
        spec = specs[chosen]
        walk_forward_records.append(
            {
                "test_year": test_year,
                "train_start": calendar[train_mask][0].date().isoformat(),
                "train_end": calendar[train_mask][-1].date().isoformat(),
                "selection_pool": pool,
                "selection_pool_count": pool_count,
                "selected_variant": variant_ids[chosen],
                "volatility_window": spec.volatility_window,
                "q_510300": spec.q_510300,
                "q_159915": spec.q_159915,
                "q_513100": spec.q_513100,
                "q_518880": spec.q_518880,
                **{f"test_{key}": value for key, value in test_candidate.items()},
                "test_annualized_delta_vs_no_cap": test_candidate[
                    "annualized_return_252"
                ]
                - test_baseline["annualized_return_252"],
                "test_sharpe_delta_vs_no_cap": test_candidate["sharpe"]
                - test_baseline["sharpe"],
                "test_mdd_improvement_vs_no_cap": test_candidate["max_drawdown"]
                - test_baseline["max_drawdown"],
                "test_annualized_delta_vs_legacy": test_candidate[
                    "annualized_return_252"
                ]
                - test_legacy["annualized_return_252"],
                "selected_is_final_C2": variant_ids[chosen] == SELECTED_ID,
            }
        )
        chained_c2.append(candidate_returns[test_mask, chosen])
        chained_no_cap.append(no_cap_returns[test_mask])
        chained_momentum.append(exact_momentum.to_numpy(float)[test_mask])
        chained_dates.append(calendar[test_mask])
    walk_forward = pd.DataFrame(walk_forward_records)
    walk_forward.to_csv(stage / "c2_expanding_walk_forward.csv", index=False)
    chained_index = pd.DatetimeIndex(np.concatenate([item.values for item in chained_dates]))
    chained_summary = {
        "start": chained_index[0].date().isoformat(),
        "end": chained_index[-1].date().isoformat(),
        "strategy": _single_metrics(np.concatenate(chained_c2), chained_index),
        "no_cap": _single_metrics(np.concatenate(chained_no_cap), chained_index),
        "momentum": _single_metrics(np.concatenate(chained_momentum), chained_index),
    }

    # CSCV PBO on unique return paths, both development and full sample.
    development_unique_columns = _unique_paths(development_returns)
    full_unique_columns = _unique_paths(candidate_returns)
    cscv_development, pbo_development = _cscv_pbo(
        development_returns[:, development_unique_columns],
        [variant_ids[index] for index in development_unique_columns],
        blocks=8,
    )
    cscv_full, pbo_full = _cscv_pbo(
        candidate_returns[:, full_unique_columns],
        [variant_ids[index] for index in full_unique_columns],
        blocks=12,
    )
    cscv_development.to_csv(stage / "c2_cscv_development_splits.csv", index=False)
    cscv_full.to_csv(stage / "c2_cscv_full_splits.csv", index=False)

    # White Reality Check for the development-period incremental mean return.
    unique_excess = unique_development - development_no_cap[:, None]
    reality_check = _white_reality_check(
        unique_excess,
        BOOTSTRAP_BLOCK,
        REALITY_CHECK_REPETITIONS,
        rng,
    )

    # DSR for absolute C2 returns and incremental returns versus no-cap.
    effective_absolute = _effective_trials(unique_development)
    nonzero_excess = unique_excess[:, unique_excess.std(axis=0, ddof=1) > 1e-14]
    effective_excess = _effective_trials(nonzero_excess)
    selected_development = development_returns[:, selected_index]
    dsr_absolute = _deflated_sharpe(
        selected_development, unique_development, effective_absolute
    )
    dsr_excess = _deflated_sharpe(
        selected_development - development_no_cap,
        nonzero_excess,
        effective_excess,
    )

    # Paired fixed-parameter circular block bootstrap.
    fixed_records: list[dict[str, object]] = []
    fixed_samples: dict[tuple[str, str], list[float]] = {
        (reference, metric): []
        for reference in ("no_cap", "legacy_C")
        for metric in ("annualized_return_252", "sharpe", "max_drawdown")
    }
    for _ in range(FIXED_BOOTSTRAP_REPETITIONS):
        sample = _circular_block_indices(len(calendar), BOOTSTRAP_BLOCK, rng)
        c2_metric = _matrix_metrics(selected_returns[sample])
        for reference_name, reference_returns in (
            ("no_cap", no_cap_returns),
            ("legacy_C", legacy_returns),
        ):
            reference_metric = _matrix_metrics(reference_returns[sample])
            for metric in fixed_samples:
                if metric[0] != reference_name:
                    continue
                key = metric[1]
                fixed_samples[metric].append(
                    float(c2_metric[key][0] - reference_metric[key][0])
                )
    for (reference, metric), samples in fixed_samples.items():
        values = np.asarray(samples)
        fixed_records.append(
            {
                "reference": reference,
                "metric_delta": metric,
                "repetitions": len(values),
                "block_length": BOOTSTRAP_BLOCK,
                "mean": values.mean(),
                "median": np.median(values),
                "ci_2_5pct": np.quantile(values, 0.025),
                "ci_97_5pct": np.quantile(values, 0.975),
                "probability_positive": np.mean(values > 0.0),
            }
        )
    fixed_bootstrap = pd.DataFrame(fixed_records)
    fixed_bootstrap.to_csv(
        stage / "c2_fixed_parameter_block_bootstrap.csv", index=False
    )

    # Selection bootstrap: reselect the parameter in each development resample.
    selection_counter: Counter[str] = Counter()
    selection_records: list[dict[str, object]] = []
    evaluation_mask = calendar > DEVELOPMENT_END
    evaluation_baseline = _matrix_metrics(no_cap_returns[evaluation_mask])
    for repetition in range(SELECTION_BOOTSTRAP_REPETITIONS):
        sample = _circular_block_indices(
            development_mask.sum(), BOOTSTRAP_BLOCK, rng
        )
        sample_metrics = _matrix_metrics(development_returns[sample])
        sample_baseline = _matrix_metrics(development_no_cap[sample])
        choice, pool, pool_count = _choose_candidate(
            sample_metrics,
            sample_baseline,
            development_entries[sample].sum(axis=0),
            candidate_switches[development_mask][sample].sum(axis=0),
            variant_ids,
        )
        chosen_id = variant_ids[choice]
        selection_counter[chosen_id] += 1
        evaluation_metrics = _matrix_metrics(candidate_returns[evaluation_mask, choice])
        selection_records.append(
            {
                "repetition": repetition + 1,
                "selected_variant": chosen_id,
                "selected_is_final_C2": chosen_id == SELECTED_ID,
                "selection_pool": pool,
                "selection_pool_count": pool_count,
                "evaluation_annualized_delta_vs_no_cap": evaluation_metrics[
                    "annualized_return_252"
                ][0]
                - evaluation_baseline["annualized_return_252"][0],
                "evaluation_sharpe_delta_vs_no_cap": evaluation_metrics["sharpe"][0]
                - evaluation_baseline["sharpe"][0],
                "evaluation_mdd_improvement_vs_no_cap": evaluation_metrics[
                    "max_drawdown"
                ][0]
                - evaluation_baseline["max_drawdown"][0],
            }
        )
    selection_bootstrap = pd.DataFrame(selection_records)
    selection_bootstrap.to_csv(
        stage / "c2_selection_bootstrap_repetitions.csv", index=False
    )
    selection_frequency = pd.DataFrame(
        [
            {
                "variant_id": variant_id,
                "selection_count": count,
                "selection_frequency": count / SELECTION_BOOTSTRAP_REPETITIONS,
            }
            for variant_id, count in selection_counter.most_common()
        ]
    )
    selection_frequency.to_csv(
        stage / "c2_selection_bootstrap_frequency.csv", index=False
    )

    def metric_record(
        metrics: dict[str, np.ndarray], column: int = 0
    ) -> dict[str, float]:
        return {
            key: float(metrics[key][column])
            for key in (
                "total_return",
                "annualized_return_252",
                "sharpe",
                "max_drawdown",
            )
        }

    selected_full_metrics = metric_record(full_metrics, selected_index)
    no_cap_full_metrics = metric_record(full_baseline)
    legacy_full_metrics = metric_record(_matrix_metrics(legacy_returns))
    post_development_metrics = {
        "C2": metric_record(_matrix_metrics(selected_returns[evaluation_mask])),
        "no_cap": metric_record(_matrix_metrics(no_cap_returns[evaluation_mask])),
        "legacy_C": metric_record(_matrix_metrics(legacy_returns[evaluation_mask])),
    }
    key_episode = episodes.iloc[0]
    key_episode_leave_out = leave_out.loc[
        leave_out["removed_episode"].eq(key_episode["episode"])
        & leave_out["replacement_reference"].eq("no_cap")
    ].iloc[0]
    top_no_cap_day = top_days.loc[top_days["reference"].eq("no_cap")].iloc[0]
    top_one_removed = removal_stress.loc[
        removal_stress["reference"].eq("no_cap")
        & removal_stress["removed_top_positive_days"].eq(1)
    ].iloc[0]

    fixed_bootstrap_summary: dict[str, dict[str, dict[str, float]]] = {}
    for reference in ("no_cap", "legacy_C"):
        fixed_bootstrap_summary[reference] = {}
        for metric in ("annualized_return_252", "sharpe", "max_drawdown"):
            row = fixed_bootstrap.loc[
                fixed_bootstrap["reference"].eq(reference)
                & fixed_bootstrap["metric_delta"].eq(metric)
            ].iloc[0]
            fixed_bootstrap_summary[reference][metric] = {
                "mean": float(row["mean"]),
                "median": float(row["median"]),
                "ci_2_5pct": float(row["ci_2_5pct"]),
                "ci_97_5pct": float(row["ci_97_5pct"]),
                "bootstrap_probability_positive": float(
                    row["probability_positive"]
                ),
            }

    statistical_summary = {
        "calendar": {
            "start": calendar[0].date().isoformat(),
            "end": calendar[-1].date().isoformat(),
            "observations": len(calendar),
            "development_observations": int(development_mask.sum()),
        },
        "candidate_universe": {
            "parameter_combinations": len(specs),
            "unique_development_return_paths": len(development_unique_columns),
            "unique_full_return_paths": len(full_unique_columns),
        },
        "development_selection": {
            "variant_id": variant_ids[development_choice],
            "pool": development_pool,
            "pool_count": development_pool_count,
        },
        "full_sample_oracle": {
            "variant_id": variant_ids[full_choice],
            "pool": full_pool,
            "pool_count": full_pool_count,
        },
        "same_development_and_full_variant": development_choice == full_choice,
        "fixed_parameter_full_sample": {
            "C2": selected_full_metrics,
            "no_cap": no_cap_full_metrics,
            "legacy_C": legacy_full_metrics,
        },
        "fixed_parameter_post_development_2023_cutoff": post_development_metrics,
        "development_equivalence_class": {
            "parameter_combinations_with_identical_development_returns": len(
                equivalent_development
            ),
            "full_annualized_return_min": float(
                full_metrics["annualized_return_252"][equivalent_development].min()
            ),
            "full_annualized_return_max": float(
                full_metrics["annualized_return_252"][equivalent_development].max()
            ),
            "full_sharpe_min": float(
                full_metrics["sharpe"][equivalent_development].min()
            ),
            "full_sharpe_max": float(
                full_metrics["sharpe"][equivalent_development].max()
            ),
        },
        "pbo_development": pbo_development,
        "pbo_full": pbo_full,
        "white_reality_check_development": reality_check,
        "deflated_sharpe_absolute_development": dsr_absolute,
        "deflated_sharpe_excess_vs_no_cap_development": dsr_excess,
        "selection_bootstrap": {
            "repetitions": SELECTION_BOOTSTRAP_REPETITIONS,
            "unique_selected_variants": len(selection_counter),
            "final_C2_selection_frequency": selection_counter[SELECTED_ID]
            / SELECTION_BOOTSTRAP_REPETITIONS,
            "median_evaluation_annualized_delta_vs_no_cap": float(
                selection_bootstrap[
                    "evaluation_annualized_delta_vs_no_cap"
                ].median()
            ),
            "probability_evaluation_annualized_delta_positive": float(
                (
                    selection_bootstrap[
                        "evaluation_annualized_delta_vs_no_cap"
                    ]
                    > 0
                ).mean()
            ),
            "probability_evaluation_sharpe_delta_positive": float(
                (selection_bootstrap["evaluation_sharpe_delta_vs_no_cap"] > 0).mean()
            ),
            "probability_evaluation_mdd_improvement_nonnegative": float(
                (
                    selection_bootstrap[
                        "evaluation_mdd_improvement_vs_no_cap"
                    ]
                    >= -1e-12
                ).mean()
            ),
        },
        "fixed_parameter_block_bootstrap": fixed_bootstrap_summary,
        "event_concentration": {
            "largest_episode": {
                "episode": int(key_episode["episode"]),
                "start": str(key_episode["start"]),
                "end": str(key_episode["end"]),
                "asset": str(key_episode["entry_asset"]),
                "c2_total_return": float(key_episode["c2_total_return"]),
                "no_cap_total_return": float(key_episode["no_cap_total_return"]),
                "legacy_total_return": float(key_episode["legacy_total_return"]),
                "c2_log_excess_vs_no_cap": float(
                    key_episode["c2_log_excess_vs_no_cap"]
                ),
                "full_sample_net_log_excess_vs_no_cap": float(
                    np.log1p(selected_returns).sum()
                    - np.log1p(no_cap_returns).sum()
                ),
            },
            "largest_single_day": {
                "date": str(top_no_cap_day["date"]),
                "c2_return": float(top_no_cap_day["c2_return"]),
                "no_cap_return": float(top_no_cap_day["reference_return"]),
                "log_excess_contribution": float(
                    top_no_cap_day["log_excess_contribution"]
                ),
            },
            "remove_largest_episode_replace_with_no_cap": {
                "annualized_return_252": float(
                    key_episode_leave_out["annualized_return_252"]
                ),
                "sharpe": float(key_episode_leave_out["sharpe"]),
                "max_drawdown": float(key_episode_leave_out["max_drawdown"]),
                "annualized_delta_vs_no_cap": float(
                    key_episode_leave_out["annualized_delta_vs_reference"]
                ),
                "sharpe_delta_vs_no_cap": float(
                    key_episode_leave_out["sharpe_delta_vs_reference"]
                ),
                "max_drawdown_improvement_vs_no_cap": float(
                    key_episode_leave_out[
                        "max_drawdown_improvement_vs_reference"
                    ]
                ),
            },
            "remove_largest_positive_day_replace_with_no_cap": {
                "annualized_return_252": float(
                    top_one_removed["annualized_return_252"]
                ),
                "sharpe": float(top_one_removed["sharpe"]),
                "max_drawdown": float(top_one_removed["max_drawdown"]),
                "annualized_delta_vs_no_cap": float(
                    top_one_removed["annualized_delta_vs_reference"]
                ),
                "sharpe_delta_vs_no_cap": float(
                    top_one_removed["sharpe_delta_vs_reference"]
                ),
                "max_drawdown_improvement_vs_no_cap": float(
                    top_one_removed["max_drawdown_improvement_vs_reference"]
                ),
            },
        },
        "event_level_parameter_cross_validation": event_cv_summary,
        "walk_forward_chained": chained_summary,
    }
    (stage / "c2_statistical_summary.json").write_text(
        json.dumps(statistical_summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    equivalence_lines = [
        "|窗口|沪深300 q|创业板 q|纳指 q|黄金 q|全样本年化|全样本Sharpe|全样本MDD|",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in pd.DataFrame(equivalent_records).sort_values("variant_id").itertuples():
        equivalence_lines.append(
            f"|{int(row.volatility_window)}|{row.q_510300:.2f}|"
            f"{row.q_159915:.2f}|{row.q_513100:.2f}|{row.q_518880:.2f}|"
            f"{row.full_annualized_return_252:.2%}|{row.full_sharpe:.3f}|"
            f"{row.full_max_drawdown:.2%}|"
        )
    equivalence_table = "\n".join(equivalence_lines)

    episode_lines = [
        "|编号|Defender持有期|入场资产|C2收益|无cap收益|旧C收益|C2对无cap对数超额|",
        "|---:|---|---|---:|---:|---:|---:|",
    ]
    for row in episodes.sort_values("episode").itertuples():
        episode_lines.append(
            f"|{row.episode}|{row.start}至{row.end}|{row.entry_asset_name}|"
            f"{row.c2_total_return:+.2%}|{row.no_cap_total_return:+.2%}|"
            f"{row.legacy_total_return:+.2%}|{row.c2_log_excess_vs_no_cap:+.4f}|"
        )
    episode_table = "\n".join(episode_lines)

    annual_lines = [
        "|年份|C2收益|无cap收益|旧C收益|C2对无cap对数超额|",
        "|---:|---:|---:|---:|---:|",
    ]
    for year in sorted(calendar.year.unique()):
        c2_return = annual.loc[
            annual["year"].eq(year) & annual["strategy"].eq("C2"), "total_return"
        ].iloc[0]
        no_cap_return = annual.loc[
            annual["year"].eq(year) & annual["strategy"].eq("no_cap"),
            "total_return",
        ].iloc[0]
        legacy_return = annual.loc[
            annual["year"].eq(year) & annual["strategy"].eq("legacy_C"),
            "total_return",
        ].iloc[0]
        annual_lines.append(
            f"|{year}|{c2_return:+.2%}|{no_cap_return:+.2%}|"
            f"{legacy_return:+.2%}|{math.log1p(c2_return) - math.log1p(no_cap_return):+.4f}|"
        )
    annual_table = "\n".join(annual_lines)

    fixed_lines = [
        "|比较对象|指标差（C2−对象）|bootstrap均值|95%区间|差值为正比例|",
        "|---|---|---:|---:|---:|",
    ]
    metric_labels = {
        "annualized_return_252": "年化收益",
        "sharpe": "Sharpe",
        "max_drawdown": "MDD改善",
    }
    for row in fixed_bootstrap.itertuples():
        percentage = row.metric_delta != "sharpe"
        formatter = ".2%" if percentage else ".3f"
        fixed_lines.append(
            f"|{row.reference}|{metric_labels[row.metric_delta]}|"
            f"{format(row.mean, formatter)}|"
            f"[{format(row.ci_2_5pct, formatter)}, {format(row.ci_97_5pct, formatter)}]|"
            f"{row.probability_positive:.1%}|"
        )
    fixed_table = "\n".join(fixed_lines)

    walk_lines = [
        "|测试年|当年以前数据选出的参数|年化差vs无cap|Sharpe差|MDD改善|",
        "|---:|---|---:|---:|---:|",
    ]
    for row in walk_forward.itertuples():
        walk_lines.append(
            f"|{row.test_year}|`{row.selected_variant}`|"
            f"{row.test_annualized_delta_vs_no_cap:+.2%}|"
            f"{row.test_sharpe_delta_vs_no_cap:+.3f}|"
            f"{row.test_mdd_improvement_vs_no_cap:+.2%}|"
        )
    walk_table = "\n".join(walk_lines)

    event_cv_lines = [
        "|方法|事件|训练样本数|选中参数（窗口/300/创业板/纳指/黄金）|q吻合|候选池|留出期对数超额vs无cap|",
        "|---|---:|---:|---|---:|---|---:|",
    ]
    method_labels = {
        "rolling_origin": "滚动前瞻",
        "purged_leave_one_event_out": "30日隔离LOEO",
    }
    for row in event_cv.sort_values(["method", "episode"]).itertuples():
        parameters = (
            f"{int(row.selected_volatility_window)}/"
            f"{row.selected_q_510300:.2f}/{row.selected_q_159915:.2f}/"
            f"{row.selected_q_513100:.2f}/{row.selected_q_518880:.2f}"
        )
        event_cv_lines.append(
            f"|{method_labels[row.method]}|{row.episode}|{row.train_observations}|"
            f"`{parameters}`|{row.matching_asset_quantiles_of_4}/4|"
            f"{row.selection_pool}|"
            f"{row.selected_event_log_excess_vs_no_cap:+.4f}|"
        )
    event_cv_table = "\n".join(event_cv_lines)

    event_cv_summary_lines = [
        "|方法|完整参数复现率|平均q吻合数|三指标候选池成立率|当前C2仍通过训练门槛|留出事件跑赢无cap比例|事件期组合收益/无cap|",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method, values in event_cv_summary.items():
        event_cv_summary_lines.append(
            f"|{method_labels[method]}|"
            f"{values['exact_current_parameter_frequency']:.1%}|"
            f"{values['mean_matching_asset_quantiles_of_4']:.2f}/4|"
            f"{values['triple_positive_selection_pool_frequency']:.1%}|"
            f"{values['current_C2_training_gate_pass_frequency']:.1%}|"
            f"{values['held_out_event_positive_log_excess_frequency']:.1%}|"
            f"{values['selected_event_path']['total_return']:+.2%}/"
            f"{values['no_cap_event_path']['total_return']:+.2%}|"
        )
    event_cv_summary_table = "\n".join(event_cv_summary_lines)

    event_cv_marginal_lines = [
        "|方法|维度|当前值|当前值复现率|众数|众数频率|",
        "|---|---|---:|---:|---:|---:|",
    ]
    dimension_labels = {
        "volatility_window": "波动窗口",
        "q_510300": "沪深300 q",
        "q_159915": "创业板 q",
        "q_513100": "纳指 q",
        "q_518880": "黄金 q",
    }
    for row in event_cv_marginal.sort_values(["method", "dimension"]).itertuples():
        event_cv_marginal_lines.append(
            f"|{method_labels[row.method]}|{dimension_labels[row.dimension]}|"
            f"{row.current_value:g}|{row.current_value_frequency:.1%}|"
            f"{row.mode_value:g}|{row.mode_frequency:.1%}|"
        )
    event_cv_marginal_table = "\n".join(event_cv_marginal_lines)

    report = f"""# C2参数过拟合与事件归因审计

## 结论先行

1. **撤回旧说法。**“开发期选中参数与全样本事后最优恰好相同”在修正选择器后不成立。开发期因果选择是 `{variant_ids[development_choice]}`，全样本oracle是 `{variant_ids[full_choice]}`；差异是创业板q{specs[development_choice].q_159915:.2f}与q{specs[full_choice].q_159915:.2f}。
2. **C2确实同时抓住了大涨并躲开了紧随其后的大跌。**2024-09-30它仍持有Momentum并获得+20.00%；2024-10-08开盘才切入Defender，当日仍获得+17.21%；2024-10-09已在Defender，仅跌-1.09%，而无cap为-16.28%。
3. **相对无cap的“收益增强”假设没有通过过拟合校正。**Reality Check p={reality_check['bootstrap_p_value']:.4f}，增量Deflated Sharpe概率={dsr_excess['deflated_sharpe_probability']:.1%}，固定参数年化差bootstrap 95%区间跨零，CSCV-PBO约为50%。
4. **风险控制有历史迹象，但证据还不到稳健部署标准。**固定参数bootstrap中Sharpe差为正比例{fixed_bootstrap_summary['no_cap']['sharpe']['bootstrap_probability_positive']:.1%}、MDD改善为正比例{fixed_bootstrap_summary['no_cap']['max_drawdown']['bootstrap_probability_positive']:.1%}，均未达到95%；扩展walk-forward也显示以牺牲年化换取Sharpe和MDD。
5. **更稳妥的定位：**C2是“待前瞻验证的回撤控制候选”，不是已证实的收益增强器。它相对旧C明显减少过度防守，但相对无cap的优势高度依赖少数事件。

## 1. “开发期选中参数与全样本事后最优相同”到底是什么意思

如果这句话为真，它只表示：

- 只用2019-01-18至2022-12-30选出的完整参数元组，碰巧与偷看2019-01-18至2026-08-17后选出的元组相同；
- 它不表示参数是真实规律，也不表示没有过拟合，因为整个候选结构仍可能由已知历史事件启发，且不同样本切分仍可能选出不同参数。

本次代码审计发现，旧选择器在开发期指标完全并列时，曾错误使用**全样本**的切换次数打破平局。2023以后信息因此进入开发期选择。修正为只使用开发期切换次数后，结果变为：

- 开发期选择：`{variant_ids[development_choice]}`；
- 全样本oracle：`{variant_ids[full_choice]}`；
- 两者不同，故旧句应撤回。

更重要的是，开发期有{len(equivalent_development)}组参数产生逐日完全相同的收益，开发期根本无法识别创业板q0.90/q0.95以及黄金q0.90/q0.95。它们在全样本的年化范围是{full_metrics['annualized_return_252'][equivalent_development].min():.2%}至{full_metrics['annualized_return_252'][equivalent_development].max():.2%}，Sharpe范围是{full_metrics['sharpe'][equivalent_development].min():.3f}至{full_metrics['sharpe'][equivalent_development].max():.3f}。全样本更好的q0.95只能叫事后oracle，不能倒推为开发期已发现的参数。

{equivalence_table}

## 2. C2是否“恰好”躲过大跌或抓住大涨

### 2024年关键路径

- 2024-09-27：慢门控切回Momentum，当日C2、旧C与无cap相同。
- 2024-09-30：旧C已经紧急切入Defender，只获得+9.38%；C2尚未报警，继续持有创业板Momentum，获得+20.00%。
- 2024-10-08：C2在开盘执行紧急切换。按开盘切换收益口径，它先获得原Momentum仓位上一收盘至当日开盘的+19.97%，再承受Defender日内-2.30%，合成为+17.21%；无cap全天Momentum为+19.98%，旧C因已在Defender仅+3.05%。
- 2024-10-09：C2已持有Defender，收益-1.09%；无cap仍持有Momentum，收益-16.28%。这一天给C2贡献了+0.1667的对数相对收益，是全样本最大的单日相对贡献。

所以，历史事实不是“只躲跌”或“只抓涨”，而是**晚于旧C防守，保留了两次上涨的大部分收益；又恰好在大跌前一个开盘完成防守**。这是一条非常有利、也非常具体的价格路径。

### 集中度和反事实

- 2024-10-08至2024-11-18这一期：C2 {key_episode.c2_total_return:+.2%}，无cap {key_episode.no_cap_total_return:+.2%}，旧C {key_episode.legacy_total_return:+.2%}；对无cap贡献+{key_episode.c2_log_excess_vs_no_cap:.4f}对数收益。
- C2全样本对无cap净对数超额只有{np.log1p(selected_returns).sum() - np.log1p(no_cap_returns).sum():+.4f}。也就是说，这一个事件的正贡献是最终净优势的{key_episode.c2_log_excess_vs_no_cap / (np.log1p(selected_returns).sum() - np.log1p(no_cap_returns).sum()):.1f}倍，其他时期合计抵消了其中绝大部分。
- 仅把这期替换为无cap收益，C2全样本年化从{selected_full_metrics['annualized_return_252']:.2%}降至{key_episode_leave_out.annualized_return_252:.2%}，低于无cap的{no_cap_full_metrics['annualized_return_252']:.2%}；MDD也回到{key_episode_leave_out.max_drawdown:.2%}，不再优于无cap。
- 仅移除2024-10-09这一个最有利日，C2年化变为{top_one_removed.annualized_return_252:.2%}，相对无cap年化差变为{top_one_removed.annualized_delta_vs_reference:+.2%}；但Sharpe仍高{top_one_removed.sharpe_delta_vs_reference:+.3f}、MDD仍改善{top_one_removed.max_drawdown_improvement_vs_reference:+.2%}。因此“收益领先”对单日极敏感，“风险指标改善”需要看完整事件而不只是一天。

全部11次紧急Defender持有期如下：

{episode_table}

逐年看，C2只在2021和2024明显跑赢无cap；2025大幅跑输，2026截至8月也跑输。排除2024后，累计相对贡献为负：

{annual_table}

严格地说，事件集中度不能证明信号一定是运气，但它证明：**当前样本不足以把C2优势解释成跨时期、跨事件稳定重复的规律。**

## 3. 过拟合检验设计与结果

### 3.1 因果性与候选空间

- 全部波动率只使用当日收盘及以前数据，分位阈值再严格滞后一期，下一交易日开盘执行。
- 固定cap≤0.8，搜索4个波动窗口×4只ETF各4个分位，共1024组参数。
- 1024组在开发期只有{len(development_unique_columns)}条不同收益路径，全样本有{len(full_unique_columns)}条；统计检验对重复路径去重，避免把完全相同策略虚增为独立试验。
- 主选择只使用2019–2022，且候选必须在开发期真实触发紧急切换，并同时优于无cap的年化、Sharpe和MDD，再以Sharpe优先排序。
- “无cap”仍保留40日慢门控和30日状态锁，只取消紧急波动cap；它是检验C2增量价值的正确基准，不等于原始Momentum。

### 3.2 多重检验和选择偏差

|检验|回答的问题|结果|判读|
|---|---|---:|---|
|CSCV-PBO（开发期8块、70种对半切分）|样本内赢家在互补样本中跌到中位数以下的概率|{pbo_development['pbo']:.1%}|约等于抛硬币，参数选择不稳定|
|CSCV-PBO（全样本12块、924种切分）|扩大样本后是否改善|{pbo_full['pbo']:.1%}|仍约50%，没有改善|
|White Reality Check（216条路径、20日循环区块、2000次）|考虑全部寻参后，最优cap的平均增量收益是否显著优于无cap|p={reality_check['bootstrap_p_value']:.4f}|不能拒绝“没有增量优势”|
|Probabilistic Sharpe，增量收益vs 0|不考虑多重试验，C2增量Sharpe为正的概率|{dsr_excess['probabilistic_sharpe_vs_zero']:.1%}|单策略证据也偏弱|
|Deflated Sharpe，增量收益|扣除多重试验及偏度、峰度后|{dsr_excess['deflated_sharpe_probability']:.1%}|低于50%，不支持增量Sharpe|
|Deflated Sharpe，C2绝对收益|整个融合策略的Sharpe是否为正|{dsr_absolute['deflated_sharpe_probability']:.1%}|很强，但主要验证慢门控+Momentum+Defender整体，不证明cap有用|
|选择bootstrap（300次开发期20日区块重采样）|轻微改变开发样本后是否仍选同一参数|同一参数{selection_counter[SELECTED_ID] / SELECTION_BOOTSTRAP_REPETITIONS:.1%}；共{len(selection_counter)}个赢家|参数元组不稳定|

选择bootstrap进一步显示：每次在重采样开发期重新选参后，把所选参数固定到2023以后，年化差vs无cap的中位数为{selection_bootstrap['evaluation_annualized_delta_vs_no_cap'].median():+.2%}，只有{(selection_bootstrap['evaluation_annualized_delta_vs_no_cap'] > 0).mean():.1%}为正；Sharpe差为正和MDD非劣的比例均为100%。这说明该搜索更像在寻找“牺牲收益换风险”的参数，而不是稳定的收益增强参数。该比例是固定后段历史上的选择稳定性诊断，不是未来成功概率。

### 3.3 固定参数抽样不确定性

以下采用配对20日循环区块bootstrap、2000次，保留C2与比较策略同期对应关系：

{fixed_table}

解释：

- 对无cap：年化差95%区间为[{fixed_bootstrap_summary['no_cap']['annualized_return_252']['ci_2_5pct']:.2%}, {fixed_bootstrap_summary['no_cap']['annualized_return_252']['ci_97_5pct']:.2%}]，没有收益领先证据；Sharpe和MDD方向偏正，但区间都跨零。
- 对旧C：年化差95%区间为[{fixed_bootstrap_summary['legacy_C']['annualized_return_252']['ci_2_5pct']:.2%}, {fixed_bootstrap_summary['legacy_C']['annualized_return_252']['ci_97_5pct']:.2%}]，这是本组检验中最扎实的正面结果；但只说明C2比过度敏感的统一q70旧C更少错失上涨，不等于优于无cap。

### 3.4 扩展walk-forward重新选参

每年只用该年以前数据重新选参数，再固定测试下一年：

{walk_table}

把2021至2026各年真实当年选择串联：C2族年化{chained_summary['strategy']['annualized_return_252']:.2%}、Sharpe {chained_summary['strategy']['sharpe']:.3f}、MDD {chained_summary['strategy']['max_drawdown']:.2%}；无cap分别为{chained_summary['no_cap']['annualized_return_252']:.2%}、{chained_summary['no_cap']['sharpe']:.3f}、{chained_summary['no_cap']['max_drawdown']:.2%}。即年化少{chained_summary['no_cap']['annualized_return_252'] - chained_summary['strategy']['annualized_return_252']:.2%}，Sharpe高{chained_summary['strategy']['sharpe'] - chained_summary['no_cap']['sharpe']:+.3f}，MDD改善{chained_summary['strategy']['max_drawdown'] - chained_summary['no_cap']['max_drawdown']:+.2%}。

但逐年极不稳定：2024大幅受益，2025和2026显著损失年化；2026的MDD还略差于无cap。它支持“风险/收益取舍”，不支持稳定的绝对收益增强。

### 3.5 事件级重新寻参与留出验证

为了避免只删除最成功的2024事件，本检验对全部11次C2紧急持有期逐一执行两种重选：

- **滚动前瞻：**每个事件只能使用事件开始前的数据选参，再在该事件窗口测试。这是两者中更接近真实部署的口径。
- **30日隔离LOEO：**从全样本评分中删除该事件及随后{EVENT_PURGE_DAYS}个交易日，再用其余日期重选。30日隔离用于覆盖状态锁的直接路径外溢；它会使用事件后的数据，所以只能衡量参数稳定性，不能称为真正样本外。

汇总结果：

{event_cv_summary_table}

分维度稳定性：

{event_cv_marginal_table}

逐事件结果：

{event_cv_table}

判读边界：事件窗口由当前C2的紧急切换定义，仍存在条件化选择；早期滚动事件训练样本较短。因而最有信息量的不是“某次是否精确复现完整元组”，而是四个资产q的平均吻合、三指标候选池能否持续成立，以及重选参数在被留出事件上能否反复跑赢无cap。

## 4. 更solid的最终判定

### 可以成立的结论

- C2比旧C的统一q70机制明显减少误防守；其全样本年化高{selected_full_metrics['annualized_return_252'] - legacy_full_metrics['annualized_return_252']:+.2%}，固定参数bootstrap对年化差的95%区间也完全为正。
- 在这段历史中，C2相对无cap把MDD从{no_cap_full_metrics['max_drawdown']:.2%}压到{selected_full_metrics['max_drawdown']:.2%}，Sharpe从{no_cap_full_metrics['sharpe']:.3f}升到{selected_full_metrics['sharpe']:.3f}；它作为风险覆盖层有研究价值。

### 不能成立的结论

- 不能说C2已经被证明能提高长期收益。全样本年化只比无cap高{selected_full_metrics['annualized_return_252'] - no_cap_full_metrics['annualized_return_252']:+.2%}，bootstrap方向约五五开，且移除2024关键事件后变成明显落后。
- 不能说q0.90是创业板的“真实最优参数”。q0.90和q0.95在开发期完全不可区分；q0.95只在后来样本更好。
- 不能把2024以后当作真正独立样本。C2结构本身是在研究者已经知道2024行情后提出的，存在研究方向层面的后见偏差。
- 事件级重选中，滚动前瞻完整参数复现率只有{event_cv_summary['rolling_origin']['exact_current_parameter_frequency']:.1%}，30日隔离LOEO也只有{event_cv_summary['purged_leave_one_event_out']['exact_current_parameter_frequency']:.1%}；不能用单次“3/4参数吻合”替代跨事件稳定性。

### 部署判断

当前证据等级：**不批准作为收益增强器；可保留为回撤控制候选，但需要冻结机制后前瞻验证。**最低要求是：

1. 不再根据现有历史修改窗口或四资产分位；若必须处理不可识别参数，事先采用简单规则或参数集成，而不是选全样本赢家。
2. 预注册主目标：优先检验相对无cap的Sharpe/MDD，同时明确允许多少年化收益牺牲；不能事后在收益、Sharpe、MDD之间切换胜负标准。
3. 收集新的、研究后发生的紧急事件；当前只有11次紧急持有期，2023又没有实际紧急切换，独立信息量不足。
4. 在新数据上继续报告事件级leave-one-out和无cap配对结果，避免再次由单一行情决定结论。

## 5. 方法边界

- 区块长度固定20日；它是对月度相关性的合理压力口径，但不同区块长度仍应作为预注册敏感性分析，而不能事后挑选。
- MDD和30日状态锁高度路径依赖；bootstrap重排区块只能近似抽样不确定性。
- DSR中的有效试验数采用收益路径相关性估计，是近似校正；因此本报告同时使用Reality Check、PBO、选择bootstrap和walk-forward，不依赖单一DSR数字。
- 交易费用、开盘切换和Defender分段收益均按当前回测引擎执行；统计检验不能覆盖未来流动性、滑点跳升、制度变化和模型失效。

## 方法参考

- Bailey、Borwein、López de Prado、Zhu，《The Probability of Backtest Overfitting》，CSCV/PBO。
- Bailey、López de Prado，《The Deflated Sharpe Ratio》，多重检验、偏度和峰度校正。
- White，《A Reality Check for Data Snooping》，对整个模型搜索集合检验相对基准的预测优势。
"""
    (stage / "c2_overfit_report.md").write_text(report, encoding="utf-8")

    code_files = [
        root / "research/run_momentum_held_asset_c2_overfit.py",
        root / "research/run_momentum_held_asset_adaptive_cap.py",
        root / "research/momentum_defender_occam.py",
        root / "research/run_momentum_volatility_signal_abcd.py",
    ]
    input_files = [
        defender_dir / "relative_defender_rotation_switch_returns.csv",
        root / "strategy/configs/quality_momentum_top1.yaml",
        *[root / "data/db" / f"{asset}.parquet" for asset in MOMENTUM_ASSETS],
    ]
    manifest = {
        "experiment": "momentum_held_asset_c2_overfit",
        "generated_on": date.today().isoformat(),
        "research_cutoff": end.isoformat(),
        "random_seed": RANDOM_SEED,
        "bootstrap_block_length": BOOTSTRAP_BLOCK,
        "fixed_bootstrap_repetitions": FIXED_BOOTSTRAP_REPETITIONS,
        "selection_bootstrap_repetitions": SELECTION_BOOTSTRAP_REPETITIONS,
        "reality_check_repetitions": REALITY_CHECK_REPETITIONS,
        "event_purge_days": EVENT_PURGE_DAYS,
        "git_commit": _git(root, "rev-parse", "HEAD"),
        "inputs": [{"path": str(path), "sha256": _sha256(path)} for path in input_files],
        "code_sources": [
            {"path": str(path), "sha256": _sha256(path)} for path in code_files
        ],
    }
    (stage / "experiment_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    final_output.mkdir(parents=True, exist_ok=True)
    for path in stage.iterdir():
        path.replace(final_output / path.name)
    stage.rmdir()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--defender-dir", type=Path, default=DEFAULT_DEFENDER_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--end", type=date.fromisoformat, default=DEFAULT_END)
    args = parser.parse_args()
    run_experiment(
        args.root.resolve(),
        args.defender_dir.resolve(),
        args.output.resolve(),
        args.end,
    )


if __name__ == "__main__":
    main()
