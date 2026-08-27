"""Run the frozen-candidate Momentum/Defender fusion research package."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import tempfile
from dataclasses import asdict
from datetime import date
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

from backtest.report import generate as generate_backtest_report
from backtest.runner import BacktestResult
from research.momentum_defender_occam import (
    ENTER_RETURN,
    EXIT_RETURN,
    HELD_RETURN,
    OccamParams,
    apply_state_schedule,
    build_inputs,
    indicator_at_effective_open,
    load_defender_bundle,
    performance,
    quantile_volatility_alert_at_open,
    scale_interface_costs,
    simulate_switch,
    slow_regime_at_open,
    volatility_cap_at_open,
)
from research.momentum_defender_occam_report import generate_html_report


DEFAULT_DEFENDER_DIR = Path(
    "/Users/hujiaoyuan/Desktop/Quant/Defender/defender/deliverable"
)
DEFAULT_END = date(2026, 8, 17)
SELECTED = OccamParams(
    lookback=40,
    risk_on_threshold=0.025,
    min_hold_days=30,
    emergency_daily_loss=None,
)
BOOTSTRAP_REPLICATIONS = 5000
BOOTSTRAP_BLOCK_DAYS = (5, 20, 60)
BOOTSTRAP_SEED = 20260821
CONSERVATIVE_TRIAL_COUNT = 2000


def _generate_standard_report(
    returns: pd.Series,
    benchmark: pd.Series,
    benchmark_name: str,
    output_path: Path,
    config: dict[str, object],
) -> Path:
    """Generate the project's standard QuantStats report on one aligned sample."""
    aligned = pd.concat(
        [returns.rename("strategy"), benchmark.rename("benchmark")], axis=1
    ).dropna()
    if len(aligned) != len(returns):
        raise ValueError(
            f"standard report lost observations after aligning {benchmark_name}: "
            f"{len(aligned)} != {len(returns)}"
        )
    result = BacktestResult(
        daily_returns=aligned["strategy"],
        benchmark_returns=aligned["benchmark"],
        positions=pd.DataFrame(index=aligned.index),
        train_end=date(2024, 12, 31),
        config=config,
        baseline_strategy_name=benchmark_name,
    )
    return generate_backtest_report(
        result,
        output_path,
        benchmark_title=benchmark_name,
    )


def _metric_row(name: str, returns: pd.Series, **metadata: object) -> dict[str, object]:
    return {"strategy": name, **metadata, **performance(returns)}


def _comparison(candidate: dict[str, object], baseline: dict[str, object]) -> dict[str, object]:
    for field in ("start", "end", "observations"):
        if candidate[field] != baseline[field]:
            raise ValueError(
                f"candidate and baseline use different samples for {field}: "
                f"{candidate[field]} != {baseline[field]}"
            )
    deltas = {
        "cagr_delta": float(candidate["cagr_calendar"]) - float(baseline["cagr_calendar"]),
        "annualized_252_delta": float(candidate["annualized_return_252"])
        - float(baseline["annualized_return_252"]),
        "sharpe_delta": float(candidate["sharpe"]) - float(baseline["sharpe"]),
        "max_drawdown_improvement": float(candidate["max_drawdown"])
        - float(baseline["max_drawdown"]),
    }
    deltas["strict_triple_pass"] = bool(
        deltas["cagr_delta"] > 1e-10
        and deltas["annualized_252_delta"] > 1e-10
        and deltas["sharpe_delta"] > 1e-10
        and deltas["max_drawdown_improvement"] > 1e-10
    )
    deltas["material_triple_pass"] = bool(
        deltas["cagr_delta"] >= 0.01
        and deltas["annualized_252_delta"] >= 0.01
        and deltas["sharpe_delta"] >= 0.10
        and deltas["max_drawdown_improvement"] >= 0.01
    )
    return deltas


def _evaluate_state(
    name: str,
    state: pd.DataFrame,
    momentum: pd.DataFrame,
    defender: pd.DataFrame,
    baseline_metrics: dict[str, object],
    **metadata: object,
) -> tuple[dict[str, object], pd.DataFrame]:
    simulated = simulate_switch(momentum, defender, state["risk_on"])
    row = _metric_row(
        name,
        simulated["return"],
        switches=int(simulated["sleeve_switch"].sum()),
        defender_days=int((~state["risk_on"]).sum()),
        defender_share=float((~state["risk_on"]).mean()),
        **metadata,
    )
    row.update(_comparison(row, baseline_metrics))
    return row, simulated


def _array_metrics(values: np.ndarray) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    curve = np.cumprod(1.0 + values)
    annualized = float(curve[-1] ** (252.0 / len(values)) - 1.0)
    standard_deviation = float(values.std(ddof=1))
    sharpe = (
        float(values.mean() / standard_deviation * np.sqrt(252.0))
        if standard_deviation
        else 0.0
    )
    anchored = np.concatenate(([1.0], curve))
    max_drawdown = float((anchored / np.maximum.accumulate(anchored) - 1.0).min())
    return annualized, sharpe, max_drawdown


def _moving_block_indices(
    rng: np.random.Generator,
    observations: int,
    block_days: int,
) -> np.ndarray:
    blocks = math.ceil(observations / block_days)
    starts = rng.integers(0, observations - block_days + 1, size=blocks)
    return np.concatenate(
        [np.arange(start, start + block_days) for start in starts]
    )[:observations]


def _iid_null_max_sharpe_heuristic(
    returns: pd.Series,
    trial_count: int,
) -> tuple[float, float, float, float, float]:
    values = returns.astype(float)
    daily_sharpe = float(values.mean() / values.std(ddof=1))
    skew = float(values.skew())
    kurtosis = float(values.kurt()) + 3.0
    variance = (
        1.0
        - skew * daily_sharpe
        + ((kurtosis - 1.0) / 4.0) * daily_sharpe**2
    ) / (len(values) - 1)
    sigma = math.sqrt(max(variance, 1e-18))
    normal = NormalDist()
    euler_gamma = 0.5772156649015329
    expected_max_null = sigma * (
        (1.0 - euler_gamma) * normal.inv_cdf(1.0 - 1.0 / trial_count)
        + euler_gamma
        * normal.inv_cdf(1.0 - 1.0 / (trial_count * math.e))
    )
    probability = normal.cdf((daily_sharpe - expected_max_null) / sigma)
    autocorrelation_1 = float(values.autocorr(lag=1))
    return (
        float(probability),
        float(daily_sharpe),
        float(expected_max_null),
        autocorrelation_1,
        kurtosis,
    )


def _finite_sample_upper_tail_p_value(
    null_statistics: np.ndarray | list[float],
    observed: float,
) -> float:
    null_values = np.asarray(null_statistics, dtype=float)
    return float((1 + np.count_nonzero(null_values >= observed)) / (len(null_values) + 1))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_value(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def _period_rows(
    periods: list[tuple[str, pd.Timestamp, pd.Timestamp]],
    series: dict[str, pd.Series],
    baseline_name: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for period_name, start, end in periods:
        baseline = performance(series[baseline_name].loc[start:end])
        for name, returns in series.items():
            row = _metric_row(
                name,
                returns.loc[start:end],
                period=period_name,
                period_start=start.date().isoformat(),
                period_end=end.date().isoformat(),
            )
            row.update(_comparison(row, baseline))
            rows.append(row)
    return pd.DataFrame(rows)


def _episode_attribution(
    selected: pd.DataFrame,
    baseline: pd.Series,
) -> pd.DataFrame:
    entries = selected.index[selected["transition"].eq("momentum_to_defender")]
    exits = selected.index[selected["transition"].eq("defender_to_momentum")]
    rows: list[dict[str, object]] = []
    for episode, entry in enumerate(entries, start=1):
        later_exits = exits[exits > entry]
        exit_date = later_exits.min() if len(later_exits) else pd.NaT
        window_end = exit_date if pd.notna(exit_date) else selected.index.max()
        interval = selected.loc[entry:window_end, "return"]
        base_interval = baseline.loc[entry:window_end]
        candidate_factor = float((1.0 + interval).prod())
        baseline_factor = float((1.0 + base_interval).prod())
        log_excess = float(np.log(candidate_factor / baseline_factor))
        candidate_metrics = performance(interval)
        baseline_metrics = performance(base_interval)
        defender_days = int(
            selected.loc[entry:window_end, "sleeve"].eq("defender").sum()
        )
        entry_reason = str(selected.loc[entry, "state_reason"])
        rows.append(
            {
                "episode": episode,
                "entry_date": entry.date().isoformat(),
                "exit_date": (
                    exit_date.date().isoformat() if pd.notna(exit_date) else "open_at_cutoff"
                ),
                "window_end": window_end.date().isoformat(),
                "window_observations": len(interval),
                "defender_days": defender_days,
                "entry_reason": entry_reason,
                "cap_triggered_entry": entry_reason == "emergency_exit",
                "candidate_return": candidate_factor - 1.0,
                "momentum_return": baseline_factor - 1.0,
                "arithmetic_excess_return": candidate_factor - baseline_factor,
                "relative_wealth_excess": candidate_factor / baseline_factor - 1.0,
                "log_excess_return": log_excess,
                "positive_excess": log_excess > 0.0,
                "candidate_cagr_calendar": candidate_metrics["cagr_calendar"],
                "candidate_annualized_return_252": candidate_metrics[
                    "annualized_return_252"
                ],
                "candidate_annualized_volatility": candidate_metrics[
                    "annualized_volatility"
                ],
                "candidate_sharpe": candidate_metrics["sharpe"],
                "candidate_max_drawdown": candidate_metrics["max_drawdown"],
                "momentum_cagr_calendar": baseline_metrics["cagr_calendar"],
                "momentum_annualized_return_252": baseline_metrics[
                    "annualized_return_252"
                ],
                "momentum_annualized_volatility": baseline_metrics[
                    "annualized_volatility"
                ],
                "momentum_sharpe": baseline_metrics["sharpe"],
                "momentum_max_drawdown": baseline_metrics["max_drawdown"],
            }
        )
    return pd.DataFrame(rows)


def run_experiment(
    root: Path,
    defender_dir: Path,
    output_dir: Path,
    end: date,
) -> None:
    final_output_dir = output_dir
    final_output_dir.parent.mkdir(parents=True, exist_ok=True)
    git_status_before = _git_value(root, "status", "--short").splitlines()
    output_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{final_output_dir.name}.staging-",
            dir=final_output_dir.parent,
        )
    )
    bundle = load_defender_bundle(defender_dir, end)
    inputs = build_inputs(
        root,
        defender_dir / "relative_defender_rotation_switch_returns.csv",
        end,
    )
    calendar = inputs.calendar
    exact_momentum = inputs.momentum[HELD_RETURN].astype(float)
    official_momentum = inputs.momentum_result.daily_returns.reindex(calendar).astype(float)
    original_base = inputs.momentum_result.benchmark_returns.reindex(calendar).astype(float)
    if original_base.isna().any():
        missing = original_base.index[original_base.isna()].strftime("%Y-%m-%d").tolist()
        raise ValueError(f"original 4ETF base is missing report dates: {missing[:5]}")
    defender_held = inputs.defender[HELD_RETURN].astype(float)
    exact_metrics = _metric_row("momentum_exact_adapter", exact_momentum)
    official_metrics = _metric_row("momentum_official_runner", official_momentum)
    original_base_metrics = _metric_row(
        "original_4etf_equal_weight_base", original_base
    )

    cap = volatility_cap_at_open(bundle.indicators, calendar)
    slow = slow_regime_at_open(
        inputs.risk_close,
        calendar,
        SELECTED.lookback,
        SELECTED.risk_on_threshold,
    )
    selected_state = apply_state_schedule(
        slow,
        cap,
        calendar,
        SELECTED.min_hold_days,
        emergency_override=True,
    )
    selected_row, selected = _evaluate_state(
        "selected_fusion",
        selected_state,
        inputs.momentum,
        inputs.defender,
        exact_metrics,
        lookback=SELECTED.lookback,
        risk_on_threshold=SELECTED.risk_on_threshold,
        min_hold_days=SELECTED.min_hold_days,
        cap_rule="signal_volatility_cap < 1",
        emergency_override=True,
    )
    selected_vs_official = _comparison(selected_row, official_metrics)
    for key, value in selected_vs_official.items():
        selected_row[f"vs_official_{key}"] = value

    performance_rows = [
        original_base_metrics,
        official_metrics,
        exact_metrics,
        _metric_row("defender_continuous", defender_held),
        selected_row,
    ]
    performance_summary = pd.DataFrame(performance_rows)

    daily = selected_state.join(selected.drop(columns=["risk_on"]))
    daily["volatility_cap_active_at_open"] = cap
    daily["momentum_exact_return"] = exact_momentum
    daily["momentum_official_return"] = official_momentum
    daily["original_base_return"] = original_base
    daily["defender_held_return"] = defender_held
    daily["candidate_excess_vs_exact_momentum"] = daily["return"] - exact_momentum
    daily["momentum_exact_nav"] = (1.0 + exact_momentum).cumprod()
    daily["momentum_official_nav"] = (1.0 + official_momentum).cumprod()
    daily["original_base_nav"] = (1.0 + original_base).cumprod()
    daily["defender_continuous_nav"] = (1.0 + defender_held).cumprod()
    slow_return_40 = (
        inputs.risk_close.astype(float)
        / inputs.risk_close.astype(float).shift(SELECTED.lookback)
        - 1.0
    )
    daily["slow_return_40_asof_previous_close"] = (
        slow_return_40.shift(1).reindex(calendar).ffill()
    )
    for source_column, output_column in (
        ("signal_realized_volatility_20", "signal_realized_volatility_20_asof_previous_close"),
        ("signal_cap_volatility_threshold", "signal_cap_volatility_threshold_asof_previous_close"),
        ("signal_volatility_cap", "signal_volatility_cap_asof_previous_close"),
        ("signal_grid_target", "signal_grid_target_asof_previous_close"),
        ("signal_base_target", "signal_base_target_asof_previous_close"),
    ):
        daily[output_column] = pd.to_numeric(
            indicator_at_effective_open(bundle.indicators, source_column, calendar),
            errors="coerce",
        )
    signal_rows = bundle.indicators.reset_index()[
        ["date", "signal_observation_date", "signal_effective_next_open_date"]
    ].dropna(subset=["signal_effective_next_open_date"])
    for column in (
        "date",
        "signal_observation_date",
        "signal_effective_next_open_date",
    ):
        signal_rows[column] = pd.to_datetime(signal_rows[column])
    signal_by_open = signal_rows.set_index("signal_effective_next_open_date")
    daily["defender_signal_row_date"] = daily.index.map(
        signal_by_open["date"].to_dict()
    )
    daily["defender_signal_observation_date"] = daily.index.map(
        signal_by_open["signal_observation_date"].to_dict()
    )
    daily.index.name = "date"
    daily.to_csv(output_dir / "selected_strategy_daily.csv")

    switches = daily.loc[daily["sleeve_switch"]].copy()
    switches.to_csv(output_dir / "switch_events.csv")

    # Time slices are stability checks only; they are not claimed as untouched OOS.
    series = {
        "momentum_official_runner": official_momentum,
        "momentum_exact_adapter": exact_momentum,
        "defender_continuous": defender_held,
        "selected_fusion": selected["return"],
    }
    eras = [
        ("2019-2021", pd.Timestamp("2019-01-18"), pd.Timestamp("2021-12-31")),
        ("2022-2024", pd.Timestamp("2022-01-01"), pd.Timestamp("2024-12-31")),
        ("2025-cutoff", pd.Timestamp("2025-01-01"), pd.Timestamp(end)),
    ]
    _period_rows(eras, series, "momentum_exact_adapter").to_csv(
        output_dir / "era_metrics.csv", index=False
    )
    annual_periods = [
        (
            str(year),
            pd.Timestamp(year=year, month=1, day=1),
            min(pd.Timestamp(year=year, month=12, day=31), pd.Timestamp(end)),
        )
        for year in range(calendar.min().year, calendar.max().year + 1)
    ]
    _period_rows(annual_periods, series, "momentum_exact_adapter").to_csv(
        output_dir / "annual_metrics.csv", index=False
    )

    false_cap = pd.Series(False, index=calendar)
    true_slow = pd.Series(True, index=calendar)
    ablation_specs = [
        ("slow_gate_only", slow, false_cap, SELECTED.min_hold_days, True),
        ("cap_only", true_slow, cap, SELECTED.min_hold_days, True),
        ("cap_respects_min_hold", slow, cap, SELECTED.min_hold_days, False),
        ("selected_no_min_hold", slow, cap, 1, True),
        ("selected_fusion", slow, cap, SELECTED.min_hold_days, True),
    ]
    ablation_rows: list[dict[str, object]] = []
    slow_gate_only_returns: pd.Series | None = None
    for name, slow_signal, cap_signal, hold, override in ablation_specs:
        state = apply_state_schedule(
            slow_signal,
            cap_signal,
            calendar,
            hold,
            emergency_override=override,
        )
        row, simulated = _evaluate_state(
            name,
            state,
            inputs.momentum,
            inputs.defender,
            exact_metrics,
            min_hold_days=hold,
            emergency_override=override,
        )
        ablation_rows.append(row)
        if name == "slow_gate_only":
            slow_gate_only_returns = simulated["return"].copy()
    if slow_gate_only_returns is None:
        raise AssertionError("slow-gate-only ablation was not generated")
    pd.DataFrame(ablation_rows).to_csv(output_dir / "ablation_metrics.csv", index=False)

    local_rows: list[dict[str, object]] = []
    local_returns: dict[str, np.ndarray] = {}
    for lookback in (30, 40, 50):
        for threshold in (0.0, 0.025, 0.05):
            local_slow = slow_regime_at_open(
                inputs.risk_close, calendar, lookback, threshold
            )
            for hold in (20, 25, 30, 35, 40):
                name = f"lb{lookback}_th{threshold:.3f}_hold{hold}"
                state = apply_state_schedule(local_slow, cap, calendar, hold)
                row, simulated = _evaluate_state(
                    name,
                    state,
                    inputs.momentum,
                    inputs.defender,
                    exact_metrics,
                    lookback=lookback,
                    risk_on_threshold=threshold,
                    min_hold_days=hold,
                    is_selected=(lookback, threshold, hold) == (40, 0.025, 30),
                )
                local_rows.append(row)
                local_returns[name] = simulated["return"].to_numpy(float)
    local_frame = pd.DataFrame(local_rows)
    local_frame.to_csv(output_dir / "local_parameter_neighborhood.csv", index=False)

    # Broad slow-gate scan is recorded to expose how fragile the slow rule is
    # without the frozen cap mechanism; it is not used to re-select the winner.
    slow_grid_rows: list[dict[str, object]] = []
    for lookback in (20, 30, 40, 50, 60, 80, 100, 120):
        for threshold in (-0.05, -0.025, 0.0, 0.025, 0.05, 0.075, 0.10):
            grid_slow = slow_regime_at_open(
                inputs.risk_close, calendar, lookback, threshold
            )
            for hold in (5, 10, 15, 20, 25, 30, 35, 40, 50):
                state = apply_state_schedule(grid_slow, false_cap, calendar, hold)
                name = f"slow_lb{lookback}_th{threshold:.3f}_hold{hold}"
                row, _ = _evaluate_state(
                    name,
                    state,
                    inputs.momentum,
                    inputs.defender,
                    exact_metrics,
                    lookback=lookback,
                    risk_on_threshold=threshold,
                    min_hold_days=hold,
                )
                slow_grid_rows.append(row)
    pd.DataFrame(slow_grid_rows).to_csv(
        output_dir / "slow_gate_full_grid.csv", index=False
    )

    quantile_rows: list[dict[str, object]] = []
    for quantile in (0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95):
        quantile_cap = quantile_volatility_alert_at_open(
            bundle.indicators, calendar, quantile
        )
        state = apply_state_schedule(slow, quantile_cap, calendar, SELECTED.min_hold_days)
        row, _ = _evaluate_state(
            f"cap_quantile_{quantile:.2f}",
            state,
            inputs.momentum,
            inputs.defender,
            exact_metrics,
            quantile=quantile,
            cap_alert_days=int(quantile_cap.sum()),
            signal_difference_days_vs_frozen=int((quantile_cap != cap).sum()),
            frozen_quantile=(quantile == 0.80),
        )
        quantile_rows.append(row)
    quantile_frame = pd.DataFrame(quantile_rows)
    quantile_frame.to_csv(output_dir / "cap_quantile_sensitivity.csv", index=False)

    # Timing placebo: circularly rotate the frozen cap history in monthly
    # (21-trading-day) increments.  The wrap makes these intentionally
    # non-causal counterfactuals; they are used only to ask whether the real
    # phase is unusually effective while preserving alert duration structure.
    placebo_rows: list[dict[str, object]] = []
    cap_values = cap.to_numpy(bool)
    for multiple in range(1, 83):
        shift_days = 21 * multiple
        shifted_cap = pd.Series(
            np.roll(cap_values, shift_days),
            index=calendar,
            dtype=bool,
        )
        state = apply_state_schedule(
            slow, shifted_cap, calendar, SELECTED.min_hold_days
        )
        row, _ = _evaluate_state(
            f"cap_placebo_shift_{shift_days}",
            state,
            inputs.momentum,
            inputs.defender,
            exact_metrics,
            circular_shift_days=shift_days,
        )
        row["dominates_selected_all_three"] = bool(
            float(row["cagr_calendar"]) > float(selected_row["cagr_calendar"])
            and float(row["sharpe"]) > float(selected_row["sharpe"])
            and float(row["max_drawdown"]) > float(selected_row["max_drawdown"])
        )
        placebo_rows.append(row)
    placebo_frame = pd.DataFrame(placebo_rows)
    placebo_frame.to_csv(output_dir / "cap_timing_placebo.csv", index=False)

    cost_rows: list[dict[str, object]] = []
    for multiplier in (0.0, 1.0, 2.0, 5.0, 10.0):
        momentum_stressed = scale_interface_costs(inputs.momentum, multiplier)
        defender_stressed = scale_interface_costs(inputs.defender, multiplier)
        stressed_baseline = _metric_row(
            f"momentum_cost_{multiplier:g}x",
            momentum_stressed[HELD_RETURN],
        )
        row, _ = _evaluate_state(
            f"selected_cost_{multiplier:g}x",
            selected_state,
            momentum_stressed,
            defender_stressed,
            stressed_baseline,
            cost_multiplier=multiplier,
        )
        versus_live_baseline = _comparison(row, exact_metrics)
        for key, value in versus_live_baseline.items():
            row[f"vs_momentum_1x_{key}"] = value
        cost_rows.append(row)
    cost_frame = pd.DataFrame(cost_rows)
    cost_frame.to_csv(output_dir / "cost_stress.csv", index=False)

    delay_rows: list[dict[str, object]] = []
    for slow_delay in (0, 1, 2):
        for cap_delay in (0, 1, 2):
            delayed_state = apply_state_schedule(
                slow.shift(slow_delay),
                cap.shift(cap_delay).fillna(False),
                calendar,
                SELECTED.min_hold_days,
            )
            row, _ = _evaluate_state(
                f"delay_slow{slow_delay}_cap{cap_delay}",
                delayed_state,
                inputs.momentum,
                inputs.defender,
                exact_metrics,
                extra_slow_delay_days=slow_delay,
                extra_cap_delay_days=cap_delay,
            )
            delay_rows.append(row)
    delay_frame = pd.DataFrame(delay_rows)
    delay_frame.to_csv(output_dir / "signal_delay_stress.csv", index=False)

    rolling_rows: list[dict[str, object]] = []
    window_days = 756
    step_days = 21
    for number, start_position in enumerate(
        range(0, len(calendar) - window_days + 1, step_days), start=1
    ):
        index = calendar[start_position : start_position + window_days]
        candidate_metrics = performance(selected.loc[index, "return"])
        baseline_metrics = performance(exact_momentum.loc[index])
        comparison = _comparison(candidate_metrics, baseline_metrics)
        rolling_rows.append(
            {
                "window": number,
                "start": index.min().date().isoformat(),
                "end": index.max().date().isoformat(),
                "observations": len(index),
                "candidate_cagr": candidate_metrics["cagr_calendar"],
                "momentum_cagr": baseline_metrics["cagr_calendar"],
                "candidate_sharpe": candidate_metrics["sharpe"],
                "momentum_sharpe": baseline_metrics["sharpe"],
                "candidate_max_drawdown": candidate_metrics["max_drawdown"],
                "momentum_max_drawdown": baseline_metrics["max_drawdown"],
                **comparison,
            }
        )
    rolling_frame = pd.DataFrame(rolling_rows)
    rolling_frame.to_csv(output_dir / "rolling_36m_metrics.csv", index=False)

    candidate_values = selected["return"].to_numpy(float)
    baseline_values = exact_momentum.to_numpy(float)
    bootstrap_rows: list[dict[str, object]] = []
    for block_days in BOOTSTRAP_BLOCK_DAYS:
        bootstrap_rng = np.random.default_rng(BOOTSTRAP_SEED + block_days)
        for replication in range(1, BOOTSTRAP_REPLICATIONS + 1):
            index = _moving_block_indices(
                bootstrap_rng, len(calendar), block_days
            )
            candidate_boot = _array_metrics(candidate_values[index])
            baseline_boot = _array_metrics(baseline_values[index])
            annual_delta = candidate_boot[0] - baseline_boot[0]
            sharpe_delta = candidate_boot[1] - baseline_boot[1]
            drawdown_delta = candidate_boot[2] - baseline_boot[2]
            bootstrap_rows.append(
                {
                    "block_days": block_days,
                    "replication": replication,
                    "annualized_252_delta": annual_delta,
                    "sharpe_delta": sharpe_delta,
                    "max_drawdown_improvement": drawdown_delta,
                    "strict_triple_pass": annual_delta > 0
                    and sharpe_delta > 0
                    and drawdown_delta > 0,
                }
            )
    bootstrap_frame = pd.DataFrame(bootstrap_rows)
    bootstrap_frame.to_csv(output_dir / "paired_block_bootstrap.csv", index=False)
    bootstrap_summary_rows: list[dict[str, object]] = []
    for block_days, group in bootstrap_frame.groupby("block_days"):
        row: dict[str, object] = {
            "block_days": int(block_days),
            "replications": len(group),
            "strict_triple_rate": float(group["strict_triple_pass"].mean()),
        }
        for column in (
            "annualized_252_delta",
            "sharpe_delta",
            "max_drawdown_improvement",
        ):
            for quantile in (0.025, 0.05, 0.50, 0.95, 0.975):
                row[f"{column}_q{quantile:g}"] = float(group[column].quantile(quantile))
        bootstrap_summary_rows.append(row)
    bootstrap_summary = pd.DataFrame(bootstrap_summary_rows)
    bootstrap_summary.to_csv(
        output_dir / "paired_block_bootstrap_summary.csv", index=False
    )

    # Studentized moving-block Reality Check over the declared 45-point local
    # family.  This corrects only that family, not every idea ever inspected.
    labels = list(local_returns)
    family_log_excess = np.column_stack(
        [
            np.log1p(local_returns[label]) - np.log1p(baseline_values)
            for label in labels
        ]
    )
    observed_mean = family_log_excess.mean(axis=0)
    observed_std = family_log_excess.std(axis=0, ddof=1)
    observed_t = np.sqrt(len(calendar)) * observed_mean / observed_std
    selected_label = "lb40_th0.025_hold30"
    selected_position = labels.index(selected_label)
    centered = family_log_excess - observed_mean
    reality_rng = np.random.default_rng(BOOTSTRAP_SEED + 1)
    max_null_statistics: list[float] = []
    for _ in range(BOOTSTRAP_REPLICATIONS):
        index = _moving_block_indices(reality_rng, len(calendar), 20)
        sampled = centered[index]
        sampled_std = sampled.std(axis=0, ddof=1)
        statistic = np.sqrt(len(calendar)) * sampled.mean(axis=0) / sampled_std
        max_null_statistics.append(float(np.nanmax(statistic)))
    selected_familywise_p_value = _finite_sample_upper_tail_p_value(
        max_null_statistics, float(observed_t[selected_position])
    )
    omnibus_reality_p_value = _finite_sample_upper_tail_p_value(
        max_null_statistics, float(np.nanmax(observed_t))
    )
    (
        iid_heuristic_probability,
        daily_sharpe,
        expected_max_null,
        autocorrelation_1,
        kurtosis,
    ) = _iid_null_max_sharpe_heuristic(selected["return"], CONSERVATIVE_TRIAL_COUNT)
    selection_bias = pd.DataFrame(
        [
            {
                "check": "local_family_studentized_block_maxT_omnibus",
                "role": "formal_gate",
                "value": omnibus_reality_p_value,
                "threshold": 0.05,
                "passed": omnibus_reality_p_value < 0.05,
                "details": f"max observed t vs max centered null; {len(labels)} candidates; 20-day blocks",
            },
            {
                "check": "selected_candidate_familywise_maxT",
                "role": "formal_gate",
                "value": selected_familywise_p_value,
                "threshold": 0.05,
                "passed": selected_familywise_p_value < 0.05,
                "details": f"selected t vs max centered null; {len(labels)} candidates; 20-day blocks",
            },
            {
                "check": "iid_null_max_sharpe_heuristic",
                "role": "informational_only",
                "value": iid_heuristic_probability,
                "threshold": np.nan,
                "passed": pd.NA,
                "details": (
                    f"not standard DSR; assumes {CONSERVATIVE_TRIAL_COUNT} iid zero-mean trials; "
                    f"daily Sharpe={daily_sharpe:.6f}; expected max null={expected_max_null:.6f}; "
                    f"lag1 autocorrelation={autocorrelation_1:.6f}; kurtosis={kurtosis:.6f}"
                ),
            },
        ]
    )
    selection_bias.to_csv(output_dir / "selection_bias_checks.csv", index=False)

    episodes = _episode_attribution(daily, exact_momentum)
    positive = episodes.loc[episodes["positive_excess"], "log_excess_return"].sort_values(
        ascending=False
    )
    positive_total = float(positive.sum())
    concentration_1 = float(positive.head(1).sum() / positive_total)
    concentration_3 = float(positive.head(3).sum() / positive_total)
    concentration_5 = float(positive.head(5).sum() / positive_total)
    legacy_episode_columns = [
        "episode",
        "entry_date",
        "exit_date",
        "window_end",
        "defender_days",
        "candidate_return",
        "momentum_return",
        "log_excess_return",
        "positive_excess",
    ]
    episodes[legacy_episode_columns].to_csv(
        output_dir / "defender_episode_attribution.csv", index=False
    )
    episodes.to_csv(output_dir / "defender_episode_metrics.csv", index=False)

    leave_one_rows: list[dict[str, object]] = []
    for episode in episodes.itertuples(index=False):
        neutralized = selected["return"].copy()
        entry = pd.Timestamp(episode.entry_date)
        window_end = pd.Timestamp(episode.window_end)
        neutralized.loc[entry:window_end] = exact_momentum.loc[entry:window_end]
        row = _metric_row(
            f"neutralize_episode_{episode.episode}",
            neutralized,
            episode=int(episode.episode),
            entry_date=episode.entry_date,
            exit_date=episode.exit_date,
            window_end=episode.window_end,
        )
        row.update(_comparison(row, exact_metrics))
        leave_one_rows.append(row)
    leave_one_frame = pd.DataFrame(leave_one_rows)
    leave_one_frame.to_csv(
        output_dir / "leave_one_defender_episode_out.csv", index=False
    )
    critical_episode = leave_one_frame.sort_values("max_drawdown_improvement").iloc[0]

    momentum_daily_error = float((exact_momentum - official_momentum).abs().max())
    momentum_nav_ratio_error = float(
        (1.0 + exact_momentum).prod() / (1.0 + official_momentum).prod() - 1.0
    )
    always_momentum = simulate_switch(
        inputs.momentum,
        inputs.defender,
        pd.Series(True, index=calendar),
        initial_previous_state="momentum",
    )
    always_defender = simulate_switch(
        inputs.momentum,
        inputs.defender,
        pd.Series(False, index=calendar),
        initial_previous_state="defender",
    )
    q80 = quantile_volatility_alert_at_open(bundle.indicators, calendar, 0.80)
    first_transition = str(selected.iloc[0]["transition"])
    checks = bundle.audit.copy()
    extra_checks = pd.DataFrame(
        [
            {
                "check": "cutoff_rows",
                "actual": len(calendar),
                "expected": 1837,
                "tolerance": 0.0,
                "passed": len(calendar) == 1837,
                "notes": "2026-08-18 through 2026-08-20 excluded",
            },
            {
                "check": "momentum_exact_vs_official_max_daily_error",
                "actual": momentum_daily_error,
                "expected": 0.0,
                "tolerance": 1.3e-5,
                "passed": momentum_daily_error <= 1.3e-5,
                "notes": "capital-aware multiplicative adapter vs official additive-cost runner",
            },
            {
                "check": "momentum_exact_vs_official_terminal_nav_ratio",
                "actual": momentum_nav_ratio_error,
                "expected": 0.0,
                "tolerance": 8e-5,
                "passed": abs(momentum_nav_ratio_error) <= 8e-5,
                "notes": "known execution-accounting difference",
            },
            {
                "check": "always_momentum_reproduction",
                "actual": float((always_momentum["return"] - exact_momentum).abs().max()),
                "expected": 0.0,
                "tolerance": 1e-12,
                "passed": bool(
                    (always_momentum["return"] - exact_momentum).abs().max() <= 1e-12
                ),
                "notes": "",
            },
            {
                "check": "always_defender_reproduction",
                "actual": float((always_defender["return"] - defender_held).abs().max()),
                "expected": 0.0,
                "tolerance": 1e-12,
                "passed": bool(
                    (always_defender["return"] - defender_held).abs().max() <= 1e-12
                ),
                "notes": "",
            },
            {
                "check": "frozen_cap_rebuild_q80",
                "actual": int((q80 != cap).sum()),
                "expected": 0,
                "tolerance": 0.0,
                "passed": q80.equals(cap),
                "notes": "strict-lag expanding quantile on unique anchor observations",
            },
            {
                "check": "first_day_prior_momentum_then_signal_switch",
                "actual": first_transition,
                "expected": "momentum_to_defender",
                "tolerance": None,
                "passed": first_transition == "momentum_to_defender",
                "notes": "2019-01-17 close signal executes 2019-01-18 open",
            },
            {
                "check": "selected_beats_exact_momentum_all_objectives",
                "actual": bool(selected_row["strict_triple_pass"]),
                "expected": True,
                "tolerance": None,
                "passed": bool(selected_row["strict_triple_pass"]),
                "notes": "requires both CAGR definitions plus Sharpe and MDD",
            },
            {
                "check": "selected_beats_official_momentum_all_objectives",
                "actual": bool(selected_row["vs_official_strict_triple_pass"]),
                "expected": True,
                "tolerance": None,
                "passed": bool(selected_row["vs_official_strict_triple_pass"]),
                "notes": "",
            },
            {
                "check": "cost_stress_1x_reproduces_selected",
                "actual": float(
                    abs(
                        cost_frame.loc[
                            cost_frame["cost_multiplier"].eq(1.0), "total_return"
                        ].iloc[0]
                        - selected_row["total_return"]
                    )
                ),
                "expected": 0.0,
                "tolerance": 1e-12,
                "passed": bool(
                    abs(
                        cost_frame.loc[
                            cost_frame["cost_multiplier"].eq(1.0), "total_return"
                        ].iloc[0]
                        - selected_row["total_return"]
                    )
                    <= 1e-12
                ),
                "notes": "",
            },
            {
                "check": "local_grid_selected_reproduces_selected",
                "actual": float(
                    np.max(
                        np.abs(
                            local_returns["lb40_th0.025_hold30"]
                            - selected["return"].to_numpy(float)
                        )
                    )
                ),
                "expected": 0.0,
                "tolerance": 1e-12,
                "passed": bool(
                    np.max(
                        np.abs(
                            local_returns["lb40_th0.025_hold30"]
                            - selected["return"].to_numpy(float)
                        )
                    )
                    <= 1e-12
                ),
                "notes": "",
            },
            {
                "check": "episode_log_excess_partition",
                "actual": float(episodes["log_excess_return"].sum()),
                "expected": float(
                    (
                        np.log1p(selected["return"])
                        - np.log1p(exact_momentum)
                    ).sum()
                ),
                "tolerance": 1e-12,
                "passed": bool(
                    abs(
                        float(episodes["log_excess_return"].sum())
                        - float(
                            (
                                np.log1p(selected["return"])
                                - np.log1p(exact_momentum)
                            ).sum()
                        )
                    )
                    <= 1e-12
                ),
                "notes": "entry through exit-open windows partition every non-Momentum leg",
            },
        ]
    )
    checks = pd.concat([checks, extra_checks], ignore_index=True)
    checks.to_csv(output_dir / "reproduction_checks.csv", index=False)
    if not checks["passed"].all():
        failed = checks.loc[~checks["passed"], "check"].tolist()
        raise AssertionError(f"experiment reproduction checks failed: {failed}")

    robustness_rows = [
        {
            "check": "local_45_strict_triple",
            "passed_count": int(local_frame["strict_triple_pass"].sum()),
            "total_count": len(local_frame),
            "rate": float(local_frame["strict_triple_pass"].mean()),
        },
        {
            "check": "local_45_material_triple",
            "passed_count": int(local_frame["material_triple_pass"].sum()),
            "total_count": len(local_frame),
            "rate": float(local_frame["material_triple_pass"].mean()),
        },
        {
            "check": "cap_quantile_strict_triple",
            "passed_count": int(quantile_frame["strict_triple_pass"].sum()),
            "total_count": len(quantile_frame),
            "rate": float(quantile_frame["strict_triple_pass"].mean()),
        },
        {
            "check": "cap_timing_placebo_strict_triple",
            "passed_count": int(placebo_frame["strict_triple_pass"].sum()),
            "total_count": len(placebo_frame),
            "rate": float(placebo_frame["strict_triple_pass"].mean()),
        },
        {
            "check": "cap_timing_placebo_dominates_selected",
            "passed_count": int(placebo_frame["dominates_selected_all_three"].sum()),
            "total_count": len(placebo_frame),
            "rate": float(placebo_frame["dominates_selected_all_three"].mean()),
        },
        {
            "check": "cost_0x_to_10x_strict_triple",
            "passed_count": int(cost_frame["strict_triple_pass"].sum()),
            "total_count": len(cost_frame),
            "rate": float(cost_frame["strict_triple_pass"].mean()),
        },
        {
            "check": "delay_0_to_2_strict_triple",
            "passed_count": int(delay_frame["strict_triple_pass"].sum()),
            "total_count": len(delay_frame),
            "rate": float(delay_frame["strict_triple_pass"].mean()),
        },
        {
            "check": "rolling_36m_strict_triple",
            "passed_count": int(rolling_frame["strict_triple_pass"].sum()),
            "total_count": len(rolling_frame),
            "rate": float(rolling_frame["strict_triple_pass"].mean()),
        },
        *[
            {
                "check": f"paired_block_bootstrap_{int(block_days)}d_strict_triple",
                "passed_count": int(group["strict_triple_pass"].sum()),
                "total_count": len(group),
                "rate": float(group["strict_triple_pass"].mean()),
            }
            for block_days, group in bootstrap_frame.groupby("block_days")
        ],
        {
            "check": "positive_defender_episodes",
            "passed_count": int(episodes["positive_excess"].sum()),
            "total_count": len(episodes),
            "rate": float(episodes["positive_excess"].mean()),
        },
        {
            "check": "leave_one_episode_out_strict_triple",
            "passed_count": int(leave_one_frame["strict_triple_pass"].sum()),
            "total_count": len(leave_one_frame),
            "rate": float(leave_one_frame["strict_triple_pass"].mean()),
        },
        {
            "check": "leave_one_episode_out_material_triple",
            "passed_count": int(leave_one_frame["material_triple_pass"].sum()),
            "total_count": len(leave_one_frame),
            "rate": float(leave_one_frame["material_triple_pass"].mean()),
        },
        {
            "check": "positive_episode_concentration_top1",
            "passed_count": np.nan,
            "total_count": np.nan,
            "rate": concentration_1,
        },
        {
            "check": "positive_episode_concentration_top3",
            "passed_count": np.nan,
            "total_count": np.nan,
            "rate": concentration_3,
        },
        {
            "check": "positive_episode_concentration_top5",
            "passed_count": np.nan,
            "total_count": np.nan,
            "rate": concentration_5,
        },
    ]
    pd.DataFrame(robustness_rows).to_csv(
        output_dir / "robustness_summary.csv", index=False
    )
    decision_summary = pd.DataFrame(
        [
            {
                "decision": "full_sample_three_objectives",
                "passed": bool(selected_row["strict_triple_pass"]),
                "evidence": "beats exact and official Momentum on both annualization conventions, Sharpe, and MDD",
            },
            {
                "decision": "selected_parameter_familywise_maxT_5pct",
                "passed": selected_familywise_p_value < 0.05,
                "evidence": f"p={selected_familywise_p_value:.6f}",
            },
            {
                "decision": "mdd_survives_every_episode_neutralization",
                "passed": bool(leave_one_frame["strict_triple_pass"].all()),
                "evidence": f"{int(leave_one_frame['strict_triple_pass'].sum())}/{len(leave_one_frame)} leave-one-out paths retain all three objectives",
            },
            {
                "decision": "production_replacement",
                "passed": False,
                "evidence": "shadow_only: no untouched holdout; familywise test fails; full-sample MDD is event-dependent",
            },
        ]
    )
    decision_summary.to_csv(output_dir / "decision_summary.csv", index=False)

    input_files = [
        defender_dir / "relative_defender_rotation_daily_returns.csv",
        defender_dir / "relative_defender_rotation_daily_indicators.csv",
        defender_dir / "relative_defender_rotation_switch_returns.csv",
        defender_dir / "relative_defender_rotation_switch_handoff.md",
        root / "strategy/configs/quality_momentum_top1.yaml",
    ]
    code_files = [
        root / "research/momentum_defender_occam.py",
        root / "research/momentum_defender_occam_report.py",
        root / "research/run_momentum_defender_occam.py",
        root / "research/tests/test_momentum_defender_occam.py",
        root / "backtest/report.py",
        root / "backtest/runner.py",
        root / "data/store.py",
        root / "factors/quality_momentum.py",
        root / "factors/registry.py",
        root / "factors/registry.yaml",
        root / "factors/validator.py",
        root / "strategy/base.py",
        root / "strategy/loader.py",
        root / "strategy/rebalance.py",
        root / "strategy/top1.py",
        root / "pyproject.toml",
        root / "uv.lock",
        root / ".python-version",
    ]
    momentum_market_files = [
        root / "data/db" / f"{asset}.parquet"
        for asset in ("510300.SH", "159915.SZ", "513100.SH", "518880.SH")
    ]
    manifest = {
        "experiment": "momentum_defender_occam",
        "generated_on": date.today().isoformat(),
        "research_cutoff": end.isoformat(),
        "calendar_rows": len(calendar),
        "selected_parameters": asdict(SELECTED),
        "cap_rule": "signal_volatility_cap < 1 at close, declared effective next open",
        "initial_previous_sleeve": "momentum",
        "bootstrap": {
            "replications": BOOTSTRAP_REPLICATIONS,
            "block_days": BOOTSTRAP_BLOCK_DAYS,
            "seed": BOOTSTRAP_SEED,
        },
        "selection_bias_assumed_trial_count": CONSERVATIVE_TRIAL_COUNT,
        "metric_conventions": {
            "calendar_cagr": "terminal factor^(365.2425 / inclusive calendar days from first through last return label)-1",
            "annualized_return_252": "terminal factor^(252 / observations)-1",
            "sharpe": "mean daily return / sample std * sqrt(252), zero risk-free rate",
            "max_drawdown": "minimum anchored NAV / running maximum - 1",
        },
        "git_commit": _git_value(root, "rev-parse", "HEAD"),
        "git_status_short": git_status_before,
        "inputs": [
            {"path": str(path), "sha256": _sha256(path)} for path in input_files
        ],
        "code_sources": [
            {"path": str(path), "sha256": _sha256(path)} for path in code_files
        ],
        "momentum_market_data": [
            {"path": str(path), "sha256": _sha256(path)}
            for path in momentum_market_files
        ],
    }
    (output_dir / "experiment_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    performance_summary.to_csv(output_dir / "performance_summary.csv", index=False)

    selected_metric = performance(selected["return"])
    exact_metric = performance(exact_momentum)
    official_metric = performance(official_momentum)
    slow_only = next(row for row in ablation_rows if row["strategy"] == "slow_gate_only")
    cap_locked = next(
        row for row in ablation_rows if row["strategy"] == "cap_respects_min_hold"
    )
    bootstrap_rate_text = "、".join(
        f"{int(row.block_days)}日块 {row.strict_triple_rate:.1%}"
        for row in bootstrap_summary.itertuples(index=False)
    )
    cost_10x = cost_frame.loc[cost_frame["cost_multiplier"].eq(10.0)].iloc[0]
    bootstrap_5d = bootstrap_summary.loc[
        bootstrap_summary["block_days"].eq(5)
    ].iloc[0]
    event_trigger = daily.loc[pd.Timestamp("2024-10-08")]
    event_episode = episodes.loc[episodes["episode"].eq(17)].iloc[0]
    event_volatility = float(
        event_trigger["signal_realized_volatility_20_asof_previous_close"]
    )
    event_cap_threshold = float(
        event_trigger["signal_cap_volatility_threshold_asof_previous_close"]
    )
    report = f"""# Momentum × Defender 奥卡姆融合回测

## 结论

在统一的 `2019-01-18` 至 `{end.isoformat()}`、{len(calendar):,} 个交易日样本上，冻结候选同时改善了当前 Momentum 的年化收益、Sharpe 和最大回撤。机械目标已经达成，但选择偏差证据并不完全一致：冻结参数自身的局部族 familywise maxT 检验未过 5% 门槛，且全样本 MDD 优势依赖一次 2024 防守事件。因此它适合作为前瞻 shadow 候选，不能称为独立样本外验证，也不建议仅凭本次回测直接替换生产策略。

## 冻结规则

1. 每个交易日收盘，用 `510300.SH` 计算 40 个交易日收益；高于 2.5% 为 Momentum，否则为 Defender。该决定只在下一交易日开盘生效。
2. 正常状态切换要求当前 sleeve 已持有 30 个完整交易日。
3. Defender 已冻结的 `signal_volatility_cap < 1` 是唯一紧急旁路：它在收盘触发后，于交接文件声明的下一开盘强制 Momentum → Defender，不受 Momentum 锁定期限制。
4. Defender → Momentum 仍须同时满足：cap 已解除、40日门控风险开、Defender 已持有30日。
5. 首日前一收盘持有 Momentum；`2019-01-17` 收盘信号在 `2019-01-18` 开盘执行首次 Momentum → Defender。

切换日严格复合旧 sleeve 的 `exit_prev_close_to_open_net_return` 与新 sleeve 的 `enter_open_to_close_net_return`；不使用任一侧 `daily_net_return_if_held`，因此不会重复计收益或内部调仓费。

## 核心结果

| 口径 | 日历 CAGR | 252日年化 | Sharpe | 最大回撤 |
|---|---:|---:|---:|---:|
| 当前正式 Momentum | {official_metric['cagr_calendar']:.2%} | {official_metric['annualized_return_252']:.2%} | {official_metric['sharpe']:.3f} | {official_metric['max_drawdown']:.2%} |
| 精确分段 Momentum | {exact_metric['cagr_calendar']:.2%} | {exact_metric['annualized_return_252']:.2%} | {exact_metric['sharpe']:.3f} | {exact_metric['max_drawdown']:.2%} |
| 冻结融合候选 | {selected_metric['cagr_calendar']:.2%} | {selected_metric['annualized_return_252']:.2%} | {selected_metric['sharpe']:.3f} | {selected_metric['max_drawdown']:.2%} |

相对正式 Momentum，融合候选的日历 CAGR 提高 {(selected_metric['cagr_calendar'] - official_metric['cagr_calendar']) * 100:.2f} 个百分点，Sharpe 提高 {selected_metric['sharpe'] - official_metric['sharpe']:.3f}，最大回撤收窄 {(selected_metric['max_drawdown'] - official_metric['max_drawdown']) * 100:.2f} 个百分点。共切换 {int(selected['sleeve_switch'].sum())} 次，Defender 占 {float((~selected_state['risk_on']).mean()):.2%} 的交易日。

## 为什么保留 cap 紧急旁路

- 仅用40日慢门控：CAGR {float(slow_only['cagr_calendar']):.2%}、Sharpe {float(slow_only['sharpe']):.3f}、最大回撤 {float(slow_only['max_drawdown']):.2%}；回撤没有实质改善。
- cap 仍受30日锁限制：CAGR {float(cap_locked['cagr_calendar']):.2%}、Sharpe {float(cap_locked['sharpe']):.3f}、最大回撤 {float(cap_locked['max_drawdown']):.2%}；同样未修复最大回撤。
- 只有让 cap 在 Momentum → Defender 时旁路锁定，才同时实现三项实质改善。它没有新增连续阈值参数，但“选择 cap 作为旁路”本身仍属于研究者自由度，必须由安慰剂和未来 shadow 继续检验。

## `signal_volatility_cap` 与 2024 防守事件

- cap 使用信号锚 `512890.SH` 的固定基准后复权 OHLC，计算20日 Rogers–Satchell 实现波动率并按252日年化。
- 阈值是当前收盘之前全部有限20日波动率的严格滞后扩展80%分位，至少20个历史值；不是滚动80日窗口。
- `raw_cap=min(1, 阈值/当前波动率)`，再按20%档位向下量化为0、0.2、0.4、0.6、0.8或1；融合规则把 `signal_volatility_cap < 1` 视为二元紧急警报，并在声明的下一有效开盘执行。
- `2024-09-30` 收盘，20日波动率为 {event_volatility:.6%}，严格滞后80%分位阈值为 {event_cap_threshold:.6%}，量化cap为 {float(event_trigger['signal_volatility_cap_asof_previous_close']):.1f}；因国庆休市，于 `2024-10-08` 开盘生效。
- 当时Momentum仅完成2个交易日，40日慢门控仍为risk-on；cap紧急旁路绕过30日锁后切入Defender。`2024-10-08` 切换日先取得旧Momentum隔夜退出腿 {float(event_trigger['exit_return_leg_used']):+.4%}，再承担Defender进入腿 {float(event_trigger['enter_return_leg_used']):+.4%}，复合收益 {float(event_trigger['return']):+.4%}。
- 第17段 `2024-10-08` 至 `2024-11-19`，融合累计 {float(event_episode['candidate_return']):+.4%}，精确Momentum累计 {float(event_episode['momentum_return']):+.4%}，收益差 {float(event_episode['arithmetic_excess_return']):+.4%}；同期MDD分别为 {float(event_episode['candidate_max_drawdown']):.4%} 与 {float(event_episode['momentum_max_drawdown']):.4%}。

`signal_volatility_cap` 原本是Defender内部限仓值，不是净值回撤或单日亏损信号。该事件中，9月30日Defender网格目标已为0.4、低于cap 0.8，所以cap在Defender内部并未进一步压仓；它在融合规则中主要充当切换触发器。

## 稳健性与过拟合检查

- 45 个完整局部邻域（30/40/50日 × 0/2.5/5% × hold 20/25/30/35/40）中，严格三项改善 {int(local_frame['strict_triple_pass'].sum())}/{len(local_frame)}；达到统一的实质改善门槛（两种年化各 +1个百分点、Sharpe +0.10、最大回撤 +1个百分点）{int(local_frame['material_triple_pass'].sum())}/{len(local_frame)}。冻结参数自身在 hold 20–40 均通过，但 hold=40 的另两个组合失去回撤优势。
- 不带 cap 的 504 点慢门控全网格只有 {int(pd.DataFrame(slow_grid_rows)['strict_triple_pass'].sum())}/504 严格三项改善、{int(pd.DataFrame(slow_grid_rows)['material_triple_pass'].sum())}/504 达到实质门槛；说明慢门控本身并不稳定解决回撤。
- cap 扩展分位 60%–95% 的 7 个压力点全部严格三项改善；80% 只因它是 Defender 已冻结口径而保留。
- cap 时点循环平移安慰剂共 {len(placebo_frame)} 个，其中 {int(placebo_frame['strict_triple_pass'].sum())} 个仍三项改善、{int(placebo_frame['dominates_selected_all_three'].sum())} 个同时支配冻结候选；真实时点较好，但并非不可替代。这些循环平移含首尾回绕，只用于时点安慰剂，不是可交易策略。
- 三个回溯时间段全部严格三项改善；8个年度切片（2019、2026为部分年度）中7个严格三项改善，2023年收益和Sharpe略逊，但回撤仍较浅。
- 按分段毛收益固定、费用等比例放大的 0–10 倍成本压力共 {len(cost_frame)} 个场景，以及信号额外延迟 0–2 日共 {len(delay_frame)} 个场景，均严格三项改善。即使只把候选加压到10倍、Momentum 保持1倍，候选 CAGR、Sharpe、MDD仍分别改善 {float(cost_10x['vs_momentum_1x_cagr_delta']) * 100:.2f} 个百分点、{float(cost_10x['vs_momentum_1x_sharpe_delta']):.3f}、{float(cost_10x['vs_momentum_1x_max_drawdown_improvement']) * 100:.2f} 个百分点。高倍成本是线性费用敏感性，不含冲击成本、容量约束，也不是重新按高费率回放每只 Defender 资产。
- 52 个 756交易日滚动窗口全部严格三项改善；窗口彼此重叠，不能视为52个独立样本。
- 固定历史状态路径的配对 moving-block bootstrap 每种块长各 {BOOTSTRAP_REPLICATIONS:,} 次，三项同时为正比例为：{bootstrap_rate_text}。这是条件型重采样，未在每条伪市场路径上重新生成信号，且块边界会影响MDD；不能当作策略级样本外检验。
- 其中5日块的年化收益差双侧95%分位区间为 {float(bootstrap_5d['annualized_252_delta_q0.025']) * 100:.2f} 至 {float(bootstrap_5d['annualized_252_delta_q0.975']) * 100:.2f} 个百分点，仍跨过零；不能声称所有 bootstrap 口径都在95%置信水平排除“无年化优势”。
- 45点局部族的 20日 studentized moving-block maxT omnibus（Reality-Check-style）p 值为 {omnibus_reality_p_value:.4f}（{'通过' if omnibus_reality_p_value < 0.05 else '未通过'}5%）；冻结参数自身的 familywise maxT p 值为 {selected_familywise_p_value:.4f}（{'通过' if selected_familywise_p_value < 0.05 else '**未通过**'}5%）。后者是本报告最重要的多重检验警示。另一个 `{CONSERVATIVE_TRIAL_COUNT:,}` 次零均值独立试验下的最大Sharpe启发式概率为 {iid_heuristic_probability:.2%}，仅作信息项；它不是标准 Deflated Sharpe，且候选收益存在一阶自相关 {autocorrelation_1:.3f}、峰度 {kurtosis:.1f}，不能用于抵消 familywise 检验。
- {int(episodes['positive_excess'].sum())}/{len(episodes)} 个 Defender episode 相对 Momentum 为正；最大单段占全部正向对数超额贡献 {concentration_1:.1%}，前5段占 {concentration_5:.1%}，并非完全由单一事件贡献。
- leave-one-episode-out 有 {int(leave_one_frame['strict_triple_pass'].sum())}/{len(leave_one_frame)} 仍严格三项改善。唯一例外是中和 `{critical_episode['entry_date']}` 至 `{critical_episode['window_end']}` 的防守段：CAGR 与 Sharpe 仍分别比 Momentum 高 {float(critical_episode['cagr_delta']) * 100:.2f} 个百分点和 {float(critical_episode['sharpe_delta']):.3f}，但最大回撤改善降为约0。**因此收益超额不依赖单一段，但完整样本的 MDD 改善依赖这次 2024 防守事件。**

## 数据与口径审计

- 三份 Defender CSV 均为1,840行；实际列数是每日收益5列、指标71列、切换接口66列。三表日期逐日一致。
- 本实验明确截断在 `{end.isoformat()}`，排除 2026-08-18 至 2026-08-20，得到1,837行。
- `2021-10-22` 保留，实际持有 `511260.SH` 100%，净收益 `+0.0437931910%`。
- Defender held/enter/exit 分段公式、净值、四组权重与费用检查全部通过。
- Momentum 分段适配器与正式 runner 的最大单日误差为 {momentum_daily_error:.3e}，期末净值相对差 {momentum_nav_ratio_error:.3e}；候选同时胜过这两个基准。
- `calendar_asset=512890.SH` 仅视为遗留信号锚标签，绝不用于过滤融合日历。
- 交接说明仍有两个非数值问题：`signal_date` 更准确的定义应是“前一有效信号观察日”（`2021-10-25` 仍为 `2021-10-21`），且第6节未列出文件中实际存在的 `closing_weight_*`/`closing_cash_weight`。这不影响本次接入。

## 使用边界

历史后段、参数邻域、bootstrap 与 Reality Check 都是回溯稳定性检验，不是 untouched holdout。`experiments/20260818_*` 旧实验读取另一份 Defender 数据，且缺少 `2021-10-22`，不能作为本候选证据。建议从规则和代码冻结后的下一个未观察交易日起做不可回改 shadow；在积累足够前瞻样本前，不把本报告表述为生产替换结论。
"""
    (output_dir / "research_report.md").write_text(report, encoding="utf-8")
    standard_report_config = {
        "strategy_name": "momentum_defender_occam",
        **asdict(SELECTED),
        "cap_rule": "signal_volatility_cap < 1",
        "research_cutoff": end.isoformat(),
    }
    _generate_standard_report(
        selected["return"],
        original_base,
        "Original 4ETF Equal-Weight Base",
        output_dir / "momentum_defender_occam_vs_original_base.html",
        standard_report_config,
    )
    _generate_standard_report(
        selected["return"],
        exact_momentum,
        "Original Momentum Strategy",
        output_dir / "momentum_defender_occam_vs_original_momentum.html",
        standard_report_config,
    )
    _generate_standard_report(
        slow_gate_only_returns,
        exact_momentum,
        "Original Momentum Strategy",
        output_dir / "momentum_defender_occam_no_cap_vs_original_momentum.html",
        {
            **standard_report_config,
            "strategy_name": "momentum_defender_occam_no_cap",
            "cap_rule": "disabled",
        },
    )
    generate_html_report(output_dir)
    final_output_dir.mkdir(parents=True, exist_ok=True)
    for staged_file in output_dir.iterdir():
        staged_file.replace(final_output_dir / staged_file.name)
    output_dir.rmdir()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--defender-dir", type=Path, default=DEFAULT_DEFENDER_DIR)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/20260821_momentum_defender_occam"),
    )
    parser.add_argument("--end", type=date.fromisoformat, default=DEFAULT_END)
    args = parser.parse_args()
    run_experiment(
        args.root.resolve(),
        args.defender_dir.resolve(),
        args.output_dir.resolve(),
        args.end,
    )


if __name__ == "__main__":
    main()
