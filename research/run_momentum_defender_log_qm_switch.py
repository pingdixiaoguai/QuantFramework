"""Robust low-dimensional switch search with log-MOM/log-ER Momentum frozen."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from dataclasses import asdict
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from factors.quality_momentum import METADATA as QUALITY_METADATA
from research.gold_min5_risk_adjusted_momentum import risk_adjusted_momentum_at_open
from research.gold_min5_risk_adjusted_momentum_w5 import (
    GoldRAQMW5Params,
    run_gold_raqm_w5,
)
from research.momentum_defender_gold_override import build_gold_override_context
from research.momentum_defender_gold_override_overfit import (
    cscv_pbo,
    expanding_walk_forward,
    full_metrics,
    leave_one_year_selection,
    paired_block_bootstrap,
    yearly_reality_check,
)
from research.momentum_defender_log_qm_switch import (
    CLOSE_LOG_STD,
    EXPANDING_HISTORY,
    LOG_RETURN,
    NO_EMERGENCY,
    ROGERS_SATCHELL,
    ROLLING_HISTORY,
    SIMPLE_RETURN,
    SwitchRun,
    SwitchSpec,
    build_fast_switch_data,
    fast_run_switch_spec,
    held_asset_alert,
    pareto_frontier,
    realized_volatility,
    run_switch_spec,
    slow_regime_at_open,
    strict_lag_volatility_cap,
)
from research.momentum_defender_occam import MOMENTUM_ASSETS, performance
from research.momentum_volatility import asof_previous_close, load_ohlc
from research.standard_report import generate_standard_report


DEFAULT_CONFIG = Path(
    "research/configs/momentum_defender_log_qm_switch_search.yaml"
)
DEFAULT_OUTPUT = Path(
    "experiments/20260824_momentum_defender_log_qm_switch_robust"
)


def _load_config(path: Path) -> dict:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("search config must be a mapping")
    return config


def _periods(config: dict) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
    return {
        label: (pd.Timestamp(values[0]), pd.Timestamp(values[1]))
        for label, values in config["periods"].items()
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _return_hash(values: pd.Series) -> str:
    return hashlib.sha256(values.to_numpy(dtype="<f8").tobytes()).hexdigest()


def _spec_record(spec: SwitchSpec) -> dict[str, object]:
    return {"candidate_id": spec.candidate_id(), **asdict(spec)}


def _cap_cache(context, config: dict) -> dict[tuple[str, str, int, str, float], pd.Series]:
    emergency = config["emergency_joint_stage"]
    estimators = set(emergency["volatility_estimators"]) | {
        config["slow_gate_stage"]["fixed_emergency"]["volatility_estimator"]
    }
    windows = set(map(int, emergency["volatility_windows"])) | {
        int(config["slow_gate_stage"]["fixed_emergency"]["volatility_window"])
    }
    histories = set(emergency["quantile_histories"]) | {
        config["slow_gate_stage"]["fixed_emergency"]["quantile_history"]
    }
    quantiles = {
        float(value)
        for scheme in emergency["quantile_schemes"].values()
        for value in scheme.values()
    }
    result: dict[tuple[str, str, int, str, float], pd.Series] = {}
    end = context.calendar.max().date()
    for asset in MOMENTUM_ASSETS:
        prices = load_ohlc(asset, end)
        for estimator in sorted(estimators):
            for window in sorted(windows):
                volatility = realized_volatility(
                    prices, estimator=estimator, window=window
                )
                for history in sorted(histories):
                    for quantile in sorted(quantiles):
                        close_cap = strict_lag_volatility_cap(
                            volatility,
                            quantile,
                            history=history,
                            step=float(emergency["cap_step"]),
                            minimum_history=int(emergency["quantile_min_history"]),
                            rolling_history=int(emergency["rolling_history"]),
                        )["cap"]
                        result[(asset, estimator, window, history, quantile)] = (
                            asof_previous_close(close_cap, context.calendar).fillna(1.0)
                        )
    return result


def _caps_for_spec(
    spec: SwitchSpec,
    schemes: dict[str, dict[str, float]],
    cache: dict[tuple[str, str, int, str, float], pd.Series],
) -> dict[str, pd.Series] | None:
    if not spec.emergency_enabled:
        return None
    scheme = schemes[spec.quantile_scheme]
    return {
        asset: cache[
            (
                asset,
                spec.volatility_estimator,
                spec.volatility_window,
                spec.quantile_history,
                float(scheme[asset]),
            )
        ]
        for asset in MOMENTUM_ASSETS
    }


def _evaluate_specs(
    context,
    specs: list[SwitchSpec],
    schemes: dict[str, dict[str, float]],
    cache: dict[tuple[str, str, int, str, float], pd.Series],
    gold_metrics: pd.DataFrame,
    fast_data,
    slow_cache: dict[tuple[str, int, float], pd.Series],
    alert_cache: dict[tuple[object, ...], pd.Series],
    *,
    label: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    records: list[dict[str, object]] = []
    returns: dict[str, np.ndarray] = {}
    for position, spec in enumerate(specs, start=1):
        slow_key = (
            spec.slow_return_mode,
            spec.slow_lookback,
            spec.slow_threshold,
        )
        if slow_key not in slow_cache:
            slow_cache[slow_key] = slow_regime_at_open(
                context.integrated.result.inputs.risk_close,
                context.calendar,
                mode=spec.slow_return_mode,
                lookback=spec.slow_lookback,
                threshold=spec.slow_threshold,
            )
        if spec.emergency_enabled:
            alert_key = (
                spec.volatility_estimator,
                spec.volatility_window,
                spec.quantile_history,
                spec.quantile_scheme,
                spec.cap_trigger_maximum,
            )
            if alert_key not in alert_cache:
                alert_cache[alert_key] = held_asset_alert(
                    _caps_for_spec(spec, schemes, cache),
                    context.integrated.result.previous_asset,
                    spec.cap_trigger_maximum,
                )
            emergency = alert_cache[alert_key]
        else:
            emergency = None
        run = fast_run_switch_spec(
            fast_data,
            spec,
            slow_cache[slow_key],
            emergency,
        )
        candidate_id = spec.candidate_id()
        returns[candidate_id] = run.returns
        records.append(
            {
                **_spec_record(spec),
                "emergency_entries": run.emergency_entries,
                "defender_days": run.defender_days,
                "base_switches": run.base_switches,
                "gold_entries": run.gold_entries,
                "gold_days": run.gold_days,
                "formal_switches": run.formal_switches,
            }
        )
        if position % 50 == 0 or position == len(specs):
            print(f"{label}: evaluated {position}/{len(specs)}", flush=True)
    return (
        pd.DataFrame(records).drop_duplicates("candidate_id").set_index("candidate_id"),
        pd.DataFrame(returns, index=context.calendar),
    )


def _add_period_metrics(
    metadata: pd.DataFrame,
    returns: pd.DataFrame,
    baseline: pd.Series,
    periods: dict[str, tuple[pd.Timestamp, pd.Timestamp]],
) -> pd.DataFrame:
    table = metadata.copy()
    for label, (start, end) in periods.items():
        sample = returns.loc[start:end]
        reference = baseline.loc[start:end]
        measured = full_metrics(sample, reference)
        measured = measured.rename(columns={column: f"{label}_{column}" for column in measured})
        table = table.join(measured)
    return table


def _rank_development(
    table: pd.DataFrame,
    baseline: pd.Series,
    periods: dict[str, tuple[pd.Timestamp, pd.Timestamp]],
    selection: dict,
) -> pd.DataFrame:
    start, end = periods["development"]
    baseline_metrics = performance(baseline.loc[start:end])
    eligible = (
        table["emergency_entries"].ge(int(selection["minimum_defender_entries"]))
        & table["defender_days"].ge(int(selection["minimum_defender_days"]))
        & table["development_max_drawdown"].ge(
            float(baseline_metrics["max_drawdown"])
            - float(selection["maximum_development_mdd_worsening"])
        )
    )
    ranked = table.copy()
    ranked["development_eligible"] = eligible
    metric_columns = [
        "development_annualized_return_252",
        "development_sharpe",
        "development_max_drawdown",
    ]
    ranked["development_pareto"] = False
    if eligible.any():
        ranked.loc[eligible, "development_pareto"] = pareto_frontier(
            ranked.loc[eligible], metric_columns
        )
    percentiles = ranked.loc[eligible, metric_columns].rank(pct=True)
    ranked.loc[eligible, "minimum_metric_percentile"] = percentiles.min(axis=1)
    ranked.loc[eligible, "mean_metric_percentile"] = percentiles.mean(axis=1)
    ranked["minimum_metric_percentile"] = ranked["minimum_metric_percentile"].fillna(-1.0)
    ranked["mean_metric_percentile"] = ranked["mean_metric_percentile"].fillna(-1.0)
    return ranked


def _select_rows(ranked: pd.DataFrame, count: int) -> pd.DataFrame:
    pool = ranked.loc[ranked["development_eligible"]].copy()
    frontier = pool.loc[pool["development_pareto"]]
    if not frontier.empty:
        pool = frontier
    return pool.sort_values(
        [
            "minimum_metric_percentile",
            "mean_metric_percentile",
            "development_sharpe",
            "development_annualized_return_252",
            "formal_switches",
        ],
        ascending=[False, False, False, False, True],
    ).head(count)


def _unique_paths(returns: pd.DataFrame) -> pd.DataFrame:
    seen: set[str] = set()
    columns: list[str] = []
    for column in returns:
        digest = hashlib.sha1(returns[column].to_numpy(float).tobytes()).hexdigest()
        if digest not in seen:
            seen.add(digest)
            columns.append(column)
    return returns[columns]


def _validation_gate(
    selected: pd.Series,
    config: dict,
) -> dict[str, object]:
    gates = config["validation_gates"]
    checks: dict[str, bool] = {}
    for period in ("validation", "recent"):
        checks[f"{period}_annual_floor"] = float(
            selected[f"{period}_delta_annualized_return_252"]
        ) >= float(gates[f"{period}_annualized_delta_floor"])
        checks[f"{period}_sharpe_floor"] = float(
            selected[f"{period}_delta_sharpe"]
        ) >= float(gates[f"{period}_sharpe_delta_floor"])
        checks[f"{period}_mdd_floor"] = float(
            selected[f"{period}_delta_max_drawdown"]
        ) >= float(gates[f"{period}_mdd_delta_floor"])
        nonnegative = sum(
            float(selected[f"{period}_delta_{field}"]) >= 0.0
            for field in (
                "annualized_return_252",
                "sharpe",
                "max_drawdown",
            )
        )
        checks[f"{period}_nonnegative_metric_count"] = nonnegative >= int(
            gates["minimum_nonnegative_metric_deltas_each_holdout"]
        )
    return {"passed": bool(all(checks.values())), "checks": checks}


def _grid_holdout_audit(table: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, dict]:
    gates = config["validation_gates"]
    audited = pd.DataFrame(index=table.index)
    for period in ("validation", "recent"):
        nonnegative = sum(
            table[f"{period}_delta_{field}"].ge(0.0).astype(int)
            for field in ("annualized_return_252", "sharpe", "max_drawdown")
        )
        audited[f"{period}_nonnegative_metrics"] = nonnegative
        audited[f"{period}_gate_passed"] = (
            table[f"{period}_delta_annualized_return_252"].ge(
                float(gates[f"{period}_annualized_delta_floor"])
            )
            & table[f"{period}_delta_sharpe"].ge(
                float(gates[f"{period}_sharpe_delta_floor"])
            )
            & table[f"{period}_delta_max_drawdown"].ge(
                float(gates[f"{period}_mdd_delta_floor"])
            )
            & nonnegative.ge(
                int(gates["minimum_nonnegative_metric_deltas_each_holdout"])
            )
        )
    audited["both_holdout_gates_passed"] = (
        audited["validation_gate_passed"] & audited["recent_gate_passed"]
    )
    audited["full_dominates_baseline"] = (
        table["full_delta_annualized_return_252"].ge(0.0)
        & table["full_delta_sharpe"].ge(0.0)
        & table["full_delta_max_drawdown"].ge(0.0)
    )
    summary = {
        "candidates": int(len(audited)),
        "validation_gate_passed": int(audited["validation_gate_passed"].sum()),
        "recent_gate_passed": int(audited["recent_gate_passed"].sum()),
        "both_holdout_gates_passed": int(
            audited["both_holdout_gates_passed"].sum()
        ),
        "full_dominates_baseline": int(audited["full_dominates_baseline"].sum()),
        "full_dominates_and_both_holdouts": int(
            (
                audited["full_dominates_baseline"]
                & audited["both_holdout_gates_passed"]
            ).sum()
        ),
    }
    return audited, summary


def _neighborhood(table: pd.DataFrame, selected: pd.Series) -> pd.DataFrame:
    parameters = [
        "slow_return_mode",
        "slow_lookback",
        "slow_threshold",
        "min_hold_days",
        "emergency_enabled",
        "volatility_estimator",
        "volatility_window",
        "quantile_history",
        "quantile_scheme",
        "cap_trigger_maximum",
    ]
    distance = pd.Series(0, index=table.index, dtype=int)
    for parameter in parameters:
        distance += ~table[parameter].eq(selected[parameter])
    result = table.loc[distance.le(1)].copy()
    result["parameter_hamming_distance"] = distance.loc[result.index]
    return result.sort_values(
        ["parameter_hamming_distance", "development_sharpe"],
        ascending=[True, False],
    )


def _events(
    selected: SwitchRun,
    baseline_state: pd.DataFrame,
    baseline_daily: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidate_target = selected.formal_state["target_candidate"].astype(str)
    baseline_target = baseline_state["target_candidate"].astype(str)
    changed = candidate_target.ne(baseline_target)
    groups = changed.ne(changed.shift()).cumsum()
    calendar = selected.formal_daily.index
    records: list[dict[str, object]] = []
    leave_records: list[dict[str, object]] = []
    candidate_returns = selected.formal_daily["return"].astype(float)
    baseline_returns = baseline_daily["return"].astype(float)
    for event, (_, sample) in enumerate(
        selected.formal_state.loc[changed].groupby(groups.loc[changed]), start=1
    ):
        start_location = calendar.get_loc(sample.index.min())
        end_location = min(calendar.get_loc(sample.index.max()) + 1, len(calendar) - 1)
        interval = calendar[start_location : end_location + 1]
        candidate_total = float((1.0 + candidate_returns.loc[interval]).prod() - 1.0)
        baseline_total = float((1.0 + baseline_returns.loc[interval]).prod() - 1.0)
        records.append(
            {
                "event": event,
                "start": interval.min().date().isoformat(),
                "end_including_exit": interval.max().date().isoformat(),
                "observations": len(interval),
                "candidate_return": candidate_total,
                "baseline_return": baseline_total,
                "log_excess": float(np.log1p(candidate_total) - np.log1p(baseline_total)),
                "candidate_targets": "|".join(candidate_target.loc[interval].drop_duplicates()),
                "baseline_targets": "|".join(baseline_target.loc[interval].drop_duplicates()),
            }
        )
        counterfactual = candidate_returns.copy()
        counterfactual.loc[interval] = baseline_returns.loc[interval]
        leave_records.append({"removed_event": event, **performance(counterfactual)})
    return pd.DataFrame(records), pd.DataFrame(leave_records)


def _friction_stress(run: SwitchRun, multipliers: list[float]) -> pd.DataFrame:
    returns = run.formal_daily["return"].astype(float)
    costs = run.formal_daily["cost_rate_at_open"].astype(float).clip(0.0, 0.99)
    gross = (1.0 + returns) / (1.0 - costs) - 1.0
    rows = []
    for multiplier in multipliers:
        stressed = (1.0 + gross) * (1.0 - float(multiplier) * costs) - 1.0
        rows.append({"cost_multiplier": multiplier, **performance(stressed)})
    return pd.DataFrame(rows)


def run_experiment(root: Path, config_path: Path, output: Path) -> dict[str, object]:
    config = _load_config(config_path)
    periods = _periods(config)
    frozen = config["frozen_momentum"]
    if QUALITY_METADATA["version"] != str(frozen["factor_version"]):
        raise AssertionError("quality_momentum version differs from preregistration")
    if frozen["momentum_convention"] != "log_return" or frozen[
        "efficiency_ratio_convention"
    ] != "log_price_path":
        raise AssertionError("Momentum calculation convention is not frozen to log/log")

    full_end = periods["full"][1].date()
    context = build_gold_override_context(root, end=full_end)
    gold_metrics = risk_adjusted_momentum_at_open(context.curves, window=5)
    fast_data = build_fast_switch_data(context, gold_metrics)
    slow_cache: dict[tuple[str, int, float], pd.Series] = {}
    alert_cache: dict[tuple[object, ...], pd.Series] = {}
    baseline_formal = run_gold_raqm_w5(
        context, GoldRAQMW5Params(2.20, 0.60), metrics=gold_metrics
    )
    baseline_returns = baseline_formal.daily["return"].astype(float)
    schemes = config["emergency_joint_stage"]["quantile_schemes"]
    cache = _cap_cache(context, config)

    slow_grid = config["slow_gate_stage"]
    fixed = slow_grid["fixed_emergency"]
    baseline_spec = SwitchSpec(
        slow_return_mode=SIMPLE_RETURN,
        slow_lookback=40,
        slow_threshold=0.025,
        min_hold_days=30,
        emergency_enabled=True,
        volatility_estimator=str(fixed["volatility_estimator"]),
        volatility_window=int(fixed["volatility_window"]),
        quantile_history=str(fixed["quantile_history"]),
        quantile_scheme=str(fixed["quantile_scheme"]),
        cap_trigger_maximum=float(fixed["cap_trigger_maximum"]),
    )
    baseline_slow = slow_regime_at_open(
        context.integrated.result.inputs.risk_close,
        context.calendar,
        mode=baseline_spec.slow_return_mode,
        lookback=baseline_spec.slow_lookback,
        threshold=baseline_spec.slow_threshold,
    )
    baseline_alert = held_asset_alert(
        _caps_for_spec(baseline_spec, schemes, cache),
        context.integrated.result.previous_asset,
        baseline_spec.cap_trigger_maximum,
    )
    fast_baseline = fast_run_switch_spec(
        fast_data,
        baseline_spec,
        baseline_slow,
        baseline_alert,
    )
    fast_parity = float(
        np.max(np.abs(fast_baseline.returns - baseline_returns.to_numpy(float)))
    )
    if fast_parity > 1e-12:
        raise AssertionError(f"fast search path fails baseline parity: {fast_parity:.3e}")
    slow_cache[(SIMPLE_RETURN, 40, 0.025)] = baseline_slow
    alert_cache[
        (
            baseline_spec.volatility_estimator,
            baseline_spec.volatility_window,
            baseline_spec.quantile_history,
            baseline_spec.quantile_scheme,
            baseline_spec.cap_trigger_maximum,
        )
    ] = baseline_alert
    stage_one_specs = [
        SwitchSpec(
            slow_return_mode=str(mode),
            slow_lookback=int(lookback),
            slow_threshold=float(threshold),
            min_hold_days=int(hold),
            emergency_enabled=True,
            volatility_estimator=str(fixed["volatility_estimator"]),
            volatility_window=int(fixed["volatility_window"]),
            quantile_history=str(fixed["quantile_history"]),
            quantile_scheme=str(fixed["quantile_scheme"]),
            cap_trigger_maximum=float(fixed["cap_trigger_maximum"]),
        )
        for mode in slow_grid["return_modes"]
        for lookback in slow_grid["lookbacks"]
        for threshold in slow_grid["thresholds"]
        for hold in slow_grid["min_hold_days"]
    ]
    stage_one_meta, stage_one_returns = _evaluate_specs(
        context,
        stage_one_specs,
        schemes,
        cache,
        gold_metrics,
        fast_data,
        slow_cache,
        alert_cache,
        label="slow-stage",
    )
    stage_one_table = _add_period_metrics(
        stage_one_meta, stage_one_returns, baseline_returns, periods
    )
    stage_one_ranked = _rank_development(
        stage_one_table, baseline_returns, periods, config["selection"]
    )
    slow_selected = _select_rows(
        stage_one_ranked,
        int(slow_grid["top_development_candidates_for_joint_stage"]),
    )
    if slow_selected.empty:
        raise RuntimeError("slow stage produced no eligible candidates")

    emergency = config["emergency_joint_stage"]
    stage_two_specs: list[SwitchSpec] = []
    for row in slow_selected.itertuples():
        slow_values = dict(
            slow_return_mode=row.slow_return_mode,
            slow_lookback=int(row.slow_lookback),
            slow_threshold=float(row.slow_threshold),
            min_hold_days=int(row.min_hold_days),
        )
        if bool(emergency["include_no_emergency"]):
            stage_two_specs.append(SwitchSpec(**slow_values, emergency_enabled=False))
        for estimator in emergency["volatility_estimators"]:
            for window in emergency["volatility_windows"]:
                for history in emergency["quantile_histories"]:
                    for scheme in emergency["quantile_schemes"]:
                        for trigger in emergency["cap_trigger_maximums"]:
                            stage_two_specs.append(
                                SwitchSpec(
                                    **slow_values,
                                    emergency_enabled=True,
                                    volatility_estimator=str(estimator),
                                    volatility_window=int(window),
                                    quantile_history=str(history),
                                    quantile_scheme=str(scheme),
                                    cap_trigger_maximum=float(trigger),
                                )
                            )
    stage_two_specs = list(
        {spec.candidate_id(): spec for spec in stage_two_specs}.values()
    )
    stage_two_meta, stage_two_returns = _evaluate_specs(
        context,
        stage_two_specs,
        schemes,
        cache,
        gold_metrics,
        fast_data,
        slow_cache,
        alert_cache,
        label="joint-stage",
    )
    stage_two_table = _add_period_metrics(
        stage_two_meta, stage_two_returns, baseline_returns, periods
    )
    stage_two_ranked = _rank_development(
        stage_two_table, baseline_returns, periods, config["selection"]
    )
    selected_row = _select_rows(stage_two_ranked, 1).iloc[0]
    selected_id = str(selected_row.name)
    selected_spec = next(
        spec for spec in stage_two_specs if spec.candidate_id() == selected_id
    )
    selected_run = run_switch_spec(
        context,
        selected_spec,
        _caps_for_spec(selected_spec, schemes, cache),
        gold_metrics=gold_metrics,
    )
    selected_returns = selected_run.formal_daily["return"].astype(float)
    validation = _validation_gate(selected_row, config)
    grid_holdout, grid_holdout_summary = _grid_holdout_audit(
        stage_two_ranked, config
    )

    all_returns = pd.concat([stage_one_returns, stage_two_returns], axis=1)
    all_returns = all_returns.loc[:, ~all_returns.columns.duplicated()]
    unique_returns = _unique_paths(all_returns)
    overfit = config["overfit_checks"]
    pbo_frame, pbo_summary = cscv_pbo(
        unique_returns,
        baseline_returns,
        block_count=int(overfit["cscv_blocks"]),
    )
    walk_forward = expanding_walk_forward(unique_returns, baseline_returns)
    leave_year = leave_one_year_selection(unique_returns, baseline_returns)
    bootstrap_frame, bootstrap_summary = paired_block_bootstrap(
        selected_returns,
        baseline_returns,
        block_size=int(overfit["paired_block_bootstrap_block"]),
        repetitions=int(overfit["paired_block_bootstrap_repetitions"]),
        seed=int(overfit["random_seed"]),
    )
    reality = yearly_reality_check(
        unique_returns,
        baseline_returns,
        repetitions=int(overfit["yearly_reality_check_repetitions"]),
        seed=int(overfit["random_seed"]),
    )
    events, leave_events = _events(
        selected_run, baseline_formal.state, baseline_formal.daily
    )
    neighborhood = _neighborhood(stage_two_ranked, selected_row)
    friction = _friction_stress(
        selected_run, list(overfit["friction_cost_multipliers"])
    )
    neighborhood_summary = {
        "candidates": int(len(neighborhood)),
        "annualized_improvement_rate": float(
            neighborhood["full_delta_annualized_return_252"].gt(0.0).mean()
        ),
        "sharpe_improvement_rate": float(
            neighborhood["full_delta_sharpe"].gt(0.0).mean()
        ),
        "mdd_nonworsening_rate": float(
            neighborhood["full_delta_max_drawdown"].ge(0.0).mean()
        ),
        "all_three_rate": float(
            (
                neighborhood["full_delta_annualized_return_252"].gt(0.0)
                & neighborhood["full_delta_sharpe"].gt(0.0)
                & neighborhood["full_delta_max_drawdown"].ge(0.0)
            ).mean()
        ),
    }
    positive_log_excess = events.loc[events["log_excess"].gt(0.0), "log_excess"]
    event_summary = {
        "events": int(len(events)),
        "positive_events": int(events["log_excess"].gt(0.0).sum()),
        "negative_events": int(events["log_excess"].lt(0.0).sum()),
        "top_two_positive_share": (
            float(positive_log_excess.nlargest(2).sum() / positive_log_excess.sum())
            if not positive_log_excess.empty and positive_log_excess.sum() > 0.0
            else 0.0
        ),
        "leave_one_event_min_annualized_return_252": (
            float(leave_events["annualized_return_252"].min())
            if not leave_events.empty
            else float("nan")
        ),
        "leave_one_event_min_sharpe": (
            float(leave_events["sharpe"].min())
            if not leave_events.empty
            else float("nan")
        ),
    }

    selected_metrics = performance(selected_returns)
    baseline_metrics = performance(baseline_returns)
    promotion_supported = bool(
        validation["passed"]
        and selected_metrics["annualized_return_252"]
        >= baseline_metrics["annualized_return_252"]
        and selected_metrics["sharpe"] >= baseline_metrics["sharpe"]
        and selected_metrics["max_drawdown"] >= baseline_metrics["max_drawdown"]
        and bootstrap_summary["annualized_return_delta_ci_lower"] > 0.0
        and bootstrap_summary["sharpe_delta_ci_lower"] > 0.0
        and reality["p_value"] < 0.05
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    stage_one_ranked.to_csv(stage / "slow_stage_candidate_grid.csv")
    unique_returns.to_parquet(stage / "unique_candidate_returns.parquet")
    slow_selected.to_csv(stage / "slow_stage_selected_for_joint.csv")
    stage_two_ranked.to_csv(stage / "joint_candidate_grid.csv")
    stage_two_ranked.loc[stage_two_ranked["development_pareto"]].to_csv(
        stage / "development_pareto_frontier.csv"
    )
    neighborhood.to_csv(stage / "selected_parameter_neighborhood.csv")
    grid_holdout.to_csv(stage / "grid_holdout_gate_audit.csv")
    pd.DataFrame(
        [
            {"strategy": "log_qm_baseline", **baseline_metrics},
            {"strategy": "development_selected", **selected_metrics},
        ]
    ).to_csv(stage / "full_strategy_metrics.csv", index=False)
    selected_row.to_frame().T.to_csv(stage / "selected_period_metrics.csv")
    selected_run.state.join(
        selected_run.base_daily.drop(columns=["risk_on"])
    ).to_csv(stage / "selected_base_c2_daily.csv")
    selected_run.formal_state.join(
        selected_run.formal_daily, rsuffix="_execution"
    ).to_csv(stage / "selected_formal_daily.csv")
    baseline_formal.state.join(
        baseline_formal.daily, rsuffix="_execution"
    ).to_csv(stage / "baseline_formal_daily.csv")
    pbo_frame.to_csv(stage / "cscv_pbo.csv", index=False)
    walk_forward.to_csv(stage / "expanding_walk_forward.csv", index=False)
    leave_year.to_csv(stage / "leave_one_year.csv", index=False)
    bootstrap_frame.to_csv(stage / "paired_block_bootstrap.csv", index=False)
    events.to_csv(stage / "event_attribution.csv", index=False)
    leave_events.to_csv(stage / "leave_one_event.csv", index=False)
    friction.to_csv(stage / "friction_stress.csv", index=False)

    selected_config = {
        "strategy_id": "momentum_defender_log_qm_switch_selected_v1",
        "status": "research_candidate" if promotion_supported else "research_rejected",
        "momentum": config["frozen_momentum"],
        "switch": asdict(selected_spec),
        "gold_override": config["fixed_layers"]["gold_override"],
        "selection_period": config["selection"]["selection_period"],
        "validation_passed": validation["passed"],
        "production_promotion_supported": promotion_supported,
        "checkpoint": {
            "start": selected_metrics["start"],
            "end": selected_metrics["end"],
            "observations": selected_metrics["observations"],
            "annualized_return_252": selected_metrics["annualized_return_252"],
            "sharpe": selected_metrics["sharpe"],
            "max_drawdown": selected_metrics["max_drawdown"],
            "daily_return_sha256_float64_le": _return_hash(selected_returns),
        },
    }
    (stage / "selected_research_config.yaml").write_text(
        yaml.safe_dump(selected_config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    (stage / "search_config.yaml").write_text(
        config_path.read_text(encoding="utf-8"), encoding="utf-8"
    )
    audit = {
        "status": "passed",
        "experiment_id": config["experiment"]["id"],
        "frozen_momentum_version": QUALITY_METADATA["version"],
        "fast_baseline_parity_max_abs_error": fast_parity,
        "stage_one_candidates": len(stage_one_specs),
        "stage_two_candidates": len(stage_two_specs),
        "actual_candidate_ids": int(all_returns.shape[1]),
        "unique_return_paths": int(unique_returns.shape[1]),
        "selected_candidate": selected_id,
        "selected_spec": asdict(selected_spec),
        "selected_run_audit": dict(selected_run.audit),
        "baseline_metrics": baseline_metrics,
        "selected_metrics": selected_metrics,
        "validation": validation,
        "grid_holdout_gate_audit": grid_holdout_summary,
        "parameter_neighborhood": neighborhood_summary,
        "cscv_pbo": pbo_summary,
        "paired_block_bootstrap": bootstrap_summary,
        "yearly_reality_check": reality,
        "walk_forward": {
            "years": int(len(walk_forward)),
            "return_win_rate": float(walk_forward["test_return_delta"].gt(0).mean()),
            "sharpe_win_rate": float(walk_forward["test_sharpe_delta"].gt(0).mean()),
        },
        "leave_one_year": {
            "years": int(len(leave_year)),
            "return_win_rate": float(leave_year["test_return_delta"].gt(0).mean()),
            "sharpe_win_rate": float(leave_year["test_sharpe_delta"].gt(0).mean()),
        },
        "event_attribution": event_summary,
        "production_promotion_supported": promotion_supported,
    }
    (stage / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    generate_standard_report(
        selected_returns,
        baseline_returns,
        "Current log-QM formal baseline",
        stage / "selected_vs_log_qm_baseline.html",
        selected_config,
    )

    annual_delta = (
        float(selected_metrics["annualized_return_252"])
        - float(baseline_metrics["annualized_return_252"])
    )
    sharpe_delta = float(selected_metrics["sharpe"]) - float(
        baseline_metrics["sharpe"]
    )
    mdd_delta = float(selected_metrics["max_drawdown"]) - float(
        baseline_metrics["max_drawdown"]
    )
    report = f"""# 双对数Momentum下的Defender切换稳健寻优

Momentum固定为20日对数收益乘对数路径ER，未参与搜索。慢门阶段测试
{len(stage_one_specs)}组，联合阶段测试{len(stage_two_specs)}组；合并后实际候选ID
{all_returns.shape[1]}个、唯一收益路径{unique_returns.shape[1]}条。参数只按development段选择，
validation、recent和full均未参与选参。

## 选中候选

`{selected_id}`

|指标|基线|候选|差值|
|---|---:|---:|---:|
|年化收益|{float(baseline_metrics['annualized_return_252']):.2%}|{float(selected_metrics['annualized_return_252']):.2%}|{annual_delta:+.2%}|
|Sharpe|{float(baseline_metrics['sharpe']):.3f}|{float(selected_metrics['sharpe']):.3f}|{sharpe_delta:+.3f}|
|最大回撤|{float(baseline_metrics['max_drawdown']):.2%}|{float(selected_metrics['max_drawdown']):.2%}|{mdd_delta:+.2%}|

Validation门槛：{'通过' if validation['passed'] else '未通过'}。CSCV-PBO为
{pbo_summary['pbo']:.1%}；年度Reality Check p={float(reality['p_value']):.4f}；20日配对分块
Bootstrap中年化差95%区间为
[{float(bootstrap_summary['annualized_return_delta_ci_lower']):+.2%},
{float(bootstrap_summary['annualized_return_delta_ci_upper']):+.2%}]，Sharpe差95%区间为
[{float(bootstrap_summary['sharpe_delta_ci_lower']):+.3f},
{float(bootstrap_summary['sharpe_delta_ci_upper']):+.3f}]。

全联合网格中有{grid_holdout_summary['full_dominates_baseline']}组在全样本同时改善三指标，
但同时通过validation与recent预注册门槛的候选为
{grid_holdout_summary['both_holdout_gates_passed']}组；同时满足全样本三指标与两段门槛的候选为
{grid_holdout_summary['full_dominates_and_both_holdouts']}组。因此不能从全样本
{grid_holdout_summary['full_dominates_baseline']}个表面赢家中
事后挑选替代开发期候选。

## 决策

{'统计与分段门槛全部满足，但仍需用户明确决定才能晋升。' if promotion_supported else '当前证据不支持生产晋升；保留完整失败证据，不替换正式策略。'}
"""
    (stage / "research_report.md").write_text(report, encoding="utf-8")

    source_paths = [
        config_path,
        root / "factors/quality_momentum.py",
        root / "research/momentum_defender_log_qm_switch.py",
        root / "research/run_momentum_defender_log_qm_switch.py",
        root / "research/DEVELOPMENT_VALIDATION.md",
    ]
    manifest = {
        "experiment_id": config["experiment"]["id"],
        "generated_on": date.today().isoformat(),
        "sources": [
            {"path": str(path.relative_to(root)), "sha256": _sha256(path)}
            for path in source_paths
        ],
    }
    (stage / "experiment_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    output.mkdir()
    for path in stage.iterdir():
        path.replace(output / path.name)
    stage.rmdir()
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    root = args.root.resolve()
    config_path = args.config if args.config.is_absolute() else root / args.config
    output = args.output if args.output.is_absolute() else root / args.output
    audit = run_experiment(root, config_path, output)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
