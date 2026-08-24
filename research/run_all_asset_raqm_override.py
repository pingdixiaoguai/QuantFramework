"""Search a common RAQM factor with asset-specific override thresholds."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from itertools import product
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
import yaml

from research.all_asset_raqm_override import (
    AssetRAQMThresholds,
    CommonRAQMSpec,
    common_raqm_at_open,
    policy_set_id,
    run_all_asset_raqm,
)
from research.gold_min5_risk_adjusted_momentum_w5 import (
    GoldRAQMW5Params,
    run_gold_raqm_w5,
)
from research.momentum_defender_gold_override import build_gold_override_context
from research.momentum_defender_gold_override_overfit import (
    cscv_pbo,
    expanding_walk_forward,
    full_metrics,
    paired_block_bootstrap,
    yearly_reality_check,
)
from research.momentum_defender_occam import HELD_RETURN, MOMENTUM_ASSETS, performance
from research.standard_report import generate_standard_report


DEFAULT_CONFIG = Path("research/configs/all_asset_raqm_override_search.yaml")
DEFAULT_OUTPUT = Path("experiments/20260824_all_asset_raqm_override")
FORMAL_GOLD_POLICY = AssetRAQMThresholds(2.20, 0.60)


def _threshold_grid(config: dict) -> list[AssetRAQMThresholds]:
    policies = []
    for entry, exit_ in product(
        config["threshold_grid"]["entry_differences"],
        config["threshold_grid"]["exit_differences"],
    ):
        if float(exit_) <= float(entry):
            policies.append(AssetRAQMThresholds(float(entry), float(exit_)))
    return policies


def _specs(config: dict) -> list[CommonRAQMSpec]:
    factor = config["factor_grid"]
    return [
        CommonRAQMSpec(
            window=int(window),
            efficiency_power=float(power),
            vol_floor_annual=float(factor["vol_floor_annual"]),
        )
        for window, power in product(
            factor["windows"], factor["efficiency_powers"]
        )
    ]


def _blank_policies() -> dict[str, AssetRAQMThresholds | None]:
    return {asset: None for asset in MOMENTUM_ASSETS}


def _return_hash(values: pd.Series) -> str:
    return hashlib.sha256(values.to_numpy(dtype="<f8").tobytes()).hexdigest()


def _json_clean(value):
    if isinstance(value, dict):
        return {str(key): _json_clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_clean(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _attach_period_metrics(
    returns: pd.DataFrame,
    baseline: pd.Series,
    periods: Mapping[str, tuple[pd.Timestamp, pd.Timestamp]],
) -> pd.DataFrame:
    result = full_metrics(returns, baseline)
    for label, (start, end) in periods.items():
        period = full_metrics(returns.loc[start:end], baseline.loc[start:end])
        for field in (
            "annualized_return_252",
            "sharpe",
            "max_drawdown",
            "delta_annualized_return_252",
            "delta_sharpe",
            "delta_max_drawdown",
        ):
            result[f"{label}_{field}"] = period[field]
    result["worst_split_sharpe"] = result[
        ["development_sharpe", "validation_sharpe", "recent_sharpe"]
    ].min(axis=1)
    return result


def _unique_paths(returns: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    representatives: dict[str, str] = {}
    rows = []
    for candidate in returns.columns:
        digest = _return_hash(returns[candidate])
        representative = representatives.setdefault(digest, str(candidate))
        rows.append(
            {
                "candidate_id": str(candidate),
                "return_path_sha256": digest,
                "representative_candidate_id": representative,
                "is_representative": representative == str(candidate),
            }
        )
    mapping = pd.DataFrame(rows).set_index("candidate_id")
    unique = returns.loc[:, mapping["is_representative"].to_numpy(bool)]
    return mapping, unique


def _policies_from_row(row: pd.Series) -> dict[str, AssetRAQMThresholds | None]:
    policies: dict[str, AssetRAQMThresholds | None] = {}
    for asset in MOMENTUM_ASSETS:
        entry = row[f"entry_{asset}"]
        exit_ = row[f"exit_{asset}"]
        policies[asset] = (
            None
            if pd.isna(entry)
            else AssetRAQMThresholds(float(entry), float(exit_))
        )
    return policies


def _spec_from_row(row: pd.Series) -> CommonRAQMSpec:
    return CommonRAQMSpec(
        window=int(row["window"]),
        efficiency_power=float(row["efficiency_power"]),
        vol_floor_annual=float(row["vol_floor_annual"]),
    )


def _metadata_row(
    spec: CommonRAQMSpec,
    policies: Mapping[str, AssetRAQMThresholds | None],
    audit: Mapping[str, object],
) -> dict[str, object]:
    row: dict[str, object] = {
        "candidate_id": policy_set_id(spec, policies),
        **asdict(spec),
        "raqm_entries": audit["raqm_entries"],
        "raqm_rotations": audit["raqm_rotations"],
        "raqm_days": audit["raqm_days"],
        "switches": audit["switches"],
    }
    for asset in MOMENTUM_ASSETS:
        policy = policies[asset]
        row[f"entry_{asset}"] = (
            policy.entry_difference if policy is not None else np.nan
        )
        row[f"exit_{asset}"] = (
            policy.exit_difference if policy is not None else np.nan
        )
        row[f"raqm_days_{asset}"] = audit["raqm_asset_days"].get(asset, 0)
        row[f"raqm_entries_{asset}"] = audit["raqm_entry_assets"].get(asset, 0)
    return row


def _select_distinct_policy_options(
    metrics: pd.DataFrame,
    returns: pd.DataFrame,
    policy_grid: list[AssetRAQMThresholds],
    count: int,
) -> list[AssetRAQMThresholds]:
    working = metrics.copy()
    working["return_hash"] = [
        _return_hash(returns[column]) for column in working.index
    ]
    ordered = working.sort_values(
        ["development_sharpe", "development_annualized_return_252"],
        ascending=False,
    ).drop_duplicates("return_hash")
    result = []
    by_id = {policy.policy_id(): policy for policy in policy_grid}
    for _, row in ordered.head(count).iterrows():
        result.append(by_id[str(row["policy_id"])])
    return result


def _event_attribution(
    state: pd.DataFrame,
    candidate: pd.Series,
    baseline: pd.Series,
) -> pd.DataFrame:
    active = state["raqm_active"].astype(bool)
    groups = active.ne(active.shift()).cumsum()
    rows = []
    for episode, (_, sample) in enumerate(
        state.loc[active].groupby(groups.loc[active]), start=1
    ):
        start = state.index.get_loc(sample.index.min())
        finish = min(state.index.get_loc(sample.index.max()) + 1, len(state) - 1)
        interval = state.index[start : finish + 1]
        candidate_return = float((1.0 + candidate.loc[interval]).prod() - 1.0)
        baseline_return = float((1.0 + baseline.loc[interval]).prod() - 1.0)
        asset_sequence = "→".join(
            sample["raqm_active_asset"]
            .astype(str)
            .loc[lambda values: values.ne(values.shift())]
            .tolist()
        )
        rows.append(
            {
                "episode": episode,
                "start": interval.min().date().isoformat(),
                "end_including_exit": interval.max().date().isoformat(),
                "observations": int(len(interval)),
                "asset_sequence": asset_sequence,
                "candidate_return": candidate_return,
                "formal_baseline_return": baseline_return,
                "relative_return": (1.0 + candidate_return)
                / (1.0 + baseline_return)
                - 1.0,
            }
        )
    return pd.DataFrame(rows)


def _leave_one_event(
    events: pd.DataFrame,
    candidate: pd.Series,
    baseline: pd.Series,
) -> pd.DataFrame:
    rows = []
    for event in events.itertuples(index=False):
        counterfactual = candidate.copy()
        interval = counterfactual.loc[
            pd.Timestamp(event.start) : pd.Timestamp(event.end_including_exit)
        ].index
        counterfactual.loc[interval] = baseline.loc[interval]
        rows.append({"removed_episode": int(event.episode), **performance(counterfactual)})
    return pd.DataFrame(rows)


def _cost_stress(
    candidate_daily: pd.DataFrame,
    baseline_daily: pd.DataFrame,
    multipliers: Iterable[int],
) -> pd.DataFrame:
    rows = []
    for multiplier in multipliers:
        extra = float(int(multiplier) - 1)
        candidate = (
            (1.0 + candidate_daily["return"])
            * (1.0 - candidate_daily["cost_rate_at_open"]).pow(extra)
            - 1.0
        )
        baseline = (
            (1.0 + baseline_daily["return"])
            * (1.0 - baseline_daily["cost_rate_at_open"]).pow(extra)
            - 1.0
        )
        cm = performance(candidate)
        bm = performance(baseline)
        rows.append(
            {
                "cost_multiplier": int(multiplier),
                "candidate_annualized_return_252": cm["annualized_return_252"],
                "formal_annualized_return_252": bm["annualized_return_252"],
                "annualized_return_delta": cm["annualized_return_252"]
                - bm["annualized_return_252"],
                "candidate_sharpe": cm["sharpe"],
                "formal_sharpe": bm["sharpe"],
                "sharpe_delta": cm["sharpe"] - bm["sharpe"],
                "candidate_max_drawdown": cm["max_drawdown"],
                "formal_max_drawdown": bm["max_drawdown"],
            }
        )
    return pd.DataFrame(rows)


def _annual_returns(strategies: Mapping[str, pd.Series]) -> pd.DataFrame:
    rows = []
    for strategy, returns in strategies.items():
        for year, sample in returns.groupby(returns.index.year):
            rows.append(
                {
                    "strategy": strategy,
                    "year": int(year),
                    "observations": int(len(sample)),
                    "total_return": float((1.0 + sample).prod() - 1.0),
                }
            )
    return pd.DataFrame(rows)


def run_experiment(root: Path, config_path: Path, output: Path) -> dict[str, object]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    context = build_gold_override_context(root)
    formal_run = run_gold_raqm_w5(context, GoldRAQMW5Params(2.20, 0.60))
    formal = formal_run.daily["return"].astype(float)
    current_c2 = context.integrated.result.simulated["return"].astype(float)
    original_momentum = context.integrated.result.inputs.momentum[
        HELD_RETURN
    ].astype(float)
    periods = {
        label: (pd.Timestamp(values[0]), pd.Timestamp(values[1]))
        for label, values in config["periods"].items()
    }
    specs = _specs(config)
    policy_grid = _threshold_grid(config)
    metrics_cache = {
        spec.spec_id(): common_raqm_at_open(context.curves, spec) for spec in specs
    }

    single_metric_frames = []
    options_by_spec: dict[
        str, dict[str, list[AssetRAQMThresholds | None]]
    ] = {}
    top_count = int(config["staged_selection"]["policies_per_asset_per_spec"])
    for spec in specs:
        spec_options: dict[str, list[AssetRAQMThresholds | None]] = {}
        applied_metrics = metrics_cache[spec.spec_id()]
        for asset in MOMENTUM_ASSETS:
            records = []
            return_values: dict[str, np.ndarray] = {}
            for policy in policy_grid:
                policies = _blank_policies()
                policies[asset] = policy
                run = run_all_asset_raqm(
                    context, spec, policies, metrics=applied_metrics
                )
                candidate_id = f"{spec.spec_id()}|{asset}|{policy.policy_id()}"
                records.append(
                    {
                        "candidate_id": candidate_id,
                        "asset": asset,
                        "policy_id": policy.policy_id(),
                        **asdict(spec),
                        **asdict(policy),
                        "raqm_entries": run.audit["raqm_entries"],
                        "raqm_days": run.audit["raqm_days"],
                    }
                )
                return_values[candidate_id] = run.daily["return"].to_numpy(float)
            asset_returns = pd.DataFrame(return_values, index=context.calendar)
            asset_records = pd.DataFrame(records).set_index("candidate_id")
            asset_metrics = _attach_period_metrics(
                asset_returns, formal, periods
            ).join(asset_records)
            single_metric_frames.append(asset_metrics)
            eligible = asset_metrics.loc[
                asset_metrics["raqm_entries"].ge(
                    int(config["staged_selection"]["minimum_single_asset_entries"])
                )
                & asset_metrics["raqm_days"].ge(
                    int(config["staged_selection"]["minimum_single_asset_days"])
                )
            ]
            selection_pool = eligible if not eligible.empty else asset_metrics
            selected_options = _select_distinct_policy_options(
                selection_pool,
                asset_returns.loc[:, selection_pool.index],
                policy_grid,
                top_count,
            )
            if (
                spec.window == 5
                and spec.efficiency_power == 1.0
                and asset == "518880.SH"
                and FORMAL_GOLD_POLICY.policy_id()
                not in {policy.policy_id() for policy in selected_options}
            ):
                selected_options.append(FORMAL_GOLD_POLICY)
            spec_options[asset] = [None, *selected_options]
        options_by_spec[spec.spec_id()] = spec_options

    single_metrics = pd.concat(single_metric_frames)
    combo_records = []
    combo_return_values: dict[str, np.ndarray] = {}
    combo_policy_sets: dict[str, dict[str, AssetRAQMThresholds | None]] = {}
    spec_lookup = {spec.spec_id(): spec for spec in specs}
    for spec_id, option_map in options_by_spec.items():
        spec = spec_lookup[spec_id]
        applied_metrics = metrics_cache[spec_id]
        for selected in product(*(option_map[asset] for asset in MOMENTUM_ASSETS)):
            policies = dict(zip(MOMENTUM_ASSETS, selected, strict=True))
            candidate_id = policy_set_id(spec, policies)
            if candidate_id in combo_return_values:
                continue
            run = run_all_asset_raqm(
                context, spec, policies, metrics=applied_metrics
            )
            combo_policy_sets[candidate_id] = policies
            metadata_row = _metadata_row(spec, policies, run.audit)
            metadata_row["search_stage"] = "staged_combination"
            combo_records.append(metadata_row)
            combo_return_values[candidate_id] = run.daily["return"].to_numpy(float)

    refinement = config["local_refinement"]
    refinement_spec = CommonRAQMSpec(**refinement["common_spec"])
    if refinement_spec.spec_id() not in metrics_cache:
        raise AssertionError("local refinement common spec is outside factor grid")
    refinement_options = {
        asset: [AssetRAQMThresholds(**values) for values in refinement["policies"][asset]]
        for asset in MOMENTUM_ASSETS
    }
    for selected in product(
        *(refinement_options[asset] for asset in MOMENTUM_ASSETS)
    ):
        policies = dict(zip(MOMENTUM_ASSETS, selected, strict=True))
        candidate_id = policy_set_id(refinement_spec, policies)
        if candidate_id in combo_return_values:
            continue
        run = run_all_asset_raqm(
            context,
            refinement_spec,
            policies,
            metrics=metrics_cache[refinement_spec.spec_id()],
        )
        combo_policy_sets[candidate_id] = policies
        metadata_row = _metadata_row(refinement_spec, policies, run.audit)
        metadata_row["search_stage"] = "local_refinement"
        combo_records.append(metadata_row)
        combo_return_values[candidate_id] = run.daily["return"].to_numpy(float)
    combo_returns = pd.DataFrame(combo_return_values, index=context.calendar)
    combo_metadata = pd.DataFrame(combo_records).set_index("candidate_id")
    combo_metrics = _attach_period_metrics(combo_returns, formal, periods).join(
        combo_metadata
    )

    control_spec = CommonRAQMSpec(window=5, efficiency_power=1.0)
    control_policies = _blank_policies()
    control_policies["518880.SH"] = FORMAL_GOLD_POLICY
    control_id = policy_set_id(control_spec, control_policies)
    if control_id not in combo_returns:
        raise AssertionError("formal Gold control missing from combination grid")
    parity_error = float((combo_returns[control_id] - formal).abs().max())
    if parity_error > 1e-12:
        raise AssertionError(f"formal Gold parity failed: {parity_error:.3e}")

    eligible_combo = combo_metrics.loc[
        combo_metrics["raqm_entries"].ge(2) & combo_metrics["raqm_days"].ge(10)
    ]
    entry_columns = [f"entry_{asset}" for asset in MOMENTUM_ASSETS]
    strict_all_enabled = eligible_combo.loc[
        eligible_combo[entry_columns].notna().all(axis=1)
    ]
    if strict_all_enabled.empty:
        raise AssertionError("combination grid contains no all-assets-enabled candidate")
    development_selected = strict_all_enabled.sort_values(
        ["development_sharpe", "development_annualized_return_252"],
        ascending=False,
    ).iloc[0]
    strict_core_improved = strict_all_enabled.loc[
        strict_all_enabled["delta_annualized_return_252"].gt(0)
        & strict_all_enabled["delta_sharpe"].gt(0)
        & strict_all_enabled["delta_max_drawdown"].ge(-1e-12)
    ]
    observed_pool = (
        strict_core_improved
        if not strict_core_improved.empty
        else strict_all_enabled
    )
    observed = observed_pool.sort_values(
        ["annualized_return_252", "sharpe"], ascending=False
    ).iloc[0]
    subset_core_improved = eligible_combo.loc[
        eligible_combo["delta_annualized_return_252"].gt(0)
        & eligible_combo["delta_sharpe"].gt(0)
        & eligible_combo["delta_max_drawdown"].ge(-1e-12)
    ]
    diagnostic_subset = (
        subset_core_improved
        if not subset_core_improved.empty
        else eligible_combo
    ).sort_values(["annualized_return_252", "sharpe"], ascending=False).iloc[0]
    observed_id = str(observed.name)
    selected_spec = _spec_from_row(observed)
    selected_policies = _policies_from_row(observed)
    selected_run = run_all_asset_raqm(
        context,
        selected_spec,
        selected_policies,
        metrics=metrics_cache[selected_spec.spec_id()],
    )

    path_mapping, unique_returns = _unique_paths(combo_returns)
    cscv, pbo = cscv_pbo(
        unique_returns,
        formal,
        block_count=int(config["robustness"]["cscv_blocks"]),
    )
    walk = expanding_walk_forward(unique_returns, formal)
    bootstrap, bootstrap_summary = paired_block_bootstrap(
        combo_returns[observed_id],
        formal,
        repetitions=int(config["robustness"]["bootstrap_repetitions"]),
    )
    reality = yearly_reality_check(
        unique_returns,
        formal,
        repetitions=int(config["robustness"]["reality_check_repetitions"]),
    )
    events = _event_attribution(
        selected_run.state, selected_run.daily["return"], formal
    )
    leave_one = _leave_one_event(events, selected_run.daily["return"], formal)
    cost_stress = _cost_stress(
        selected_run.daily,
        formal_run.daily,
        config["robustness"]["cost_multipliers"],
    )

    sensitivity_rows = []
    selected_metrics_frame = metrics_cache[selected_spec.spec_id()]
    for asset in MOMENTUM_ASSETS:
        for replacement in [None, *policy_grid]:
            policies = dict(selected_policies)
            policies[asset] = replacement
            run = run_all_asset_raqm(
                context,
                selected_spec,
                policies,
                metrics=selected_metrics_frame,
            )
            measured = performance(run.daily["return"])
            sensitivity_rows.append(
                {
                    "changed_asset": asset,
                    "replacement_policy": (
                        replacement.policy_id() if replacement else "off"
                    ),
                    **measured,
                    "delta_annualized_return_vs_formal": measured[
                        "annualized_return_252"
                    ]
                    - performance(formal)["annualized_return_252"],
                    "delta_sharpe_vs_formal": measured["sharpe"]
                    - performance(formal)["sharpe"],
                    "delta_mdd_vs_formal": measured["max_drawdown"]
                    - performance(formal)["max_drawdown"],
                }
            )
    sensitivity = pd.DataFrame(sensitivity_rows)

    improvement_counts = {
        "annualized_return": int(
            strict_all_enabled["delta_annualized_return_252"].gt(0).sum()
        ),
        "sharpe": int(strict_all_enabled["delta_sharpe"].gt(0).sum()),
        "max_drawdown_not_worse": int(
            strict_all_enabled["delta_max_drawdown"].ge(-1e-12).sum()
        ),
        "all_three": int(len(strict_core_improved)),
    }
    subset_improvement_counts = {
        "annualized_return": int(
            eligible_combo["delta_annualized_return_252"].gt(0).sum()
        ),
        "sharpe": int(eligible_combo["delta_sharpe"].gt(0).sum()),
        "max_drawdown_not_worse": int(
            eligible_combo["delta_max_drawdown"].ge(-1e-12).sum()
        ),
        "all_three": int(len(subset_core_improved)),
    }
    positive_event_log = np.log1p(
        events.loc[events["relative_return"].gt(0), "relative_return"]
    )
    positive_total = float(positive_event_log.sum())
    top_event_share = (
        float(positive_event_log.nlargest(2).sum() / positive_total)
        if positive_total > 0.0
        else np.nan
    )
    overfit_flags = {
        "reality_check_not_significant_10pct": float(reality["p_value"]) >= 0.10,
        "bootstrap_return_probability_below_90pct": float(
            bootstrap_summary["annualized_return_delta_positive_probability"]
        )
        < 0.90,
        "bootstrap_sharpe_probability_below_90pct": float(
            bootstrap_summary["sharpe_delta_positive_probability"]
        )
        < 0.90,
        "walk_forward_return_win_below_half": float(
            walk["test_return_delta"].gt(0).mean()
        )
        < 0.5,
        "top_two_event_share_above_half": bool(
            pd.notna(top_event_share) and top_event_share > 0.5
        ),
    }
    overfit_assessment = (
        "high"
        if sum(overfit_flags.values()) >= 3
        else "moderate" if any(overfit_flags.values()) else "low"
    )

    output.mkdir(parents=True, exist_ok=True)
    single_metrics.to_csv(output / "single_asset_candidates.csv")
    combo_metrics.sort_values(
        ["annualized_return_252", "sharpe"], ascending=False
    ).to_csv(output / "combination_candidates.csv")
    combo_metrics.loc[combo_metrics["search_stage"].eq("local_refinement")].sort_values(
        ["annualized_return_252", "sharpe"], ascending=False
    ).to_csv(output / "local_refinement_candidates.csv")
    factor_spec_best = (
        strict_all_enabled.sort_values(
            ["annualized_return_252", "sharpe"], ascending=False
        )
        .groupby(["window", "efficiency_power"], as_index=False)
        .first()
        .sort_values("annualized_return_252", ascending=False)
    )
    factor_spec_best.to_csv(output / "factor_spec_best.csv", index=False)
    path_mapping.to_csv(output / "return_path_mapping.csv")
    selected_run.state.join(
        selected_run.daily, rsuffix="_execution"
    ).to_csv(output / "daily_observed_candidate.csv")
    events.to_csv(output / "override_episodes.csv", index=False)
    leave_one.to_csv(output / "leave_one_event.csv", index=False)
    cost_stress.to_csv(output / "cost_stress.csv", index=False)
    sensitivity.to_csv(output / "one_at_a_time_threshold_sensitivity.csv", index=False)
    cscv.to_csv(output / "cscv_pbo.csv", index=False)
    walk.to_csv(output / "expanding_walk_forward.csv", index=False)
    bootstrap.to_csv(output / "paired_block_bootstrap.csv", index=False)
    pd.DataFrame(
        [
            {"strategy": name, **performance(values)}
            for name, values in {
                "observed_all_asset_raqm": selected_run.daily["return"],
                "formal_gold_raqm_w5": formal,
                "current_c2": current_c2,
                "original_momentum": original_momentum,
            }.items()
        ]
    ).to_csv(output / "strategy_metrics.csv", index=False)
    _annual_returns(
        {
            "observed_all_asset_raqm": selected_run.daily["return"],
            "formal_gold_raqm_w5": formal,
            "current_c2": current_c2,
            "original_momentum": original_momentum,
        }
    ).to_csv(output / "calendar_year_returns.csv", index=False)
    (output / "strategy_config.yaml").write_text(
        config_path.read_text(encoding="utf-8"), encoding="utf-8"
    )

    report_config = {
        "experiment_id": config["experiment"]["id"],
        "candidate_id": observed_id,
        "common_factor": asdict(selected_spec),
        "asset_policies": {
            asset: asdict(policy) if policy else None
            for asset, policy in selected_policies.items()
        },
        "production_replacement": False,
    }
    generate_standard_report(
        selected_run.daily["return"],
        formal,
        "Formal Gold RAQM-W5",
        output / "observed_vs_formal_strategy.html",
        report_config,
    )
    generate_standard_report(
        selected_run.daily["return"],
        original_momentum,
        "Original Momentum",
        output / "observed_vs_original_momentum.html",
        report_config,
    )
    generate_standard_report(
        selected_run.daily["return"],
        current_c2,
        "Current Integrated C2",
        output / "observed_vs_current_c2.html",
        report_config,
    )

    formal_metrics = performance(formal)
    selected_metrics = combo_metrics.loc[observed_id]
    summary = {
        "experiment_id": config["experiment"]["id"],
        "single_asset_candidate_count": int(len(single_metrics)),
        "combination_candidate_count": int(len(combo_metrics)),
        "staged_combination_candidate_count": int(
            combo_metrics["search_stage"].eq("staged_combination").sum()
        ),
        "local_refinement_candidate_count": int(
            combo_metrics["search_stage"].eq("local_refinement").sum()
        ),
        "strict_all_assets_enabled_candidate_count": int(len(strict_all_enabled)),
        "unique_combination_return_paths": int(len(unique_returns.columns)),
        "formal_control_id": control_id,
        "formal_control_parity_max_abs_error": parity_error,
        "formal_baseline": formal_metrics,
        "development_selected_id": str(development_selected.name),
        "development_selected": development_selected.to_dict(),
        "observed_candidate_id": observed_id,
        "observed_candidate": selected_metrics.to_dict(),
        "diagnostic_subset_candidate_id": str(diagnostic_subset.name),
        "diagnostic_subset_candidate": diagnostic_subset.to_dict(),
        "selected_common_factor": asdict(selected_spec),
        "selected_asset_policies": {
            asset: asdict(policy) if policy else None
            for asset, policy in selected_policies.items()
        },
        "selected_audit": selected_run.audit,
        "improvement_counts": improvement_counts,
        "subset_allowed_improvement_counts": subset_improvement_counts,
        "event_count": int(len(events)),
        "top_two_positive_event_share": top_event_share,
        "leave_one_event_minimum_annualized_return": (
            float(leave_one["annualized_return_252"].min())
            if not leave_one.empty
            else None
        ),
        "leave_one_event_minimum_sharpe": (
            float(leave_one["sharpe"].min()) if not leave_one.empty else None
        ),
        "pbo": pbo,
        "walk_forward_return_win_rate": float(
            walk["test_return_delta"].gt(0).mean()
        ),
        "walk_forward_sharpe_win_rate": float(
            walk["test_sharpe_delta"].gt(0).mean()
        ),
        "bootstrap": bootstrap_summary,
        "reality_check": reality,
        "overfit_flags": overfit_flags,
        "overfit_assessment": overfit_assessment,
        "production_replacement": False,
    }
    summary = _json_clean(summary)
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )

    policy_lines = "\n".join(
        f"- {asset}: `{policy.policy_id()}`" if policy else f"- {asset}: 关闭"
        for asset, policy in selected_policies.items()
    )
    report = f"""# 四Momentum资产共同RAQM覆盖研究

## 固定边界

基础C2、慢门、30日锁、emergency、Momentum轮动、费用和开盘执行全部不变。RAQM覆盖层
沿用5日硬持有状态机，只把可覆盖标的从Gold推广到四只Momentum ETF。同一候选的窗口、
波动率地板和效率比指数完全相同；仅入场/退出差值阈值可按资产不同。

关闭其余资产、使用当前Gold W5/ER^1/2.20/0.60时，对正式策略逐日最大误差
{parity_error:.3e}。

## 严格四资产回溯观察冠军

共同因子：window={selected_spec.window}，efficiency_power={selected_spec.efficiency_power:.1f}，
vol_floor_annual={selected_spec.vol_floor_annual:.2f}。

{policy_lines}

- 年化{float(selected_metrics['annualized_return_252']):.2%}，相对正式策略
  {float(selected_metrics['delta_annualized_return_252']):+.2%}。
- Sharpe {float(selected_metrics['sharpe']):.3f}，相对正式策略
  {float(selected_metrics['delta_sharpe']):+.3f}。
- MDD {float(selected_metrics['max_drawdown']):.2%}，相对正式策略
  {float(selected_metrics['delta_max_drawdown']):+.2%}。
- 覆盖{int(selected_metrics['raqm_entries'])}次、{int(selected_metrics['raqm_days'])}日。

共评估{len(single_metrics)}个单资产候选、{len(combo_metrics)}个组合、
{len(unique_returns.columns)}条唯一组合收益路径；其中四只资产全部启用的严格候选
{len(strict_all_enabled)}个，同时改善年化、Sharpe和MDD的严格候选有
{improvement_counts['all_three']}个。允许关闭部分资产时，三项同时改善的诊断候选有
{subset_improvement_counts['all_three']}个，但不满足本实验“四资产均覆盖”的目标。

第一阶段之后追加了{int(combo_metrics['search_stage'].eq('local_refinement').sum())}组
W5/ER^1严格全启用局部组合；该步骤使用了已观察结果，已经计入全部候选和多重试验校正，
不能称为预注册或样本外。

## 稳健性

- PBO {float(pbo['pbo']):.1%}；年度块多重试验校正p={float(reality['p_value']):.3f}。
- walk-forward收益/Sharpe胜率分别为
  {summary['walk_forward_return_win_rate']:.1%}/{summary['walk_forward_sharpe_win_rate']:.1%}。
- 分块bootstrap年化差为正概率
  {bootstrap_summary['annualized_return_delta_positive_probability']:.1%}，95%区间
  [{bootstrap_summary['annualized_return_delta_ci_lower']:+.2%},
  {bootstrap_summary['annualized_return_delta_ci_upper']:+.2%}]。
- 事件数{len(events)}，前两大正事件占正贡献{top_event_share:.1%}。
- 综合过拟合风险：**{overfit_assessment.upper()}**。

## 决策

严格四资产历史组合可以小幅改善核心指标，但阈值与共同因子经过两阶段回溯选择，增益很薄，
并且稳健性审计不支持自动替换正式策略。当前结论为保留研究候选、不晋升；未来是否采用取决于
独立前瞻事件，而不只看全样本冠军。
"""
    (output / "report.md").write_text(report, encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    root = args.root.resolve()
    config = args.config if args.config.is_absolute() else root / args.config
    output = args.output if args.output.is_absolute() else root / args.output
    summary = run_experiment(root, config, output)
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
