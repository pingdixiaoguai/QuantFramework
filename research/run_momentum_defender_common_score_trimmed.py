"""Search common-score selected-asset DRAQM on candidate-independent ordinary blocks."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from factors.quality_momentum import METADATA as QUALITY_METADATA
from research.momentum_defender_common_score_trimmed import (
    ExtremeBlockSpec,
    build_extreme_block_mask,
    validate_common_score_policies,
)
from research.momentum_defender_downside_raqm import (
    DownsideRAQMSpec,
    FactorProfile,
    build_downside_raqm_features,
    build_exact_execution_data,
    exact_candidate_schedule,
    run_downside_raqm_spec,
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
from research.momentum_defender_log_qm_switch import pareto_frontier
from research.momentum_defender_occam import performance
from research.momentum_defender_selected_asset_draqm import (
    STICKY_ENTRY_ASSET,
    AssetDRAQMPolicy,
    SelectedAssetDRAQMSpec,
    run_selected_asset_draqm_spec,
)
from research.momentum_volatility import load_ohlc
from research.run_momentum_defender_log_qm_robust import (
    _event_stress,
    _friction,
    _selected_cost_schedule,
    _unique_paths,
)
from research.standard_report import generate_standard_report


DEFAULT_CONFIG = Path(
    "research/configs/momentum_defender_common_score_trimmed_search.yaml"
)
DEFAULT_OUTPUT = Path(
    "experiments/20260824_momentum_defender_common_score_trimmed"
)
ASSETS = ("510300.SH", "518880.SH")


def _load_config(path: Path) -> dict:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("common-score trimmed config must be a mapping")
    return config


def _profiles(config: dict) -> dict[str, FactorProfile]:
    return {
        profile_id: FactorProfile(
            profile_id,
            tuple(map(int, values["horizons"])),
            tuple(map(float, values["weights"])),
        )
        for profile_id, values in config["common_score"]["profiles"].items()
    }


def _policy_grid(
    config: dict,
    profiles: dict[str, FactorProfile],
) -> dict[str, dict[str, list[AssetDRAQMPolicy]]]:
    search = config["single_asset_stage"]
    minimum_gap = float(search["minimum_hysteresis_gap"])
    result = {asset: {profile: [] for profile in profiles} for asset in ASSETS}
    for asset in ASSETS:
        for profile_id, profile in profiles.items():
            unique = {}
            for values in product(
                search["defender_entry_percentiles"],
                search["momentum_recovery_percentiles"],
                search["defender_entry_confirmation_days"],
                search["momentum_recovery_confirmation_days"],
            ):
                entry, recovery, entry_c, recovery_c = values
                if float(entry) - float(recovery) + 1e-12 < minimum_gap:
                    continue
                policy = AssetDRAQMPolicy(
                    asset,
                    profile,
                    float(entry),
                    float(recovery),
                    int(entry_c),
                    int(recovery_c),
                )
                unique[policy.policy_id()] = policy
            result[asset][profile_id] = list(unique.values())
    return result


def _add_metrics(
    metadata: pd.DataFrame,
    returns: pd.DataFrame,
    baseline: pd.Series,
    selection_mask: pd.Series,
    config: dict,
) -> pd.DataFrame:
    result = metadata.join(full_metrics(returns, baseline).add_prefix("full_"))
    ordinary = selection_mask.reindex(returns.index).fillna(False).astype(bool)
    result = result.join(
        full_metrics(returns.loc[ordinary], baseline.loc[ordinary]).add_prefix(
            "trimmed_"
        )
    )
    for period in ("development", "validation", "recent"):
        start, end = map(pd.Timestamp, config["periods"][period])
        period_mask = returns.index.to_series().between(start, end).to_numpy()
        period_returns = returns.loc[period_mask]
        period_base = baseline.loc[period_mask]
        metrics = full_metrics(period_returns, period_base)
        for field in ("annualized_return_252", "sharpe", "max_drawdown"):
            result[f"{period}_{field}"] = metrics[field]
        trimmed_mask = ordinary & period_mask
        trimmed_metrics = full_metrics(
            returns.loc[trimmed_mask], baseline.loc[trimmed_mask]
        )
        for field in ("annualized_return_252", "sharpe"):
            result[f"trimmed_{period}_{field}"] = trimmed_metrics[field]
    result["trimmed_minimum_segment_sharpe"] = result[
        [
            "trimmed_development_sharpe",
            "trimmed_validation_sharpe",
            "trimmed_recent_sharpe",
        ]
    ].min(axis=1)
    return result


def _evaluate_single(
    data,
    momentum_target,
    features,
    policy_grid,
    baseline,
    selection_mask,
    config,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, AssetDRAQMPolicy]]:
    records = []
    returns = {}
    lookup = {}
    stage = config["single_asset_stage"]
    total = sum(
        len(policy_grid[asset][profile])
        for asset in ASSETS
        for profile in policy_grid[asset]
    )
    completed = 0
    for asset in ASSETS:
        for profile_id, policies in policy_grid[asset].items():
            for policy in policies:
                spec = SelectedAssetDRAQMSpec(
                    {candidate: policy if candidate == asset else None for candidate in ASSETS},
                    int(stage["fixed_momentum_lock_days"]),
                    int(stage["fixed_defender_lock_days"]),
                    str(stage["fixed_recovery_reference"]),
                )
                run = run_selected_asset_draqm_spec(
                    data, momentum_target, features, spec
                )
                candidate_id = policy.policy_id()
                lookup[candidate_id] = policy
                returns[candidate_id] = run.returns
                records.append(
                    {
                        "candidate_id": candidate_id,
                        "asset": asset,
                        "profile_id": profile_id,
                        "horizons": "|".join(map(str, policy.profile.horizons)),
                        "weights": "|".join(
                            f"{value:.3f}" for value in policy.profile.weights
                        ),
                        "entry_percentile": policy.entry_percentile,
                        "recovery_percentile": policy.recovery_percentile,
                        "entry_confirmation_days": policy.entry_confirmation_days,
                        "recovery_confirmation_days": policy.recovery_confirmation_days,
                        "defender_entries": run.defender_entries,
                        "defender_days": run.defender_days,
                    }
                )
                completed += 1
                if completed % 500 == 0 or completed == total:
                    print(f"single: evaluated {completed}/{total}", flush=True)
    frame = pd.DataFrame(returns, index=data.calendar)
    metadata = pd.DataFrame(records).set_index("candidate_id")
    return (
        _add_metrics(metadata, frame, baseline, selection_mask, config),
        frame,
        lookup,
    )


def _threshold_neighborhood(table: pd.DataFrame, config: dict) -> pd.DataFrame:
    search = config["single_asset_stage"]
    dimensions = {
        "entry_percentile": list(map(float, search["defender_entry_percentiles"])),
        "recovery_percentile": list(
            map(float, search["momentum_recovery_percentiles"])
        ),
        "entry_confirmation_days": list(
            map(int, search["defender_entry_confirmation_days"])
        ),
        "recovery_confirmation_days": list(
            map(int, search["momentum_recovery_confirmation_days"])
        ),
    }
    result = table.copy()
    positions = []
    for field, values in dimensions.items():
        column = f"_{field}_position"
        result[column] = result[field].map(
            {value: position for position, value in enumerate(values)}
        )
        positions.append(column)
    rows = {}
    for _, group in result.groupby(["asset", "profile_id"], sort=False):
        coords = group[positions].to_numpy(int)
        full_annual = group["full_annualized_return_252"].to_numpy(float)
        full_sharpe = group["full_sharpe"].to_numpy(float)
        annual = group["trimmed_annualized_return_252"].to_numpy(float)
        sharpe = group["trimmed_sharpe"].to_numpy(float)
        for position, candidate_id in enumerate(group.index):
            members = np.all(np.abs(coords - coords[position]) <= 1, axis=1)
            rows[str(candidate_id)] = {
                "threshold_neighborhood_count": int(members.sum()),
                "threshold_neighborhood_full_annualized_q25": float(
                    np.quantile(full_annual[members], 0.25)
                ),
                "threshold_neighborhood_full_annualized_median": float(
                    np.median(full_annual[members])
                ),
                "threshold_neighborhood_full_sharpe_q25": float(
                    np.quantile(full_sharpe[members], 0.25)
                ),
                "threshold_neighborhood_full_sharpe_median": float(
                    np.median(full_sharpe[members])
                ),
                "threshold_neighborhood_trimmed_annualized_q25": float(
                    np.quantile(annual[members], 0.25)
                ),
                "threshold_neighborhood_trimmed_annualized_median": float(
                    np.median(annual[members])
                ),
                "threshold_neighborhood_trimmed_sharpe_q25": float(
                    np.quantile(sharpe[members], 0.25)
                ),
                "threshold_neighborhood_trimmed_sharpe_median": float(
                    np.median(sharpe[members])
                ),
            }
    return result.drop(columns=positions).join(
        pd.DataFrame.from_dict(rows, orient="index")
    )


def _select_options(
    table: pd.DataFrame,
    lookup: dict[str, AssetDRAQMPolicy],
    config: dict,
) -> dict[str, dict[str, list[AssetDRAQMPolicy]]]:
    stage = config["single_asset_stage"]
    count = int(stage["policies_per_asset_per_profile_for_joint_stage"])
    result = {asset: {} for asset in ASSETS}
    fields = list(stage["selection_metrics"])
    for asset in ASSETS:
        for profile_id in table.loc[table["asset"].eq(asset), "profile_id"].unique():
            pool = table.loc[
                table["asset"].eq(asset)
                & table["profile_id"].eq(profile_id)
                & table["defender_entries"].ge(int(stage["minimum_defender_entries"]))
                & table["defender_days"].ge(int(stage["minimum_defender_days"]))
            ].copy()
            ranks = pool[fields].rank(pct=True)
            pool["robust_min_percentile"] = ranks.min(axis=1)
            pool["robust_mean_percentile"] = ranks.mean(axis=1)
            selected = pool.sort_values(
                [
                    "robust_min_percentile",
                    "robust_mean_percentile",
                    "trimmed_annualized_return_252",
                    "trimmed_sharpe",
                ],
                ascending=False,
            ).head(count)
            result[asset][str(profile_id)] = [lookup[str(value)] for value in selected.index]
    return result


def _evaluate_joint(
    data,
    momentum_target,
    features,
    options,
    baseline,
    selection_mask,
    config,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, SelectedAssetDRAQMSpec]]:
    joint = config["joint_stage"]
    combinations = []
    for profile_id in options["510300.SH"]:
        for values in product(
            options["510300.SH"][profile_id],
            options["518880.SH"][profile_id],
            joint["momentum_lock_days"],
            joint["defender_lock_days"],
            joint["recovery_reference_modes"],
        ):
            combinations.append((profile_id, *values))
    matrix = np.empty((len(data.calendar), len(combinations)), dtype=np.float32)
    records = []
    ids = []
    specs = {}
    for position, values in enumerate(combinations):
        profile_id, csi, gold, momentum_hold, defender_hold, mode = values
        validate_common_score_policies({"510300.SH": csi, "518880.SH": gold})
        spec = SelectedAssetDRAQMSpec(
            {"510300.SH": csi, "518880.SH": gold},
            int(momentum_hold),
            int(defender_hold),
            str(mode),
        )
        run = run_selected_asset_draqm_spec(data, momentum_target, features, spec)
        candidate_id = spec.candidate_id()
        ids.append(candidate_id)
        specs[candidate_id] = spec
        matrix[:, position] = run.returns
        records.append(
            {
                "candidate_id": candidate_id,
                "profile_id": profile_id,
                "policy_510300": csi.policy_id(),
                "policy_518880": gold.policy_id(),
                "momentum_lock_days": int(momentum_hold),
                "defender_lock_days": int(defender_hold),
                "recovery_mode": str(mode),
                "defender_entries": run.defender_entries,
                "defender_days": run.defender_days,
                "sleeve_switches": run.sleeve_switches,
                "candidate_switches": run.candidate_switches,
            }
        )
        if (position + 1) % 250 == 0 or position + 1 == len(combinations):
            print(f"joint: evaluated {position + 1}/{len(combinations)}", flush=True)
    frame = pd.DataFrame(matrix, index=data.calendar, columns=ids)
    metadata = pd.DataFrame(records).set_index("candidate_id")
    return (
        _add_metrics(metadata, frame, baseline, selection_mask, config),
        frame,
        specs,
    )


def _joint_neighborhood(
    table: pd.DataFrame,
    single: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    joint = config["joint_stage"]
    result = table.copy()
    momentum_positions = {
        int(value): position for position, value in enumerate(joint["momentum_lock_days"])
    }
    defender_positions = {
        int(value): position for position, value in enumerate(joint["defender_lock_days"])
    }
    result["_mh"] = result["momentum_lock_days"].map(momentum_positions)
    result["_dh"] = result["defender_lock_days"].map(defender_positions)
    rows = {}
    for _, group in result.groupby(
        ["profile_id", "policy_510300", "policy_518880", "recovery_mode"],
        sort=False,
    ):
        coords = group[["_mh", "_dh"]].to_numpy(int)
        full_annual = group["full_annualized_return_252"].to_numpy(float)
        trimmed_annual = group["trimmed_annualized_return_252"].to_numpy(float)
        trimmed_sharpe = group["trimmed_sharpe"].to_numpy(float)
        for position, candidate_id in enumerate(group.index):
            members = np.all(np.abs(coords - coords[position]) <= 1, axis=1)
            rows[str(candidate_id)] = {
                "lock_neighborhood_count": int(members.sum()),
                "lock_neighborhood_full_annualized_pass_rate": float(
                    np.mean(full_annual[members] >= 0.45)
                ),
                "lock_neighborhood_trimmed_annualized_q25": float(
                    np.quantile(trimmed_annual[members], 0.25)
                ),
                "lock_neighborhood_trimmed_sharpe_q25": float(
                    np.quantile(trimmed_sharpe[members], 0.25)
                ),
            }
    result = result.drop(columns=["_mh", "_dh"]).join(
        pd.DataFrame.from_dict(rows, orient="index")
    )
    policy_ann = single["threshold_neighborhood_trimmed_annualized_q25"].to_dict()
    policy_sharpe = single["threshold_neighborhood_trimmed_sharpe_q25"].to_dict()
    result["policy_neighborhood_trimmed_annualized_q25_min"] = result.apply(
        lambda row: min(policy_ann[row["policy_510300"]], policy_ann[row["policy_518880"]]),
        axis=1,
    )
    result["policy_neighborhood_trimmed_sharpe_q25_min"] = result.apply(
        lambda row: min(
            policy_sharpe[row["policy_510300"]], policy_sharpe[row["policy_518880"]]
        ),
        axis=1,
    )
    return result


def _select_joint(table: pd.DataFrame, config: dict) -> tuple[pd.Series, pd.DataFrame]:
    joint = config["joint_stage"]
    result = table.copy()
    result["hard_eligible"] = (
        result["defender_entries"].ge(int(joint["minimum_defender_entries"]))
        & result["defender_days"].ge(int(joint["minimum_defender_days"]))
        & result["full_annualized_return_252"].ge(
            float(joint["full_annualized_return_floor"])
        )
        & result["validation_sharpe"].ge(float(joint["validation_sharpe_floor"]))
        & result["recent_sharpe"].ge(float(joint["recent_sharpe_floor"]))
        & result["lock_neighborhood_full_annualized_pass_rate"].ge(
            float(joint["lock_neighborhood_full_annualized_pass_rate_floor"])
        )
    )
    pool = result.loc[result["hard_eligible"]].copy()
    if pool.empty:
        pool = result.copy()
    fields = list(joint["selection_priority"])
    fields.extend(
        [
            "policy_neighborhood_trimmed_annualized_q25_min",
            "policy_neighborhood_trimmed_sharpe_q25_min",
        ]
    )
    ranks = pool[fields].rank(pct=True)
    pool["robust_min_percentile"] = ranks.min(axis=1)
    pool["robust_mean_percentile"] = ranks.mean(axis=1)
    result.loc[pool.index, "robust_min_percentile"] = pool["robust_min_percentile"]
    result.loc[pool.index, "robust_mean_percentile"] = pool["robust_mean_percentile"]
    selected = pool.sort_values(
        [
            "robust_min_percentile",
            "robust_mean_percentile",
            "trimmed_annualized_return_252",
            "trimmed_sharpe",
            "full_annualized_return_252",
            "full_sharpe",
        ],
        ascending=False,
    ).iloc[0]
    return selected, result


def _block_performance(
    blocks: pd.DataFrame,
    returns: dict[str, pd.Series],
) -> pd.DataFrame:
    rows = []
    for block_id, block in blocks.iterrows():
        start, end = pd.Timestamp(block["start"]), pd.Timestamp(block["end"])
        row = {
            "block_id": int(block_id),
            "start": start.date().isoformat(),
            "end": end.date().isoformat(),
            "observations": int(block["observations"]),
            "shock_score": float(block["shock_score"]),
            "excluded_from_selection": bool(block["excluded_from_selection"]),
        }
        for name, series in returns.items():
            row[f"{name}_return"] = float((1.0 + series.loc[start:end]).prod() - 1.0)
        rows.append(row)
    return pd.DataFrame(rows)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_experiment(root: Path, config_path: Path, output: Path) -> dict[str, object]:
    config = _load_config(config_path)
    if QUALITY_METADATA["version"] != config["frozen_layers"][
        "momentum_factor_version"
    ]:
        raise AssertionError("frozen log-quality Momentum factor mismatch")
    profiles = _profiles(config)
    policy_grid = _policy_grid(config, profiles)
    end = pd.Timestamp(config["periods"]["full"][1])
    context = build_gold_override_context(root, end=end.date())
    data = build_exact_execution_data(context)
    closes = {asset: load_ohlc(asset, end.date())["close"] for asset in ASSETS}
    common = config["common_score"]
    features = {
        asset: build_downside_raqm_features(
            closes[asset],
            data.calendar,
            profiles,
            {"rolling_504_strict_lag": 504},
            min_history=int(common["percentile_min_history"]),
            volatility_floor_annual=float(common["volatility_floor_annual"]),
            winsor_limit=float(common["winsor_limit"]),
        )
        for asset in ASSETS
    }
    trim_config = config["extreme_block_trim"]
    extreme = build_extreme_block_mask(
        closes,
        data.calendar,
        ExtremeBlockSpec(
            shock_return_window=int(trim_config["shock_return_window"]),
            shock_volatility_window=int(trim_config["shock_volatility_window"]),
            volatility_floor_annual=float(trim_config["volatility_floor_annual"]),
            block_length_sessions=int(trim_config["block_length_sessions"]),
            excluded_block_fraction=float(trim_config["excluded_block_fraction"]),
            normalization_mode=str(
                trim_config.get("normalization_mode", "volatility_adjusted")
            ),
        ),
    )
    momentum_values, momentum_actual, _ = exact_candidate_schedule(
        data, data.momentum_target
    )
    momentum_returns = pd.Series(
        momentum_values, index=data.calendar, name="log_qm_momentum"
    )
    momentum_target = context.momentum_target.astype(str)
    anchor_profile = FactorProfile(
        "universal_w30_40_25_75", (30, 40), (0.25, 0.75)
    )
    anchor_features = build_downside_raqm_features(
        closes["510300.SH"],
        data.calendar,
        {anchor_profile.profile_id: anchor_profile},
        {"rolling_504_strict_lag": 504},
        min_history=int(common["percentile_min_history"]),
        volatility_floor_annual=float(common["volatility_floor_annual"]),
        winsor_limit=float(common["winsor_limit"]),
    )
    anchor_run = run_downside_raqm_spec(
        data,
        anchor_features,
        DownsideRAQMSpec(
            anchor_profile,
            "rolling_504_strict_lag",
            0.55,
            0.20,
            30,
            30,
            3,
            1,
        ),
    )
    anchor_returns = pd.Series(
        anchor_run.returns, index=data.calendar, name="universal_510300_draqm"
    )
    anchor_target = pd.Series(
        [data.candidates[value] for value in anchor_run.actual_target],
        index=data.calendar,
    )

    single, single_returns, lookup = _evaluate_single(
        data,
        momentum_target,
        features,
        policy_grid,
        momentum_returns,
        extreme.selection_mask,
        config,
    )
    single = _threshold_neighborhood(single, config)
    options = _select_options(single, lookup, config)
    option_frame = pd.DataFrame(
        [
            {
                "asset": asset,
                "profile_id": profile,
                "rank": rank + 1,
                "policy_id": policy.policy_id(),
            }
            for asset in ASSETS
            for profile, policies in options[asset].items()
            for rank, policy in enumerate(policies)
        ]
    )
    joint, joint_returns, specs = _evaluate_joint(
        data,
        momentum_target,
        features,
        options,
        momentum_returns,
        extreme.selection_mask,
        config,
    )
    joint = _joint_neighborhood(joint, single, config)
    selected, joint = _select_joint(joint, config)
    selected_id = str(selected.name)
    selected_spec = specs[selected_id]
    selected_run = run_selected_asset_draqm_spec(
        data, momentum_target, features, selected_spec
    )
    selected_returns = pd.Series(
        selected_run.returns, index=data.calendar, name=selected_id
    )
    selected_target = pd.Series(
        [data.candidates[value] for value in selected_run.actual_target],
        index=data.calendar,
    )
    momentum_actual_target = pd.Series(
        [data.candidates[value] for value in momentum_actual], index=data.calendar
    )

    all_returns = _unique_paths(pd.concat([single_returns, joint_returns], axis=1))
    ordinary = extreme.selection_mask.astype(bool)
    checks = config["overfit_checks"]
    pbo_full, pbo_full_summary = cscv_pbo(
        all_returns, momentum_returns, block_count=int(checks["cscv_blocks"])
    )
    pbo_trimmed, pbo_trimmed_summary = cscv_pbo(
        all_returns.loc[ordinary],
        momentum_returns.loc[ordinary],
        block_count=int(checks["cscv_blocks"]),
    )
    reality_full = yearly_reality_check(
        all_returns,
        momentum_returns,
        repetitions=int(checks["yearly_reality_check_repetitions"]),
        seed=int(checks["random_seed"]),
    )
    reality_trimmed = yearly_reality_check(
        all_returns.loc[ordinary],
        momentum_returns.loc[ordinary],
        repetitions=int(checks["yearly_reality_check_repetitions"]),
        seed=int(checks["random_seed"]),
    )
    walk = expanding_walk_forward(all_returns, momentum_returns)
    leave_selection = leave_one_year_selection(all_returns, momentum_returns)
    bootstrap_momentum, bootstrap_momentum_summary = paired_block_bootstrap(
        selected_returns,
        momentum_returns,
        block_size=int(checks["paired_block_bootstrap_block"]),
        repetitions=int(checks["paired_block_bootstrap_repetitions"]),
        seed=int(checks["random_seed"]),
    )
    bootstrap_anchor, bootstrap_anchor_summary = paired_block_bootstrap(
        selected_returns,
        anchor_returns,
        block_size=int(checks["paired_block_bootstrap_block"]),
        repetitions=int(checks["paired_block_bootstrap_repetitions"]),
        seed=int(checks["random_seed"]),
    )
    events_momentum, leave_momentum, top_momentum, event_momentum_summary = _event_stress(
        selected_returns,
        momentum_returns,
        selected_target,
        momentum_actual_target,
        list(map(int, checks["top_positive_event_deletions"])),
    )
    events_anchor, leave_anchor, top_anchor, event_anchor_summary = _event_stress(
        selected_returns,
        anchor_returns,
        selected_target,
        anchor_target,
        list(map(int, checks["top_positive_event_deletions"])),
    )
    costs = _selected_cost_schedule(context, data, selected_run.actual_target)
    friction = _friction(
        selected_returns,
        costs,
        list(map(float, checks["friction_cost_multipliers"])),
    )
    block_performance = _block_performance(
        extreme.blocks,
        {
            "selected": selected_returns,
            "momentum": momentum_returns,
            "universal_anchor": anchor_returns,
        },
    )

    output.mkdir(parents=True, exist_ok=True)
    extreme.blocks.to_csv(output / "shock_blocks.csv")
    pd.concat(
        [
            extreme.selection_mask,
            extreme.shock_score_at_open,
            extreme.asset_shock_at_open,
        ],
        axis=1,
    ).to_csv(output / "selection_mask.csv")
    block_performance.to_csv(output / "block_performance.csv", index=False)
    single.sort_values(
        ["trimmed_annualized_return_252", "trimmed_sharpe"], ascending=False
    ).to_csv(output / "single_asset_grid.csv")
    option_frame.to_csv(output / "single_asset_options_for_joint.csv", index=False)
    joint.sort_values(
        ["hard_eligible", "robust_min_percentile", "trimmed_annualized_return_252"],
        ascending=False,
    ).to_csv(output / "joint_grid.csv")
    frontier = pareto_frontier(
        joint,
        ["full_annualized_return_252", "full_sharpe", "full_max_drawdown"],
    )
    joint.loc[frontier].to_csv(output / "joint_full_pareto_frontier.csv")
    all_returns.to_parquet(output / "unique_candidate_returns.parquet")
    pbo_full.to_csv(output / "cscv_full.csv", index=False)
    pbo_trimmed.to_csv(output / "cscv_trimmed.csv", index=False)
    walk.to_csv(output / "expanding_walk_forward.csv", index=False)
    leave_selection.to_csv(output / "leave_one_year_selection.csv", index=False)
    bootstrap_momentum.to_csv(output / "bootstrap_vs_momentum.csv", index=False)
    bootstrap_anchor.to_csv(output / "bootstrap_vs_universal_anchor.csv", index=False)
    events_momentum.to_csv(output / "events_vs_momentum.csv", index=False)
    leave_momentum.to_csv(output / "leave_one_event_vs_momentum.csv", index=False)
    top_momentum.to_csv(output / "top_event_deletion_vs_momentum.csv", index=False)
    events_anchor.to_csv(output / "events_vs_universal_anchor.csv", index=False)
    leave_anchor.to_csv(output / "leave_one_event_vs_universal_anchor.csv", index=False)
    top_anchor.to_csv(output / "top_event_deletion_vs_universal_anchor.csv", index=False)
    friction.to_csv(output / "friction_stress.csv", index=False)

    selected_daily = selected_run.state.copy()
    selected_daily["ordinary_selection_day"] = ordinary
    selected_daily["shock_score_at_open"] = extreme.shock_score_at_open
    selected_daily["return"] = selected_returns
    selected_daily["nav"] = (1.0 + selected_returns).cumprod()
    selected_daily["requested_candidate"] = [
        data.candidates[value] for value in selected_run.requested_target
    ]
    selected_daily["actual_candidate"] = selected_target
    selected_daily["cost_rate_at_open"] = costs
    selected_daily.to_csv(output / "selected_daily.csv")
    selected_daily.to_parquet(output / "selected_daily.parquet")

    metrics = performance(selected_returns)
    trimmed_metrics = performance(selected_returns.loc[ordinary])
    shock_metrics = performance(selected_returns.loc[~ordinary])
    momentum_metrics = performance(momentum_returns)
    anchor_metrics = performance(anchor_returns)
    strategy_metrics = pd.DataFrame(
        [
            {"strategy": "log_qm_momentum", **momentum_metrics},
            {"strategy": "universal_510300_draqm", **anchor_metrics},
            {"strategy": selected_id, **metrics},
        ]
    )
    strategy_metrics.to_csv(output / "strategy_metrics.csv", index=False)
    policies = {
        asset: {
            "profile_id": policy.profile.profile_id,
            "horizons": list(policy.profile.horizons),
            "weights": list(policy.profile.weights),
            "entry_percentile": policy.entry_percentile,
            "recovery_percentile": policy.recovery_percentile,
            "entry_confirmation_days": policy.entry_confirmation_days,
            "recovery_confirmation_days": policy.recovery_confirmation_days,
        }
        for asset, policy in selected_spec.policies.items()
        if policy is not None
    }
    validate_common_score_policies(
        {asset: selected_spec.policies[asset] for asset in ASSETS}  # type: ignore[arg-type]
    )
    three_x = friction.loc[friction["cost_multiplier"].eq(3.0)].iloc[0]
    excluded = extreme.blocks.loc[extreme.blocks["excluded_from_selection"]]
    audit = {
        "experiment_id": config["experiment"]["id"],
        "selected_candidate": selected_id,
        "common_score_profile": policies["510300.SH"]["profile_id"],
        "selected_policies": policies,
        "state_policy": {
            "momentum_lock_days": selected_spec.momentum_lock_days,
            "defender_lock_days": selected_spec.defender_lock_days,
            "recovery_mode": selected_spec.recovery_mode,
            "other_momentum_assets_gated": False,
        },
        "trim": {
            "candidate_independent": True,
            "block_length_sessions": int(trim_config["block_length_sessions"]),
            "blocks": int(len(extreme.blocks)),
            "excluded_blocks": int(excluded.shape[0]),
            "excluded_sessions": int((~ordinary).sum()),
            "excluded_session_fraction": float((~ordinary).mean()),
            "excluded_start_end": [
                {
                    "start": pd.Timestamp(row["start"]).date().isoformat(),
                    "end": pd.Timestamp(row["end"]).date().isoformat(),
                    "shock_score": float(row["shock_score"]),
                }
                for _, row in excluded.sort_values("start").iterrows()
            ],
        },
        "metrics": metrics,
        "trimmed_selection_metrics": trimmed_metrics,
        "excluded_shock_sessions_metrics": shock_metrics,
        "momentum_metrics": momentum_metrics,
        "universal_anchor_metrics": anchor_metrics,
        "single_candidate_ids": int(len(single)),
        "joint_candidate_ids": int(len(joint)),
        "unique_return_paths": int(all_returns.shape[1]),
        "cscv_full": pbo_full_summary,
        "cscv_trimmed": pbo_trimmed_summary,
        "reality_full": reality_full,
        "reality_trimmed": reality_trimmed,
        "walk_forward_return_win_rate": float(walk["test_return_delta"].gt(0).mean()),
        "walk_forward_sharpe_win_rate": float(walk["test_sharpe_delta"].gt(0).mean()),
        "bootstrap_vs_momentum": bootstrap_momentum_summary,
        "bootstrap_vs_universal_anchor": bootstrap_anchor_summary,
        "events_vs_momentum": event_momentum_summary,
        "events_vs_universal_anchor": event_anchor_summary,
        "three_x_cost_annualized_return_252": float(
            three_x["annualized_return_252"]
        ),
        "three_x_cost_sharpe": float(three_x["sharpe"]),
        "daily_return_sha256_float64_le": hashlib.sha256(
            selected_returns.to_numpy(dtype="<f8").tobytes()
        ).hexdigest(),
        "latest_state": {
            "date": selected_daily.index[-1].date().isoformat(),
            "risk_on": bool(selected_daily.iloc[-1]["risk_on"]),
            "momentum_top1": str(selected_daily.iloc[-1]["momentum_top1_at_open"]),
            "state_reason": str(selected_daily.iloc[-1]["state_reason"]),
            "actual_candidate": str(selected_daily.iloc[-1]["actual_candidate"]),
        },
    }
    (output / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    selected_config = {
        "strategy_id": "momentum_defender_common_score_trimmed_v1",
        "status": "research_candidate_not_production",
        "selected_on": config["experiment"]["created_on"],
        "common_score": {
            "profile_id": audit["common_score_profile"],
            "formula": common["formula"],
            "volatility_floor_annual": common["volatility_floor_annual"],
            "winsor_limit": common["winsor_limit"],
            "percentile_history": common["percentile_history"],
            "percentile_min_history": common["percentile_min_history"],
        },
        "asset_policies": policies,
        "state_policy": audit["state_policy"],
        "selection_trim": audit["trim"],
        "checkpoint": {
            **metrics,
            "defender_entries": selected_run.defender_entries,
            "defender_days": selected_run.defender_days,
            "sleeve_switches": selected_run.sleeve_switches,
            "candidate_switches": selected_run.candidate_switches,
            "daily_return_sha256_float64_le": audit[
                "daily_return_sha256_float64_le"
            ],
        },
        "decision": {
            "automatic_production_promotion": False,
            "require_explicit_user_promotion": True,
        },
    }
    (output / "selected_research_config.yaml").write_text(
        yaml.safe_dump(selected_config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    (output / "search_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    generate_standard_report(
        selected_returns,
        momentum_returns,
        "Log-QM Momentum",
        output / "selected_vs_momentum.html",
        selected_config,
    )
    generate_standard_report(
        selected_returns,
        anchor_returns,
        "Universal 510300 DRAQM",
        output / "selected_vs_universal_anchor.html",
        selected_config,
    )
    report = f"""# 共同score × 极端块修剪寻参

两个ETF强制使用相同score口径：`{audit['common_score_profile']}`。候选只按剩余
{ordinary.mean():.1%}普通交易日排序；最终指标仍包含全部日期。

|策略|全样本年化|Sharpe|MDD|
|---|---:|---:|---:|
|Log-QM Momentum|{momentum_metrics['annualized_return_252']:.2%}|{momentum_metrics['sharpe']:.3f}|{momentum_metrics['max_drawdown']:.2%}|
|通用510300门控|{anchor_metrics['annualized_return_252']:.2%}|{anchor_metrics['sharpe']:.3f}|{anchor_metrics['max_drawdown']:.2%}|
|共同score修剪候选|{metrics['annualized_return_252']:.2%}|{metrics['sharpe']:.3f}|{metrics['max_drawdown']:.2%}|

普通区间年化{trimmed_metrics['annualized_return_252']:.2%}、Sharpe {trimmed_metrics['sharpe']:.3f}；
被剔除极端区间年化{shock_metrics['annualized_return_252']:.2%}、Sharpe {shock_metrics['sharpe']:.3f}。
搜索{len(single)}个单资产政策、{len(joint)}个联合候选、{all_returns.shape[1]}条唯一收益路径。
"""
    (output / "research_report.md").write_text(report, encoding="utf-8")

    sources = [
        config_path,
        root / "research/momentum_defender_common_score_trimmed.py",
        root / "research/run_momentum_defender_common_score_trimmed.py",
        root / "research/momentum_defender_selected_asset_draqm.py",
        root / "data/db/510300.SH.parquet",
        root / "data/db/518880.SH.parquet",
    ]
    manifest = {
        "experiment_id": config["experiment"]["id"],
        "sources": {str(path.relative_to(root)): _sha(path) for path in sources},
        "artifacts": {
            path.name: _sha(path)
            for path in output.iterdir()
            if path.is_file() and path.name != "experiment_manifest.json"
        },
    }
    (output / "experiment_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config = args.config if args.config.is_absolute() else root / args.config
    output = args.output if args.output.is_absolute() else root / args.output
    if args.check:
        with tempfile.TemporaryDirectory() as directory:
            audit = run_experiment(root, config, Path(directory))
    else:
        audit = run_experiment(root, config, output)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
