"""Staged search for DRAQM gating only on selected Momentum Top-1 assets."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from collections import Counter
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from factors.quality_momentum import METADATA as QUALITY_METADATA
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
    SHADOW_TOP1_RECOVER_OTHER,
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
    "research/configs/momentum_defender_selected_asset_draqm_search.yaml"
)
DEFAULT_OUTPUT = Path(
    "experiments/20260824_momentum_defender_selected_asset_draqm"
)
ASSETS = ("510300.SH", "518880.SH")


def _load_config(path: Path) -> dict:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("selected-asset DRAQM config must be a mapping")
    return config


def _profiles(config: dict) -> dict[str, dict[str, FactorProfile]]:
    result: dict[str, dict[str, FactorProfile]] = {}
    for asset in ASSETS:
        result[asset] = {
            profile_id: FactorProfile(
                profile_id,
                tuple(map(int, values["horizons"])),
                tuple(map(float, values["weights"])),
            )
            for profile_id, values in config["factor"]["assets"][asset][
                "profiles"
            ].items()
        }
    return result


def _policy_grid(
    config: dict,
    profiles: dict[str, dict[str, FactorProfile]],
) -> dict[str, list[AssetDRAQMPolicy]]:
    search = config["single_asset_stage"]
    minimum_gap = float(search["minimum_hysteresis_gap"])
    result: dict[str, list[AssetDRAQMPolicy]] = {}
    for asset in ASSETS:
        policies: dict[str, AssetDRAQMPolicy] = {}
        for values in product(
            profiles[asset].values(),
            search["defender_entry_percentiles"],
            search["momentum_recovery_percentiles"],
            search["defender_entry_confirmation_days"],
            search["momentum_recovery_confirmation_days"],
        ):
            profile, entry, recovery, entry_c, recovery_c = values
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
            policies[policy.policy_id()] = policy
        result[asset] = list(policies.values())
    return result


def _segment_metrics(
    returns: pd.DataFrame,
    baseline: pd.Series,
    config: dict,
) -> pd.DataFrame:
    frames = []
    for name in ("development", "validation", "recent", "full"):
        start, end = map(pd.Timestamp, config["periods"][name])
        frames.append(
            full_metrics(returns.loc[start:end], baseline.loc[start:end]).add_prefix(
                f"{name}_"
            )
        )
    return pd.concat(frames, axis=1)


def _evaluate_single(
    data,
    momentum_target,
    features,
    policy_grid,
    config,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, AssetDRAQMPolicy]]:
    policies_by_id: dict[str, AssetDRAQMPolicy] = {}
    records = []
    returns = {}
    stage = config["single_asset_stage"]
    for asset in ASSETS:
        for position, policy in enumerate(policy_grid[asset], start=1):
            spec = SelectedAssetDRAQMSpec(
                {
                    candidate: policy if candidate == asset else None
                    for candidate in ASSETS
                },
                int(stage["fixed_momentum_lock_days"]),
                int(stage["fixed_defender_lock_days"]),
                str(stage["fixed_recovery_reference"]),
            )
            run = run_selected_asset_draqm_spec(
                data, momentum_target, features, spec
            )
            candidate_id = policy.policy_id()
            policies_by_id[candidate_id] = policy
            records.append(
                {
                    "candidate_id": candidate_id,
                    "asset": asset,
                    "profile_id": policy.profile.profile_id,
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
                    "sleeve_switches": run.sleeve_switches,
                    "candidate_switches": run.candidate_switches,
                }
            )
            returns[candidate_id] = run.returns
            if position % 250 == 0 or position == len(policy_grid[asset]):
                print(
                    f"single {asset}: evaluated {position}/{len(policy_grid[asset])}",
                    flush=True,
                )
    return (
        pd.DataFrame(records).set_index("candidate_id"),
        pd.DataFrame(returns, index=data.calendar),
        policies_by_id,
    )


def _single_neighborhood(table: pd.DataFrame, config: dict) -> pd.DataFrame:
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
    position_columns = []
    for field, values in dimensions.items():
        column = f"_{field}_position"
        result[column] = result[field].map(
            {value: position for position, value in enumerate(values)}
        )
        position_columns.append(column)
    rows = {}
    for _, group in result.groupby(["asset", "profile_id"], sort=False):
        coordinates = group[position_columns].to_numpy(int)
        annual = group["full_annualized_return_252"].to_numpy(float)
        sharpe = group["full_sharpe"].to_numpy(float)
        for position, candidate_id in enumerate(group.index):
            members = np.all(np.abs(coordinates - coordinates[position]) <= 1, axis=1)
            rows[str(candidate_id)] = {
                "neighborhood_count": int(members.sum()),
                "neighborhood_annualized_q25": float(
                    np.quantile(annual[members], 0.25)
                ),
                "neighborhood_annualized_median": float(np.median(annual[members])),
                "neighborhood_sharpe_q25": float(
                    np.quantile(sharpe[members], 0.25)
                ),
                "neighborhood_sharpe_median": float(np.median(sharpe[members])),
                "neighborhood_annualized_45pct_pass_rate": float(
                    np.mean(annual[members] >= 0.45)
                ),
            }
    return result.drop(columns=position_columns).join(
        pd.DataFrame.from_dict(rows, orient="index")
    )


def _select_policy_options(
    table: pd.DataFrame,
    policies_by_id: dict[str, AssetDRAQMPolicy],
    config: dict,
) -> dict[str, list[AssetDRAQMPolicy | None]]:
    stage = config["single_asset_stage"]
    count = int(stage["policies_per_asset_for_joint_stage"])
    result = {}
    for asset in ASSETS:
        pool = table.loc[
            table["asset"].eq(asset)
            & table["defender_entries"].ge(int(stage["minimum_defender_entries"]))
            & table["defender_days"].ge(int(stage["minimum_defender_days"]))
        ].copy()
        pool["minimum_segment_sharpe"] = pool[
            ["development_sharpe", "validation_sharpe", "recent_sharpe"]
        ].min(axis=1)
        score_fields = [
            "full_annualized_return_252",
            "full_sharpe",
            "minimum_segment_sharpe",
            "neighborhood_annualized_q25",
            "neighborhood_sharpe_q25",
        ]
        percentiles = pool[score_fields].rank(pct=True)
        pool["robust_min_percentile"] = percentiles.min(axis=1)
        pool["robust_mean_percentile"] = percentiles.mean(axis=1)
        ordered = pool.sort_values(
            [
                "robust_min_percentile",
                "robust_mean_percentile",
                "full_annualized_return_252",
                "full_sharpe",
            ],
            ascending=False,
        )
        selected_ids = []
        profile_counts: Counter[str] = Counter()
        for candidate_id, row in ordered.iterrows():
            profile = str(row["profile_id"])
            if profile_counts[profile] >= 2:
                continue
            selected_ids.append(str(candidate_id))
            profile_counts[profile] += 1
            if len(selected_ids) >= count:
                break
        if len(selected_ids) < count:
            for candidate_id in ordered.index:
                value = str(candidate_id)
                if value not in selected_ids:
                    selected_ids.append(value)
                if len(selected_ids) >= count:
                    break
        result[asset] = [None, *[policies_by_id[value] for value in selected_ids]]
    return result


def _evaluate_joint(
    data,
    momentum_target,
    features,
    options,
    config,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict[str, SelectedAssetDRAQMSpec],
]:
    joint = config["joint_stage"]
    combinations = list(
        product(
            options["510300.SH"],
            options["518880.SH"],
            joint["momentum_lock_days"],
            joint["defender_lock_days"],
            joint["recovery_reference_modes"],
        )
    )
    matrix = np.empty((len(data.calendar), len(combinations)), dtype=np.float32)
    records = []
    specs = {}
    ids = []
    for position, values in enumerate(combinations):
        csi, gold, momentum_hold, defender_hold, mode = values
        spec = SelectedAssetDRAQMSpec(
            {"510300.SH": csi, "518880.SH": gold},
            int(momentum_hold),
            int(defender_hold),
            str(mode),
        )
        candidate_id = spec.candidate_id()
        run = run_selected_asset_draqm_spec(data, momentum_target, features, spec)
        matrix[:, position] = run.returns
        ids.append(candidate_id)
        specs[candidate_id] = spec
        records.append(
            {
                "candidate_id": candidate_id,
                "policy_510300": csi.policy_id() if csi else "off",
                "policy_518880": gold.policy_id() if gold else "off",
                "momentum_lock_days": int(momentum_hold),
                "defender_lock_days": int(defender_hold),
                "recovery_mode": str(mode),
                "enabled_assets": int(csi is not None) + int(gold is not None),
                "defender_entries": run.defender_entries,
                "defender_days": run.defender_days,
                "sleeve_switches": run.sleeve_switches,
                "candidate_switches": run.candidate_switches,
            }
        )
        if (position + 1) % 250 == 0 or position + 1 == len(combinations):
            print(
                f"joint: evaluated {position + 1}/{len(combinations)}", flush=True
            )
    return (
        pd.DataFrame(records).set_index("candidate_id"),
        pd.DataFrame(matrix, index=data.calendar, columns=ids),
        specs,
    )


def _joint_neighborhood(table: pd.DataFrame, config: dict) -> pd.DataFrame:
    joint = config["joint_stage"]
    momentum_positions = {
        int(value): position for position, value in enumerate(joint["momentum_lock_days"])
    }
    defender_positions = {
        int(value): position for position, value in enumerate(joint["defender_lock_days"])
    }
    result = table.copy()
    result["_mh"] = result["momentum_lock_days"].map(momentum_positions)
    result["_dh"] = result["defender_lock_days"].map(defender_positions)
    rows = {}
    for _, group in result.groupby(
        ["policy_510300", "policy_518880", "recovery_mode"], sort=False
    ):
        coords = group[["_mh", "_dh"]].to_numpy(int)
        annual = group["full_annualized_return_252"].to_numpy(float)
        sharpe = group["full_sharpe"].to_numpy(float)
        for position, candidate_id in enumerate(group.index):
            members = np.all(np.abs(coords - coords[position]) <= 1, axis=1)
            rows[str(candidate_id)] = {
                "lock_neighborhood_count": int(members.sum()),
                "lock_neighborhood_annualized_pass_rate": float(
                    np.mean(annual[members] >= 0.45)
                ),
                "lock_neighborhood_annualized_q25": float(
                    np.quantile(annual[members], 0.25)
                ),
                "lock_neighborhood_sharpe_q25": float(
                    np.quantile(sharpe[members], 0.25)
                ),
            }
    return result.drop(columns=["_mh", "_dh"]).join(
        pd.DataFrame.from_dict(rows, orient="index")
    )


def _select_joint(
    table: pd.DataFrame,
    single: pd.DataFrame,
    config: dict,
) -> tuple[pd.Series, pd.DataFrame]:
    result = table.copy()
    joint = config["joint_stage"]
    result["minimum_segment_sharpe"] = result[
        ["development_sharpe", "validation_sharpe", "recent_sharpe"]
    ].min(axis=1)
    policy_ann = single["neighborhood_annualized_q25"].to_dict()
    policy_sharpe = single["neighborhood_sharpe_q25"].to_dict()

    def enabled_min(row, values):
        selected = [
            values[policy]
            for policy in (row["policy_510300"], row["policy_518880"])
            if policy != "off"
        ]
        return min(selected) if selected else -np.inf

    result["policy_neighborhood_annualized_q25_min"] = result.apply(
        enabled_min, axis=1, values=policy_ann
    )
    result["policy_neighborhood_sharpe_q25_min"] = result.apply(
        enabled_min, axis=1, values=policy_sharpe
    )
    result["hard_eligible"] = (
        result["enabled_assets"].ge(1)
        & result["defender_entries"].ge(int(joint["minimum_defender_entries"]))
        & result["defender_days"].ge(int(joint["minimum_defender_days"]))
        & result["full_annualized_return_252"].ge(
            float(joint["hard_full_annualized_return_floor"])
        )
        & result["validation_annualized_return_252"].ge(
            float(joint["validation_annualized_return_floor"])
        )
        & result["recent_annualized_return_252"].ge(
            float(joint["recent_annualized_return_floor"])
        )
        & result["validation_sharpe"].ge(float(joint["validation_sharpe_floor"]))
        & result["recent_sharpe"].ge(float(joint["recent_sharpe_floor"]))
        & result["lock_neighborhood_annualized_pass_rate"].ge(0.5)
    )
    pool = result.loc[result["hard_eligible"]].copy()
    if pool.empty:
        pool = result.loc[result["enabled_assets"].ge(1)].copy()
    score_fields = [
        "full_annualized_return_252",
        "full_sharpe",
        "minimum_segment_sharpe",
        "lock_neighborhood_annualized_q25",
        "lock_neighborhood_sharpe_q25",
        "policy_neighborhood_annualized_q25_min",
        "policy_neighborhood_sharpe_q25_min",
    ]
    percentiles = pool[score_fields].rank(pct=True)
    pool["robust_min_percentile"] = percentiles.min(axis=1)
    pool["robust_mean_percentile"] = percentiles.mean(axis=1)
    result.loc[pool.index, "robust_min_percentile"] = pool[
        "robust_min_percentile"
    ]
    result.loc[pool.index, "robust_mean_percentile"] = pool[
        "robust_mean_percentile"
    ]
    selected = pool.sort_values(
        [
            "robust_min_percentile",
            "robust_mean_percentile",
            "full_annualized_return_252",
            "full_sharpe",
        ],
        ascending=False,
    ).iloc[0]
    return selected, result


def _fixed_leave_year(returns: pd.Series) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "removed_year": int(year),
                **performance(returns.loc[returns.index.year != year]),
            }
            for year in sorted(returns.index.year.unique())
        ]
    )


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
    full_end = pd.Timestamp(config["periods"]["full"][1])
    context = build_gold_override_context(root, end=full_end.date())
    data = build_exact_execution_data(context)
    features = {
        asset: build_downside_raqm_features(
            load_ohlc(asset, full_end.date())["close"],
            data.calendar,
            profiles[asset],
            {"rolling_504_strict_lag": 504},
            min_history=int(config["factor"]["percentile_min_history"]),
            volatility_floor_annual=float(
                config["factor"]["volatility_floor_annual"]
            ),
            winsor_limit=float(config["factor"]["winsor_limit"]),
        )
        for asset in ASSETS
    }
    momentum_values, momentum_actual, _ = exact_candidate_schedule(
        data, data.momentum_target
    )
    momentum_returns = pd.Series(
        momentum_values, index=data.calendar, name="log_qm_momentum"
    )
    momentum_target = context.momentum_target.astype(str)

    anchor_profile = profiles["510300.SH"]["w30_40_25_75"]
    anchor_spec = DownsideRAQMSpec(
        anchor_profile,
        "rolling_504_strict_lag",
        0.55,
        0.20,
        30,
        30,
        3,
        1,
    )
    anchor_run = run_downside_raqm_spec(data, features["510300.SH"], anchor_spec)
    anchor_returns = pd.Series(
        anchor_run.returns, index=data.calendar, name="universal_510300_draqm"
    )
    anchor_target = pd.Series(
        [data.candidates[value] for value in anchor_run.actual_target],
        index=data.calendar,
    )

    single_meta, single_returns, policies_by_id = _evaluate_single(
        data, momentum_target, features, policy_grid, config
    )
    single_table = single_meta.join(
        _segment_metrics(single_returns, momentum_returns, config)
    )
    single_table = _single_neighborhood(single_table, config)
    options = _select_policy_options(single_table, policies_by_id, config)
    selected_options = pd.DataFrame(
        [
            {
                "asset": asset,
                "option_rank": rank,
                "policy_id": policy.policy_id() if policy else "off",
            }
            for asset in ASSETS
            for rank, policy in enumerate(options[asset])
        ]
    )

    joint_meta, joint_returns, joint_specs = _evaluate_joint(
        data, momentum_target, features, options, config
    )
    joint_table = joint_meta.join(
        _segment_metrics(joint_returns, momentum_returns, config)
    )
    joint_table = _joint_neighborhood(joint_table, config)
    selected, joint_table = _select_joint(joint_table, single_table, config)
    selected_id = str(selected.name)
    selected_spec = joint_specs[selected_id]
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

    all_returns = pd.concat([single_returns, joint_returns], axis=1)
    all_returns = _unique_paths(all_returns)
    checks = config["overfit_checks"]
    pbo_momentum, pbo_momentum_summary = cscv_pbo(
        all_returns, momentum_returns, block_count=int(checks["cscv_blocks"])
    )
    pbo_anchor, pbo_anchor_summary = cscv_pbo(
        all_returns, anchor_returns, block_count=int(checks["cscv_blocks"])
    )
    reality_momentum = yearly_reality_check(
        all_returns,
        momentum_returns,
        repetitions=int(checks["yearly_reality_check_repetitions"]),
        seed=int(checks["random_seed"]),
    )
    reality_anchor = yearly_reality_check(
        all_returns,
        anchor_returns,
        repetitions=int(checks["yearly_reality_check_repetitions"]),
        seed=int(checks["random_seed"]),
    )
    walk_momentum = expanding_walk_forward(all_returns, momentum_returns)
    walk_anchor = expanding_walk_forward(all_returns, anchor_returns)
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
    events, leave_event, top_deletion, event_summary = _event_stress(
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
    fixed_leave = _fixed_leave_year(selected_returns)

    output.mkdir(parents=True, exist_ok=True)
    single_table.sort_values(
        ["full_annualized_return_252", "full_sharpe"], ascending=False
    ).to_csv(output / "single_asset_grid.csv")
    selected_options.to_csv(output / "single_asset_options_for_joint.csv", index=False)
    joint_table.sort_values(
        ["hard_eligible", "robust_min_percentile", "full_annualized_return_252"],
        ascending=False,
    ).to_csv(output / "joint_grid.csv")
    frontier = pareto_frontier(
        joint_table,
        ["full_annualized_return_252", "full_sharpe", "full_max_drawdown"],
    )
    joint_table.loc[frontier].to_csv(output / "joint_pareto_frontier.csv")
    all_returns.to_parquet(output / "unique_candidate_returns.parquet")
    pbo_momentum.to_csv(output / "cscv_vs_momentum.csv", index=False)
    pbo_anchor.to_csv(output / "cscv_vs_universal_anchor.csv", index=False)
    walk_momentum.to_csv(output / "walk_forward_vs_momentum.csv", index=False)
    walk_anchor.to_csv(output / "walk_forward_vs_universal_anchor.csv", index=False)
    leave_selection.to_csv(output / "leave_one_year_selection.csv", index=False)
    bootstrap_momentum.to_csv(output / "bootstrap_vs_momentum.csv", index=False)
    bootstrap_anchor.to_csv(output / "bootstrap_vs_universal_anchor.csv", index=False)
    events.to_csv(output / "event_attribution_vs_universal_anchor.csv", index=False)
    leave_event.to_csv(output / "leave_one_event.csv", index=False)
    top_deletion.to_csv(output / "top_positive_event_deletion.csv", index=False)
    friction.to_csv(output / "friction_stress.csv", index=False)
    fixed_leave.to_csv(output / "fixed_candidate_leave_one_year.csv", index=False)

    selected_daily = selected_run.state.copy()
    selected_daily["return"] = selected_returns
    selected_daily["nav"] = (1.0 + selected_returns).cumprod()
    selected_daily["requested_candidate"] = [
        data.candidates[value] for value in selected_run.requested_target
    ]
    selected_daily["actual_candidate"] = selected_target
    selected_daily["cost_rate_at_open"] = costs
    selected_daily.to_csv(output / "selected_daily.csv")
    selected_daily.to_parquet(output / "selected_daily.parquet")

    selected_metrics = performance(selected_returns)
    momentum_metrics = performance(momentum_returns)
    anchor_metrics = performance(anchor_returns)
    strategy_metrics = pd.DataFrame(
        [
            {"strategy": "log_qm_momentum", **momentum_metrics},
            {"strategy": "universal_510300_draqm", **anchor_metrics},
            {"strategy": selected_id, **selected_metrics},
        ]
    )
    strategy_metrics.to_csv(output / "strategy_metrics.csv", index=False)

    policies = {
        asset: (
            {
                "profile_id": policy.profile.profile_id,
                "horizons": list(policy.profile.horizons),
                "weights": list(policy.profile.weights),
                "entry_percentile": policy.entry_percentile,
                "recovery_percentile": policy.recovery_percentile,
                "entry_confirmation_days": policy.entry_confirmation_days,
                "recovery_confirmation_days": policy.recovery_confirmation_days,
            }
            if policy
            else None
        )
        for asset, policy in selected_spec.policies.items()
    }
    three_x = friction.loc[friction["cost_multiplier"].eq(3.0)].iloc[0]
    audit = {
        "experiment_id": config["experiment"]["id"],
        "requested_asset_interpretation": "510330 interpreted as existing 510300.SH",
        "single_candidate_ids": int(len(single_table)),
        "joint_candidate_ids": int(len(joint_table)),
        "unique_return_paths": int(all_returns.shape[1]),
        "selected_candidate": selected_id,
        "selected_policies": policies,
        "selected_state_policy": {
            "momentum_lock_days": selected_spec.momentum_lock_days,
            "defender_lock_days": selected_spec.defender_lock_days,
            "recovery_mode": selected_spec.recovery_mode,
        },
        "requirements": {
            "only_510300_and_518880_gated": True,
            "other_momentum_assets_not_gated": True,
            "all_horizons_at_least_20": all(
                horizon >= 20
                for policy in selected_spec.policies.values()
                if policy
                for horizon in policy.profile.horizons
            ),
            "both_locks_in_20_30": 20
            <= selected_spec.momentum_lock_days
            <= 30
            and 20 <= selected_spec.defender_lock_days <= 30,
        },
        "metrics": selected_metrics,
        "momentum_metrics": momentum_metrics,
        "universal_anchor_metrics": anchor_metrics,
        "selected_vs_universal_anchor": {
            "annualized_return_delta": selected_metrics["annualized_return_252"]
            - anchor_metrics["annualized_return_252"],
            "sharpe_delta": selected_metrics["sharpe"] - anchor_metrics["sharpe"],
            "max_drawdown_delta": selected_metrics["max_drawdown"]
            - anchor_metrics["max_drawdown"],
        },
        "cscv_vs_momentum": pbo_momentum_summary,
        "cscv_vs_universal_anchor": pbo_anchor_summary,
        "reality_vs_momentum": reality_momentum,
        "reality_vs_universal_anchor": reality_anchor,
        "walk_forward_vs_momentum_return_win_rate": float(
            walk_momentum["test_return_delta"].gt(0.0).mean()
        ),
        "walk_forward_vs_momentum_sharpe_win_rate": float(
            walk_momentum["test_sharpe_delta"].gt(0.0).mean()
        ),
        "walk_forward_vs_anchor_return_win_rate": float(
            walk_anchor["test_return_delta"].gt(0.0).mean()
        ),
        "walk_forward_vs_anchor_sharpe_win_rate": float(
            walk_anchor["test_sharpe_delta"].gt(0.0).mean()
        ),
        "bootstrap_vs_momentum": bootstrap_momentum_summary,
        "bootstrap_vs_universal_anchor": bootstrap_anchor_summary,
        "events_vs_universal_anchor": event_summary,
        "fixed_leave_year_min_annualized_return_252": float(
            fixed_leave["annualized_return_252"].min()
        ),
        "fixed_leave_year_min_sharpe": float(fixed_leave["sharpe"].min()),
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
            "trigger_asset": (
                None
                if pd.isna(selected_daily.iloc[-1]["trigger_asset"])
                else str(selected_daily.iloc[-1]["trigger_asset"])
            ),
            "state_reason": str(selected_daily.iloc[-1]["state_reason"]),
            "actual_candidate": str(selected_daily.iloc[-1]["actual_candidate"]),
        },
    }
    (output / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    selected_config = {
        "strategy_id": "momentum_defender_selected_asset_draqm_v1",
        "status": "research_candidate_not_production",
        "selected_on": config["experiment"]["created_on"],
        "requested_asset_interpretation": audit["requested_asset_interpretation"],
        "frozen_layers": config["frozen_layers"],
        "factor": {
            "formula": config["factor"]["formula"],
            "volatility_floor_annual": config["factor"][
                "volatility_floor_annual"
            ],
            "winsor_limit": config["factor"]["winsor_limit"],
            "percentile_history": config["factor"]["percentile_history"],
            "percentile_min_history": config["factor"]["percentile_min_history"],
            "asset_policies": policies,
        },
        "state_policy": audit["selected_state_policy"],
        "execution": {
            "signal_timing": "previous_close_to_next_open",
            "costs": "inherited_exact_asset_interfaces",
        },
        "checkpoint": {
            **selected_metrics,
            "defender_entries": selected_run.defender_entries,
            "defender_days": selected_run.defender_days,
            "sleeve_switches": selected_run.sleeve_switches,
            "candidate_switches": selected_run.candidate_switches,
            "daily_return_sha256_float64_le": audit[
                "daily_return_sha256_float64_le"
            ],
        },
        "comparison": audit["selected_vs_universal_anchor"],
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

    report = f"""# 指定Momentum持仓资产的下行RAQM门控

请求中的510330按当前资产池中的510300解释；没有新增ETF或改写Momentum历史。

最终候选：`{selected_id}`。

|策略|年化|Sharpe|MDD|
|---|---:|---:|---:|
|Log-QM Momentum|{momentum_metrics['annualized_return_252']:.2%}|{momentum_metrics['sharpe']:.3f}|{momentum_metrics['max_drawdown']:.2%}|
|通用510300 DRAQM|{anchor_metrics['annualized_return_252']:.2%}|{anchor_metrics['sharpe']:.3f}|{anchor_metrics['max_drawdown']:.2%}|
|指定资产DRAQM|{selected_metrics['annualized_return_252']:.2%}|{selected_metrics['sharpe']:.3f}|{selected_metrics['max_drawdown']:.2%}|

指定资产策略相对通用510300门控：年化{audit['selected_vs_universal_anchor']['annualized_return_delta']:+.2%}、
Sharpe {audit['selected_vs_universal_anchor']['sharpe_delta']:+.3f}、MDD {audit['selected_vs_universal_anchor']['max_drawdown_delta']:+.2%}。

搜索{len(single_table)}个单资产政策、{len(joint_table)}个联合候选，合计{all_returns.shape[1]}条
唯一收益路径。对Momentum的Reality Check p={reality_momentum['p_value']:.4f}；对通用510300
门控p={reality_anchor['p_value']:.4f}。结果是研究候选，不自动修改生产策略。
"""
    (output / "research_report.md").write_text(report, encoding="utf-8")

    source_paths = [
        config_path,
        root / "research/momentum_defender_selected_asset_draqm.py",
        root / "research/run_momentum_defender_selected_asset_draqm.py",
        root / "research/momentum_defender_downside_raqm.py",
        root / "factors/quality_momentum.py",
        root / "data/db/510300.SH.parquet",
        root / "data/db/518880.SH.parquet",
    ]
    manifest = {
        "experiment_id": config["experiment"]["id"],
        "sources": {
            str(path.relative_to(root)): _sha(path) for path in source_paths
        },
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
