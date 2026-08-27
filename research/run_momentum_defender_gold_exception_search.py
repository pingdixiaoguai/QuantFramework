"""Staged robust search for universal 510300 gating with a Gold exception."""

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
)
from research.momentum_defender_downside_raqm import (
    DownsideRAQMSpec,
    FactorProfile,
    build_downside_raqm_features,
    build_exact_execution_data,
    exact_candidate_schedule,
    run_downside_raqm_spec,
)
from research.momentum_defender_gold_exception_gate import (
    GOLD_BIDIRECTIONAL,
    GOLD_EXEMPTION_ONLY,
)
from research.momentum_defender_gold_overlay import (
    GoldOverlaySpec,
    independent_anchor_state,
    independent_gold_state,
    run_gold_overlay_spec,
)
from research.momentum_defender_gold_override import build_gold_override_context
from research.momentum_defender_gold_override_overfit import (
    cscv_pbo,
    expanding_walk_forward,
    paired_block_bootstrap,
    yearly_reality_check,
)
from research.momentum_defender_occam import performance
from research.momentum_defender_selected_asset_draqm import AssetDRAQMPolicy
from research.momentum_volatility import load_ohlc
from research.run_momentum_defender_common_score_trimmed import _add_metrics
from research.run_momentum_defender_log_qm_robust import (
    _event_stress,
    _friction,
    _selected_cost_schedule,
    _unique_paths,
)
from research.standard_report import generate_standard_report


DEFAULT_CONFIG = Path("research/configs/momentum_defender_gold_exception_search.yaml")
DEFAULT_OUTPUT = Path("experiments/20260825_momentum_defender_gold_exception_search")


def _load(path: Path) -> dict:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Gold-exception config must be a mapping")
    return config


def _profiles(values: dict) -> dict[str, FactorProfile]:
    return {
        name: FactorProfile(
            name,
            tuple(map(int, profile["horizons"])),
            tuple(map(float, profile["weights"])),
        )
        for name, profile in values.items()
    }


def _policies(asset: str, profiles: dict[str, FactorProfile], grid: dict):
    result = {profile: [] for profile in profiles}
    gap = float(grid["minimum_hysteresis_gap"])
    for profile_id, profile in profiles.items():
        unique = {}
        for entry, recovery, entry_c, recovery_c in product(
            grid["entry_percentiles"],
            grid["recovery_percentiles"],
            grid["entry_confirmation_days"],
            grid["recovery_confirmation_days"],
        ):
            if float(entry) - float(recovery) + 1e-12 < gap:
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
        result[profile_id] = list(unique.values())
    return result


def _minimum_segment(table: pd.DataFrame):
    result = table.copy()
    result["full_minimum_segment_sharpe"] = result[
        ["development_sharpe", "validation_sharpe", "recent_sharpe"]
    ].min(axis=1)
    return result


def _add_threshold_neighborhood(table: pd.DataFrame, grid: dict, group_fields):
    result = table.copy()
    dimensions = {
        "entry_percentile": list(map(float, grid["entry_percentiles"])),
        "recovery_percentile": list(map(float, grid["recovery_percentiles"])),
        "entry_confirmation_days": list(map(int, grid["entry_confirmation_days"])),
        "recovery_confirmation_days": list(
            map(int, grid["recovery_confirmation_days"])
        ),
    }
    columns = []
    for field, values in dimensions.items():
        column = f"_{field}"
        result[column] = result[field].map(
            {value: position for position, value in enumerate(values)}
        )
        columns.append(column)
    rows = {}
    for _, group in result.groupby(group_fields, sort=False):
        coords = group[columns].to_numpy(int)
        arrays = {
            "full_annualized": group["full_annualized_return_252"].to_numpy(float),
            "full_sharpe": group["full_sharpe"].to_numpy(float),
            "trimmed_annualized": group[
                "trimmed_annualized_return_252"
            ].to_numpy(float),
            "trimmed_sharpe": group["trimmed_sharpe"].to_numpy(float),
        }
        for position, candidate_id in enumerate(group.index):
            members = np.all(np.abs(coords - coords[position]) <= 1, axis=1)
            rows[str(candidate_id)] = {
                "threshold_neighborhood_count": int(members.sum()),
                **{
                    f"threshold_neighborhood_{name}_q25": float(
                        np.quantile(values[members], 0.25)
                    )
                    for name, values in arrays.items()
                },
            }
    return result.drop(columns=columns).join(pd.DataFrame.from_dict(rows, orient="index"))


def _rank(pool: pd.DataFrame, fields, prefix):
    result = pool.copy()
    ranks = result[fields].rank(pct=True)
    result[f"{prefix}_min"] = ranks.min(axis=1)
    result[f"{prefix}_mean"] = ranks.mean(axis=1)
    return result.sort_values(
        [f"{prefix}_min", f"{prefix}_mean", fields[0], fields[1]],
        ascending=False,
    )


def _anchor_stage(data, features, policies, baseline, ordinary, config):
    records = []
    returns = {}
    lookup = {}
    total = sum(map(len, policies.values()))
    done = 0
    mh, dh = map(int, config["anchor_policy_grid"]["fixed_stage_locks"])
    for profile_id, values in policies.items():
        for policy in values:
            run = run_downside_raqm_spec(
                data,
                features,
                DownsideRAQMSpec(
                    policy.profile,
                    "rolling_504_strict_lag",
                    policy.entry_percentile,
                    policy.recovery_percentile,
                    mh,
                    dh,
                    policy.entry_confirmation_days,
                    policy.recovery_confirmation_days,
                ),
            )
            candidate_id = policy.policy_id()
            lookup[candidate_id] = policy
            returns[candidate_id] = run.returns
            records.append(
                {
                    "candidate_id": candidate_id,
                    "profile_id": profile_id,
                    "entry_percentile": policy.entry_percentile,
                    "recovery_percentile": policy.recovery_percentile,
                    "entry_confirmation_days": policy.entry_confirmation_days,
                    "recovery_confirmation_days": policy.recovery_confirmation_days,
                    "defender_entries": run.defender_entries,
                    "defender_days": run.defender_days,
                }
            )
            done += 1
            if done % 100 == 0 or done == total:
                print(f"anchor: {done}/{total}", flush=True)
    frame = pd.DataFrame(returns, index=data.calendar)
    table = _minimum_segment(
        _add_metrics(
            pd.DataFrame(records).set_index("candidate_id"),
            frame,
            baseline,
            ordinary,
            config,
        )
    )
    table = _add_threshold_neighborhood(
        table, config["anchor_policy_grid"], ["profile_id"]
    )
    options = []
    option_rows = []
    full_fields = [
        "full_annualized_return_252",
        "full_sharpe",
        "full_minimum_segment_sharpe",
        "threshold_neighborhood_full_annualized_q25",
        "threshold_neighborhood_full_sharpe_q25",
    ]
    trimmed_fields = [
        "trimmed_annualized_return_252",
        "trimmed_sharpe",
        "trimmed_minimum_segment_sharpe",
        "threshold_neighborhood_trimmed_annualized_q25",
        "threshold_neighborhood_trimmed_sharpe_q25",
    ]
    for profile_id, sample in table.groupby("profile_id"):
        selected = {}
        for role, fields in (
            ("best_full", full_fields),
            ("best_trimmed", trimmed_fields),
            ("best_combined", full_fields[:2] + trimmed_fields[:2]),
        ):
            candidate_id = str(_rank(sample, fields, role).index[0])
            selected[candidate_id] = role
        for candidate_id, role in selected.items():
            options.append(lookup[candidate_id])
            option_rows.append(
                {
                    "profile_id": profile_id,
                    "policy_id": candidate_id,
                    "selection_role": role,
                }
            )
    forced = config["forced_controls"]["anchor"]
    forced_policy = next(
        policy
        for policy in policies[str(forced["profile_id"])]
        if policy.entry_percentile == float(forced["entry_percentile"])
        and policy.recovery_percentile == float(forced["recovery_percentile"])
        and policy.entry_confirmation_days
        == int(forced["entry_confirmation_days"])
        and policy.recovery_confirmation_days
        == int(forced["recovery_confirmation_days"])
    )
    if forced_policy.policy_id() not in {policy.policy_id() for policy in options}:
        options.append(forced_policy)
        option_rows.append(
            {
                "profile_id": forced_policy.profile.profile_id,
                "policy_id": forced_policy.policy_id(),
                "selection_role": "forced_universal_control",
            }
        )
    return table, frame, options, pd.DataFrame(option_rows)


def _gold_stage(
    data,
    momentum_target,
    features,
    anchor_options,
    gold_policies,
    baseline,
    ordinary,
    config,
):
    mh, dh = map(int, config["gold_policy_grid"]["fixed_stage_locks"])
    combinations = [
        (anchor, gold, mode)
        for anchor in anchor_options
        for profile in gold_policies.values()
        for gold in profile
        for mode in config["gold_policy_grid"]["override_modes"]
    ]
    matrix = np.empty((len(data.calendar), len(combinations)), dtype=np.float32)
    records = []
    lookup = {}
    base_cache = {}
    gold_cache = {}
    for position, (anchor, gold, mode) in enumerate(combinations):
        spec = GoldOverlaySpec(anchor, gold, mh, dh, str(mode))
        base_key = anchor.policy_id()
        if base_key not in base_cache:
            base_cache[base_key] = independent_anchor_state(
                data.calendar, features, spec
            )
        gold_key = gold.policy_id()
        if gold_key not in gold_cache:
            gold_cache[gold_key] = independent_gold_state(
                data.calendar, momentum_target, features, gold
            )
        run = run_gold_overlay_spec(
            data,
            momentum_target,
            features,
            spec,
            base_state=base_cache[base_key],
            gold_state=gold_cache[gold_key],
        )
        candidate_id = spec.candidate_id()
        matrix[:, position] = run.returns
        lookup[candidate_id] = spec
        records.append(
            {
                "candidate_id": candidate_id,
                "anchor_profile": anchor.profile.profile_id,
                "anchor_policy": anchor.policy_id(),
                "gold_profile": gold.profile.profile_id,
                "gold_policy": gold.policy_id(),
                "override_mode": str(mode),
                "entry_percentile": gold.entry_percentile,
                "recovery_percentile": gold.recovery_percentile,
                "entry_confirmation_days": gold.entry_confirmation_days,
                "recovery_confirmation_days": gold.recovery_confirmation_days,
                "defender_entries": run.base_defender_entries,
                "defender_days": run.effective_defender_days,
            }
        )
        if (position + 1) % 500 == 0 or position + 1 == len(combinations):
            print(f"gold: {position + 1}/{len(combinations)}", flush=True)
    frame = pd.DataFrame(matrix, index=data.calendar, columns=[r["candidate_id"] for r in records])
    table = _minimum_segment(
        _add_metrics(
            pd.DataFrame(records).set_index("candidate_id"),
            frame,
            baseline,
            ordinary,
            config,
        )
    )
    table = _add_threshold_neighborhood(
        table,
        config["gold_policy_grid"],
        ["anchor_policy", "gold_profile", "override_mode"],
    )
    full_fields = [
        "full_annualized_return_252",
        "full_sharpe",
        "full_minimum_segment_sharpe",
        "threshold_neighborhood_full_annualized_q25",
        "threshold_neighborhood_full_sharpe_q25",
    ]
    trimmed_fields = [
        "trimmed_annualized_return_252",
        "trimmed_sharpe",
        "trimmed_minimum_segment_sharpe",
        "threshold_neighborhood_trimmed_annualized_q25",
        "threshold_neighborhood_trimmed_sharpe_q25",
    ]
    options = []
    option_rows = []
    for keys, sample in table.groupby(
        ["anchor_profile", "gold_profile", "override_mode"]
    ):
        selected = {}
        for role, fields in (
            ("best_full", full_fields),
            ("best_trimmed", trimmed_fields),
            ("best_combined", full_fields[:2] + trimmed_fields[:2]),
        ):
            candidate_id = str(_rank(sample, fields, role).index[0])
            selected[candidate_id] = role
        for candidate_id, role in selected.items():
            options.append(lookup[candidate_id])
            option_rows.append(
                {
                    "anchor_profile": keys[0],
                    "gold_profile": keys[1],
                    "override_mode": keys[2],
                    "pair_id": candidate_id,
                    "selection_role": role,
                }
            )
    forced_anchor = config["forced_controls"]["anchor"]
    forced_gold = config["forced_controls"]["gold"]
    anchor = next(
        policy
        for policy in anchor_options
        if policy.profile.profile_id == str(forced_anchor["profile_id"])
        and policy.entry_percentile == float(forced_anchor["entry_percentile"])
        and policy.recovery_percentile == float(forced_anchor["recovery_percentile"])
        and policy.entry_confirmation_days
        == int(forced_anchor["entry_confirmation_days"])
        and policy.recovery_confirmation_days
        == int(forced_anchor["recovery_confirmation_days"])
    )
    gold = next(
        policy
        for policy in gold_policies[str(forced_gold["profile_id"])]
        if policy.entry_percentile == float(forced_gold["entry_percentile"])
        and policy.recovery_percentile == float(forced_gold["recovery_percentile"])
        and policy.entry_confirmation_days == int(forced_gold["entry_confirmation_days"])
        and policy.recovery_confirmation_days
        == int(forced_gold["recovery_confirmation_days"])
    )
    for mode in config["forced_controls"]["override_modes"]:
        forced_id = GoldOverlaySpec(anchor, gold, mh, dh, str(mode)).candidate_id()
        if forced_id not in {spec.candidate_id() for spec in options}:
            options.append(lookup[forced_id])
            option_rows.append(
                {
                    "anchor_profile": anchor.profile.profile_id,
                    "gold_profile": gold.profile.profile_id,
                    "override_mode": mode,
                    "pair_id": forced_id,
                    "selection_role": "forced_diagnostic_control",
                }
            )
    return table, frame, options, pd.DataFrame(option_rows)


def _joint_stage(
    data,
    momentum_target,
    features,
    pair_options,
    baseline,
    ordinary,
    config,
):
    combinations = [
        (pair.anchor_policy, pair.gold_policy, pair.override_mode, mh, dh)
        for pair in pair_options
        for mh in config["joint_stage"]["momentum_lock_days"]
        for dh in config["joint_stage"]["defender_lock_days"]
    ]
    matrix = np.empty((len(data.calendar), len(combinations)), dtype=np.float32)
    records = []
    specs = {}
    ids = []
    base_cache = {}
    gold_cache = {}
    for position, (anchor, gold, mode, mh, dh) in enumerate(combinations):
        spec = GoldOverlaySpec(anchor, gold, int(mh), int(dh), str(mode))
        base_key = (anchor.policy_id(), int(mh), int(dh))
        if base_key not in base_cache:
            base_cache[base_key] = independent_anchor_state(
                data.calendar, features, spec
            )
        gold_key = gold.policy_id()
        if gold_key not in gold_cache:
            gold_cache[gold_key] = independent_gold_state(
                data.calendar, momentum_target, features, gold
            )
        run = run_gold_overlay_spec(
            data,
            momentum_target,
            features,
            spec,
            base_state=base_cache[base_key],
            gold_state=gold_cache[gold_key],
        )
        candidate_id = spec.candidate_id()
        matrix[:, position] = run.returns
        ids.append(candidate_id)
        specs[candidate_id] = spec
        records.append(
            {
                "candidate_id": candidate_id,
                "anchor_profile": anchor.profile.profile_id,
                "gold_profile": gold.profile.profile_id,
                "anchor_policy": anchor.policy_id(),
                "gold_policy": gold.policy_id(),
                "override_mode": str(mode),
                "momentum_lock_days": int(mh),
                "defender_lock_days": int(dh),
                "defender_entries": run.base_defender_entries,
                "defender_days": run.effective_defender_days,
                "sleeve_switches": int(
                    run.state["effective_risk_on"].ne(
                        run.state["effective_risk_on"].shift()
                    ).sum()
                    - 1
                ),
                "candidate_switches": run.candidate_switches,
            }
        )
        if (position + 1) % 500 == 0 or position + 1 == len(combinations):
            print(f"joint: {position + 1}/{len(combinations)}", flush=True)
    frame = pd.DataFrame(matrix, index=data.calendar, columns=ids)
    table = _minimum_segment(
        _add_metrics(
            pd.DataFrame(records).set_index("candidate_id"),
            frame,
            baseline,
            ordinary,
            config,
        )
    )
    locks_m = list(map(int, config["joint_stage"]["momentum_lock_days"]))
    locks_d = list(map(int, config["joint_stage"]["defender_lock_days"]))
    table["_mh"] = table["momentum_lock_days"].map(
        {value: position for position, value in enumerate(locks_m)}
    )
    table["_dh"] = table["defender_lock_days"].map(
        {value: position for position, value in enumerate(locks_d)}
    )
    rows = {}
    for _, group in table.groupby(
        ["anchor_policy", "gold_policy", "override_mode"], sort=False
    ):
        coords = group[["_mh", "_dh"]].to_numpy(int)
        arrays = {
            "full_annualized": group["full_annualized_return_252"].to_numpy(float),
            "full_sharpe": group["full_sharpe"].to_numpy(float),
            "trimmed_annualized": group[
                "trimmed_annualized_return_252"
            ].to_numpy(float),
            "trimmed_sharpe": group["trimmed_sharpe"].to_numpy(float),
        }
        for position, candidate_id in enumerate(group.index):
            members = np.all(np.abs(coords - coords[position]) <= 1, axis=1)
            rows[str(candidate_id)] = {
                "lock_neighborhood_count": int(members.sum()),
                **{
                    f"lock_neighborhood_{name}_q25": float(
                        np.quantile(values[members], 0.25)
                    )
                    for name, values in arrays.items()
                },
            }
    table = table.drop(columns=["_mh", "_dh"]).join(
        pd.DataFrame.from_dict(rows, orient="index")
    )
    return table, frame, specs


def _complexity(row: pd.Series, config: dict) -> float:
    values = config["occam_complexity"]
    score = 0.0
    score += values["weighted_profile"] if "_" in str(row["anchor_profile"])[1:] else values["single_window_profile"]
    score += values["weighted_profile"] if "_" in str(row["gold_profile"])[1:] else values["single_window_profile"]
    score += values[str(row["override_mode"])]
    # Confirmation complexity is encoded in policy IDs and is only a final tie-breaker.
    return float(score)


def _select(table: pd.DataFrame, config: dict, selection_key: str, prefix: str):
    result = table.copy()
    eligible = (
        result["defender_entries"].ge(int(config["joint_stage"]["minimum_defender_entries"]))
        & result["defender_days"].ge(int(config["joint_stage"]["minimum_defender_days"]))
        & result["full_minimum_segment_sharpe"].gt(
            float(config["joint_stage"]["minimum_segment_sharpe"])
        )
        & result["trimmed_minimum_segment_sharpe"].gt(
            float(config["joint_stage"]["minimum_segment_sharpe"])
        )
    )
    pool = result.loc[eligible].copy()
    fields = list(config[selection_key]["ranking_fields"])
    ranked = _rank(pool, fields, prefix)
    result.loc[ranked.index, f"{prefix}_min"] = ranked[f"{prefix}_min"]
    result.loc[ranked.index, f"{prefix}_mean"] = ranked[f"{prefix}_mean"]
    best_min = float(ranked[f"{prefix}_min"].max())
    best_mean = float(ranked.loc[ranked[f"{prefix}_min"].ge(best_min - 0.03), f"{prefix}_mean"].max())
    stable = ranked.loc[
        ranked[f"{prefix}_min"].ge(best_min - 0.03)
        & ranked[f"{prefix}_mean"].ge(best_mean - 0.03)
    ].copy()
    annual_field, sharpe_field = fields[:2]
    max_annual = float(stable[annual_field].max())
    max_sharpe = float(stable[sharpe_field].max())
    near = stable.loc[
        stable[annual_field].ge(
            max_annual - float(config[selection_key]["occam_annualized_tolerance"])
        )
        & stable[sharpe_field].ge(
            max_sharpe - float(config[selection_key]["occam_sharpe_tolerance"])
        )
    ].copy()
    near["occam_complexity"] = near.apply(_complexity, axis=1, config=config)
    selected = near.sort_values(
        ["occam_complexity", annual_field, sharpe_field, f"{prefix}_mean"],
        ascending=[True, False, False, False],
    ).iloc[0]
    result[f"eligible_{prefix}"] = eligible
    return selected, result


def _candidate_detail(
    label,
    selected,
    specs,
    data,
    momentum_target,
    features,
    momentum_returns,
    momentum_actual_target,
    universal_returns,
    universal_target,
    context,
    ordinary,
    output,
    config,
):
    candidate_id = str(selected.name)
    spec = specs[candidate_id]
    run = run_gold_overlay_spec(data, momentum_target, features, spec)
    returns = pd.Series(run.returns, index=data.calendar, name=candidate_id)
    target = pd.Series(
        [data.candidates[value] for value in run.actual_target], index=data.calendar
    )
    boot_m, boot_m_summary = paired_block_bootstrap(
        returns, momentum_returns, repetitions=5000, seed=20260825
    )
    boot_u, boot_u_summary = paired_block_bootstrap(
        returns, universal_returns, repetitions=5000, seed=20260825
    )
    events_m, leave_m, top_m, event_m_summary = _event_stress(
        returns, momentum_returns, target, momentum_actual_target, [1, 2, 3]
    )
    events_u, leave_u, top_u, event_u_summary = _event_stress(
        returns, universal_returns, target, universal_target, [1, 2, 3]
    )
    costs = _selected_cost_schedule(context, data, run.actual_target)
    friction = _friction(returns, costs, [1.0, 2.0, 3.0])
    daily = run.state.copy()
    daily["ordinary_selection_day"] = ordinary
    daily["return"] = returns
    daily["nav"] = (1.0 + returns).cumprod()
    daily["actual_candidate"] = target
    daily["cost_rate_at_open"] = costs
    daily.to_csv(output / f"selected_{label}_daily.csv")
    daily.to_parquet(output / f"selected_{label}_daily.parquet")
    boot_m.to_csv(output / f"selected_{label}_bootstrap_vs_momentum.csv", index=False)
    boot_u.to_csv(output / f"selected_{label}_bootstrap_vs_universal.csv", index=False)
    events_m.to_csv(output / f"selected_{label}_events_vs_momentum.csv", index=False)
    leave_m.to_csv(output / f"selected_{label}_leave_event_vs_momentum.csv", index=False)
    top_m.to_csv(output / f"selected_{label}_top_events_vs_momentum.csv", index=False)
    events_u.to_csv(output / f"selected_{label}_events_vs_universal.csv", index=False)
    leave_u.to_csv(output / f"selected_{label}_leave_event_vs_universal.csv", index=False)
    top_u.to_csv(output / f"selected_{label}_top_events_vs_universal.csv", index=False)
    friction.to_csv(output / f"selected_{label}_friction.csv", index=False)
    generate_standard_report(
        returns,
        universal_returns,
        "Universal 510300 Gate",
        output / f"selected_{label}_vs_universal.html",
        {"candidate_id": candidate_id},
    )
    return {
        "candidate_id": candidate_id,
        "anchor_policy": {
            "profile_id": spec.anchor_policy.profile.profile_id,
            "horizons": list(spec.anchor_policy.profile.horizons),
            "weights": list(spec.anchor_policy.profile.weights),
            "entry_percentile": spec.anchor_policy.entry_percentile,
            "recovery_percentile": spec.anchor_policy.recovery_percentile,
            "entry_confirmation_days": spec.anchor_policy.entry_confirmation_days,
            "recovery_confirmation_days": spec.anchor_policy.recovery_confirmation_days,
        },
        "gold_policy": {
            "profile_id": spec.gold_policy.profile.profile_id,
            "horizons": list(spec.gold_policy.profile.horizons),
            "weights": list(spec.gold_policy.profile.weights),
            "entry_percentile": spec.gold_policy.entry_percentile,
            "recovery_percentile": spec.gold_policy.recovery_percentile,
            "entry_confirmation_days": spec.gold_policy.entry_confirmation_days,
            "recovery_confirmation_days": spec.gold_policy.recovery_confirmation_days,
        },
        "state_policy": {
            "momentum_lock_days": spec.momentum_lock_days,
            "defender_lock_days": spec.defender_lock_days,
            "override_mode": spec.override_mode,
        },
        "full_metrics": performance(returns),
        "ordinary_metrics": performance(returns.loc[ordinary]),
        "defender_entries": run.base_defender_entries,
        "defender_days": run.effective_defender_days,
        "sleeve_switches": int(
            run.state["effective_risk_on"].ne(
                run.state["effective_risk_on"].shift()
            ).sum()
            - 1
        ),
        "candidate_switches": run.candidate_switches,
        "bootstrap_vs_momentum": boot_m_summary,
        "bootstrap_vs_universal": boot_u_summary,
        "events_vs_momentum": event_m_summary,
        "events_vs_universal": event_u_summary,
        "three_x_cost_metrics": friction.loc[
            friction["cost_multiplier"].eq(3.0)
        ].iloc[0].to_dict(),
        "daily_return_sha256_float64_le": hashlib.sha256(
            returns.to_numpy(dtype="<f8").tobytes()
        ).hexdigest(),
    }


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_experiment(root: Path, config_path: Path, output: Path):
    config = _load(config_path)
    if QUALITY_METADATA["version"] != config["frozen_layers"][
        "momentum_factor_version"
    ]:
        raise AssertionError("Momentum factor mismatch")
    end = pd.Timestamp(config["periods"]["full"][1])
    context = build_gold_override_context(root, end=end.date())
    data = build_exact_execution_data(context)
    anchor_profiles = _profiles(config["anchor_policy_grid"]["profiles"])
    gold_profiles = _profiles(config["gold_policy_grid"]["profiles"])
    closes = {
        "510300.SH": load_ohlc("510300.SH", end.date())["close"],
        "518880.SH": load_ohlc("518880.SH", end.date())["close"],
    }
    defaults = config["factor_defaults"]
    features = {
        "510300.SH": build_downside_raqm_features(
            closes["510300.SH"],
            data.calendar,
            anchor_profiles,
            {"rolling_504_strict_lag": 504},
            min_history=int(defaults["percentile_min_history"]),
            volatility_floor_annual=float(defaults["volatility_floor_annual"]),
            winsor_limit=float(defaults["winsor_limit"]),
        ),
        "518880.SH": build_downside_raqm_features(
            closes["518880.SH"],
            data.calendar,
            gold_profiles,
            {"rolling_504_strict_lag": 504},
            min_history=int(defaults["percentile_min_history"]),
            volatility_floor_annual=float(defaults["volatility_floor_annual"]),
            winsor_limit=float(defaults["winsor_limit"]),
        ),
    }
    trim = config["extreme_block_trim"]
    extreme = build_extreme_block_mask(
        closes,
        data.calendar,
        ExtremeBlockSpec(
            shock_return_window=int(trim["shock_return_window"]),
            block_length_sessions=int(trim["block_length_sessions"]),
            excluded_block_fraction=float(trim["excluded_block_fraction"]),
            normalization_mode=str(trim["normalization_mode"]),
        ),
    )
    ordinary = extreme.selection_mask.astype(bool)
    momentum_values, momentum_actual, _ = exact_candidate_schedule(
        data, data.momentum_target
    )
    momentum_returns = pd.Series(
        momentum_values, index=data.calendar, name="log_qm_momentum"
    )
    momentum_target = pd.Series(
        [data.candidates[value] for value in momentum_actual], index=data.calendar
    )
    universal = pd.read_parquet(
        root
        / "experiments/20260824_momentum_defender_downside_raqm_final_selection/selected_daily.parquet"
    )
    universal_returns = universal["return"].astype(float)
    universal_target = universal["actual_candidate"].astype(str)

    anchor_policies = _policies("510300.SH", anchor_profiles, config["anchor_policy_grid"])
    gold_policies = _policies("518880.SH", gold_profiles, config["gold_policy_grid"])
    anchor_table, anchor_returns, anchor_options, anchor_option_frame = _anchor_stage(
        data,
        features["510300.SH"],
        anchor_policies,
        momentum_returns,
        ordinary,
        config,
    )
    gold_table, gold_returns, pair_options, pair_option_frame = _gold_stage(
        data,
        context.momentum_target,
        features,
        anchor_options,
        gold_policies,
        momentum_returns,
        ordinary,
        config,
    )
    joint_table, joint_returns, specs = _joint_stage(
        data,
        context.momentum_target,
        features,
        pair_options,
        momentum_returns,
        ordinary,
        config,
    )
    selected_full, joint_table = _select(
        joint_table, config, "selection_including_extremes", "full"
    )
    selected_trimmed, joint_table = _select(
        joint_table, config, "selection_excluding_extremes", "trimmed"
    )
    output.mkdir(parents=True, exist_ok=True)
    extreme.blocks.to_csv(output / "shock_blocks.csv")
    anchor_table.to_csv(output / "anchor_grid.csv")
    anchor_option_frame.to_csv(output / "anchor_options.csv", index=False)
    gold_table.to_csv(output / "gold_grid.csv")
    pair_option_frame.to_csv(output / "gold_pair_options.csv", index=False)
    joint_table.to_csv(output / "joint_grid.csv")
    all_returns = _unique_paths(
        pd.concat(
            [
                anchor_returns.add_prefix("anchor::"),
                gold_returns.add_prefix("gold::"),
                joint_returns.add_prefix("joint::"),
            ],
            axis=1,
        )
    )
    all_returns.to_parquet(output / "unique_candidate_returns.parquet")
    pbo_full, pbo_full_summary = cscv_pbo(
        all_returns, momentum_returns, block_count=12
    )
    pbo_trimmed, pbo_trimmed_summary = cscv_pbo(
        all_returns.loc[ordinary], momentum_returns.loc[ordinary], block_count=12
    )
    reality_full = yearly_reality_check(
        all_returns, momentum_returns, repetitions=5000, seed=20260825
    )
    reality_trimmed = yearly_reality_check(
        all_returns.loc[ordinary],
        momentum_returns.loc[ordinary],
        repetitions=5000,
        seed=20260825,
    )
    walk = expanding_walk_forward(all_returns, momentum_returns)
    pbo_full.to_csv(output / "cscv_full.csv", index=False)
    pbo_trimmed.to_csv(output / "cscv_trimmed.csv", index=False)
    walk.to_csv(output / "walk_forward.csv", index=False)
    candidates = {
        "including_extremes": _candidate_detail(
            "including_extremes",
            selected_full,
            specs,
            data,
            context.momentum_target,
            features,
            momentum_returns,
            momentum_target,
            universal_returns,
            universal_target,
            context,
            ordinary,
            output,
            config,
        ),
        "excluding_extremes": _candidate_detail(
            "excluding_extremes",
            selected_trimmed,
            specs,
            data,
            context.momentum_target,
            features,
            momentum_returns,
            momentum_target,
            universal_returns,
            universal_target,
            context,
            ordinary,
            output,
            config,
        ),
    }
    audit = {
        "experiment_id": config["experiment"]["id"],
        "candidate_counts": {
            "anchor": int(len(anchor_table)),
            "gold_pairs": int(len(gold_table)),
            "joint": int(len(joint_table)),
            "unique_paths": int(all_returns.shape[1]),
        },
        "trim": {
            "blocks": int(len(extreme.blocks)),
            "excluded_blocks": int(extreme.blocks["excluded_from_selection"].sum()),
            "excluded_sessions": int((~ordinary).sum()),
        },
        "candidates": candidates,
        "family_audit": {
            "cscv_full": pbo_full_summary,
            "cscv_trimmed": pbo_trimmed_summary,
            "reality_full": reality_full,
            "reality_trimmed": reality_trimmed,
            "walk_forward_return_win_rate": float(
                walk["test_return_delta"].gt(0.0).mean()
            ),
            "walk_forward_sharpe_win_rate": float(
                walk["test_sharpe_delta"].gt(0.0).mean()
            ),
        },
        "benchmarks": {
            "momentum": performance(momentum_returns),
            "universal_gate": performance(universal_returns),
        },
    }
    (output / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for label, candidate in candidates.items():
        config_out = {
            "strategy_id": f"momentum_defender_gold_exception_{label}_v1",
            "status": "research_candidate_not_production",
            "selection_objective": label,
            **candidate,
        }
        (output / f"selected_{label}_config.yaml").write_text(
            yaml.safe_dump(config_out, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
    report = f"""# 通用510300门控 + Gold例外

|选择目标|全样本年化|Sharpe|普通区间年化|普通区间Sharpe|
|---|---:|---:|---:|---:|
|包含极端行情|{candidates['including_extremes']['full_metrics']['annualized_return_252']:.2%}|{candidates['including_extremes']['full_metrics']['sharpe']:.3f}|{candidates['including_extremes']['ordinary_metrics']['annualized_return_252']:.2%}|{candidates['including_extremes']['ordinary_metrics']['sharpe']:.3f}|
|不包含极端行情|{candidates['excluding_extremes']['full_metrics']['annualized_return_252']:.2%}|{candidates['excluding_extremes']['full_metrics']['sharpe']:.3f}|{candidates['excluding_extremes']['ordinary_metrics']['annualized_return_252']:.2%}|{candidates['excluding_extremes']['ordinary_metrics']['sharpe']:.3f}|
"""
    (output / "research_report.md").write_text(report, encoding="utf-8")
    sources = [
        config_path,
        root / "research/momentum_defender_gold_exception_gate.py",
        root / "research/run_momentum_defender_gold_exception_search.py",
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


def main():
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
