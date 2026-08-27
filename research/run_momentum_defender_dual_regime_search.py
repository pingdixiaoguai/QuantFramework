"""Dual search with asset-specific scores and five-day-multiple sleeve locks."""

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
from research.momentum_defender_gold_override import build_gold_override_context
from research.momentum_defender_gold_override_overfit import (
    cscv_pbo,
    expanding_walk_forward,
    paired_block_bootstrap,
    yearly_reality_check,
)
from research.momentum_defender_occam import performance
from research.momentum_defender_selected_asset_draqm import (
    STICKY_ENTRY_ASSET,
    AssetDRAQMPolicy,
    SelectedAssetDRAQMSpec,
    run_selected_asset_draqm_spec,
)
from research.momentum_volatility import load_ohlc
from research.run_momentum_defender_common_score_trimmed import (
    _add_metrics,
    _block_performance,
    _evaluate_single,
    _threshold_neighborhood,
)
from research.run_momentum_defender_log_qm_robust import (
    _event_stress,
    _friction,
    _selected_cost_schedule,
    _unique_paths,
)
from research.standard_report import generate_standard_report


DEFAULT_CONFIG = Path("research/configs/momentum_defender_dual_regime_research.yaml")
DEFAULT_OUTPUT = Path("experiments/20260825_momentum_defender_dual_regime_search")
ASSETS = ("510300.SH", "518880.SH")


def _load_config(path: Path) -> dict:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("dual-regime config must be a mapping")
    return config


def _profiles(config: dict) -> dict[str, FactorProfile]:
    return {
        profile_id: FactorProfile(
            profile_id,
            tuple(map(int, values["horizons"])),
            tuple(map(float, values["weights"])),
        )
        for profile_id, values in config["score_profiles"][
            "available_to_each_asset"
        ].items()
    }


def _policy_grid(config: dict, profiles: dict[str, FactorProfile]):
    stage = config["single_asset_stage"]
    gap = float(stage["minimum_hysteresis_gap"])
    result = {asset: {profile: [] for profile in profiles} for asset in ASSETS}
    for asset in ASSETS:
        for profile_id, profile in profiles.items():
            policies = {}
            for entry, recovery, entry_c, recovery_c in product(
                stage["defender_entry_percentiles"],
                stage["momentum_recovery_percentiles"],
                stage["defender_entry_confirmation_days"],
                stage["momentum_recovery_confirmation_days"],
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
                policies[policy.policy_id()] = policy
            result[asset][profile_id] = list(policies.values())
    return result


def _minimum_segment(table: pd.DataFrame) -> pd.DataFrame:
    result = table.copy()
    result["full_minimum_segment_sharpe"] = result[
        ["development_sharpe", "validation_sharpe", "recent_sharpe"]
    ].min(axis=1)
    return result


def _rank_one(pool: pd.DataFrame, fields: list[str], prefix: str) -> pd.DataFrame:
    ranked = pool.copy()
    percentiles = ranked[fields].rank(pct=True)
    ranked[f"{prefix}_min_percentile"] = percentiles.min(axis=1)
    ranked[f"{prefix}_mean_percentile"] = percentiles.mean(axis=1)
    return ranked.sort_values(
        [f"{prefix}_min_percentile", f"{prefix}_mean_percentile", *fields[:2]],
        ascending=False,
    )


def _select_policy_options(
    single: pd.DataFrame,
    lookup: dict[str, AssetDRAQMPolicy],
    config: dict,
) -> tuple[dict[str, dict[str, list[AssetDRAQMPolicy]]], pd.DataFrame]:
    stage = config["single_asset_stage"]
    options = {asset: {} for asset in ASSETS}
    rows = []
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
    combined_fields = [
        "full_annualized_return_252",
        "full_sharpe",
        "trimmed_annualized_return_252",
        "trimmed_sharpe",
        "threshold_neighborhood_full_sharpe_q25",
        "threshold_neighborhood_trimmed_sharpe_q25",
    ]
    for asset in ASSETS:
        for profile_id in single.loc[single["asset"].eq(asset), "profile_id"].unique():
            pool = single.loc[
                single["asset"].eq(asset)
                & single["profile_id"].eq(profile_id)
                & single["defender_entries"].ge(int(stage["minimum_defender_entries"]))
                & single["defender_days"].ge(int(stage["minimum_defender_days"]))
            ]
            selected: dict[str, str] = {}
            rankings = {
                "best_full": _rank_one(pool, full_fields, "full_policy"),
                "best_trimmed": _rank_one(pool, trimmed_fields, "trimmed_policy"),
                "best_combined_robust": _rank_one(
                    pool, combined_fields, "combined_policy"
                ),
            }
            for role, ranking in rankings.items():
                candidate_id = str(ranking.index[0])
                selected[candidate_id] = role
            options[asset][str(profile_id)] = [lookup[value] for value in selected]
            for rank, (candidate_id, role) in enumerate(selected.items(), start=1):
                rows.append(
                    {
                        "asset": asset,
                        "profile_id": profile_id,
                        "option_rank": rank,
                        "selection_role": role,
                        "policy_id": candidate_id,
                    }
                )
    return options, pd.DataFrame(rows)


def _evaluate_joint(
    data,
    momentum_target,
    features,
    options,
    baseline,
    ordinary_mask,
    config,
):
    joint = config["joint_stage"]
    combinations = []
    for csi_profile, gold_profile in product(
        options["510300.SH"], options["518880.SH"]
    ):
        for values in product(
            options["510300.SH"][csi_profile],
            options["518880.SH"][gold_profile],
            joint["momentum_lock_days"],
            joint["defender_lock_days"],
            joint["recovery_reference_modes"],
        ):
            combinations.append((csi_profile, gold_profile, *values))
    matrix = np.empty((len(data.calendar), len(combinations)), dtype=np.float32)
    records = []
    ids = []
    specs = {}
    for position, values in enumerate(combinations):
        csi_profile, gold_profile, csi, gold, momentum_hold, defender_hold, mode = values
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
                "profile_510300": csi_profile,
                "profile_518880": gold_profile,
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
        if (position + 1) % 1000 == 0 or position + 1 == len(combinations):
            print(f"joint: evaluated {position + 1}/{len(combinations)}", flush=True)
    returns = pd.DataFrame(matrix, index=data.calendar, columns=ids)
    metadata = pd.DataFrame(records).set_index("candidate_id")
    table = _minimum_segment(
        _add_metrics(metadata, returns, baseline, ordinary_mask, config)
    )
    return table, returns, specs


def _add_joint_neighborhood(
    table: pd.DataFrame,
    single: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    result = table.copy()
    locks_m = list(map(int, config["joint_stage"]["momentum_lock_days"]))
    locks_d = list(map(int, config["joint_stage"]["defender_lock_days"]))
    map_m = {value: position for position, value in enumerate(locks_m)}
    map_d = {value: position for position, value in enumerate(locks_d)}
    result["_mh"] = result["momentum_lock_days"].map(map_m)
    result["_dh"] = result["defender_lock_days"].map(map_d)
    rows = {}
    for _, group in result.groupby(
        [
            "profile_510300",
            "profile_518880",
            "policy_510300",
            "policy_518880",
            "recovery_mode",
        ],
        sort=False,
    ):
        coords = group[["_mh", "_dh"]].to_numpy(int)
        values = {
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
                    f"lock_neighborhood_{field}_q25": float(
                        np.quantile(array[members], 0.25)
                    )
                    for field, array in values.items()
                },
            }
    result = result.drop(columns=["_mh", "_dh"]).join(
        pd.DataFrame.from_dict(rows, orient="index")
    )
    maps = {
        "full_annualized": single[
            "threshold_neighborhood_full_annualized_q25"
        ].to_dict(),
        "full_sharpe": single["threshold_neighborhood_full_sharpe_q25"].to_dict(),
        "trimmed_annualized": single[
            "threshold_neighborhood_trimmed_annualized_q25"
        ].to_dict(),
        "trimmed_sharpe": single[
            "threshold_neighborhood_trimmed_sharpe_q25"
        ].to_dict(),
    }
    for label, values in maps.items():
        result[f"policy_neighborhood_{label}_q25_min"] = result.apply(
            lambda row: min(
                values[row["policy_510300"]], values[row["policy_518880"]]
            ),
            axis=1,
        )
    return result


def _select(
    table: pd.DataFrame,
    config: dict,
    selection_key: str,
    prefix: str,
) -> tuple[pd.Series, pd.DataFrame]:
    result = table.copy()
    joint = config["joint_stage"]
    eligible = (
        result["defender_entries"].ge(int(joint["minimum_defender_entries"]))
        & result["defender_days"].ge(int(joint["minimum_defender_days"]))
        & result["full_minimum_segment_sharpe"].gt(
            float(joint["minimum_segment_sharpe"])
        )
        & result["trimmed_minimum_segment_sharpe"].gt(
            float(joint["minimum_segment_sharpe"])
        )
    )
    pool = result.loc[eligible].copy()
    fields = list(config[selection_key]["ranking_fields"])
    ranks = pool[fields].rank(pct=True)
    pool[f"{prefix}_robust_min_percentile"] = ranks.min(axis=1)
    pool[f"{prefix}_robust_mean_percentile"] = ranks.mean(axis=1)
    result.loc[pool.index, f"{prefix}_robust_min_percentile"] = pool[
        f"{prefix}_robust_min_percentile"
    ]
    result.loc[pool.index, f"{prefix}_robust_mean_percentile"] = pool[
        f"{prefix}_robust_mean_percentile"
    ]
    order = [
        f"{prefix}_robust_min_percentile",
        f"{prefix}_robust_mean_percentile",
        *config[selection_key]["tie_breakers"],
    ]
    selected = pool.sort_values(order, ascending=False).iloc[0]
    result[f"eligible_{prefix}"] = eligible
    return selected, result


def _candidate_outputs(
    label,
    selected,
    specs,
    data,
    momentum_target,
    features,
    momentum_returns,
    momentum_actual_target,
    anchor_returns,
    anchor_target,
    context,
    ordinary,
    extreme,
    output,
    config,
):
    candidate_id = str(selected.name)
    spec = specs[candidate_id]
    run = run_selected_asset_draqm_spec(data, momentum_target, features, spec)
    returns = pd.Series(run.returns, index=data.calendar, name=candidate_id)
    target = pd.Series(
        [data.candidates[value] for value in run.actual_target], index=data.calendar
    )
    bootstrap_momentum, bootstrap_momentum_summary = paired_block_bootstrap(
        returns,
        momentum_returns,
        block_size=20,
        repetitions=5000,
        seed=int(config["overfit_checks"]["random_seed"]),
    )
    bootstrap_anchor, bootstrap_anchor_summary = paired_block_bootstrap(
        returns,
        anchor_returns,
        block_size=20,
        repetitions=5000,
        seed=int(config["overfit_checks"]["random_seed"]),
    )
    events_momentum, leave_momentum, top_momentum, event_momentum_summary = _event_stress(
        returns, momentum_returns, target, momentum_actual_target, [1, 2, 3]
    )
    events_anchor, leave_anchor, top_anchor, event_anchor_summary = _event_stress(
        returns, anchor_returns, target, anchor_target, [1, 2, 3]
    )
    costs = _selected_cost_schedule(context, data, run.actual_target)
    friction = _friction(returns, costs, [1.0, 2.0, 3.0])
    daily = run.state.copy()
    daily["ordinary_selection_day"] = ordinary
    daily["shock_score_at_open"] = extreme.shock_score_at_open
    daily["return"] = returns
    daily["nav"] = (1.0 + returns).cumprod()
    daily["requested_candidate"] = [
        data.candidates[value] for value in run.requested_target
    ]
    daily["actual_candidate"] = target
    daily["cost_rate_at_open"] = costs
    daily.to_csv(output / f"selected_{label}_daily.csv")
    daily.to_parquet(output / f"selected_{label}_daily.parquet")
    bootstrap_momentum.to_csv(output / f"selected_{label}_bootstrap_vs_momentum.csv", index=False)
    bootstrap_anchor.to_csv(output / f"selected_{label}_bootstrap_vs_universal_anchor.csv", index=False)
    events_momentum.to_csv(output / f"selected_{label}_events_vs_momentum.csv", index=False)
    leave_momentum.to_csv(output / f"selected_{label}_leave_event_vs_momentum.csv", index=False)
    top_momentum.to_csv(output / f"selected_{label}_top_event_deletion_vs_momentum.csv", index=False)
    events_anchor.to_csv(output / f"selected_{label}_events_vs_universal_anchor.csv", index=False)
    leave_anchor.to_csv(output / f"selected_{label}_leave_event_vs_universal_anchor.csv", index=False)
    top_anchor.to_csv(output / f"selected_{label}_top_event_deletion_vs_universal_anchor.csv", index=False)
    friction.to_csv(output / f"selected_{label}_friction.csv", index=False)
    generate_standard_report(
        returns,
        momentum_returns,
        "Log-QM Momentum",
        output / f"selected_{label}_vs_momentum.html",
        {"candidate_id": candidate_id, "selection": label},
    )
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
        for asset, policy in spec.policies.items()
        if policy is not None
    }
    return {
        "candidate_id": candidate_id,
        "policies": policies,
        "state_policy": {
            "momentum_lock_days": spec.momentum_lock_days,
            "defender_lock_days": spec.defender_lock_days,
            "recovery_mode": spec.recovery_mode,
        },
        "full_metrics": performance(returns),
        "ordinary_metrics": performance(returns.loc[ordinary]),
        "shock_block_metrics": performance(returns.loc[~ordinary]),
        "defender_entries": run.defender_entries,
        "defender_days": run.defender_days,
        "sleeve_switches": run.sleeve_switches,
        "candidate_switches": run.candidate_switches,
        "bootstrap_vs_momentum": bootstrap_momentum_summary,
        "bootstrap_vs_universal_anchor": bootstrap_anchor_summary,
        "events_vs_momentum": event_momentum_summary,
        "events_vs_universal_anchor": event_anchor_summary,
        "three_x_cost_metrics": friction.loc[
            friction["cost_multiplier"].eq(3.0)
        ].iloc[0].to_dict(),
        "daily_return_sha256_float64_le": hashlib.sha256(
            returns.to_numpy(dtype="<f8").tobytes()
        ).hexdigest(),
        "latest_state": {
            "date": daily.index[-1].date().isoformat(),
            "risk_on": bool(daily.iloc[-1]["risk_on"]),
            "momentum_top1": str(daily.iloc[-1]["momentum_top1_at_open"]),
            "actual_candidate": str(daily.iloc[-1]["actual_candidate"]),
        },
    }


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_experiment(root: Path, config_path: Path, output: Path):
    config = _load_config(config_path)
    if QUALITY_METADATA["version"] != config["frozen_layers"][
        "momentum_factor_version"
    ]:
        raise AssertionError("frozen Momentum factor mismatch")
    profiles = _profiles(config)
    policy_grid = _policy_grid(config, profiles)
    end = pd.Timestamp(config["periods"]["full"][1])
    context = build_gold_override_context(root, end=end.date())
    data = build_exact_execution_data(context)
    closes = {asset: load_ohlc(asset, end.date())["close"] for asset in ASSETS}
    score = config["score_profiles"]
    features = {
        asset: build_downside_raqm_features(
            closes[asset],
            data.calendar,
            profiles,
            {"rolling_504_strict_lag": 504},
            min_history=int(score["percentile_min_history"]),
            volatility_floor_annual=float(score["volatility_floor_annual"]),
            winsor_limit=float(score["winsor_limit"]),
        )
        for asset in ASSETS
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
    momentum_actual_target = pd.Series(
        [data.candidates[value] for value in momentum_actual], index=data.calendar
    )
    anchor_profile = FactorProfile(
        "universal_w30_40_25_75", (30, 40), (0.25, 0.75)
    )
    anchor_features = build_downside_raqm_features(
        closes["510300.SH"],
        data.calendar,
        {anchor_profile.profile_id: anchor_profile},
        {"rolling_504_strict_lag": 504},
        min_history=252,
        volatility_floor_annual=0.08,
        winsor_limit=3.0,
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
        context.momentum_target,
        features,
        policy_grid,
        momentum_returns,
        ordinary,
        config,
    )
    single = _minimum_segment(_threshold_neighborhood(single, config))
    options, option_frame = _select_policy_options(single, lookup, config)
    joint, joint_returns, specs = _evaluate_joint(
        data,
        context.momentum_target,
        features,
        options,
        momentum_returns,
        ordinary,
        config,
    )
    joint = _add_joint_neighborhood(joint, single, config)
    selected_full, joint = _select(
        joint,
        config,
        "selection_including_extremes",
        "full_selection",
    )
    selected_trimmed, joint = _select(
        joint,
        config,
        "selection_excluding_extremes",
        "trimmed_selection",
    )
    output.mkdir(parents=True, exist_ok=True)
    extreme.blocks.to_csv(output / "shock_blocks.csv")
    pd.concat(
        [extreme.selection_mask, extreme.shock_score_at_open, extreme.asset_shock_at_open],
        axis=1,
    ).to_csv(output / "selection_mask.csv")
    single.to_csv(output / "single_asset_grid.csv")
    option_frame.to_csv(output / "single_asset_options_for_joint.csv", index=False)
    joint.to_csv(output / "joint_grid.csv")
    all_returns = _unique_paths(pd.concat([single_returns, joint_returns], axis=1))
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
    walk.to_csv(output / "expanding_walk_forward.csv", index=False)
    block_returns = _block_performance(
        extreme.blocks,
        {
            "momentum": momentum_returns,
            "universal_anchor": anchor_returns,
        },
    )
    block_returns.to_csv(output / "block_performance.csv", index=False)
    candidates = {
        "including_extremes": _candidate_outputs(
            "including_extremes",
            selected_full,
            specs,
            data,
            context.momentum_target,
            features,
            momentum_returns,
            momentum_actual_target,
            anchor_returns,
            anchor_target,
            context,
            ordinary,
            extreme,
            output,
            config,
        ),
        "excluding_extremes": _candidate_outputs(
            "excluding_extremes",
            selected_trimmed,
            specs,
            data,
            context.momentum_target,
            features,
            momentum_returns,
            momentum_actual_target,
            anchor_returns,
            anchor_target,
            context,
            ordinary,
            extreme,
            output,
            config,
        ),
    }
    audit = {
        "experiment_id": config["experiment"]["id"],
        "lock_grid": {
            "momentum": config["joint_stage"]["momentum_lock_days"],
            "defender": config["joint_stage"]["defender_lock_days"],
            "all_nonnegative_multiples_of_five": True,
        },
        "trim": {
            "candidate_independent": True,
            "blocks": int(len(extreme.blocks)),
            "excluded_blocks": int(
                extreme.blocks["excluded_from_selection"].sum()
            ),
            "excluded_sessions": int((~ordinary).sum()),
        },
        "single_candidate_ids": int(len(single)),
        "joint_candidate_ids": int(len(joint)),
        "unique_return_paths": int(all_returns.shape[1]),
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
            "universal_anchor": performance(anchor_returns),
        },
    }
    (output / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for label, candidate in candidates.items():
        selected_config = {
            "strategy_id": f"momentum_defender_dual_regime_{label}_v1",
            "status": "research_candidate_not_production",
            "selection_objective": label,
            "policies": candidate["policies"],
            "state_policy": candidate["state_policy"],
            "checkpoint": {
                **candidate["full_metrics"],
                "defender_entries": candidate["defender_entries"],
                "defender_days": candidate["defender_days"],
                "sleeve_switches": candidate["sleeve_switches"],
                "candidate_switches": candidate["candidate_switches"],
                "daily_return_sha256_float64_le": candidate[
                    "daily_return_sha256_float64_le"
                ],
            },
        }
        (output / f"selected_{label}_config.yaml").write_text(
            yaml.safe_dump(selected_config, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
    report = f"""# 包含/不包含极端行情的双目标寻参

同一候选族分别按全样本和普通区间独立排序。全样本候选：
`{candidates['including_extremes']['candidate_id']}`；普通区间候选：
`{candidates['excluding_extremes']['candidate_id']}`。

|选择目标|全样本年化|全样本Sharpe|普通区间年化|普通区间Sharpe|
|---|---:|---:|---:|---:|
|包含极端行情|{candidates['including_extremes']['full_metrics']['annualized_return_252']:.2%}|{candidates['including_extremes']['full_metrics']['sharpe']:.3f}|{candidates['including_extremes']['ordinary_metrics']['annualized_return_252']:.2%}|{candidates['including_extremes']['ordinary_metrics']['sharpe']:.3f}|
|不包含极端行情|{candidates['excluding_extremes']['full_metrics']['annualized_return_252']:.2%}|{candidates['excluding_extremes']['full_metrics']['sharpe']:.3f}|{candidates['excluding_extremes']['ordinary_metrics']['annualized_return_252']:.2%}|{candidates['excluding_extremes']['ordinary_metrics']['sharpe']:.3f}|
"""
    (output / "research_report.md").write_text(report, encoding="utf-8")
    sources = [
        config_path,
        root / "research/momentum_defender_selected_asset_draqm.py",
        root / "research/run_momentum_defender_dual_regime_search.py",
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
