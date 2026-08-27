"""Staged rolling-500 A/B/C quantile search for formal W40 Top1 escapes."""

from __future__ import annotations

import argparse
import hashlib
import json
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
from research.momentum_defender_downside_raqm import build_exact_execution_data
from research.momentum_defender_gold_override_overfit import (
    cscv_pbo,
    expanding_walk_forward,
    leave_one_year_selection,
    paired_block_bootstrap,
    yearly_reality_check,
)
from research.momentum_defender_occam import MOMENTUM_ASSETS, performance
from research.momentum_defender_w40_quantile_escape import (
    HISTORY_WINDOW,
    MIN_HISTORY,
    QuantileXYPolicy,
    policy_set_id,
    rolling_quantiles_at_open,
    run_quantile_escape,
)
from research.momentum_defender_w40_top1_escape import (
    DEFENDER_ELIGIBILITY_DAYS,
    QUALITY_WINDOW,
    TOP1_HARD_HOLD_DAYS,
    quality_metrics_at_open,
)
from research.momentum_volatility import load_ohlc
from research.run_momentum_defender_common_score_trimmed import _add_metrics
from research.run_momentum_defender_log_qm_robust import (
    _event_stress,
    _friction,
    _selected_cost_schedule,
    _unique_paths,
)
from research.run_momentum_defender_w40_asset_specific_escape_search import (
    _joint_select,
    _plain,
    _rank_select,
)
from research.standard_report import generate_standard_report
from strategy.momentum_defender_w40_full_equity import (
    FORMAL_STRATEGY_ID,
    run_formal_strategy,
)


DEFAULT_CONFIG = Path(
    "research/configs/momentum_defender_w40_quantile_escape_search.yaml"
)
DEFAULT_OUTPUT = Path(
    "experiments/20260825_momentum_defender_w40_quantile_escape_search"
)
PRIOR_PATHS = Path(
    "experiments/20260825_momentum_defender_w40_asset_specific_escape_search/"
    "global_unique_candidate_returns.parquet"
)


def _load(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("quantile escape config must be a mapping")
    return value


def _policy_grid(config: dict) -> list[QuantileXYPolicy]:
    grid = config["quantile_grid"]
    result = {}
    for a, b in product(grid["momentum_entry_a"], grid["momentum_exit_b"]):
        if float(b) > float(a):
            continue
        policy = QuantileXYPolicy(float(a), float(b))
        result[policy.policy_id] = policy
    return list(result.values())


def _metrics_table(
    metadata: pd.DataFrame,
    returns: pd.DataFrame,
    baseline: pd.Series,
    ordinary: pd.Series,
    config: dict,
) -> pd.DataFrame:
    result = _add_metrics(metadata, returns, baseline, ordinary, config)
    result = result.rename(
        columns={
            column: column.replace("trimmed_", "ordinary_", 1)
            for column in result.columns
            if column.startswith("trimmed_")
        }
    )
    result["full_minimum_segment_sharpe"] = result[
        ["development_sharpe", "validation_sharpe", "recent_sharpe"]
    ].min(axis=1)
    return result


def _single_neighborhood(table: pd.DataFrame, config: dict) -> pd.DataFrame:
    grid = config["quantile_grid"]
    lookups = {
        "entry_a": {
            float(value): position
            for position, value in enumerate(grid["momentum_entry_a"])
        },
        "exit_b": {
            float(value): position
            for position, value in enumerate(grid["momentum_exit_b"])
        },
        "defender_c": {
            float(value): position
            for position, value in enumerate(grid["common_defender_c"])
        },
    }
    summaries = {}
    for asset, group in table.groupby("asset", sort=False):
        coords = np.column_stack(
            [group[field].map(lookup).to_numpy(int) for field, lookup in lookups.items()]
        )
        arrays = {
            "full_annualized": group["full_annualized_return_252"].to_numpy(float),
            "full_sharpe": group["full_sharpe"].to_numpy(float),
            "ordinary_annualized": group[
                "ordinary_annualized_return_252"
            ].to_numpy(float),
            "ordinary_sharpe": group["ordinary_sharpe"].to_numpy(float),
        }
        for position, candidate_id in enumerate(group.index):
            members = np.all(np.abs(coords - coords[position]) <= 1, axis=1)
            summaries[str(candidate_id)] = {
                "neighborhood_count": int(members.sum()),
                **{
                    f"neighborhood_{name}_q25": float(
                        np.quantile(values[members], 0.25)
                    )
                    for name, values in arrays.items()
                },
            }
    return table.join(pd.DataFrame.from_dict(summaries, orient="index"))


def _select_options_by_c(
    table: pd.DataFrame,
    policies: dict[str, QuantileXYPolicy],
    config: dict,
) -> dict[float, dict[str, list[QuantileXYPolicy | None]]]:
    staged = config["staged_selection"]
    count = int(staged["policies_per_asset_per_c"])
    fields_joint = [
        "full_annualized_return_252",
        "full_sharpe",
        "ordinary_annualized_return_252",
        "ordinary_sharpe",
        "neighborhood_full_sharpe_q25",
        "neighborhood_ordinary_sharpe_q25",
        "full_minimum_segment_sharpe",
    ]
    result = {}
    for c in map(float, config["quantile_grid"]["common_defender_c"]):
        result[c] = {}
        for asset in MOMENTUM_ASSETS:
            rows = table["asset"].eq(asset) & table["defender_c"].eq(c)
            eligible = (
                rows
                & table["escape_entries"].ge(
                    int(staged["minimum_single_asset_entries"])
                )
                & table["lock_break_entries"].ge(
                    int(staged["minimum_single_asset_lock_break_entries"])
                )
                & table["escape_days"].ge(
                    int(staged["minimum_single_asset_days"])
                )
            )
            ids = []
            for fields in (
                ["full_annualized_return_252", "full_sharpe"],
                ["ordinary_annualized_return_252", "ordinary_sharpe"],
                fields_joint,
            ):
                for candidate_id in _rank_select(table, eligible, fields, 1):
                    if candidate_id not in ids:
                        ids.append(candidate_id)
            if len(ids) < count:
                for candidate_id in _rank_select(
                    table, eligible, fields_joint, count
                ):
                    if candidate_id not in ids:
                        ids.append(candidate_id)
                    if len(ids) == count:
                        break
            selected = [
                policies[str(table.at[candidate_id, "policy_id"])]
                for candidate_id in ids[:count]
            ]
            result[c][asset] = [None, *selected]
    return result


def _joint_neighborhood(table: pd.DataFrame) -> pd.DataFrame:
    option_columns = [f"option_{asset}" for asset in MOMENTUM_ASSETS]
    coords = table[option_columns].to_numpy(int)
    c_values = table["defender_c"].to_numpy(float)
    arrays = {
        "full_annualized": table["full_annualized_return_252"].to_numpy(float),
        "full_sharpe": table["full_sharpe"].to_numpy(float),
        "ordinary_annualized": table[
            "ordinary_annualized_return_252"
        ].to_numpy(float),
        "ordinary_sharpe": table["ordinary_sharpe"].to_numpy(float),
        "full_delta_annualized": table[
            "full_delta_annualized_return_252"
        ].to_numpy(float),
        "full_delta_sharpe": table["full_delta_sharpe"].to_numpy(float),
    }
    summaries = {}
    for position, candidate_id in enumerate(table.index):
        members = (c_values == c_values[position]) & (
            np.sum(coords != coords[position], axis=1) <= 1
        )
        summaries[str(candidate_id)] = {
            "neighborhood_count": int(members.sum()),
            **{
                f"neighborhood_{name}_q25": float(
                    np.quantile(values[members], 0.25)
                )
                for name, values in arrays.items()
            },
            "neighborhood_full_dual_pass_rate": float(
                np.mean(
                    (arrays["full_delta_annualized"][members] > 0.0)
                    & (arrays["full_delta_sharpe"][members] > 0.0)
                )
            ),
        }
    return table.join(pd.DataFrame.from_dict(summaries, orient="index"))


def run_experiment(root: Path, config_path: Path, output: Path) -> dict:
    config = _load(config_path)
    frozen = config["frozen_layers"]
    if frozen["formal_strategy_id"] != FORMAL_STRATEGY_ID:
        raise AssertionError("quantile search is not pinned to formal strategy")
    if QUALITY_METADATA["version"] != frozen["momentum_factor_version"]:
        raise AssertionError("Momentum factor version mismatch")
    if not (
        int(frozen["metric_window"]) == QUALITY_WINDOW
        and int(frozen["quantile_history_window"]) == HISTORY_WINDOW
        and int(frozen["quantile_min_history"]) == MIN_HISTORY
        and int(frozen["defender_eligibility_days"]) == DEFENDER_ELIGIBILITY_DAYS
        and int(frozen["top1_hard_hold_days"]) == TOP1_HARD_HOLD_DAYS
    ):
        raise AssertionError("fixed quantile mechanism mismatch")
    output.mkdir(parents=True, exist_ok=True)
    cutoff = pd.Timestamp(config["periods"]["full"][1]).date()
    formal = run_formal_strategy(root, end=cutoff)
    context = formal.context
    baseline = formal.daily["return"].astype(float)
    metrics = quality_metrics_at_open(context)
    all_quantiles = sorted(
        {
            *map(float, config["quantile_grid"]["momentum_entry_a"]),
            *map(float, config["quantile_grid"]["momentum_exit_b"]),
            *map(float, config["quantile_grid"]["common_defender_c"]),
        }
    )
    quantile_frames = rolling_quantiles_at_open(metrics, all_quantiles)
    policies = _policy_grid(config)
    policy_lookup = {policy.policy_id: policy for policy in policies}

    trim = config["extreme_block_trim"]
    extreme = build_extreme_block_mask(
        {
            asset: load_ohlc(asset, cutoff)["close"]
            for asset in trim["shock_assets"]
        },
        context.calendar,
        ExtremeBlockSpec(
            shock_return_window=int(trim["shock_return_window"]),
            block_length_sessions=int(trim["block_length_sessions"]),
            excluded_block_fraction=float(trim["excluded_block_fraction"]),
            normalization_mode=str(trim["normalization_mode"]),
        ),
    )
    ordinary = extreme.selection_mask.astype(bool)
    extreme.blocks.to_csv(output / "shock_blocks.csv")

    single_records = []
    single_returns = {}
    for c in map(float, config["quantile_grid"]["common_defender_c"]):
        for asset in MOMENTUM_ASSETS:
            for policy in policies:
                policy_set = {
                    candidate: policy if candidate == asset else None
                    for candidate in MOMENTUM_ASSETS
                }
                run = run_quantile_escape(
                    context,
                    formal.state,
                    c,
                    policy_set,
                    metrics=metrics,
                    quantile_frames=quantile_frames,
                )
                candidate_id = f"c{c:.2f}|{asset}|{policy.policy_id}"
                single_returns[candidate_id] = run.daily["return"].to_numpy(float)
                single_records.append(
                    {
                        "candidate_id": candidate_id,
                        "asset": asset,
                        "defender_c": c,
                        "policy_id": policy.policy_id,
                        "entry_a": policy.entry_a,
                        "exit_b": policy.exit_b,
                        "escape_entries": run.audit["escape_entries"],
                        "lock_break_entries": run.audit["lock_break_entries"],
                        "escape_days": run.audit["escape_days"],
                    }
                )
        print(f"single quantile C complete: {c:.2f}", flush=True)
    single_returns_frame = pd.DataFrame(single_returns, index=context.calendar)
    single_metadata = pd.DataFrame(single_records).set_index("candidate_id")
    single_table = _single_neighborhood(
        _metrics_table(
            single_metadata,
            single_returns_frame,
            baseline,
            ordinary,
            config,
        ),
        config,
    )
    options_by_c = _select_options_by_c(
        single_table, policy_lookup, config
    )
    single_table.to_csv(output / "single_asset_candidates.csv")
    (output / "staged_options.json").write_text(
        json.dumps(
            {
                f"c{c:.2f}": {
                    asset: [policy.policy_id if policy else "off" for policy in values]
                    for asset, values in options.items()
                }
                for c, options in options_by_c.items()
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    combo_records = []
    combo_returns = {}
    combo_runs = {}
    combo_policies = {}
    for c, options in options_by_c.items():
        ordered = [options[asset] for asset in MOMENTUM_ASSETS]
        for selected in product(*ordered):
            policy_set = dict(zip(MOMENTUM_ASSETS, selected, strict=True))
            run = run_quantile_escape(
                context,
                formal.state,
                c,
                policy_set,
                metrics=metrics,
                quantile_frames=quantile_frames,
            )
            candidate_id = policy_set_id(c, policy_set)
            combo_returns[candidate_id] = run.daily["return"].to_numpy(float)
            combo_runs[candidate_id] = run
            combo_policies[candidate_id] = policy_set
            combo_records.append(
                {
                    "candidate_id": candidate_id,
                    "defender_c": c,
                    "enabled_assets": run.audit["enabled_assets"],
                    "escape_entries": run.audit["escape_entries"],
                    "lock_break_entries": run.audit["lock_break_entries"],
                    "escape_days": run.audit["escape_days"],
                    **{
                        f"policy_{asset}": policy_set[asset].policy_id
                        if policy_set[asset]
                        else "off"
                        for asset in MOMENTUM_ASSETS
                    },
                    **{
                        f"option_{asset}": options[asset].index(policy_set[asset])
                        for asset in MOMENTUM_ASSETS
                    },
                }
            )
    combo_returns_frame = pd.DataFrame(combo_returns, index=context.calendar)
    disabled_columns = [
        column
        for column in combo_returns_frame
        if "510300=off__159915=off__513100=off__518880=off" in column
    ]
    disabled_parity = max(
        float((combo_returns_frame[column] - baseline).abs().max())
        for column in disabled_columns
    )
    if disabled_parity > 1e-14:
        raise AssertionError("disabled quantile policies do not match baseline")
    combo_metadata = pd.DataFrame(combo_records).set_index("candidate_id")
    combo_table = _joint_neighborhood(
        _metrics_table(
            combo_metadata,
            combo_returns_frame,
            baseline,
            ordinary,
            config,
        )
    )
    staged = config["staged_selection"]
    eligible = (
        combo_table["escape_entries"].ge(int(staged["minimum_joint_entries"]))
        & combo_table["lock_break_entries"].ge(
            int(staged["minimum_joint_lock_break_entries"])
        )
        & combo_table["escape_days"].ge(int(staged["minimum_joint_days"]))
        & combo_table["full_max_drawdown"].ge(
            float(staged["maximum_full_drawdown"])
        )
        & combo_table["full_minimum_segment_sharpe"].ge(
            float(staged["minimum_full_segment_sharpe"])
        )
        & combo_table["ordinary_minimum_segment_sharpe"].ge(
            float(staged["minimum_ordinary_segment_sharpe"])
        )
    )
    strict_dual = (
        eligible
        & combo_table["full_delta_annualized_return_252"].gt(0.0)
        & combo_table["full_delta_sharpe"].gt(0.0)
    )
    pool = strict_dual if strict_dual.any() else eligible
    selection = config["selection"]
    selected_full = _joint_select(
        combo_table, pool, list(selection["ranking_fields_full"])
    )
    selected_ordinary = _joint_select(
        combo_table, pool, list(selection["ranking_fields_ordinary"])
    )
    selected_joint = _joint_select(
        combo_table, pool, list(selection["ranking_fields_joint"])
    )
    combo_table["eligible"] = eligible
    combo_table["strict_full_dual_improvement"] = strict_dual
    combo_table["selected_full"] = combo_table.index == str(selected_full.name)
    combo_table["selected_ordinary"] = combo_table.index == str(selected_ordinary.name)
    combo_table["selected_joint"] = combo_table.index == str(selected_joint.name)
    combo_table.to_csv(output / "combination_candidates.csv")
    unique_combo = _unique_paths(combo_returns_frame)
    unique_combo.to_parquet(output / "unique_combination_returns.parquet")
    all_current = _unique_paths(
        pd.concat([single_returns_frame, combo_returns_frame], axis=1)
    )
    all_current.to_parquet(output / "all_current_unique_returns.parquet")

    cscv, cscv_summary = cscv_pbo(unique_combo, baseline, block_count=12)
    cscv_ordinary, cscv_ordinary_summary = cscv_pbo(
        unique_combo.loc[ordinary], baseline.loc[ordinary], block_count=12
    )
    cscv.to_csv(output / "cscv_joint.csv", index=False)
    cscv_ordinary.to_csv(output / "cscv_joint_ordinary.csv", index=False)
    walk = expanding_walk_forward(unique_combo, baseline)
    leave = leave_one_year_selection(unique_combo, baseline)
    walk.to_csv(output / "walk_forward.csv", index=False)
    leave.to_csv(output / "leave_one_year.csv", index=False)
    family_reality = yearly_reality_check(
        all_current,
        baseline,
        repetitions=int(
            config["overfit_checks"]["yearly_reality_check_repetitions"]
        ),
        seed=int(config["overfit_checks"]["random_seed"]),
    )
    prior = pd.read_parquet(root / PRIOR_PATHS)
    prior.columns = [f"prior::{column}" for column in prior]
    current = all_current.copy()
    current.columns = [f"quantile::{column}" for column in current]
    global_paths = _unique_paths(pd.concat([prior, current], axis=1))
    global_paths.to_parquet(output / "global_unique_candidate_returns.parquet")
    global_reality = {
        "full": yearly_reality_check(
            global_paths,
            baseline,
            repetitions=int(
                config["overfit_checks"]["yearly_reality_check_repetitions"]
            ),
            seed=int(config["overfit_checks"]["random_seed"]),
        ),
        "ordinary": yearly_reality_check(
            global_paths.loc[ordinary],
            baseline.loc[ordinary],
            repetitions=int(
                config["overfit_checks"]["yearly_reality_check_repetitions"]
            ),
            seed=int(config["overfit_checks"]["random_seed"]),
        ),
    }

    selected_id = str(selected_joint.name)
    selected_run = combo_runs[selected_id]
    selected_returns = selected_run.daily["return"].astype(float)
    bootstrap, bootstrap_summary = paired_block_bootstrap(
        selected_returns,
        baseline,
        block_size=int(config["overfit_checks"]["paired_block_bootstrap_block"]),
        repetitions=int(
            config["overfit_checks"]["paired_block_bootstrap_repetitions"]
        ),
        seed=int(config["overfit_checks"]["random_seed"]),
    )
    bootstrap.to_csv(output / "selected_bootstrap.csv", index=False)
    events, leave_events, deleted, event_summary = _event_stress(
        selected_returns,
        baseline,
        selected_run.state["target_candidate"].astype(str),
        formal.daily["candidate"].astype(str),
        list(map(int, config["overfit_checks"]["top_positive_event_deletions"])),
    )
    events.to_csv(output / "selected_events.csv", index=False)
    leave_events.to_csv(output / "selected_leave_event.csv", index=False)
    deleted.to_csv(output / "selected_delete_top_events.csv", index=False)
    data = build_exact_execution_data(context)
    actual_target = selected_run.daily["candidate"].map(
        data.candidate_index
    ).to_numpy(int)
    costs = _selected_cost_schedule(context, data, actual_target)
    friction = _friction(
        selected_returns,
        costs,
        list(map(float, config["overfit_checks"]["friction_cost_multipliers"])),
    )
    friction.to_csv(output / "selected_friction.csv", index=False)
    selected_run.state.join(
        selected_run.daily, rsuffix="_execution"
    ).to_csv(output / "selected_daily.csv")
    selected_run.state.join(
        selected_run.daily, rsuffix="_execution"
    ).to_parquet(output / "selected_daily.parquet")
    generate_standard_report(
        selected_returns,
        baseline,
        "Current Formal W40 Full Equity",
        output / "selected_vs_formal.html",
        {"experiment": config["experiment"], "selected_id": selected_id},
    )
    policy_set = combo_policies[selected_id]
    target = config["comparison_target"]
    matched_fixed_candidate = bool(
        performance(selected_returns)["annualized_return_252"]
        >= float(target["require_point_annualized_not_below"])
        and performance(selected_returns)["sharpe"]
        >= float(target["require_point_sharpe_not_below"])
    )
    selected_config = {
        "strategy_id": "momentum_defender_w40_quantile_escape_candidate_v1",
        "status": "research_candidate_not_production",
        "base_strategy_id": FORMAL_STRATEGY_ID,
        "common_defender_c": float(selected_joint["defender_c"]),
        "policies": {
            asset: (
                {"entry_a": policy.entry_a, "exit_b": policy.exit_b}
                if policy
                else None
            )
            for asset, policy in policy_set.items()
        },
        "history_window": HISTORY_WINDOW,
        "minimum_history": MIN_HISTORY,
        "checkpoint": {
            **performance(selected_returns),
            "ordinary": performance(selected_returns.loc[ordinary]),
            "escape_entries": selected_run.audit["escape_entries"],
            "lock_break_entries": selected_run.audit["lock_break_entries"],
            "escape_days": selected_run.audit["escape_days"],
            "daily_return_sha256_float64_le": hashlib.sha256(
                selected_returns.to_numpy(dtype="<f8").tobytes()
            ).hexdigest(),
        },
        "matched_or_better_than_fixed_gold_candidate": matched_fixed_candidate,
        "bootstrap_vs_formal": bootstrap_summary,
        "events_vs_formal": event_summary,
        "three_x_cost": friction.loc[
            friction["cost_multiplier"].eq(3.0)
        ].iloc[0].to_dict(),
        "production_replacement": False,
    }
    (output / "selected_config.yaml").write_text(
        yaml.safe_dump(_plain(selected_config), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    audit = {
        "experiment": config["experiment"],
        "baseline": performance(baseline),
        "disabled_parity_max_abs_error": disabled_parity,
        "search": {
            "single_asset_parameter_ids": len(single_table),
            "combination_ids": len(combo_table),
            "unique_combination_paths": int(unique_combo.shape[1]),
            "all_current_unique_paths": int(all_current.shape[1]),
            "eligible_combinations": int(eligible.sum()),
            "strict_full_dual_improvement": int(strict_dual.sum()),
            "selected_full": str(selected_full.name),
            "selected_ordinary": str(selected_ordinary.name),
            "selected_joint": selected_id,
        },
        "selected": _plain(selected_joint.to_dict()),
        "selected_checkpoint": _plain(selected_config),
        "matched_or_better_than_fixed_gold_candidate": matched_fixed_candidate,
        "joint_cscv": {
            "full": cscv_summary,
            "ordinary": cscv_ordinary_summary,
        },
        "family_reality_check": family_reality,
        "global_paths": {
            "input_prior": int(prior.shape[1]),
            "input_current": int(all_current.shape[1]),
            "unique": int(global_paths.shape[1]),
            "reality_check": global_reality,
        },
        "walk_forward": {
            "return_win_rate": float(walk["test_return_delta"].gt(0.0).mean()),
            "sharpe_win_rate": float(walk["test_sharpe_delta"].gt(0.0).mean()),
        },
        "leave_one_year": {
            "return_win_rate": float(leave["test_return_delta"].gt(0.0).mean()),
            "sharpe_win_rate": float(leave["test_sharpe_delta"].gt(0.0).mean()),
        },
        "decision": config["decision"],
    }
    (output / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    audit = run_experiment(root, args.config.resolve(), args.output.resolve())
    print(json.dumps(audit["search"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
