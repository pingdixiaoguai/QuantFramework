"""Cross-base robust search for simpler Gold RAQM regularization and thresholds."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from research.defender_curve_momentum import DEFENDER_CANDIDATE
from research.gold_raqm_regularization import (
    GoldRuleSpec,
    RAQMSpec,
    metric_at_open,
    run_gold_rule,
)
from research.momentum_defender_gold_override import build_gold_override_context
from research.momentum_defender_gold_override_overfit import (
    cscv_pbo,
    expanding_walk_forward,
    full_metrics,
    paired_block_bootstrap,
    yearly_reality_check,
)
from research.momentum_defender_log_qm_robust import robust_leave_year_metrics
from research.momentum_defender_log_qm_switch import (
    build_fast_switch_data,
    fast_candidate_schedule,
)
from research.momentum_defender_occam import performance
from research.run_momentum_defender_log_qm_robust import (
    _event_stress,
    _friction,
    _selected_cost_schedule,
    _unique_paths,
)
from research.run_momentum_held_asset_c2_overfit import (
    _deflated_sharpe,
    _effective_trials,
)
from research.standard_report import generate_standard_report


DEFAULT_CONFIG = Path(
    "research/configs/gold_raqm_regularization_robust_search.yaml"
)
DEFAULT_OUTPUT = Path(
    "experiments/20260824_gold_raqm_regularization_robust_search"
)
STABILITY_DAILY = Path(
    "experiments/20260824_momentum_defender_log_qm_absolute_stability_candidate/daily.csv"
)


def _load(path: Path) -> dict:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Gold search config must be a mapping")
    return config


def _factor_specs(config: dict) -> list[RAQMSpec]:
    specs = []
    for family, values in config["factor_grid"]["regularization_families"].items():
        for window in config["factor_grid"]["windows"]:
            for floor in values["volatility_floors_annual"]:
                for winsor in values["winsor_limits"]:
                    specs.append(
                        RAQMSpec(
                            str(family),
                            int(window),
                            None if floor is None else float(floor),
                            None if winsor is None else float(winsor),
                            int(values["extra_numeric_parameters"]),
                        )
                    )
    return specs


def _rule_specs(config: dict) -> list[GoldRuleSpec]:
    hold = int(config["fixed"]["hard_min_gold_hold_days"])
    return [
        GoldRuleSpec(factor, float(entry), float(exit_), hold)
        for factor in _factor_specs(config)
        for entry in config["threshold_grid"]["entry_differences"]
        for exit_ in config["threshold_grid"]["exit_differences"]
        if float(exit_) <= float(entry)
    ]


def _baseline_target(data, risk_on):
    defender = data.candidate_index[DEFENDER_CANDIDATE]
    requested = np.where(risk_on, data.momentum_target, defender).astype(int)
    returns, actual, switches = fast_candidate_schedule(data, requested)
    return returns, actual, switches


def _rank(
    metadata: pd.DataFrame,
    primary_returns: pd.DataFrame,
    secondary_returns: pd.DataFrame,
    primary_no_gold: pd.Series,
    secondary_no_gold: pd.Series,
    years: list[int],
    config: dict,
) -> pd.DataFrame:
    table = metadata.join(
        full_metrics(primary_returns, primary_no_gold).add_prefix("formal_")
    ).join(
        full_metrics(secondary_returns, secondary_no_gold).add_prefix("stable_")
    ).join(
        robust_leave_year_metrics(
            primary_returns, primary_no_gold, years
        ).add_prefix("formal_")
    ).join(
        robust_leave_year_metrics(
            secondary_returns, secondary_no_gold, years
        ).add_prefix("stable_")
    )
    selection = config["selection"]
    table["worst_full_annual_delta"] = table[
        ["formal_delta_annualized_return_252", "stable_delta_annualized_return_252"]
    ].min(axis=1)
    table["worst_full_sharpe_delta"] = table[
        ["formal_delta_sharpe", "stable_delta_sharpe"]
    ].min(axis=1)
    table["worst_full_mdd_delta"] = table[
        ["formal_delta_max_drawdown", "stable_delta_max_drawdown"]
    ].min(axis=1)
    table["worst_leave_annual_q25"] = table[
        [
            "formal_leave_year_annualized_return_252_q25",
            "stable_leave_year_annualized_return_252_q25",
        ]
    ].min(axis=1)
    table["worst_leave_annual_median"] = table[
        [
            "formal_leave_year_annualized_return_252_median",
            "stable_leave_year_annualized_return_252_median",
        ]
    ].min(axis=1)
    table["worst_leave_sharpe_q25"] = table[
        ["formal_leave_year_sharpe_q25", "stable_leave_year_sharpe_q25"]
    ].min(axis=1)
    table["worst_leave_sharpe_median"] = table[
        ["formal_leave_year_sharpe_median", "stable_leave_year_sharpe_median"]
    ].min(axis=1)
    table["robust_eligible"] = (
        table["formal_gold_entries"].ge(int(selection["minimum_gold_entries_each_base"]))
        & table["stable_gold_entries"].ge(int(selection["minimum_gold_entries_each_base"]))
        & table["worst_full_annual_delta"].ge(
            float(selection["full_annualized_delta_floor_each_base"])
        )
        & table["worst_full_sharpe_delta"].ge(
            float(selection["full_sharpe_delta_floor_each_base"])
        )
        & table["worst_full_mdd_delta"].ge(
            float(selection["full_mdd_delta_floor_each_base"])
        )
        & table["worst_leave_annual_q25"].ge(
            float(selection["leave_year_annualized_delta_q25_floor"])
        )
        & table["worst_leave_annual_median"].ge(
            float(selection["leave_year_annualized_delta_median_floor"])
        )
        & table["worst_leave_sharpe_q25"].ge(
            float(selection["leave_year_sharpe_delta_q25_floor"])
        )
        & table["worst_leave_sharpe_median"].ge(
            float(selection["leave_year_sharpe_delta_median_floor"])
        )
    )
    score_columns = [
        "worst_leave_annual_q25",
        "worst_leave_sharpe_q25",
        "worst_full_mdd_delta",
    ]
    pool = table["robust_eligible"]
    if not pool.any():
        pool = pd.Series(True, index=table.index)
    ranks = table.loc[pool, score_columns].rank(pct=True)
    table.loc[pool, "minimum_robust_percentile"] = ranks.min(axis=1)
    table.loc[pool, "mean_robust_percentile"] = ranks.mean(axis=1)
    table["minimum_robust_percentile"] = table[
        "minimum_robust_percentile"
    ].fillna(-1.0)
    table["mean_robust_percentile"] = table["mean_robust_percentile"].fillna(-1.0)
    return table


def _select(table: pd.DataFrame) -> pd.Series:
    pool = table.loc[table["robust_eligible"]].copy()
    if pool.empty:
        pool = table.copy()
    minimum_extra = int(pool["extra_numeric_parameters"].min())
    simplest = pool.loc[pool["extra_numeric_parameters"].eq(minimum_extra)].copy()
    simplest["_candidate_sort_key"] = simplest.index.astype(str)
    return simplest.sort_values(
        [
            "minimum_robust_percentile",
            "mean_robust_percentile",
            "worst_full_sharpe_delta",
            "worst_full_annual_delta",
            "formal_switches",
            "_candidate_sort_key",
        ],
        ascending=[False, False, False, False, True, True],
    ).iloc[0]


def _event_audit(
    returns,
    target,
    no_gold_returns,
    no_gold_target,
    deletions,
):
    return _event_stress(
        returns,
        no_gold_returns,
        target,
        no_gold_target,
        deletions,
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_experiment(root: Path, config_path: Path, output: Path) -> dict[str, object]:
    config = _load(config_path)
    full_start, full_end = map(pd.Timestamp, config["periods"]["full"])
    years = list(map(int, config["periods"]["calendar_years"]))
    context = build_gold_override_context(root, end=full_end.date())
    current_metric = metric_at_open(
        context.curves, RAQMSpec("current", 5, 0.08, 3.0, 2)
    )
    data = build_fast_switch_data(context, current_metric)
    formal_risk = context.integrated.result.state["risk_on"].astype(bool).to_numpy()
    stable_frame = pd.read_csv(
        root / STABILITY_DAILY, parse_dates=["date"]
    ).set_index("date").reindex(context.calendar)
    stable_risk = stable_frame["risk_on"].astype(bool).to_numpy()
    formal_no_gold_values, formal_no_gold_target_codes, _ = _baseline_target(
        data, formal_risk
    )
    stable_no_gold_values, stable_no_gold_target_codes, _ = _baseline_target(
        data, stable_risk
    )
    formal_no_gold = pd.Series(formal_no_gold_values, index=context.calendar)
    stable_no_gold = pd.Series(stable_no_gold_values, index=context.calendar)
    formal_no_gold_target = pd.Series(
        [data.candidates[index] for index in formal_no_gold_target_codes],
        index=context.calendar,
    )
    stable_no_gold_target = pd.Series(
        [data.candidates[index] for index in stable_no_gold_target_codes],
        index=context.calendar,
    )

    specs = _rule_specs(config)
    metric_cache = {
        factor.factor_id(): metric_at_open(context.curves, factor)
        for factor in _factor_specs(config)
    }
    records = []
    formal_returns = {}
    stable_returns = {}
    for position, spec in enumerate(specs, start=1):
        difference = metric_cache[spec.factor.factor_id()]["difference"]
        formal = run_gold_rule(data, formal_risk, difference, spec)
        stable = run_gold_rule(data, stable_risk, difference, spec)
        candidate_id = spec.candidate_id()
        formal_returns[candidate_id] = formal.returns
        stable_returns[candidate_id] = stable.returns
        records.append(
            {
                "candidate_id": candidate_id,
                **asdict(spec.factor),
                "entry_difference": spec.entry_difference,
                "exit_difference": spec.exit_difference,
                "hard_min_hold_days": spec.hard_min_hold_days,
                "total_numeric_parameters": 3
                + spec.factor.extra_numeric_parameters,
                "formal_gold_entries": formal.gold_entries,
                "formal_gold_days": formal.gold_days,
                "formal_switches": formal.switches,
                "stable_gold_entries": stable.gold_entries,
                "stable_gold_days": stable.gold_days,
                "stable_switches": stable.switches,
            }
        )
        if position % 500 == 0 or position == len(specs):
            print(f"Gold search: evaluated {position}/{len(specs)}", flush=True)
    metadata = pd.DataFrame(records).set_index("candidate_id", drop=False)
    formal_matrix = pd.DataFrame(formal_returns, index=context.calendar)
    stable_matrix = pd.DataFrame(stable_returns, index=context.calendar)
    table = _rank(
        metadata,
        formal_matrix,
        stable_matrix,
        formal_no_gold,
        stable_no_gold,
        years,
        config,
    )
    selected_row = _select(table)
    selected_id = str(selected_row.name)
    selected_spec = next(spec for spec in specs if spec.candidate_id() == selected_id)
    selected_difference = metric_cache[selected_spec.factor.factor_id()]["difference"]
    selected_formal = run_gold_rule(data, formal_risk, selected_difference, selected_spec)
    selected_stable = run_gold_rule(data, stable_risk, selected_difference, selected_spec)
    formal_selected_returns = pd.Series(
        selected_formal.returns, index=context.calendar
    )
    stable_selected_returns = pd.Series(
        selected_stable.returns, index=context.calendar
    )
    formal_selected_target = pd.Series(
        [data.candidates[index] for index in selected_formal.target_candidate],
        index=context.calendar,
    )
    stable_selected_target = pd.Series(
        [data.candidates[index] for index in selected_stable.target_candidate],
        index=context.calendar,
    )

    formal_unique = _unique_paths(formal_matrix)
    stable_unique = _unique_paths(stable_matrix)
    checks = config["overfit_checks"]
    formal_pbo_frame, formal_pbo = cscv_pbo(
        formal_unique, formal_no_gold, block_count=int(checks["cscv_blocks"])
    )
    stable_pbo_frame, stable_pbo = cscv_pbo(
        stable_unique, stable_no_gold, block_count=int(checks["cscv_blocks"])
    )
    formal_reality = yearly_reality_check(
        formal_unique,
        formal_no_gold,
        repetitions=int(checks["yearly_reality_check_repetitions"]),
        seed=int(checks["random_seed"]),
    )
    stable_reality = yearly_reality_check(
        stable_unique,
        stable_no_gold,
        repetitions=int(checks["yearly_reality_check_repetitions"]),
        seed=int(checks["random_seed"]),
    )
    formal_walk = expanding_walk_forward(formal_unique, formal_no_gold)
    stable_walk = expanding_walk_forward(stable_unique, stable_no_gold)
    formal_bootstrap, formal_bootstrap_summary = paired_block_bootstrap(
        formal_selected_returns,
        formal_no_gold,
        block_size=int(checks["paired_block_bootstrap_block"]),
        repetitions=int(checks["paired_block_bootstrap_repetitions"]),
        seed=int(checks["random_seed"]),
    )
    stable_bootstrap, stable_bootstrap_summary = paired_block_bootstrap(
        stable_selected_returns,
        stable_no_gold,
        block_size=int(checks["paired_block_bootstrap_block"]),
        repetitions=int(checks["paired_block_bootstrap_repetitions"]),
        seed=int(checks["random_seed"]),
    )
    deletions = list(map(int, checks["top_positive_event_deletions"]))
    formal_events, formal_leave, formal_delete, formal_event_summary = _event_audit(
        formal_selected_returns,
        formal_selected_target,
        formal_no_gold,
        formal_no_gold_target,
        deletions,
    )
    stable_events, stable_leave, stable_delete, stable_event_summary = _event_audit(
        stable_selected_returns,
        stable_selected_target,
        stable_no_gold,
        stable_no_gold_target,
        deletions,
    )
    formal_costs = _selected_cost_schedule(
        context, data, selected_formal.target_candidate
    )
    stable_costs = _selected_cost_schedule(
        context, data, selected_stable.target_candidate
    )
    formal_friction = _friction(
        formal_selected_returns,
        formal_costs,
        list(map(float, checks["friction_cost_multipliers"])),
    )
    stable_friction = _friction(
        stable_selected_returns,
        stable_costs,
        list(map(float, checks["friction_cost_multipliers"])),
    )
    formal_effective = _effective_trials(formal_unique.to_numpy(float))
    stable_effective = _effective_trials(stable_unique.to_numpy(float))
    formal_excess_matrix = (
        formal_unique.to_numpy(float) - formal_no_gold.to_numpy(float)[:, None]
    )
    stable_excess_matrix = (
        stable_unique.to_numpy(float) - stable_no_gold.to_numpy(float)[:, None]
    )
    formal_excess_matrix = formal_excess_matrix[
        :, formal_excess_matrix.std(axis=0, ddof=1) > 1e-14
    ]
    stable_excess_matrix = stable_excess_matrix[
        :, stable_excess_matrix.std(axis=0, ddof=1) > 1e-14
    ]
    formal_dsr = _deflated_sharpe(
        formal_selected_returns.to_numpy(float) - formal_no_gold.to_numpy(float),
        formal_excess_matrix,
        _effective_trials(formal_excess_matrix),
    )
    stable_dsr = _deflated_sharpe(
        stable_selected_returns.to_numpy(float) - stable_no_gold.to_numpy(float),
        stable_excess_matrix,
        _effective_trials(stable_excess_matrix),
    )

    current_id = (
        "floor_and_winsor_w5_floor0.08_clip3.0_"
        "en+2.20_ex+0.60_h5"
    )
    if current_id not in table.index:
        raise AssertionError("current frozen Gold parameter row is missing")
    current_row = table.loc[current_id]
    neighborhood_parameters = [
        "family",
        "window",
        "volatility_floor_annual",
        "winsor_limit",
        "entry_difference",
        "exit_difference",
    ]
    normalized = table[neighborhood_parameters].copy()
    normalized["volatility_floor_annual"] = normalized[
        "volatility_floor_annual"
    ].fillna(-999.0)
    normalized["winsor_limit"] = normalized["winsor_limit"].fillna(-999.0)
    selected_values = normalized.loc[selected_id]
    distance = sum(
        (~normalized[column].eq(selected_values[column])).astype(int)
        for column in neighborhood_parameters
    )
    neighborhood = table.loc[distance.le(1)].copy()
    neighborhood["parameter_hamming_distance"] = distance.loc[neighborhood.index]
    family_eligible_counts = {
        str(family): int(sample["robust_eligible"].sum())
        for family, sample in table.groupby("family")
    }

    robust_eligible = bool(selected_row["robust_eligible"])
    production_supported = bool(
        robust_eligible
        and formal_reality["p_value"] < 0.05
        and stable_reality["p_value"] < 0.05
        and formal_bootstrap_summary["annualized_return_delta_ci_lower"] > 0.0
        and stable_bootstrap_summary["annualized_return_delta_ci_lower"] > 0.0
        and formal_bootstrap_summary["sharpe_delta_ci_lower"] > 0.0
        and stable_bootstrap_summary["sharpe_delta_ci_lower"] > 0.0
        and formal_walk["test_return_delta"].gt(0.0).mean() >= 0.60
        and stable_walk["test_return_delta"].gt(0.0).mean() >= 0.60
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    table.to_csv(stage / "candidate_grid.csv")
    neighborhood.to_csv(stage / "selected_parameter_neighborhood.csv")
    formal_unique.to_parquet(stage / "formal_unique_returns.parquet")
    stable_unique.to_parquet(stage / "stable_unique_returns.parquet")
    pd.DataFrame(
        [
            {"base": "formal", "strategy": "no_gold", **performance(formal_no_gold)},
            {"base": "formal", "strategy": "selected_gold", **performance(formal_selected_returns)},
            {"base": "stable", "strategy": "no_gold", **performance(stable_no_gold)},
            {"base": "stable", "strategy": "selected_gold", **performance(stable_selected_returns)},
        ]
    ).to_csv(stage / "strategy_metrics.csv", index=False)
    selected_row.to_frame().T.to_csv(stage / "selected_metrics.csv")
    formal_pbo_frame.to_csv(stage / "formal_cscv_pbo.csv", index=False)
    stable_pbo_frame.to_csv(stage / "stable_cscv_pbo.csv", index=False)
    formal_walk.to_csv(stage / "formal_walk_forward.csv", index=False)
    stable_walk.to_csv(stage / "stable_walk_forward.csv", index=False)
    formal_bootstrap.to_csv(stage / "formal_paired_bootstrap.csv", index=False)
    stable_bootstrap.to_csv(stage / "stable_paired_bootstrap.csv", index=False)
    formal_events.to_csv(stage / "formal_events.csv", index=False)
    stable_events.to_csv(stage / "stable_events.csv", index=False)
    formal_leave.to_csv(stage / "formal_leave_one_event.csv", index=False)
    stable_leave.to_csv(stage / "stable_leave_one_event.csv", index=False)
    formal_delete.to_csv(stage / "formal_top_event_deletion.csv", index=False)
    stable_delete.to_csv(stage / "stable_top_event_deletion.csv", index=False)
    formal_friction.to_csv(stage / "formal_friction.csv", index=False)
    stable_friction.to_csv(stage / "stable_friction.csv", index=False)
    selected_config = {
        "strategy_id": "gold_raqm_regularization_selected_v2",
        "status": "promotion_supported" if production_supported else "research_rejected",
        "factor": asdict(selected_spec.factor),
        "entry_difference": selected_spec.entry_difference,
        "exit_difference": selected_spec.exit_difference,
        "hard_min_hold_days": selected_spec.hard_min_hold_days,
        "total_numeric_parameters": 3
        + selected_spec.factor.extra_numeric_parameters,
        "robust_eligible": robust_eligible,
        "production_promotion_supported": production_supported,
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
        "candidate_ids": len(specs),
        "formal_unique_paths": int(formal_unique.shape[1]),
        "stable_unique_paths": int(stable_unique.shape[1]),
        "selected_candidate": selected_id,
        "selected_config": selected_config,
        "robust_eligible_candidates": int(table["robust_eligible"].sum()),
        "robust_eligible_by_family": family_eligible_counts,
        "simplest_eligible_extra_parameters": (
            int(table.loc[table["robust_eligible"], "extra_numeric_parameters"].min())
            if table["robust_eligible"].any()
            else None
        ),
        "formal_metrics": {
            "no_gold": performance(formal_no_gold),
            "selected": performance(formal_selected_returns),
        },
        "stable_metrics": {
            "no_gold": performance(stable_no_gold),
            "selected": performance(stable_selected_returns),
        },
        "formal_gold_entries": selected_formal.gold_entries,
        "stable_gold_entries": selected_stable.gold_entries,
        "current_frozen_candidate": {
            "candidate_id": current_id,
            "formal_annualized_return_252": float(
                current_row["formal_annualized_return_252"]
            ),
            "formal_sharpe": float(current_row["formal_sharpe"]),
            "formal_max_drawdown": float(current_row["formal_max_drawdown"]),
            "stable_annualized_return_252": float(
                current_row["stable_annualized_return_252"]
            ),
            "stable_sharpe": float(current_row["stable_sharpe"]),
            "stable_max_drawdown": float(current_row["stable_max_drawdown"]),
        },
        "formal_pbo": formal_pbo,
        "stable_pbo": stable_pbo,
        "formal_walk_forward": {
            "years": int(len(formal_walk)),
            "return_win_rate": float(
                formal_walk["test_return_delta"].gt(0.0).mean()
            ),
            "sharpe_win_rate": float(
                formal_walk["test_sharpe_delta"].gt(0.0).mean()
            ),
        },
        "stable_walk_forward": {
            "years": int(len(stable_walk)),
            "return_win_rate": float(
                stable_walk["test_return_delta"].gt(0.0).mean()
            ),
            "sharpe_win_rate": float(
                stable_walk["test_sharpe_delta"].gt(0.0).mean()
            ),
        },
        "formal_reality_check": formal_reality,
        "stable_reality_check": stable_reality,
        "formal_bootstrap": formal_bootstrap_summary,
        "stable_bootstrap": stable_bootstrap_summary,
        "formal_event_stress": formal_event_summary,
        "stable_event_stress": stable_event_summary,
        "formal_effective_trials": formal_effective,
        "stable_effective_trials": stable_effective,
        "formal_deflated_sharpe_excess": formal_dsr,
        "stable_deflated_sharpe_excess": stable_dsr,
        "selected_parameter_neighborhood": {
            "candidates": int(len(neighborhood)),
            "robust_eligible": int(neighborhood["robust_eligible"].sum()),
            "raw_candidates": int(neighborhood["family"].eq("raw").sum()),
        },
        "production_promotion_supported": production_supported,
    }
    (stage / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    generate_standard_report(
        formal_selected_returns,
        formal_no_gold,
        "Formal base without Gold",
        stage / "formal_selected_vs_no_gold.html",
        selected_config,
    )
    generate_standard_report(
        stable_selected_returns,
        stable_no_gold,
        "Stability base without Gold",
        stage / "stable_selected_vs_no_gold.html",
        selected_config,
    )
    report = f"""# Gold RAQM正则化与阈值重寻优

同时在正式C2和绝对稳定候选两种基础状态上测试{len(specs)}个参数ID。优先选择无波动率
地板、无剪裁的raw模型；只有通过所有跨基础状态与留一年门槛的候选才进入最简复杂度比较。

选中：`{selected_id}`，总数值参数{selected_config['total_numeric_parameters']}个。

|基础状态|无Gold年化/Sharpe/MDD|候选年化/Sharpe/MDD|Gold入场|
|---|---|---|---:|
|正式C2|{performance(formal_no_gold)['annualized_return_252']:.2%}/{performance(formal_no_gold)['sharpe']:.3f}/{performance(formal_no_gold)['max_drawdown']:.2%}|{performance(formal_selected_returns)['annualized_return_252']:.2%}/{performance(formal_selected_returns)['sharpe']:.3f}/{performance(formal_selected_returns)['max_drawdown']:.2%}|{selected_formal.gold_entries}|
|稳定候选|{performance(stable_no_gold)['annualized_return_252']:.2%}/{performance(stable_no_gold)['sharpe']:.3f}/{performance(stable_no_gold)['max_drawdown']:.2%}|{performance(stable_selected_returns)['annualized_return_252']:.2%}/{performance(stable_selected_returns)['sharpe']:.3f}/{performance(stable_selected_returns)['max_drawdown']:.2%}|{selected_stable.gold_entries}|

当前冻结2.20/0.60方案在正式基础上的年化/Sharpe/MDD为
{float(current_row['formal_annualized_return_252']):.2%}/
{float(current_row['formal_sharpe']):.3f}/
{float(current_row['formal_max_drawdown']):.2%}；在稳定基础上为
{float(current_row['stable_annualized_return_252']):.2%}/
{float(current_row['stable_sharpe']):.3f}/
{float(current_row['stable_max_drawdown']):.2%}。

正式/稳定基础的Reality Check p分别为{formal_reality['p_value']:.4f}/
{stable_reality['p_value']:.4f}。结论：
Walk-forward收益/Sharpe胜率分别为
{formal_walk['test_return_delta'].gt(0.0).mean():.1%}/
{formal_walk['test_sharpe_delta'].gt(0.0).mean():.1%}（正式基础）和
{stable_walk['test_return_delta'].gt(0.0).mean():.1%}/
{stable_walk['test_sharpe_delta'].gt(0.0).mean():.1%}（稳定基础）。
{'全部稳健门槛通过，可提交用户决定是否晋升。' if production_supported else '至少一项全局稳健门槛失败，不自动替换生产参数。'}
"""
    (stage / "research_report.md").write_text(report, encoding="utf-8")
    sources = [
        config_path,
        root / "research/gold_raqm_regularization.py",
        root / "research/run_gold_raqm_regularization_robust.py",
        root / "research/DEVELOPMENT_VALIDATION.md",
        root / STABILITY_DAILY,
    ]
    manifest = {
        "experiment_id": config["experiment"]["id"],
        "sources": [
            {"path": str(path.relative_to(root)), "sha256": _sha(path)}
            for path in sources
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
    print(json.dumps(run_experiment(root, config_path, output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
