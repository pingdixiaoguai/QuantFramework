"""Second-round multi-horizon ensemble search for robust Defender switching."""

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
from research.defender_curve_momentum import DEFENDER_CANDIDATE
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
from research.momentum_defender_log_qm_robust import (
    COMBINED_VOTE,
    RELATIVE_VOTE,
    TREND_VOTE,
    EmergencySpec,
    EnsembleGateSpec,
    RobustEnsembleSpec,
    StatePolicy,
    build_feature_bundle,
    robust_leave_year_metrics,
    run_ensemble_spec,
)
from research.momentum_defender_log_qm_switch import (
    build_fast_switch_data,
    fast_candidate_schedule,
)
from research.momentum_defender_occam import HELD_RETURN, performance
from research.run_momentum_defender_log_qm_robust import (
    _emergency_specs,
    _event_stress,
    _friction,
    _selected_cost_schedule,
    _unique_paths,
)
from research.standard_report import generate_standard_report


DEFAULT_CONFIG = Path(
    "research/configs/momentum_defender_log_qm_ensemble_search.yaml"
)
DEFAULT_OUTPUT = Path(
    "experiments/20260824_momentum_defender_log_qm_multihorizon_ensemble"
)


def _load_config(path: Path) -> dict:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("ensemble config must be a mapping")
    return config


def _policies(config: dict) -> dict[str, StatePolicy]:
    return {
        policy_id: StatePolicy(policy_id=policy_id, **values)
        for policy_id, values in config["ensemble_gate_stage"]["state_policies"].items()
    }


def _ensemble_gates(config: dict) -> list[EnsembleGateSpec]:
    gate = config["ensemble_gate_stage"]
    policies = _policies(config)
    specs = []
    for mode in gate["modes"]:
        thresholds = [0.0] if mode == TREND_VOTE else gate["relative_thresholds"]
        for horizons in gate["horizon_sets"]:
            for fraction in gate["vote_fractions"]:
                for threshold in thresholds:
                    for policy in policies.values():
                        specs.append(
                            EnsembleGateSpec(
                                str(mode),
                                tuple(map(int, horizons)),
                                float(fraction),
                                float(threshold),
                                policy,
                            )
                        )
    return list({spec.candidate_id(): spec for spec in specs}.values())


def _record(spec: RobustEnsembleSpec) -> dict[str, object]:
    return {
        "candidate_id": spec.candidate_id(),
        "ensemble_mode": spec.gate.mode,
        "horizons": "|".join(map(str, spec.gate.horizons)),
        "vote_fraction": spec.gate.vote_fraction,
        "relative_threshold": spec.gate.relative_threshold,
        **{f"policy_{key}": value for key, value in asdict(spec.gate.policy).items()},
        **{f"emergency_{key}": value for key, value in asdict(spec.emergency).items()},
    }


def _evaluate(data, features, specs, negative_window, label):
    records = []
    returns = {}
    for position, spec in enumerate(specs, start=1):
        result = run_ensemble_spec(
            data, features, spec, negative_trend_window=negative_window
        )
        candidate_id = spec.candidate_id()
        returns[candidate_id] = result.returns
        records.append(
            {
                **_record(spec),
                "defender_entries": result.defender_entries,
                "defender_days": result.defender_days,
                "base_switches": result.base_switches,
                "gold_entries": result.gold_entries,
                "gold_days": result.gold_days,
                "formal_switches": result.formal_switches,
            }
        )
        if position % 100 == 0 or position == len(specs):
            print(f"{label}: evaluated {position}/{len(specs)}", flush=True)
    return (
        pd.DataFrame(records).set_index("candidate_id"),
        pd.DataFrame(returns, index=features.calendar),
    )


def _rank(metadata, returns, baseline, years, config):
    full = full_metrics(returns, baseline).add_prefix("full_")
    leave = robust_leave_year_metrics(returns, baseline, years)
    table = metadata.join(full).join(leave)
    selection = config["robust_selection"]
    table["full_three_metric_dominance"] = (
        table["full_delta_annualized_return_252"].ge(0.0)
        & table["full_delta_sharpe"].ge(0.0)
        & table["full_delta_max_drawdown"].ge(0.0)
    )
    table["robust_eligible"] = (
        table["leave_year_annualized_return_252_q25"].ge(
            float(selection["annualized_delta_q25_floor"])
        )
        & table["leave_year_annualized_return_252_median"].ge(
            float(selection["annualized_delta_median_floor"])
        )
        & table["leave_year_sharpe_q25"].ge(
            float(selection["sharpe_delta_q25_floor"])
        )
        & table["leave_year_sharpe_median"].ge(
            float(selection["sharpe_delta_median_floor"])
        )
        & table["full_delta_max_drawdown"].ge(0.0)
    )
    score_columns = [
        "leave_year_annualized_return_252_q25",
        "leave_year_sharpe_q25",
        "full_delta_max_drawdown",
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


def _select(table, count, require_dominance):
    pool = table.loc[table["robust_eligible"]].copy()
    if require_dominance:
        dominating = pool.loc[pool["full_three_metric_dominance"]]
        if not dominating.empty:
            pool = dominating
    if pool.empty:
        pool = table.copy()
    return pool.sort_values(
        [
            "minimum_robust_percentile",
            "mean_robust_percentile",
            "full_sharpe",
            "full_annualized_return_252",
            "formal_switches",
        ],
        ascending=[False, False, False, False, True],
    ).head(count)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_experiment(root: Path, config_path: Path, output: Path) -> dict[str, object]:
    config = _load_config(config_path)
    if QUALITY_METADATA["version"] != config["frozen_layers"][
        "momentum_factor_version"
    ]:
        raise AssertionError("frozen Momentum factor mismatch")
    full_start, full_end = map(pd.Timestamp, config["periods"]["full"])
    years = list(map(int, config["periods"]["calendar_years"]))
    context = build_gold_override_context(root, end=full_end.date())
    gold_metrics = risk_adjusted_momentum_at_open(context.curves, window=5)
    data = build_fast_switch_data(context, gold_metrics)
    baseline_formal = run_gold_raqm_w5(
        context, GoldRAQMW5Params(2.20, 0.60), metrics=gold_metrics
    )
    baseline_returns = baseline_formal.daily["return"].astype(float)
    baseline_codes = baseline_formal.state["target_candidate"].map(
        data.candidate_index
    ).to_numpy(int)
    replay, _, _ = fast_candidate_schedule(data, baseline_codes)
    parity = float(np.max(np.abs(replay - baseline_returns.to_numpy(float))))
    if parity > 1e-12:
        raise AssertionError("ensemble fast path baseline parity failed")

    gate = config["ensemble_gate_stage"]
    emergency = config["emergency_stage"]
    horizons = sorted(
        {int(value) for values in gate["horizon_sets"] for value in values}
        | {1, int(emergency["negative_trend_confirmation_window"])}
    )
    momentum_curve = (
        1.0 + context.integrated.result.inputs.momentum[HELD_RETURN].astype(float)
    ).cumprod()
    features = build_feature_bundle(
        context.calendar,
        context.integrated.result.previous_asset,
        end=full_end.date(),
        return_lookbacks=horizons,
        drawdown_windows=list(map(int, emergency["drawdown_windows"])),
        volatility_windows=list(map(int, emergency["volatility_windows"])),
        quantiles=list(map(float, emergency["common_quantiles"])),
        histories=list(map(str, emergency["quantile_histories"])),
        minimum_history=int(emergency["quantile_min_history"]),
        rolling_history=int(emergency["rolling_history"]),
        momentum_curve=momentum_curve,
        defender_curve=context.curves[DEFENDER_CANDIDATE],
    )
    gates = _ensemble_gates(config)
    gate_specs = [RobustEnsembleSpec(value, EmergencySpec()) for value in gates]
    gate_meta, gate_returns = _evaluate(
        data,
        features,
        gate_specs,
        int(emergency["negative_trend_confirmation_window"]),
        "ensemble-gate",
    )
    gate_table = _rank(gate_meta, gate_returns, baseline_returns, years, config)
    selected_gates = _select(
        gate_table,
        int(gate["top_robust_gates_for_emergency_stage"]),
        False,
    )
    lookup = {value.candidate_id(): value for value in gates}
    joint_specs = [
        RobustEnsembleSpec(
            lookup[gate_id.removesuffix("__em_none")], emergency_spec
        )
        for gate_id in selected_gates.index
        for emergency_spec in _emergency_specs(config)
    ]
    joint_specs = list({value.candidate_id(): value for value in joint_specs}.values())
    joint_meta, joint_returns = _evaluate(
        data,
        features,
        joint_specs,
        int(emergency["negative_trend_confirmation_window"]),
        "ensemble-emergency",
    )
    joint_table = _rank(joint_meta, joint_returns, baseline_returns, years, config)
    selected_row = _select(joint_table, 1, True).iloc[0]
    selected_id = str(selected_row.name)
    selected_spec = next(
        value for value in joint_specs if value.candidate_id() == selected_id
    )
    selected_fast = run_ensemble_spec(
        data,
        features,
        selected_spec,
        negative_trend_window=int(emergency["negative_trend_confirmation_window"]),
    )
    selected_returns = pd.Series(selected_fast.returns, index=context.calendar)
    selected_target = pd.Series(
        [data.candidates[index] for index in selected_fast.target_candidate],
        index=context.calendar,
    )
    baseline_target = baseline_formal.state["target_candidate"].astype(str)

    all_returns = pd.concat([gate_returns, joint_returns], axis=1)
    all_returns = all_returns.loc[:, ~all_returns.columns.duplicated()]
    unique_returns = _unique_paths(all_returns)
    checks = config["overfit_checks"]
    pbo_frame, pbo_summary = cscv_pbo(
        unique_returns, baseline_returns, block_count=int(checks["cscv_blocks"])
    )
    walk = expanding_walk_forward(unique_returns, baseline_returns)
    leave_selection = leave_one_year_selection(unique_returns, baseline_returns)
    bootstrap_frame, bootstrap_summary = paired_block_bootstrap(
        selected_returns,
        baseline_returns,
        block_size=int(checks["paired_block_bootstrap_block"]),
        repetitions=int(checks["paired_block_bootstrap_repetitions"]),
        seed=int(checks["random_seed"]),
    )
    reality = yearly_reality_check(
        unique_returns,
        baseline_returns,
        repetitions=int(checks["yearly_reality_check_repetitions"]),
        seed=int(checks["random_seed"]),
    )
    events, leave_events, deletions, event_summary = _event_stress(
        selected_returns,
        baseline_returns,
        selected_target,
        baseline_target,
        list(map(int, checks["top_positive_event_deletions"])),
    )
    selected_metrics = performance(selected_returns)
    baseline_metrics = performance(baseline_returns)
    robust_eligible = bool(selected_row["robust_eligible"])
    full_dominates = bool(selected_row["full_three_metric_dominance"])
    stressed = deletions.loc[deletions["removed_top_positive_events"].gt(0)]
    deletion_dominates = bool(
        stressed["annualized_return_252"].min()
        >= baseline_metrics["annualized_return_252"]
        and stressed["sharpe"].min() >= baseline_metrics["sharpe"]
        and stressed["max_drawdown"].min() >= baseline_metrics["max_drawdown"]
    )
    promotion_supported = bool(
        robust_eligible
        and full_dominates
        and deletion_dominates
        and bootstrap_summary["annualized_return_delta_ci_lower"] > 0.0
        and bootstrap_summary["sharpe_delta_ci_lower"] > 0.0
        and bootstrap_summary["max_drawdown_delta_ci_lower"] >= 0.0
        and reality["p_value"] < 0.05
        and pbo_summary["pbo"] < 0.40
        and walk["test_return_delta"].gt(0.0).mean() >= 0.60
        and walk["test_sharpe_delta"].gt(0.0).mean() >= 0.60
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    gate_table.to_csv(stage / "ensemble_gate_grid.csv")
    unique_returns.to_parquet(stage / "unique_candidate_returns.parquet")
    selected_gates.to_csv(stage / "ensemble_gates_selected.csv")
    joint_table.to_csv(stage / "ensemble_emergency_grid.csv")
    pd.DataFrame(
        [
            {"strategy": "baseline", **baseline_metrics},
            {"strategy": "ensemble_selected", **selected_metrics},
        ]
    ).to_csv(stage / "strategy_metrics.csv", index=False)
    selected_row.to_frame().T.to_csv(stage / "selected_robust_metrics.csv")
    pd.DataFrame(
        {
            "risk_on": selected_fast.risk_on,
            "target_candidate": selected_target,
            "return": selected_returns,
            "nav": (1.0 + selected_returns).cumprod(),
        },
        index=context.calendar,
    ).to_csv(stage / "selected_daily.csv")
    pbo_frame.to_csv(stage / "cscv_pbo.csv", index=False)
    walk.to_csv(stage / "expanding_walk_forward.csv", index=False)
    leave_selection.to_csv(stage / "leave_one_year_selection.csv", index=False)
    bootstrap_frame.to_csv(stage / "paired_block_bootstrap.csv", index=False)
    events.to_csv(stage / "event_attribution.csv", index=False)
    leave_events.to_csv(stage / "leave_one_event.csv", index=False)
    deletions.to_csv(stage / "top_positive_event_deletion.csv", index=False)
    friction = _friction(
        selected_returns,
        _selected_cost_schedule(context, data, selected_fast.target_candidate),
        list(map(float, checks["friction_cost_multipliers"])),
    )
    friction.to_csv(stage / "friction_stress.csv", index=False)

    selected_config = {
        "strategy_id": "momentum_defender_log_qm_ensemble_selected_v3",
        "status": "promotion_supported" if promotion_supported else "research_rejected",
        "ensemble_gate": {
            **asdict(selected_spec.gate),
            "policy": asdict(selected_spec.gate.policy),
        },
        "emergency": asdict(selected_spec.emergency),
        "full_metrics": selected_metrics,
        "robust_eligible": robust_eligible,
        "full_three_metric_dominance": full_dominates,
        "top_event_deletion_three_metric_dominance": deletion_dominates,
        "production_promotion_supported": promotion_supported,
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
        "fast_baseline_parity_max_abs_error": parity,
        "gate_candidates": len(gates),
        "joint_candidates": len(joint_specs),
        "actual_candidate_ids": int(all_returns.shape[1]),
        "unique_return_paths": int(unique_returns.shape[1]),
        "selected_candidate": selected_id,
        "selected_spec": selected_config,
        "baseline_metrics": baseline_metrics,
        "selected_metrics": selected_metrics,
        "robust_eligible_candidates": int(joint_table["robust_eligible"].sum()),
        "full_dominating_candidates": int(
            joint_table["full_three_metric_dominance"].sum()
        ),
        "robust_and_full_dominating_candidates": int(
            (
                joint_table["robust_eligible"]
                & joint_table["full_three_metric_dominance"]
            ).sum()
        ),
        "cscv_pbo": pbo_summary,
        "paired_block_bootstrap": bootstrap_summary,
        "yearly_reality_check": reality,
        "walk_forward": {
            "years": len(walk),
            "return_win_rate": float(walk["test_return_delta"].gt(0.0).mean()),
            "sharpe_win_rate": float(walk["test_sharpe_delta"].gt(0.0).mean()),
        },
        "event_stress": event_summary,
        "top_event_deletion_three_metric_dominance": deletion_dominates,
        "production_promotion_supported": promotion_supported,
    }
    (stage / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    generate_standard_report(
        selected_returns,
        baseline_returns,
        "Current log-QM formal baseline",
        stage / "ensemble_selected_vs_baseline.html",
        selected_config,
    )
    report = f"""# 双对数Momentum：多窗口集成切换搜索

第二轮用多窗口等权投票替代单窗口硬门控，并加入Momentum相对Defender趋势票。门控
{len(gates)}组、联合紧急退出{len(joint_specs)}组，实际候选ID {all_returns.shape[1]}个、
唯一收益路径{unique_returns.shape[1]}条。

选中候选：`{selected_id}`。

|指标|基线|候选|差值|
|---|---:|---:|---:|
|年化收益|{baseline_metrics['annualized_return_252']:.2%}|{selected_metrics['annualized_return_252']:.2%}|{selected_metrics['annualized_return_252']-baseline_metrics['annualized_return_252']:+.2%}|
|Sharpe|{baseline_metrics['sharpe']:.3f}|{selected_metrics['sharpe']:.3f}|{selected_metrics['sharpe']-baseline_metrics['sharpe']:+.3f}|
|最大回撤|{baseline_metrics['max_drawdown']:.2%}|{selected_metrics['max_drawdown']:.2%}|{selected_metrics['max_drawdown']-baseline_metrics['max_drawdown']:+.2%}|

稳健合格{audit['robust_eligible_candidates']}组，全样本三指标占优
{audit['full_dominating_candidates']}组，交集{audit['robust_and_full_dominating_candidates']}组。
PBO={pbo_summary['pbo']:.1%}，Reality Check p={reality['p_value']:.4f}，Walk-forward
收益/Sharpe胜率{audit['walk_forward']['return_win_rate']:.1%}/
{audit['walk_forward']['sharpe_win_rate']:.1%}，删最大正贡献事件后仍三指标占优：
{deletion_dominates}。

结论：{'全部门槛通过，可提交用户决定是否晋升。' if promotion_supported else '至少一项稳健门槛失败，不替换生产。'}
"""
    (stage / "research_report.md").write_text(report, encoding="utf-8")
    sources = [
        config_path,
        root / "factors/quality_momentum.py",
        root / "research/momentum_defender_log_qm_robust.py",
        root / "research/run_momentum_defender_log_qm_ensemble.py",
        root / "research/run_momentum_defender_log_qm_robust.py",
        root / "research/DEVELOPMENT_VALIDATION.md",
    ]
    manifest = {
        "experiment_id": config["experiment"]["id"],
        "generated_on": date.today().isoformat(),
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
