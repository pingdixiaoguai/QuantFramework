"""Search broader robust switch mechanisms for frozen log-MOM/log-ER Momentum."""

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
from research.momentum_defender_log_qm_robust import (
    ANCHOR_OR_BREADTH,
    BREADTH,
    DOWNSIDE_VOL,
    DRAWDOWN,
    HELD_OR_BREADTH,
    NEGATIVE_RS_VOL,
    NO_EMERGENCY,
    ONE_DAY_LOSS,
    EmergencySpec,
    GateSpec,
    RobustSpec,
    StatePolicy,
    build_feature_bundle,
    robust_leave_year_metrics,
    run_robust_spec,
)
from research.momentum_defender_log_qm_switch import (
    EXPANDING_HISTORY,
    ROLLING_HISTORY,
    build_fast_switch_data,
    fast_candidate_schedule,
)
from research.momentum_defender_occam import performance
from research.momentum_defender_occam import (
    ENTRY_COST,
    EXIT_COST,
    INTERNAL_COST,
)
from research.standard_report import generate_standard_report


DEFAULT_CONFIG = Path(
    "research/configs/momentum_defender_log_qm_robust_mechanism_search.yaml"
)
DEFAULT_OUTPUT = Path(
    "experiments/20260824_momentum_defender_log_qm_robust_mechanisms"
)


def _load_config(path: Path) -> dict:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("search config must be a mapping")
    return config


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _policies(config: dict) -> dict[str, StatePolicy]:
    return {
        policy_id: StatePolicy(policy_id=policy_id, **values)
        for policy_id, values in config["gate_stage"]["state_policies"].items()
    }


def _gate_specs(config: dict) -> list[GateSpec]:
    gate = config["gate_stage"]
    policies = _policies(config)
    specs: list[GateSpec] = []
    for mode in gate["return_gate_modes"]:
        for lookback in gate["log_return_lookbacks"]:
            for threshold in gate["return_thresholds"]:
                for policy in policies.values():
                    specs.append(
                        GateSpec(str(mode), int(lookback), float(threshold), 2, policy)
                    )
    for lookback in gate["log_return_lookbacks"]:
        for breadth in gate["breadth_required"]:
            for policy in policies.values():
                specs.append(
                    GateSpec(BREADTH, int(lookback), 0.0, int(breadth), policy)
                )
    for mode in (ANCHOR_OR_BREADTH, HELD_OR_BREADTH):
        for lookback in gate["log_return_lookbacks"]:
            for threshold in gate["return_thresholds"]:
                for breadth in gate["breadth_required"]:
                    for policy in policies.values():
                        specs.append(
                            GateSpec(
                                mode,
                                int(lookback),
                                float(threshold),
                                int(breadth),
                                policy,
                            )
                        )
    return list({spec.candidate_id(): spec for spec in specs}.values())


def _emergency_specs(config: dict) -> list[EmergencySpec]:
    emergency = config["emergency_stage"]
    specs: list[EmergencySpec] = []
    if emergency["include_none"]:
        specs.append(EmergencySpec())
    specs.extend(
        EmergencySpec(ONE_DAY_LOSS, 1, float(threshold))
        for threshold in emergency["one_day_loss_thresholds"]
    )
    specs.extend(
        EmergencySpec(DRAWDOWN, int(window), float(threshold))
        for window in emergency["drawdown_windows"]
        for threshold in emergency["drawdown_thresholds"]
    )
    for mode in (DOWNSIDE_VOL, NEGATIVE_RS_VOL):
        specs.extend(
            EmergencySpec(
                mode,
                int(window),
                quantile=float(quantile),
                history=str(history),
            )
            for window in emergency["volatility_windows"]
            for quantile in emergency["common_quantiles"]
            for history in emergency["quantile_histories"]
        )
    return specs


def _spec_record(spec: RobustSpec) -> dict[str, object]:
    return {
        "candidate_id": spec.candidate_id(),
        "gate_mode": spec.gate.mode,
        "gate_lookback": spec.gate.lookback,
        "gate_return_threshold": spec.gate.return_threshold,
        "gate_breadth_required": spec.gate.breadth_required,
        **{f"policy_{key}": value for key, value in asdict(spec.gate.policy).items()},
        **{f"emergency_{key}": value for key, value in asdict(spec.emergency).items()},
    }


def _evaluate(
    data,
    features,
    specs: list[RobustSpec],
    *,
    negative_trend_window: int,
    label: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    records = []
    returns: dict[str, np.ndarray] = {}
    for position, spec in enumerate(specs, start=1):
        result = run_robust_spec(
            data,
            features,
            spec,
            negative_trend_window=negative_trend_window,
        )
        candidate_id = spec.candidate_id()
        returns[candidate_id] = result.returns
        records.append(
            {
                **_spec_record(spec),
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


def _add_metrics(
    metadata: pd.DataFrame,
    returns: pd.DataFrame,
    baseline: pd.Series,
    years: list[int],
) -> pd.DataFrame:
    full = full_metrics(returns, baseline).add_prefix("full_")
    leave = robust_leave_year_metrics(returns, baseline, years)
    return metadata.join(full).join(leave)


def _rank_robust(table: pd.DataFrame, config: dict) -> pd.DataFrame:
    selection = config["robust_selection"]
    ranked = table.copy()
    ranked["full_three_metric_dominance"] = (
        ranked["full_delta_annualized_return_252"].ge(0.0)
        & ranked["full_delta_sharpe"].ge(0.0)
        & ranked["full_delta_max_drawdown"].ge(0.0)
    )
    ranked["robust_eligible"] = (
        ranked["defender_entries"].ge(int(selection["minimum_defender_entries"]))
        & ranked["defender_days"].ge(int(selection["minimum_defender_days"]))
        & ranked["leave_year_annualized_return_252_q25"].ge(
            float(selection["annualized_delta_q25_floor"])
        )
        & ranked["leave_year_annualized_return_252_median"].ge(
            float(selection["annualized_delta_median_floor"])
        )
        & ranked["leave_year_sharpe_q25"].ge(
            float(selection["sharpe_delta_q25_floor"])
        )
        & ranked["leave_year_sharpe_median"].ge(
            float(selection["sharpe_delta_median_floor"])
        )
        & ranked["leave_year_max_drawdown_worst"].ge(
            float(selection["mdd_delta_worst_floor"])
        )
    )
    robust_columns = [
        "leave_year_annualized_return_252_q25",
        "leave_year_sharpe_q25",
        "leave_year_max_drawdown_worst",
    ]
    pool = ranked["robust_eligible"]
    if not pool.any():
        pool = pd.Series(True, index=ranked.index)
    percentiles = ranked.loc[pool, robust_columns].rank(pct=True)
    ranked.loc[pool, "minimum_robust_percentile"] = percentiles.min(axis=1)
    ranked.loc[pool, "mean_robust_percentile"] = percentiles.mean(axis=1)
    ranked["minimum_robust_percentile"] = ranked[
        "minimum_robust_percentile"
    ].fillna(-1.0)
    ranked["mean_robust_percentile"] = ranked["mean_robust_percentile"].fillna(-1.0)
    return ranked


def _select(table: pd.DataFrame, count: int, *, require_dominance: bool) -> pd.DataFrame:
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


def _unique_paths(frame: pd.DataFrame) -> pd.DataFrame:
    seen: set[str] = set()
    columns = []
    for column in frame:
        digest = hashlib.sha1(frame[column].to_numpy(float).tobytes()).hexdigest()
        if digest not in seen:
            seen.add(digest)
            columns.append(column)
    return frame[columns]


def _event_stress(
    candidate_returns: pd.Series,
    baseline_returns: pd.Series,
    candidate_target: pd.Series,
    baseline_target: pd.Series,
    deletions: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    changed = candidate_target.ne(baseline_target)
    groups = changed.ne(changed.shift()).cumsum()
    calendar = candidate_returns.index
    rows = []
    masks: dict[int, pd.DatetimeIndex] = {}
    for event, (_, sample) in enumerate(
        candidate_target.loc[changed].groupby(groups.loc[changed]), start=1
    ):
        start = calendar.get_loc(sample.index.min())
        end = min(calendar.get_loc(sample.index.max()) + 1, len(calendar) - 1)
        interval = calendar[start : end + 1]
        candidate_total = float((1.0 + candidate_returns.loc[interval]).prod() - 1.0)
        baseline_total = float((1.0 + baseline_returns.loc[interval]).prod() - 1.0)
        rows.append(
            {
                "event": event,
                "start": interval.min().date().isoformat(),
                "end_including_exit": interval.max().date().isoformat(),
                "observations": len(interval),
                "candidate_return": candidate_total,
                "baseline_return": baseline_total,
                "log_excess": float(np.log1p(candidate_total) - np.log1p(baseline_total)),
            }
        )
        masks[event] = interval
    events = pd.DataFrame(rows)
    leave_rows = []
    for event, interval in masks.items():
        stressed = candidate_returns.copy()
        stressed.loc[interval] = baseline_returns.loc[interval]
        leave_rows.append({"removed_event": event, **performance(stressed)})
    deletion_rows = []
    order = events.sort_values("log_excess", ascending=False)["event"].tolist()
    for count in [0, *deletions]:
        stressed = candidate_returns.copy()
        for event in order[:count]:
            interval = masks[int(event)]
            stressed.loc[interval] = baseline_returns.loc[interval]
        deletion_rows.append(
            {"removed_top_positive_events": count, **performance(stressed)}
        )
    positive = events.loc[events["log_excess"].gt(0.0), "log_excess"]
    summary = {
        "events": int(len(events)),
        "positive": int(events["log_excess"].gt(0.0).sum()),
        "negative": int(events["log_excess"].lt(0.0).sum()),
        "top_two_positive_share": (
            float(positive.nlargest(2).sum() / positive.sum())
            if not positive.empty and positive.sum() > 0.0
            else 0.0
        ),
        "leave_one_min_annualized_return_252": float(
            pd.DataFrame(leave_rows)["annualized_return_252"].min()
        ),
        "leave_one_min_sharpe": float(pd.DataFrame(leave_rows)["sharpe"].min()),
    }
    return events, pd.DataFrame(leave_rows), pd.DataFrame(deletion_rows), summary


def _friction(returns: pd.Series, costs: pd.Series, multipliers: list[float]) -> pd.DataFrame:
    costs = costs.astype(float).clip(0.0, 0.99)
    gross = (1.0 + returns) / (1.0 - costs) - 1.0
    return pd.DataFrame(
        [
            {
                "cost_multiplier": multiplier,
                **performance((1.0 + gross) * (1.0 - multiplier * costs) - 1.0),
            }
            for multiplier in multipliers
        ]
    )


def _selected_cost_schedule(context, data, actual_target: np.ndarray) -> pd.Series:
    candidates = data.candidates
    internal = np.vstack(
        [context.interfaces[candidate][INTERNAL_COST].to_numpy(float) for candidate in candidates]
    )
    entry = np.vstack(
        [context.interfaces[candidate][ENTRY_COST].to_numpy(float) for candidate in candidates]
    )
    exit_ = np.vstack(
        [context.interfaces[candidate][EXIT_COST].to_numpy(float) for candidate in candidates]
    )
    costs = np.empty(len(actual_target), dtype=float)
    current = int(data.initial_candidate)
    for position, target_value in enumerate(actual_target):
        target = int(target_value)
        if target != current:
            costs[position] = 1.0 - (
                1.0 - float(exit_[current, position])
            ) * (1.0 - float(entry[target, position]))
            current = target
        else:
            costs[position] = float(internal[current, position])
    return pd.Series(costs, index=context.calendar, name="cost_rate_at_open")


def run_experiment(root: Path, config_path: Path, output: Path) -> dict[str, object]:
    config = _load_config(config_path)
    frozen = config["frozen_layers"]
    if QUALITY_METADATA["version"] != frozen["momentum_factor_version"]:
        raise AssertionError("frozen Momentum factor version mismatch")
    full_start, full_end = map(pd.Timestamp, config["periods"]["full"])
    years = list(map(int, config["periods"]["calendar_years"]))
    context = build_gold_override_context(root, end=full_end.date())
    gold_metrics = risk_adjusted_momentum_at_open(context.curves, window=5)
    data = build_fast_switch_data(context, gold_metrics)
    baseline_formal = run_gold_raqm_w5(
        context, GoldRAQMW5Params(2.20, 0.60), metrics=gold_metrics
    )
    baseline_returns = baseline_formal.daily["return"].astype(float)
    baseline_target_codes = baseline_formal.state["target_candidate"].map(
        data.candidate_index
    ).to_numpy(int)
    replay, _, _ = fast_candidate_schedule(data, baseline_target_codes)
    baseline_parity = float(
        np.max(np.abs(replay - baseline_returns.to_numpy(float)))
    )
    if baseline_parity > 1e-12:
        raise AssertionError("fast candidate executor fails formal baseline parity")

    gate = config["gate_stage"]
    emergency = config["emergency_stage"]
    feature_lookbacks = sorted(
        set(map(int, gate["log_return_lookbacks"]))
        | {1, int(emergency["negative_trend_confirmation_window"])}
    )
    features = build_feature_bundle(
        context.calendar,
        context.integrated.result.previous_asset,
        end=full_end.date(),
        return_lookbacks=feature_lookbacks,
        drawdown_windows=list(map(int, emergency["drawdown_windows"])),
        volatility_windows=list(map(int, emergency["volatility_windows"])),
        quantiles=list(map(float, emergency["common_quantiles"])),
        histories=list(map(str, emergency["quantile_histories"])),
        minimum_history=int(emergency["quantile_min_history"]),
        rolling_history=int(emergency["rolling_history"]),
    )
    gate_specs = _gate_specs(config)
    gate_robust_specs = [
        RobustSpec(spec, EmergencySpec()) for spec in gate_specs
    ]
    gate_meta, gate_returns = _evaluate(
        data,
        features,
        gate_robust_specs,
        negative_trend_window=int(emergency["negative_trend_confirmation_window"]),
        label="gate-stage",
    )
    gate_table = _rank_robust(
        _add_metrics(gate_meta, gate_returns, baseline_returns, years), config
    )
    selected_gates = _select(
        gate_table,
        int(gate["top_robust_gates_for_emergency_stage"]),
        require_dominance=False,
    )
    chosen_gate_ids = set(selected_gates.index)
    gate_lookup = {spec.candidate_id(): spec for spec in gate_specs}

    joint_specs = [
        RobustSpec(gate_lookup[gate_id.removesuffix("__em_none")], emergency_spec)
        for gate_id in chosen_gate_ids
        for emergency_spec in _emergency_specs(config)
    ]
    joint_specs = list({spec.candidate_id(): spec for spec in joint_specs}.values())
    joint_meta, joint_returns = _evaluate(
        data,
        features,
        joint_specs,
        negative_trend_window=int(emergency["negative_trend_confirmation_window"]),
        label="emergency-stage",
    )
    joint_table = _rank_robust(
        _add_metrics(joint_meta, joint_returns, baseline_returns, years), config
    )
    selected_row = _select(joint_table, 1, require_dominance=True).iloc[0]
    selected_id = str(selected_row.name)
    selected_spec = next(
        spec for spec in joint_specs if spec.candidate_id() == selected_id
    )
    selected_fast = run_robust_spec(
        data,
        features,
        selected_spec,
        negative_trend_window=int(emergency["negative_trend_confirmation_window"]),
    )
    selected_returns = pd.Series(
        selected_fast.returns, index=context.calendar, name="return"
    )
    selected_target = pd.Series(
        [data.candidates[index] for index in selected_fast.target_candidate],
        index=context.calendar,
        name="target_candidate",
    )
    baseline_target = baseline_formal.state["target_candidate"].astype(str)

    all_returns = pd.concat([gate_returns, joint_returns], axis=1)
    all_returns = all_returns.loc[:, ~all_returns.columns.duplicated()]
    unique_returns = _unique_paths(all_returns)
    overfit = config["overfit_checks"]
    pbo_frame, pbo_summary = cscv_pbo(
        unique_returns,
        baseline_returns,
        block_count=int(overfit["cscv_blocks"]),
    )
    walk = expanding_walk_forward(unique_returns, baseline_returns)
    leave_year_selection = leave_one_year_selection(unique_returns, baseline_returns)
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
    events, leave_events, event_deletions, event_summary = _event_stress(
        selected_returns,
        baseline_returns,
        selected_target,
        baseline_target,
        list(map(int, overfit["top_positive_event_deletions"])),
    )
    selected_metrics = performance(selected_returns)
    baseline_metrics = performance(baseline_returns)
    selection = config["robust_selection"]
    full_dominates = bool(selected_row["full_three_metric_dominance"])
    robust_eligible = bool(selected_row["robust_eligible"])
    deletion_three_metric = bool(
        event_deletions.loc[event_deletions["removed_top_positive_events"].gt(0),
            "annualized_return_252"].min()
        >= baseline_metrics["annualized_return_252"]
        and event_deletions.loc[event_deletions["removed_top_positive_events"].gt(0),
            "sharpe"].min()
        >= baseline_metrics["sharpe"]
        and event_deletions.loc[event_deletions["removed_top_positive_events"].gt(0),
            "max_drawdown"].min()
        >= baseline_metrics["max_drawdown"]
    )
    promotion_supported = bool(
        robust_eligible
        and full_dominates
        and deletion_three_metric
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
        raise FileExistsError(f"output already exists: {output}")
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    gate_table.to_csv(stage / "gate_candidate_grid.csv")
    unique_returns.to_parquet(stage / "unique_candidate_returns.parquet")
    selected_gates.to_csv(stage / "gates_selected_for_emergency_stage.csv")
    joint_table.to_csv(stage / "emergency_candidate_grid.csv")
    pd.DataFrame(
        [
            {"strategy": "baseline", **baseline_metrics},
            {"strategy": "robust_selected", **selected_metrics},
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
    leave_year_selection.to_csv(stage / "leave_one_year_selection.csv", index=False)
    bootstrap_frame.to_csv(stage / "paired_block_bootstrap.csv", index=False)
    events.to_csv(stage / "event_attribution.csv", index=False)
    leave_events.to_csv(stage / "leave_one_event.csv", index=False)
    event_deletions.to_csv(stage / "top_positive_event_deletion.csv", index=False)
    friction = _friction(
        selected_returns,
        _selected_cost_schedule(context, data, selected_fast.target_candidate),
        list(map(float, overfit["friction_cost_multipliers"])),
    )
    friction.to_csv(stage / "friction_stress.csv", index=False)

    selected_config = {
        "strategy_id": "momentum_defender_log_qm_robust_selected_v2",
        "status": "promotion_supported" if promotion_supported else "research_rejected",
        "gate": {
            **asdict(selected_spec.gate),
            "policy": asdict(selected_spec.gate.policy),
        },
        "emergency": asdict(selected_spec.emergency),
        "full_metrics": selected_metrics,
        "robust_eligible": robust_eligible,
        "full_three_metric_dominance": full_dominates,
        "top_event_deletion_three_metric_dominance": deletion_three_metric,
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
        "fast_baseline_parity_max_abs_error": baseline_parity,
        "gate_candidates": len(gate_specs),
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
            "years": int(len(walk)),
            "return_win_rate": float(walk["test_return_delta"].gt(0.0).mean()),
            "sharpe_win_rate": float(walk["test_sharpe_delta"].gt(0.0).mean()),
        },
        "event_stress": event_summary,
        "top_event_deletion_three_metric_dominance": deletion_three_metric,
        "production_promotion_supported": promotion_supported,
    }
    (stage / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    generate_standard_report(
        selected_returns,
        baseline_returns,
        "Current log-QM formal baseline",
        stage / "robust_selected_vs_baseline.html",
        selected_config,
    )
    annual_delta = (
        selected_metrics["annualized_return_252"]
        - baseline_metrics["annualized_return_252"]
    )
    sharpe_delta = selected_metrics["sharpe"] - baseline_metrics["sharpe"]
    mdd_delta = selected_metrics["max_drawdown"] - baseline_metrics["max_drawdown"]
    report = f"""# 双对数Momentum：宽机制稳健搜索

本轮不沿用原40日慢门结构，测试跨资产门控、非对称锁、确认期，以及方向敏感紧急退出。
门控阶段{len(gate_specs)}组，紧急退出联合阶段{len(joint_specs)}组；实际候选ID
{all_returns.shape[1]}个、唯一收益路径{unique_returns.shape[1]}条。删除年份和事件只用于压力
测试，所有最终绩效始终使用完整2019-01-18至2026-08-21历史。

## 稳健目标选中候选

`{selected_id}`

|指标|基线|候选|差值|
|---|---:|---:|---:|
|年化收益|{baseline_metrics['annualized_return_252']:.2%}|{selected_metrics['annualized_return_252']:.2%}|{annual_delta:+.2%}|
|Sharpe|{baseline_metrics['sharpe']:.3f}|{selected_metrics['sharpe']:.3f}|{sharpe_delta:+.3f}|
|最大回撤|{baseline_metrics['max_drawdown']:.2%}|{selected_metrics['max_drawdown']:.2%}|{mdd_delta:+.2%}|

稳健合格候选{audit['robust_eligible_candidates']}组；全样本三指标同时占优
{audit['full_dominating_candidates']}组；两者交集{audit['robust_and_full_dominating_candidates']}组。
CSCV-PBO={pbo_summary['pbo']:.1%}，Reality Check p={reality['p_value']:.4f}，Walk-forward
收益/Sharpe胜率分别为{audit['walk_forward']['return_win_rate']:.1%}/
{audit['walk_forward']['sharpe_win_rate']:.1%}。删除最大1/2/3个正贡献事件后是否仍三指标占优：
{deletion_three_metric}。

## 决策

{'全部预注册稳健门槛通过，可提交用户决定是否晋升。' if promotion_supported else '至少一项预注册稳健门槛失败，不替换生产策略。'}
"""
    (stage / "research_report.md").write_text(report, encoding="utf-8")
    sources = [
        config_path,
        root / "factors/quality_momentum.py",
        root / "research/momentum_defender_log_qm_robust.py",
        root / "research/run_momentum_defender_log_qm_robust.py",
        root / "research/DEVELOPMENT_VALIDATION.md",
    ]
    manifest = {
        "experiment_id": config["experiment"]["id"],
        "generated_on": date.today().isoformat(),
        "sources": [
            {"path": str(path.relative_to(root)), "sha256": _sha256(path)}
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
