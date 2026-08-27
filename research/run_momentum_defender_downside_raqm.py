"""Run the preregistered downside-RAQM Momentum/Defender search."""

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
from research.defender_curve_momentum import DEFENDER_CANDIDATE
from research.momentum_defender_downside_raqm import (
    DownsideRAQMRun,
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
from research.momentum_volatility import load_ohlc
from research.run_momentum_defender_log_qm_robust import (
    _event_stress,
    _friction,
    _selected_cost_schedule,
    _unique_paths,
)
from research.standard_report import generate_standard_report


DEFAULT_CONFIG = Path("research/configs/momentum_defender_downside_raqm_search.yaml")
DEFAULT_OUTPUT = Path("experiments/20260824_momentum_defender_downside_raqm")


def _load_config(path: Path) -> dict:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("downside-RAQM search config must be a mapping")
    return config


def _profiles(config: dict) -> dict[str, FactorProfile]:
    result = {}
    for profile_id, values in config["factor"]["profiles"].items():
        result[profile_id] = FactorProfile(
            profile_id=str(profile_id),
            horizons=tuple(map(int, values["horizons"])),
            weights=tuple(map(float, values["weights"])),
        )
    return result


def _specs(config: dict, profiles: dict[str, FactorProfile]) -> list[DownsideRAQMSpec]:
    factor = config["factor"]
    search = config["state_search"]
    minimum_gap = float(search["minimum_hysteresis_gap"])
    result: dict[str, DownsideRAQMSpec] = {}
    for values in product(
        profiles.values(),
        factor["percentile_history_modes"],
        search["defender_entry_percentiles"],
        search["defender_exit_percentiles"],
        search["momentum_lock_days"],
        search["defender_lock_days"],
        search["defender_entry_confirmation_days"],
        search["momentum_recovery_confirmation_days"],
    ):
        profile, history, entry, exit_, momentum_hold, defender_hold, entry_c, recovery_c = values
        if float(entry) - float(exit_) + 1e-12 < minimum_gap:
            continue
        spec = DownsideRAQMSpec(
            profile=profile,
            history_mode=str(history),
            entry_percentile=float(entry),
            exit_percentile=float(exit_),
            momentum_lock_days=int(momentum_hold),
            defender_lock_days=int(defender_hold),
            entry_confirmation_days=int(entry_c),
            recovery_confirmation_days=int(recovery_c),
        )
        result[spec.candidate_id()] = spec
    return list(result.values())


def _record(run: DownsideRAQMRun) -> dict[str, object]:
    spec = run.spec
    return {
        "candidate_id": spec.candidate_id(),
        "profile_id": spec.profile.profile_id,
        "horizons": "|".join(map(str, spec.profile.horizons)),
        "weights": "|".join(f"{value:.2f}" for value in spec.profile.weights),
        "history_mode": spec.history_mode,
        "entry_percentile": spec.entry_percentile,
        "exit_percentile": spec.exit_percentile,
        "momentum_lock_days": spec.momentum_lock_days,
        "defender_lock_days": spec.defender_lock_days,
        "entry_confirmation_days": spec.entry_confirmation_days,
        "recovery_confirmation_days": spec.recovery_confirmation_days,
        "defender_entries": run.defender_entries,
        "defender_days": run.defender_days,
        "sleeve_switches": run.sleeve_switches,
        "candidate_switches": run.candidate_switches,
    }


def _evaluate(data, features, specs: list[DownsideRAQMSpec]):
    matrix = np.empty((len(data.calendar), len(specs)), dtype=np.float32)
    records: list[dict[str, object]] = []
    for position, spec in enumerate(specs):
        run = run_downside_raqm_spec(data, features, spec)
        matrix[:, position] = run.returns
        records.append(_record(run))
        completed = position + 1
        if completed % 500 == 0 or completed == len(specs):
            print(f"downside-RAQM: evaluated {completed}/{len(specs)}", flush=True)
    ids = [record["candidate_id"] for record in records]
    return (
        pd.DataFrame(records).set_index("candidate_id"),
        pd.DataFrame(matrix, index=data.calendar, columns=ids),
    )


def _segment_metrics(
    returns: pd.DataFrame,
    baseline: pd.Series,
    config: dict,
) -> pd.DataFrame:
    result = []
    for name in ("development", "validation", "recent", "full"):
        start, end = map(pd.Timestamp, config["periods"][name])
        sample = returns.loc[start:end]
        base = baseline.loc[start:end]
        result.append(full_metrics(sample, base).add_prefix(f"{name}_"))
    return pd.concat(result, axis=1)


def _add_neighborhood_metrics(table: pd.DataFrame, config: dict) -> pd.DataFrame:
    search = config["state_search"]
    dimensions = {
        "entry_percentile": list(map(float, search["defender_entry_percentiles"])),
        "exit_percentile": list(map(float, search["defender_exit_percentiles"])),
        "momentum_lock_days": list(map(int, search["momentum_lock_days"])),
        "defender_lock_days": list(map(int, search["defender_lock_days"])),
    }
    include_profiles = bool(
        config["selection"].get("neighborhood_include_adjacent_profiles", False)
    )
    if include_profiles:
        dimensions = {
            "profile_id": list(config["factor"]["profiles"]),
            **dimensions,
        }
    positions = {
        name: {value: position for position, value in enumerate(values)}
        for name, values in dimensions.items()
    }
    ranked = table.copy()
    for name, lookup in positions.items():
        ranked[f"_{name}_position"] = ranked[name].map(lookup).astype(int)
    metrics: dict[str, dict[str, float | int]] = {}
    group_fields = [
        "history_mode",
        "entry_confirmation_days",
        "recovery_confirmation_days",
    ]
    if not include_profiles:
        group_fields.insert(0, "profile_id")
    coordinate_fields = [f"_{name}_position" for name in dimensions]
    annual_floor = float(config["selection"]["hard_full_annualized_return_floor"])
    for _, group in ranked.groupby(group_fields, sort=False):
        coordinates = group[coordinate_fields].to_numpy(int)
        annualized = group["full_annualized_return_252"].to_numpy(float)
        sharpe = group["full_sharpe"].to_numpy(float)
        for row_position, candidate_id in enumerate(group.index):
            distance = np.abs(coordinates - coordinates[row_position])
            members = np.all(distance <= 1, axis=1)
            metrics[str(candidate_id)] = {
                "neighborhood_count": int(members.sum()),
                "neighborhood_annualized_pass_rate": float(
                    np.mean(annualized[members] >= annual_floor)
                ),
                "neighborhood_annualized_q25": float(
                    np.quantile(annualized[members], 0.25)
                ),
                "neighborhood_annualized_median": float(np.median(annualized[members])),
                "neighborhood_sharpe_q25": float(np.quantile(sharpe[members], 0.25)),
                "neighborhood_sharpe_median": float(np.median(sharpe[members])),
            }
    neighbor = pd.DataFrame.from_dict(metrics, orient="index")
    ranked = ranked.join(neighbor)
    return ranked.drop(columns=coordinate_fields)


def _selected_neighborhood_mask(
    table: pd.DataFrame,
    selected: pd.Series,
    config: dict,
) -> pd.Series:
    search = config["state_search"]
    dimensions: dict[str, list[object]] = {
        "entry_percentile": list(map(float, search["defender_entry_percentiles"])),
        "exit_percentile": list(map(float, search["defender_exit_percentiles"])),
        "momentum_lock_days": list(map(int, search["momentum_lock_days"])),
        "defender_lock_days": list(map(int, search["defender_lock_days"])),
    }
    include_profiles = bool(
        config["selection"].get("neighborhood_include_adjacent_profiles", False)
    )
    if include_profiles:
        dimensions = {"profile_id": list(config["factor"]["profiles"]), **dimensions}
    mask = (
        table["history_mode"].eq(selected["history_mode"])
        & table["entry_confirmation_days"].eq(selected["entry_confirmation_days"])
        & table["recovery_confirmation_days"].eq(
            selected["recovery_confirmation_days"]
        )
    )
    if not include_profiles:
        mask &= table["profile_id"].eq(selected["profile_id"])
    for field, values in dimensions.items():
        lookup = {value: position for position, value in enumerate(values)}
        selected_position = lookup[selected[field]]
        positions = table[field].map(lookup)
        mask &= positions.notna() & positions.sub(selected_position).abs().le(1)
    return mask


def _select(table: pd.DataFrame, config: dict) -> tuple[pd.Series, pd.DataFrame]:
    selection = config["selection"]
    ranked = table.copy()
    ranked["minimum_segment_sharpe"] = ranked[
        ["development_sharpe", "validation_sharpe", "recent_sharpe"]
    ].min(axis=1)
    ranked["minimum_segment_annualized_return"] = ranked[
        [
            "development_annualized_return_252",
            "validation_annualized_return_252",
            "recent_annualized_return_252",
        ]
    ].min(axis=1)
    ranked["hard_eligible"] = (
        ranked["full_annualized_return_252"].ge(
            float(selection["hard_full_annualized_return_floor"])
        )
        & ranked["defender_entries"].ge(int(selection["minimum_defender_entries"]))
        & ranked["defender_days"].ge(int(selection["minimum_defender_days"]))
        & ranked["validation_annualized_return_252"].ge(
            float(selection["validation_annualized_return_floor"])
        )
        & ranked["recent_annualized_return_252"].ge(
            float(selection["recent_annualized_return_floor"])
        )
        & ranked["validation_sharpe"].ge(float(selection["validation_sharpe_floor"]))
        & ranked["recent_sharpe"].ge(float(selection["recent_sharpe_floor"]))
        & ranked["neighborhood_annualized_pass_rate"].ge(
            float(selection["neighborhood_annualized_pass_rate_floor"])
        )
    )
    pool = ranked.loc[ranked["hard_eligible"]].copy()
    if pool.empty:
        pool = ranked.loc[
            ranked["full_annualized_return_252"].ge(
                float(selection["hard_full_annualized_return_floor"])
            )
        ].copy()
    if pool.empty:
        pool = ranked.copy()
    ordered = pool.sort_values(
        [
            "neighborhood_annualized_pass_rate",
            "neighborhood_sharpe_q25",
            "minimum_segment_sharpe",
            "full_sharpe",
            "full_annualized_return_252",
            "candidate_switches",
        ],
        ascending=[False, False, False, False, False, True],
    )
    return ordered.iloc[0], ranked


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _selected_config(
    config: dict,
    selected: pd.Series,
    selected_returns: pd.Series,
    selected_run: DownsideRAQMRun,
) -> dict:
    return {
        "strategy_id": "momentum_defender_downside_raqm_v1",
        "status": "research_candidate_not_production",
        "selected_on": config["experiment"]["created_on"],
        "evidence_status": "retrospective_robust_selection_not_independent_oos",
        "frozen_layers": config["frozen_layers"],
        "factor": {
            "anchor_asset": config["factor"]["anchor_asset"],
            "formula": config["factor"]["formula"],
            "horizons": list(selected_run.spec.profile.horizons),
            "weights": list(selected_run.spec.profile.weights),
            "volatility_floor_annual": config["factor"]["volatility_floor_annual"],
            "winsor_limit": config["factor"]["winsor_limit"],
            "percentile_history": selected_run.spec.history_mode,
            "percentile_min_history": config["factor"]["percentile_min_history"],
            "signal_timing": "previous_close_to_next_open",
        },
        "state_policy": {
            "defender_entry_percentile": selected_run.spec.entry_percentile,
            "defender_exit_percentile": selected_run.spec.exit_percentile,
            "momentum_lock_days": selected_run.spec.momentum_lock_days,
            "defender_lock_days": selected_run.spec.defender_lock_days,
            "defender_entry_confirmation_days": selected_run.spec.entry_confirmation_days,
            "momentum_recovery_confirmation_days": selected_run.spec.recovery_confirmation_days,
            "emergency_override": False,
        },
        "execution": {
            "costs": "inherited_exact_asset_interfaces",
            "untradable_switch": "retain_previous_candidate",
        },
        "checkpoint": {
            "start": selected_returns.index.min().date().isoformat(),
            "end": selected_returns.index.max().date().isoformat(),
            "observations": len(selected_returns),
            **performance(selected_returns),
            "defender_entries": selected_run.defender_entries,
            "defender_days": selected_run.defender_days,
            "sleeve_switches": selected_run.sleeve_switches,
            "candidate_switches": selected_run.candidate_switches,
            "daily_return_sha256_float64_le": hashlib.sha256(
                selected_returns.to_numpy(dtype="<f8").tobytes()
            ).hexdigest(),
        },
        "parameter_stability": {
            "neighborhood_count": int(selected["neighborhood_count"]),
            "annualized_45pct_pass_rate": float(
                selected["neighborhood_annualized_pass_rate"]
            ),
            "annualized_q25": float(selected["neighborhood_annualized_q25"]),
            "annualized_median": float(
                selected["neighborhood_annualized_median"]
            ),
            "sharpe_q25": float(selected["neighborhood_sharpe_q25"]),
            "sharpe_median": float(selected["neighborhood_sharpe_median"]),
        },
        "decision": {
            "full_annualized_return_at_least_45pct": bool(
                float(selected["full_annualized_return_252"]) >= 0.45
            ),
            "automatic_production_promotion": False,
            "requires_explicit_user_promotion": True,
        },
    }


def _write_report(
    output: Path,
    config: dict,
    selected: pd.Series,
    baseline_metrics: dict,
    selected_metrics: dict,
    audit: dict,
) -> None:
    goal_met = selected_metrics["annualized_return_252"] >= 0.45
    report = f"""# 510300 下行 RAQM：Momentum/Defender 重设计与稳健寻参

## 结论

本轮只允许 510300 的 X 日下行风险调整质量动量决定顶层 Momentum/Defender
状态，所有 X 均不小于 20；Momentum 与 Defender 锁定期均限制在 20—30 个交易日。
5 日桥接、Gold 覆盖和紧急破锁全部关闭。

选中候选：`{selected.name}`。

年化 45% 硬门槛：**{'通过' if goal_met else '未通过'}**。

|指标|Log-QM Momentum 基线|下行 RAQM 候选|
|---|---:|---:|
|年化收益|{baseline_metrics['annualized_return_252']:.2%}|{selected_metrics['annualized_return_252']:.2%}|
|年化波动|{baseline_metrics['annualized_volatility']:.2%}|{selected_metrics['annualized_volatility']:.2%}|
|Sharpe|{baseline_metrics['sharpe']:.3f}|{selected_metrics['sharpe']:.3f}|
|最大回撤|{baseline_metrics['max_drawdown']:.2%}|{selected_metrics['max_drawdown']:.2%}|

## 选中机制

- RAQM 窗口：{selected['horizons']}，权重：{selected['weights']}；
- 历史标准化：{selected['history_mode']}；
- Defender 入场/退出分位：{selected['entry_percentile']:.2f}/{selected['exit_percentile']:.2f}；
- Momentum/Defender 锁定：{int(selected['momentum_lock_days'])}/{int(selected['defender_lock_days'])} 日；
- 入场/恢复确认：{int(selected['entry_confirmation_days'])}/{int(selected['recovery_confirmation_days'])} 日；
- Defender 入场 {int(selected['defender_entries'])} 次，占用 {int(selected['defender_days'])} 日。

## 参数平台

- 邻域候选数：{int(selected['neighborhood_count'])}；
- 邻域年化达到 45% 的比例：{selected['neighborhood_annualized_pass_rate']:.1%}；
- 邻域年化中位数/Q25：{selected['neighborhood_annualized_median']:.2%}/{selected['neighborhood_annualized_q25']:.2%}；
- 邻域 Sharpe 中位数/Q25：{selected['neighborhood_sharpe_median']:.3f}/{selected['neighborhood_sharpe_q25']:.3f}。

## 过拟合与压力测试

- 实际候选 ID：{audit['candidate_ids']}，唯一收益路径：{audit['unique_return_paths']}；
- CSCV-PBO：{audit['cscv']['pbo']:.1%}；
- 年度 Reality Check p 值：{audit['reality_check']['p_value']:.4f}；
- Walk-forward 收益/Sharpe 胜率：{audit['walk_forward_return_win_rate']:.1%}/{audit['walk_forward_sharpe_win_rate']:.1%}；
- 20 日配对分块 Bootstrap 的 Sharpe 差 95% 区间：[{audit['bootstrap']['sharpe_delta_ci_lower']:.3f}, {audit['bootstrap']['sharpe_delta_ci_upper']:.3f}]；
- 删除任一差异事件后的最低年化/Sharpe：{audit['events']['leave_one_min_annualized_return_252']:.2%}/{audit['events']['leave_one_min_sharpe']:.3f}；
- 3 倍费用年化/Sharpe：{audit['three_x_cost']['annualized_return_252']:.2%}/{audit['three_x_cost']['sharpe']:.3f}。

## 证据边界

搜索空间在运行前冻结，但本项目历史已经被反复研究，因此 development、validation 和 recent
只能作为固定分段压力测试，不能宣称真正独立样本外。该结果保留为研究候选，不自动替换生产策略。
"""
    (output / "research_report.md").write_text(report, encoding="utf-8")


def run_experiment(root: Path, config_path: Path, output: Path) -> dict[str, object]:
    config = _load_config(config_path)
    if QUALITY_METADATA["version"] != config["frozen_layers"][
        "momentum_factor_version"
    ]:
        raise AssertionError("frozen log-quality Momentum factor mismatch")
    profiles = _profiles(config)
    specs = _specs(config, profiles)
    if not specs:
        raise AssertionError("downside-RAQM grid is empty")
    if any(horizon < 20 for profile in profiles.values() for horizon in profile.horizons):
        raise AssertionError("search contains a forbidden horizon below 20")
    if any(
        not 20 <= hold <= 30
        for spec in specs
        for hold in (spec.momentum_lock_days, spec.defender_lock_days)
    ):
        raise AssertionError("search contains a sleeve lock outside [20, 30]")

    full_start, full_end = map(pd.Timestamp, config["periods"]["full"])
    context = build_gold_override_context(root, end=full_end.date())
    data = build_exact_execution_data(context)
    anchor = load_ohlc(
        str(config["factor"]["anchor_asset"]), full_end.date()
    )["close"]
    factor = config["factor"]
    features = build_downside_raqm_features(
        anchor,
        data.calendar,
        profiles,
        {
            str(mode): (None if window is None else int(window))
            for mode, window in factor["percentile_history_modes"].items()
        },
        min_history=int(factor["percentile_min_history"]),
        volatility_floor_annual=float(factor["volatility_floor_annual"]),
        winsor_limit=float(factor["winsor_limit"]),
    )

    # Exact executor parity against the frozen integrated C2 target schedule.
    frozen_codes = context.baseline_target.map(data.candidate_index).to_numpy(int)
    replay, _, _ = exact_candidate_schedule(data, frozen_codes)
    parity = float(
        np.max(
            np.abs(
                replay
                - context.integrated.result.simulated["return"].to_numpy(float)
            )
        )
    )
    if parity > 5e-8:
        raise AssertionError(f"exact executor baseline parity failed: {parity:.3e}")

    metadata, returns = _evaluate(data, features, specs)
    momentum_values, momentum_actual, momentum_switches = exact_candidate_schedule(
        data, data.momentum_target
    )
    momentum_returns = pd.Series(
        momentum_values, index=data.calendar, name="log_qm_momentum"
    )
    metrics = _segment_metrics(returns, momentum_returns, config)
    table = metadata.join(metrics)
    table = _add_neighborhood_metrics(table, config)
    selected, table = _select(table, config)
    selected_id = str(selected.name)
    selected_spec = next(spec for spec in specs if spec.candidate_id() == selected_id)
    selected_run = run_downside_raqm_spec(data, features, selected_spec)
    selected_returns = pd.Series(
        selected_run.returns, index=data.calendar, name=selected_id
    )
    selected_target = pd.Series(
        [data.candidates[value] for value in selected_run.actual_target],
        index=data.calendar,
        name="actual_candidate",
    )
    momentum_target = pd.Series(
        [data.candidates[value] for value in momentum_actual],
        index=data.calendar,
        name="momentum_candidate",
    )

    unique_returns = _unique_paths(returns)
    checks = config["overfit_checks"]
    pbo_frame, pbo_summary = cscv_pbo(
        unique_returns,
        momentum_returns,
        block_count=int(checks["cscv_blocks"]),
    )
    walk = expanding_walk_forward(unique_returns, momentum_returns)
    leave_year = leave_one_year_selection(unique_returns, momentum_returns)
    bootstrap_frame, bootstrap_summary = paired_block_bootstrap(
        selected_returns,
        momentum_returns,
        block_size=int(checks["paired_block_bootstrap_block"]),
        repetitions=int(checks["paired_block_bootstrap_repetitions"]),
        seed=int(checks["random_seed"]),
    )
    reality = yearly_reality_check(
        unique_returns,
        momentum_returns,
        repetitions=int(checks["yearly_reality_check_repetitions"]),
        seed=int(checks["random_seed"]),
    )
    events, leave_event, top_deletion, event_summary = _event_stress(
        selected_returns,
        momentum_returns,
        selected_target,
        momentum_target,
        list(map(int, checks["top_positive_event_deletions"])),
    )
    costs = _selected_cost_schedule(context, data, selected_run.actual_target)
    friction = _friction(
        selected_returns,
        costs,
        list(map(float, checks["friction_cost_multipliers"])),
    )

    output.mkdir(parents=True, exist_ok=True)
    search_path = output / "search_grid.csv"
    table.sort_values(
        ["hard_eligible", "neighborhood_sharpe_q25", "full_sharpe"],
        ascending=[False, False, False],
    ).to_csv(search_path)
    frontier = pareto_frontier(
        table,
        ["full_annualized_return_252", "full_sharpe", "full_max_drawdown"],
    )
    table.loc[frontier].to_csv(output / "pareto_frontier.csv")
    unique_returns.to_parquet(output / "unique_candidate_returns.parquet")
    pbo_frame.to_csv(output / "cscv_pbo.csv", index=False)
    walk.to_csv(output / "expanding_walk_forward.csv", index=False)
    leave_year.to_csv(output / "leave_one_year_selection.csv", index=False)
    bootstrap_frame.to_csv(output / "paired_block_bootstrap.csv", index=False)
    events.to_csv(output / "event_attribution.csv", index=False)
    leave_event.to_csv(output / "leave_one_event.csv", index=False)
    top_deletion.to_csv(output / "top_positive_event_deletion.csv", index=False)
    friction.to_csv(output / "friction_stress.csv", index=False)

    selected_daily = selected_run.state.copy()
    selected_daily["return"] = selected_returns
    selected_daily["nav"] = (1.0 + selected_returns).cumprod()
    selected_daily["requested_candidate"] = [
        data.candidates[value] for value in selected_run.requested_target
    ]
    selected_daily["actual_candidate"] = selected_target
    selected_daily["cost_rate_at_open"] = costs
    for horizon, values in features.raw_at_open.items():
        selected_daily[f"downside_raqm_{horizon}_at_open"] = values
    selected_daily.to_csv(output / "selected_daily.csv")

    neighbor_mask = _selected_neighborhood_mask(table, selected, config)
    neighborhood = table.loc[neighbor_mask].sort_values("full_sharpe", ascending=False)
    neighborhood.to_csv(output / "selected_parameter_neighborhood.csv")

    baseline_metrics = performance(momentum_returns)
    selected_metrics = performance(selected_returns)
    strategy_metrics = pd.DataFrame(
        [
            {"strategy": "log_qm_momentum", **baseline_metrics},
            {"strategy": selected_id, **selected_metrics},
        ]
    )
    strategy_metrics.to_csv(output / "strategy_metrics.csv", index=False)

    three_x = friction.loc[friction["cost_multiplier"].eq(3.0)].iloc[0].to_dict()
    audit = {
        "experiment_id": config["experiment"]["id"],
        "candidate_ids": len(specs),
        "unique_return_paths": int(unique_returns.shape[1]),
        "selected_candidate": selected_id,
        "annualized_45pct_gate_passed": bool(
            selected_metrics["annualized_return_252"] >= 0.45
        ),
        "all_factor_horizons_at_least_20": True,
        "all_sleeve_locks_between_20_and_30": True,
        "gold_override_disabled": True,
        "emergency_override_disabled": True,
        "signal_timing": "strictly_previous_close_to_open",
        "exact_executor_parity_max_abs_error": parity,
        "momentum_baseline_candidate_switches": momentum_switches,
        "cscv": pbo_summary,
        "bootstrap": bootstrap_summary,
        "reality_check": reality,
        "walk_forward_return_win_rate": float(walk["test_return_delta"].gt(0.0).mean()),
        "walk_forward_sharpe_win_rate": float(walk["test_sharpe_delta"].gt(0.0).mean()),
        "leave_one_year_return_win_rate": float(leave_year["test_return_delta"].gt(0.0).mean()),
        "leave_one_year_sharpe_win_rate": float(leave_year["test_sharpe_delta"].gt(0.0).mean()),
        "events": event_summary,
        "three_x_cost": three_x,
    }
    (output / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    selected_config = _selected_config(
        config, selected, selected_returns, selected_run
    )
    (output / "selected_research_config.yaml").write_text(
        yaml.safe_dump(selected_config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    (output / "search_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    _write_report(
        output, config, selected, baseline_metrics, selected_metrics, audit
    )
    generate_standard_report(
        selected_returns,
        momentum_returns,
        "Log-QM Momentum",
        output / "selected_vs_momentum.html",
        selected_config,
    )

    source_paths = [
        config_path,
        root / "research/momentum_defender_downside_raqm.py",
        root / "research/run_momentum_defender_downside_raqm.py",
        root / "factors/quality_momentum.py",
        root / "data/db/510300.SH.parquet",
    ]
    manifest = {
        "experiment_id": config["experiment"]["id"],
        "selected_candidate": selected_id,
        "sources": {
            str(path.relative_to(root)): _sha256(path) for path in source_paths
        },
        "artifacts": {
            path.name: _sha256(path)
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
    config_path = args.config if args.config.is_absolute() else root / args.config
    output = args.output if args.output.is_absolute() else root / args.output
    if args.check:
        with tempfile.TemporaryDirectory() as directory:
            audit = run_experiment(root, config_path, Path(directory))
    else:
        audit = run_experiment(root, config_path, output)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
