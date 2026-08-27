"""Preregistered dual-objective search for the single W40 loss gate."""

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
from research.momentum_defender_downside_raqm import (
    build_exact_execution_data,
    exact_candidate_schedule,
)
from research.momentum_defender_gold_override import build_gold_override_context
from research.momentum_defender_gold_override_overfit import (
    cscv_pbo,
    expanding_walk_forward,
    leave_one_year_selection,
    paired_block_bootstrap,
    yearly_reality_check,
)
from research.momentum_defender_occam import performance
from research.momentum_defender_w40_loss_gate import (
    W40LossGateSpec,
    run_w40_loss_gate,
    w40_loss_percentile_at_open,
)
from research.momentum_volatility import load_ohlc
from research.run_momentum_defender_common_score_trimmed import _add_metrics
from research.run_momentum_defender_log_qm_robust import (
    _event_stress,
    _friction,
    _selected_cost_schedule,
    _unique_paths,
)
from research.standard_report import generate_standard_report


DEFAULT_CONFIG = Path(
    "research/configs/momentum_defender_w40_loss_occam_search.yaml"
)
DEFAULT_OUTPUT = Path(
    "experiments/20260825_momentum_defender_w40_loss_occam_search"
)
PRIOR_EXPERIMENTS = (
    "20260824_momentum_defender_downside_raqm",
    "20260824_momentum_defender_downside_raqm_focused_stability",
    "20260824_momentum_defender_downside_raqm_weighted_profiles",
)


def _load(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("W40 loss search config must be a mapping")
    return value


def _specs(config: dict) -> dict[str, W40LossGateSpec]:
    grid = config["state_grid"]
    gap = float(grid["minimum_hysteresis_gap"])
    result = {}
    for entry, recovery, entry_c, recovery_c, momentum_lock, defender_lock in product(
        grid["entry_percentiles"],
        grid["recovery_percentiles"],
        grid["entry_confirmation_days"],
        grid["recovery_confirmation_days"],
        grid["momentum_lock_days"],
        grid["defender_lock_days"],
    ):
        if float(entry) - float(recovery) + 1e-12 < gap:
            continue
        spec = W40LossGateSpec(
            float(entry),
            float(recovery),
            int(entry_c),
            int(recovery_c),
            int(momentum_lock),
            int(defender_lock),
        )
        result[spec.candidate_id()] = spec
    return result


def _add_search_metrics(
    metadata: pd.DataFrame,
    returns: pd.DataFrame,
    baseline: pd.Series,
    ordinary: pd.Series,
    config: dict,
) -> pd.DataFrame:
    result = _add_metrics(metadata, returns, baseline, ordinary, config)
    result = result.rename(
        columns={column: column.replace("trimmed_", "ordinary_", 1)
                 for column in result.columns if column.startswith("trimmed_")}
    )
    result["full_minimum_segment_sharpe"] = result[
        ["development_sharpe", "validation_sharpe", "recent_sharpe"]
    ].min(axis=1)
    return result


def _add_neighborhood(table: pd.DataFrame, config: dict) -> pd.DataFrame:
    result = table.copy()
    grid = config["state_grid"]
    dimensions = {
        "entry_percentile": list(map(float, grid["entry_percentiles"])),
        "recovery_percentile": list(map(float, grid["recovery_percentiles"])),
        "entry_confirmation_days": list(map(int, grid["entry_confirmation_days"])),
        "recovery_confirmation_days": list(
            map(int, grid["recovery_confirmation_days"])
        ),
        "momentum_lock_days": list(map(int, grid["momentum_lock_days"])),
        "defender_lock_days": list(map(int, grid["defender_lock_days"])),
    }
    coordinates = []
    for field, values in dimensions.items():
        column = f"_{field}"
        result[column] = result[field].map(
            {value: position for position, value in enumerate(values)}
        )
        coordinates.append(column)
    coords = result[coordinates].to_numpy(int)
    arrays = {
        "full_annualized": result["full_annualized_return_252"].to_numpy(float),
        "full_sharpe": result["full_sharpe"].to_numpy(float),
        "full_mdd": result["full_max_drawdown"].to_numpy(float),
        "ordinary_annualized": result[
            "ordinary_annualized_return_252"
        ].to_numpy(float),
        "ordinary_sharpe": result["ordinary_sharpe"].to_numpy(float),
        "ordinary_mdd": result["ordinary_max_drawdown"].to_numpy(float),
    }
    rows = {}
    for position, candidate_id in enumerate(result.index):
        members = np.all(np.abs(coords - coords[position]) <= 1, axis=1)
        rows[str(candidate_id)] = {
            "neighborhood_count": int(members.sum()),
            **{
                f"neighborhood_{label}_q25": float(np.quantile(values[members], 0.25))
                for label, values in arrays.items()
            },
            **{
                f"neighborhood_{label}_median": float(np.median(values[members]))
                for label, values in arrays.items()
            },
        }
    return result.drop(columns=coordinates).join(
        pd.DataFrame.from_dict(rows, orient="index")
    )


def _complexity(row: pd.Series, config: dict) -> float:
    values = config["occam_tiebreak"]
    confirmation = (
        int(row["entry_confirmation_days"])
        + int(row["recovery_confirmation_days"])
        - 2
    )
    shortfall = (
        (30 - int(row["momentum_lock_days"]))
        + (30 - int(row["defender_lock_days"]))
    ) / 5
    asymmetry = abs(
        int(row["momentum_lock_days"]) - int(row["defender_lock_days"])
    ) / 5
    return float(
        values["extra_confirmation_day_unit"] * confirmation
        + values["five_day_lock_shortfall_unit"] * shortfall
        + values["asymmetric_lock_unit"] * asymmetry
    )


def _select(table: pd.DataFrame, config: dict, key: str, prefix: str):
    eligibility = config["eligibility"]
    eligible = (
        table["defender_entries"].ge(int(eligibility["minimum_defender_entries"]))
        & table["defender_days"].ge(int(eligibility["minimum_defender_days"]))
        & table["full_minimum_segment_sharpe"].gt(
            float(eligibility["minimum_full_segment_sharpe"])
        )
        & table["ordinary_minimum_segment_sharpe"].gt(
            float(eligibility["minimum_ordinary_segment_sharpe"])
        )
    )
    result = table.copy()
    pool = result.loc[eligible].copy()
    fields = list(config[key]["ranking_fields"])
    ranks = pool[fields].rank(pct=True)
    pool[f"{prefix}_rank_min"] = ranks.min(axis=1)
    pool[f"{prefix}_rank_mean"] = ranks.mean(axis=1)
    best_min = float(pool[f"{prefix}_rank_min"].max())
    stable = pool.loc[pool[f"{prefix}_rank_min"].ge(best_min - 0.03)].copy()
    best_mean = float(stable[f"{prefix}_rank_mean"].max())
    stable = stable.loc[stable[f"{prefix}_rank_mean"].ge(best_mean - 0.03)]
    annual_field, sharpe_field = fields[:2]
    max_annual = float(stable[annual_field].max())
    max_sharpe = float(stable[sharpe_field].max())
    near = stable.loc[
        stable[annual_field].ge(
            max_annual - float(config[key]["occam_annualized_tolerance"])
        )
        & stable[sharpe_field].ge(
            max_sharpe - float(config[key]["occam_sharpe_tolerance"])
        )
    ].copy()
    near["occam_complexity"] = near.apply(_complexity, axis=1, config=config)
    selected = near.sort_values(
        ["occam_complexity", annual_field, sharpe_field, f"{prefix}_rank_mean"],
        ascending=[True, False, False, False],
    ).iloc[0]
    result[f"eligible_{prefix}"] = eligible
    result.loc[pool.index, f"{prefix}_rank_min"] = pool[f"{prefix}_rank_min"]
    result.loc[pool.index, f"{prefix}_rank_mean"] = pool[f"{prefix}_rank_mean"]
    result[f"{prefix}_occam_near"] = result.index.isin(near.index)
    return selected, result


def _pareto(table: pd.DataFrame, prefix: str) -> pd.DataFrame:
    fields = [
        f"{prefix}_annualized_return_252",
        f"{prefix}_sharpe",
        f"{prefix}_max_drawdown",
    ]
    values = table[fields].to_numpy(float)
    keep = np.ones(len(table), dtype=bool)
    for position in range(len(table)):
        dominated = np.all(values >= values[position], axis=1) & np.any(
            values > values[position], axis=1
        )
        if dominated.any():
            keep[position] = False
    return table.loc[keep].sort_values(fields[:2], ascending=False)


def _neighborhood_detail(table: pd.DataFrame, selected: pd.Series, config: dict):
    grid = config["state_grid"]
    values_by_field = {
        "entry_percentile": list(map(float, grid["entry_percentiles"])),
        "recovery_percentile": list(map(float, grid["recovery_percentiles"])),
        "entry_confirmation_days": list(map(int, grid["entry_confirmation_days"])),
        "recovery_confirmation_days": list(
            map(int, grid["recovery_confirmation_days"])
        ),
        "momentum_lock_days": list(map(int, grid["momentum_lock_days"])),
        "defender_lock_days": list(map(int, grid["defender_lock_days"])),
    }
    member = np.ones(len(table), dtype=bool)
    for field, values in values_by_field.items():
        mapping = {value: position for position, value in enumerate(values)}
        member &= table[field].map(mapping).sub(mapping[selected[field]]).abs().le(1)
    neighborhood = table.loc[member].copy()
    summary = {"count": int(len(neighborhood))}
    for scope in ("full", "ordinary"):
        for metric in ("annualized_return_252", "sharpe", "max_drawdown"):
            field = f"{scope}_{metric}"
            values = neighborhood[field]
            summary[f"{scope}_{metric}_q25"] = float(values.quantile(0.25))
            summary[f"{scope}_{metric}_median"] = float(values.median())
            summary[f"{scope}_{metric}_improve_rate"] = float(
                values.gt(float(selected[field])).mean()
            )
    return neighborhood, summary


def _selected_detail(
    label: str,
    selected: pd.Series,
    spec: W40LossGateSpec,
    data,
    score,
    raw_loss,
    context,
    weighted_daily,
    ordinary,
    output: Path,
    config: dict,
) -> dict:
    run = run_w40_loss_gate(data, score, spec)
    returns = pd.Series(run.returns, index=data.calendar, name=spec.candidate_id())
    actual = pd.Series(
        [data.candidates[value] for value in run.actual_target], index=data.calendar
    )
    daily = run.state.copy()
    daily["w40_downside_log_loss_at_open"] = raw_loss
    daily["w40_loss_percentile_at_open"] = score
    daily["return"] = returns
    daily["nav"] = (1.0 + returns).cumprod()
    daily["requested_candidate"] = [
        data.candidates[value] for value in run.requested_target
    ]
    daily["actual_candidate"] = actual
    daily.to_csv(output / f"selected_{label}_daily.csv")
    daily.to_parquet(output / f"selected_{label}_daily.parquet")
    baseline_returns = weighted_daily["return"].astype(float)
    baseline_target = weighted_daily["candidate"].astype(str)
    events, leave_events, deleted, event_summary = _event_stress(
        returns,
        baseline_returns,
        actual,
        baseline_target,
        list(map(int, config["overfit_checks"]["top_positive_event_deletions"])),
    )
    events.to_csv(output / f"selected_{label}_events_vs_formal.csv", index=False)
    leave_events.to_csv(
        output / f"selected_{label}_leave_event_vs_formal.csv", index=False
    )
    deleted.to_csv(
        output / f"selected_{label}_delete_top_events_vs_formal.csv", index=False
    )
    bootstrap, bootstrap_summary = paired_block_bootstrap(
        returns,
        baseline_returns,
        block_size=int(config["overfit_checks"]["paired_block_bootstrap_block"]),
        repetitions=int(
            config["overfit_checks"]["paired_block_bootstrap_repetitions"]
        ),
        seed=int(config["overfit_checks"]["random_seed"]),
    )
    bootstrap.to_csv(output / f"selected_{label}_bootstrap.csv", index=False)
    costs = _selected_cost_schedule(context, data, run.actual_target)
    friction = _friction(
        returns,
        costs,
        list(map(float, config["overfit_checks"]["friction_cost_multipliers"])),
    )
    friction.to_csv(output / f"selected_{label}_friction.csv", index=False)
    generate_standard_report(
        returns,
        baseline_returns,
        "Current Weighted DRAQM Formal",
        output / f"selected_{label}_vs_formal.html",
        {"experiment": config["experiment"], "selected": spec.__dict__},
    )
    selected_config = {
        "strategy_id": f"momentum_defender_w40_loss_{label}_v1",
        "status": "research_candidate_not_production",
        "selected_on": config["experiment"]["created_on"],
        "evidence_status": config["experiment"]["evidence_status"],
        "factor": config["factor"],
        "state_policy": spec.__dict__,
        "checkpoint": {
            **performance(returns),
            "ordinary": performance(returns.loc[ordinary]),
            "weighted_formal": performance(baseline_returns),
            "defender_entries": run.defender_entries,
            "defender_days": run.defender_days,
            "sleeve_switches": run.sleeve_switches,
            "candidate_switches": run.candidate_switches,
            "daily_return_sha256_float64_le": hashlib.sha256(
                returns.to_numpy(dtype="<f8").tobytes()
            ).hexdigest(),
        },
        "bootstrap_vs_weighted_formal": bootstrap_summary,
        "event_stress_vs_weighted_formal": event_summary,
        "three_x_cost": friction.loc[
            friction["cost_multiplier"].eq(3.0)
        ].iloc[0].to_dict(),
        "decision": config["decision"],
    }
    (output / f"selected_{label}_config.yaml").write_text(
        yaml.safe_dump(selected_config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return {
        "candidate_id": spec.candidate_id(),
        "config": selected_config,
        "bootstrap": bootstrap_summary,
        "events": event_summary,
    }


def _global_prior_paths(root: Path, current: pd.DataFrame):
    seen: set[str] = set()
    frames = []
    counts = []
    for name in PRIOR_EXPERIMENTS:
        frame = pd.read_parquet(
            root / "experiments" / name / "unique_candidate_returns.parquet"
        )
        keep = []
        for column in frame:
            digest = hashlib.sha1(frame[column].to_numpy(float).tobytes()).hexdigest()
            if digest not in seen:
                seen.add(digest)
                keep.append(str(column))
        selected = frame.loc[:, keep].copy()
        selected.columns = [f"{name}::{column}" for column in keep]
        frames.append(selected)
        counts.append(
            {
                "experiment": name,
                "input_unique_paths": int(frame.shape[1]),
                "new_global_unique_paths": len(keep),
            }
        )
    keep = []
    for column in current:
        digest = hashlib.sha1(current[column].to_numpy(float).tobytes()).hexdigest()
        if digest not in seen:
            seen.add(digest)
            keep.append(str(column))
    selected = current.loc[:, keep].copy()
    selected.columns = [f"current::{column}" for column in keep]
    frames.append(selected)
    counts.append(
        {
            "experiment": "current_w40_loss_occam",
            "input_unique_paths": int(current.shape[1]),
            "new_global_unique_paths": len(keep),
        }
    )
    return pd.concat(frames, axis=1), counts


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_experiment(root: Path, config_path: Path, output: Path) -> dict:
    config = _load(config_path)
    if QUALITY_METADATA["version"] != config["frozen_layers"][
        "momentum_factor_version"
    ]:
        raise AssertionError("Momentum factor version mismatch")
    output.mkdir(parents=True, exist_ok=True)
    end = pd.Timestamp(config["periods"]["full"][1])
    context = build_gold_override_context(root, end=end.date())
    data = build_exact_execution_data(context)
    close = load_ohlc("510300.SH", end.date())["close"]
    raw_loss, score = w40_loss_percentile_at_open(close, data.calendar)
    specs = _specs(config)
    weighted_daily = pd.read_parquet(
        root / config["frozen_layers"]["current_formal_daily"]
    ).reindex(data.calendar)
    if weighted_daily[["return", "candidate"]].isna().any().any():
        raise ValueError("current formal daily does not cover search calendar")
    baseline = weighted_daily["return"].astype(float)
    trim = config["extreme_block_trim"]
    closes = {
        asset: load_ohlc(asset, end.date())["close"]
        for asset in trim["shock_assets"]
    }
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
    extreme.blocks.to_csv(output / "shock_blocks.csv")
    ordinary = extreme.selection_mask.astype(bool)

    ids = list(specs)
    matrix = np.empty((len(data.calendar), len(ids)), dtype=np.float32)
    records = []
    for position, candidate_id in enumerate(ids):
        run = run_w40_loss_gate(data, score, specs[candidate_id])
        matrix[:, position] = run.returns
        records.append(
            {
                "candidate_id": candidate_id,
                **specs[candidate_id].__dict__,
                "defender_entries": run.defender_entries,
                "defender_days": run.defender_days,
                "sleeve_switches": run.sleeve_switches,
                "candidate_switches": run.candidate_switches,
            }
        )
        if (position + 1) % 250 == 0 or position + 1 == len(ids):
            print(f"W40 loss: {position + 1}/{len(ids)}", flush=True)
    returns = pd.DataFrame(matrix, index=data.calendar, columns=ids)
    metadata = pd.DataFrame(records).set_index("candidate_id")
    table = _add_neighborhood(
        _add_search_metrics(metadata, returns, baseline, ordinary, config), config
    )
    selected_full, table = _select(
        table, config, "selection_full_sample", "full"
    )
    selected_ordinary, table = _select(
        table, config, "selection_excluding_extremes", "ordinary"
    )
    table.to_csv(output / "candidate_grid.csv")
    _pareto(table, "full").to_csv(output / "pareto_full.csv")
    _pareto(table, "ordinary").to_csv(output / "pareto_ordinary.csv")
    neighborhood_summaries = {}
    for label, selected in (
        ("full_sample", selected_full),
        ("excluding_extremes", selected_ordinary),
    ):
        neighborhood, summary = _neighborhood_detail(table, selected, config)
        neighborhood.to_csv(output / f"selected_{label}_neighborhood.csv")
        neighborhood_summaries[label] = summary

    unique = _unique_paths(returns)
    unique.to_parquet(output / "unique_candidate_returns.parquet")
    cscv_full, cscv_full_summary = cscv_pbo(
        unique, baseline, block_count=int(config["overfit_checks"]["cscv_blocks"])
    )
    cscv_ordinary, cscv_ordinary_summary = cscv_pbo(
        unique.loc[ordinary],
        baseline.loc[ordinary],
        block_count=int(config["overfit_checks"]["cscv_blocks"]),
    )
    cscv_full.to_csv(output / "cscv_full.csv", index=False)
    cscv_ordinary.to_csv(output / "cscv_ordinary.csv", index=False)
    current_reality = {
        "full": yearly_reality_check(
            unique,
            baseline,
            repetitions=int(config["overfit_checks"]["yearly_reality_check_repetitions"]),
            seed=int(config["overfit_checks"]["random_seed"]),
        ),
        "ordinary": yearly_reality_check(
            unique.loc[ordinary],
            baseline.loc[ordinary],
            repetitions=int(config["overfit_checks"]["yearly_reality_check_repetitions"]),
            seed=int(config["overfit_checks"]["random_seed"]),
        ),
    }
    walk = expanding_walk_forward(unique, baseline)
    leave = leave_one_year_selection(unique, baseline)
    walk.to_csv(output / "walk_forward.csv", index=False)
    leave.to_csv(output / "leave_one_year.csv", index=False)

    global_audit = None
    if config["overfit_checks"].get("include_prior_downside_raqm_paths", False):
        global_paths, family_counts = _global_prior_paths(root, unique)
        global_audit = {
            "input_candidate_ids": 72144 + len(specs),
            "global_unique_paths": int(global_paths.shape[1]),
            "family_path_counts": family_counts,
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
        (output / "global_prior_audit.json").write_text(
            json.dumps(global_audit, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    details = {
        "full_sample": _selected_detail(
            "full_sample",
            selected_full,
            specs[str(selected_full.name)],
            data,
            score,
            raw_loss,
            context,
            weighted_daily,
            ordinary,
            output,
            config,
        ),
        "excluding_extremes": _selected_detail(
            "excluding_extremes",
            selected_ordinary,
            specs[str(selected_ordinary.name)],
            data,
            score,
            raw_loss,
            context,
            weighted_daily,
            ordinary,
            output,
            config,
        ),
    }
    audit = {
        "experiment": config["experiment"],
        "calendar": {
            "start": data.calendar.min().date().isoformat(),
            "end": data.calendar.max().date().isoformat(),
            "observations": len(data.calendar),
            "ordinary_observations": int(ordinary.sum()),
        },
        "search": {
            "candidate_ids": len(specs),
            "unique_paths": int(unique.shape[1]),
            "selected_full_sample": str(selected_full.name),
            "selected_excluding_extremes": str(selected_ordinary.name),
        },
        "weighted_formal": performance(baseline),
        "neighborhoods": neighborhood_summaries,
        "cscv": {"full": cscv_full_summary, "ordinary": cscv_ordinary_summary},
        "reality_check": current_reality,
        "global_prior_audit": global_audit,
        "walk_forward": {
            "return_win_rate": float(walk["test_return_delta"].gt(0.0).mean()),
            "sharpe_win_rate": float(walk["test_sharpe_delta"].gt(0.0).mean()),
        },
        "leave_one_year": {
            "return_win_rate": float(leave["test_return_delta"].gt(0.0).mean()),
            "sharpe_win_rate": float(leave["test_sharpe_delta"].gt(0.0).mean()),
        },
        "selected": details,
        "causality": {
            "factor": "previous_close_40_day_downside_log_loss",
            "percentile": "rolling_504_strict_lag_min_252",
            "exact_exit_and_entry_legs": True,
            "untradable_switch_retains_previous_candidate": True,
            "baseline_parity_max_abs_error": context.baseline_parity_max_abs_error,
        },
        "decision": config["decision"],
    }
    (output / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    source_paths = [
        config_path,
        Path(__file__),
        root / "research/momentum_defender_w40_loss_gate.py",
        root / "research/tests/test_momentum_defender_w40_loss_gate.py",
        root / "research/momentum_defender_downside_raqm.py",
        root / "research/configs/momentum_defender_w40_loss_full_sample_selected.yaml",
        root / "research/configs/momentum_defender_w40_loss_excluding_extremes_selected.yaml",
        root / "docs/research/2026-08-25_w40_loss_occam_dual_objective.md",
        root / "experiments/20260825_momentum_defender_downside_raqm_weighted_v1_formal/daily_backtest.parquet",
    ]
    manifest = {
        "experiment_id": config["experiment"]["id"],
        "files": {
            str(path.resolve().relative_to(root)): _sha(path.resolve())
            for path in source_paths
        },
    }
    (output / "experiment_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report = f"""# 单一W40下跌幅度分位双目标寻参

固定因子为510300的40日对数下跌幅度，不使用加权、路径效率、波动率调整、地板或clip。
实际搜索{len(specs):,}个状态参数ID，去重为{unique.shape[1]:,}条收益路径。完整样本选择为
`{selected_full.name}`；剔除极端行情选择为`{selected_ordinary.name}`。两类选择都只在选参
目标上不同，最终检查点保留完整1,841日。

当前正式加权DRAQM为{performance(baseline)['annualized_return_252']:.2%}年化、
Sharpe {performance(baseline)['sharpe']:.3f}。本研究不自动替换正式策略；详细结论以
`audit.json`、两份selected config及标准HTML为准。
"""
    (output / "research_report.md").write_text(report, encoding="utf-8")
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
