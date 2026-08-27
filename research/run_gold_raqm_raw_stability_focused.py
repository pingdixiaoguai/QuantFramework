"""Focused raw Gold RAQM search on the absolute-stability base state only."""

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
    "research/configs/gold_raqm_raw_stability_focused_search.yaml"
)
DEFAULT_OUTPUT = Path(
    "experiments/20260824_gold_raqm_raw_stability_focused"
)
STABILITY_DAILY = Path(
    "experiments/20260824_momentum_defender_log_qm_absolute_stability_candidate/daily.csv"
)


def _specs(config: dict) -> list[GoldRuleSpec]:
    hold = int(config["factor"]["hard_min_gold_hold_days"])
    return [
        GoldRuleSpec(
            RAQMSpec("raw", int(window), None, None, 0),
            float(entry),
            float(exit_),
            hold,
        )
        for window in config["factor"]["windows"]
        for entry in config["threshold_grid"]["entry_differences"]
        for exit_ in config["threshold_grid"]["exit_differences"]
        if float(exit_) <= float(entry)
    ]


def _no_gold(data, risk_on):
    defender = data.candidate_index[DEFENDER_CANDIDATE]
    requested = np.where(risk_on, data.momentum_target, defender).astype(int)
    return fast_candidate_schedule(data, requested)


def _rank(metadata, returns, baseline, years, config):
    table = metadata.join(full_metrics(returns, baseline)).join(
        robust_leave_year_metrics(returns, baseline, years)
    )
    selection = config["selection"]
    table["robust_eligible"] = (
        table["gold_entries"].ge(int(selection["minimum_gold_entries"]))
        & table["delta_annualized_return_252"].ge(
            float(selection["full_annualized_delta_floor"])
        )
        & table["delta_sharpe"].ge(float(selection["full_sharpe_delta_floor"]))
        & table["delta_max_drawdown"].ge(float(selection["full_mdd_delta_floor"]))
        & table["leave_year_annualized_return_252_q25"].ge(
            float(selection["leave_year_annualized_delta_q25_floor"])
        )
        & table["leave_year_annualized_return_252_median"].ge(
            float(selection["leave_year_annualized_delta_median_floor"])
        )
        & table["leave_year_sharpe_q25"].ge(
            float(selection["leave_year_sharpe_delta_q25_floor"])
        )
        & table["leave_year_sharpe_median"].ge(
            float(selection["leave_year_sharpe_delta_median_floor"])
        )
    )
    columns = [
        "leave_year_annualized_return_252_q25",
        "leave_year_sharpe_q25",
        "delta_max_drawdown",
    ]
    pool = table["robust_eligible"]
    if not pool.any():
        pool = pd.Series(True, index=table.index)
    ranks = table.loc[pool, columns].rank(pct=True)
    table.loc[pool, "minimum_robust_percentile"] = ranks.min(axis=1)
    table.loc[pool, "mean_robust_percentile"] = ranks.mean(axis=1)
    table["minimum_robust_percentile"] = table[
        "minimum_robust_percentile"
    ].fillna(-1.0)
    table["mean_robust_percentile"] = table["mean_robust_percentile"].fillna(-1.0)
    return table


def _select(table):
    pool = table.loc[table["robust_eligible"]].copy()
    if pool.empty:
        pool = table.copy()
    pool["_candidate_sort_key"] = pool.index.astype(str)
    return pool.sort_values(
        [
            "minimum_robust_percentile",
            "mean_robust_percentile",
            "sharpe",
            "annualized_return_252",
            "switches",
            "_candidate_sort_key",
        ],
        ascending=[False, False, False, False, True, True],
    ).iloc[0]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_experiment(root: Path, config_path: Path, output: Path) -> dict[str, object]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    full_start, full_end = map(pd.Timestamp, config["periods"]["full"])
    years = list(map(int, config["periods"]["calendar_years"]))
    context = build_gold_override_context(root, end=full_end.date())
    data = build_fast_switch_data(
        context,
        metric_at_open(context.curves, RAQMSpec("raw", 5, None, None, 0)),
    )
    stable = pd.read_csv(root / STABILITY_DAILY, parse_dates=["date"]).set_index(
        "date"
    ).reindex(context.calendar)
    risk_on = stable["risk_on"].astype(bool).to_numpy()
    no_gold_values, no_gold_codes, _ = _no_gold(data, risk_on)
    no_gold = pd.Series(no_gold_values, index=context.calendar)
    no_gold_target = pd.Series(
        [data.candidates[index] for index in no_gold_codes], index=context.calendar
    )
    specs = _specs(config)
    metrics = {
        window: metric_at_open(
            context.curves, RAQMSpec("raw", int(window), None, None, 0)
        )["difference"]
        for window in config["factor"]["windows"]
    }
    records = []
    returns = {}
    for spec in specs:
        result = run_gold_rule(data, risk_on, metrics[spec.factor.window], spec)
        candidate_id = spec.candidate_id()
        returns[candidate_id] = result.returns
        records.append(
            {
                "candidate_id": candidate_id,
                "window": spec.factor.window,
                "entry_difference": spec.entry_difference,
                "exit_difference": spec.exit_difference,
                "hard_min_hold_days": spec.hard_min_hold_days,
                "gold_entries": result.gold_entries,
                "gold_days": result.gold_days,
                "switches": result.switches,
            }
        )
    metadata = pd.DataFrame(records).set_index("candidate_id")
    matrix = pd.DataFrame(returns, index=context.calendar)
    table = _rank(metadata, matrix, no_gold, years, config)
    selected_row = _select(table)
    selected_id = str(selected_row.name)
    selected_spec = next(value for value in specs if value.candidate_id() == selected_id)
    selected_result = run_gold_rule(
        data, risk_on, metrics[selected_spec.factor.window], selected_spec
    )
    selected_returns = pd.Series(selected_result.returns, index=context.calendar)
    selected_target = pd.Series(
        [data.candidates[index] for index in selected_result.target_candidate],
        index=context.calendar,
    )

    unique = _unique_paths(matrix)
    checks = config["overfit_checks"]
    pbo_frame, pbo = cscv_pbo(unique, no_gold, block_count=int(checks["cscv_blocks"]))
    walk = expanding_walk_forward(unique, no_gold)
    reality = yearly_reality_check(
        unique,
        no_gold,
        repetitions=int(checks["yearly_reality_check_repetitions"]),
        seed=int(checks["random_seed"]),
    )
    bootstrap, bootstrap_summary = paired_block_bootstrap(
        selected_returns,
        no_gold,
        block_size=int(checks["paired_block_bootstrap_block"]),
        repetitions=int(checks["paired_block_bootstrap_repetitions"]),
        seed=int(checks["random_seed"]),
    )
    events, leave_events, deletions, event_summary = _event_stress(
        selected_returns,
        no_gold,
        selected_target,
        no_gold_target,
        list(map(int, checks["top_positive_event_deletions"])),
    )
    costs = _selected_cost_schedule(context, data, selected_result.target_candidate)
    friction = _friction(
        selected_returns,
        costs,
        list(map(float, checks["friction_cost_multipliers"])),
    )
    excess_matrix = unique.to_numpy(float) - no_gold.to_numpy(float)[:, None]
    excess_matrix = excess_matrix[:, excess_matrix.std(axis=0, ddof=1) > 1e-14]
    dsr = _deflated_sharpe(
        selected_returns.to_numpy(float) - no_gold.to_numpy(float),
        excess_matrix,
        _effective_trials(excess_matrix),
    )
    window_summary = pd.DataFrame(
        [
            {
                "window": int(window),
                "candidate_ids": int(table["window"].eq(window).sum()),
                "robust_eligible": int(
                    (table["window"].eq(window) & table["robust_eligible"]).sum()
                ),
                "best_annualized_return_252": float(
                    table.loc[table["window"].eq(window), "annualized_return_252"].max()
                ),
                "best_sharpe": float(
                    table.loc[table["window"].eq(window), "sharpe"].max()
                ),
                "best_max_drawdown": float(
                    table.loc[table["window"].eq(window), "max_drawdown"].max()
                ),
            }
            for window in config["factor"]["windows"]
        ]
    )
    selected_metrics = performance(selected_returns)
    baseline_metrics = performance(no_gold)
    production_supported = bool(
        selected_row["robust_eligible"]
        and reality["p_value"] < 0.05
        and bootstrap_summary["annualized_return_delta_ci_lower"] > 0.0
        and bootstrap_summary["sharpe_delta_ci_lower"] > 0.0
        and walk["test_return_delta"].gt(0.0).mean() >= 0.60
        and walk["test_sharpe_delta"].gt(0.0).mean() >= 0.60
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    table.to_csv(stage / "candidate_grid.csv")
    unique.to_parquet(stage / "unique_returns.parquet")
    window_summary.to_csv(stage / "window_summary.csv", index=False)
    pd.DataFrame(
        [
            {"strategy": "no_gold", **baseline_metrics},
            {"strategy": "selected_raw_gold", **selected_metrics},
        ]
    ).to_csv(stage / "strategy_metrics.csv", index=False)
    selected_row.to_frame().T.to_csv(stage / "selected_metrics.csv")
    pbo_frame.to_csv(stage / "cscv_pbo.csv", index=False)
    walk.to_csv(stage / "walk_forward.csv", index=False)
    bootstrap.to_csv(stage / "paired_bootstrap.csv", index=False)
    events.to_csv(stage / "events.csv", index=False)
    leave_events.to_csv(stage / "leave_one_event.csv", index=False)
    deletions.to_csv(stage / "top_event_deletion.csv", index=False)
    friction.to_csv(stage / "friction.csv", index=False)
    selected_config = {
        "strategy_id": "gold_raqm_raw_stability_focused_v1",
        "status": "promotion_supported" if production_supported else "research_rejected",
        "window": selected_spec.factor.window,
        "entry_difference": selected_spec.entry_difference,
        "exit_difference": selected_spec.exit_difference,
        "hard_min_hold_days": selected_spec.hard_min_hold_days,
        "volatility_floor_annual": None,
        "winsor_limit": None,
        "robust_eligible": bool(selected_row["robust_eligible"]),
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
        "unique_paths": int(unique.shape[1]),
        "selected_candidate": selected_id,
        "selected_config": selected_config,
        "window_summary": window_summary.to_dict(orient="records"),
        "metrics": {"no_gold": baseline_metrics, "selected": selected_metrics},
        "gold_entries": selected_result.gold_entries,
        "pbo": pbo,
        "walk_forward": {
            "return_win_rate": float(walk["test_return_delta"].gt(0.0).mean()),
            "sharpe_win_rate": float(walk["test_sharpe_delta"].gt(0.0).mean()),
        },
        "reality_check": reality,
        "bootstrap": bootstrap_summary,
        "event_stress": event_summary,
        "deflated_sharpe_excess": dsr,
        "production_promotion_supported": production_supported,
    }
    (stage / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    generate_standard_report(
        selected_returns,
        no_gold,
        "Absolute-stability base without Gold",
        stage / "selected_vs_no_gold.html",
        selected_config,
    )
    report = f"""# Raw Gold RAQM：绝对稳定性基础状态专项搜索

只测试无地板、无剪裁的5/10/20日窗口，共{len(specs)}个参数ID、{unique.shape[1]}条唯一
收益路径。选中`{selected_id}`。

|策略|年化|Sharpe|MDD|
|---|---:|---:|---:|
|关闭Gold|{baseline_metrics['annualized_return_252']:.2%}|{baseline_metrics['sharpe']:.3f}|{baseline_metrics['max_drawdown']:.2%}|
|专项候选|{selected_metrics['annualized_return_252']:.2%}|{selected_metrics['sharpe']:.3f}|{selected_metrics['max_drawdown']:.2%}|

5/10/20日稳健合格组合分别为
{int(window_summary.loc[window_summary['window'].eq(5),'robust_eligible'].iloc[0])}/
{int(window_summary.loc[window_summary['window'].eq(10),'robust_eligible'].iloc[0])}/
{int(window_summary.loc[window_summary['window'].eq(20),'robust_eligible'].iloc[0])}。
PBO={pbo['pbo']:.1%}，Reality Check p={reality['p_value']:.4f}，Walk-forward收益/Sharpe
胜率{walk['test_return_delta'].gt(0.0).mean():.1%}/
{walk['test_sharpe_delta'].gt(0.0).mean():.1%}。

结论：{'全部门槛通过，可提交用户决定是否晋升。' if production_supported else '至少一项多重试验门槛失败，不自动替换生产。'}
"""
    (stage / "research_report.md").write_text(report, encoding="utf-8")
    sources = [
        config_path,
        root / "research/gold_raqm_regularization.py",
        root / "research/run_gold_raqm_raw_stability_focused.py",
        root / STABILITY_DAILY,
        root / "research/DEVELOPMENT_VALIDATION.md",
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
