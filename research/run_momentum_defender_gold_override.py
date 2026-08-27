"""Search and report Gold overrides on top of the integrated C2 strategy."""

from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import asdict
from datetime import date
from pathlib import Path

import pandas as pd
import yaml

from research.momentum_defender_gold_override import (
    GoldOverrideParams,
    build_gold_override_context,
    candidate_record,
    run_gold_override,
    search_grid,
)
from research.momentum_defender_occam import performance
from research.standard_report import generate_standard_report


DEFAULT_CONFIG = Path("research/configs/momentum_defender_gold_override_search.yaml")
DEFAULT_OUTPUT = Path("experiments/20260823_momentum_defender_gold_override")


def _load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("gold override config must be a mapping")
    return config


def _periods(config: dict) -> dict[str, tuple[date, date]]:
    return {
        label: (date.fromisoformat(values[0]), date.fromisoformat(values[1]))
        for label, values in config["periods"].items()
    }


def _baseline_record(returns: pd.Series, periods) -> dict[str, object]:
    row: dict[str, object] = {"candidate_id": "current_integrated_c2"}
    for label, (start, end) in periods.items():
        metrics = performance(returns.loc[pd.Timestamp(start) : pd.Timestamp(end)])
        for field in (
            "observations",
            "total_return",
            "annualized_return_252",
            "annualized_volatility",
            "sharpe",
            "max_drawdown",
        ):
            row[f"{label}_{field}"] = metrics[field]
    row["worst_split_sharpe"] = min(
        float(row[f"{label}_sharpe"])
        for label in periods
        if label != "full"
    )
    return row


def _params(row: pd.Series) -> GoldOverrideParams:
    return GoldOverrideParams(
        metric=str(row["metric"]),
        window=int(row["window"]),
        entry_threshold=float(row["entry_threshold"]),
        exit_threshold=float(row["exit_threshold"]),
        min_gold_hold_days=int(row["min_gold_hold_days"]),
    )


def _neighbors(grid: pd.DataFrame, selected: pd.Series) -> pd.DataFrame:
    parameters = [
        "metric",
        "window",
        "entry_threshold",
        "exit_threshold",
        "min_gold_hold_days",
    ]
    mask = pd.Series(False, index=grid.index)
    for changed in parameters:
        same = pd.Series(True, index=grid.index)
        for parameter in parameters:
            if parameter != changed:
                same &= grid[parameter].eq(selected[parameter])
        mask |= same
    columns = [
        "candidate_id",
        *parameters,
        "full_annualized_return_252",
        "full_sharpe",
        "full_max_drawdown",
        "development_sharpe",
        "validation_sharpe",
        "recent_sharpe",
        "worst_split_sharpe",
        "gold_override_entries",
        "gold_override_days",
    ]
    return grid.loc[mask, columns].sort_values(
        ["worst_split_sharpe", "full_sharpe"], ascending=False
    )


def run_experiment(root: Path, config_path: Path, output: Path) -> dict[str, object]:
    config = _load_config(config_path)
    periods = _periods(config)
    context = build_gold_override_context(root)
    baseline_returns = context.integrated.result.simulated["return"].astype(float)
    baseline = _baseline_record(baseline_returns, periods)

    literal_config = config["literal_variants"]
    literals = []
    literal_runs = {}
    for metric in literal_config["metrics"]:
        params = GoldOverrideParams(
            metric=metric,
            window=int(literal_config["window"]),
            entry_threshold=float(literal_config["entry_threshold"]),
            exit_threshold=float(literal_config["exit_threshold"]),
            min_gold_hold_days=int(literal_config["min_gold_hold_days"]),
        )
        run = run_gold_override(context, params)
        literal_runs[metric] = run
        literals.append(candidate_record(run, periods))
    literal_frame = pd.DataFrame(literals)

    grid = search_grid(context, periods, config["grid"])
    for field in (
        "annualized_return_252",
        "sharpe",
        "max_drawdown",
        "worst_split_sharpe",
    ):
        baseline_field = field if field == "worst_split_sharpe" else f"full_{field}"
        grid[f"delta_{field}"] = grid[baseline_field] - float(
            baseline[baseline_field]
        )
    selection = config["selection"]
    eligible = grid.loc[
        grid["gold_override_entries"].ge(
            int(selection["minimum_gold_override_entries"])
        )
        & grid["gold_override_days"].ge(
            int(selection["minimum_gold_override_days"])
        )
    ].copy()
    if eligible.empty:
        raise RuntimeError("gold override search produced no eligible candidates")
    best_annual = eligible.sort_values(
        ["full_annualized_return_252", "full_sharpe"], ascending=False
    ).iloc[0]
    best_sharpe = eligible.sort_values(
        ["full_sharpe", "full_annualized_return_252"], ascending=False
    ).iloc[0]
    best_mdd = eligible.sort_values(
        ["full_max_drawdown", "full_sharpe"], ascending=False
    ).iloc[0]
    robust = eligible.sort_values(
        ["worst_split_sharpe", "full_sharpe", "full_annualized_return_252"],
        ascending=False,
    ).iloc[0]
    tolerance = selection["balanced_tolerances"]
    balanced_pool = eligible.loc[
        eligible["delta_annualized_return_252"].ge(
            float(tolerance["annualized_return_delta"])
        )
        & eligible["delta_sharpe"].ge(float(tolerance["sharpe_delta"]))
        & eligible["delta_max_drawdown"].ge(
            float(tolerance["max_drawdown_delta"])
        )
        & (
            eligible[
                [
                    "delta_annualized_return_252",
                    "delta_sharpe",
                    "delta_max_drawdown",
                ]
            ].gt(0.0).any(axis=1)
        )
    ].copy()
    selected = (
        balanced_pool.sort_values(
            selection["balanced_sort"], ascending=False
        ).iloc[0]
        if not balanced_pool.empty
        else robust
    )
    selected_run = run_gold_override(context, _params(selected))
    winner_runs = {
        "selected_balanced": selected_run,
        "best_annual": run_gold_override(context, _params(best_annual)),
        "best_sharpe": run_gold_override(context, _params(best_sharpe)),
        "best_mdd": run_gold_override(context, _params(best_mdd)),
        "best_robust": run_gold_override(context, _params(robust)),
    }

    improvement_counts = {
        "annualized_return": int(eligible["delta_annualized_return_252"].gt(0).sum()),
        "sharpe": int(eligible["delta_sharpe"].gt(0).sum()),
        "max_drawdown": int(eligible["delta_max_drawdown"].gt(0).sum()),
        "worst_split_sharpe": int(
            eligible["delta_worst_split_sharpe"].gt(0).sum()
        ),
        "all_three": int(
            eligible[
                [
                    "delta_annualized_return_252",
                    "delta_sharpe",
                    "delta_max_drawdown",
                ]
            ].gt(0).all(axis=1).sum()
        ),
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    grid.sort_values(["worst_split_sharpe", "full_sharpe"], ascending=False).to_csv(
        stage / "candidate_grid.csv", index=False
    )
    literal_frame.to_csv(stage / "literal_metrics.csv", index=False)
    pd.DataFrame([baseline]).to_csv(stage / "baseline_metrics.csv", index=False)
    _neighbors(grid, selected).to_csv(stage / "selected_neighborhood.csv", index=False)
    for name, run in winner_runs.items():
        run.state.join(run.daily, rsuffix="_execution").to_csv(
            stage / f"daily_{name}.csv"
        )
    (stage / "strategy_config.yaml").write_text(
        config_path.read_text(encoding="utf-8"), encoding="utf-8"
    )

    report_config = {
        "strategy_name": "momentum_defender_c2_gold_override",
        "base_strategy": config["experiment"]["base_strategy"],
        "selected_params": asdict(selected_run.params),
        "evidence_status": config["experiment"]["evidence_status"],
    }
    generate_standard_report(
        selected_run.daily["return"],
        baseline_returns,
        "Current Integrated C2",
        stage / "selected_balanced_vs_current_c2.html",
        report_config,
    )
    generate_standard_report(
        winner_runs["best_annual"].daily["return"],
        baseline_returns,
        "Current Integrated C2",
        stage / "best_annual_vs_current_c2.html",
        {**report_config, "selected_params": asdict(winner_runs["best_annual"].params)},
    )
    generate_standard_report(
        winner_runs["best_sharpe"].daily["return"],
        baseline_returns,
        "Current Integrated C2",
        stage / "best_sharpe_vs_current_c2.html",
        {**report_config, "selected_params": asdict(winner_runs["best_sharpe"].params)},
    )

    report = f"""# C2基础上的黄金趋势覆盖

## 机制

基础C2状态机保持不变。仅当C2仍希望持有Defender时，比较黄金ETF与Defender整体连续净值
的上一收盘X指标；黄金领先超过入场阈值时，下一开盘直接切黄金，不受510300 40日门槛或
C2 30日锁限制。黄金优势跌破退出阈值后回Defender；若基础C2恢复Momentum，则无条件回到
原Momentum目标。

## 基线与搜索

- 当前C2：年化 {float(baseline['full_annualized_return_252']):.2%}，Sharpe {float(baseline['full_sharpe']):.3f}，MDD {float(baseline['full_max_drawdown']):.2%}。
- 候选数：{len(grid)}；合格候选：{len(eligible)}。
- 超过基线的候选数：年化 {improvement_counts['annualized_return']}，Sharpe {improvement_counts['sharpe']}，MDD {improvement_counts['max_drawdown']}，最差分段Sharpe {improvement_counts['worst_split_sharpe']}，三项同时 {improvement_counts['all_three']}。

## 推荐折中候选

`{selected['candidate_id']}`

- 参数：指标 `{selected['metric']}`，窗口 {int(selected['window'])}日，入场差值 > {float(selected['entry_threshold']):.4f}，退出差值 ≤ {float(selected['exit_threshold']):.4f}，黄金最短持有 {int(selected['min_gold_hold_days'])}日。
- 全样本：年化 {float(selected['full_annualized_return_252']):.2%}，Sharpe {float(selected['full_sharpe']):.3f}，MDD {float(selected['full_max_drawdown']):.2%}。
- 相对基线：年化 {float(selected['delta_annualized_return_252']):+.2%}，Sharpe {float(selected['delta_sharpe']):+.3f}，MDD {float(selected['delta_max_drawdown']):+.2%}。
- 三段Sharpe：development {float(selected['development_sharpe']):.3f}，validation {float(selected['validation_sharpe']):.3f}，recent {float(selected['recent_sharpe']):.3f}；最差 {float(selected['worst_split_sharpe']):.3f}。

## 单指标最优

- 最高年化：`{best_annual['candidate_id']}`，年化 {float(best_annual['full_annualized_return_252']):.2%}，Sharpe {float(best_annual['full_sharpe']):.3f}，MDD {float(best_annual['full_max_drawdown']):.2%}。
- 最高Sharpe：`{best_sharpe['candidate_id']}`，年化 {float(best_sharpe['full_annualized_return_252']):.2%}，Sharpe {float(best_sharpe['full_sharpe']):.3f}，MDD {float(best_sharpe['full_max_drawdown']):.2%}。
- 最浅MDD：`{best_mdd['candidate_id']}`，年化 {float(best_mdd['full_annualized_return_252']):.2%}，Sharpe {float(best_mdd['full_sharpe']):.3f}，MDD {float(best_mdd['full_max_drawdown']):.2%}。
- 最佳分段下限：`{robust['candidate_id']}`，最差分段Sharpe {float(robust['worst_split_sharpe']):.3f}。

## 证据边界

所有参数都使用了已观察历史搜索，属于回溯证据，不是独立样本外。报告会给出邻域与分段表现，
但不会自动覆盖生产C2；是否晋升应结合提升幅度、机制可解释性和前瞻记录决定。
"""
    (stage / "research_report.md").write_text(report, encoding="utf-8")

    summary = {
        "experiment_id": config["experiment"]["id"],
        "candidate_count": int(len(grid)),
        "eligible_candidate_count": int(len(eligible)),
        "baseline_parity_max_abs_error": context.baseline_parity_max_abs_error,
        "improvement_counts": improvement_counts,
        "selected_balanced": selected.to_dict(),
        "best_annual": best_annual.to_dict(),
        "best_sharpe": best_sharpe.to_dict(),
        "best_mdd": best_mdd.to_dict(),
        "best_robust": robust.to_dict(),
        "winner_audits": {name: run.audit for name, run in winner_runs.items()},
    }
    (stage / "search_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    output.mkdir(parents=True, exist_ok=True)
    for path in stage.iterdir():
        path.replace(output / path.name)
    stage.rmdir()
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    summary = run_experiment(args.root.resolve(), args.config, args.output)
    print(
        f"searched={summary['candidate_count']} "
        f"improved_all_three={summary['improvement_counts']['all_three']} "
        f"selected={summary['selected_balanced']['candidate_id']}"
    )
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
