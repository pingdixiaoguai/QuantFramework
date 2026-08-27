"""Search only the entry/exit thresholds of the fixed Gold min-5 mechanism."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from research.gold_min5_risk_adjusted_escape import (
    GoldMin5Params,
    collect_grid,
    run_gold_min5,
)
from research.momentum_defender_gold_override import build_gold_override_context
from research.momentum_defender_gold_override_overfit import (
    cscv_pbo,
    expanding_walk_forward,
    full_metrics,
    paired_block_bootstrap,
    yearly_reality_check,
)
from research.standard_report import generate_standard_report


DEFAULT_CONFIG = Path("research/configs/gold_min5_risk_adjusted_escape_search.yaml")
DEFAULT_OUTPUT = Path("experiments/20260823_gold_min5_risk_adjusted_escape")


def _range(start: float, end: float, step: float) -> list[float]:
    count = int(round((end - start) / step))
    return [round(start + position * step, 10) for position in range(count + 1)]


def _params(row: pd.Series) -> GoldMin5Params:
    return GoldMin5Params(
        entry_difference=float(row["entry_difference"]),
        exit_difference=float(row["exit_difference"]),
    )


def run_experiment(root: Path, config_path: Path, output: Path) -> dict[str, object]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    context = build_gold_override_context(root)
    baseline = context.integrated.result.simulated["return"].astype(float)
    grid_config = config["threshold_grid"]
    entries = _range(
        float(grid_config["entry_start"]),
        float(grid_config["entry_end"]),
        float(grid_config["entry_step"]),
    )
    exits = _range(
        float(grid_config["exit_start"]),
        float(grid_config["exit_end"]),
        float(grid_config["exit_step"]),
    )
    metadata, returns = collect_grid(context, entries, exits)
    metrics = full_metrics(returns, baseline).join(metadata)
    periods = {
        label: (pd.Timestamp(values[0]), pd.Timestamp(values[1]))
        for label, values in config["periods"].items()
    }
    for label, (start, end) in periods.items():
        period = full_metrics(returns.loc[start:end], baseline.loc[start:end])
        for field in (
            "annualized_return_252",
            "sharpe",
            "max_drawdown",
            "delta_annualized_return_252",
            "delta_sharpe",
            "delta_max_drawdown",
        ):
            metrics[f"{label}_{field}"] = period[field]
    metrics["worst_split_sharpe"] = metrics[
        ["development_sharpe", "validation_sharpe", "recent_sharpe"]
    ].min(axis=1)
    selection = config["selection"]
    eligible = metrics.loc[
        metrics["gold_entries"].ge(int(selection["minimum_gold_entries"]))
        & metrics["gold_days"].ge(int(selection["minimum_gold_days"]))
    ].copy()
    active_candidates = metrics.loc[metrics["gold_entries"].gt(0)].copy()
    best_annual = active_candidates.sort_values(
        ["annualized_return_252", "sharpe"], ascending=False
    ).iloc[0]
    best_sharpe = active_candidates.sort_values(
        ["sharpe", "annualized_return_252"], ascending=False
    ).iloc[0]
    eligible["annual_rank"] = eligible["annualized_return_252"].rank(
        method="min", ascending=False
    )
    eligible["sharpe_rank"] = eligible["sharpe"].rank(
        method="min", ascending=False
    )
    eligible["joint_rank"] = eligible["annual_rank"] + eligible["sharpe_rank"]
    balanced = eligible.sort_values(
        ["joint_rank", "annualized_return_252", "sharpe"], ascending=[True, False, False]
    ).iloc[0]
    pareto_mask = []
    values = eligible[["annualized_return_252", "sharpe"]].to_numpy(float)
    for index, value in enumerate(values):
        dominated = np.any(
            np.all(values >= value, axis=1)
            & np.any(values > value, axis=1)
        )
        pareto_mask.append(not dominated)
    pareto = eligible.loc[pareto_mask].copy()

    winners = {
        "balanced": balanced,
        "best_annual": best_annual,
        "best_sharpe": best_sharpe,
    }
    winner_runs = {
        name: run_gold_min5(context, _params(row))
        for name, row in winners.items()
    }
    cscv, pbo = cscv_pbo(returns, baseline, block_count=16)
    walk = expanding_walk_forward(returns, baseline)
    bootstrap_results = {}
    bootstrap_frames = {}
    for name, row in winners.items():
        frame, summary = paired_block_bootstrap(
            returns[str(row.name)], baseline, repetitions=5000
        )
        bootstrap_frames[name] = frame
        bootstrap_results[name] = summary
    reality = yearly_reality_check(returns, baseline, repetitions=5000)

    output.mkdir(parents=True, exist_ok=True)
    metrics.sort_values(
        ["annualized_return_252", "sharpe"], ascending=False
    ).to_csv(output / "candidate_metrics.csv")
    pareto.sort_values(
        ["annualized_return_252", "sharpe"], ascending=False
    ).to_csv(output / "pareto_frontier.csv")
    cscv.to_csv(output / "cscv_pbo.csv", index=False)
    walk.to_csv(output / "expanding_walk_forward.csv", index=False)
    for name, frame in bootstrap_frames.items():
        frame.to_csv(output / f"paired_block_bootstrap_{name}.csv", index=False)
    for name, run in winner_runs.items():
        run.state.join(run.daily, rsuffix="_execution").to_csv(
            output / f"daily_{name}.csv"
        )
        generate_standard_report(
            run.daily["return"],
            baseline,
            "Current Integrated C2",
            output / f"{name}_vs_current_c2.html",
            {
                "strategy_name": config["experiment"]["id"],
                "variant": name,
                "params": run.params.__dict__,
                "fixed_mechanism": config["fixed_mechanism"],
            },
        )
    (output / "strategy_config.yaml").write_text(
        config_path.read_text(encoding="utf-8"), encoding="utf-8"
    )

    improvement_counts = {
        "annualized_return": int(metrics["delta_annualized_return_252"].gt(0).sum()),
        "sharpe": int(metrics["delta_sharpe"].gt(0).sum()),
        "mdd_nonworsening": int(metrics["delta_max_drawdown"].ge(-1e-12).sum()),
        "annual_and_sharpe": int(
            (
                metrics["delta_annualized_return_252"].gt(0)
                & metrics["delta_sharpe"].gt(0)
            ).sum()
        ),
        "all_three": int(
            (
                metrics["delta_annualized_return_252"].gt(0)
                & metrics["delta_sharpe"].gt(0)
                & metrics["delta_max_drawdown"].ge(-1e-12)
            ).sum()
        ),
    }
    selected_bootstrap = bootstrap_results["balanced"]
    flags = {
        "pbo_above_half": float(pbo["pbo"]) >= 0.5,
        "reality_not_significant": float(reality["p_value"]) >= 0.10,
        "bootstrap_return_below_90pct": float(
            selected_bootstrap["annualized_return_delta_positive_probability"]
        ) < 0.90,
        "bootstrap_sharpe_below_90pct": float(
            selected_bootstrap["sharpe_delta_positive_probability"]
        ) < 0.90,
        "walk_forward_return_win_below_half": float(
            walk["test_return_delta"].gt(0).mean()
        ) < 0.5,
    }
    assessment = (
        "high" if sum(flags.values()) >= 3 else "moderate" if any(flags.values()) else "low"
    )
    next_state = winner_runs["balanced"].state.iloc[-1]
    summary = {
        "experiment_id": config["experiment"]["id"],
        "candidate_count": int(len(metrics)),
        "eligible_candidate_count": int(len(eligible)),
        "pareto_candidate_count": int(len(pareto)),
        "improvement_counts": improvement_counts,
        "balanced": balanced.to_dict(),
        "balanced_id": str(balanced.name),
        "best_annual": best_annual.to_dict(),
        "best_annual_id": str(best_annual.name),
        "best_sharpe": best_sharpe.to_dict(),
        "best_sharpe_id": str(best_sharpe.name),
        "winner_audits": {name: run.audit for name, run in winner_runs.items()},
        "pbo": pbo,
        "walk_forward_return_win_rate": float(walk["test_return_delta"].gt(0).mean()),
        "walk_forward_sharpe_win_rate": float(walk["test_sharpe_delta"].gt(0).mean()),
        "bootstrap": bootstrap_results,
        "reality_check": reality,
        "overfit_flags": flags,
        "overfit_assessment": assessment,
        "latest_balanced_state": next_state.to_dict(),
        "production_replacement": False,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )

    def line(label: str, row: pd.Series) -> str:
        return (
            f"|{label}|{float(row['entry_difference']):.3f}|{float(row['exit_difference']):.3f}|"
            f"{float(row['annualized_return_252']):.2%}|{float(row['sharpe']):.3f}|"
            f"{float(row['max_drawdown']):.2%}|{int(row['gold_entries'])}|{int(row['gold_days'])}|"
        )

    report = f"""# 固定10日风险调整收益差：黄金最少持有5日

## 固定机制

X为10日收益率除以10日年化日波动率，Gold与Defender完全同口径。只有黄金可以从Defender
逃生；切入后前5个完整交易日无条件持有。第6个开盘起，若基础C2已为Momentum则切原
Momentum Top1，否则仅由风险调整收益差与退出阈值决定是否回Defender。

## 阈值搜索

- 入场阈值0.000至0.800，步长0.025；退出阈值-0.800至0.200，步长0.025。
- 有效候选{len(metrics)}组，Pareto前沿{len(pareto)}组。
- 超过当前C2：年化{improvement_counts['annualized_return']}组、Sharpe{improvement_counts['sharpe']}组、年化和Sharpe同时{improvement_counts['annual_and_sharpe']}组、三项同时{improvement_counts['all_three']}组。

|候选|入场差|退出差|年化|Sharpe|MDD|黄金入场|黄金日数|
|---|---:|---:|---:|---:|---:|---:|---:|
{line('年化/Sharpe折中', balanced)}
{line('最高年化', best_annual)}
{line('最高Sharpe', best_sharpe)}

## 稳健性

- PBO {float(pbo['pbo']):.1%}；walk-forward收益/Sharpe胜率{summary['walk_forward_return_win_rate']:.1%}/{summary['walk_forward_sharpe_win_rate']:.1%}。
- 折中候选bootstrap年化差为正概率{selected_bootstrap['annualized_return_delta_positive_probability']:.1%}、Sharpe差为正概率{selected_bootstrap['sharpe_delta_positive_probability']:.1%}。
- 年度块多重试验校正p={float(reality['p_value']):.3f}；综合过拟合风险**{assessment.upper()}**。

## 最新状态

截至{context.calendar.max().date()}，折中候选黄金状态{'激活' if bool(next_state['gold_active']) else '未激活'}，
10日风险调整收益差为{float(next_state['metric_difference_at_open']):+.3f}。

## 结论

本实验只搜索两个阈值，其他机制完全固定。最终是否值得采用应同时看年化、Sharpe、回撤和
搜索后统计审计；不会自动修改生产C2。
"""
    (output / "report.md").write_text(report, encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    summary = run_experiment(args.root.resolve(), args.config, args.output)
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
