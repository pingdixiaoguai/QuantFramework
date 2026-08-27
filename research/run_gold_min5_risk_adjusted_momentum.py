"""Search thresholds for registered 20-day risk-adjusted momentum Gold escape."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from research.gold_min5_risk_adjusted_momentum import (
    GoldRAQMParams,
    collect_grid,
    risk_adjusted_momentum_at_open,
    run_gold_raqm,
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


DEFAULT_CONFIG = Path(
    "research/configs/gold_min5_risk_adjusted_momentum_search.yaml"
)
DEFAULT_OUTPUT = Path(
    "experiments/20260823_gold_min5_risk_adjusted_momentum"
)


def _range(start: float, end: float, step: float) -> list[float]:
    count = int(round((end - start) / step))
    return [round(start + position * step, 10) for position in range(count + 1)]


def _params(row: pd.Series) -> GoldRAQMParams:
    return GoldRAQMParams(
        float(row["entry_difference"]), float(row["exit_difference"])
    )


def run_experiment(root: Path, config_path: Path, output: Path) -> dict[str, object]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    context = build_gold_override_context(root)
    baseline = context.integrated.result.simulated["return"].astype(float)
    grid = config["threshold_grid"]
    metadata, returns = collect_grid(
        context,
        _range(float(grid["entry_start"]), float(grid["entry_end"]), float(grid["entry_step"])),
        _range(float(grid["exit_start"]), float(grid["exit_end"]), float(grid["exit_step"])),
    )
    metrics = full_metrics(returns, baseline).join(metadata)
    for label, values in config["periods"].items():
        start, end = map(pd.Timestamp, values)
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
    active = metrics.loc[metrics["gold_entries"].gt(0)].copy()
    selection = config["selection"]
    eligible = active.loc[
        active["gold_entries"].ge(int(selection["minimum_gold_entries"]))
        & active["gold_days"].ge(int(selection["minimum_gold_days"]))
    ].copy()
    best_annual = active.sort_values(
        ["annualized_return_252", "sharpe"], ascending=False
    ).iloc[0]
    best_sharpe = active.sort_values(
        ["sharpe", "annualized_return_252"], ascending=False
    ).iloc[0]
    eligible["annual_rank"] = eligible["annualized_return_252"].rank(
        method="min", ascending=False
    )
    eligible["sharpe_rank"] = eligible["sharpe"].rank(method="min", ascending=False)
    eligible["joint_rank"] = eligible["annual_rank"] + eligible["sharpe_rank"]
    balanced = eligible.sort_values(
        ["joint_rank", "annualized_return_252", "sharpe"],
        ascending=[True, False, False],
    ).iloc[0]
    values = active[["annualized_return_252", "sharpe"]].to_numpy(float)
    pareto_mask = [
        not np.any(np.all(values >= value, axis=1) & np.any(values > value, axis=1))
        for value in values
    ]
    pareto = active.loc[pareto_mask]

    winners = {"balanced": balanced, "best_annual": best_annual, "best_sharpe": best_sharpe}
    runs = {name: run_gold_raqm(context, _params(row)) for name, row in winners.items()}
    cscv, pbo = cscv_pbo(returns, baseline, block_count=16)
    walk = expanding_walk_forward(returns, baseline)
    bootstrap = {}
    bootstrap_frames = {}
    for name, row in winners.items():
        frame, summary = paired_block_bootstrap(
            returns[str(row.name)], baseline, repetitions=5000
        )
        bootstrap[name] = summary
        bootstrap_frames[name] = frame
    reality = yearly_reality_check(returns, baseline, repetitions=5000)

    output.mkdir(parents=True, exist_ok=True)
    metrics.sort_values(["annualized_return_252", "sharpe"], ascending=False).to_csv(
        output / "candidate_metrics.csv"
    )
    pareto.sort_values(["annualized_return_252", "sharpe"], ascending=False).to_csv(
        output / "pareto_frontier.csv"
    )
    cscv.to_csv(output / "cscv_pbo.csv", index=False)
    walk.to_csv(output / "expanding_walk_forward.csv", index=False)
    for name, frame in bootstrap_frames.items():
        frame.to_csv(output / f"paired_block_bootstrap_{name}.csv", index=False)
    for name, run in runs.items():
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
    improvement = {
        "annualized_return": int(metrics["delta_annualized_return_252"].gt(0).sum()),
        "sharpe": int(metrics["delta_sharpe"].gt(0).sum()),
        "annual_and_sharpe": int(
            (metrics["delta_annualized_return_252"].gt(0) & metrics["delta_sharpe"].gt(0)).sum()
        ),
        "all_three": int(
            (
                metrics["delta_annualized_return_252"].gt(0)
                & metrics["delta_sharpe"].gt(0)
                & metrics["delta_max_drawdown"].ge(-1e-12)
            ).sum()
        ),
    }
    selected_bootstrap = bootstrap["balanced"]
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
    assessment = "high" if sum(flags.values()) >= 3 else "moderate" if any(flags.values()) else "low"
    latest_metrics = risk_adjusted_momentum_at_open(context.curves).iloc[-1]
    summary = {
        "experiment_id": config["experiment"]["id"],
        "candidate_count": int(len(metrics)),
        "active_candidate_count": int(len(active)),
        "eligible_candidate_count": int(len(eligible)),
        "pareto_count": int(len(pareto)),
        "improvement_counts": improvement,
        "balanced_id": str(balanced.name),
        "balanced": balanced.to_dict(),
        "best_annual_id": str(best_annual.name),
        "best_annual": best_annual.to_dict(),
        "best_sharpe_id": str(best_sharpe.name),
        "best_sharpe": best_sharpe.to_dict(),
        "audits": {name: run.audit for name, run in runs.items()},
        "pbo": pbo,
        "walk_forward_return_win_rate": float(walk["test_return_delta"].gt(0).mean()),
        "walk_forward_sharpe_win_rate": float(walk["test_sharpe_delta"].gt(0).mean()),
        "bootstrap": bootstrap,
        "reality_check": reality,
        "overfit_flags": flags,
        "overfit_assessment": assessment,
        "latest_metric": latest_metrics.to_dict(),
        "production_replacement": False,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )

    def row(label: str, item: pd.Series) -> str:
        return (
            f"|{label}|{item.entry_difference:.3f}|{item.exit_difference:.3f}|"
            f"{item.annualized_return_252:.2%}|{item.sharpe:.3f}|"
            f"{item.max_drawdown:.2%}|{int(item.gold_entries)}|{int(item.gold_days)}|"
        )

    report = f"""# 20日注册风险调整动量：黄金硬持有5日

## 固定口径

Gold和Defender整体NAV均调用项目注册的`risk_adjusted_quality_momentum`：20日对数收益除以
带8%年化波动率地板的20日波动，再乘Kaufman路径效率并裁剪。黄金前5个完整交易日硬持有，
第6个开盘起基础C2 Momentum优先；仅搜索入场差与退出差。

## 搜索结果

- 候选{len(metrics)}组，实际触发{len(active)}组，Pareto前沿{len(pareto)}组。
- 超过C2：年化{improvement['annualized_return']}组、Sharpe{improvement['sharpe']}组、两者同时{improvement['annual_and_sharpe']}组、三项同时{improvement['all_three']}组。

|候选|入场差|退出差|年化|Sharpe|MDD|入场|黄金日数|
|---|---:|---:|---:|---:|---:|---:|---:|
{row('折中', balanced)}
{row('最高年化', best_annual)}
{row('最高Sharpe', best_sharpe)}

## 稳健性

- PBO {float(pbo['pbo']):.1%}；walk-forward收益/Sharpe胜率{summary['walk_forward_return_win_rate']:.1%}/{summary['walk_forward_sharpe_win_rate']:.1%}。
- 折中候选bootstrap年化/Sharpe差为正概率{selected_bootstrap['annualized_return_delta_positive_probability']:.1%}/{selected_bootstrap['sharpe_delta_positive_probability']:.1%}。
- 多重试验校正p={float(reality['p_value']):.3f}；过拟合风险**{assessment.upper()}**。

## 最新指标

截至{context.calendar.max().date()}，Gold={float(latest_metrics['518880.SH']):+.3f}、Defender={float(latest_metrics['DEFENDER']):+.3f}、差值={float(latest_metrics['difference']):+.3f}。

## 结论

该实验对齐注册风险调整动量，只改变窗口为20日并搜索两个阈值；不会自动修改生产C2。
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
