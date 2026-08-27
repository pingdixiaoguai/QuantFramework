"""Validate the unified Momentum Top-1 versus Defender escape gate."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import pandas as pd
import yaml

from research.momentum_defender_gold_override import build_gold_override_context
from research.momentum_defender_gold_override_overfit import (
    cscv_pbo,
    expanding_walk_forward,
    full_metrics,
    paired_block_bootstrap,
    yearly_reality_check,
)
from research.momentum_top1_defender_escape import (
    Top1EscapeParams,
    collect_candidate_returns,
    run_top1_escape,
)
from research.standard_report import generate_standard_report


DEFAULT_CONFIG = Path("research/configs/momentum_top1_defender_escape.yaml")
DEFAULT_OUTPUT = Path("experiments/20260823_momentum_top1_defender_escape")


def _params(row: pd.Series | dict) -> Top1EscapeParams:
    return Top1EscapeParams(
        metric=str(row["metric"]),
        window=int(row["window"]),
        entry_difference=float(row["entry_difference"]),
        exit_difference=float(row["exit_difference"]),
        absolute_minimum=float(row.get("absolute_minimum", 0.0)),
        min_escape_hold_days=int(row["min_escape_hold_days"]),
    )


def run_experiment(root: Path, config_path: Path, output: Path) -> dict[str, object]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    context = build_gold_override_context(root)
    baseline = context.integrated.result.simulated["return"].astype(float)
    metadata, returns = collect_candidate_returns(context, config["grid"])
    metrics = full_metrics(returns, baseline).join(metadata)
    periods = {
        label: (date.fromisoformat(values[0]), date.fromisoformat(values[1]))
        for label, values in config["periods"].items()
    }
    for label, (start, end) in periods.items():
        period = full_metrics(
            returns.loc[pd.Timestamp(start) : pd.Timestamp(end)],
            baseline.loc[pd.Timestamp(start) : pd.Timestamp(end)],
        )
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

    preregistered_params = _params(config["pre_registered"])
    preregistered_id = preregistered_params.candidate_id()
    if preregistered_id not in metrics.index:
        raise AssertionError("pre-registered candidate missing from grid")
    selection = config["selection"]
    eligible = metrics.loc[
        metrics["escape_entries"].ge(int(selection["minimum_escape_entries"]))
        & metrics["escape_days"].ge(int(selection["minimum_escape_days"]))
    ].copy()
    best_annual = eligible.sort_values(
        ["annualized_return_252", "sharpe"], ascending=False
    ).iloc[0]
    best_sharpe = eligible.sort_values(
        ["sharpe", "annualized_return_252"], ascending=False
    ).iloc[0]
    best_mdd = eligible.sort_values(
        ["max_drawdown", "sharpe"], ascending=False
    ).iloc[0]
    robust = eligible.sort_values(
        ["worst_split_sharpe", "sharpe", "annualized_return_252"], ascending=False
    ).iloc[0]
    tolerance = selection["balanced_tolerances"]
    balanced = eligible.loc[
        eligible["delta_annualized_return_252"].ge(
            float(tolerance["annualized_return_delta"])
        )
        & eligible["delta_sharpe"].ge(float(tolerance["sharpe_delta"]))
        & eligible["delta_max_drawdown"].ge(
            float(tolerance["max_drawdown_delta"])
        )
        & eligible[
            [
                "delta_annualized_return_252",
                "delta_sharpe",
                "delta_max_drawdown",
            ]
        ].gt(0).any(axis=1)
    ]
    selected = (
        balanced.sort_values(
            ["worst_split_sharpe", "sharpe", "annualized_return_252"],
            ascending=False,
        ).iloc[0]
        if not balanced.empty
        else robust
    )

    selected_id = str(selected.name)
    selected_run = run_top1_escape(context, _params(selected))
    preregistered_run = run_top1_escape(context, preregistered_params)
    cscv, pbo = cscv_pbo(returns, baseline, block_count=16)
    walk = expanding_walk_forward(returns, baseline)
    bootstrap, bootstrap_summary = paired_block_bootstrap(
        returns[selected_id], baseline, repetitions=5000
    )
    reality = yearly_reality_check(returns, baseline, repetitions=5000)

    improvement_counts = {
        "annualized_return": int(eligible["delta_annualized_return_252"].gt(0).sum()),
        "sharpe": int(eligible["delta_sharpe"].gt(0).sum()),
        "max_drawdown": int(eligible["delta_max_drawdown"].ge(-1e-12).sum()),
        "all_three": int(
            (
                eligible["delta_annualized_return_252"].gt(0)
                & eligible["delta_sharpe"].gt(0)
                & eligible["delta_max_drawdown"].ge(-1e-12)
            ).sum()
        ),
    }
    overfit_flags = {
        "pbo_above_half": float(pbo["pbo"]) >= 0.5,
        "reality_check_not_significant_10pct": float(reality["p_value"]) >= 0.10,
        "bootstrap_return_probability_below_90pct": float(
            bootstrap_summary["annualized_return_delta_positive_probability"]
        ) < 0.90,
        "bootstrap_sharpe_probability_below_90pct": float(
            bootstrap_summary["sharpe_delta_positive_probability"]
        ) < 0.90,
        "walk_forward_return_win_below_half": float(
            walk["test_return_delta"].gt(0).mean()
        ) < 0.5,
    }
    overfit_assessment = (
        "high"
        if sum(overfit_flags.values()) >= 3
        else "moderate" if any(overfit_flags.values()) else "low"
    )

    output.mkdir(parents=True, exist_ok=True)
    metrics.sort_values(["worst_split_sharpe", "sharpe"], ascending=False).to_csv(
        output / "candidate_metrics.csv"
    )
    selected_run.state.join(selected_run.daily, rsuffix="_execution").to_csv(
        output / "daily_selected.csv"
    )
    preregistered_run.state.join(
        preregistered_run.daily, rsuffix="_execution"
    ).to_csv(output / "daily_pre_registered.csv")
    cscv.to_csv(output / "cscv_pbo.csv", index=False)
    walk.to_csv(output / "expanding_walk_forward.csv", index=False)
    bootstrap.to_csv(output / "paired_block_bootstrap.csv", index=False)
    (output / "strategy_config.yaml").write_text(
        config_path.read_text(encoding="utf-8"), encoding="utf-8"
    )
    generate_standard_report(
        selected_run.daily["return"],
        baseline,
        "Current Integrated C2",
        output / "selected_vs_current_c2.html",
        {
            "strategy_name": config["experiment"]["id"],
            "selected_params": selected_run.params.__dict__,
            "evidence_status": config["experiment"]["evidence_status"],
        },
    )
    generate_standard_report(
        preregistered_run.daily["return"],
        baseline,
        "Current Integrated C2",
        output / "pre_registered_vs_current_c2.html",
        {
            "strategy_name": "pre_registered_unified_top1_escape",
            "params": preregistered_run.params.__dict__,
        },
    )

    summary = {
        "experiment_id": config["experiment"]["id"],
        "candidate_count": int(len(metrics)),
        "eligible_candidate_count": int(len(eligible)),
        "baseline_parity_max_abs_error": context.baseline_parity_max_abs_error,
        "pre_registered": metrics.loc[preregistered_id].to_dict(),
        "selected": selected.to_dict(),
        "selected_id": selected_id,
        "best_annual": best_annual.to_dict(),
        "best_sharpe": best_sharpe.to_dict(),
        "best_mdd": best_mdd.to_dict(),
        "best_robust": robust.to_dict(),
        "improvement_counts": improvement_counts,
        "pbo": pbo,
        "walk_forward_return_win_rate": float(walk["test_return_delta"].gt(0).mean()),
        "walk_forward_sharpe_win_rate": float(walk["test_sharpe_delta"].gt(0).mean()),
        "bootstrap": bootstrap_summary,
        "reality_check": reality,
        "overfit_flags": overfit_flags,
        "overfit_assessment": overfit_assessment,
        "production_replacement": False,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )

    pre = metrics.loc[preregistered_id]
    report = f"""# C2 + 统一Momentum Top1逃生门控

## 规则

基础C2为Momentum时保持原Momentum Top1。基础C2为Defender时，四只ETF共用同一X指标、
窗口和阈值；只有当前Momentum Top1自身X>0且相对Defender整体X显著领先，才允许下一开盘
临时离开Defender。没有任何资产专用参数。

## 预注册方案

20日quality momentum、差值>0入场/≤0退出、最短持有1日：年化
{float(pre['annualized_return_252']):.2%}、Sharpe {float(pre['sharpe']):.3f}、MDD
{float(pre['max_drawdown']):.2%}；逃生{int(pre['escape_entries'])}次、{int(pre['escape_days'])}日。

## 小网格结果

- 共{len(metrics)}组，合格{len(eligible)}组。
- 超过基线：年化{improvement_counts['annualized_return']}组、Sharpe{improvement_counts['sharpe']}组、MDD不恶化{improvement_counts['max_drawdown']}组、三项同时{improvement_counts['all_three']}组。
- 推荐折中：`{selected_id}`。
- 参数：{selected['metric']}，{int(selected['window'])}日，入场差>{float(selected['entry_difference']):.4f}，退出差≤{float(selected['exit_difference']):.4f}，最短持有{int(selected['min_escape_hold_days'])}日。
- 全样本：年化{float(selected['annualized_return_252']):.2%}、Sharpe{float(selected['sharpe']):.3f}、MDD{float(selected['max_drawdown']):.2%}。
- 相对基线：年化{float(selected['delta_annualized_return_252']):+.2%}、Sharpe{float(selected['delta_sharpe']):+.3f}、MDD{float(selected['delta_max_drawdown']):+.2%}。
- 逃生资产日数：沪深300 {int(selected['escape_days_510300.SH'])}、创业板 {int(selected['escape_days_159915.SZ'])}、纳指 {int(selected['escape_days_513100.SH'])}、黄金 {int(selected['escape_days_518880.SH'])}。

## 过拟合审计

- PBO {float(pbo['pbo']):.1%}，测试排名中位{float(pbo['median_test_rank_percentile']):.1%}。
- 扩展式walk-forward收益/Sharpe胜率分别为{summary['walk_forward_return_win_rate']:.1%}/{summary['walk_forward_sharpe_win_rate']:.1%}。
- 分块bootstrap年化差为正概率{bootstrap_summary['annualized_return_delta_positive_probability']:.1%}，95%区间[{bootstrap_summary['annualized_return_delta_ci_lower']:+.2%}, {bootstrap_summary['annualized_return_delta_ci_upper']:+.2%}]；Sharpe差为正概率{bootstrap_summary['sharpe_delta_positive_probability']:.1%}。
- 年度块多重试验校正p={float(reality['p_value']):.3f}。
- 综合过拟合风险：**{overfit_assessment.upper()}**。

## 结论

统一Top1逃生解决了“只看510300”的结构盲点，但是否值得采用必须同时看核心指标和外推审计。
本实验不会自动替换生产C2。
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
