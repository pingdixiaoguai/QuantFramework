"""Run multiple-testing and temporal overfitting audits for Gold Override."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from research.momentum_defender_gold_override import build_gold_override_context
from research.momentum_defender_gold_override_overfit import (
    collect_candidate_returns,
    cscv_pbo,
    expanding_walk_forward,
    full_metrics,
    leave_one_year_selection,
    paired_block_bootstrap,
    yearly_reality_check,
)


FIRST_GRID = Path("research/configs/momentum_defender_gold_override_search.yaml")
REFINEMENT_GRID = Path(
    "research/configs/momentum_defender_gold_override_refinement.yaml"
)
FINAL_CONFIG = Path("research/configs/momentum_defender_gold_override_best.yaml")
DEFAULT_OUTPUT = Path("experiments/20260823_momentum_defender_gold_override")


def _yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def run_audit(root: Path, output: Path) -> dict[str, object]:
    first = _yaml(FIRST_GRID)
    refinement = _yaml(REFINEMENT_GRID)
    final = _yaml(FINAL_CONFIG)
    selected_id = final["strategy"]["id"]
    selected_candidate_id = (
        f"risk_adjusted_return_w5_en+0.6000_ex-0.4000_h7"
    )
    context = build_gold_override_context(root)
    metadata, returns = collect_candidate_returns(
        context, [first["grid"], refinement["grid"]]
    )
    if selected_candidate_id not in returns:
        raise AssertionError("final candidate is absent from searched family")
    baseline = context.integrated.result.simulated["return"].astype(float)
    metrics = full_metrics(returns, baseline).join(metadata)

    selected = metrics.loc[selected_candidate_id]
    neighborhood = metrics.loc[
        metrics["metric"].eq("risk_adjusted_return")
        & metrics["window"].isin([5, 7])
        & metrics["entry_threshold"].between(0.50, 0.70)
        & metrics["exit_threshold"].between(-0.40, -0.20)
        & metrics["min_gold_hold_days"].between(5, 10)
    ].copy()
    neighborhood_summary = {
        "candidate_count": int(len(neighborhood)),
        "annualized_return_improvement_rate": float(
            neighborhood["delta_annualized_return_252"].gt(0).mean()
        ),
        "sharpe_improvement_rate": float(
            neighborhood["delta_sharpe"].gt(0).mean()
        ),
        "mdd_nonworsening_rate": float(
            neighborhood["delta_max_drawdown"].ge(-1e-12).mean()
        ),
        "all_three_nonworsening_rate": float(
            (
                neighborhood["delta_annualized_return_252"].ge(0)
                & neighborhood["delta_sharpe"].ge(0)
                & neighborhood["delta_max_drawdown"].ge(-1e-12)
            ).mean()
        ),
        "median_annualized_return_delta": float(
            neighborhood["delta_annualized_return_252"].median()
        ),
        "median_sharpe_delta": float(neighborhood["delta_sharpe"].median()),
    }

    cscv, pbo_summary = cscv_pbo(returns, baseline, block_count=16)
    walk_forward = expanding_walk_forward(returns, baseline)
    leave_year = leave_one_year_selection(returns, baseline)
    bootstrap, bootstrap_summary = paired_block_bootstrap(
        returns[selected_candidate_id], baseline
    )
    reality = yearly_reality_check(returns, baseline)

    selected_frequency = {
        "cscv_selected_rate": float(
            cscv["selected_candidate"].eq(selected_candidate_id).mean()
        ),
        "walk_forward_selected_rate": float(
            walk_forward["selected_candidate"].eq(selected_candidate_id).mean()
        ),
        "leave_one_year_selected_rate": float(
            leave_year["selected_candidate"].eq(selected_candidate_id).mean()
        ),
    }
    temporal_summary = {
        "walk_forward_folds": int(len(walk_forward)),
        "walk_forward_return_win_rate": float(
            walk_forward["test_return_delta"].gt(0).mean()
        ),
        "walk_forward_sharpe_win_rate": float(
            walk_forward["test_sharpe_delta"].gt(0).mean()
        ),
        "leave_one_year_return_win_rate": float(
            leave_year["test_return_delta"].gt(0).mean()
        ),
        "leave_one_year_sharpe_win_rate": float(
            leave_year["test_sharpe_delta"].gt(0).mean()
        ),
    }

    event_audit_path = output / "final_candidate_audit.json"
    event_audit = json.loads(event_audit_path.read_text(encoding="utf-8"))
    high_risk_flags = {
        "pbo_above_half": float(pbo_summary["pbo"]) >= 0.5,
        "reality_check_not_significant_10pct": float(reality["p_value"]) >= 0.10,
        "bootstrap_return_positive_below_90pct": float(
            bootstrap_summary["annualized_return_delta_positive_probability"]
        ) < 0.90,
        "bootstrap_sharpe_positive_below_90pct": float(
            bootstrap_summary["sharpe_delta_positive_probability"]
        ) < 0.90,
        "walk_forward_return_win_below_half": float(
            temporal_summary["walk_forward_return_win_rate"]
        ) < 0.5,
    }
    if sum(high_risk_flags.values()) >= 3:
        assessment = "high"
    elif any(high_risk_flags.values()):
        assessment = "moderate"
    else:
        assessment = "low"

    output.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(output / "overfit_candidate_metadata.csv")
    metrics.to_csv(output / "overfit_full_metrics.csv")
    neighborhood.to_csv(output / "overfit_selected_neighborhood.csv")
    cscv.to_csv(output / "overfit_cscv_pbo.csv", index=False)
    walk_forward.to_csv(output / "overfit_expanding_walk_forward.csv", index=False)
    leave_year.to_csv(output / "overfit_leave_one_year.csv", index=False)
    bootstrap.to_csv(output / "overfit_paired_block_bootstrap.csv", index=False)

    summary = {
        "strategy_id": selected_id,
        "candidate_id": selected_candidate_id,
        "searched_unique_candidates": int(len(returns.columns)),
        "observations": int(len(returns)),
        "baseline_parity_max_abs_error": context.baseline_parity_max_abs_error,
        "selected_full_metrics": selected.to_dict(),
        "parameter_neighborhood": neighborhood_summary,
        "pbo": pbo_summary,
        "paired_block_bootstrap": bootstrap_summary,
        "year_block_reality_check": reality,
        "temporal_selection": temporal_summary,
        "selection_frequency": selected_frequency,
        "event_audit": {
            "event_count": event_audit["event_count"],
            "positive_event_count": event_audit["positive_event_count"],
            "negative_event_count": event_audit["negative_event_count"],
            "top_two_positive_event_share": event_audit[
                "top_two_positive_event_share"
            ],
            "leave_one_event_min_annualized_return_252": event_audit[
                "leave_one_event_min_annualized_return_252"
            ],
            "leave_one_event_min_sharpe": event_audit[
                "leave_one_event_min_sharpe"
            ],
        },
        "high_risk_flags": high_risk_flags,
        "overfit_risk_assessment": assessment,
        "production_replacement": False,
    }
    (output / "overfit_audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )

    report = f"""# C2 Gold Override 过拟合审计

## 审计范围

- 重建首轮与局部细化的全部唯一候选：{len(returns.columns)}组，逐日{len(returns)}个观测。
- 最终候选：`{selected_candidate_id}`。
- 关闭覆盖复现当前C2的最大日收益误差：{context.baseline_parity_max_abs_error:.3e}。

## 参数邻域

- 邻域候选：{neighborhood_summary['candidate_count']}组。
- 年化提高比例：{neighborhood_summary['annualized_return_improvement_rate']:.1%}；Sharpe提高比例：{neighborhood_summary['sharpe_improvement_rate']:.1%}；MDD不恶化比例：{neighborhood_summary['mdd_nonworsening_rate']:.1%}；三项均不恶化比例：{neighborhood_summary['all_three_nonworsening_rate']:.1%}。
- 邻域中位年化变化：{neighborhood_summary['median_annualized_return_delta']:+.2%}；中位Sharpe变化：{neighborhood_summary['median_sharpe_delta']:+.3f}。

## CSCV / PBO

- 16个顺序块、{pbo_summary['split_count']}个对称训练/测试划分。
- PBO：{pbo_summary['pbo']:.1%}；样本外排名中位分位：{pbo_summary['median_test_rank_percentile']:.1%}。
- 每折样本内冠军在测试集击败当前C2的比例：{pbo_summary['selected_beats_baseline_rate']:.1%}。

## 时间外推

- 扩展式walk-forward：收益胜率{temporal_summary['walk_forward_return_win_rate']:.1%}、Sharpe胜率{temporal_summary['walk_forward_sharpe_win_rate']:.1%}。
- 留一年选择：收益胜率{temporal_summary['leave_one_year_return_win_rate']:.1%}、Sharpe胜率{temporal_summary['leave_one_year_sharpe_win_rate']:.1%}。
- 最终参数被CSCV、walk-forward、留一年重新选中的频率分别为{selected_frequency['cscv_selected_rate']:.1%}、{selected_frequency['walk_forward_selected_rate']:.1%}、{selected_frequency['leave_one_year_selected_rate']:.1%}。

## Bootstrap与多重试验

- 20日成对分块bootstrap {bootstrap_summary['repetitions']}次：年化差为正概率{bootstrap_summary['annualized_return_delta_positive_probability']:.1%}，95%区间[{bootstrap_summary['annualized_return_delta_ci_lower']:+.2%}, {bootstrap_summary['annualized_return_delta_ci_upper']:+.2%}]。
- Sharpe差为正概率{bootstrap_summary['sharpe_delta_positive_probability']:.1%}，95%区间[{bootstrap_summary['sharpe_delta_ci_lower']:+.3f}, {bootstrap_summary['sharpe_delta_ci_upper']:+.3f}]。
- 以年度为相关块的White式最大均值校正：p={reality['p_value']:.3f}，观测最优候选为`{reality['observed_best_candidate']}`。

## 事件审计

- 9次事件中6正3负，前两大正事件占正贡献{event_audit['top_two_positive_event_share']:.1%}。
- 删除任一事件后最低年化{event_audit['leave_one_event_min_annualized_return_252']:.2%}、最低Sharpe{event_audit['leave_one_event_min_sharpe']:.3f}。

## 结论

综合过拟合风险评级：**{assessment.upper()}**。高风险标记：{', '.join(name for name, value in high_risk_flags.items() if value) or '无'}。

该评级针对“搜索后提升能否外推”，不是对实现正确性的否定。无论评级结果如何，现有证据仍是回溯的，生产晋升需要独立前瞻期。
"""
    (output / "overfit_audit_report.md").write_text(report, encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    summary = run_audit(args.root.resolve(), args.output)
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
