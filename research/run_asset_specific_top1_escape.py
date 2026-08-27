"""Staged asset-specific escape search with multiple-testing diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from research.asset_specific_top1_escape import (
    AssetEscapePolicy,
    build_policy_grid,
    combination_search,
    run_asset_specific_escape,
    single_asset_search,
)
from research.momentum_defender_gold_override import build_gold_override_context
from research.momentum_defender_gold_override_overfit import (
    cscv_pbo,
    expanding_walk_forward,
    full_metrics,
    paired_block_bootstrap,
    yearly_reality_check,
)
from research.momentum_defender_occam import MOMENTUM_ASSETS
from research.momentum_top1_defender_escape import all_metrics_at_open
from research.standard_report import generate_standard_report


DEFAULT_CONFIG = Path("research/configs/asset_specific_top1_escape_search.yaml")
DEFAULT_OUTPUT = Path("experiments/20260823_asset_specific_top1_escape")


def _policy_from_id(policy_id: str, policies: list[AssetEscapePolicy]) -> AssetEscapePolicy:
    return next(policy for policy in policies if policy.policy_id() == policy_id)


def run_experiment(root: Path, config_path: Path, output: Path) -> dict[str, object]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    context = build_gold_override_context(root)
    baseline = context.integrated.result.simulated["return"].astype(float)
    policy_grid = build_policy_grid(config["single_asset_policy_grid"])
    metric_frames = {
        (policy.metric, policy.window): all_metrics_at_open(
            context.curves, policy.metric, policy.window
        )
        for policy in policy_grid
    }

    single_records = []
    single_return_frames = []
    top_options: dict[str, list[AssetEscapePolicy | None]] = {}
    staged = config["staged_selection"]
    for asset in MOMENTUM_ASSETS:
        records, returns = single_asset_search(
            context, asset, policy_grid, metric_frames
        )
        candidate_metrics = full_metrics(returns, baseline).join(records)
        single_records.append(candidate_metrics)
        single_return_frames.append(returns)
        eligible = candidate_metrics.loc[
            candidate_metrics["escape_entries"].ge(
                int(staged["minimum_single_asset_entries"])
            )
            & candidate_metrics["escape_days"].ge(
                int(staged["minimum_single_asset_days"])
            )
        ]
        top = eligible.sort_values(
            ["annualized_return_252", "sharpe"], ascending=False
        ).head(int(staged["policies_per_asset"]))
        options = [
            _policy_from_id(str(row["metric"]) + "_w" + str(int(row["window"])) +
                f"_en{float(row['entry_difference']):+.4f}_ex{float(row['exit_difference']):+.4f}_abs{float(row['absolute_minimum']):+.4f}_h{int(row['min_hold_days'])}",
                policy_grid)
            for _, row in top.iterrows()
        ]
        # Preserve distinct policy IDs while collapsing equivalent top rows.
        unique_options = {policy.policy_id(): policy for policy in options}
        top_options[asset] = [None, *unique_options.values()]

    single_metrics = pd.concat(single_records)
    single_returns = pd.concat(single_return_frames, axis=1)
    combo_records, combo_returns, policy_sets = combination_search(
        context, top_options, metric_frames
    )
    combo_metrics = full_metrics(combo_returns, baseline).join(combo_records)

    periods = {
        label: (
            pd.Timestamp(values[0]),
            pd.Timestamp(values[1]),
        )
        for label, values in config["periods"].items()
    }
    for label, (start, end) in periods.items():
        period = full_metrics(
            combo_returns.loc[start:end], baseline.loc[start:end]
        )
        for field in (
            "annualized_return_252",
            "sharpe",
            "max_drawdown",
            "delta_annualized_return_252",
            "delta_sharpe",
            "delta_max_drawdown",
        ):
            combo_metrics[f"{label}_{field}"] = period[field]
    combo_metrics["worst_split_sharpe"] = combo_metrics[
        ["development_sharpe", "validation_sharpe", "recent_sharpe"]
    ].min(axis=1)

    selected = combo_metrics.sort_values(
        ["annualized_return_252", "sharpe"], ascending=False
    ).iloc[0]
    selected_id = str(selected.name)
    selected_policies = policy_sets[selected_id]
    selected_run = run_asset_specific_escape(
        context, selected_policies, metric_frames=metric_frames
    )

    cscv, pbo = cscv_pbo(combo_returns, baseline, block_count=16)
    walk = expanding_walk_forward(combo_returns, baseline)
    bootstrap, bootstrap_summary = paired_block_bootstrap(
        combo_returns[selected_id], baseline, repetitions=5000
    )
    all_tested_returns = pd.concat([single_returns, combo_returns], axis=1)
    all_tested_returns = all_tested_returns.loc[
        :, ~all_tested_returns.columns.duplicated()
    ]
    reality = yearly_reality_check(
        all_tested_returns, baseline, repetitions=5000
    )

    output.mkdir(parents=True, exist_ok=True)
    single_metrics.sort_values(
        ["annualized_return_252", "sharpe"], ascending=False
    ).to_csv(output / "single_asset_candidates.csv")
    combo_metrics.sort_values(
        ["annualized_return_252", "sharpe"], ascending=False
    ).to_csv(output / "combination_candidates.csv")
    selected_run.state.join(selected_run.daily, rsuffix="_execution").to_csv(
        output / "daily_selected.csv"
    )
    cscv.to_csv(output / "cscv_pbo.csv", index=False)
    walk.to_csv(output / "expanding_walk_forward.csv", index=False)
    bootstrap.to_csv(output / "paired_block_bootstrap.csv", index=False)
    (output / "strategy_config.yaml").write_text(
        config_path.read_text(encoding="utf-8"), encoding="utf-8"
    )
    selected_config = {
        "strategy": {
            "id": "c2_asset_specific_top1_escape_best_annual",
            "status": "research_candidate_not_promoted",
            "evidence_status": config["experiment"]["evidence_status"],
        },
        "objective": config["experiment"]["objective"],
        "policies": {
            asset: policy.__dict__ if policy is not None else None
            for asset, policy in selected_policies.items()
        },
        "checkpoint": {
            "annualized_return_252": float(selected["annualized_return_252"]),
            "sharpe": float(selected["sharpe"]),
            "max_drawdown": float(selected["max_drawdown"]),
            "escape_entries": int(selected["escape_entries"]),
            "escape_days": int(selected["escape_days"]),
        },
        "decision": {"production_replacement": False},
    }
    (output / "selected_config.yaml").write_text(
        yaml.safe_dump(selected_config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    generate_standard_report(
        selected_run.daily["return"],
        baseline,
        "Current Integrated C2",
        output / "selected_vs_current_c2.html",
        selected_config,
    )

    improvement_counts = {
        "annualized_return": int(combo_metrics["delta_annualized_return_252"].gt(0).sum()),
        "sharpe": int(combo_metrics["delta_sharpe"].gt(0).sum()),
        "max_drawdown": int(combo_metrics["delta_max_drawdown"].ge(-1e-12).sum()),
        "all_three": int(
            (
                combo_metrics["delta_annualized_return_252"].gt(0)
                & combo_metrics["delta_sharpe"].gt(0)
                & combo_metrics["delta_max_drawdown"].ge(-1e-12)
            ).sum()
        ),
    }
    overfit_flags = {
        "pbo_above_half": float(pbo["pbo"]) >= 0.5,
        "reality_not_significant": float(reality["p_value"]) >= 0.10,
        "bootstrap_return_below_90pct": float(
            bootstrap_summary["annualized_return_delta_positive_probability"]
        ) < 0.90,
        "bootstrap_sharpe_below_90pct": float(
            bootstrap_summary["sharpe_delta_positive_probability"]
        ) < 0.90,
        "walk_forward_return_win_below_half": float(
            walk["test_return_delta"].gt(0).mean()
        ) < 0.5,
    }
    assessment = (
        "high"
        if sum(overfit_flags.values()) >= 3
        else "moderate" if any(overfit_flags.values()) else "low"
    )
    summary = {
        "experiment_id": config["experiment"]["id"],
        "single_asset_candidate_count": int(len(single_metrics)),
        "combination_candidate_count": int(len(combo_metrics)),
        "all_tested_unique_returns": int(len(all_tested_returns.columns)),
        "selected_id": selected_id,
        "selected": selected.to_dict(),
        "selected_policies": {
            asset: policy.__dict__ if policy else None
            for asset, policy in selected_policies.items()
        },
        "improvement_counts": improvement_counts,
        "pbo": pbo,
        "walk_forward_return_win_rate": float(walk["test_return_delta"].gt(0).mean()),
        "walk_forward_sharpe_win_rate": float(walk["test_sharpe_delta"].gt(0).mean()),
        "bootstrap": bootstrap_summary,
        "reality_check": reality,
        "overfit_flags": overfit_flags,
        "overfit_assessment": assessment,
        "production_replacement": False,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )

    policy_lines = "\n".join(
        f"- {asset}: `{policy.policy_id()}`" if policy else f"- {asset}: 禁用"
        for asset, policy in selected_policies.items()
    )
    report = f"""# C2资产专用Momentum Top1逃生

## 分阶段搜索

- 单资产候选：{len(single_metrics)}组。
- 每只资产保留3个最高年化参数和禁用选项，组合候选：{len(combo_metrics)}组。
- 全部测试的唯一收益路径：{len(all_tested_returns.columns)}条。

## 最高年化组合

{policy_lines}

- 年化{float(selected['annualized_return_252']):.2%}（相对C2 {float(selected['delta_annualized_return_252']):+.2%}）。
- Sharpe {float(selected['sharpe']):.3f}（{float(selected['delta_sharpe']):+.3f}）。
- MDD {float(selected['max_drawdown']):.2%}（{float(selected['delta_max_drawdown']):+.2%}）。
- 逃生{int(selected['escape_entries'])}次、{int(selected['escape_days'])}日。
- 三段Sharpe：development {float(selected['development_sharpe']):.3f}、validation {float(selected['validation_sharpe']):.3f}、recent {float(selected['recent_sharpe']):.3f}。

## 候选族与稳健性

- 组合中超过C2：年化{improvement_counts['annualized_return']}组、Sharpe{improvement_counts['sharpe']}组、MDD不恶化{improvement_counts['max_drawdown']}组、三项同时{improvement_counts['all_three']}组。
- PBO {float(pbo['pbo']):.1%}；walk-forward收益/Sharpe胜率{summary['walk_forward_return_win_rate']:.1%}/{summary['walk_forward_sharpe_win_rate']:.1%}。
- Bootstrap年化差为正概率{bootstrap_summary['annualized_return_delta_positive_probability']:.1%}，Sharpe差为正概率{bootstrap_summary['sharpe_delta_positive_probability']:.1%}。
- 对全部测试路径做年度块多重试验校正：p={float(reality['p_value']):.3f}。
- 综合过拟合风险：**{assessment.upper()}**。

## 结论

该实验允许资产拥有不同阈值和机制，并明确以年化为主目标。结果仍属于分阶段回溯搜索，
不会自动修改生产C2；是否有可用提升必须以统计审计为准。
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
