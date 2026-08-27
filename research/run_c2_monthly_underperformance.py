"""Generate monthly underperformance attribution and simple counterfactuals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.c2_monthly_underperformance import build_analysis
from research.standard_report import generate_standard_report


DEFAULT_OUTPUT = Path("experiments/20260823_c2_monthly_underperformance")


def run_report(root: Path, output: Path) -> dict[str, object]:
    (
        integrated,
        daily,
        monthly,
        causes,
        counterfactuals,
        counterfactual_monthly,
    ) = build_analysis(root)
    losing = monthly.loc[monthly["underperformed"]].copy()
    winning = monthly.loc[~monthly["underperformed"]].copy()
    losing["loss_log_magnitude"] = -np.log1p(losing["relative_return"])
    total_loss = float(losing["loss_log_magnitude"].sum())
    top_five_share = (
        float(losing.nlargest(5, "loss_log_magnitude")["loss_log_magnitude"].sum())
        / total_loss
        if total_loss > 0.0
        else 0.0
    )
    cause_counts = losing["primary_cause"].value_counts()
    asset_counts = losing[
        "dominant_momentum_asset_during_defender"
    ].value_counts()
    negative_total = -float(causes["negative_log_excess"].sum())
    cause_loss_shares = {
        category: (
            -float(row["negative_log_excess"]) / negative_total
            if negative_total > 0.0
            else 0.0
        )
        for category, row in causes.iterrows()
    }

    baseline = counterfactuals.loc["defender_exit_lock_30"]
    alternatives = counterfactuals.drop(index="defender_exit_lock_30").copy()
    for field in ("annualized_return_252", "sharpe", "max_drawdown"):
        alternatives[f"delta_{field}"] = alternatives[field] - float(baseline[field])
    best_exit_lock_name = (
        alternatives.loc[
            alternatives.index.str.startswith("defender_exit_lock_")
        ]["sharpe"].idxmax()
    )
    best_exit_lock = alternatives.loc[best_exit_lock_name]

    output.mkdir(parents=True, exist_ok=True)
    daily.to_csv(output / "daily_excess_attribution.csv")
    monthly.to_csv(output / "monthly_returns.csv")
    losing.sort_values("relative_return").to_csv(output / "losing_months.csv")
    causes.to_csv(output / "cause_summary.csv")
    counterfactuals.to_csv(output / "counterfactual_metrics.csv")
    counterfactual_monthly.to_csv(output / "counterfactual_monthly.csv", index=False)

    best_returns = counterfactual_monthly.loc[
        counterfactual_monthly["variant"].eq(best_exit_lock_name)
    ]
    # Regenerate the daily best variant from the already returned integrated
    # analysis by matching the stored counterfactual name.
    from research.c2_monthly_underperformance import run_counterfactuals

    best_variant = next(
        item for item in run_counterfactuals(integrated) if item.name == best_exit_lock_name
    )
    generate_standard_report(
        best_variant.simulated["return"],
        integrated.result.simulated["return"],
        "Current Integrated C2",
        output / "best_exit_lock_vs_current_c2.html",
        {
            "strategy_name": best_exit_lock_name,
            "experiment": "C2 monthly underperformance attribution",
        },
    )
    del best_returns

    summary = {
        "sample_start": monthly.iloc[0]["start"],
        "sample_end": monthly.iloc[-1]["end"],
        "months": int(len(monthly)),
        "underperform_months": int(len(losing)),
        "underperform_rate": float(len(losing) / len(monthly)),
        "average_losing_month_relative_return": float(losing["relative_return"].mean()),
        "median_losing_month_relative_return": float(losing["relative_return"].median()),
        "top_five_loss_share": top_five_share,
        "momentum_positive_in_losing_months": int(losing["momentum_return"].gt(0).sum()),
        "mostly_defender_losing_months": int(losing["defender_day_share"].ge(0.5).sum()),
        "losing_months_with_exit_lock_delay": int(losing["exit_lock_delay_days"].gt(0).sum()),
        "losing_months_with_emergency_cap_days": int(losing["emergency_cap_days"].gt(0).sum()),
        "primary_cause_counts": {str(key): int(value) for key, value in cause_counts.items()},
        "dominant_momentum_asset_counts": {
            str(key): int(value) for key, value in asset_counts.items()
        },
        "cause_negative_loss_shares": cause_loss_shares,
        "worst_months": losing.nsmallest(10, "relative_return")[
            ["c2_return", "momentum_return", "relative_return", "primary_cause"]
        ].reset_index().to_dict("records"),
        "best_exit_lock_counterfactual": {
            "variant": best_exit_lock_name,
            **{key: value for key, value in best_exit_lock.to_dict().items()},
        },
        "no_emergency_counterfactual": counterfactuals.loc[
            "no_emergency_cap"
        ].to_dict(),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )

    worst_lines = "\n".join(
        f"|{month}|{row.c2_return:+.2%}|{row.momentum_return:+.2%}|{row.relative_return:+.2%}|{row.primary_cause}|{row.dominant_momentum_asset_during_defender}|"
        for month, row in losing.nsmallest(12, "relative_return").iterrows()
    )
    cause_lines = "\n".join(
        f"|{category}|{int(row.days)}|{row.net_log_excess:+.4f}|{cause_loss_shares[category]:.1%}|"
        for category, row in causes.sort_values("negative_log_excess").iterrows()
    )
    report = f"""# C2相对原Momentum的月度跑输审计

## 有多少个月跑输

样本为{summary['sample_start']}至{summary['sample_end']}，共{summary['months']}个自然月；
C2跑输原Momentum {summary['underperform_months']}个月，占{summary['underperform_rate']:.1%}。
跑输月平均相对收益{summary['average_losing_month_relative_return']:+.2%}，中位数
{summary['median_losing_month_relative_return']:+.2%}；最差5个月占全部跑输log损失
{summary['top_five_loss_share']:.1%}。

## 最差月份

|月份|C2|原Momentum|相对收益|主要原因|Defender期间Momentum主标的|
|---|---:|---:|---:|---|---|
{worst_lines}

## 逐日原因贡献

|原因|天数|净log excess|负向日损失占比|
|---|---:|---:|---:|
{cause_lines}

- 跑输月份中，Momentum本身上涨的有{summary['momentum_positive_in_losing_months']}个月；C2至少半个月处于Defender的有{summary['mostly_defender_losing_months']}个月。
- 出现Defender退出锁延迟的跑输月有{summary['losing_months_with_exit_lock_delay']}个；出现紧急cap维持Defender的有{summary['losing_months_with_emergency_cap_days']}个。
- Defender期间原Momentum主标的分布：{summary['dominant_momentum_asset_counts']}。

## 简单反事实

只缩短Defender→Momentum退出锁、保持Momentum→Defender 30日和紧急cap不变。Sharpe最高的
简单版本是`{best_exit_lock_name}`：年化{float(best_exit_lock['annualized_return_252']):.2%}
（相对基线{float(best_exit_lock['delta_annualized_return_252']):+.2%}）、Sharpe
{float(best_exit_lock['sharpe']):.3f}（{float(best_exit_lock['delta_sharpe']):+.3f}）、MDD
{float(best_exit_lock['max_drawdown']):.2%}（{float(best_exit_lock['delta_max_drawdown']):+.2%}），
跑输月份{int(best_exit_lock['underperform_months'])}个。

移除紧急cap后的年化为{float(counterfactuals.loc['no_emergency_cap','annualized_return_252']):.2%}、
Sharpe {float(counterfactuals.loc['no_emergency_cap','sharpe']):.3f}、MDD
{float(counterfactuals.loc['no_emergency_cap','max_drawdown']):.2%}。

## 解释与优化边界

月度跑输本身不是缺陷：C2以较低波动和更浅回撤换取部分上涨月份。真正可优化的只应是
“慢门控已恢复但锁仓仍延迟退出”这类可识别摩擦；若主要损失来自慢门控仍判定风险关闭时
Momentum突然上涨，则缩短锁无法解决，只能增加新的资产特异性旁路，而此前Gold Override
已经被多重试验与walk-forward审计判定为高过拟合风险。
"""
    (output / "report.md").write_text(report, encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    summary = run_report(args.root.resolve(), args.output)
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
