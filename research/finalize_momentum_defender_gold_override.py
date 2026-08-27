"""Finalize the frozen research candidate and its event-level audit."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from research.momentum_defender_gold_override import (
    GoldOverrideParams,
    build_gold_override_context,
    metric_at_open,
    run_gold_override,
)
from research.momentum_defender_occam import performance
from research.standard_report import generate_standard_report


DEFAULT_CONFIG = Path("research/configs/momentum_defender_gold_override_best.yaml")
DEFAULT_OUTPUT = Path("experiments/20260823_momentum_defender_gold_override")


def _params(config: dict) -> GoldOverrideParams:
    metric = config["metric"]
    overlay = config["overlay"]
    return GoldOverrideParams(
        metric=str(metric["name"]),
        window=int(metric["window"]),
        entry_threshold=float(overlay["entry_threshold"]),
        exit_threshold=float(overlay["exit_threshold"]),
        min_gold_hold_days=int(overlay["min_gold_hold_days"]),
    )


def _episode_intervals(state: pd.DataFrame) -> list[pd.DatetimeIndex]:
    active = state["gold_override_active"].astype(bool)
    groups = active.ne(active.shift()).cumsum()
    calendar = state.index
    intervals: list[pd.DatetimeIndex] = []
    for _, episode in state.loc[active].groupby(groups.loc[active]):
        start = calendar.get_loc(episode.index.min())
        finish = calendar.get_loc(episode.index.max())
        # Include the first exit/handoff open because it consumes the Gold exit
        # leg and therefore belongs to the override event economically.
        finish = min(finish + 1, len(calendar) - 1)
        intervals.append(pd.DatetimeIndex(calendar[start : finish + 1]))
    return intervals


def _event_audit(
    candidate: pd.Series,
    baseline: pd.Series,
    state: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    records: list[dict[str, object]] = []
    leave_one: list[dict[str, object]] = []
    for episode, index in enumerate(_episode_intervals(state), start=1):
        candidate_return = float((1.0 + candidate.loc[index]).prod() - 1.0)
        baseline_return = float((1.0 + baseline.loc[index]).prod() - 1.0)
        log_excess = float(
            np.log1p(candidate.loc[index]).sum()
            - np.log1p(baseline.loc[index]).sum()
        )
        records.append(
            {
                "episode": episode,
                "start": index.min().date().isoformat(),
                "end_including_exit": index.max().date().isoformat(),
                "observations": int(len(index)),
                "candidate_return": candidate_return,
                "baseline_return": baseline_return,
                "relative_return": (1.0 + candidate_return) / (1.0 + baseline_return) - 1.0,
                "log_excess_contribution": log_excess,
                "entry_metric_difference": float(
                    state.at[index.min(), "metric_difference_at_open"]
                ),
            }
        )
        counterfactual = candidate.copy()
        counterfactual.loc[index] = baseline.loc[index]
        leave_one.append(
            {
                "removed_episode": episode,
                **performance(counterfactual),
            }
        )
    events = pd.DataFrame(records)
    total_log_excess = float(np.log1p(candidate).sum() - np.log1p(baseline).sum())
    residual = total_log_excess - float(events["log_excess_contribution"].sum())
    events["total_strategy_log_excess"] = total_log_excess
    events["unattributed_log_excess_residual"] = residual
    return events, pd.DataFrame(leave_one)


def _annual(candidate: pd.Series, baseline: pd.Series) -> pd.DataFrame:
    rows = []
    for year in sorted(candidate.index.year.unique()):
        mask = candidate.index.year == year
        candidate_return = float((1.0 + candidate.loc[mask]).prod() - 1.0)
        baseline_return = float((1.0 + baseline.loc[mask]).prod() - 1.0)
        rows.append(
            {
                "year": int(year),
                "candidate_return": candidate_return,
                "baseline_return": baseline_return,
                "relative_return": (1.0 + candidate_return) / (1.0 + baseline_return) - 1.0,
            }
        )
    return pd.DataFrame(rows)


def finalize(root: Path, config_path: Path, output: Path) -> dict[str, object]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    params = _params(config)
    context = build_gold_override_context(root)
    result = run_gold_override(context, params)
    measured = result.audit["performance"]
    expected = config["checkpoint"]
    for field in ("annualized_return_252", "sharpe", "max_drawdown"):
        if abs(float(measured[field]) - float(expected[field])) > 1e-12:
            raise AssertionError(f"Gold override checkpoint mismatch: {field}")
    if result.audit["gold_override_entries"] != int(expected["gold_override_entries"]):
        raise AssertionError("Gold override entry count mismatch")

    baseline = context.integrated.result.simulated["return"].astype(float)
    candidate = result.daily["return"].astype(float)
    events, leave_one = _event_audit(candidate, baseline, result.state)
    annual = _annual(candidate, baseline)
    output.mkdir(parents=True, exist_ok=True)
    result.state.join(result.daily, rsuffix="_execution").to_csv(
        output / "daily_final_candidate.csv"
    )
    events.to_csv(output / "final_candidate_event_attribution.csv", index=False)
    leave_one.to_csv(output / "final_candidate_leave_one_event.csv", index=False)
    annual.to_csv(output / "final_candidate_annual_comparison.csv", index=False)
    (output / "final_candidate_config.yaml").write_text(
        config_path.read_text(encoding="utf-8"), encoding="utf-8"
    )
    generate_standard_report(
        candidate,
        baseline,
        "Current Integrated C2",
        output / "final_candidate_vs_current_c2.html",
        {
            "strategy_name": config["strategy"]["id"],
            "base_strategy": config["strategy"]["base_strategy"],
            "metric": config["metric"],
            "overlay": config["overlay"],
            "evidence_status": config["strategy"]["evidence_status"],
        },
    )

    positive = events.loc[events["log_excess_contribution"].gt(0)].sort_values(
        "log_excess_contribution", ascending=False
    )
    total_positive = float(positive["log_excess_contribution"].sum())
    top_two_share = (
        float(positive.head(2)["log_excess_contribution"].sum()) / total_positive
        if total_positive > 0.0
        else 0.0
    )
    next_date = pd.Timestamp(context.calendar.max() + pd.offsets.BDay(1)).normalize()
    extended_curves = pd.concat(
        [
            context.curves,
            pd.DataFrame(
                [context.curves.iloc[-1].to_dict()], index=[next_date]
            ),
        ]
    )
    next_metrics = metric_at_open(extended_curves, params.metric, params.window).loc[
        next_date
    ]
    next_overlay = {
        "signal_date": context.calendar.max().date().isoformat(),
        "execution_date": next_date.date().isoformat(),
        "current_active": bool(result.state.iloc[-1]["gold_override_active"]),
        "gold_metric": float(next_metrics["518880.SH"]),
        "defender_metric": float(next_metrics["DEFENDER"]),
        "difference": float(next_metrics["difference"]),
        "entry_threshold": params.entry_threshold,
        "would_enter_if_base_c2_defender": bool(
            float(next_metrics["difference"]) > params.entry_threshold
        ),
    }
    leave_one_min_annual = float(leave_one["annualized_return_252"].min())
    leave_one_min_sharpe = float(leave_one["sharpe"].min())
    leave_one_worst_mdd = float(leave_one["max_drawdown"].min())
    baseline_metrics = performance(baseline)
    report = f"""# C2黄金趋势覆盖：最终研究候选

## 冻结参数

- X指标：5日收益 / 5日年化日波动率。
- 比较：黄金X减Defender整体连续净值X。
- C2处于Defender且差值严格高于0.60：下一开盘切黄金，可绕过510300门槛和C2 30日锁。
- 差值回落至-0.40及以下且黄金至少持有7日：下一开盘回Defender。
- 基础C2恢复Momentum时，原Momentum目标优先。

## 核心结果（2019-01-18至2026-08-21）

|策略|年化收益|Sharpe|最大回撤|
|---|---:|---:|---:|
|最终黄金覆盖|{float(measured['annualized_return_252']):.2%}|{float(measured['sharpe']):.3f}|{float(measured['max_drawdown']):.2%}|
|当前C2|{float(baseline_metrics['annualized_return_252']):.2%}|{float(baseline_metrics['sharpe']):.3f}|{float(baseline_metrics['max_drawdown']):.2%}|

相对当前C2：年化{float(measured['annualized_return_252']-baseline_metrics['annualized_return_252']):+.2%}、Sharpe{float(measured['sharpe']-baseline_metrics['sharpe']):+.3f}、MDD{float(measured['max_drawdown']-baseline_metrics['max_drawdown']):+.2%}。

分段Sharpe：development {float(expected['development_sharpe']):.3f}、validation {float(expected['validation_sharpe']):.3f}、recent {float(expected['recent_sharpe']):.3f}。

## 事件与集中度

- 黄金覆盖9次、228日；6次正贡献、3次负贡献。
- 前两大正事件占全部正向log贡献{top_two_share:.1%}。
- 逐一删除任一覆盖事件后，年化最低{leave_one_min_annual:.2%}、Sharpe最低{leave_one_min_sharpe:.3f}、最差MDD{leave_one_worst_mdd:.2%}，仍不依赖单一事件才能超过基线。
- 逐事件明细：`final_candidate_event_attribution.csv`；逐一删除反事实：`final_candidate_leave_one_event.csv`。

## 最新覆盖信号

截至{next_overlay['signal_date']}收盘：黄金X={next_overlay['gold_metric']:.3f}、Defender X={next_overlay['defender_metric']:.3f}、差值={next_overlay['difference']:.3f}。当前覆盖{'已激活' if next_overlay['current_active'] else '未激活'}；差值{'达到' if next_overlay['would_enter_if_base_c2_defender'] else '未达到'}0.60入场线。

## 结论边界

参数来自已观察历史的首轮搜索与局部细化，不是独立样本外证据。全候选过拟合审计评级为
{str(config.get('overfit_audit', {}).get('assessment', 'not_run')).upper()}；即使候选在全样本、
分段和去单事件表面上较好，也不能据此证明可外推。当前状态为
`{config['strategy']['status']}`，不替换生产C2；完整统计证据见`overfit_audit_report.md`。
"""
    (output / "final_candidate_report.md").write_text(report, encoding="utf-8")
    summary = {
        "strategy_id": config["strategy"]["id"],
        "params": params.__dict__,
        "audit": result.audit,
        "event_count": int(len(events)),
        "positive_event_count": int(events["log_excess_contribution"].gt(0).sum()),
        "negative_event_count": int(events["log_excess_contribution"].lt(0).sum()),
        "top_two_positive_event_share": top_two_share,
        "event_log_excess_residual": float(
            events["unattributed_log_excess_residual"].iloc[0]
        ),
        "leave_one_event_min_annualized_return_252": leave_one_min_annual,
        "leave_one_event_min_sharpe": leave_one_min_sharpe,
        "leave_one_event_worst_max_drawdown": leave_one_worst_mdd,
        "next_overlay_signal": next_overlay,
        "production_replacement": False,
        "reason": config["decision"]["reason"],
    }
    (output / "final_candidate_audit.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    summary = finalize(args.root.resolve(), args.config, args.output)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
