"""Generate reports for direct momentum comparison with the whole Defender NAV."""

from __future__ import annotations

import argparse
import json
import tempfile
from datetime import date
from pathlib import Path

import pandas as pd

from research.defender_curve_momentum import (
    ALL_CANDIDATES,
    DEFENDER_CANDIDATE,
    CurveMomentumParams,
    build_candidate_bundle,
    run_curve_momentum_from_bundle,
)
from research.momentum_defender_integrated import run_integrated_c2
from research.momentum_defender_occam import HELD_RETURN, performance
from research.standard_report import generate_standard_report


DEFAULT_CONFIG = Path("research/configs/defender_curve_quality_momentum.yaml")
DEFAULT_OUTPUT = Path("experiments/20260823_defender_curve_quality_momentum")
REPORT_START = date(2019, 1, 18)


def _metrics(strategies: dict[str, pd.Series]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"strategy": name, **performance(returns.astype(float))}
            for name, returns in strategies.items()
        ]
    )


def _annual(strategies: dict[str, pd.Series]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for name, returns in strategies.items():
        for year, sample in returns.groupby(returns.index.year):
            rows.append(
                {
                    "strategy": name,
                    "year": int(year),
                    "observations": int(len(sample)),
                    "total_return": float((1.0 + sample).prod() - 1.0),
                }
            )
    return pd.DataFrame(rows)


def run_experiment(root: Path, output: Path, config_path: Path) -> dict[str, object]:
    _, curves, scores, interfaces = build_candidate_bundle(window=20)
    daily = run_curve_momentum_from_bundle(
        CurveMomentumParams(window=20, rebalance_days=1, start=REPORT_START),
        curves,
        scores,
        interfaces,
    )
    five_day = run_curve_momentum_from_bundle(
        CurveMomentumParams(window=20, rebalance_days=5, start=REPORT_START),
        curves,
        scores,
        interfaces,
    )
    extended = run_curve_momentum_from_bundle(
        CurveMomentumParams(window=20, rebalance_days=1, start=date(2013, 1, 1)),
        curves,
        scores,
        interfaces,
    )
    integrated = run_integrated_c2(root, end=daily.calendar.max().date())
    baselines = {
        "current_integrated_c2": integrated.result.simulated["return"].astype(float),
        "original_momentum": integrated.result.inputs.momentum[HELD_RETURN].astype(float),
        "always_defender": integrated.result.inputs.defender[HELD_RETURN].astype(float),
    }
    strategies = {
        "direct_curve_momentum_daily": daily.daily["return"].astype(float),
        "direct_curve_momentum_5d": five_day.daily["return"].astype(float),
        **baselines,
    }
    metrics = _metrics(strategies)
    annual = _annual(strategies)
    extended_metrics = performance(extended.daily["return"])

    latest_scores = scores.iloc[-1].astype(float).sort_values(ascending=False)
    next_open_desired = str(latest_scores.index[0])
    latest = {
        "signal_date": scores.index[-1].date().isoformat(),
        "next_open_desired": next_open_desired,
        "scores": {candidate: float(latest_scores[candidate]) for candidate in ALL_CANDIDATES},
        "daily_current_candidate": str(daily.daily.iloc[-1]["candidate"]),
        "five_day_current_candidate": str(five_day.daily.iloc[-1]["candidate"]),
        "five_day_held_days_at_open": int(five_day.daily.iloc[-1]["held_days_at_open"]),
    }
    curve = (1.0 + daily.daily["return"].astype(float)).cumprod()
    drawdown = curve / curve.cummax() - 1.0
    trough = pd.Timestamp(drawdown.idxmin())
    peak = pd.Timestamp(curve.loc[:trough].idxmax())
    drawdown_slice = daily.daily.loc[peak:trough]
    drawdown_audit = {
        "peak": peak.date().isoformat(),
        "trough": trough.date().isoformat(),
        "drawdown": float(drawdown.loc[trough]),
        "switches": int(drawdown_slice["switched"].sum()),
        "candidate_days": {
            str(candidate): int(count)
            for candidate, count in drawdown_slice["candidate"].value_counts().items()
        },
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    daily.daily.to_csv(stage / "daily_direct_top1.csv")
    five_day.daily.to_csv(stage / "daily_5day_hold.csv")
    extended.daily.to_csv(stage / "daily_extended_2013.csv")
    metrics.to_csv(stage / "strategy_metrics.csv", index=False)
    annual.to_csv(stage / "calendar_year_returns.csv", index=False)
    scores.tail(120).to_csv(stage / "latest_120d_candidate_scores.csv")
    pd.DataFrame(
        {
            "defender_nav": curves[DEFENDER_CANDIDATE],
            "defender_quality_momentum": scores[DEFENDER_CANDIDATE],
        }
    ).to_csv(stage / "defender_whole_curve_factor.csv")
    audits = {
        "direct_daily": daily.audit,
        "five_day": five_day.audit,
        "extended_daily": extended.audit,
        "latest_signal": latest,
        "maximum_drawdown_episode": drawdown_audit,
    }
    (stage / "audit.json").write_text(
        json.dumps(audits, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (stage / "strategy_config.yaml").write_text(
        config_path.read_text(encoding="utf-8"), encoding="utf-8"
    )

    report_config = {
        "strategy_name": "defender_curve_quality_momentum_top1",
        "factor": "quality_momentum(window=20)",
        "defender_representation": "whole continuous-hold net NAV",
        "signal_timing": "previous_close_to_next_open",
    }
    generate_standard_report(
        strategies["direct_curve_momentum_daily"],
        baselines["current_integrated_c2"],
        "Current Integrated C2",
        stage / "direct_daily_vs_current_c2.html",
        {**report_config, "rebalance_days": 1},
    )
    generate_standard_report(
        strategies["direct_curve_momentum_daily"],
        baselines["original_momentum"],
        "Original 4ETF Momentum",
        stage / "direct_daily_vs_original_momentum.html",
        {**report_config, "rebalance_days": 1},
    )
    generate_standard_report(
        strategies["direct_curve_momentum_5d"],
        baselines["current_integrated_c2"],
        "Current Integrated C2",
        stage / "five_day_vs_current_c2.html",
        {**report_config, "rebalance_days": 5},
    )

    table = metrics.set_index("strategy")
    literal = table.loc["direct_curve_momentum_daily"]
    held = table.loc["direct_curve_momentum_5d"]
    current = table.loc["current_integrated_c2"]
    original = table.loc["original_momentum"]
    report = f"""# Defender整体收益曲线直接参与质量动量Top1

## 实现口径

Defender没有拆成内部ETF。其完整连续持有净收益曲线的`nav_if_held`被视为第五个候选资产的
收盘价，与510300、159915、513100、518880完全调用同一个
`quality_momentum(window=20)`：20日收益乘Kaufman路径效率。所有得分只在收盘后可知，
下一交易日开盘选择Top1；切换使用旧候选退出腿和新候选进入腿，并保留Defender内部费用。

## 2019-01-18至2026-08-21结果

|策略|年化收益|Sharpe|最大回撤|切换次数|Defender日数|
|---|---:|---:|---:|---:|---:|
|直接五选一，每日|{float(literal.annualized_return_252):.2%}|{float(literal.sharpe):.3f}|{float(literal.max_drawdown):.2%}|{daily.audit['switches']}|{daily.audit['defender_days']}|
|直接五选一，5日锁|{float(held.annualized_return_252):.2%}|{float(held.sharpe):.3f}|{float(held.max_drawdown):.2%}|{five_day.audit['switches']}|{five_day.audit['defender_days']}|
|原四ETF Momentum|{float(original.annualized_return_252):.2%}|{float(original.sharpe):.3f}|{float(original.max_drawdown):.2%}|—|—|
|当前 C2|{float(current.annualized_return_252):.2%}|{float(current.sharpe):.3f}|{float(current.max_drawdown):.2%}|—|—|

每日直接比较相对原Momentum年化变化{float(literal.annualized_return_252-original.annualized_return_252):+.2%}、Sharpe变化{float(literal.sharpe-original.sharpe):+.3f}、MDD变化{float(literal.max_drawdown-original.max_drawdown):+.2%}；相对当前C2，年化变化{float(literal.annualized_return_252-current.annualized_return_252):+.2%}、Sharpe变化{float(literal.sharpe-current.sharpe):+.3f}、MDD变化{float(literal.max_drawdown-current.max_drawdown):+.2%}。

完整可用历史从{extended_metrics['start']}开始：每日五选一年化{float(extended_metrics['annualized_return_252']):.2%}、Sharpe{float(extended_metrics['sharpe']):.3f}、MDD{float(extended_metrics['max_drawdown']):.2%}。

## 最新收盘比较

截至{latest['signal_date']}，下一开盘得分最高的是`{next_open_desired}`。五条曲线得分见
`latest_120d_candidate_scores.csv`；Defender整体NAV及其因子逐日值见
`defender_whole_curve_factor.csv`。

- 510300：{latest['scores']['510300.SH']:.5f}
- 159915：{latest['scores']['159915.SZ']:.5f}
- 513100：{latest['scores']['513100.SH']:.5f}
- 518880：{latest['scores']['518880.SH']:.5f}
- Defender整体：{latest['scores']['DEFENDER']:.5f}

每日版下一开盘会选择黄金；5日版当前仍持有Defender，处于持有窗口内，暂不切换。

## 结论

把Defender整体NAV直接加入Top1，只给原Momentum带来很小的收益/Sharpe变化，却显著加深
最大回撤；5日持有约束也没有改善。它不应替换当前C2。原因是相对动量只能回答“最近哪条
曲线更强”，没有保留C2慢门控在风险阶段持续持有低波Defender的机制。

每日版最大回撤发生于{drawdown_audit['peak']}至{drawdown_audit['trough']}，区间回撤
{drawdown_audit['drawdown']:.2%}；其中创业板持有{drawdown_audit['candidate_days'].get('159915.SZ', 0)}日、
Defender仅{drawdown_audit['candidate_days'].get('DEFENDER', 0)}日，并发生{drawdown_audit['switches']}次切换。
"""
    (stage / "research_report.md").write_text(report, encoding="utf-8")

    summary = {
        "strategy_id": "defender_curve_quality_momentum_top1",
        "direct_daily": daily.audit,
        "five_day": five_day.audit,
        "extended_daily": extended.audit,
        "latest_signal": latest,
        "maximum_drawdown_episode": drawdown_audit,
        "production_replacement": False,
        "reason": "materially_worse_than_current_c2_and_deeper_drawdown_than_original_momentum",
    }
    (stage / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    output.mkdir(parents=True, exist_ok=True)
    for path in stage.iterdir():
        path.replace(output / path.name)
    stage.rmdir()
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    summary = run_experiment(args.root.resolve(), args.output, args.config)
    print(
        f"direct_daily={summary['direct_daily']['performance']} "
        f"latest={summary['latest_signal']['next_open_desired']}"
    )
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
