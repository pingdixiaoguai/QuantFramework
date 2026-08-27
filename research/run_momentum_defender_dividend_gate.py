"""Evaluate and tune the joint 510300-return/Defender-dividend gate family."""

from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import asdict
from datetime import date
from pathlib import Path

import pandas as pd
import yaml

from research.momentum_defender_dividend_gate import (
    CONJUNCTION,
    ENTRY_ONLY,
    DividendGateParams,
    candidate_record,
    run_dividend_gate,
    search_grid,
    validate_dividend_gate,
)
from research.momentum_defender_integrated import run_integrated_c2
from research.momentum_defender_occam import performance
from research.standard_report import generate_standard_report


DEFAULT_CONFIG = Path("research/configs/momentum_defender_dividend_gate_search.yaml")
DEFAULT_OUTPUT = Path("experiments/20260822_momentum_defender_dividend_gate")


def _load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("search config must be a mapping")
    return config


def _periods(config: dict) -> dict[str, tuple[date, date]]:
    return {
        label: (date.fromisoformat(values[0]), date.fromisoformat(values[1]))
        for label, values in config["periods"].items()
    }


def _literal_params(config: dict, mode: str) -> DividendGateParams:
    literal = config["literal_strategy"]
    return DividendGateParams(
        lookback=int(literal["lookback"]),
        slow_return_threshold=float(literal["slow_return_threshold"]),
        defender_primary_minimum=float(literal["defender_primary_minimum"]),
        min_hold_days=int(literal["min_hold_days"]),
        exit_mode=mode,
    )


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


def _neighbor_rows(grid: pd.DataFrame, selected: pd.Series) -> pd.DataFrame:
    parameters = [
        "lookback",
        "slow_return_threshold",
        "defender_primary_minimum",
        "min_hold_days",
        "exit_mode",
    ]
    mask = pd.Series(False, index=grid.index)
    for changed in parameters:
        same = pd.Series(True, index=grid.index)
        for parameter in parameters:
            if parameter == changed:
                continue
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
        "defender_entries",
        "defender_days",
    ]
    return grid.loc[mask, columns].sort_values(
        ["worst_split_sharpe", "full_sharpe"], ascending=False
    )


def _daily(backtest) -> pd.DataFrame:
    return backtest.state.join(
        backtest.simulated.drop(columns=["risk_on"])
    )


def run_experiment(root: Path, config_path: Path, output: Path) -> dict[str, object]:
    config = _load_config(config_path)
    periods = _periods(config)
    end = periods["full"][1]
    integrated = run_integrated_c2(root, end=end)
    baseline_returns = integrated.result.simulated["return"].astype(float)
    baseline_record = _baseline_record(baseline_returns, periods)

    literal_entry = run_dividend_gate(integrated, _literal_params(config, ENTRY_ONLY))
    literal_conjunction = run_dividend_gate(
        integrated, _literal_params(config, CONJUNCTION)
    )
    literal_records = pd.DataFrame(
        [
            candidate_record(literal_entry, periods),
            candidate_record(literal_conjunction, periods),
        ]
    )
    literal_is_bad = bool(
        literal_records["full_sharpe"].max() < float(baseline_record["full_sharpe"])
        or literal_records["full_max_drawdown"].max()
        < float(baseline_record["full_max_drawdown"])
    )

    grid_config = config["grid"]
    grid = search_grid(
        integrated,
        periods,
        lookbacks=grid_config["lookbacks"],
        slow_thresholds=grid_config["slow_return_thresholds"],
        primary_minimums=grid_config["defender_primary_minimums"],
        min_hold_days=grid_config["min_hold_days"],
        exit_modes=grid_config["exit_modes"],
    )
    selection = config["selection"]
    eligible = grid.loc[
        grid["defender_entries"].ge(int(selection["minimum_defender_entries"]))
    ].copy()
    if eligible.empty:
        raise RuntimeError("search produced no eligible candidates")
    robust = eligible.sort_values(
        ["worst_split_sharpe", "full_sharpe", "full_annualized_return_252"],
        ascending=False,
    ).iloc[0]
    best_full = eligible.sort_values(
        ["full_sharpe", "full_annualized_return_252"], ascending=False
    ).iloc[0]
    best_development = eligible.sort_values(
        ["development_sharpe", "development_annualized_return_252"],
        ascending=False,
    ).iloc[0]
    dominating = eligible.loc[
        eligible["full_annualized_return_252"].ge(
            float(baseline_record["full_annualized_return_252"])
        )
        & eligible["full_sharpe"].ge(float(baseline_record["full_sharpe"]))
        & eligible["full_max_drawdown"].ge(
            float(baseline_record["full_max_drawdown"])
        )
    ].copy()
    improvement_counts = {
        "annualized_return": int(
            eligible["full_annualized_return_252"].gt(
                float(baseline_record["full_annualized_return_252"])
            ).sum()
        ),
        "sharpe": int(
            eligible["full_sharpe"].gt(float(baseline_record["full_sharpe"])).sum()
        ),
        "max_drawdown": int(
            eligible["full_max_drawdown"].gt(
                float(baseline_record["full_max_drawdown"])
            ).sum()
        ),
        "worst_split_sharpe": int(
            eligible["worst_split_sharpe"].gt(
                float(baseline_record["worst_split_sharpe"])
            ).sum()
        ),
    }

    robust_params = DividendGateParams(
        lookback=int(robust["lookback"]),
        slow_return_threshold=float(robust["slow_return_threshold"]),
        defender_primary_minimum=float(robust["defender_primary_minimum"]),
        min_hold_days=int(robust["min_hold_days"]),
        exit_mode=str(robust["exit_mode"]),
    )
    best = run_dividend_gate(integrated, robust_params)
    audits = {
        "literal_entry_only": validate_dividend_gate(literal_entry),
        "literal_conjunction": validate_dividend_gate(literal_conjunction),
        "best_robust": validate_dividend_gate(best),
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    grid.sort_values(["worst_split_sharpe", "full_sharpe"], ascending=False).to_csv(
        stage / "candidate_grid.csv", index=False
    )
    literal_records.to_csv(stage / "literal_strategy_metrics.csv", index=False)
    pd.DataFrame([baseline_record]).to_csv(stage / "baseline_metrics.csv", index=False)
    _neighbor_rows(grid, robust).to_csv(stage / "best_robust_neighborhood.csv", index=False)
    _daily(literal_entry).to_csv(stage / "daily_literal_entry_only.csv")
    _daily(literal_conjunction).to_csv(stage / "daily_literal_conjunction.csv")
    _daily(best).to_csv(stage / "daily_best_robust.csv")
    (stage / "audit.json").write_text(
        json.dumps(audits, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (stage / "strategy_config.yaml").write_text(
        config_path.read_text(encoding="utf-8"), encoding="utf-8"
    )

    standard_config = {
        "strategy_name": config["experiment"]["id"],
        "candidate": asdict(robust_params),
        "evidence_status": config["experiment"]["evidence_status"],
    }
    generate_standard_report(
        literal_entry.simulated["return"],
        baseline_returns,
        "Current Integrated C2",
        stage / "literal_80pct_entry_only_vs_current_c2.html",
        {**standard_config, "candidate": asdict(literal_entry.params)},
    )
    generate_standard_report(
        best.simulated["return"],
        baseline_returns,
        "Current Integrated C2",
        stage / "best_robust_vs_current_c2.html",
        standard_config,
    )

    literal = literal_records.set_index("exit_mode").loc[ENTRY_ONLY]
    report = f"""# 510300慢门控 × Defender红利目标联合门控

## 用户指定方案

字面方案仅在510300的40日收益严格低于2.5%，且Defender下一开盘红利ETF目标合计
不低于80%时，允许从Momentum切入Defender。30日状态锁保留，紧急cap不能绕过联合门控。

- 当前C2基线：年化 {float(baseline_record['full_annualized_return_252']):.2%}，Sharpe {float(baseline_record['full_sharpe']):.3f}，MDD {float(baseline_record['full_max_drawdown']):.2%}。
- 指定方案（仅限制入场）：年化 {float(literal['full_annualized_return_252']):.2%}，Sharpe {float(literal['full_sharpe']):.3f}，MDD {float(literal['full_max_drawdown']):.2%}。
- 判定：{'效果较差，已自动执行寻参。' if literal_is_bad else '未触发寻参。'}

## 搜索范围与选择

共评估 {len(grid)} 个联合门控候选。稳健候选按 development、validation、recent 三段
Sharpe的最小值优先，再按全样本Sharpe和年化收益打破平局；每个候选至少需要
{int(selection['minimum_defender_entries'])}次Defender入场。

最佳稳健候选：`{robust_params.candidate_id()}`

- 参数：{robust_params.lookback}日510300收益 < {robust_params.slow_return_threshold:.2%}；Defender红利目标 ≥ {robust_params.defender_primary_minimum:.0%}；状态锁 {robust_params.min_hold_days}日；退出语义 `{robust_params.exit_mode}`。
- 全样本：年化 {float(robust['full_annualized_return_252']):.2%}，Sharpe {float(robust['full_sharpe']):.3f}，MDD {float(robust['full_max_drawdown']):.2%}。
- 三段Sharpe：development {float(robust['development_sharpe']):.3f}，validation {float(robust['validation_sharpe']):.3f}，recent {float(robust['recent_sharpe']):.3f}；最差分段 {float(robust['worst_split_sharpe']):.3f}。
- 全样本Sharpe最高候选：`{best_full['candidate_id']}`，Sharpe {float(best_full['full_sharpe']):.3f}。
- 仅用development选择的候选：`{best_development['candidate_id']}`；validation Sharpe {float(best_development['validation_sharpe']):.3f}，recent Sharpe {float(best_development['recent_sharpe']):.3f}。
- 同时不劣于当前基线年化、Sharpe和MDD的候选数：{len(dominating)}。
- 单项超过当前基线的候选数：年化 {improvement_counts['annualized_return']}，Sharpe {improvement_counts['sharpe']}，MDD {improvement_counts['max_drawdown']}，最差分段Sharpe {improvement_counts['worst_split_sharpe']}。

## 结论边界

这是回溯参数搜索，不是独立样本外证据。即使某个候选优于80%字面方案，也只有在分段表现、
邻域稳定性和相对当前生产基线均可接受时才值得进一步前瞻观察；本报告不会自动替换生产信号。
"""
    (stage / "research_report.md").write_text(report, encoding="utf-8")

    summary = {
        "experiment_id": config["experiment"]["id"],
        "literal_is_bad": literal_is_bad,
        "candidate_count": int(len(grid)),
        "eligible_candidate_count": int(len(eligible)),
        "dominating_baseline_count": int(len(dominating)),
        "improvement_counts": improvement_counts,
        "best_robust": robust.to_dict(),
        "best_full": best_full.to_dict(),
        "best_development": best_development.to_dict(),
        "audits": audits,
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
        f"dominating_baseline={summary['dominating_baseline_count']} "
        f"best_robust={summary['best_robust']['candidate_id']}"
    )
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
