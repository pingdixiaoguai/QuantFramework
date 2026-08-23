"""Run the versioned frozen Momentum/Defender C2 research strategy."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
from datetime import date
from pathlib import Path

import pandas as pd

from research.momentum_defender_c2 import (
    DEFAULT_CONFIG_PATH,
    FrozenC2Config,
    load_frozen_c2_config,
    run_frozen_c2,
    validate_frozen_checkpoint,
)
from research.momentum_defender_occam import (
    HELD_RETURN,
    MOMENTUM_ASSETS,
    apply_state_schedule,
    performance,
    simulate_switch,
)
from research.standard_report import generate_standard_report


DEFAULT_OUTPUT = Path("experiments/20260821_momentum_defender_c2_frozen_v2")
PERIODS = {
    "development_2019_2022": (
        pd.Timestamp("2019-01-18"),
        pd.Timestamp("2022-12-30"),
    ),
    "2023": (pd.Timestamp("2023-01-01"), pd.Timestamp("2023-12-31")),
    "2024": (pd.Timestamp("2024-01-01"), pd.Timestamp("2024-12-31")),
    "2025": (pd.Timestamp("2025-01-01"), pd.Timestamp("2025-12-31")),
    "2026_ytd": (pd.Timestamp("2026-01-01"), pd.Timestamp("2026-08-17")),
    "full": (pd.Timestamp("2019-01-18"), pd.Timestamp("2026-08-17")),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()


def _metric_records(
    strategies: dict[str, pd.Series],
    end: date,
) -> pd.DataFrame:
    periods = {
        **PERIODS,
        "2026_ytd": (pd.Timestamp("2026-01-01"), pd.Timestamp(end)),
        "full": (pd.Timestamp("2019-01-18"), pd.Timestamp(end)),
    }
    records: list[dict[str, object]] = []
    for period, (start, finish) in periods.items():
        for strategy, returns in strategies.items():
            records.append(
                {
                    "period": period,
                    "strategy": strategy,
                    **performance(returns.loc[start:finish]),
                }
            )
    return pd.DataFrame(records)


def _calendar_year_records(strategies: dict[str, pd.Series]) -> pd.DataFrame:
    calendar = next(iter(strategies.values())).index
    records: list[dict[str, object]] = []
    for year in sorted(calendar.year.unique()):
        for strategy, returns in strategies.items():
            sample = returns.loc[returns.index.year == year]
            records.append(
                {
                    "year": int(year),
                    "strategy": strategy,
                    "observations": len(sample),
                    "total_return": float((1.0 + sample).prod() - 1.0),
                }
            )
    return pd.DataFrame(records)


def _defender_periods(
    state: pd.DataFrame,
    strategies: dict[str, pd.Series],
) -> pd.DataFrame:
    defender = ~state["risk_on"].astype(bool)
    groups = defender.ne(defender.shift()).cumsum()
    records: list[dict[str, object]] = []
    for episode, (_, active) in enumerate(
        state.loc[defender].groupby(groups.loc[defender]), start=1
    ):
        index = active.index
        row: dict[str, object] = {
            "episode": episode,
            "start": index.min().date().isoformat(),
            "end": index.max().date().isoformat(),
            "observations": len(index),
            "entry_reason": active.iloc[0]["state_reason"],
        }
        for strategy, returns in strategies.items():
            sample = returns.loc[index]
            row[f"{strategy}_total_return"] = float((1.0 + sample).prod() - 1.0)
        records.append(row)
    return pd.DataFrame(records)


def _metric(metrics: pd.DataFrame, strategy: str) -> pd.Series:
    return metrics.loc[
        metrics["period"].eq("full") & metrics["strategy"].eq(strategy)
    ].iloc[0]


def _summary_table(metrics: pd.DataFrame) -> str:
    labels = {
        "frozen_c2": "冻结C2",
        "no_cap_fusion": "无cap融合",
        "original_momentum": "原动量策略",
        "original_base": "原4ETF等权base",
    }
    lines = [
        "|方案|年化收益|年化波动|Sharpe|最大回撤|",
        "|---|---:|---:|---:|---:|",
    ]
    for strategy, label in labels.items():
        row = _metric(metrics, strategy)
        lines.append(
            f"|{label}|{row.annualized_return_252:.2%}|"
            f"{row.annualized_volatility:.2%}|{row.sharpe:.3f}|"
            f"{row.max_drawdown:.2%}|"
        )
    return "\n".join(lines)


def _year_table(yearly: pd.DataFrame) -> str:
    pivot = yearly.pivot(index="year", columns="strategy", values="total_return")
    lines = [
        "|年份|冻结C2|无cap融合|原动量|原base|",
        "|---:|---:|---:|---:|---:|",
    ]
    for year, row in pivot.iterrows():
        lines.append(
            f"|{year}|{row.frozen_c2:+.2%}|{row.no_cap_fusion:+.2%}|"
            f"{row.original_momentum:+.2%}|{row.original_base:+.2%}|"
        )
    return "\n".join(lines)


def run_experiment(
    root: Path,
    config_path: Path,
    final_output: Path,
    defender_dir: Path | None = None,
    end: date | None = None,
) -> None:
    config = load_frozen_c2_config(config_path)
    cutoff = end or config.research_cutoff
    if cutoff != config.research_cutoff:
        raise ValueError(
            "frozen report cutoff must match the versioned checkpoint: "
            f"{config.research_cutoff.isoformat()}"
        )
    final_output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(prefix=f".{final_output.name}.staging-", dir=final_output.parent)
    )
    result = run_frozen_c2(
        root,
        config,
        defender_dir=defender_dir,
        end=cutoff,
    )
    checkpoint_audit = validate_frozen_checkpoint(result)
    (stage / "checkpoint_audit.json").write_text(
        json.dumps(checkpoint_audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    calendar = result.inputs.calendar
    exact_momentum = result.inputs.momentum[HELD_RETURN].astype(float)
    original_base = result.inputs.momentum_result.benchmark_returns.reindex(
        calendar
    ).astype(float)
    if original_base.isna().any():
        raise ValueError("original 4ETF base has missing report dates")
    no_cap_state = apply_state_schedule(
        result.slow_signal,
        pd.Series(False, index=calendar),
        calendar,
        config.min_hold_days,
        emergency_override=config.emergency_override,
        initial_risk_on=True,
    )
    no_cap_simulated = simulate_switch(
        result.inputs.momentum,
        result.inputs.defender,
        no_cap_state["risk_on"],
        initial_previous_state=config.initial_previous_sleeve,
    )
    strategies = {
        "frozen_c2": result.simulated["return"],
        "no_cap_fusion": no_cap_simulated["return"],
        "original_momentum": exact_momentum,
        "original_base": original_base,
    }
    metrics = _metric_records(strategies, cutoff)
    metrics.to_csv(stage / "strategy_period_metrics.csv", index=False)
    yearly = _calendar_year_records(strategies)
    yearly.to_csv(stage / "calendar_year_returns.csv", index=False)
    periods = _defender_periods(result.state, strategies)
    periods.to_csv(stage / "defender_periods.csv", index=False)
    result.daily.to_csv(stage / "daily_backtest.csv")
    (stage / "strategy_config.yaml").write_text(
        config_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    standard_config = config.serializable()
    reports = {
        "momentum_defender_c2_vs_original_base.html": (
            original_base,
            "Original 4ETF Equal-weight Base",
        ),
        "momentum_defender_c2_vs_original_momentum.html": (
            exact_momentum,
            "Original Momentum Strategy",
        ),
        "momentum_defender_c2_vs_no_cap_fusion.html": (
            no_cap_simulated["return"],
            "No-cap Slow-gate Fusion",
        ),
    }
    for filename, (benchmark, benchmark_name) in reports.items():
        generate_standard_report(
            result.simulated["return"],
            benchmark,
            benchmark_name,
            stage / filename,
            standard_config,
        )

    frozen = _metric(metrics, "frozen_c2")
    no_cap = _metric(metrics, "no_cap_fusion")
    momentum = _metric(metrics, "original_momentum")
    emergency_entries = (
        result.state["state_changed"].astype(bool)
        & result.state["state_reason"].eq("emergency_exit")
    )
    report = f"""# Momentum × Defender C2 冻结版本回测

## 版本结论

当前分支固定版本为`{config.strategy_id}`，参数ID为`{config.variant_id()}`。创业板ETF分位数固定为q{int(config.asset_quantiles['159915.SZ'] * 100)}；本报告不执行寻参，也不从任何消融实验读取参数。

{_summary_table(metrics)}

相对无cap融合，冻结C2年化变化{frozen.annualized_return_252 - no_cap.annualized_return_252:+.2%}、Sharpe变化{frozen.sharpe - no_cap.sharpe:+.3f}、MDD变化{frozen.max_drawdown - no_cap.max_drawdown:+.2%}；相对原动量策略，年化变化{frozen.annualized_return_252 - momentum.annualized_return_252:+.2%}、Sharpe变化{frozen.sharpe - momentum.sharpe:+.3f}、MDD变化{frozen.max_drawdown - momentum.max_drawdown:+.2%}。

## 固定规则

- Momentum：`strategy/configs/quality_momentum_top1.yaml`的四ETF Top-1策略。
- 慢门控：沪深300ETF 40日收益高于2.5%为Momentum，否则为Defender；上一收盘信号下一开盘执行。
- 紧急cap：每只ETF使用10日Rogers–Satchell波动率，严格滞后的全历史扩展分位数；沪深300q{int(config.asset_quantiles['510300.SH'] * 100)}、创业板q{int(config.asset_quantiles['159915.SZ'] * 100)}、纳指q{int(config.asset_quantiles['513100.SH'] * 100)}、黄金q{int(config.asset_quantiles['518880.SH'] * 100)}。
- 只读取上一收盘Momentum实际持有ETF的cap；`cap≤0.8`时下一开盘紧急切入Defender。
- 状态锁：30个交易日；紧急Momentum→Defender可绕过锁，Defender→Momentum不得绕过。
- 开盘切换：卖出旧袖套的退出腿并买入新袖套的进入腿，使用两份交接接口中的既有净费用。

## 状态统计

- 报警日：{int(result.emergency_alert.sum())}。
- 紧急入场：{int(emergency_entries.sum())}次。
- Defender持有：{int((~result.state['risk_on']).sum())}日。
- 袖套切换：{int(result.simulated['sleeve_switch'].sum())}次。
- 检查点：`checkpoint_audit.json`已逐项通过，包括1,837个交易日逐日收益哈希。

## 逐年收益

{_year_table(yearly)}

## 证据边界

这是当前分支冻结的研究候选，不等于已接入`run_daily.py`的生产策略。C2相对无cap的历史优势高度依赖少数事件，既有过拟合审计仍然有效；冻结的含义是后续不再用当前历史修改参数，并从下一未观察交易日起做前瞻验证。
"""
    (stage / "research_report.md").write_text(report, encoding="utf-8")

    deliverable = defender_dir or config.defender_deliverable_dir
    input_files = [
        config_path,
        deliverable / config.defender_switch_returns_file,
        root / config.momentum_config_path,
        *[root / "data/db" / f"{asset}.parquet" for asset in MOMENTUM_ASSETS],
    ]
    code_files = [
        root / "research/momentum_defender_c2.py",
        root / "research/momentum_volatility.py",
        root / "research/standard_report.py",
        root / "research/run_momentum_defender_c2_frozen.py",
        root / "research/momentum_defender_occam.py",
    ]
    manifest = {
        "experiment": config.strategy_id,
        "generated_on": date.today().isoformat(),
        "research_cutoff": cutoff.isoformat(),
        "git_branch": _git(root, "branch", "--show-current"),
        "git_commit": _git(root, "rev-parse", "HEAD"),
        "variant_id": config.variant_id(),
        "checkpoint_status": checkpoint_audit["status"],
        "inputs": [{"path": str(path), "sha256": _sha256(path)} for path in input_files],
        "code_sources": [
            {"path": str(path), "sha256": _sha256(path)} for path in code_files
        ],
    }
    (stage / "experiment_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    final_output.mkdir(parents=True, exist_ok=True)
    for path in stage.iterdir():
        path.replace(final_output / path.name)
    stage.rmdir()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--defender-dir", type=Path)
    parser.add_argument("--end", type=date.fromisoformat)
    args = parser.parse_args()
    run_experiment(
        args.root,
        args.config,
        args.output,
        defender_dir=args.defender_dir,
        end=args.end,
    )


if __name__ == "__main__":
    main()
