"""Generate the standard report for the integrated Defender-main C2 strategy."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from datetime import date
from pathlib import Path

import pandas as pd

from research.momentum_defender_integrated import (
    ALL_ASSETS,
    INTEGRATED_STRATEGY_ID,
    UPSTREAM_COMMIT,
    run_integrated_c2,
)
from research.momentum_defender_occam import HELD_RETURN, performance
from research.standard_report import generate_standard_report


DEFAULT_OUTPUT = Path("experiments/20260822_momentum_defender_main_integration")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metrics(strategies: dict[str, pd.Series]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"strategy": name, **performance(returns)} for name, returns in strategies.items()]
    )


def _annual(strategies: dict[str, pd.Series]) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for name, returns in strategies.items():
        for year, sample in returns.groupby(returns.index.year):
            records.append(
                {
                    "strategy": name,
                    "year": int(year),
                    "observations": int(len(sample)),
                    "total_return": float((1.0 + sample).prod() - 1.0),
                }
            )
    return pd.DataFrame(records)


def run_report(
    root: Path,
    output: Path,
    *,
    end: date | None = None,
) -> Path:
    integrated = run_integrated_c2(root, end=end)
    result = integrated.result
    strategies = {
        "integrated_c2": result.simulated["return"].astype(float),
        "original_momentum": result.inputs.momentum[HELD_RETURN].astype(float),
        "always_defender_main": result.inputs.defender[HELD_RETURN].astype(float),
    }
    metrics = _metrics(strategies)
    annual = _annual(strategies)
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))

    integrated.daily.to_csv(stage / "daily_backtest.csv")
    integrated.targets.to_csv(stage / "target_weights.csv")
    metrics.to_csv(stage / "strategy_metrics.csv", index=False)
    annual.to_csv(stage / "calendar_year_returns.csv", index=False)
    (stage / "integration_audit.json").write_text(
        json.dumps(integrated.audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    config_path = root / "strategy/configs/momentum_defender_c2_main.yaml"
    (stage / "strategy_config.yaml").write_text(
        config_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    report_path = stage / "momentum_defender_c2_main_vs_original_momentum.html"
    generate_standard_report(
        strategies["integrated_c2"],
        strategies["original_momentum"],
        "Original Momentum Strategy",
        report_path,
        {
            "strategy_name": INTEGRATED_STRATEGY_ID,
            "defender_upstream_commit": UPSTREAM_COMMIT,
            "defender_strategy_id": integrated.audit["defender_strategy_id"],
            "asset_pool": list(ALL_ASSETS),
            "signal_timing": "previous_close_to_next_open",
        },
    )

    table = metrics.set_index("strategy")
    integrated_metric = table.loc["integrated_c2"]
    momentum_metric = table.loc["original_momentum"]
    emergency_entries = int(
        (
            result.state["state_changed"].astype(bool)
            & result.state["state_reason"].eq("emergency_exit")
        ).sum()
    )
    summary = f"""# Momentum × Defender main 内嵌版回测

## 结论

本报告使用内嵌的 Defender main 提交 `{UPSTREAM_COMMIT}`，正式 Defender ID 为
`{integrated.audit['defender_strategy_id']}`。运行时不读取外部 Defender 仓库或交付 CSV。

- 样本：{integrated.audit['start']} 至 {integrated.audit['end']}，{integrated.audit['observations']} 个交易日。
- 复合策略年化收益：{float(integrated_metric['annualized_return_252']):.2%}。
- 复合策略 Sharpe：{float(integrated_metric['sharpe']):.3f}。
- 复合策略最大回撤：{float(integrated_metric['max_drawdown']):.2%}。
- 相对原 Momentum 年化变化：{float(integrated_metric['annualized_return_252'] - momentum_metric['annualized_return_252']):+.2%}。
- 相对原 Momentum Sharpe 变化：{float(integrated_metric['sharpe'] - momentum_metric['sharpe']):+.3f}。
- Defender 持有日：{integrated.audit['defender_days']}；袖套切换：{integrated.audit['sleeve_switches']}；紧急入场：{emergency_entries}。

## 机械审计

- 目标权重与现金合计最大误差：{float(integrated.audit['target_sum_max_abs_error']):.3e}。
- 复合收益重构净值最大误差：{float(integrated.audit['nav_reconstruction_max_abs_error']):.3e}。
- 信号时序因果校验：{'通过' if integrated.audit['signal_timing_causal'] else '失败'}。
- HTML：`{report_path.name}`。

## 证据边界

Defender main 的 2013 历史扩展和 C2 参数都使用过已观察历史；本次结果用于实现与逻辑
校验，不会把回溯收益重新解释成独立样本外证据。后续生产运行应继续记录冻结后的前瞻信号。
"""
    (stage / "research_report.md").write_text(summary, encoding="utf-8")

    source_paths = [
        root / "defender/UPSTREAM.md",
        root / "defender/__init__.py",
        root / "defender/current_strategy.py",
        root / "defender/defender_opt_v2.py",
        root / "defender/engine.py",
        root / "defender/grid_reproduction.py",
        root / "defender/market_risk_overlay.py",
        root / "defender/relative_defender.py",
        root / "defender/relative_defender_champion.py",
        root / "defender/relative_defender_rotation.py",
        root / "defender/relative_defender_rotation_2013_report.py",
        root / "defender/relative_defender_rotation_2013_export.py",
        root / "defender/live.py",
        root / "research/momentum_defender_c2.py",
        root / "research/momentum_defender_integrated.py",
        root / "research/momentum_defender_occam.py",
        root / "research/momentum_volatility.py",
        root / "research/run_momentum_defender_integrated.py",
        root / "research/standard_report.py",
        root / "backtest/report.py",
        root / "strategy/momentum_defender.py",
        root / "run_daily_momentum_defender.py",
        config_path,
    ]
    manifest = {
        "strategy_id": INTEGRATED_STRATEGY_ID,
        "generated_on": date.today().isoformat(),
        "defender_upstream_commit": UPSTREAM_COMMIT,
        "audit_status": integrated.audit["status"],
        "sources": [
            {"path": str(path.relative_to(root)), "sha256": _sha256(path)}
            for path in source_paths
        ],
    }
    (stage / "experiment_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    output.mkdir(parents=True, exist_ok=True)
    for path in stage.iterdir():
        path.replace(output / path.name)
    stage.rmdir()
    return output / report_path.name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--end", type=date.fromisoformat)
    args = parser.parse_args()
    report = run_report(args.root.resolve(), args.output, end=args.end)
    print(f"HTML report: {report}")


if __name__ == "__main__":
    main()
