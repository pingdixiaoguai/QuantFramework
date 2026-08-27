"""Generate standard reports for the ChiNext-q95 C2 counterfactual."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
from dataclasses import replace
from datetime import date
from pathlib import Path

import pandas as pd

from research.momentum_defender_c2 import load_frozen_c2_config, run_frozen_c2
from research.momentum_defender_occam import (
    HELD_RETURN,
    MOMENTUM_ASSETS,
    apply_state_schedule,
    performance,
    simulate_switch,
)
from research.standard_report import generate_standard_report


DEFAULT_OUTPUT = Path(
    "experiments/20260821_momentum_defender_c2_chinext_q95_counterfactual"
)
Q90_PARENT_CONFIG_PATH = (
    Path(__file__).resolve().parent
    / "configs"
    / "momentum_defender_c2_frozen_v1.yaml"
)
COUNTERFACTUAL_ID = "momentum_defender_c2_chinext_q95_counterfactual"


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


def _metric_row(strategy: str, returns: pd.Series) -> dict[str, object]:
    return {"strategy": strategy, **performance(returns)}


def run_experiment(
    root: Path,
    config_path: Path,
    final_output: Path,
    defender_dir: Path | None = None,
) -> None:
    frozen = load_frozen_c2_config(config_path)
    quantiles = dict(frozen.asset_quantiles)
    quantiles["159915.SZ"] = 0.95
    q95_config = replace(
        frozen,
        strategy_id=COUNTERFACTUAL_ID,
        status="counterfactual_not_frozen",
        asset_quantiles=quantiles,
        checkpoint={},
    )
    if any(
        q95_config.asset_quantiles[asset] != frozen.asset_quantiles[asset]
        for asset in MOMENTUM_ASSETS
        if asset != "159915.SZ"
    ):
        raise AssertionError("q95 counterfactual changed a non-ChiNext quantile")
    if q95_config.asset_quantiles["159915.SZ"] != 0.95:
        raise AssertionError("q95 counterfactual did not set ChiNext to q95")

    final_output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(prefix=f".{final_output.name}.staging-", dir=final_output.parent)
    )
    q95 = run_frozen_c2(
        root,
        q95_config,
        defender_dir=defender_dir,
        end=frozen.research_cutoff,
    )
    calendar = q95.inputs.calendar
    no_cap_state = apply_state_schedule(
        q95.slow_signal,
        pd.Series(False, index=calendar),
        calendar,
        q95_config.min_hold_days,
        emergency_override=q95_config.emergency_override,
        initial_risk_on=True,
    )
    no_cap = simulate_switch(
        q95.inputs.momentum,
        q95.inputs.defender,
        no_cap_state["risk_on"],
        initial_previous_state=q95_config.initial_previous_sleeve,
    )
    momentum = q95.inputs.momentum[HELD_RETURN].astype(float)

    reports = {
        "momentum_defender_c2_chinext_q95_vs_no_cap.html": (
            no_cap["return"],
            "No-cap Slow-gate Fusion",
        ),
        "momentum_defender_c2_chinext_q95_vs_original_momentum.html": (
            momentum,
            "Original Momentum Strategy",
        ),
    }
    report_config = q95_config.serializable()
    report_config["counterfactual_notice"] = (
        "Only ChiNext quantile changed from frozen q90 to hindsight q95"
    )
    for filename, (benchmark, benchmark_name) in reports.items():
        generate_standard_report(
            q95.simulated["return"],
            benchmark,
            benchmark_name,
            stage / filename,
            report_config,
        )

    metrics = pd.DataFrame(
        [
            _metric_row("c2_chinext_q95", q95.simulated["return"]),
            _metric_row("no_cap_fusion", no_cap["return"]),
            _metric_row("original_momentum", momentum),
        ]
    )
    metrics.to_csv(stage / "strategy_metrics.csv", index=False)
    years: list[dict[str, object]] = []
    strategies = {
        "c2_chinext_q95": q95.simulated["return"],
        "no_cap_fusion": no_cap["return"],
        "original_momentum": momentum,
    }
    for year in sorted(calendar.year.unique()):
        for strategy, returns in strategies.items():
            sample = returns.loc[returns.index.year == year]
            years.append(
                {
                    "year": int(year),
                    "strategy": strategy,
                    "total_return": float((1.0 + sample).prod() - 1.0),
                }
            )
    pd.DataFrame(years).to_csv(stage / "calendar_year_returns.csv", index=False)
    q95.daily.to_csv(stage / "daily_backtest.csv")
    (stage / "counterfactual_config.json").write_text(
        json.dumps(q95_config.serializable(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    q95_metrics = performance(q95.simulated["return"])
    no_cap_metrics = performance(no_cap["return"])
    momentum_metrics = performance(momentum)
    q95_2025 = float(
        (1.0 + q95.simulated["return"].loc["2025-01-01":"2025-12-31"]).prod()
        - 1.0
    )
    no_cap_2025 = float(
        (1.0 + no_cap["return"].loc["2025-01-01":"2025-12-31"]).prod() - 1.0
    )
    momentum_2025 = float(
        (1.0 + momentum.loc["2025-01-01":"2025-12-31"]).prod() - 1.0
    )
    summary = f"""# C2创业板q95反事实回测

该版本只把当前冻结C2的创业板分位数从q90改为q95，其他参数、全历史扩展分位数、30日状态锁和开盘执行口径全部不变。它是全样本事后反事实，不替代`momentum_defender_c2_frozen_v1`。

|方案|年化收益|Sharpe|最大回撤|
|---|---:|---:|---:|
|C2创业板q95|{q95_metrics['annualized_return_252']:.2%}|{q95_metrics['sharpe']:.3f}|{q95_metrics['max_drawdown']:.2%}|
|无cap融合|{no_cap_metrics['annualized_return_252']:.2%}|{no_cap_metrics['sharpe']:.3f}|{no_cap_metrics['max_drawdown']:.2%}|
|原动量策略|{momentum_metrics['annualized_return_252']:.2%}|{momentum_metrics['sharpe']:.3f}|{momentum_metrics['max_drawdown']:.2%}|

2025年：C2创业板q95 {q95_2025:+.2%}，无cap融合 {no_cap_2025:+.2%}，原动量策略 {momentum_2025:+.2%}。

q95相对无cap全样本年化高{q95_metrics['annualized_return_252'] - no_cap_metrics['annualized_return_252']:+.2%}，2025年相对原动量高{q95_2025 - momentum_2025:+.2%}，但2025年仍比无cap低{q95_2025 - no_cap_2025:+.2%}。q90和q95在2019—2023年路径相同，q95优势来自已经观察到的后段历史，不能解释为独立样本外证据。
"""
    (stage / "research_report.md").write_text(summary, encoding="utf-8")

    deliverable = defender_dir or frozen.defender_deliverable_dir
    input_files = [
        config_path,
        deliverable / frozen.defender_switch_returns_file,
        root / frozen.momentum_config_path,
        *[root / "data/db" / f"{asset}.parquet" for asset in MOMENTUM_ASSETS],
    ]
    code_files = [
        root / "research/momentum_defender_c2.py",
        root / "research/momentum_volatility.py",
        root / "research/standard_report.py",
        root / "research/run_momentum_defender_c2_chinext_q95.py",
        root / "research/momentum_defender_occam.py",
    ]
    manifest = {
        "experiment": COUNTERFACTUAL_ID,
        "generated_on": date.today().isoformat(),
        "research_cutoff": frozen.research_cutoff.isoformat(),
        "git_branch": _git(root, "branch", "--show-current"),
        "git_commit": _git(root, "rev-parse", "HEAD"),
        "frozen_parent_strategy": frozen.strategy_id,
        "changed_parameter": {"asset": "159915.SZ", "from": 0.90, "to": 0.95},
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
    parser.add_argument("--config", type=Path, default=Q90_PARENT_CONFIG_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--defender-dir", type=Path)
    args = parser.parse_args()
    run_experiment(
        args.root,
        args.config,
        args.output,
        defender_dir=args.defender_dir,
    )


if __name__ == "__main__":
    main()
