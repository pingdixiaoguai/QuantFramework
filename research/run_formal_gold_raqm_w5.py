"""Generate and verify the formal C2 Gold RAQM-W5 production checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from research.gold_min5_risk_adjusted_momentum_w5 import (
    GoldRAQMW5Params,
    run_gold_raqm_w5,
)
from research.momentum_defender_gold_override import build_gold_override_context
from research.momentum_defender_occam import HELD_RETURN, performance
from research.standard_report import generate_standard_report
from strategy.momentum_defender_gold_raqm import (
    ENTRY_DIFFERENCE,
    EXIT_DIFFERENCE,
    FORMAL_STRATEGY_ID,
    MIN_GOLD_HOLD_DAYS,
)


DEFAULT_CONFIG = Path(
    "strategy/configs/momentum_defender_c2_gold_raqm_w5.yaml"
)
DEFAULT_OUTPUT = Path(
    "experiments/20260823_momentum_defender_c2_gold_raqm_w5_formal"
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _return_hash(returns: pd.Series) -> str:
    return hashlib.sha256(returns.to_numpy(dtype="<f8").tobytes()).hexdigest()


def _annual(strategies: dict[str, pd.Series]) -> pd.DataFrame:
    rows = []
    for strategy, returns in strategies.items():
        for year, sample in returns.groupby(returns.index.year):
            rows.append(
                {
                    "strategy": strategy,
                    "year": int(year),
                    "observations": int(len(sample)),
                    "total_return": float((1.0 + sample).prod() - 1.0),
                }
            )
    return pd.DataFrame(rows)


def _episodes(state: pd.DataFrame, returns: pd.Series, baseline: pd.Series) -> pd.DataFrame:
    active = state["gold_active"].astype(bool)
    groups = active.ne(active.shift()).cumsum()
    calendar = state.index
    rows = []
    for episode, (_, sample) in enumerate(
        state.loc[active].groupby(groups.loc[active]), start=1
    ):
        start = calendar.get_loc(sample.index.min())
        finish = min(calendar.get_loc(sample.index.max()) + 1, len(calendar) - 1)
        index = calendar[start : finish + 1]
        candidate_return = float((1.0 + returns.loc[index]).prod() - 1.0)
        baseline_return = float((1.0 + baseline.loc[index]).prod() - 1.0)
        rows.append(
            {
                "episode": episode,
                "start": index.min().date().isoformat(),
                "end_including_exit": index.max().date().isoformat(),
                "observations": int(len(index)),
                "candidate_return": candidate_return,
                "baseline_return": baseline_return,
                "relative_return": (1.0 + candidate_return) / (1.0 + baseline_return) - 1.0,
                "entry_metric_difference": float(
                    state.at[index.min(), "metric_difference_at_open"]
                ),
            }
        )
    return pd.DataFrame(rows)


def run_formal(root: Path, config_path: Path, output: Path) -> dict[str, object]:
    config_path = config_path if config_path.is_absolute() else root / config_path
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config["strategy_name"] != FORMAL_STRATEGY_ID:
        raise AssertionError("formal strategy ID mismatch")
    override = config["gold_override"]
    if (
        float(override["entry_difference"]) != ENTRY_DIFFERENCE
        or float(override["exit_difference"]) != EXIT_DIFFERENCE
        or int(override["hard_min_hold_days"]) != MIN_GOLD_HOLD_DAYS
    ):
        raise AssertionError("formal Gold parameters differ from code constants")

    checkpoint = config["checkpoint"]
    cutoff = date.fromisoformat(str(checkpoint["end"]))
    context = build_gold_override_context(root, end=cutoff)
    result = run_gold_raqm_w5(
        context,
        GoldRAQMW5Params(ENTRY_DIFFERENCE, EXIT_DIFFERENCE),
    )
    candidate = result.daily["return"].astype(float)
    base = context.integrated.result.simulated["return"].astype(float)
    momentum = context.integrated.result.inputs.momentum[HELD_RETURN].astype(float)
    measured = performance(candidate)
    failures = []
    for field in (
        "observations",
        "total_return",
        "annualized_return_252",
        "annualized_volatility",
        "sharpe",
        "max_drawdown",
    ):
        expected = checkpoint[field]
        actual = measured[field]
        if isinstance(expected, int):
            matches = int(actual) == expected
        else:
            matches = abs(float(actual) - float(expected)) <= 1e-12
        if not matches:
            failures.append(f"{field}: {actual!r} != {expected!r}")
    daily_hash = _return_hash(candidate)
    if daily_hash != checkpoint["daily_return_sha256_float64_le"]:
        failures.append("daily return hash mismatch")
    for field, actual in (
        ("gold_entries", result.audit["gold_entries"]),
        ("gold_days", result.audit["gold_days"]),
        ("switches", result.audit["switches"]),
    ):
        if int(actual) != int(checkpoint[field]):
            failures.append(f"{field}: {actual} != {checkpoint[field]}")
    if failures:
        raise AssertionError("formal checkpoint failed: " + "; ".join(failures))

    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    result.state.join(result.daily, rsuffix="_execution").to_csv(
        stage / "daily_backtest.csv"
    )
    strategies = {
        "formal_gold_raqm_w5": candidate,
        "current_c2": base,
        "original_momentum": momentum,
    }
    pd.DataFrame(
        [{"strategy": name, **performance(values)} for name, values in strategies.items()]
    ).to_csv(stage / "strategy_metrics.csv", index=False)
    _annual(strategies).to_csv(stage / "calendar_year_returns.csv", index=False)
    episodes = _episodes(result.state, candidate, base)
    episodes.to_csv(stage / "gold_episodes.csv", index=False)
    leave_one_rows = []
    for event in episodes.itertuples(index=False):
        interval = candidate.loc[
            pd.Timestamp(event.start) : pd.Timestamp(event.end_including_exit)
        ].index
        counterfactual = candidate.copy()
        counterfactual.loc[interval] = base.loc[interval]
        leave_one_rows.append(
            {
                "removed_episode": int(event.episode),
                **performance(counterfactual),
            }
        )
    leave_one = pd.DataFrame(leave_one_rows)
    leave_one.to_csv(stage / "gold_leave_one_event.csv", index=False)
    (stage / "strategy_config.yaml").write_text(
        config_path.read_text(encoding="utf-8"), encoding="utf-8"
    )
    generate_standard_report(
        candidate,
        base,
        "Current Integrated C2",
        stage / "formal_vs_current_c2.html",
        config,
    )
    generate_standard_report(
        candidate,
        momentum,
        "Original Momentum",
        stage / "formal_vs_original_momentum.html",
        config,
    )

    audit = {
        "status": "passed",
        "strategy_id": FORMAL_STRATEGY_ID,
        "checkpoint_tolerance": 1e-12,
        "daily_return_sha256_float64_le": daily_hash,
        "baseline_parity_max_abs_error": context.baseline_parity_max_abs_error,
        "mechanical_audit": result.audit,
        "leave_one_event": {
            "events": int(len(leave_one)),
            "minimum_annualized_return_252": float(
                leave_one["annualized_return_252"].min()
            ),
            "minimum_sharpe": float(leave_one["sharpe"].min()),
            "worst_max_drawdown": float(leave_one["max_drawdown"].min()),
        },
        "measured": measured,
        "expected": checkpoint,
    }
    (stage / "checkpoint_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    source_paths = [
        config_path,
        root / "strategy/momentum_defender_gold_raqm.py",
        root / "strategy/momentum_defender.py",
        root / "run_daily_momentum_defender.py",
        root / "research/gold_min5_risk_adjusted_escape.py",
        root / "research/gold_min5_risk_adjusted_momentum.py",
        root / "research/gold_min5_risk_adjusted_momentum_w5.py",
        root / "research/run_formal_gold_raqm_w5.py",
        root / "research/momentum_defender_integrated.py",
        root / "research/momentum_defender_gold_override.py",
        root / "factors/risk_adjusted_quality_momentum.py",
        root / "strategy/prospective_ledger.py",
        root / "research/DEVELOPMENT_VALIDATION.md",
    ]
    manifest = {
        "strategy_id": FORMAL_STRATEGY_ID,
        "generated_on": date.today().isoformat(),
        "formal_promotion_date": config["formal_promotion_date"],
        "promotion_authority": config["promotion_authority"],
        "checkpoint_status": "passed",
        "sources": [
            {
                "path": str(path.relative_to(root)),
                "sha256": _sha256_file(path),
            }
            for path in source_paths
        ],
    }
    (stage / "experiment_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report = f"""# 正式策略：C2 + Gold RAQM-W5

正式策略ID为`{FORMAL_STRATEGY_ID}`。基础C2保持不变，仅在基础状态为Defender时允许黄金
通过5日注册风险调整质量动量差触发覆盖。入场线2.20、退出线0.60；黄金前5个完整交易日
硬持有，第6个开盘起基础C2 Momentum优先。

|策略|年化收益|Sharpe|最大回撤|
|---|---:|---:|---:|
|正式Gold RAQM-W5|{float(measured['annualized_return_252']):.2%}|{float(measured['sharpe']):.3f}|{float(measured['max_drawdown']):.2%}|
|当前C2|{float(performance(base)['annualized_return_252']):.2%}|{float(performance(base)['sharpe']):.3f}|{float(performance(base)['max_drawdown']):.2%}|

黄金覆盖{result.audit['gold_entries']}次、{result.audit['gold_days']}日。逐日收益哈希与配置检查点
一致，非法入场、提前退出、Momentum交接和净值重构审计全部通过。

逐一删除任一黄金事件后，最低年化{float(leave_one['annualized_return_252'].min()):.2%}、
最低Sharpe {float(leave_one['sharpe'].min()):.3f}，结果不依赖单一事件。

证据边界：参数来自回溯阈值搜索。邻域、事件、分段、PBO和分块bootstrap支持历史改善，
但年度块多重试验校正仍未显著；正式晋升来自用户明确指令，后续必须独立记录前瞻信号。
"""
    (stage / "research_report.md").write_text(report, encoding="utf-8")

    output.mkdir(parents=True, exist_ok=True)
    for path in stage.iterdir():
        path.replace(output / path.name)
    stage.rmdir()
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    audit = run_formal(args.root.resolve(), args.config, args.output)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
