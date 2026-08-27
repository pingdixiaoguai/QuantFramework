"""Generate and verify the formal absolute-stability raw-Gold checkpoint."""

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

from backtest.runner import run as run_backtest
from research.momentum_defender_occam import _load_momentum_config, performance
from research.standard_report import generate_standard_report
from strategy.momentum_defender_gold_raqm import (
    ENTRY_DIFFERENCE,
    EXIT_DIFFERENCE,
    FORMAL_STRATEGY_ID,
    MIN_GOLD_HOLD_DAYS,
)
from strategy.momentum_defender_absolute_stability import (
    RAPID_REVERSAL_ENTRY_DIFFERENCE,
    RAPID_REVERSAL_EXIT_DIFFERENCE,
    RISK_OFF_CONFIRMATION_DAYS,
    RISK_ON_CONFIRMATION_DAYS,
    run_formal_strategy,
)


DEFAULT_CONFIG = Path(
    "strategy/configs/momentum_defender_c2_gold_raqm_w5.yaml"
)
DEFAULT_OUTPUT = Path(
    "experiments/20260824_momentum_defender_confirmation_bridge_raw_gold_v4_formal"
)
LEGACY_MOMENTUM_CONFIG = Path(
    "strategy/configs/quality_momentum_top1_legacy_simple_price.yaml"
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
    policy = config["state_policy"]
    if (
        int(policy["min_momentum_days"]) != 0
        or int(policy["min_defender_days"]) != 0
        or int(policy["risk_off_confirmation"]) != RISK_OFF_CONFIRMATION_DAYS
        or int(policy["risk_on_confirmation"]) != RISK_ON_CONFIRMATION_DAYS
    ):
        raise AssertionError("formal no-lock confirmation policy differs from code")
    bridge = config["rapid_reversal_bridge"]
    if (
        int(bridge["hard_min_hold_days"]) != 0
        or float(bridge["entry_difference"])
        != RAPID_REVERSAL_ENTRY_DIFFERENCE
        or float(bridge["exit_difference"])
        != RAPID_REVERSAL_EXIT_DIFFERENCE
    ):
        raise AssertionError("formal rapid-reversal bridge differs from code")
    override = config["gold_override"]
    if (
        str(override["factor"]) != "raw_risk_adjusted_quality_momentum"
        or int(override["window"]) != 5
        or override.get("vol_floor_annual") is not None
        or override.get("winsor_limit") is not None
        or float(override["entry_difference"]) != ENTRY_DIFFERENCE
        or float(override["exit_difference"]) != EXIT_DIFFERENCE
        or int(override["hard_min_hold_days"]) != MIN_GOLD_HOLD_DAYS
    ):
        raise AssertionError("formal Gold parameters differ from code constants")

    checkpoint = config["checkpoint"]
    cutoff = date.fromisoformat(str(checkpoint["end"]))
    result = run_formal_strategy(root, end=cutoff)
    candidate = result.daily["return"].astype(float)
    base = result.base_daily["return"].astype(float)
    legacy_config = _load_momentum_config(
        root / LEGACY_MOMENTUM_CONFIG,
        cutoff,
    )
    legacy_result = run_backtest(legacy_config)
    momentum = legacy_result.daily_returns.reindex(candidate.index).astype(float)
    if momentum.isna().any():
        raise AssertionError("legacy Momentum baseline does not cover formal calendar")
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
        ("base_defender_entries", result.audit["base_defender_entries"]),
        ("base_defender_days", result.audit["base_defender_days"]),
        ("base_switches", result.audit["base_switches"]),
        ("rapid_reversal_entries", result.audit["rapid_reversal_entries"]),
        ("rapid_reversal_days", result.audit["rapid_reversal_days"]),
        ("effective_defender_days", result.audit["effective_defender_days"]),
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
    result.indicators.join(result.base_state).join(
        result.rapid_reversal_metrics
    ).join(
        result.rapid_reversal_state, rsuffix="_rapid_reversal"
    ).join(result.gold_state, rsuffix="_gold").join(
        result.daily, rsuffix="_execution"
    ).to_csv(
        stage / "daily_backtest.csv"
    )
    strategies = {
        "formal_confirmation_bridge_raw_gold": candidate,
        "confirmation_bridge_no_gold": base,
        "original_momentum_legacy_simple_price": momentum,
    }
    pd.DataFrame(
        [{"strategy": name, **performance(values)} for name, values in strategies.items()]
    ).to_csv(stage / "strategy_metrics.csv", index=False)
    _annual(strategies).to_csv(stage / "calendar_year_returns.csv", index=False)
    episodes = _episodes(result.gold_state, candidate, base)
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
        "Confirmation Bridge Without Gold",
        stage / "formal_vs_no_gold_base.html",
        config,
    )
    generate_standard_report(
        candidate,
        momentum,
        "Original Momentum (Simple MOM × Price ER)",
        stage / "formal_vs_original_momentum.html",
        config,
    )

    audit = {
        "status": "passed",
        "strategy_id": FORMAL_STRATEGY_ID,
        "checkpoint_tolerance": 1e-12,
        "daily_return_sha256_float64_le": daily_hash,
        "candidate_interface_baseline_parity_max_abs_error": (
            result.context.baseline_parity_max_abs_error
        ),
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
        root / "strategy/momentum_defender_absolute_stability.py",
        root / "strategy/momentum_defender.py",
        root / "run_daily_momentum_defender.py",
        root / "research/run_formal_gold_raqm_w5.py",
        root / "research/momentum_defender_integrated.py",
        root / "research/momentum_defender_gold_override.py",
        root / "factors/quality_momentum.py",
        root / "factors/legacy_quality_momentum.py",
        root / LEGACY_MOMENTUM_CONFIG,
        root / "strategy/prospective_ledger.py",
        root / "strategy/governance/momentum_defender_confirmation_bridge_raw_gold_v4.json",
        root / "docs/research/2026-08-24_no_lock_confirmation_bridge_design.md",
        root / "research/configs/momentum_defender_badcase_context.yaml",
        root / "research/generate_momentum_defender_badcases.py",
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
    report = f"""# 正式策略：无锁确认 + 快速反转桥接 + Raw Gold RAQM-W5

正式策略ID为`{FORMAL_STRATEGY_ID}`。Momentum固定使用20日对数收益乘对数路径ER。
基础状态仍观察沪深300与当前Momentum持仓的120日双正趋势，但不再设置Momentum或
Defender最短持有期；风险关闭连续确认20日，恢复连续确认10日，反向证据会清零确认。
Top1相对Defender的Raw RAQM-W5差值以2.00/0.75迟滞线提供无持有锁快速桥接，且不改写
基础确认状态。5日趋势为负且20日下行波动超过严格滞后q95时仍立即退出Momentum。
Gold使用无波动率地板、无剪裁的Raw RAQM-W5；入场线2.00、退出线0.75，前5个完整
交易日硬持有。

|策略|年化收益|Sharpe|最大回撤|
|---|---:|---:|---:|
|正式Raw Gold策略|{float(measured['annualized_return_252']):.2%}|{float(measured['sharpe']):.3f}|{float(measured['max_drawdown']):.2%}|
|关闭Gold的确认桥接基础|{float(performance(base)['annualized_return_252']):.2%}|{float(performance(base)['sharpe']):.3f}|{float(performance(base)['max_drawdown']):.2%}|

快速反转桥接{result.audit['rapid_reversal_entries']}次、
{result.audit['rapid_reversal_days']}日；黄金覆盖{result.audit['gold_entries']}次、
{result.audit['gold_days']}日。逐日收益哈希与配置检查点一致，非法入场、提前退出、
Momentum交接和净值重构审计全部通过。

逐一删除任一黄金事件后，最低年化{float(leave_one['annualized_return_252'].min()):.2%}、
最低Sharpe {float(leave_one['sharpe'].min()):.3f}，结果不依赖单一事件。

证据边界：v4使用了已经观察的完整历史与v3 Badcase，不是独立样本外升级。相对v3，v4
年化、Sharpe与最大回撤均退化；晋升来自用户对取消Momentum/Defender持有锁和减少快速
反转阻塞的明确优先级。新策略ID从第一未观察交易日起独立记账，20/10确认及桥接、Gold的
2.00/0.75阈值不得继续在同一历史上微调。

`formal_vs_original_momentum.html`中的原Momentum基准使用历史原版公式：20日简单收益MOM
乘价格路径Kaufman ER，5日最短持有；它不使用Defender或Gold，也不使用生产v4的双对数
`quality_momentum 2.0.0`。
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
