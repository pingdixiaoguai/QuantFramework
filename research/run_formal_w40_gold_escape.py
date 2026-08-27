"""Generate and verify the promoted W40 Gold-only QM20 escape checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from datetime import date
from pathlib import Path

import pandas as pd
import yaml

from backtest.runner import run as run_backtest
from defender.w40_reversal_full_equity import FORMAL_DIVIDEND_ASSETS
from research.formal_strategy_holdings import build_formal_target_schedule
from research.momentum_defender_occam import (
    HELD_RETURN,
    _load_momentum_config,
    performance,
)
from research.run_momentum_defender_log_qm_robust import _event_stress
from research.standard_report import generate_standard_report
from strategy.momentum_defender_w40_gold_escape import (
    DEFENDER_ELIGIBILITY_DAYS,
    FORMAL_STRATEGY_ID,
    GOLD_ASSET,
    GOLD_ENTRY_X,
    GOLD_EXIT_Y,
    GOLD_HARD_HOLD_DAYS,
    IMMEDIATE_DEFENDER_ENTRY_GOLD_VETO,
    run_formal_strategy,
)
from strategy.momentum_defender_w40_loss import (
    DEFENDER_ENTRY_CONFIRMATION_DAYS,
    DEFENDER_ENTRY_PERCENTILE,
    DEFENDER_LOCK_DAYS,
    MOMENTUM_LOCK_DAYS,
    MOMENTUM_RECOVERY_CONFIRMATION_DAYS,
    MOMENTUM_RECOVERY_PERCENTILE,
)


DEFAULT_CONFIG = Path(
    "experiments/20260826_momentum_defender_w40_gold_qm20_escape_v3_formal/"
    "strategy_config.yaml"
)
DEFAULT_OUTPUT = Path(
    "experiments/20260826_momentum_defender_w40_gold_qm20_escape_v3_formal"
)
RESEARCH_SELECTED = Path(
    "experiments/20260826_momentum_defender_immediate_gold_entry_veto/candidate_daily.parquet"
)
RESEARCH_SELECTED_COLUMN = "candidate_return"
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


def _validate_config(config: dict) -> None:
    if config["strategy_name"] != FORMAL_STRATEGY_ID:
        raise AssertionError("formal Gold escape strategy ID mismatch")
    if config["strategy_mode"] != "w40_gold_qm20_escape":
        raise AssertionError("formal Gold escape strategy mode mismatch")
    if str(config.get("backtest_start")) != "2013-01-01":
        raise AssertionError("formal backtest start must remain 2013-01-01")
    gate = config["regime_gate"]
    expected_gate = {
        "window": 40,
        "defender_entry_percentile": DEFENDER_ENTRY_PERCENTILE,
        "momentum_recovery_percentile": MOMENTUM_RECOVERY_PERCENTILE,
    }
    for field, expected in expected_gate.items():
        if gate[field] != expected:
            raise AssertionError(f"formal gate mismatch for {field}")
    state = config["state_policy"]
    expected_state = {
        "momentum_lock_days": MOMENTUM_LOCK_DAYS,
        "defender_lock_days": DEFENDER_LOCK_DAYS,
        "defender_entry_confirmation_days": DEFENDER_ENTRY_CONFIRMATION_DAYS,
        "momentum_recovery_confirmation_days": MOMENTUM_RECOVERY_CONFIRMATION_DAYS,
    }
    for field, expected in expected_state.items():
        if state[field] != expected:
            raise AssertionError(f"formal state mismatch for {field}")
    selector = config["defender_selector"]
    if not (
        selector["frequency"] == "monthly"
        and selector["lookback_sessions"] == 40
        and selector["direction"] == "lowest"
        and tuple(selector["candidate_assets"]) == FORMAL_DIVIDEND_ASSETS
    ):
        raise AssertionError("formal Defender selector mismatch")
    position = config["defender_position"]
    if (
        position["selected_dividend_equity_weight"] != 1.0
        or position["government_bond_511260_weight"] != 0.0
    ):
        raise AssertionError("formal Defender position is not 100% dividend")
    escape = config["gold_escape"]
    expected_escape = {
        "enabled_assets": [GOLD_ASSET],
        "defender_eligibility_days": DEFENDER_ELIGIBILITY_DAYS,
        "gold_entry_x": GOLD_ENTRY_X,
        "gold_exit_y": GOLD_EXIT_Y,
        "gold_hard_hold_days": GOLD_HARD_HOLD_DAYS,
        "immediate_defender_entry_gold_veto": (
            IMMEDIATE_DEFENDER_ENTRY_GOLD_VETO
        ),
    }
    for field, expected in expected_escape.items():
        if escape[field] != expected:
            raise AssertionError(f"formal Gold escape mismatch for {field}")


def _checkpoint_failures(
    measured: dict[str, float],
    checkpoint: dict,
    audit: dict,
    daily_hash: str,
) -> list[str]:
    failures = []
    for field in (
        "observations",
        "total_return",
        "cagr_calendar",
        "annualized_return_252",
        "annualized_volatility",
        "sharpe",
        "max_drawdown",
    ):
        expected = checkpoint[field]
        actual = measured[field]
        matches = (
            int(actual) == expected
            if isinstance(expected, int)
            else abs(float(actual) - float(expected)) <= 1e-12
        )
        if not matches:
            failures.append(f"{field}: {actual!r} != {expected!r}")
    if daily_hash != checkpoint["daily_return_sha256_float64_le"]:
        failures.append("daily return hash mismatch")
    for field in (
        "escape_entries",
        "lock_break_entries",
        "escape_days",
        "immediate_entry_veto_entries",
        "candidate_switches",
    ):
        if int(audit[field]) != int(checkpoint[field]):
            failures.append(f"{field}: {audit[field]} != {checkpoint[field]}")
    return failures


def run_formal(root: Path, config_path: Path, output: Path) -> dict[str, object]:
    config_path = config_path if config_path.is_absolute() else root / config_path
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    _validate_config(config)
    checkpoint = config["checkpoint"]
    cutoff = date.fromisoformat(str(checkpoint["end"]))
    formal = run_formal_strategy(root, end=cutoff)
    legacy_start_formal = run_formal_strategy(
        root,
        start=date(2019, 1, 18),
        end=cutoff,
    )
    candidate = formal.daily["return"].astype(float)
    rollback = formal.base.daily["return"].astype(float)
    log_momentum = formal.context.integrated.result.inputs.momentum[
        HELD_RETURN
    ].astype(float)
    legacy_config = _load_momentum_config(root / LEGACY_MOMENTUM_CONFIG, cutoff)
    legacy = run_backtest(legacy_config).daily_returns.reindex(candidate.index).astype(float)
    if legacy.isna().any():
        raise AssertionError("legacy Momentum baseline does not cover formal calendar")

    measured = performance(candidate)
    daily_hash = _return_hash(candidate)
    failures = _checkpoint_failures(
        measured, checkpoint, dict(formal.audit), daily_hash
    )
    selected = pd.read_parquet(root / RESEARCH_SELECTED)[
        RESEARCH_SELECTED_COLUMN
    ].astype(float)
    if not selected.index.equals(candidate.index):
        failures.append("research selected calendar mismatch")
        research_parity = float("inf")
    else:
        research_parity = float((candidate - selected).abs().max())
        if research_parity > 1e-14:
            failures.append(f"research parity mismatch: {research_parity:.3e}")
    if failures:
        raise AssertionError("formal checkpoint failed: " + "; ".join(failures))

    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    daily = formal.escape.state.copy()
    daily["w40_downside_log_loss_at_open"] = formal.raw_loss_at_open
    daily["w40_loss_percentile_at_open"] = formal.score_at_open
    daily = daily.join(formal.daily, rsuffix="_execution")
    daily = daily.join(build_formal_target_schedule(formal).add_prefix("target_"))
    daily.to_csv(stage / "daily_backtest.csv")
    daily.to_parquet(stage / "daily_backtest.parquet")

    strategies = {
        "formal_w40_gold_qm20_escape": candidate,
        "formal_w40_gold_qm20_escape_2019_start": (
            legacy_start_formal.daily["return"].astype(float)
        ),
        "rollback_w40_full_equity": rollback,
        "log_quality_momentum": log_momentum,
        "original_momentum_legacy_simple_price": legacy,
    }
    pd.DataFrame(
        [
            {"strategy": name, **performance(values)}
            for name, values in strategies.items()
        ]
    ).to_csv(stage / "strategy_metrics.csv", index=False)
    _annual(strategies).to_csv(stage / "calendar_year_returns.csv", index=False)

    candidate_target = formal.daily["candidate"].astype(str)
    rollback_target = formal.base.daily["candidate"].astype(str)
    events, leave_events, deleted, event_summary = _event_stress(
        candidate,
        rollback,
        candidate_target,
        rollback_target,
        [1, 2, 3],
    )
    events.to_csv(stage / "events_vs_rollback_w40_full_equity.csv", index=False)
    leave_events.to_csv(
        stage / "leave_one_event_vs_rollback_w40_full_equity.csv", index=False
    )
    deleted.to_csv(stage / "top_positive_event_deletion.csv", index=False)
    (stage / "strategy_config.yaml").write_text(
        config_path.read_text(encoding="utf-8"), encoding="utf-8"
    )
    generate_standard_report(
        candidate,
        legacy,
        "Original Momentum (Simple MOM × Price ER)",
        stage / "formal_backtest.html",
        config,
    )
    configured_start = str(config["backtest_start"])
    dated_current_report = (
        f"formal_backtest_{configured_start}_to_{cutoff.isoformat()}.html"
    )
    shutil.copyfile(
        stage / "formal_backtest.html",
        stage / dated_current_report,
    )
    legacy_start_returns = legacy_start_formal.daily["return"].astype(float)
    legacy_start_benchmark = legacy.reindex(legacy_start_returns.index)
    if legacy_start_benchmark.isna().any():
        raise AssertionError("legacy Momentum does not cover the 2019-start report")
    dated_2019_report = (
        f"formal_backtest_2019-01-18_to_{cutoff.isoformat()}.html"
    )
    generate_standard_report(
        legacy_start_returns,
        legacy_start_benchmark,
        "Original Momentum (Simple MOM × Price ER)",
        stage / dated_2019_report,
        config,
    )
    shutil.copyfile(
        stage / "formal_backtest.html", stage / "formal_vs_original_momentum.html"
    )
    generate_standard_report(
        candidate,
        rollback,
        "Rollback W40 + 100% Dividend Defender",
        stage / "formal_vs_rollback_w40_full_equity.html",
        config,
    )
    audit = {
        "status": "passed",
        "strategy_id": FORMAL_STRATEGY_ID,
        "checkpoint_tolerance": 1e-12,
        "daily_return_sha256_float64_le": daily_hash,
        "research_selected_daily_max_abs_error": research_parity,
        "mechanical_audit": formal.audit,
        "events_vs_rollback_w40_full_equity": event_summary,
        "measured": measured,
        "expected": checkpoint,
    }
    (stage / "checkpoint_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    source_paths = [
        config_path,
        root / "defender/w40_reversal_full_equity.py",
        root / "strategy/momentum_defender_w40_gold_escape.py",
        root / "strategy/momentum_defender_w40_full_equity.py",
        root / "research/momentum_defender_dividend_universe.py",
        root / "research/momentum_defender_occam.py",
        root / "research/defender_curve_momentum.py",
        root / "research/momentum_defender_w40_asset_specific_escape.py",
        root / "research/formal_strategy_holdings.py",
        root / "research/run_formal_w40_gold_escape.py",
        root / "research/standard_report.py",
        root / "backtest/report.py",
        root / "research/run_momentum_defender_immediate_gold_entry_veto.py",
        root / "research/configs/momentum_defender_immediate_gold_entry_veto.yaml",
        root / RESEARCH_SELECTED,
        root / "run_daily_momentum_defender.py",
        root / "scripts/run_daily_job.py",
        root / "factors/quality_momentum.py",
        root / LEGACY_MOMENTUM_CONFIG,
        root / "strategy/configs/quality_momentum_top1.yaml",
        root / "strategy/configs/momentum_defender_w40_full_equity.yaml",
        root / "strategy/governance/momentum_defender_w40_gold_qm20_escape_v3.json",
        root / "strategy/governance/momentum_defender_w40_gold_qm20_escape_v2.json",
        root / "strategy/governance/momentum_defender_w40_reversal_full_equity_v2.json",
        root / "strategy/governance/momentum_defender_w40_gold_qm20_escape_v1.json",
        root / "research/configs/momentum_defender_badcase_context.yaml",
        root / "research/configs/strategy_drawdown_badcase_context.yaml",
        root / "research/generate_momentum_defender_badcases.py",
        root / "research/generate_strategy_drawdown_badcases.py",
        root / "docs/research/momentum_defender_badcases.md",
        root / "docs/research/momentum_defender_drawdown_badcases.md",
        root / "docs/research/2026-08-26_immediate_gold_entry_veto.md",
        root / "docs/research/2026-08-26_immediate_gold_entry_veto_formal_promotion.md",
        root / "README.md",
        root / "research/README.md",
        root / "strategy_changelog.md",
        root / "strategy/AGENTS.md",
        root / "defender/AGENTS.md",
        root / "ops/tencent-cloud/README.md",
        root / "ops/tencent-cloud/quant-daily.service",
        root / "strategy/tests/test_momentum_defender_w40_gold_escape.py",
        root / "research/tests/test_standard_report.py",
        root / "tests/test_run_daily_momentum_defender.py",
        root / "tests/test_run_daily_job.py",
    ]
    manifest = {
        "strategy_id": FORMAL_STRATEGY_ID,
        "generated_on": date.today().isoformat(),
        "formal_promotion_date": config["formal_promotion_date"],
        "promotion_authority": config["promotion_authority"],
        "checkpoint_status": "passed",
        "sources": [
            {"path": str(path.relative_to(root)), "sha256": _sha256_file(path)}
            for path in source_paths
        ],
    }
    (stage / "experiment_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    rollback_metrics = performance(rollback)
    report = f"""# 正式策略：W40 + 100%红利Defender + 黄金QM20逃生

策略ID：`{FORMAL_STRATEGY_ID}`。W40与Defender候选池保持冻结。W40切入Defender当日若
黄金已经满足Top1与QM20差值条件，实际持仓立即黄金；其他情况下实际Defender满5日后，仅黄金
Top1可在QM20相对连续Defender净值高于{GOLD_ENTRY_X:+.3f}时破锁。黄金硬持有
{GOLD_HARD_HOLD_DAYS}日，之后低于{GOLD_EXIT_Y:+.3f}或Top1不再是黄金时返回Defender。

正式检查点为{measured['annualized_return_252']:.2%}年化、Sharpe
{measured['sharpe']:.3f}、MDD {measured['max_drawdown']:.2%}；直接回滚为
{rollback_metrics['annualized_return_252']:.2%}/{rollback_metrics['sharpe']:.3f}。
历史共{formal.audit['escape_entries']}次黄金逃生、{formal.audit['escape_days']}个逃生日，
其中{formal.audit['lock_break_entries']}次打破未满30日的基础Defender锁。

标准HTML同时保留`{dated_current_report}`和`{dated_2019_report}`；前者为当前固定区间，后者
在2019-01-18重新初始化状态并保留原报告语义。`formal_backtest.html`始终指向当前固定区间。

正式实现与研究选中路径逐日最大误差为{research_parity:.1e}。本次晋升来自用户明确指令；
回溯研究未证明统计显著优于回滚W40，证据边界见晋升报告与`checkpoint_audit.json`。
"""
    (stage / "formal_report.md").write_text(report, encoding="utf-8")
    if output.exists():
        shutil.rmtree(output)
    stage.rename(output)
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    audit = run_formal(root, args.config, args.output.resolve())
    print(json.dumps(audit["measured"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
