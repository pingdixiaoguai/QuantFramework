"""Generate and verify the current W40/QM40 threshold v5 checkpoint."""

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
from defender.w40_qm_reversal_full_equity import (
    FORMAL_DEFENDER_STRATEGY_ID,
)
from research.formal_strategy_holdings import build_formal_target_schedule
from research.momentum_defender_occam import (
    HELD_RETURN,
    _load_momentum_config,
    performance,
)
from research.standard_report import generate_standard_report
from strategy.momentum_defender_w40_qm40_signed_exit import (
    DEFENDER_ENTRY_PERCENTILE,
    DEFENDER_FALLBACK_LOCK_DAYS,
    DEFENDER_MINIMUM_DAYS,
    MOMENTUM_RECOVERY_PERCENTILE,
    QM40_RECOVERY_CONFIRMATION_DAYS,
    W40_PERCENTILE_HISTORY,
    run_formal_strategy as run_v4_rollback,
)
from strategy.momentum_defender_w40_qm40_threshold import (
    FORMAL_STRATEGY_ID,
    QM40_RECOVERY_THRESHOLD,
    run_formal_strategy,
)


DEFAULT_CONFIG = Path("strategy/configs/momentum_defender_w40_gold_escape.yaml")
DEFAULT_OUTPUT = Path(
    "experiments/20260827_momentum_defender_w40_qm40_threshold_v5_formal"
)
RESEARCH_DAILY = Path(
    "experiments/20260826_qm40_recovery_threshold_search_2019/"
    "daily_returns.parquet"
)
LEGACY_CONFIG = Path(
    "strategy/configs/quality_momentum_top1_legacy_simple_price.yaml"
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _return_hash(returns: pd.Series) -> str:
    return hashlib.sha256(
        returns.to_numpy(dtype="<f8").tobytes()
    ).hexdigest()


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
        raise AssertionError("formal v5 strategy ID mismatch")
    if config["strategy_mode"] != "w40_qm40_threshold":
        raise AssertionError("formal v5 strategy mode mismatch")
    if str(config["backtest_start"]) != "2013-01-01":
        raise AssertionError("formal v5 start must remain 2013-01-01")
    gate = config["regime_gate"]
    expected_gate = {
        "window": 40,
        "percentile_history": "rolling_756_strict_lag",
        "percentile_min_history": 252,
        "defender_entry_percentile": DEFENDER_ENTRY_PERCENTILE,
        "momentum_recovery_percentile": MOMENTUM_RECOVERY_PERCENTILE,
    }
    for field, expected in expected_gate.items():
        if gate[field] != expected:
            raise AssertionError(f"formal v5 gate mismatch for {field}")
    state = config["state_policy"]
    expected_state = {
        "momentum_lock_days": 30,
        "defender_fallback_lock_days": DEFENDER_FALLBACK_LOCK_DAYS,
        "defender_minimum_days_for_qm40_recovery": DEFENDER_MINIMUM_DAYS,
        "qm40_recovery_confirmation_days": QM40_RECOVERY_CONFIRMATION_DAYS,
        "qm40_recovery_threshold": QM40_RECOVERY_THRESHOLD,
    }
    for field, expected in expected_state.items():
        if state[field] != expected:
            raise AssertionError(f"formal v5 state mismatch for {field}")
    selector = config["defender_selector"]
    if not (
        selector["score"] == "quality_momentum"
        and selector["lookback_sessions"] == 40
        and selector["direction"] == "lowest"
    ):
        raise AssertionError("formal v5 Defender selector mismatch")
    if config["base_strategy"]["defender_strategy_id"] != (
        FORMAL_DEFENDER_STRATEGY_ID
    ):
        raise AssertionError("formal v5 Defender strategy ID mismatch")


def _checkpoint_failures(
    measured: dict[str, object],
    checkpoint: dict,
    audit: dict,
    daily_hash: str,
) -> list[str]:
    failures: list[str] = []
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
    if int(audit["base_audit"]["qm40_early_recoveries"]) != int(
        checkpoint["qm40_early_recoveries"]
    ):
        failures.append("QM40 early recovery count mismatch")
    return failures


def run_formal(root: Path, config_path: Path, output: Path) -> dict[str, object]:
    config_path = config_path if config_path.is_absolute() else root / config_path
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    _validate_config(config)
    checkpoint = config["checkpoint"]
    cutoff = date.fromisoformat(str(checkpoint["end"]))
    formal = run_formal_strategy(root, end=cutoff)
    formal_2019 = run_formal_strategy(
        root, start=date(2019, 1, 18), end=cutoff
    )
    rollback = run_v4_rollback(root, end=cutoff)
    candidate = formal.daily["return"].astype(float)
    candidate_2019 = formal_2019.daily["return"].astype(float)
    rollback_returns = rollback.daily["return"].astype(float)
    log_momentum = formal.context.integrated.result.inputs.momentum[
        HELD_RETURN
    ].astype(float)
    legacy_config = _load_momentum_config(root / LEGACY_CONFIG, cutoff)
    legacy = run_backtest(legacy_config).daily_returns.reindex(candidate.index)
    if legacy.isna().any():
        raise AssertionError("Original Momentum does not cover formal v5 calendar")

    measured = performance(candidate)
    daily_hash = _return_hash(candidate)
    failures = _checkpoint_failures(
        measured, checkpoint, dict(formal.audit), daily_hash
    )
    research = pd.read_parquet(root / RESEARCH_DAILY)[
        "qm40_threshold_+0.00750"
    ].astype(float)
    if not research.index.equals(candidate_2019.index):
        failures.append("2019 research calendar mismatch")
        research_parity = float("inf")
    else:
        research_parity = float((candidate_2019 - research).abs().max())
        if research_parity > 1e-14:
            failures.append(f"2019 research parity mismatch: {research_parity:.3e}")
    validation = config["validation"]
    if _return_hash(candidate_2019) != validation[
        "primary_daily_return_sha256_float64_le"
    ]:
        failures.append("2019 validation hash mismatch")
    if failures:
        raise AssertionError("formal v5 checkpoint failed: " + "; ".join(failures))

    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent)
    )
    daily = formal.state.add_prefix("base_")
    daily = daily.join(formal.escape.state.add_prefix("escape_"))
    daily = daily.join(formal.daily, rsuffix="_execution")
    daily = daily.join(build_formal_target_schedule(formal).add_prefix("target_"))
    daily.to_csv(stage / "daily_backtest.csv")
    daily.to_parquet(stage / "daily_backtest.parquet")

    strategies = {
        "formal_v5": candidate,
        "formal_v5_2019_restart": candidate_2019,
        "rollback_v4": rollback_returns,
        "log_quality_momentum": log_momentum.reindex(candidate.index),
        "original_momentum": legacy.astype(float),
    }
    metrics = pd.DataFrame(
        [
            {"strategy": name, **performance(values)}
            for name, values in strategies.items()
        ]
    )
    metrics.to_csv(stage / "strategy_metrics.csv", index=False)
    _annual(strategies).to_csv(stage / "calendar_year_returns.csv", index=False)
    (stage / "strategy_config.yaml").write_text(
        config_path.read_text(encoding="utf-8"), encoding="utf-8"
    )

    generate_standard_report(
        candidate,
        legacy.astype(float),
        "Original Momentum (Simple MOM × Price ER)",
        stage / "formal_backtest.html",
        config,
    )
    dated_2013 = f"formal_backtest_2013-01-01_to_{cutoff.isoformat()}.html"
    shutil.copyfile(stage / "formal_backtest.html", stage / dated_2013)
    legacy_2019 = legacy.reindex(candidate_2019.index)
    generate_standard_report(
        candidate_2019,
        legacy_2019,
        "Original Momentum (Simple MOM × Price ER)",
        stage / f"formal_backtest_2019-01-18_to_{cutoff.isoformat()}.html",
        config,
    )
    rollback_aligned = rollback_returns.reindex(candidate.index)
    generate_standard_report(
        candidate,
        rollback_aligned,
        "Rollback Production v4 (QM40 > 0)",
        stage / "formal_vs_rollback_v4.html",
        config,
    )

    audit = {
        "status": "passed",
        "strategy_id": FORMAL_STRATEGY_ID,
        "checkpoint_tolerance": 1e-12,
        "daily_return_sha256_float64_le": daily_hash,
        "research_2019_max_abs_error": research_parity,
        "mechanical_audit": formal.audit,
        "measured": measured,
        "expected": checkpoint,
        "validation_2019": performance(candidate_2019),
        "rollback_v4": performance(rollback_returns),
    }
    (stage / "checkpoint_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    source_paths = [
        config_path,
        root / "defender/w40_qm_reversal_full_equity.py",
        root / "defender/w40_reversal_full_equity.py",
        root / "strategy/momentum_defender_w40_qm40_signed_exit.py",
        root / "strategy/momentum_defender_w40_qm40_threshold.py",
        root / "strategy/momentum_defender_w40_full_equity.py",
        root / "strategy/momentum_defender_w40_gold_escape.py",
        root / "research/run_formal_w40_qm40_signed_exit.py",
        root / "research/run_formal_w40_qm40_threshold.py",
        root / "research/run_qm40_recovery_threshold_search_2019.py",
        root / "research/configs/qm40_recovery_threshold_search_2019.yaml",
        root / RESEARCH_DAILY,
        root / "research/formal_strategy_holdings.py",
        root / "research/standard_report.py",
        root / "backtest/report.py",
        root / "run_daily_momentum_defender.py",
        root / "scripts/run_daily_job.py",
        root / "factors/quality_momentum.py",
        root / LEGACY_CONFIG,
        root / "strategy/configs/quality_momentum_top1.yaml",
        root / "strategy/governance/momentum_defender_w40_qm40_threshold_v5.json",
        root / "strategy/governance/momentum_defender_w40_qm40_signed_exit_v4.json",
        root / "strategy/governance/momentum_defender_w40_gold_qm20_escape_v3.json",
        root / "research/configs/momentum_defender_badcase_context.yaml",
        root / "research/configs/strategy_drawdown_badcase_context.yaml",
        root / "research/generate_momentum_defender_badcases.py",
        root / "research/generate_strategy_drawdown_badcases.py",
        root / "docs/research/momentum_defender_badcases.md",
        root / "docs/research/momentum_defender_drawdown_badcases.md",
        root / "docs/research/2026-08-26_qm40_recovery_threshold_search_2019.md",
        root / "docs/research/2026-08-27_qm40_recovery_threshold_v5_formal_promotion.md",
        root / "README.md",
        root / "research/README.md",
        root / "strategy_changelog.md",
        root / "strategy/AGENTS.md",
        root / "defender/AGENTS.md",
        root / "notification/AGENTS.md",
        root / "ops/tencent-cloud/README.md",
        root / "ops/tencent-cloud/quant-daily.service",
        root / "strategy/tests/test_momentum_defender_w40_qm40_signed_exit.py",
        root / "strategy/tests/test_momentum_defender_w40_qm40_threshold.py",
        root / "defender/tests/test_w40_qm_reversal_full_equity.py",
        root / "tests/test_run_daily_momentum_defender.py",
        root / "tests/test_run_daily_job.py",
    ]
    missing = [path for path in source_paths if not path.exists()]
    if missing:
        raise AssertionError(f"formal v5 manifest sources missing: {missing}")
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

    rollback_metrics = performance(rollback_returns)
    primary_metrics = performance(candidate_2019)
    report = f"""# 正式v5：QM40恢复阈值0.0075

策略ID：`{FORMAL_STRATEGY_ID}`。用户明确将0.005–0.010平台中央0.0075晋升生产；v4保留直接回滚。

2013固定正式口径为{measured['annualized_return_252']:.2%}年化、Sharpe
{measured['sharpe']:.3f}、MDD {measured['max_drawdown']:.2%}；v4回滚为
{rollback_metrics['annualized_return_252']:.2%}/{rollback_metrics['sharpe']:.3f}/
{rollback_metrics['max_drawdown']:.2%}。2019重启主研究口径为
{primary_metrics['annualized_return_252']:.2%}/{primary_metrics['sharpe']:.3f}/
{primary_metrics['max_drawdown']:.2%}。

正式实现与阈值研究选中路径在2019逐日最大误差为{research_parity:.1e}。Bootstrap跨0、
Reality Check不显著且筛选规则有已披露的事后修正；本次生产变化完全来自用户明确指令。
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
    output = args.output if args.output.is_absolute() else root / args.output
    audit = run_formal(root, args.config, output)
    print(json.dumps(audit["measured"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
