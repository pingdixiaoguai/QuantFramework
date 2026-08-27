"""Generate and verify the promoted W40/full-equity formal checkpoint."""

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
from research.momentum_defender_occam import HELD_RETURN, _load_momentum_config, performance
from research.run_momentum_defender_log_qm_robust import _event_stress
from research.standard_report import generate_standard_report
from strategy.momentum_defender_w40_full_equity import (
    FORMAL_STRATEGY_ID,
    run_formal_strategy,
)
from strategy.momentum_defender_w40_loss import (
    DEFENDER_ENTRY_CONFIRMATION_DAYS,
    DEFENDER_ENTRY_PERCENTILE,
    DEFENDER_LOCK_DAYS,
    MOMENTUM_LOCK_DAYS,
    MOMENTUM_RECOVERY_CONFIRMATION_DAYS,
    MOMENTUM_RECOVERY_PERCENTILE,
    run_formal_strategy as run_rollback_w40,
)


DEFAULT_CONFIG = Path("strategy/configs/momentum_defender_w40_full_equity.yaml")
DEFAULT_OUTPUT = Path(
    "experiments/20260826_momentum_defender_w40_reversal_full_equity_v2_formal"
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


def _validate_config(config: dict) -> None:
    if config["strategy_name"] != FORMAL_STRATEGY_ID:
        raise AssertionError("formal W40/full-equity strategy ID mismatch")
    if config["strategy_mode"] != "w40_reversal_full_equity":
        raise AssertionError("formal strategy mode mismatch")
    if str(config.get("backtest_start")) != "2013-01-01":
        raise AssertionError("formal backtest start must remain 2013-01-01")
    gate = config["regime_gate"]
    expected_gate = {
        "window": 40,
        "defender_entry_percentile": DEFENDER_ENTRY_PERCENTILE,
        "momentum_recovery_percentile": MOMENTUM_RECOVERY_PERCENTILE,
    }
    for field, value in expected_gate.items():
        if gate[field] != value:
            raise AssertionError(f"formal gate mismatch for {field}")
    state = config["state_policy"]
    expected_state = {
        "momentum_lock_days": MOMENTUM_LOCK_DAYS,
        "defender_lock_days": DEFENDER_LOCK_DAYS,
        "defender_entry_confirmation_days": DEFENDER_ENTRY_CONFIRMATION_DAYS,
        "momentum_recovery_confirmation_days": MOMENTUM_RECOVERY_CONFIRMATION_DAYS,
    }
    for field, value in expected_state.items():
        if state[field] != value:
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


def run_formal(root: Path, config_path: Path, output: Path) -> dict[str, object]:
    config_path = config_path if config_path.is_absolute() else root / config_path
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    _validate_config(config)
    checkpoint = config["checkpoint"]
    cutoff = date.fromisoformat(str(checkpoint["end"]))
    formal = run_formal_strategy(root, end=cutoff)
    candidate = formal.daily["return"].astype(float)
    rollback_run = run_rollback_w40(root, end=cutoff)
    rollback = rollback_run.daily["return"].astype(float)
    log_momentum = formal.context.integrated.result.inputs.momentum[
        HELD_RETURN
    ].astype(float)
    legacy_config = _load_momentum_config(root / LEGACY_MOMENTUM_CONFIG, cutoff)
    legacy = run_backtest(legacy_config).daily_returns.reindex(candidate.index).astype(float)
    if legacy.isna().any():
        raise AssertionError("legacy Momentum baseline does not cover formal calendar")

    measured = performance(candidate)
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
    daily_hash = _return_hash(candidate)
    if daily_hash != checkpoint["daily_return_sha256_float64_le"]:
        failures.append("daily return hash mismatch")
    for field in (
        "defender_entries",
        "defender_days",
        "sleeve_switches",
        "candidate_switches",
    ):
        if int(formal.audit[field]) != int(checkpoint[field]):
            failures.append(f"{field}: {formal.audit[field]} != {checkpoint[field]}")
    selection_switches = int(
        formal.audit["defender_audit"]["selection_switches"]
    )
    if selection_switches != int(checkpoint["defender_selection_switches"]):
        failures.append(
            "defender_selection_switches: "
            f"{selection_switches} != {checkpoint['defender_selection_switches']}"
        )
    if failures:
        raise AssertionError("formal checkpoint failed: " + "; ".join(failures))

    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    daily = formal.state.copy()
    daily["w40_downside_log_loss_at_open"] = formal.raw_loss_at_open
    daily["w40_loss_percentile_at_open"] = formal.score_at_open
    daily = daily.join(formal.daily, rsuffix="_execution")
    daily = daily.join(build_formal_target_schedule(formal).add_prefix("target_"))
    daily.to_csv(stage / "daily_backtest.csv")
    daily.to_parquet(stage / "daily_backtest.parquet")
    strategies = {
        "formal_w40_full_equity": candidate,
        "rollback_w40_legacy_defender": rollback,
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

    risk_on = formal.state["risk_on"].astype(bool)
    candidate_event_target = pd.Series(
        ["MOMENTUM" if value else "FULL_EQUITY_DEFENDER" for value in risk_on],
        index=candidate.index,
    )
    rollback_event_target = pd.Series(
        ["MOMENTUM" if value else "LEGACY_DEFENDER" for value in risk_on],
        index=candidate.index,
    )
    rollback_common = candidate.index.intersection(rollback.index)
    events, leave_events, deleted, event_summary = _event_stress(
        candidate.loc[rollback_common],
        rollback.loc[rollback_common],
        candidate_event_target.loc[rollback_common],
        rollback_event_target.loc[rollback_common],
        [1, 2, 3],
    )
    events.to_csv(stage / "events_vs_rollback_w40.csv", index=False)
    leave_events.to_csv(stage / "leave_one_event_vs_rollback_w40.csv", index=False)
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
    shutil.copyfile(
        stage / "formal_backtest.html", stage / "formal_vs_original_momentum.html"
    )
    generate_standard_report(
        candidate.loc[rollback_common],
        rollback,
        "Rollback Formal W40 (common history)",
        stage / "formal_vs_rollback_w40.html",
        config,
    )
    audit = {
        "status": "passed",
        "strategy_id": FORMAL_STRATEGY_ID,
        "checkpoint_tolerance": 1e-12,
        "daily_return_sha256_float64_le": daily_hash,
        "research_selected_daily_max_abs_error": None,
        "mechanical_audit": formal.audit,
        "events_vs_rollback_w40": event_summary,
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
        root / "strategy/momentum_defender_w40_full_equity.py",
        root / "run_daily_momentum_defender.py",
        root / "scripts/run_daily_job.py",
        root / "research/run_formal_w40_full_equity.py",
        root / "research/standard_report.py",
        root / "backtest/report.py",
        root / "research/momentum_defender_w40_loss_gate.py",
        root / "research/formal_strategy_holdings.py",
        root / "factors/quality_momentum.py",
        root / LEGACY_MOMENTUM_CONFIG,
        root / "strategy/configs/quality_momentum_top1.yaml",
        root / "strategy/configs/momentum_defender_w40_loss.yaml",
        root / "strategy/governance/momentum_defender_w40_reversal_full_equity_v2.json",
        root / "strategy/governance/momentum_defender_w40_loss_excluding_extremes_v1.json",
        root / "research/configs/momentum_defender_dividend_universe_2007_validation.yaml",
        root / "research/momentum_defender_dividend_universe.py",
        root / "research/configs/momentum_defender_badcase_context.yaml",
        root / "research/configs/strategy_drawdown_badcase_context.yaml",
        root / "research/generate_momentum_defender_badcases.py",
        root / "research/generate_strategy_drawdown_badcases.py",
        root / "docs/research/momentum_defender_badcases.md",
        root / "docs/research/momentum_defender_drawdown_badcases.md",
        root / "docs/research/2026-08-26_defender_2007_2026_validation.md",
        root / "docs/research/2026-08-26_defender_dividend_universe_formal_promotion.md",
        root / "research/DEVELOPMENT_VALIDATION.md",
        root / "README.md",
        root / "research/README.md",
        root / "strategy_changelog.md",
        root / "strategy/AGENTS.md",
        root / "defender/AGENTS.md",
        root / "ops/tencent-cloud/README.md",
        root / "ops/tencent-cloud/quant-daily.service",
        root / "defender/tests/test_w40_reversal_full_equity.py",
        root / "strategy/tests/test_momentum_defender_w40_full_equity.py",
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
    report = f"""# 正式策略：W40门控 + 月度40日最弱红利ETF + 100%红利

策略ID：`{FORMAL_STRATEGY_ID}`。顶层W40状态与回滚版本逐日完全一致；Defender在固定六只
红利ETF中每月选择40日对数收益最低者并100%持有，不再使用国债、网格、波动率上限或champion
满仓覆盖。正式检查点为{measured['annualized_return_252']:.2%}年化、
Sharpe {measured['sharpe']:.3f}、MDD {measured['max_drawdown']:.2%}。

本次晋升来自用户明确指令；回溯研究未证明统计显著优于回滚W40，详见正式配置、治理记录、
两本badcase台账和`checkpoint_audit.json`。
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
