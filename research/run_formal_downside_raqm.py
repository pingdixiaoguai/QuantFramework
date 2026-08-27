"""Generate and verify the formal frozen 510300 downside-RAQM checkpoint."""

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
from research.momentum_defender_occam import HELD_RETURN, _load_momentum_config, performance
from research.run_momentum_defender_log_qm_robust import _event_stress
from research.standard_report import generate_standard_report
from strategy.momentum_defender_absolute_stability import (
    run_formal_strategy as run_superseded_v4,
)
from strategy.momentum_defender_downside_raqm import (
    ANCHOR_ASSET,
    DEFENDER_ENTRY_CONFIRMATION_DAYS,
    DEFENDER_ENTRY_PERCENTILE,
    DEFENDER_LOCK_DAYS,
    FORMAL_STRATEGY_ID,
    HORIZONS,
    MOMENTUM_LOCK_DAYS,
    MOMENTUM_RECOVERY_CONFIRMATION_DAYS,
    MOMENTUM_RECOVERY_PERCENTILE,
    PERCENTILE_HISTORY_WINDOW,
    PERCENTILE_MIN_HISTORY,
    VOLATILITY_FLOOR_ANNUAL,
    WEIGHTS,
    WINSOR_LIMIT,
    run_formal_strategy,
)


DEFAULT_CONFIG = Path("strategy/configs/momentum_defender_downside_raqm.yaml")
DEFAULT_OUTPUT = Path(
    "experiments/20260825_momentum_defender_downside_raqm_weighted_v1_formal"
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
        raise AssertionError("formal strategy ID mismatch")
    gate = config["regime_gate"]
    expected_gate = {
        "anchor_asset": ANCHOR_ASSET,
        "horizons": list(HORIZONS),
        "weights": list(WEIGHTS),
        "volatility_floor_annual": VOLATILITY_FLOOR_ANNUAL,
        "winsor_limit": WINSOR_LIMIT,
        "percentile_min_history": PERCENTILE_MIN_HISTORY,
        "defender_entry_percentile": DEFENDER_ENTRY_PERCENTILE,
        "momentum_recovery_percentile": MOMENTUM_RECOVERY_PERCENTILE,
    }
    for field, expected in expected_gate.items():
        if gate[field] != expected:
            raise AssertionError(f"formal gate mismatch for {field}")
    if gate["percentile_history"] != "rolling_504_strict_lag":
        raise AssertionError("formal percentile history mismatch")
    if PERCENTILE_HISTORY_WINDOW != 504:
        raise AssertionError("formal percentile history constant mismatch")
    state = config["state_policy"]
    expected_state = {
        "momentum_lock_days": MOMENTUM_LOCK_DAYS,
        "defender_lock_days": DEFENDER_LOCK_DAYS,
        "defender_entry_confirmation_days": DEFENDER_ENTRY_CONFIRMATION_DAYS,
        "momentum_recovery_confirmation_days": (
            MOMENTUM_RECOVERY_CONFIRMATION_DAYS
        ),
        "emergency_override": False,
    }
    for field, expected in expected_state.items():
        if state[field] != expected:
            raise AssertionError(f"formal state mismatch for {field}")
    if any(value != "disabled" for value in config["overlays"].values()):
        raise AssertionError("formal overlays must all be disabled")


def run_formal(root: Path, config_path: Path, output: Path) -> dict[str, object]:
    config_path = config_path if config_path.is_absolute() else root / config_path
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    _validate_config(config)
    checkpoint = config["checkpoint"]
    cutoff = date.fromisoformat(str(checkpoint["end"]))
    formal = run_formal_strategy(root, end=cutoff)
    candidate = formal.daily["return"].astype(float)
    log_momentum = formal.context.integrated.result.inputs.momentum[
        HELD_RETURN
    ].astype(float)
    superseded = run_superseded_v4(root, end=cutoff).daily["return"].astype(float)
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
    selected_daily = pd.read_parquet(
        root
        / "experiments/20260824_momentum_defender_downside_raqm_final_selection/selected_daily.parquet"
    )["return"].astype(float)
    research_parity = float((candidate - selected_daily).abs().max())
    if research_parity > 1e-14:
        failures.append(f"research parity mismatch: {research_parity:.3e}")
    if failures:
        raise AssertionError("formal checkpoint failed: " + "; ".join(failures))

    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    daily = formal.state.copy()
    daily["downside_raqm_30_at_open"] = formal.features.raw_at_open[30]
    daily["downside_raqm_40_at_open"] = formal.features.raw_at_open[40]
    daily = daily.join(formal.daily, rsuffix="_execution")
    daily.to_csv(stage / "daily_backtest.csv")
    daily.to_parquet(stage / "daily_backtest.parquet")
    strategies = {
        "formal_downside_raqm": candidate,
        "superseded_v4": superseded,
        "log_quality_momentum": log_momentum,
        "original_momentum_legacy_simple_price": legacy,
    }
    pd.DataFrame(
        [{"strategy": name, **performance(values)} for name, values in strategies.items()]
    ).to_csv(stage / "strategy_metrics.csv", index=False)
    _annual(strategies).to_csv(stage / "calendar_year_returns.csv", index=False)

    events, leave_events, top_deleted, event_summary = _event_stress(
        candidate,
        log_momentum,
        formal.daily["candidate"].astype(str),
        formal.context.momentum_target.astype(str),
        [1, 2, 3],
    )
    events.to_csv(stage / "defender_events_vs_log_momentum.csv", index=False)
    leave_events.to_csv(stage / "leave_one_defender_event.csv", index=False)
    top_deleted.to_csv(stage / "top_positive_event_deletion.csv", index=False)
    (stage / "strategy_config.yaml").write_text(
        config_path.read_text(encoding="utf-8"), encoding="utf-8"
    )
    generate_standard_report(
        candidate,
        superseded,
        "Superseded Confirmation Bridge Raw Gold v4",
        stage / "formal_vs_superseded_v4.html",
        config,
    )
    generate_standard_report(
        candidate,
        legacy,
        "Original Momentum (Simple MOM × Price ER)",
        stage / "formal_backtest.html",
        config,
    )
    shutil.copyfile(
        stage / "formal_backtest.html",
        stage / "formal_vs_original_momentum.html",
    )
    audit = {
        "status": "passed",
        "strategy_id": FORMAL_STRATEGY_ID,
        "checkpoint_tolerance": 1e-12,
        "daily_return_sha256_float64_le": daily_hash,
        "research_selected_daily_max_abs_error": research_parity,
        "candidate_interface_baseline_parity_max_abs_error": (
            formal.context.baseline_parity_max_abs_error
        ),
        "mechanical_audit": formal.audit,
        "event_stress_vs_log_momentum": event_summary,
        "measured": measured,
        "expected": checkpoint,
    }
    (stage / "checkpoint_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    source_paths = [
        config_path,
        root / "strategy/momentum_defender_downside_raqm.py",
        root / "run_daily_momentum_defender.py",
        root / "scripts/run_daily_job.py",
        root / "research/run_formal_downside_raqm.py",
        root / "research/momentum_defender_downside_raqm.py",
        root / "research/momentum_defender_integrated.py",
        root / "research/momentum_defender_gold_override.py",
        root / "factors/quality_momentum.py",
        root / LEGACY_MOMENTUM_CONFIG,
        root / "strategy/configs/quality_momentum_top1.yaml",
        root / "strategy/governance/momentum_defender_downside_raqm_weighted_v1.json",
        root / "research/configs/momentum_defender_badcase_context.yaml",
        root / "research/generate_momentum_defender_badcases.py",
        root / "docs/research/momentum_defender_badcases.md",
        root / "docs/research/2026-08-25_downside_raqm_formal_promotion.md",
        root / "research/DEVELOPMENT_VALIDATION.md",
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
    report = f"""# 正式策略：冻结通用510300下行DRAQM门控

正式策略ID为`{FORMAL_STRATEGY_ID}`。Momentum固定使用20日对数收益乘对数路径ER；
510300的30/40日下行DRAQM严格滞后504日分位按25%/75%合成。组合分位连续3日不低于
0.55进入Defender，分位不高于0.20恢复Momentum，两个袖套均有30日不可绕过状态锁。
Gold、快速反转、持仓标的紧急退出和全部破锁逻辑关闭。

|策略|年化收益|Sharpe|最大回撤|
|---|---:|---:|---:|
|正式通用门控|{float(measured['annualized_return_252']):.2%}|{float(measured['sharpe']):.3f}|{float(measured['max_drawdown']):.2%}|
|已取代v4|{float(performance(superseded)['annualized_return_252']):.2%}|{float(performance(superseded)['sharpe']):.3f}|{float(performance(superseded)['max_drawdown']):.2%}|
|双对数Momentum|{float(performance(log_momentum)['annualized_return_252']):.2%}|{float(performance(log_momentum)['sharpe']):.3f}|{float(performance(log_momentum)['max_drawdown']):.2%}|

正式路径共有{formal.audit['defender_entries']}次Defender进入、
{formal.audit['defender_days']}个Defender交易日、{formal.audit['candidate_switches']}次实际候选切换。
正式实现与研究选中逐日收益最大误差为{research_parity:.1e}，收益哈希与配置完全一致。

证据边界：该参数来自已观察历史，Reality Check p=0.1742，不能表述为统计显著的未来优势。
本次晋升来自用户明确选择；2026-08-25之后进入独立前瞻账本，禁止继续用相同历史调整
30/40权重、0.55/0.20阈值、3/1日确认或30/30日锁。
"""
    (stage / "research_report.md").write_text(report, encoding="utf-8")

    output.mkdir(parents=True, exist_ok=True)
    obsolete_log_report = output / "formal_vs_log_quality_momentum.html"
    if obsolete_log_report.exists():
        obsolete_log_report.unlink()
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
