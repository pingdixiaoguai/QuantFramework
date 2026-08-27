"""Generate and verify the formal W40 loss checkpoint and HTML reports."""

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
from strategy.momentum_defender_downside_raqm import (
    run_formal_strategy as run_superseded_weighted,
)
from strategy.momentum_defender_w40_loss import (
    ANCHOR_ASSET,
    DEFENDER_ENTRY_CONFIRMATION_DAYS,
    DEFENDER_ENTRY_PERCENTILE,
    DEFENDER_LOCK_DAYS,
    FORMAL_STRATEGY_ID,
    HISTORY_WINDOW,
    MIN_HISTORY,
    MOMENTUM_LOCK_DAYS,
    MOMENTUM_RECOVERY_CONFIRMATION_DAYS,
    MOMENTUM_RECOVERY_PERCENTILE,
    WINDOW,
    run_formal_strategy,
)


DEFAULT_CONFIG = Path("strategy/configs/momentum_defender_w40_loss.yaml")
DEFAULT_OUTPUT = Path(
    "experiments/20260825_momentum_defender_w40_loss_excluding_extremes_v1_formal"
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
        raise AssertionError("formal W40 strategy ID mismatch")
    gate = config["regime_gate"]
    expected = {
        "anchor_asset": ANCHOR_ASSET,
        "window": WINDOW,
        "percentile_min_history": MIN_HISTORY,
        "defender_entry_percentile": DEFENDER_ENTRY_PERCENTILE,
        "momentum_recovery_percentile": MOMENTUM_RECOVERY_PERCENTILE,
    }
    for field, value in expected.items():
        if gate[field] != value:
            raise AssertionError(f"formal W40 gate mismatch for {field}")
    if gate["percentile_history"] != "rolling_504_strict_lag" or HISTORY_WINDOW != 504:
        raise AssertionError("formal W40 percentile history mismatch")
    if any(gate[field] != "disabled" for field in (
        "path_efficiency", "volatility_adjustment", "volatility_floor", "winsor_clip"
    )):
        raise AssertionError("formal W40 factor contains an unsupported regularizer")
    state = config["state_policy"]
    expected_state = {
        "momentum_lock_days": MOMENTUM_LOCK_DAYS,
        "defender_lock_days": DEFENDER_LOCK_DAYS,
        "defender_entry_confirmation_days": DEFENDER_ENTRY_CONFIRMATION_DAYS,
        "momentum_recovery_confirmation_days": MOMENTUM_RECOVERY_CONFIRMATION_DAYS,
        "emergency_override": False,
    }
    for field, value in expected_state.items():
        if state[field] != value:
            raise AssertionError(f"formal W40 state mismatch for {field}")


def run_formal(root: Path, config_path: Path, output: Path) -> dict[str, object]:
    config_path = config_path if config_path.is_absolute() else root / config_path
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    _validate_config(config)
    checkpoint = config["checkpoint"]
    cutoff = date.fromisoformat(str(checkpoint["end"]))
    formal = run_formal_strategy(root, end=cutoff)
    candidate = formal.daily["return"].astype(float)
    weighted = run_superseded_weighted(root, end=cutoff).daily["return"].astype(float)
    log_momentum = formal.context.integrated.result.inputs.momentum[HELD_RETURN].astype(float)
    legacy_config = _load_momentum_config(root / LEGACY_MOMENTUM_CONFIG, cutoff)
    legacy = run_backtest(legacy_config).daily_returns.reindex(candidate.index).astype(float)
    if legacy.isna().any():
        raise AssertionError("legacy Momentum baseline does not cover W40 calendar")
    measured = performance(candidate)
    failures = []
    for field in (
        "observations", "total_return", "cagr_calendar", "annualized_return_252",
        "annualized_volatility", "sharpe", "max_drawdown",
    ):
        expected = checkpoint[field]
        actual = measured[field]
        matches = int(actual) == expected if isinstance(expected, int) else abs(float(actual) - float(expected)) <= 1e-12
        if not matches:
            failures.append(f"{field}: {actual!r} != {expected!r}")
    daily_hash = _return_hash(candidate)
    if daily_hash != checkpoint["daily_return_sha256_float64_le"]:
        failures.append("daily return hash mismatch")
    for field in ("defender_entries", "defender_days", "sleeve_switches", "candidate_switches"):
        if int(formal.audit[field]) != int(checkpoint[field]):
            failures.append(f"{field}: {formal.audit[field]} != {checkpoint[field]}")
    selected = pd.read_parquet(
        root / "experiments/20260825_momentum_defender_w40_loss_occam_search/selected_excluding_extremes_daily.parquet"
    )["return"].astype(float)
    research_parity = float((candidate - selected).abs().max())
    if research_parity > 1e-14:
        failures.append(f"research parity mismatch: {research_parity:.3e}")
    if failures:
        raise AssertionError("formal W40 checkpoint failed: " + "; ".join(failures))

    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    daily = formal.state.copy()
    daily["w40_downside_log_loss_at_open"] = formal.raw_loss_at_open
    daily["w40_loss_percentile_at_open"] = formal.score_at_open
    daily = daily.join(formal.daily, rsuffix="_execution")
    daily.to_csv(stage / "daily_backtest.csv")
    daily.to_parquet(stage / "daily_backtest.parquet")
    strategies = {
        "formal_w40_loss": candidate,
        "superseded_weighted_draqm": weighted,
        "log_quality_momentum": log_momentum,
        "original_momentum_legacy_simple_price": legacy,
    }
    pd.DataFrame(
        [{"strategy": name, **performance(values)} for name, values in strategies.items()]
    ).to_csv(stage / "strategy_metrics.csv", index=False)
    _annual(strategies).to_csv(stage / "calendar_year_returns.csv", index=False)
    events, leave_events, deleted, event_summary = _event_stress(
        candidate,
        weighted,
        formal.daily["candidate"].astype(str),
        run_superseded_weighted(root, end=cutoff).daily["candidate"].astype(str),
        [1, 2, 3],
    )
    events.to_csv(stage / "events_vs_superseded_weighted.csv", index=False)
    leave_events.to_csv(stage / "leave_one_event_vs_superseded_weighted.csv", index=False)
    deleted.to_csv(stage / "top_positive_event_deletion.csv", index=False)
    (stage / "strategy_config.yaml").write_text(
        config_path.read_text(encoding="utf-8"), encoding="utf-8"
    )
    generate_standard_report(
        candidate, legacy, "Original Momentum (Simple MOM × Price ER)",
        stage / "formal_backtest.html", config,
    )
    shutil.copyfile(stage / "formal_backtest.html", stage / "formal_vs_original_momentum.html")
    generate_standard_report(
        candidate, weighted, "Superseded Weighted DRAQM Formal",
        stage / "formal_vs_superseded_weighted.html", config,
    )
    audit = {
        "status": "passed",
        "strategy_id": FORMAL_STRATEGY_ID,
        "checkpoint_tolerance": 1e-12,
        "daily_return_sha256_float64_le": daily_hash,
        "research_selected_daily_max_abs_error": research_parity,
        "candidate_interface_baseline_parity_max_abs_error": formal.context.baseline_parity_max_abs_error,
        "mechanical_audit": formal.audit,
        "events_vs_superseded_weighted": event_summary,
        "measured": measured,
        "expected": checkpoint,
    }
    (stage / "checkpoint_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    source_paths = [
        config_path,
        root / "strategy/momentum_defender_w40_loss.py",
        root / "run_daily_momentum_defender.py",
        root / "scripts/run_daily_job.py",
        root / "research/run_formal_w40_loss.py",
        root / "research/momentum_defender_w40_loss_gate.py",
        root / "research/momentum_defender_integrated.py",
        root / "factors/quality_momentum.py",
        root / LEGACY_MOMENTUM_CONFIG,
        root / "strategy/configs/quality_momentum_top1.yaml",
        root / "strategy/governance/momentum_defender_w40_loss_excluding_extremes_v1.json",
        root / "research/configs/momentum_defender_w40_loss_excluding_extremes_selected.yaml",
        root / "research/configs/momentum_defender_badcase_context.yaml",
        root / "research/generate_momentum_defender_badcases.py",
        root / "docs/research/momentum_defender_badcases.md",
        root / "docs/research/2026-08-25_w40_loss_formal_promotion.md",
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
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report = f"""# 正式策略：510300单一40日下跌幅度分位

正式策略ID为`{FORMAL_STRATEGY_ID}`。40日对数下跌幅度的严格滞后504日分位以55%/40%
进出线、1/1日确认和30/30日不可绕过状态锁控制Momentum/Defender。因子不使用加权、
路径效率、波动率调整、地板或clip。

|策略|年化收益|Sharpe|最大回撤|
|---|---:|---:|---:|
|正式W40|{measured['annualized_return_252']:.2%}|{measured['sharpe']:.3f}|{measured['max_drawdown']:.2%}|
|已取代加权DRAQM|{performance(weighted)['annualized_return_252']:.2%}|{performance(weighted)['sharpe']:.3f}|{performance(weighted)['max_drawdown']:.2%}|

正式代码与研究选中路径逐日最大误差{research_parity:.1e}，收益哈希与配置一致。该候选的
全局普通区间Reality Check p=0.9932，晋升来自用户明确偏好而非统计显著性。参数自
2026-08-25冻结，后续只允许使用未观察数据评价。
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
