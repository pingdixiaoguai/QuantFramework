"""Backtest C2 with ChiNext emergency cap disabled and no 30-day lock."""

from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import asdict
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from research.momentum_defender_occam import (
    HELD_RETURN,
    MOMENTUM_ASSETS,
    apply_state_schedule,
    build_inputs,
    simulate_switch,
    slow_regime_at_open,
)
from research.run_momentum_defender_occam import _generate_standard_report
from research.run_momentum_held_asset_adaptive_cap import held_asset_cap_alert
from research.run_momentum_held_asset_c2_no_chinext_cap import (
    CHINEXT_ASSET,
    SELECTED_C2,
    SLOW_PARAMS,
    _git,
    _metric,
    _metric_records,
    _sha256,
    suppress_alert_for_asset,
)
from research.run_momentum_volatility_signal_abcd import (
    DEFAULT_DEFENDER_DIR,
    DEFAULT_END,
    _load_ohlc,
    asof_previous_close,
    expanding_volatility_cap,
    momentum_asset_at_previous_close,
    rogers_satchell_volatility,
)


DEFAULT_OUTPUT = Path(
    "experiments/20260821_momentum_held_asset_c2_no_chinext_cap_no_lock"
)
NO_LOCK_MIN_HOLD_DAYS = 1


def _simulate(
    slow: pd.Series,
    alert: pd.Series,
    calendar: pd.DatetimeIndex,
    momentum: pd.DataFrame,
    defender: pd.DataFrame,
    min_hold_days: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    state = apply_state_schedule(
        slow,
        alert,
        calendar,
        min_hold_days,
        emergency_override=True,
    )
    return state, simulate_switch(momentum, defender, state["risk_on"])


def _risk_off_episode_lengths(state: pd.DataFrame) -> list[int]:
    risk_off = ~state["risk_on"].astype(bool)
    if not risk_off.any():
        return []
    groups = risk_off.ne(risk_off.shift()).cumsum()
    return [int(len(group)) for _, group in risk_off.loc[risk_off].groupby(groups)]


def _state_diagnostics(
    states: dict[str, pd.DataFrame],
    simulated: dict[str, pd.DataFrame],
    alerts: dict[str, pd.Series],
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for strategy, state in states.items():
        emergency_entries = (
            state["state_changed"].astype(bool)
            & state["state_reason"].eq("emergency_exit")
        )
        lengths = _risk_off_episode_lengths(state)
        records.append(
            {
                "strategy": strategy,
                "alert_days": int(alerts[strategy].sum()),
                "emergency_entries": int(emergency_entries.sum()),
                "defender_days": int((~state["risk_on"]).sum()),
                "defender_episodes": len(lengths),
                "median_defender_episode_days": (
                    float(np.median(lengths)) if lengths else 0.0
                ),
                "maximum_defender_episode_days": max(lengths, default=0),
                "sleeve_switches": int(simulated[strategy]["sleeve_switch"].sum()),
            }
        )
    return pd.DataFrame(records)


def _calendar_year_records(strategies: dict[str, pd.Series]) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    calendar = next(iter(strategies.values())).index
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


def _full_table(metrics: pd.DataFrame) -> str:
    labels = {
        "c2_no_chinext_cap_no_lock": "取消创业板cap、无30日锁",
        "c2_no_chinext_cap_30d_lock": "取消创业板cap、保留30日锁",
        "selected_c2": "原C2",
        "no_cap_fusion": "无cap融合（保留30日锁）",
        "original_momentum": "原动量策略",
        "original_base": "原4ETF等权base",
    }
    lines = [
        "|方案|年化收益|年化波动|Sharpe|最大回撤|",
        "|---|---:|---:|---:|---:|",
    ]
    for strategy, label in labels.items():
        row = _metric(metrics, "full", strategy)
        lines.append(
            f"|{label}|{row.annualized_return_252:.2%}|"
            f"{row.annualized_volatility:.2%}|{row.sharpe:.3f}|"
            f"{row.max_drawdown:.2%}|"
        )
    return "\n".join(lines)


def _year_table(yearly: pd.DataFrame) -> str:
    pivot = yearly.pivot(index="year", columns="strategy", values="total_return")
    lines = [
        "|年份|取消创业板cap、无锁|取消创业板cap、30日锁|原C2|无cap融合|",
        "|---:|---:|---:|---:|---:|",
    ]
    for year, row in pivot.iterrows():
        lines.append(
            f"|{year}|{row.c2_no_chinext_cap_no_lock:+.2%}|"
            f"{row.c2_no_chinext_cap_30d_lock:+.2%}|"
            f"{row.selected_c2:+.2%}|{row.no_cap_fusion:+.2%}|"
        )
    return "\n".join(lines)


def _diagnostic_table(diagnostics: pd.DataFrame) -> str:
    labels = {
        "c2_no_chinext_cap_no_lock": "取消创业板cap、无锁",
        "c2_no_chinext_cap_30d_lock": "取消创业板cap、30日锁",
        "selected_c2": "原C2",
        "no_cap_fusion": "无cap融合",
    }
    lines = [
        "|方案|报警日|紧急入场|Defender持有日|Defender区间|中位区间长度|最长区间|袖套切换|",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for strategy, label in labels.items():
        row = diagnostics.loc[diagnostics["strategy"].eq(strategy)].iloc[0]
        lines.append(
            f"|{label}|{int(row.alert_days)}|{int(row.emergency_entries)}|"
            f"{int(row.defender_days)}|{int(row.defender_episodes)}|"
            f"{row.median_defender_episode_days:.1f}|"
            f"{int(row.maximum_defender_episode_days)}|"
            f"{int(row.sleeve_switches)}|"
        )
    return "\n".join(lines)


def run_experiment(
    root: Path,
    defender_dir: Path,
    final_output: Path,
    end: date,
) -> None:
    final_output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(prefix=f".{final_output.name}.staging-", dir=final_output.parent)
    )

    inputs = build_inputs(
        root,
        defender_dir / "relative_defender_rotation_switch_returns.csv",
        end,
    )
    calendar = inputs.calendar
    exact_momentum = inputs.momentum[HELD_RETURN].astype(float)
    original_base = inputs.momentum_result.benchmark_returns.reindex(calendar).astype(float)
    if original_base.isna().any():
        raise ValueError("original 4ETF base has missing report dates")
    slow = slow_regime_at_open(
        inputs.risk_close,
        calendar,
        SLOW_PARAMS.lookback,
        SLOW_PARAMS.risk_on_threshold,
    )
    previous_asset = momentum_asset_at_previous_close(inputs.momentum_result, calendar)

    caps: dict[str, pd.Series] = {}
    for asset in MOMENTUM_ASSETS:
        prices = _load_ohlc(asset, end)
        volatility = rogers_satchell_volatility(
            prices, int(SELECTED_C2.volatility_window)
        )
        close_cap = expanding_volatility_cap(
            volatility, SELECTED_C2.asset_quantiles()[asset]
        )["cap"]
        caps[asset] = asof_previous_close(close_cap, calendar).fillna(1.0)

    selected_c2_alert = held_asset_cap_alert(
        caps,
        previous_asset,
        {asset: float(SELECTED_C2.cap_trigger_maximum) for asset in MOMENTUM_ASSETS},
    )
    no_chinext_alert = suppress_alert_for_asset(
        selected_c2_alert,
        previous_asset,
        CHINEXT_ASSET,
    )
    no_cap_alert = pd.Series(False, index=calendar, name="no_cap_alert")

    strategy_specs = {
        "c2_no_chinext_cap_no_lock": (no_chinext_alert, NO_LOCK_MIN_HOLD_DAYS),
        "c2_no_chinext_cap_30d_lock": (no_chinext_alert, SLOW_PARAMS.min_hold_days),
        "selected_c2": (selected_c2_alert, SLOW_PARAMS.min_hold_days),
        "no_cap_fusion": (no_cap_alert, SLOW_PARAMS.min_hold_days),
    }
    states: dict[str, pd.DataFrame] = {}
    simulated: dict[str, pd.DataFrame] = {}
    alerts: dict[str, pd.Series] = {}
    for strategy, (alert, min_hold_days) in strategy_specs.items():
        states[strategy], simulated[strategy] = _simulate(
            slow,
            alert,
            calendar,
            inputs.momentum,
            inputs.defender,
            min_hold_days,
        )
        alerts[strategy] = alert

    strategies = {
        strategy: result["return"] for strategy, result in simulated.items()
    }
    strategies["original_momentum"] = exact_momentum
    strategies["original_base"] = original_base

    metrics = _metric_records(strategies, end)
    metrics.to_csv(stage / "strategy_period_metrics.csv", index=False)
    yearly = _calendar_year_records(strategies)
    yearly.to_csv(stage / "calendar_year_returns.csv", index=False)
    diagnostics = _state_diagnostics(states, simulated, alerts)
    diagnostics.to_csv(stage / "state_diagnostics.csv", index=False)

    daily = pd.DataFrame(index=calendar)
    daily["momentum_asset_at_previous_close"] = previous_asset
    daily["selected_c2_alert"] = selected_c2_alert
    daily["no_chinext_cap_alert"] = no_chinext_alert
    daily["alert_suppressed"] = selected_c2_alert & ~no_chinext_alert
    daily["slow_signal_asof_previous_close"] = slow
    for strategy, state in states.items():
        daily[f"{strategy}_risk_on"] = state["risk_on"]
        daily[f"{strategy}_state_reason"] = state["state_reason"]
        daily[f"{strategy}_return"] = strategies[strategy]
    daily["original_momentum_return"] = exact_momentum
    daily["original_base_return"] = original_base
    daily.index.name = "date"
    daily.to_csv(stage / "daily_comparison.csv")

    config = {
        "strategy_name": "C2_no_chinext_cap_no_30d_lock",
        **asdict(SELECTED_C2),
        "excluded_emergency_cap_asset": CHINEXT_ASSET,
        "min_hold_days": NO_LOCK_MIN_HOLD_DAYS,
        "research_cutoff": end.isoformat(),
    }
    report_benchmarks = {
        "C2_no_chinext_cap_no_lock_vs_original_base.html": (
            original_base,
            "Original 4ETF Equal-weight Base",
        ),
        "C2_no_chinext_cap_no_lock_vs_original_momentum.html": (
            exact_momentum,
            "Original Momentum Strategy",
        ),
        "C2_no_chinext_cap_no_lock_vs_no_cap_fusion.html": (
            strategies["no_cap_fusion"],
            "No-cap Slow-gate Fusion (30-day lock)",
        ),
        "C2_no_chinext_cap_no_lock_vs_30d_lock.html": (
            strategies["c2_no_chinext_cap_30d_lock"],
            "C2 No-ChiNext Cap (30-day lock)",
        ),
        "C2_no_chinext_cap_no_lock_vs_selected_C2.html": (
            strategies["selected_c2"],
            "Selected C2",
        ),
    }
    for filename, (benchmark, benchmark_name) in report_benchmarks.items():
        _generate_standard_report(
            strategies["c2_no_chinext_cap_no_lock"],
            benchmark,
            benchmark_name,
            stage / filename,
            config,
        )

    no_lock = _metric(metrics, "full", "c2_no_chinext_cap_no_lock")
    locked = _metric(metrics, "full", "c2_no_chinext_cap_30d_lock")
    original_c2 = _metric(metrics, "full", "selected_c2")
    no_cap = _metric(metrics, "full", "no_cap_fusion")
    no_lock_diag = diagnostics.loc[
        diagnostics["strategy"].eq("c2_no_chinext_cap_no_lock")
    ].iloc[0]
    locked_diag = diagnostics.loc[
        diagnostics["strategy"].eq("c2_no_chinext_cap_30d_lock")
    ].iloc[0]
    report = f"""# 取消创业板cap并取消30日状态锁：回测结论

## 结论

“无30日锁”按最早可在下一交易日再次切换实现，即`min_hold_days=1`。全样本调整后年化为{no_lock.annualized_return_252:.2%}、Sharpe为{no_lock.sharpe:.3f}、MDD为{no_lock.max_drawdown:.2%}。

相对“取消创业板cap但保留30日锁”，年化变化{no_lock.annualized_return_252 - locked.annualized_return_252:+.2%}、Sharpe变化{no_lock.sharpe - locked.sharpe:+.3f}、MDD变化{no_lock.max_drawdown - locked.max_drawdown:+.2%}。袖套切换从{int(locked_diag.sleeve_switches)}次增加到{int(no_lock_diag.sleeve_switches)}次，紧急入场从{int(locked_diag.emergency_entries)}次增加到{int(no_lock_diag.emergency_entries)}次。

## 全样本指标

{_full_table(metrics)}

## 逐年收益

{_year_table(yearly)}

## 状态与换手

{_diagnostic_table(diagnostics)}

## 对照判断

- 相对原C2：年化变化{no_lock.annualized_return_252 - original_c2.annualized_return_252:+.2%}、Sharpe变化{no_lock.sharpe - original_c2.sharpe:+.3f}、MDD变化{no_lock.max_drawdown - original_c2.max_drawdown:+.2%}。
- 相对无cap融合：年化变化{no_lock.annualized_return_252 - no_cap.annualized_return_252:+.2%}、Sharpe变化{no_lock.sharpe - no_cap.sharpe:+.3f}、MDD变化{no_lock.max_drawdown - no_cap.max_drawdown:+.2%}。
- 取消锁不会改变信号的因果口径，但会让短暂报警和慢门控反复重置持仓；是否合理必须同时看收益、MDD和切换次数，不能只看年化。

## 固定不变的口径

- 创业板ETF仍参与Momentum和40日慢门控，只不能触发紧急cap。
- 其他三只ETF仍使用C2的10日Rogers–Satchell波动率、cap≤0.8，以及沪深300q70、纳指q95、黄金q90。
- 信号使用上一收盘及更早数据，下一开盘执行；Momentum与Defender使用开盘切换分段收益和既有费用。
"""
    (stage / "research_report.md").write_text(report, encoding="utf-8")

    input_files = [
        defender_dir / "relative_defender_rotation_switch_returns.csv",
        root / "strategy/configs/quality_momentum_top1.yaml",
        *[root / "data/db" / f"{asset}.parquet" for asset in MOMENTUM_ASSETS],
    ]
    code_files = [
        root / "research/run_momentum_held_asset_c2_no_chinext_cap_no_lock.py",
        root / "research/run_momentum_held_asset_c2_no_chinext_cap.py",
        root / "research/run_momentum_held_asset_adaptive_cap.py",
        root / "research/run_momentum_volatility_signal_abcd.py",
        root / "research/momentum_defender_occam.py",
    ]
    manifest = {
        "experiment": "momentum_held_asset_c2_no_chinext_cap_no_lock",
        "generated_on": date.today().isoformat(),
        "research_cutoff": end.isoformat(),
        "git_commit": _git(root, "rev-parse", "HEAD"),
        "selected_c2": asdict(SELECTED_C2),
        "excluded_emergency_cap_asset": CHINEXT_ASSET,
        "min_hold_days": NO_LOCK_MIN_HOLD_DAYS,
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
    parser.add_argument("--defender-dir", type=Path, default=DEFAULT_DEFENDER_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--end", type=date.fromisoformat, default=DEFAULT_END)
    args = parser.parse_args()
    run_experiment(args.root, args.defender_dir, args.output, args.end)


if __name__ == "__main__":
    main()
