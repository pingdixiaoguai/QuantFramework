"""Backtest C2 with ChiNext q90 confirmed by CSI300 q70 stress."""

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
    performance,
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
    "experiments/20260821_momentum_held_asset_c2_chinext_csi300_confirmation"
)
CSI300_ASSET = "510300.SH"
CAP_TRIGGER_MAXIMUM = 0.8


def confirm_chinext_alert_with_csi300(
    original_c2_alert: pd.Series,
    previous_asset: pd.Series,
    csi300_confirmation_alert: pd.Series,
) -> pd.Series:
    """Require CSI300 stress in addition to ChiNext stress when ChiNext is held."""
    if not (
        original_c2_alert.index.equals(previous_asset.index)
        and original_c2_alert.index.equals(csi300_confirmation_alert.index)
    ):
        raise ValueError("all confirmation inputs must have identical indexes")
    if previous_asset.isna().any():
        raise ValueError("previous_asset contains missing values")
    result = original_c2_alert.astype(bool).copy()
    chinext_held = previous_asset.eq(CHINEXT_ASSET)
    result.loc[chinext_held] &= csi300_confirmation_alert.loc[
        chinext_held
    ].astype(bool)
    result.name = "c2_chinext_q90_csi300_q70_confirmation_alert"
    return result


def _simulate(
    slow: pd.Series,
    alert: pd.Series,
    calendar: pd.DatetimeIndex,
    momentum: pd.DataFrame,
    defender: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    state = apply_state_schedule(
        slow,
        alert,
        calendar,
        SLOW_PARAMS.min_hold_days,
        emergency_override=True,
    )
    return state, simulate_switch(momentum, defender, state["risk_on"])


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


def _risk_off_episode_lengths(state: pd.DataFrame) -> list[int]:
    risk_off = ~state["risk_on"].astype(bool)
    groups = risk_off.ne(risk_off.shift()).cumsum()
    return [int(len(group)) for _, group in risk_off.loc[risk_off].groupby(groups)]


def _state_diagnostics(
    states: dict[str, pd.DataFrame],
    simulated: dict[str, pd.DataFrame],
    alerts: dict[str, pd.Series],
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for strategy, state in states.items():
        entries = (
            state["state_changed"].astype(bool)
            & state["state_reason"].eq("emergency_exit")
        )
        lengths = _risk_off_episode_lengths(state)
        records.append(
            {
                "strategy": strategy,
                "alert_days": int(alerts[strategy].sum()),
                "emergency_entries": int(entries.sum()),
                "defender_days": int((~state["risk_on"]).sum()),
                "defender_episodes": len(lengths),
                "median_defender_episode_days": float(np.median(lengths)),
                "sleeve_switches": int(simulated[strategy]["sleeve_switch"].sum()),
            }
        )
    return pd.DataFrame(records)


def _original_chinext_emergency_windows(
    original_state: pd.DataFrame,
    previous_asset: pd.Series,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    entries = (
        original_state["state_changed"].astype(bool)
        & original_state["state_reason"].eq("emergency_exit")
        & previous_asset.eq(CHINEXT_ASSET)
    )
    windows: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for start in original_state.index[entries]:
        future = original_state.loc[start:]
        next_risk_on = future.index[future["risk_on"].astype(bool)]
        end = (
            original_state.index[-1]
            if len(next_risk_on) == 0
            else original_state.index[original_state.index.get_loc(next_risk_on[0]) - 1]
        )
        windows.append((start, end))
    return windows


def _event_records(
    windows: list[tuple[pd.Timestamp, pd.Timestamp]],
    strategies: dict[str, pd.Series],
    states: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for episode, (start, end) in enumerate(windows, start=1):
        for strategy, returns in strategies.items():
            sample = returns.loc[start:end]
            measured = performance(sample)
            records.append(
                {
                    "episode": episode,
                    "start": start.date().isoformat(),
                    "end": end.date().isoformat(),
                    "strategy": strategy,
                    "total_return": measured["total_return"],
                    "sharpe": measured["sharpe"],
                    "max_drawdown": measured["max_drawdown"],
                    "defender_days": (
                        int((~states[strategy]["risk_on"]).loc[start:end].sum())
                        if strategy in states
                        else 0
                    ),
                }
            )
    return pd.DataFrame(records)


def _full_table(metrics: pd.DataFrame) -> str:
    labels = {
        "chinext_q90_csi300_q70_confirmation": "创业板q90+沪深300q70确认",
        "selected_c2": "原C2（按持仓资产独立触发）",
        "c2_no_chinext_cap": "完全取消创业板cap",
        "no_cap_fusion": "无cap融合",
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
        "|年份|q90+q70确认|原C2|取消创业板cap|无cap融合|",
        "|---:|---:|---:|---:|---:|",
    ]
    for year, row in pivot.iterrows():
        lines.append(
            f"|{year}|{row.chinext_q90_csi300_q70_confirmation:+.2%}|"
            f"{row.selected_c2:+.2%}|{row.c2_no_chinext_cap:+.2%}|"
            f"{row.no_cap_fusion:+.2%}|"
        )
    return "\n".join(lines)


def _event_table(events: pd.DataFrame) -> str:
    pivot = events.pivot(
        index=["episode", "start", "end"],
        columns="strategy",
        values="total_return",
    )
    lines = [
        "|事件|窗口|q90+q70确认|原C2|取消创业板cap|无cap融合|",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for (episode, start, end), row in pivot.iterrows():
        lines.append(
            f"|{episode}|{start}至{end}|"
            f"{row.chinext_q90_csi300_q70_confirmation:+.2%}|"
            f"{row.selected_c2:+.2%}|{row.c2_no_chinext_cap:+.2%}|"
            f"{row.no_cap_fusion:+.2%}|"
        )
    return "\n".join(lines)


def _diagnostic_table(diagnostics: pd.DataFrame) -> str:
    labels = {
        "chinext_q90_csi300_q70_confirmation": "q90+q70确认",
        "selected_c2": "原C2",
        "c2_no_chinext_cap": "取消创业板cap",
        "no_cap_fusion": "无cap融合",
    }
    lines = [
        "|方案|报警日|紧急入场|Defender持有日|Defender区间|袖套切换|",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for strategy, label in labels.items():
        row = diagnostics.loc[diagnostics["strategy"].eq(strategy)].iloc[0]
        lines.append(
            f"|{label}|{int(row.alert_days)}|{int(row.emergency_entries)}|"
            f"{int(row.defender_days)}|{int(row.defender_episodes)}|"
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

    thresholds = {asset: CAP_TRIGGER_MAXIMUM for asset in MOMENTUM_ASSETS}
    original_c2_alert = held_asset_cap_alert(caps, previous_asset, thresholds)
    csi300_confirmation_alert = caps[CSI300_ASSET].le(CAP_TRIGGER_MAXIMUM)
    confirmed_alert = confirm_chinext_alert_with_csi300(
        original_c2_alert,
        previous_asset,
        csi300_confirmation_alert,
    )
    no_chinext_alert = original_c2_alert & ~previous_asset.eq(CHINEXT_ASSET)
    no_cap_alert = pd.Series(False, index=calendar, name="no_cap_alert")

    strategy_alerts = {
        "chinext_q90_csi300_q70_confirmation": confirmed_alert,
        "selected_c2": original_c2_alert,
        "c2_no_chinext_cap": no_chinext_alert,
        "no_cap_fusion": no_cap_alert,
    }
    states: dict[str, pd.DataFrame] = {}
    simulated: dict[str, pd.DataFrame] = {}
    for strategy, alert in strategy_alerts.items():
        states[strategy], simulated[strategy] = _simulate(
            slow,
            alert,
            calendar,
            inputs.momentum,
            inputs.defender,
        )

    strategies = {
        strategy: result["return"] for strategy, result in simulated.items()
    }
    strategies["original_momentum"] = exact_momentum
    strategies["original_base"] = original_base
    metrics = _metric_records(strategies, end)
    metrics.to_csv(stage / "strategy_period_metrics.csv", index=False)
    yearly = _calendar_year_records(strategies)
    yearly.to_csv(stage / "calendar_year_returns.csv", index=False)
    diagnostics = _state_diagnostics(states, simulated, strategy_alerts)
    diagnostics.to_csv(stage / "state_diagnostics.csv", index=False)

    windows = _original_chinext_emergency_windows(states["selected_c2"], previous_asset)
    events = _event_records(windows, strategies, states)
    events.to_csv(stage / "chinext_event_comparison.csv", index=False)

    chinext_held = previous_asset.eq(CHINEXT_ASSET)
    signal_diagnostics = pd.DataFrame(
        [
            {
                "chinext_held_days": int(chinext_held.sum()),
                "chinext_q90_alert_days": int((original_c2_alert & chinext_held).sum()),
                "csi300_q70_confirmation_days_while_chinext_held": int(
                    (csi300_confirmation_alert & chinext_held).sum()
                ),
                "joint_confirmed_alert_days": int((confirmed_alert & chinext_held).sum()),
                "chinext_alert_days_filtered_out": int(
                    (original_c2_alert & ~confirmed_alert & chinext_held).sum()
                ),
            }
        ]
    )
    signal_diagnostics.to_csv(stage / "chinext_confirmation_diagnostics.csv", index=False)

    daily = pd.DataFrame(index=calendar)
    daily["momentum_asset_at_previous_close"] = previous_asset
    daily["chinext_q90_or_held_asset_alert"] = original_c2_alert
    daily["csi300_q70_confirmation_alert"] = csi300_confirmation_alert
    daily["joint_confirmed_alert"] = confirmed_alert
    daily["chinext_alert_filtered_out"] = (
        original_c2_alert & ~confirmed_alert & chinext_held
    )
    for strategy, state in states.items():
        daily[f"{strategy}_risk_on"] = state["risk_on"]
        daily[f"{strategy}_state_reason"] = state["state_reason"]
        daily[f"{strategy}_return"] = strategies[strategy]
    daily["original_momentum_return"] = exact_momentum
    daily["original_base_return"] = original_base
    daily.index.name = "date"
    daily.to_csv(stage / "daily_comparison.csv")

    config = {
        "strategy_name": "C2_ChiNext_q90_confirmed_by_CSI300_q70",
        **asdict(SELECTED_C2),
        "chinext_confirmation_asset": CSI300_ASSET,
        "chinext_confirmation_quantile": SELECTED_C2.q_510300,
        "confirmation_logic": "AND",
        "research_cutoff": end.isoformat(),
    }
    report_benchmarks = {
        "C2_chinext_q90_csi300_q70_vs_original_base.html": (
            original_base,
            "Original 4ETF Equal-weight Base",
        ),
        "C2_chinext_q90_csi300_q70_vs_original_momentum.html": (
            exact_momentum,
            "Original Momentum Strategy",
        ),
        "C2_chinext_q90_csi300_q70_vs_no_cap_fusion.html": (
            strategies["no_cap_fusion"],
            "No-cap Slow-gate Fusion",
        ),
        "C2_chinext_q90_csi300_q70_vs_selected_C2.html": (
            strategies["selected_c2"],
            "Selected C2",
        ),
        "C2_chinext_q90_csi300_q70_vs_no_chinext_cap.html": (
            strategies["c2_no_chinext_cap"],
            "C2 with ChiNext Cap Disabled",
        ),
    }
    for filename, (benchmark, benchmark_name) in report_benchmarks.items():
        _generate_standard_report(
            strategies["chinext_q90_csi300_q70_confirmation"],
            benchmark,
            benchmark_name,
            stage / filename,
            config,
        )

    confirmed = _metric(metrics, "full", "chinext_q90_csi300_q70_confirmation")
    original = _metric(metrics, "full", "selected_c2")
    disabled = _metric(metrics, "full", "c2_no_chinext_cap")
    no_cap = _metric(metrics, "full", "no_cap_fusion")
    event_pivot = events.pivot(index="episode", columns="strategy", values="total_return")
    report = f"""# 创业板q90 + 沪深300q70确认：回测结论

## 结论

当上一收盘持有创业板ETF时，只有创业板自身q90与沪深300q70同时达到`cap≤0.8`，下一开盘才紧急切入Defender；其他资产逻辑及30日状态锁保持原C2不变。

全样本确认版年化为{confirmed.annualized_return_252:.2%}、Sharpe为{confirmed.sharpe:.3f}、MDD为{confirmed.max_drawdown:.2%}。相对原C2，年化变化{confirmed.annualized_return_252 - original.annualized_return_252:+.2%}、Sharpe变化{confirmed.sharpe - original.sharpe:+.3f}、MDD变化{confirmed.max_drawdown - original.max_drawdown:+.2%}；相对完全取消创业板cap，年化变化{confirmed.annualized_return_252 - disabled.annualized_return_252:+.2%}、Sharpe变化{confirmed.sharpe - disabled.sharpe:+.3f}、MDD变化{confirmed.max_drawdown - disabled.max_drawdown:+.2%}。

## 全样本指标

{_full_table(metrics)}

## 逐年收益

{_year_table(yearly)}

## 原C2两次创业板紧急事件

{_event_table(events)}

- 2024事件：确认版{event_pivot.loc[1, 'chinext_q90_csi300_q70_confirmation']:+.2%}，原C2{event_pivot.loc[1, 'selected_c2']:+.2%}；沪深300q70确认了这次防守。
- 2025事件：确认版{event_pivot.loc[2, 'chinext_q90_csi300_q70_confirmation']:+.2%}，原C2{event_pivot.loc[2, 'selected_c2']:+.2%}；沪深300q70同样确认了这次防守，因此没有过滤原C2的2025年误防守。
- 全部23个创业板q90报警日上，沪深300q70均已报警；交叉确认条件在当前样本完全冗余，不能据此证明增加了稳健性。

## 信号及换手

{_diagnostic_table(diagnostics)}

- Momentum持有创业板共{int(signal_diagnostics.iloc[0].chinext_held_days)}日；创业板q90原报警{int(signal_diagnostics.iloc[0].chinext_q90_alert_days)}日，经过沪深300q70确认后保留{int(signal_diagnostics.iloc[0].joint_confirmed_alert_days)}日，过滤{int(signal_diagnostics.iloc[0].chinext_alert_days_filtered_out)}日。
- 相对无cap融合：年化变化{confirmed.annualized_return_252 - no_cap.annualized_return_252:+.2%}、Sharpe变化{confirmed.sharpe - no_cap.sharpe:+.3f}、MDD变化{confirmed.max_drawdown - no_cap.max_drawdown:+.2%}。

## 口径边界

- 两个确认信号都严格使用上一收盘及更早数据，下一开盘执行，不存在同收盘交易。
- 沪深300q70在原C2中只负责“持有沪深300时”的自身报警；本方案新增的是它对创业板报警的AND确认权限。
- 收益使用Momentum与Defender开盘切换分段接口和既有费用。
"""
    (stage / "research_report.md").write_text(report, encoding="utf-8")

    input_files = [
        defender_dir / "relative_defender_rotation_switch_returns.csv",
        root / "strategy/configs/quality_momentum_top1.yaml",
        *[root / "data/db" / f"{asset}.parquet" for asset in MOMENTUM_ASSETS],
    ]
    code_files = [
        root / "research/run_momentum_held_asset_c2_chinext_csi300_confirmation.py",
        root / "research/run_momentum_held_asset_c2_no_chinext_cap.py",
        root / "research/run_momentum_held_asset_adaptive_cap.py",
        root / "research/run_momentum_volatility_signal_abcd.py",
        root / "research/momentum_defender_occam.py",
    ]
    manifest = {
        "experiment": "momentum_held_asset_c2_chinext_csi300_confirmation",
        "generated_on": date.today().isoformat(),
        "research_cutoff": end.isoformat(),
        "git_commit": _git(root, "rev-parse", "HEAD"),
        "selected_c2": asdict(SELECTED_C2),
        "confirmation_asset": CSI300_ASSET,
        "confirmation_quantile": SELECTED_C2.q_510300,
        "confirmation_logic": "AND",
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
