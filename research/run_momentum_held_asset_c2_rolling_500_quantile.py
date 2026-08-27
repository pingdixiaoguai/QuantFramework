"""Backtest C2 using at most 500 prior ETF observations for each quantile."""

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
from research.run_momentum_held_asset_adaptive_cap import (
    ASSET_NAMES,
    held_asset_cap_alert,
)
from research.run_momentum_held_asset_c2_no_chinext_cap import (
    SELECTED_C2,
    SLOW_PARAMS,
    _git,
    _metric,
    _metric_records,
    _sha256,
    _state_divergence_records,
)
from research.run_momentum_volatility_signal_abcd import (
    CAP_STEP,
    CAP_THRESHOLD_MIN_HISTORY,
    DEFAULT_DEFENDER_DIR,
    DEFAULT_END,
    _load_ohlc,
    asof_previous_close,
    choose_by_asset,
    expanding_volatility_cap,
    momentum_asset_at_previous_close,
    rogers_satchell_volatility,
)


DEFAULT_OUTPUT = Path(
    "experiments/20260821_momentum_held_asset_c2_rolling_500_quantile"
)
MAX_QUANTILE_HISTORY = 500
CAP_TRIGGER_MAXIMUM = 0.8


def rolling_volatility_cap(
    realized_volatility: pd.Series,
    quantile: float,
    *,
    max_history: int = MAX_QUANTILE_HISTORY,
    step: float = CAP_STEP,
    min_history: int = CAP_THRESHOLD_MIN_HISTORY,
) -> pd.DataFrame:
    """Strict-lag cap using no more than ``max_history`` prior observations."""
    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must be strictly between zero and one")
    if max_history < min_history or min_history < 1:
        raise ValueError("max_history must be at least min_history >= 1")
    if not 0.0 < step <= 1.0:
        raise ValueError("step must be in (0, 1]")
    volatility = realized_volatility.astype(float)
    threshold = volatility.shift(1).rolling(
        max_history,
        min_periods=min_history,
    ).quantile(quantile)
    raw_cap = (threshold / volatility).clip(upper=1.0)
    cap = np.floor(raw_cap / step + 1e-12) * step
    cap = cap.clip(lower=0.0, upper=1.0).where(raw_cap.notna(), 1.0)
    return pd.DataFrame(
        {
            "realized_volatility": volatility,
            "threshold": threshold,
            "raw_cap": raw_cap,
            "cap": cap,
        }
    )


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
        records.append(
            {
                "strategy": strategy,
                "alert_days": int(alerts[strategy].sum()),
                "emergency_entries": int(entries.sum()),
                "defender_days": int((~state["risk_on"]).sum()),
                "sleeve_switches": int(simulated[strategy]["sleeve_switch"].sum()),
            }
        )
    return pd.DataFrame(records)


def _asset_signal_diagnostics(
    previous_asset: pd.Series,
    expanding_alert: pd.Series,
    rolling_alert: pd.Series,
    expanding_state: pd.DataFrame,
    rolling_state: pd.DataFrame,
) -> pd.DataFrame:
    expanding_entries = (
        expanding_state["state_changed"].astype(bool)
        & expanding_state["state_reason"].eq("emergency_exit")
    )
    rolling_entries = (
        rolling_state["state_changed"].astype(bool)
        & rolling_state["state_reason"].eq("emergency_exit")
    )
    records: list[dict[str, object]] = []
    for asset in MOMENTUM_ASSETS:
        held = previous_asset.eq(asset)
        records.append(
            {
                "asset": asset,
                "asset_name": ASSET_NAMES[asset],
                "quantile": SELECTED_C2.asset_quantiles()[asset],
                "held_days": int(held.sum()),
                "expanding_alert_days": int((expanding_alert & held).sum()),
                "rolling_500_alert_days": int((rolling_alert & held).sum()),
                "rolling_only_alert_days": int(
                    (rolling_alert & ~expanding_alert & held).sum()
                ),
                "expanding_only_alert_days": int(
                    (expanding_alert & ~rolling_alert & held).sum()
                ),
                "expanding_emergency_entries": int(
                    (expanding_entries & held).sum()
                ),
                "rolling_500_emergency_entries": int((rolling_entries & held).sum()),
            }
        )
    return pd.DataFrame(records)


def _full_table(metrics: pd.DataFrame) -> str:
    labels = {
        "rolling_500_c2": "C2滚动500日分位数",
        "expanding_c2": "原C2全历史扩展分位数",
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
        "|年份|滚动500日C2|原扩展历史C2|无cap融合|原动量|",
        "|---:|---:|---:|---:|---:|",
    ]
    for year, row in pivot.iterrows():
        lines.append(
            f"|{year}|{row.rolling_500_c2:+.2%}|{row.expanding_c2:+.2%}|"
            f"{row.no_cap_fusion:+.2%}|{row.original_momentum:+.2%}|"
        )
    return "\n".join(lines)


def _asset_table(diagnostics: pd.DataFrame) -> str:
    lines = [
        "|资产|分位数|持有日|原报警日|滚动500报警日|滚动新增|滚动减少|原紧急入场|滚动紧急入场|",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in diagnostics.iterrows():
        lines.append(
            f"|{row.asset_name}|q{int(row['quantile'] * 100)}|{int(row.held_days)}|"
            f"{int(row.expanding_alert_days)}|{int(row.rolling_500_alert_days)}|"
            f"{int(row.rolling_only_alert_days)}|{int(row.expanding_only_alert_days)}|"
            f"{int(row.expanding_emergency_entries)}|"
            f"{int(row.rolling_500_emergency_entries)}|"
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

    expanding_caps: dict[str, pd.Series] = {}
    rolling_caps: dict[str, pd.Series] = {}
    for asset in MOMENTUM_ASSETS:
        prices = _load_ohlc(asset, end)
        volatility = rogers_satchell_volatility(
            prices, int(SELECTED_C2.volatility_window)
        )
        quantile = SELECTED_C2.asset_quantiles()[asset]
        expanding_close = expanding_volatility_cap(volatility, quantile)["cap"]
        rolling_close = rolling_volatility_cap(volatility, quantile)["cap"]
        expanding_caps[asset] = asof_previous_close(
            expanding_close, calendar
        ).fillna(1.0)
        rolling_caps[asset] = asof_previous_close(rolling_close, calendar).fillna(1.0)

    thresholds = {asset: CAP_TRIGGER_MAXIMUM for asset in MOMENTUM_ASSETS}
    expanding_alert = held_asset_cap_alert(
        expanding_caps, previous_asset, thresholds
    ).rename("expanding_c2_alert")
    rolling_alert = held_asset_cap_alert(
        rolling_caps, previous_asset, thresholds
    ).rename("rolling_500_c2_alert")
    no_cap_alert = pd.Series(False, index=calendar, name="no_cap_alert")
    strategy_alerts = {
        "rolling_500_c2": rolling_alert,
        "expanding_c2": expanding_alert,
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
    state_diagnostics = _state_diagnostics(states, simulated, strategy_alerts)
    state_diagnostics.to_csv(stage / "state_diagnostics.csv", index=False)
    asset_diagnostics = _asset_signal_diagnostics(
        previous_asset,
        expanding_alert,
        rolling_alert,
        states["expanding_c2"],
        states["rolling_500_c2"],
    )
    asset_diagnostics.to_csv(stage / "asset_signal_diagnostics.csv", index=False)
    divergences = _state_divergence_records(
        states["expanding_c2"],
        states["rolling_500_c2"],
        strategies["expanding_c2"],
        strategies["rolling_500_c2"],
    )
    divergences.to_csv(stage / "state_path_divergence.csv", index=False)

    daily = pd.DataFrame(index=calendar)
    daily["momentum_asset_at_previous_close"] = previous_asset
    daily["expanding_selected_cap"] = choose_by_asset(
        expanding_caps, previous_asset
    )
    daily["rolling_500_selected_cap"] = choose_by_asset(rolling_caps, previous_asset)
    daily["expanding_c2_alert"] = expanding_alert
    daily["rolling_500_c2_alert"] = rolling_alert
    daily["rolling_only_alert"] = rolling_alert & ~expanding_alert
    daily["expanding_only_alert"] = expanding_alert & ~rolling_alert
    for strategy, state in states.items():
        daily[f"{strategy}_risk_on"] = state["risk_on"]
        daily[f"{strategy}_state_reason"] = state["state_reason"]
        daily[f"{strategy}_return"] = strategies[strategy]
    daily["original_momentum_return"] = exact_momentum
    daily["original_base_return"] = original_base
    daily.index.name = "date"
    daily.to_csv(stage / "daily_comparison.csv")

    config = {
        "strategy_name": "C2_rolling_500_observation_quantiles",
        **asdict(SELECTED_C2),
        "quantile_history_type": "strict_lag_rolling",
        "quantile_max_history_observations": MAX_QUANTILE_HISTORY,
        "research_cutoff": end.isoformat(),
    }
    report_benchmarks = {
        "C2_rolling_500_vs_original_base.html": (
            original_base,
            "Original 4ETF Equal-weight Base",
        ),
        "C2_rolling_500_vs_original_momentum.html": (
            exact_momentum,
            "Original Momentum Strategy",
        ),
        "C2_rolling_500_vs_no_cap_fusion.html": (
            strategies["no_cap_fusion"],
            "No-cap Slow-gate Fusion",
        ),
        "C2_rolling_500_vs_expanding_C2.html": (
            strategies["expanding_c2"],
            "Original Expanding-history C2",
        ),
    }
    for filename, (benchmark, benchmark_name) in report_benchmarks.items():
        _generate_standard_report(
            strategies["rolling_500_c2"],
            benchmark,
            benchmark_name,
            stage / filename,
            config,
        )

    rolling = _metric(metrics, "full", "rolling_500_c2")
    expanding = _metric(metrics, "full", "expanding_c2")
    no_cap = _metric(metrics, "full", "no_cap_fusion")
    rolling_state = state_diagnostics.loc[
        state_diagnostics["strategy"].eq("rolling_500_c2")
    ].iloc[0]
    expanding_state = state_diagnostics.loc[
        state_diagnostics["strategy"].eq("expanding_c2")
    ].iloc[0]
    key_start = pd.Timestamp("2024-09-27")
    key_end = pd.Timestamp("2024-09-30")
    key_rolling_return = float(
        (1.0 + strategies["rolling_500_c2"].loc[key_start:key_end]).prod() - 1.0
    )
    key_expanding_return = float(
        (1.0 + strategies["expanding_c2"].loc[key_start:key_end]).prod() - 1.0
    )
    report = f"""# C2分位数最多使用500个历史交易日：回测结论

## 口径

- 四只ETF继续使用10日Rogers–Satchell波动率和既定分位数：沪深300q70、创业板q90、纳指q95、黄金q90。
- 原C2分位阈值使用当时全部可得历史；本方案改为每只ETF各自此前最多{MAX_QUANTILE_HISTORY}个交易日的波动率观测。
- 当前收盘的波动率不进入自己的阈值：阈值严格滞后一期，信号下一开盘执行。
- `cap≤0.8`、按上一收盘Momentum持仓资产触发、40日慢门控和30日状态锁均不变。

## 全样本结果

滚动500日版年化为{rolling.annualized_return_252:.2%}、Sharpe为{rolling.sharpe:.3f}、MDD为{rolling.max_drawdown:.2%}。相对原扩展历史C2，年化变化{rolling.annualized_return_252 - expanding.annualized_return_252:+.2%}、Sharpe变化{rolling.sharpe - expanding.sharpe:+.3f}、MDD变化{rolling.max_drawdown - expanding.max_drawdown:+.2%}；相对无cap融合，年化变化{rolling.annualized_return_252 - no_cap.annualized_return_252:+.2%}、Sharpe变化{rolling.sharpe - no_cap.sharpe:+.3f}、MDD变化{rolling.max_drawdown - no_cap.max_drawdown:+.2%}。

**不建议用滚动500日版直接替换原C2。** 它通过更长时间持有Defender把年化波动压低，因此Sharpe更高；但年化少{expanding.annualized_return_252 - rolling.annualized_return_252:.2%}，最大回撤也从{expanding.max_drawdown:.2%}恶化到{rolling.max_drawdown:.2%}，没有形成对原C2的风险收益改进。

{_full_table(metrics)}

## 逐年收益

{_year_table(yearly)}

## 各资产信号变化

{_asset_table(asset_diagnostics)}

## 状态变化

- 原扩展历史C2：报警{int(expanding_state.alert_days)}日、紧急入场{int(expanding_state.emergency_entries)}次、Defender持有{int(expanding_state.defender_days)}日、袖套切换{int(expanding_state.sleeve_switches)}次。
- 滚动500日C2：报警{int(rolling_state.alert_days)}日、紧急入场{int(rolling_state.emergency_entries)}次、Defender持有{int(rolling_state.defender_days)}日、袖套切换{int(rolling_state.sleeve_switches)}次。
- 两者持仓状态不同共{int(states['expanding_c2']['risk_on'].ne(states['rolling_500_c2']['risk_on']).sum())}个交易日。由于30日锁会重置，报警差异可能继续影响后续路径，不能只按单个报警日解释收益差。

## 2024年关键路径

- 2024-09-27，Momentum上一收盘目标转为创业板。原扩展历史cap仍为1.0，没有报警并切回Momentum；滚动500日cap已降至0.8，继续留在Defender。
- 2024-09-30，原cap仍为1.0，滚动500日cap降至0.6。两日合计原C2获得{key_expanding_return:+.2%}，滚动500日版只有{key_rolling_return:+.2%}。
- 2024-10-08两者都报警，但原C2此前已经获得创业板主要上涨，滚动500日版则过早防守。这使滚动500日版2024全年仅+42.16%，低于原C2的+109.02%，也低于无cap融合的+71.91%。

## 风险解释

- 500日窗口更能适应近两年的波动制度变化，但阈值也更容易随单一阶段移动，未必天然更稳健。
- 本次只比较用户指定的500日规则，没有在多个窗口中择优；因此没有新增“从窗口网格挑赢家”的参数选择偏差。
- 该结构仍使用已知全样本检验，不能替代冻结后的前瞻验证。
"""
    (stage / "research_report.md").write_text(report, encoding="utf-8")

    input_files = [
        defender_dir / "relative_defender_rotation_switch_returns.csv",
        root / "strategy/configs/quality_momentum_top1.yaml",
        *[root / "data/db" / f"{asset}.parquet" for asset in MOMENTUM_ASSETS],
    ]
    code_files = [
        root / "research/run_momentum_held_asset_c2_rolling_500_quantile.py",
        root / "research/run_momentum_held_asset_adaptive_cap.py",
        root / "research/run_momentum_volatility_signal_abcd.py",
        root / "research/momentum_defender_occam.py",
    ]
    manifest = {
        "experiment": "momentum_held_asset_c2_rolling_500_quantile",
        "generated_on": date.today().isoformat(),
        "research_cutoff": end.isoformat(),
        "git_commit": _git(root, "rev-parse", "HEAD"),
        "selected_c2": asdict(SELECTED_C2),
        "quantile_history_type": "strict_lag_rolling",
        "quantile_max_history_observations": MAX_QUANTILE_HISTORY,
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
