"""Backtest C2 after disabling only the ChiNext emergency volatility cap.

The ChiNext ETF remains eligible for the Momentum sleeve and continues to
participate in the frozen slow trend gate.  The only change is that a C2
emergency alert is ignored when the Momentum asset held through the previous
close was 159915.SZ.  All other C2 parameters and execution semantics remain
unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
from dataclasses import asdict
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from research.momentum_defender_occam import (
    HELD_RETURN,
    MOMENTUM_ASSETS,
    OccamParams,
    build_inputs,
    performance,
    slow_regime_at_open,
)
from research.run_momentum_defender_occam import _generate_standard_report
from research.run_momentum_held_asset_adaptive_cap import (
    AdaptiveCSpec,
    held_asset_cap_alert,
)
from research.run_momentum_volatility_signal_abcd import (
    DEFAULT_DEFENDER_DIR,
    DEFAULT_END,
    _load_ohlc,
    asof_previous_close,
    evaluate_alert,
    expanding_volatility_cap,
    momentum_asset_at_previous_close,
    rogers_satchell_volatility,
)


DEFAULT_OUTPUT = Path(
    "experiments/20260821_momentum_held_asset_c2_no_chinext_cap"
)
CHINEXT_ASSET = "159915.SZ"
SLOW_PARAMS = OccamParams(40, 0.025, 30, None)
SELECTED_C2 = AdaptiveCSpec(
    scheme="C2",
    label="Selected C2",
    volatility_window=10,
    cap_trigger_maximum=0.8,
    q_510300=0.70,
    q_159915=0.90,
    q_513100=0.95,
    q_518880=0.90,
)

PERIODS = {
    "development_2019_2022": (
        pd.Timestamp("2019-01-18"),
        pd.Timestamp("2022-12-30"),
    ),
    "2023": (pd.Timestamp("2023-01-01"), pd.Timestamp("2023-12-31")),
    "2024": (pd.Timestamp("2024-01-01"), pd.Timestamp("2024-12-31")),
    "2025": (pd.Timestamp("2025-01-01"), pd.Timestamp("2025-12-31")),
    "2026_ytd": (pd.Timestamp("2026-01-01"), pd.Timestamp(DEFAULT_END)),
    "full": (pd.Timestamp("2019-01-18"), pd.Timestamp(DEFAULT_END)),
}


def suppress_alert_for_asset(
    alert: pd.Series,
    previous_asset: pd.Series,
    excluded_asset: str,
) -> pd.Series:
    """Ignore alerts only when the excluded asset was held at prior close."""
    if not alert.index.equals(previous_asset.index):
        raise ValueError("alert and previous_asset must have identical indexes")
    if previous_asset.isna().any():
        raise ValueError("previous_asset contains missing values")
    return (alert.astype(bool) & ~previous_asset.eq(excluded_asset)).rename(
        alert.name
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()


def _metric_records(
    strategies: dict[str, pd.Series],
    end: date,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    effective_periods = {
        **PERIODS,
        "2026_ytd": (pd.Timestamp("2026-01-01"), pd.Timestamp(end)),
        "full": (pd.Timestamp("2019-01-18"), pd.Timestamp(end)),
    }
    for period, (start, finish) in effective_periods.items():
        for strategy, returns in strategies.items():
            sample = returns.loc[start:finish]
            measured = performance(sample)
            records.append(
                {
                    "period": period,
                    "strategy": strategy,
                    **measured,
                }
            )
    return pd.DataFrame(records)


def _calendar_year_records(strategies: dict[str, pd.Series]) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for year in sorted(strategies["c2_no_chinext_cap"].index.year.unique()):
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


def _defender_episode_end(
    state: pd.DataFrame,
    entry: pd.Timestamp,
) -> pd.Timestamp:
    future = state.loc[entry:]
    next_risk_on = future.index[future["risk_on"].astype(bool)]
    if len(next_risk_on) == 0:
        return state.index[-1]
    exit_position = state.index.get_loc(next_risk_on[0])
    if exit_position == 0:
        raise AssertionError("emergency entry cannot end before it starts")
    return state.index[exit_position - 1]


def _chinext_episode_records(
    original_state: pd.DataFrame,
    variant_state: pd.DataFrame,
    original_alert: pd.Series,
    variant_alert: pd.Series,
    previous_asset: pd.Series,
    strategies: dict[str, pd.Series],
) -> pd.DataFrame:
    emergency_entry = (
        original_state["state_changed"].astype(bool)
        & original_state["state_reason"].eq("emergency_exit")
        & previous_asset.eq(CHINEXT_ASSET)
    )
    records: list[dict[str, object]] = []
    for episode, entry in enumerate(original_state.index[emergency_entry], start=1):
        finish = _defender_episode_end(original_state, entry)
        for strategy, returns in strategies.items():
            sample = returns.loc[entry:finish]
            measured = performance(sample)
            records.append(
                {
                    "episode": episode,
                    "start": entry.date().isoformat(),
                    "end": finish.date().isoformat(),
                    "strategy": strategy,
                    "total_return": measured["total_return"],
                    "annualized_volatility": measured["annualized_volatility"],
                    "sharpe": measured["sharpe"],
                    "max_drawdown": measured["max_drawdown"],
                    "original_c2_alert_days": int(
                        original_alert.loc[entry:finish].sum()
                    ),
                    "variant_alert_days": int(variant_alert.loc[entry:finish].sum()),
                    "original_c2_defender_days": int(
                        (~original_state["risk_on"]).loc[entry:finish].sum()
                    ),
                    "variant_defender_days": int(
                        (~variant_state["risk_on"]).loc[entry:finish].sum()
                    ),
                }
            )
    return pd.DataFrame(records)


def _state_divergence_records(
    original_state: pd.DataFrame,
    variant_state: pd.DataFrame,
    original_returns: pd.Series,
    variant_returns: pd.Series,
) -> pd.DataFrame:
    """Summarize direct and downstream state differences caused by suppression."""
    different = original_state["risk_on"].ne(variant_state["risk_on"])
    groups = different.ne(different.shift()).cumsum()
    records: list[dict[str, object]] = []
    for episode, (_, positions) in enumerate(
        groups.loc[different].groupby(groups.loc[different]), start=1
    ):
        index = positions.index
        original_sample = original_returns.loc[index]
        variant_sample = variant_returns.loc[index]
        records.append(
            {
                "divergence_episode": episode,
                "start": index.min().date().isoformat(),
                "end": index.max().date().isoformat(),
                "observations": len(index),
                "original_c2_sleeve": (
                    "momentum"
                    if bool(original_state.at[index[0], "risk_on"])
                    else "defender"
                ),
                "variant_sleeve": (
                    "momentum"
                    if bool(variant_state.at[index[0], "risk_on"])
                    else "defender"
                ),
                "original_c2_total_return": float(
                    (1.0 + original_sample).prod() - 1.0
                ),
                "variant_total_return": float((1.0 + variant_sample).prod() - 1.0),
                "variant_log_excess_vs_original_c2": float(
                    np.log1p(variant_sample).sum() - np.log1p(original_sample).sum()
                ),
            }
        )
    return pd.DataFrame(records)


def _metric(metrics: pd.DataFrame, period: str, strategy: str) -> pd.Series:
    return metrics.loc[
        metrics["period"].eq(period) & metrics["strategy"].eq(strategy)
    ].iloc[0]


def _comparison_table(metrics: pd.DataFrame) -> str:
    labels = {
        "c2_no_chinext_cap": "C2仅取消创业板cap",
        "selected_c2": "原C2",
        "no_cap_fusion": "无cap融合",
        "original_momentum": "原动量策略",
        "original_base": "原4ETF等权base",
    }
    lines = [
        "|方案|年化收益|年化波动|Sharpe|最大回撤|",
        "|---|---:|---:|---:|---:|",
    ]
    for strategy in labels:
        row = _metric(metrics, "full", strategy)
        lines.append(
            f"|{labels[strategy]}|{row.annualized_return_252:.2%}|"
            f"{row.annualized_volatility:.2%}|{row.sharpe:.3f}|"
            f"{row.max_drawdown:.2%}|"
        )
    return "\n".join(lines)


def _year_table(yearly: pd.DataFrame) -> str:
    pivot = yearly.pivot(index="year", columns="strategy", values="total_return")
    lines = [
        "|年份|仅取消创业板cap|原C2|无cap融合|原动量|",
        "|---:|---:|---:|---:|---:|",
    ]
    for year, row in pivot.iterrows():
        lines.append(
            f"|{year}|{row.c2_no_chinext_cap:+.2%}|{row.selected_c2:+.2%}|"
            f"{row.no_cap_fusion:+.2%}|{row.original_momentum:+.2%}|"
        )
    return "\n".join(lines)


def _episode_table(episodes: pd.DataFrame) -> str:
    pivot = episodes.pivot(
        index=["episode", "start", "end"],
        columns="strategy",
        values="total_return",
    )
    lines = [
        "|创业板cap事件|窗口|仅取消创业板cap|原C2|无cap融合|原动量|",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for (episode, start, end), row in pivot.iterrows():
        lines.append(
            f"|{episode}|{start}至{end}|{row.c2_no_chinext_cap:+.2%}|"
            f"{row.selected_c2:+.2%}|{row.no_cap_fusion:+.2%}|"
            f"{row.original_momentum:+.2%}|"
        )
    return "\n".join(lines)


def _divergence_table(divergences: pd.DataFrame) -> str:
    lines = [
        "|状态差异窗口|原C2持仓|调整版持仓|原C2收益|调整版收益|调整版对数超额|",
        "|---|---|---|---:|---:|---:|",
    ]
    for _, row in divergences.iterrows():
        lines.append(
            f"|{row.start}至{row.end}|{row.original_c2_sleeve}|"
            f"{row.variant_sleeve}|{row.original_c2_total_return:+.2%}|"
            f"{row.variant_total_return:+.2%}|"
            f"{row.variant_log_excess_vs_original_c2:+.4f}|"
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
        quantile = SELECTED_C2.asset_quantiles()[asset]
        close_cap = expanding_volatility_cap(volatility, quantile)["cap"]
        caps[asset] = asof_previous_close(close_cap, calendar).fillna(1.0)

    original_alert = held_asset_cap_alert(
        caps,
        previous_asset,
        {asset: float(SELECTED_C2.cap_trigger_maximum) for asset in MOMENTUM_ASSETS},
    )
    variant_alert = suppress_alert_for_asset(
        original_alert,
        previous_asset,
        CHINEXT_ASSET,
    )
    no_cap_alert = pd.Series(False, index=calendar, name="no_cap_alert")

    _, original_state, original_simulated = evaluate_alert(
        SELECTED_C2,
        original_alert,
        slow,
        inputs.momentum,
        inputs.defender,
        exact_momentum,
    )
    variant_spec = AdaptiveCSpec(
        **{
            **asdict(SELECTED_C2),
            "scheme": "C2_no_chinext_cap",
            "label": "C2 with ChiNext emergency cap disabled",
        }
    )
    _, variant_state, variant_simulated = evaluate_alert(
        variant_spec,
        variant_alert,
        slow,
        inputs.momentum,
        inputs.defender,
        exact_momentum,
    )
    no_cap_spec = AdaptiveCSpec("N", "No emergency cap")
    _, no_cap_state, no_cap_simulated = evaluate_alert(
        no_cap_spec,
        no_cap_alert,
        slow,
        inputs.momentum,
        inputs.defender,
        exact_momentum,
    )

    strategies = {
        "c2_no_chinext_cap": variant_simulated["return"],
        "selected_c2": original_simulated["return"],
        "no_cap_fusion": no_cap_simulated["return"],
        "original_momentum": exact_momentum,
        "original_base": original_base,
    }
    metrics = _metric_records(strategies, end)
    metrics.to_csv(stage / "strategy_period_metrics.csv", index=False)
    yearly = _calendar_year_records(strategies)
    yearly.to_csv(stage / "calendar_year_returns.csv", index=False)

    episodes = _chinext_episode_records(
        original_state,
        variant_state,
        original_alert,
        variant_alert,
        previous_asset,
        strategies,
    )
    episodes.to_csv(stage / "chinext_cap_event_comparison.csv", index=False)

    divergences = _state_divergence_records(
        original_state,
        variant_state,
        original_simulated["return"],
        variant_simulated["return"],
    )
    divergences.to_csv(stage / "state_path_divergence.csv", index=False)

    original_emergency_entry = (
        original_state["state_changed"].astype(bool)
        & original_state["state_reason"].eq("emergency_exit")
    )
    variant_emergency_entry = (
        variant_state["state_changed"].astype(bool)
        & variant_state["state_reason"].eq("emergency_exit")
    )
    diagnostics = pd.DataFrame(
        [
            {
                "strategy": "selected_c2",
                "alert_days": int(original_alert.sum()),
                "emergency_entries": int(original_emergency_entry.sum()),
                "defender_days": int((~original_state["risk_on"]).sum()),
                "sleeve_switches": int(original_simulated["sleeve_switch"].sum()),
            },
            {
                "strategy": "c2_no_chinext_cap",
                "alert_days": int(variant_alert.sum()),
                "emergency_entries": int(variant_emergency_entry.sum()),
                "defender_days": int((~variant_state["risk_on"]).sum()),
                "sleeve_switches": int(variant_simulated["sleeve_switch"].sum()),
            },
            {
                "strategy": "no_cap_fusion",
                "alert_days": 0,
                "emergency_entries": 0,
                "defender_days": int((~no_cap_state["risk_on"]).sum()),
                "sleeve_switches": int(no_cap_simulated["sleeve_switch"].sum()),
            },
        ]
    )
    diagnostics.to_csv(stage / "state_diagnostics.csv", index=False)

    daily = pd.DataFrame(index=calendar)
    daily["momentum_asset_at_previous_close"] = previous_asset
    daily["selected_c2_alert"] = original_alert
    daily["c2_no_chinext_cap_alert"] = variant_alert
    daily["alert_suppressed"] = original_alert & ~variant_alert
    daily["selected_c2_risk_on"] = original_state["risk_on"]
    daily["selected_c2_state_reason"] = original_state["state_reason"]
    daily["c2_no_chinext_cap_risk_on"] = variant_state["risk_on"]
    daily["c2_no_chinext_cap_state_reason"] = variant_state["state_reason"]
    daily["selected_c2_return"] = strategies["selected_c2"]
    daily["c2_no_chinext_cap_return"] = strategies["c2_no_chinext_cap"]
    daily["no_cap_fusion_return"] = strategies["no_cap_fusion"]
    daily["original_momentum_return"] = strategies["original_momentum"]
    daily["original_base_return"] = strategies["original_base"]
    daily.index.name = "date"
    daily.to_csv(stage / "daily_comparison.csv")

    config = {
        "strategy_name": "C2_no_chinext_emergency_cap",
        **asdict(SLOW_PARAMS),
        **asdict(SELECTED_C2),
        "excluded_emergency_cap_asset": CHINEXT_ASSET,
        "research_cutoff": end.isoformat(),
    }
    report_benchmarks = {
        "C2_no_chinext_cap_vs_original_base.html": (
            original_base,
            "Original 4ETF Equal-weight Base",
        ),
        "C2_no_chinext_cap_vs_original_momentum.html": (
            exact_momentum,
            "Original Momentum Strategy",
        ),
        "C2_no_chinext_cap_vs_no_cap_fusion.html": (
            no_cap_simulated["return"],
            "No-cap Slow-gate Fusion",
        ),
        "C2_no_chinext_cap_vs_selected_C2.html": (
            original_simulated["return"],
            "Selected C2",
        ),
    }
    for filename, (benchmark, benchmark_name) in report_benchmarks.items():
        _generate_standard_report(
            variant_simulated["return"],
            benchmark,
            benchmark_name,
            stage / filename,
            config,
        )

    variant_full = _metric(metrics, "full", "c2_no_chinext_cap")
    original_full = _metric(metrics, "full", "selected_c2")
    no_cap_full = _metric(metrics, "full", "no_cap_fusion")
    event_pivot = episodes.pivot(
        index="episode", columns="strategy", values="total_return"
    )
    removed_2024_benefit = (
        event_pivot.loc[1, "c2_no_chinext_cap"]
        - event_pivot.loc[1, "selected_c2"]
    )
    avoided_2025_cost = (
        event_pivot.loc[2, "c2_no_chinext_cap"]
        - event_pivot.loc[2, "selected_c2"]
    )
    reasonable = (
        variant_full["annualized_return_252"] >= no_cap_full["annualized_return_252"]
        and variant_full["sharpe"] >= no_cap_full["sharpe"]
        and variant_full["max_drawdown"] >= no_cap_full["max_drawdown"]
    )
    verdict = (
        "历史样本支持把它作为C2的简化版继续验证。"
        if reasonable
        else "历史样本不支持把它直接替换为正式C2。"
    )
    report = f"""# C2仅取消创业板ETF紧急cap：回测结论

## 结论

{verdict} 该改动的经济直觉是成立的：创业板cap参数在既有稳定性检验中最不稳定，而且取消它不会改变创业板的Momentum资格、慢门控或其他三只ETF的cap。但它仍是看过2024和2025结果后的结构调整，不能因为参数更少就自动视为不过拟合。

全样本中，调整版年化为{variant_full.annualized_return_252:.2%}、Sharpe为{variant_full.sharpe:.3f}、MDD为{variant_full.max_drawdown:.2%}。相对原C2，年化变化{variant_full.annualized_return_252 - original_full.annualized_return_252:+.2%}、Sharpe变化{variant_full.sharpe - original_full.sharpe:+.3f}、MDD变化{variant_full.max_drawdown - original_full.max_drawdown:+.2%}；相对无cap融合，年化变化{variant_full.annualized_return_252 - no_cap_full.annualized_return_252:+.2%}、Sharpe变化{variant_full.sharpe - no_cap_full.sharpe:+.3f}、MDD变化{variant_full.max_drawdown - no_cap_full.max_drawdown:+.2%}。

## 全样本指标

{_comparison_table(metrics)}

## 逐年收益

{_year_table(yearly)}

## 被取消的创业板cap事件

{_episode_table(episodes)}

- 2024年事件中，取消创业板cap使窗口收益相对原C2变化{removed_2024_benefit:+.2%}；这是放弃当前C2最关键的一次成功防守。
- 2025年原C2事件窗口中，调整版收益相对原C2变化{avoided_2025_cost:+.2%}。但这并不只是“避免一次误防守”：调整版在2025-10-16又被其他资产cap切入Defender，30日锁又使它延迟到2025-11-26才切回Momentum，形成额外的路径收益。
- 两次创业板事件的样本数仍然太少，不能据此认定创业板cap未来无效。这个版本本质上是在“放弃一次巨大左尾保护”与“减少一次误防守”之间取舍。

## 30日锁造成的后续路径差异

{_divergence_table(divergences)}

- 只屏蔽了{int((original_alert & ~variant_alert).sum())}个报警日、少了2次创业板紧急入场，但由于30交易日状态锁会重置，两个策略共有{int(original_state['risk_on'].ne(variant_state['risk_on']).sum())}个交易日持仓状态不同。
- 因此2025年的改善不应全部归因于“创业板cap无效”；其中相当一部分来自后来另一次cap与30日锁恰好形成的有利相位。这也是该调整最主要的路径依赖风险。

## 口径

- 唯一改动：上一收盘Momentum持仓为`159915.SZ`时，忽略C2紧急波动cap。
- 冻结不变：40日慢门控、2.5%阈值、30交易日状态锁、10日Rogers–Satchell波动率、cap≤0.8，以及沪深300 q70、纳指 q95、黄金 q90。
- 创业板ETF仍可被Momentum选中；只取消它触发紧急切入Defender的权限。
- 所有信号使用上一收盘及更早数据，下一开盘执行；收益使用Momentum与Defender的开盘切换分段接口，含既有费用。
"""
    (stage / "research_report.md").write_text(report, encoding="utf-8")

    input_files = [
        defender_dir / "relative_defender_rotation_switch_returns.csv",
        root / "strategy/configs/quality_momentum_top1.yaml",
        *[root / "data/db" / f"{asset}.parquet" for asset in MOMENTUM_ASSETS],
    ]
    code_files = [
        root / "research/run_momentum_held_asset_c2_no_chinext_cap.py",
        root / "research/run_momentum_held_asset_adaptive_cap.py",
        root / "research/run_momentum_volatility_signal_abcd.py",
        root / "research/momentum_defender_occam.py",
    ]
    manifest = {
        "experiment": "momentum_held_asset_c2_no_chinext_cap",
        "generated_on": date.today().isoformat(),
        "research_cutoff": end.isoformat(),
        "git_commit": _git(root, "rev-parse", "HEAD"),
        "selected_c2": asdict(SELECTED_C2),
        "excluded_emergency_cap_asset": CHINEXT_ASSET,
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
