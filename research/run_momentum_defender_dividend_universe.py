"""Run and persist the retrospective Defender ETF-universe audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from research.momentum_defender_dividend_universe import (
    dedupe_pools,
    difference_events,
    load_harness,
    run_universe,
)
from research.momentum_defender_gold_override_overfit import (
    cscv_pbo,
    expanding_walk_forward,
    full_metrics,
    leave_one_year_selection,
    paired_block_bootstrap,
    yearly_reality_check,
)
from research.momentum_defender_occam import performance
from research.standard_report import generate_standard_report
from strategy.momentum_defender_w40_gold_escape import run_formal_strategy


DEFAULT_CONFIG = Path(
    "research/configs/momentum_defender_dividend_universe_search.yaml"
)
DEFAULT_OUTPUT = Path(
    "experiments/20260826_momentum_defender_dividend_universe"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _return_hash(series: pd.Series) -> str:
    return hashlib.sha256(series.to_numpy(dtype="<f8").tobytes()).hexdigest()


def _pool_family(config: dict) -> dict[str, tuple[str, ...]]:
    baseline = tuple(config["baseline_pool"])
    selected = tuple(config["selected_research_pool"])
    screened = [row["asset"] for row in config["screened_assets"]]
    active = [
        row["asset"]
        for row in config["screened_assets"]
        if float(row["median60_amount_yi"])
        >= float(config["screen"]["minimum_median_daily_amount_yi"])
    ]
    pools: dict[str, tuple[str, ...]] = {
        "baseline": baseline,
        "selected_liquidity_exposure": selected,
    }
    for asset in baseline:
        if asset != "510880.SH":
            pools[f"baseline_remove_{asset}"] = tuple(
                value for value in baseline if value != asset
            )
    for asset in screened:
        if asset not in baseline:
            pools[f"baseline_add_{asset}"] = (*baseline, asset)
    pools["full_size_screen"] = tuple(screened)
    pools["full_size_and_activity_screen"] = tuple(active)
    pools["representative_eight"] = (
        "512890.SH",
        "510880.SH",
        "515450.SH",
        "515180.SH",
        "513630.SH",
        "159545.SZ",
        "513530.SH",
        "159691.SZ",
    )
    structured = {
        "structured_remove_159545": (
            "512890.SH", "513530.SH", "515080.SH", "510880.SH", "563020.SH"
        ),
        "structured_remove_159545_563020": (
            "512890.SH", "513530.SH", "515080.SH", "510880.SH"
        ),
        "structured_add_515450": (
            "512890.SH", "513530.SH", "515080.SH", "510880.SH", "515450.SH"
        ),
        "structured_replace_515080_with_515180": (
            "512890.SH", "513530.SH", "515180.SH", "510880.SH", "515450.SH"
        ),
        "structured_add_159307": (
            "512890.SH", "513530.SH", "515080.SH", "510880.SH", "515450.SH", "159307.SZ"
        ),
        "structured_add_520990": (
            "512890.SH", "513530.SH", "515080.SH", "510880.SH", "515450.SH", "520990.SH"
        ),
        "structured_add_513630_159307": (
            "512890.SH", "513530.SH", "515080.SH", "510880.SH", "515450.SH", "513630.SH", "159307.SZ"
        ),
        "structured_add_159307_520990": (
            "512890.SH", "513530.SH", "515080.SH", "510880.SH", "515450.SH", "159307.SZ", "520990.SH"
        ),
        "structured_only_legacy_three": (
            "512890.SH", "513530.SH", "510880.SH"
        ),
    }
    pools.update(structured)
    for asset in selected:
        if asset != "510880.SH":
            pools[f"selected_remove_{asset}"] = tuple(
                value for value in selected if value != asset
            )
    for asset in active:
        if asset not in selected:
            pools[f"selected_add_{asset}"] = (*selected, asset)
    return dedupe_pools(pools)


def _segment_metrics(
    returns: Mapping[str, pd.Series], config: dict
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path_id, series in returns.items():
        for segment, boundaries in config["segments"].items():
            sample = series.loc[str(boundaries[0]) : str(boundaries[1])]
            rows.append({"path_id": path_id, "segment": segment, **performance(sample)})
    return pd.DataFrame(rows)


def _annual_metrics(returns: Mapping[str, pd.Series]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path_id, series in returns.items():
        for year, sample in series.groupby(series.index.year):
            rows.append(
                {
                    "path_id": path_id,
                    "year": int(year),
                    **performance(sample),
                }
            )
    return pd.DataFrame(rows)


def _leave_one_event(
    candidate: pd.Series,
    baseline: pd.Series,
    events: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for event in events.itertuples(index=False):
        counterfactual = candidate.copy()
        counterfactual.loc[event.start : event.end] = baseline.loc[event.start : event.end]
        rows.append(
            {
                "event_id": int(event.event_id),
                "removed_start": event.start,
                "removed_end": event.end,
                **performance(counterfactual),
            }
        )
    return pd.DataFrame(rows)


def _event_summary(events: pd.DataFrame, leave_one: pd.DataFrame) -> dict[str, float | int]:
    positive = events.loc[events["log_excess"].gt(0.0), "log_excess"].sort_values(
        ascending=False
    )
    positive_sum = float(positive.sum())
    return {
        "event_count": int(len(events)),
        "positive_events": int(events["log_excess"].gt(0.0).sum()),
        "negative_events": int(events["log_excess"].lt(0.0).sum()),
        "top_two_positive_log_excess_share": (
            float(positive.head(2).sum() / positive_sum) if positive_sum > 0.0 else 0.0
        ),
        "leave_one_event_min_annualized_return_252": float(
            leave_one["annualized_return_252"].min()
        ),
        "leave_one_event_min_sharpe": float(leave_one["sharpe"].min()),
    }


def _report(
    config: dict,
    full: pd.DataFrame,
    segments: pd.DataFrame,
    diagnostics: dict,
    event_summary: dict,
    path_count: int,
) -> str:
    selected_id = "selected_liquidity_exposure"
    baseline = full.loc["baseline"]
    selected = full.loc[selected_id]
    segment = segments.set_index(["path_id", "segment"])
    bootstrap = diagnostics["paired_bootstrap"]
    reality = diagnostics["reality_check"]
    cscv = diagnostics["cscv"]
    wf = diagnostics["walk_forward"]
    loo = diagnostics["leave_one_year"]
    return f"""# Defender红利ETF候选池审计（2026-08-26）

## 结论

在不改Momentum、W40、40日最弱反转、黄金逃生和执行时序的前提下，研究候选删除
`159545.SZ`与同指数重复的`563020.SH`，加入`515450.SH`和`513630.SH`。完整历史年化从
{baseline['annualized_return_252']:.2%}提高到{selected['annualized_return_252']:.2%}，Sharpe从
{baseline['sharpe']:.3f}提高到{selected['sharpe']:.3f}，最大回撤为
{selected['max_drawdown']:.2%}（基线{baseline['max_drawdown']:.2%}）。

该结果是回溯研究，不晋升生产。共实际复现{path_count}条唯一收益路径；年度Reality Check
`p={reality['p_value']:.4f}`，20日成对分块Bootstrap的年化差95%区间为
[{bootstrap['annualized_return_delta_ci_lower']:+.2%},
{bootstrap['annualized_return_delta_ci_upper']:+.2%}]，Sharpe差区间为
[{bootstrap['sharpe_delta_ci_lower']:+.3f}, {bootstrap['sharpe_delta_ci_upper']:+.3f}]。
区间跨0，不能把历史改善当成可靠的未来优势。

## 规模与活跃度口径

- 截面日：{config['screen']['snapshot_date']}；ETF名称或指数必须明确含红利/股息属性。
- 规模至少{config['screen']['minimum_size_yi']:.0f}亿元；活跃池还要求近60日成交额中位数至少
  {config['screen']['minimum_median_daily_amount_yi']:.1f}亿元。
- 规模按基金份额×收盘价估算；Tushare `fund_daily.amount`单位为千元，报告换算为亿元。
- 当前规模筛选不是历史逐日规模筛选，因此新ETF只按真实上市日和40日暖机进入回测，但存在
  当前幸存者偏差。

## 固定分段

|分段|基线年化|候选年化|基线Sharpe|候选Sharpe|
|---|---:|---:|---:|---:|
|development|{segment.at[('baseline','development'),'annualized_return_252']:.2%}|{segment.at[(selected_id,'development'),'annualized_return_252']:.2%}|{segment.at[('baseline','development'),'sharpe']:.3f}|{segment.at[(selected_id,'development'),'sharpe']:.3f}|
|validation|{segment.at[('baseline','validation'),'annualized_return_252']:.2%}|{segment.at[(selected_id,'validation'),'annualized_return_252']:.2%}|{segment.at[('baseline','validation'),'sharpe']:.3f}|{segment.at[(selected_id,'validation'),'sharpe']:.3f}|
|recent|{segment.at[('baseline','recent'),'annualized_return_252']:.2%}|{segment.at[(selected_id,'recent'),'annualized_return_252']:.2%}|{segment.at[('baseline','recent'),'sharpe']:.3f}|{segment.at[(selected_id,'recent'),'sharpe']:.3f}|

所有分段都已被本次选择过程观察，只能称回溯分段，不能称独立样本外。

## 稳健性

- CSCV-PBO：{cscv['pbo']:.1%}；训练冠军在测试段击败基线Sharpe比例
  {cscv['selected_beats_baseline_rate']:.1%}。
- 扩展walk-forward收益/Sharpe胜率：{wf['return_win_rate']:.1%}/{wf['sharpe_win_rate']:.1%}。
- 留一年重选收益/Sharpe胜率：{loo['return_win_rate']:.1%}/{loo['sharpe_win_rate']:.1%}。
- 候选与基线路径差异被切为{event_summary['event_count']}个连续事件，正/负事件
  {event_summary['positive_events']}/{event_summary['negative_events']}；前两大正事件占全部正向
  log excess的{event_summary['top_two_positive_log_excess_share']:.1%}。
- 删除任一差异事件后，最低完整年化/Sharpe为
  {event_summary['leave_one_event_min_annualized_return_252']:.2%}/
  {event_summary['leave_one_event_min_sharpe']:.3f}。

## 治理决定

生产配置保持`momentum_defender_w40_gold_qm20_escape_v1`不变。研究候选只可作为前瞻影子池，
从2026-08-26之后积累未观察数据；未经明确用户晋升决定，不得写入正式配置或实盘持仓。
"""


def run(root: Path, config_path: Path, output: Path) -> dict[str, object]:
    config_path = config_path if config_path.is_absolute() else root / config_path
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    cutoff = date.fromisoformat(str(config["formal_cutoff"]))
    pools = _pool_family(config)
    all_assets = tuple(dict.fromkeys(asset for pool in pools.values() for asset in pool))
    harness = load_harness(root, all_assets, end=cutoff)

    runs = {}
    returns = {}
    for number, (path_id, pool) in enumerate(pools.items(), start=1):
        print(f"[{number}/{len(pools)}] {path_id}", flush=True)
        result = run_universe(harness, pool)
        runs[path_id] = result
        returns[path_id] = result.returns

    formal = run_formal_strategy(root, end=cutoff).daily["return"].astype(float)
    baseline = returns["baseline"]
    parity = float(baseline.sub(formal).abs().max())
    if parity > 1e-14:
        raise AssertionError(f"custom-universe baseline parity failed: {parity:.3e}")

    selected_id = "selected_liquidity_exposure"
    selected = returns[selected_id]
    returns_frame = pd.DataFrame(returns)
    alternatives = returns_frame.drop(columns="baseline")
    full = full_metrics(alternatives, baseline)
    full.loc["baseline"] = {
        "annualized_return_252": performance(baseline)["annualized_return_252"],
        "sharpe": performance(baseline)["sharpe"],
        "max_drawdown": performance(baseline)["max_drawdown"],
        "delta_annualized_return_252": 0.0,
        "delta_sharpe": 0.0,
        "delta_max_drawdown": 0.0,
    }
    full.index.name = "path_id"
    segments = _segment_metrics(returns, config)
    annual = _annual_metrics(returns)

    checks = config["validation"]
    bootstrap_frame, bootstrap_summary = paired_block_bootstrap(
        selected,
        baseline,
        block_size=int(checks["bootstrap_block_sessions"]),
        repetitions=int(checks["bootstrap_repetitions"]),
        seed=int(checks["random_seed"]),
    )
    reality = yearly_reality_check(
        alternatives,
        baseline,
        repetitions=int(checks["reality_check_repetitions"]),
        seed=int(checks["random_seed"]),
    )
    cscv_frame, cscv_summary = cscv_pbo(
        alternatives, baseline, block_count=int(checks["cscv_blocks"])
    )
    walk_forward = expanding_walk_forward(alternatives, baseline)
    leave_year = leave_one_year_selection(alternatives, baseline)
    wf_summary = {
        "return_win_rate": float(walk_forward["test_return_delta"].gt(0.0).mean()),
        "sharpe_win_rate": float(walk_forward["test_sharpe_delta"].gt(0.0).mean()),
    }
    leave_year_summary = {
        "return_win_rate": float(leave_year["test_return_delta"].gt(0.0).mean()),
        "sharpe_win_rate": float(leave_year["test_sharpe_delta"].gt(0.0).mean()),
    }

    events = difference_events(selected, baseline)
    leave_event = _leave_one_event(selected, baseline, events)
    event_summary = _event_summary(events, leave_event)

    cost_rows = []
    for multiplier in checks["defender_cost_multipliers"]:
        for path_id, pool in (
            ("baseline", tuple(config["baseline_pool"])),
            (selected_id, tuple(config["selected_research_pool"])),
        ):
            result = run_universe(
                harness, pool, defender_cost_multiplier=float(multiplier)
            )
            cost_rows.append(
                {
                    "path_id": path_id,
                    "defender_cost_multiplier": float(multiplier),
                    **performance(result.returns),
                }
            )
    cost_stress = pd.DataFrame(cost_rows)

    selection_comparison = pd.DataFrame(
        {
            "baseline_asset": runs["baseline"].selection["selected_asset"],
            "selected_asset": runs[selected_id].selection["selected_asset"],
        }
    )
    selection_comparison["different"] = selection_comparison.nunique(axis=1).eq(2)
    selection_comparison = selection_comparison.loc[
        selection_comparison["different"]
    ]

    output.mkdir(parents=True, exist_ok=True)
    metadata = pd.DataFrame(
        [
            {"path_id": path_id, "asset_count": len(pool), "assets": "|".join(pool)}
            for path_id, pool in pools.items()
        ]
    )
    metadata.to_csv(output / "path_metadata.csv", index=False)
    returns_frame.to_parquet(output / "candidate_returns.parquet")
    full.sort_values("annualized_return_252", ascending=False).to_csv(
        output / "full_metrics.csv"
    )
    segments.to_csv(output / "segment_metrics.csv", index=False)
    annual.to_csv(output / "annual_metrics.csv", index=False)
    bootstrap_frame.to_csv(output / "paired_block_bootstrap.csv", index=False)
    cscv_frame.to_csv(output / "cscv.csv", index=False)
    walk_forward.to_csv(output / "walk_forward.csv", index=False)
    leave_year.to_csv(output / "leave_one_year.csv", index=False)
    events.to_csv(output / "difference_events.csv", index=False)
    leave_event.to_csv(output / "leave_one_event.csv", index=False)
    cost_stress.to_csv(output / "cost_stress.csv", index=False)
    selection_comparison.to_csv(output / "selection_differences.csv")
    selected_daily = pd.DataFrame(
        {
            "candidate_return": selected,
            "baseline_return": baseline,
            "daily_difference": selected - baseline,
            "candidate_nav": (1.0 + selected).cumprod(),
            "baseline_nav": (1.0 + baseline).cumprod(),
        }
    )
    selected_daily.to_csv(output / "selected_daily.csv")
    selected_daily.to_parquet(output / "selected_daily.parquet")

    diagnostics = {
        "paired_bootstrap": bootstrap_summary,
        "reality_check": reality,
        "cscv": cscv_summary,
        "walk_forward": wf_summary,
        "leave_one_year": leave_year_summary,
    }
    audit = {
        "research_id": config["research_id"],
        "status": "passed_research_only",
        "formal_baseline_parity_max_abs_error": parity,
        "unique_return_paths": int(returns_frame.T.drop_duplicates().shape[0]),
        "attempted_pool_paths": int(len(pools)),
        "baseline_return_hash": _return_hash(baseline),
        "selected_return_hash": _return_hash(selected),
        "baseline_performance": performance(baseline),
        "selected_performance": performance(selected),
        "event_summary": event_summary,
        "diagnostics": diagnostics,
        "production_changed": False,
        "evidence_limit": "retrospective_current_survivor_screen_not_independent_oos",
    }
    (output / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report = _report(
        config,
        full,
        segments,
        diagnostics,
        event_summary,
        audit["unique_return_paths"],
    )
    (output / "REPORT.md").write_text(report, encoding="utf-8")
    generate_standard_report(
        selected,
        baseline,
        "formal_w40_gold_escape_original_defender_pool",
        output / "selected_vs_formal.html",
        config,
    )

    manifest_paths = [
        config_path,
        root / "research/momentum_defender_dividend_universe.py",
        root / "research/run_momentum_defender_dividend_universe.py",
        *[root / "data/db" / f"{asset}.parquet" for asset in all_assets],
    ]
    manifest = [
        {"path": str(path.relative_to(root)), "sha256": _sha256(path)}
        for path in manifest_paths
    ]
    (output / "source_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output if args.output.is_absolute() else root / args.output
    run(root, args.config, output)


if __name__ == "__main__":
    main()
