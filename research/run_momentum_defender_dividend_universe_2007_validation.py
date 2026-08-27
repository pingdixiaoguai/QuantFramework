"""Replay Defender from 2007 and mechanically isolate 2026 as validation."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Mapping

import pandas as pd
import yaml

from research.momentum_defender_dividend_universe import (
    dedupe_pools,
    difference_events,
    load_harness,
    load_standalone_harness,
    run_standalone_universe,
    run_universe,
)
from research.momentum_defender_gold_override_overfit import (
    cscv_pbo,
    paired_block_bootstrap,
    yearly_reality_check,
)
from research.momentum_defender_occam import performance
from research.standard_report import generate_standard_report
from strategy.momentum_defender_w40_gold_escape import run_formal_strategy


DEFAULT_CONFIG = Path(
    "research/configs/momentum_defender_dividend_universe_2007_validation.yaml"
)
DEFAULT_OUTPUT = Path(
    "experiments/20260826_momentum_defender_dividend_universe_2007_validation"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pool_family(config: dict) -> dict[str, tuple[str, ...]]:
    baseline = tuple(config["baseline_pool"])
    fixed = tuple(config["fixed_candidate_under_test"])
    screened = [row["asset"] for row in config["screened_assets"]]
    active = [
        row["asset"]
        for row in config["screened_assets"]
        if float(row["median60_amount_yi"])
        >= float(config["screen"]["minimum_median_daily_amount_yi"])
    ]
    pools: dict[str, tuple[str, ...]] = {
        "baseline": baseline,
        "fixed_candidate": fixed,
    }
    for asset in baseline:
        if asset != "510880.SH":
            pools[f"baseline_remove_{asset}"] = tuple(
                value for value in baseline if value != asset
            )
    for asset in screened:
        if asset not in baseline:
            pools[f"baseline_add_{asset}"] = (*baseline, asset)
    pools["full_2025_size_screen"] = tuple(screened)
    pools["full_2025_size_activity_screen"] = tuple(active)
    for asset in fixed:
        if asset != "510880.SH":
            pools[f"fixed_remove_{asset}"] = tuple(
                value for value in fixed if value != asset
            )
    for asset in active:
        if asset not in fixed:
            pools[f"fixed_add_{asset}"] = (*fixed, asset)
    pools.update(
        {
            "economic_core_add_520990": (
                "512890.SH",
                "513530.SH",
                "515080.SH",
                "510880.SH",
                "515450.SH",
                "520990.SH",
            ),
            "economic_core_add_513630_520990": (*fixed, "520990.SH"),
            "replace_515080_with_515180": (
                "512890.SH",
                "513530.SH",
                "515180.SH",
                "510880.SH",
                "515450.SH",
                "513630.SH",
            ),
            "only_long_history_three": (
                "512890.SH",
                "513530.SH",
                "510880.SH",
            ),
        }
    )
    return dedupe_pools(pools)


def _period_metrics(
    layer: str,
    returns: Mapping[str, pd.Series],
    config: dict,
) -> pd.DataFrame:
    periods = config["periods"]
    rows: list[dict[str, object]] = []
    for path_id, series in returns.items():
        development_start = (
            periods["standalone_start"]
            if layer == "standalone_defender"
            else periods["integrated_start"]
        )
        for segment, start, end in (
            (
                "development",
                development_start,
                periods["development_end"],
            ),
            (
                "validation_2026",
                periods["validation_start"],
                periods["validation_end"],
            ),
        ):
            sample = series.loc[str(start) : str(end)]
            rows.append(
                {
                    "layer": layer,
                    "path_id": path_id,
                    "segment": segment,
                    **performance(sample),
                }
            )
    return pd.DataFrame(rows)


def _rank_table(metrics: pd.DataFrame, segment: str) -> pd.DataFrame:
    selected = metrics.loc[metrics["segment"].eq(segment)].copy()
    selected = selected.sort_values(
        ["sharpe", "annualized_return_252"], ascending=False
    ).reset_index(drop=True)
    selected.insert(0, "rank", range(1, len(selected) + 1))
    return selected


def _summary_delta(
    metrics: pd.DataFrame,
    layer: str,
    segment: str,
    path_id: str,
) -> dict[str, float]:
    indexed = metrics.set_index(["layer", "segment", "path_id"])
    candidate = indexed.loc[(layer, segment, path_id)]
    baseline = indexed.loc[(layer, segment, "baseline")]
    return {
        "total_return_delta": float(
            candidate["total_return"] - baseline["total_return"]
        ),
        "annualized_return_252_delta": float(
            candidate["annualized_return_252"]
            - baseline["annualized_return_252"]
        ),
        "sharpe_delta": float(candidate["sharpe"] - baseline["sharpe"]),
        "max_drawdown_delta": float(
            candidate["max_drawdown"] - baseline["max_drawdown"]
        ),
    }


def _report(config: dict, audit: dict, metrics: pd.DataFrame) -> str:
    indexed = metrics.set_index(["layer", "segment", "path_id"])

    def row(layer: str, segment: str, path_id: str) -> pd.Series:
        return indexed.loc[(layer, segment, path_id)]

    standalone_dev_base = row("standalone_defender", "development", "baseline")
    standalone_dev_fixed = row("standalone_defender", "development", "fixed_candidate")
    standalone_val_base = row("standalone_defender", "validation_2026", "baseline")
    standalone_val_fixed = row("standalone_defender", "validation_2026", "fixed_candidate")
    integrated_dev_base = row("integrated_composite", "development", "baseline")
    integrated_dev_fixed = row("integrated_composite", "development", "fixed_candidate")
    integrated_val_base = row("integrated_composite", "validation_2026", "baseline")
    integrated_val_fixed = row("integrated_composite", "validation_2026", "fixed_candidate")
    champion = audit["development_champion"]
    bootstrap = audit["validation_bootstrap"]
    return f"""# Defender 2007研发段 / 2026机械验证（2026-08-26）

## 结论

固定研究池在两层测试中都保持同方向领先：

- 独立Defender袖套，2007–2025研发段年化从
  {standalone_dev_base['annualized_return_252']:.2%}升至
  {standalone_dev_fixed['annualized_return_252']:.2%}；2026验证段累计收益从
  {standalone_val_base['total_return']:.2%}升至
  {standalone_val_fixed['total_return']:.2%}。
- 完整W40＋黄金逃生组合，2019–2025可执行研发段年化从
  {integrated_dev_base['annualized_return_252']:.2%}升至
  {integrated_dev_fixed['annualized_return_252']:.2%}；2026验证段累计收益从
  {integrated_val_base['total_return']:.2%}升至
  {integrated_val_fixed['total_return']:.2%}，Sharpe从
  {integrated_val_base['sharpe']:.3f}升至{integrated_val_fixed['sharpe']:.3f}。

因此“当前固定六只研究池在2026没有失效”可以成立；“候选池寻优流程已经稳定”不能成立。
只用研发段选出的Sharpe冠军为`{champion['path_id']}`，它在2026相对基线累计收益差为
{champion['validation_total_return_delta']:+.2%}、Sharpe差为
{champion['validation_sharpe_delta']:+.3f}。研发冠军没有可靠延续，说明追逐历史第一名仍会过拟合。

## 可用历史边界

510880从2007-01-18首个交易日开始。最初40个交易日还没有反转分数，研究按“唯一已上市
红利ETF满仓持有”暖机；之后恢复正式的每月40日收益最低规则。2007–2018期间其他候选尚未
上市，各候选池实际都等价于510880，因此这段历史检验的是Defender资产本身，不提供多ETF
选择证据。

完整组合不能从2007开始：510300门控、创业板、纳指、黄金以及504日门控历史当时并不存在。
强行用代理指数或回填价格会制造不存在的可交易历史。本报告只从正式可执行的2019-01-18开始
比较完整组合。

## 固定候选结果

|层/分段|基线年化|固定池年化|基线Sharpe|固定池Sharpe|基线MDD|固定池MDD|
|---|---:|---:|---:|---:|---:|---:|
|Defender 2007–2025|{standalone_dev_base['annualized_return_252']:.2%}|{standalone_dev_fixed['annualized_return_252']:.2%}|{standalone_dev_base['sharpe']:.3f}|{standalone_dev_fixed['sharpe']:.3f}|{standalone_dev_base['max_drawdown']:.2%}|{standalone_dev_fixed['max_drawdown']:.2%}|
|Defender 2026|{standalone_val_base['annualized_return_252']:.2%}|{standalone_val_fixed['annualized_return_252']:.2%}|{standalone_val_base['sharpe']:.3f}|{standalone_val_fixed['sharpe']:.3f}|{standalone_val_base['max_drawdown']:.2%}|{standalone_val_fixed['max_drawdown']:.2%}|
|完整组合 2019–2025|{integrated_dev_base['annualized_return_252']:.2%}|{integrated_dev_fixed['annualized_return_252']:.2%}|{integrated_dev_base['sharpe']:.3f}|{integrated_dev_fixed['sharpe']:.3f}|{integrated_dev_base['max_drawdown']:.2%}|{integrated_dev_fixed['max_drawdown']:.2%}|
|完整组合 2026|{integrated_val_base['annualized_return_252']:.2%}|{integrated_val_fixed['annualized_return_252']:.2%}|{integrated_val_base['sharpe']:.3f}|{integrated_val_fixed['sharpe']:.3f}|{integrated_val_base['max_drawdown']:.2%}|{integrated_val_fixed['max_drawdown']:.2%}|

2026只有{int(integrated_val_fixed['observations'])}个交易日，表中年化只是统一比较口径；更直观的
完整组合累计收益为基线{integrated_val_base['total_return']:.2%}、固定池
{integrated_val_fixed['total_return']:.2%}。

## 稳健性限制

- 固定池在研发段综合候选中Sharpe排名第{audit['fixed_candidate_development_rank']}，不是事后冠军；
  这比选择第一名更可信，但仍是上一轮观察2026后确定的池。
- 2026日收益配对Bootstrap的年化差95%区间为
  [{bootstrap['annualized_return_delta_ci_lower']:+.2%},
  {bootstrap['annualized_return_delta_ci_upper']:+.2%}]，Sharpe差区间为
  [{bootstrap['sharpe_delta_ci_lower']:+.3f}, {bootstrap['sharpe_delta_ci_upper']:+.3f}]。
- 研发段{audit['attempted_pool_paths']}条路径的Reality Check `p={audit['development_reality_check']['p_value']:.4f}`，
  CSCV-PBO为{audit['development_cscv']['pbo']:.1%}。
- 2025-12-31截面先确认ETF身份，再取份额；不再用部分沪市ETF为空的`fund_type`字段过滤。
- 2026已经在上一轮研究中被看过，所以本次是机械隔离验证，不是独立OOS。真正未观察验证从
  2026-08-26之后开始。

## 决定

固定研究池通过“方向稳定性”检查，但未通过“独立统计显著性”检查。继续保留为影子候选，
不修改正式生产配置。
"""


def run(root: Path, config_path: Path, output: Path) -> dict[str, object]:
    config_path = config_path if config_path.is_absolute() else root / config_path
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    periods = config["periods"]
    pools = _pool_family(config)
    all_assets = tuple(dict.fromkeys(asset for pool in pools.values() for asset in pool))
    market_override_paths = {
        asset: root / path
        for asset, path in config.get("research_market_overrides", {}).items()
    }
    market_overrides = {
        asset: pd.read_parquet(path)
        for asset, path in market_override_paths.items()
    }

    standalone_harness = load_standalone_harness(
        all_assets,
        start=date.fromisoformat(str(periods["standalone_start"])),
        end=date.fromisoformat(str(periods["validation_end"])),
        market_overrides=market_overrides,
    )
    integrated_harness = load_harness(
        root,
        all_assets,
        end=date.fromisoformat(str(periods["validation_end"])),
    )

    output.mkdir(parents=True, exist_ok=True)
    standalone_cache = output / "_interim_standalone_returns.parquet"
    integrated_cache = output / "_interim_integrated_returns.parquet"
    if standalone_cache.exists() and integrated_cache.exists():
        print("reusing cached candidate paths", flush=True)
        standalone_frame = pd.read_parquet(standalone_cache)
        integrated_frame = pd.read_parquet(integrated_cache)
        if list(standalone_frame.columns) != list(pools):
            raise AssertionError("standalone cache candidate family mismatch")
        if list(integrated_frame.columns) != list(pools):
            raise AssertionError("integrated cache candidate family mismatch")
        standalone_returns = {
            column: standalone_frame[column] for column in standalone_frame
        }
        integrated_returns = {
            column: integrated_frame[column] for column in integrated_frame
        }
    else:
        standalone_returns = {}
        integrated_returns = {}
        for number, (path_id, pool) in enumerate(pools.items(), start=1):
            print(f"[{number}/{len(pools)}] {path_id}", flush=True)
            standalone = run_standalone_universe(standalone_harness, pool)
            integrated = run_universe(integrated_harness, pool)
            standalone_returns[path_id] = standalone.returns
            integrated_returns[path_id] = integrated.returns
        standalone_frame = pd.DataFrame(standalone_returns)
        integrated_frame = pd.DataFrame(integrated_returns)
        standalone_frame.to_parquet(standalone_cache)
        integrated_frame.to_parquet(integrated_cache)

    standalone_runs = {
        path_id: run_standalone_universe(standalone_harness, pools[path_id])
        for path_id in ("baseline", "fixed_candidate")
    }

    formal = run_formal_strategy(
        root, end=date.fromisoformat(str(periods["validation_end"]))
    ).daily["return"].astype(float)
    parity = float(integrated_returns["baseline"].sub(formal).abs().max())
    if parity > 1e-14:
        raise AssertionError(f"integrated baseline parity failed: {parity:.3e}")

    metrics = pd.concat(
        [
            _period_metrics("standalone_defender", standalone_returns, config),
            _period_metrics("integrated_composite", integrated_returns, config),
        ],
        ignore_index=True,
    )
    integrated_dev_rank = _rank_table(
        metrics.loc[metrics["layer"].eq("integrated_composite")], "development"
    )
    integrated_val_rank = _rank_table(
        metrics.loc[metrics["layer"].eq("integrated_composite")], "validation_2026"
    )
    standalone_dev_rank = _rank_table(
        metrics.loc[metrics["layer"].eq("standalone_defender")], "development"
    )
    standalone_val_rank = _rank_table(
        metrics.loc[metrics["layer"].eq("standalone_defender")], "validation_2026"
    )
    development_champion = str(integrated_dev_rank.iloc[0]["path_id"])
    fixed_rank = int(
        integrated_dev_rank.loc[
            integrated_dev_rank["path_id"].eq("fixed_candidate"), "rank"
        ].iloc[0]
    )

    dev_index = integrated_frame.index <= pd.Timestamp(periods["development_end"])
    development = integrated_frame.loc[dev_index]
    dev_baseline = development["baseline"]
    dev_alternatives = development.drop(columns="baseline")
    checks = config["validation"]
    reality = yearly_reality_check(
        dev_alternatives,
        dev_baseline,
        repetitions=int(checks["reality_check_repetitions"]),
        seed=int(checks["random_seed"]),
    )
    cscv_frame, cscv_summary = cscv_pbo(
        dev_alternatives,
        dev_baseline,
        block_count=int(checks["cscv_blocks"]),
    )

    validation = integrated_frame.loc[
        integrated_frame.index >= pd.Timestamp(periods["validation_start"])
    ]
    bootstrap_frame, bootstrap_summary = paired_block_bootstrap(
        validation["fixed_candidate"],
        validation["baseline"],
        block_size=int(checks["bootstrap_block_sessions"]),
        repetitions=int(checks["bootstrap_repetitions"]),
        seed=int(checks["random_seed"]),
    )
    events = difference_events(
        validation["fixed_candidate"], validation["baseline"]
    )

    cost_rows: list[dict[str, object]] = []
    for multiplier in checks["defender_cost_multipliers"]:
        for path_id, pool in (
            ("baseline", tuple(config["baseline_pool"])),
            ("fixed_candidate", tuple(config["fixed_candidate_under_test"])),
        ):
            standalone = run_standalone_universe(
                standalone_harness,
                pool,
                defender_cost_multiplier=float(multiplier),
            ).returns
            integrated = run_universe(
                integrated_harness,
                pool,
                defender_cost_multiplier=float(multiplier),
            ).returns
            for layer, series, start in (
                (
                    "standalone_defender",
                    standalone,
                    periods["standalone_start"],
                ),
                (
                    "integrated_composite",
                    integrated,
                    periods["integrated_start"],
                ),
            ):
                for segment, segment_start, segment_end in (
                    ("development", start, periods["development_end"]),
                    (
                        "validation_2026",
                        periods["validation_start"],
                        periods["validation_end"],
                    ),
                ):
                    sample = series.loc[str(segment_start) : str(segment_end)]
                    cost_rows.append(
                        {
                            "layer": layer,
                            "path_id": path_id,
                            "segment": segment,
                            "defender_cost_multiplier": float(multiplier),
                            **performance(sample),
                        }
                    )
    cost_stress = pd.DataFrame(cost_rows)

    selection_months = []
    for path_id in ("baseline", "fixed_candidate"):
        selection = standalone_runs[path_id].selection.loc[
            str(periods["validation_start"]) : str(periods["validation_end"])
        ]
        for month, sample in selection.groupby(selection.index.to_period("M")):
            selection_months.append(
                {
                    "path_id": path_id,
                    "month": str(month),
                    "selected_asset": str(sample.iloc[0]["selected_asset"]),
                }
            )
    selection_months_frame = pd.DataFrame(selection_months)

    champion_validation_delta = _summary_delta(
        metrics,
        "integrated_composite",
        "validation_2026",
        development_champion,
    )
    fixed_deltas = {
        f"{layer}_{segment}": _summary_delta(
            metrics, layer, segment, "fixed_candidate"
        )
        for layer in ("standalone_defender", "integrated_composite")
        for segment in ("development", "validation_2026")
    }
    directional_pass = all(
        delta["annualized_return_252_delta"] > 0.0
        and delta["sharpe_delta"] > 0.0
        for delta in fixed_deltas.values()
    )
    statistical_pass = bool(
        bootstrap_summary["annualized_return_delta_ci_lower"] > 0.0
        and bootstrap_summary["sharpe_delta_ci_lower"] > 0.0
        and float(reality["p_value"]) < 0.10
    )
    rank_correlation = float(
        integrated_dev_rank.set_index("path_id")["rank"].corr(
            integrated_val_rank.set_index("path_id")["rank"], method="pearson"
        )
    )

    audit = {
        "research_id": config["research_id"],
        "status": "passed_research_only",
        "formal_baseline_parity_max_abs_error": parity,
        "attempted_pool_paths": int(len(pools)),
        "unique_integrated_paths": int(integrated_frame.T.drop_duplicates().shape[0]),
        "unique_standalone_paths": int(standalone_frame.T.drop_duplicates().shape[0]),
        "fixed_candidate_development_rank": fixed_rank,
        "development_champion": {
            "path_id": development_champion,
            "assets": list(pools[development_champion]),
            "validation_total_return_delta": champion_validation_delta[
                "total_return_delta"
            ],
            "validation_sharpe_delta": champion_validation_delta["sharpe_delta"],
        },
        "fixed_candidate_deltas": fixed_deltas,
        "fixed_candidate_directional_stability_pass": directional_pass,
        "fixed_candidate_statistical_significance_pass": statistical_pass,
        "development_validation_rank_correlation": rank_correlation,
        "validation_bootstrap": bootstrap_summary,
        "development_reality_check": reality,
        "development_cscv": cscv_summary,
        "production_changed": False,
        "independence_limit": checks["independence_limit"],
    }

    pd.DataFrame(
        [
            {"path_id": path_id, "asset_count": len(pool), "assets": "|".join(pool)}
            for path_id, pool in pools.items()
        ]
    ).to_csv(output / "path_metadata.csv", index=False)
    standalone_frame.to_parquet(output / "standalone_returns.parquet")
    integrated_frame.to_parquet(output / "integrated_returns.parquet")
    metrics.to_csv(output / "segment_metrics.csv", index=False)
    integrated_dev_rank.to_csv(output / "integrated_development_ranking.csv", index=False)
    integrated_val_rank.to_csv(output / "integrated_validation_ranking.csv", index=False)
    standalone_dev_rank.to_csv(output / "standalone_development_ranking.csv", index=False)
    standalone_val_rank.to_csv(output / "standalone_validation_ranking.csv", index=False)
    bootstrap_frame.to_csv(output / "validation_paired_bootstrap.csv", index=False)
    cscv_frame.to_csv(output / "development_cscv.csv", index=False)
    events.to_csv(output / "fixed_candidate_validation_events.csv", index=False)
    cost_stress.to_csv(output / "cost_stress.csv", index=False)
    selection_months_frame.to_csv(output / "validation_monthly_selections.csv", index=False)
    (output / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "REPORT.md").write_text(
        _report(config, audit, metrics), encoding="utf-8"
    )
    generate_standard_report(
        validation["fixed_candidate"],
        validation["baseline"],
        "formal_original_defender_pool_2026",
        output / "fixed_candidate_vs_formal_2026.html",
        config,
    )

    manifest_paths = [
        config_path,
        root / "research/momentum_defender_dividend_universe.py",
        root / "research/run_momentum_defender_dividend_universe_2007_validation.py",
        *market_override_paths.values(),
        *[root / "data/db" / f"{asset}.parquet" for asset in all_assets],
    ]
    (output / "source_manifest.json").write_text(
        json.dumps(
            [
                {"path": str(path.relative_to(root)), "sha256": _sha256(path)}
                for path in manifest_paths
            ],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
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
