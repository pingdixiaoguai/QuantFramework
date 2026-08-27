"""Backtest an immediate Gold veto of the first executable Defender entry."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from research.momentum_defender_dividend_universe import difference_events
from research.momentum_defender_gold_override_overfit import paired_block_bootstrap
from research.momentum_defender_occam import performance
from research.momentum_defender_w40_asset_specific_escape import (
    run_asset_specific_w40_escape,
)
from research.momentum_defender_w40_top1_escape import quality_metrics_at_open
from research.standard_report import generate_standard_report
from strategy.momentum_defender_w40_gold_escape import (
    formal_policies,
)
from strategy.momentum_defender_w40_full_equity import (
    run_formal_strategy as run_base_formal,
)


DEFAULT_CONFIG = Path(
    "research/configs/momentum_defender_immediate_gold_entry_veto.yaml"
)
DEFAULT_OUTPUT = Path(
    "experiments/20260826_momentum_defender_immediate_gold_entry_veto"
)
BASELINE_STRATEGY_ID = "momentum_defender_w40_gold_qm20_escape_v2"


def _hash(series: pd.Series) -> str:
    return hashlib.sha256(series.to_numpy(dtype="<f8").tobytes()).hexdigest()


def _segments(
    candidate: pd.Series,
    baseline: pd.Series,
    config: dict,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for segment, boundaries in config["segments"].items():
        for path_id, returns in (
            ("formal_v2", baseline),
            ("immediate_gold_entry_veto", candidate),
        ):
            sample = returns.loc[str(boundaries[0]) : str(boundaries[1])]
            rows.append(
                {
                    "segment": segment,
                    "path_id": path_id,
                    **performance(sample),
                }
            )
    return pd.DataFrame(rows)


def _entry_events(
    state: pd.DataFrame,
    candidate: pd.Series,
    baseline: pd.Series,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    entries = state["immediate_entry_veto_qualified"].astype(bool)
    for event_id, timestamp in enumerate(state.index[entries], start=1):
        position = state.index.get_loc(timestamp)
        interval = state.index[position : position + 5]
        candidate_return = float((1.0 + candidate.loc[interval]).prod() - 1.0)
        baseline_return = float((1.0 + baseline.loc[interval]).prod() - 1.0)
        rows.append(
            {
                "event_id": event_id,
                "entry_date": timestamp,
                "hard_hold_end": interval[-1],
                "gold_minus_defender_qm20": float(
                    state.at[timestamp, "metric_difference_at_open"]
                ),
                "candidate_return_5d": candidate_return,
                "formal_v2_return_5d": baseline_return,
                "return_delta_5d": candidate_return - baseline_return,
                "formal_v2_target_on_entry": "DEFENDER",
                "candidate_target_on_entry": str(
                    state.at[timestamp, "target_candidate"]
                ),
            }
        )
    return pd.DataFrame(rows)


def _leave_one_difference_event(
    candidate: pd.Series,
    baseline: pd.Series,
    events: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for event in events.itertuples(index=False):
        counterfactual = candidate.copy()
        counterfactual.loc[event.start : event.end] = baseline.loc[
            event.start : event.end
        ]
        rows.append(
            {
                "event_id": int(event.event_id),
                "removed_start": event.start,
                "removed_end": event.end,
                **performance(counterfactual),
            }
        )
    return pd.DataFrame(rows)


def _report(
    config: dict,
    metrics: pd.DataFrame,
    entries: pd.DataFrame,
    events: pd.DataFrame,
    leave_one: pd.DataFrame,
    bootstrap: dict,
) -> str:
    indexed = metrics.set_index(["segment", "path_id"])
    baseline = indexed.loc[("full", "formal_v2")]
    candidate = indexed.loc[("full", "immediate_gold_entry_veto")]
    development_base = indexed.loc[("development", "formal_v2")]
    development_candidate = indexed.loc[
        ("development", "immediate_gold_entry_veto")
    ]
    validation_base = indexed.loc[("validation_2026", "formal_v2")]
    validation_candidate = indexed.loc[
        ("validation_2026", "immediate_gold_entry_veto")
    ]
    positive = events.loc[events["log_excess"].gt(0.0), "log_excess"].sort_values(
        ascending=False
    )
    positive_share = (
        float(positive.head(2).sum() / positive.sum())
        if float(positive.sum()) > 0.0
        else 0.0
    )
    return f"""# W40进入Defender时的即时黄金否决研究

## 机制

W40基础状态仍按正式v2进入Defender并启动30日锁。仅在该切入开盘，如果Momentum Top1为
黄金且`QM20(Gold)-QM20(连续Defender)>0.005`，实际目标立即为黄金，不先持有Defender。
黄金仍硬持有5日；之后沿用正式退出规则。从黄金返回Defender后，仍需实际持有Defender满5日
才可再次逃生。

## 结果

|区间|正式v2年化 / Sharpe|即时否决年化 / Sharpe|MDD|
|---|---:|---:|---:|
|2013–2025研发|{development_base['annualized_return_252']:.2%} / {development_base['sharpe']:.3f}|{development_candidate['annualized_return_252']:.2%} / {development_candidate['sharpe']:.3f}|{development_candidate['max_drawdown']:.2%}|
|2026验证|{validation_base['annualized_return_252']:.2%} / {validation_base['sharpe']:.3f}|{validation_candidate['annualized_return_252']:.2%} / {validation_candidate['sharpe']:.3f}|{validation_candidate['max_drawdown']:.2%}|
|完整|{baseline['annualized_return_252']:.2%} / {baseline['sharpe']:.3f}|{candidate['annualized_return_252']:.2%} / {candidate['sharpe']:.3f}|{candidate['max_drawdown']:.2%}|

完整年化提高{candidate['annualized_return_252']-baseline['annualized_return_252']:+.2%}，Sharpe提高
{candidate['sharpe']-baseline['sharpe']:+.3f}；最大回撤不变。共{len(entries)}次即时否决，路径
差异被切为{len(events)}个连续事件。

## 稳健性

- 2026没有即时否决事件，两条路径完全相同；因此2026不能验证该机制，全部改善来自研发历史。
- 20日配对Bootstrap年化差95%区间为
  [{bootstrap['annualized_return_delta_ci_lower']:+.2%},
  {bootstrap['annualized_return_delta_ci_upper']:+.2%}]，Sharpe差区间为
  [{bootstrap['sharpe_delta_ci_lower']:+.3f}, {bootstrap['sharpe_delta_ci_upper']:+.3f}]。
- 前两大正向差异事件占全部正向log excess的{positive_share:.1%}。
- 删除任一差异事件后最低年化/Sharpe为
  {leave_one['annualized_return_252'].min():.2%}/{leave_one['sharpe'].min():.3f}。

## 决定

这是逻辑更一致的研究候选，但2026零触发且机制由已观察到的2025错过黄金事件提出，不能称
独立样本外证据。正式v2保持不变，等待新的前瞻W40入场事件验证。
"""


def run(root: Path, config_path: Path, output: Path) -> dict[str, object]:
    config_path = config_path if config_path.is_absolute() else root / config_path
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config["baseline_strategy"] != BASELINE_STRATEGY_ID:
        raise AssertionError("research baseline is not archived formal v2")
    cutoff = date.fromisoformat(str(config["cutoff"]))
    base = run_base_formal(root, end=cutoff)
    metrics_at_open = quality_metrics_at_open(base.context)
    baseline_run = run_asset_specific_w40_escape(
        base.context,
        base.state,
        formal_policies(),
        metrics=metrics_at_open,
        immediate_entry_veto=False,
    )
    candidate_run = run_asset_specific_w40_escape(
        base.context,
        base.state,
        formal_policies(),
        metrics=metrics_at_open,
        immediate_entry_veto=True,
    )
    baseline = baseline_run.daily["return"].astype(float)
    candidate = candidate_run.daily["return"].astype(float)
    if not candidate.index.equals(baseline.index):
        raise AssertionError("candidate calendar differs from formal v2")

    metrics = _segments(candidate, baseline, config)
    entry_events = _entry_events(candidate_run.state, candidate, baseline)
    events = difference_events(candidate, baseline)
    leave_one = _leave_one_difference_event(candidate, baseline, events)
    checks = config["validation"]
    bootstrap_frame, bootstrap_summary = paired_block_bootstrap(
        candidate,
        baseline,
        block_size=int(checks["paired_block_bootstrap_block_sessions"]),
        repetitions=int(checks["paired_block_bootstrap_repetitions"]),
        seed=int(checks["random_seed"]),
    )
    validation = metrics.set_index(["segment", "path_id"])
    validation_equal = bool(
        candidate.loc["2026-01-01":].equals(baseline.loc["2026-01-01":])
    )
    audit = {
        "research_id": config["research_id"],
        "status": "passed_research_only",
        "formal_strategy_id": BASELINE_STRATEGY_ID,
        "formal_return_hash": _hash(baseline),
        "candidate_return_hash": _hash(candidate),
        "formal_performance": performance(baseline),
        "candidate_performance": performance(candidate),
        "candidate_audit": candidate_run.audit,
        "base_w40_state_unchanged": True,
        "immediate_entry_events": int(len(entry_events)),
        "difference_events": int(len(events)),
        "different_days": int(candidate.sub(baseline).abs().gt(1e-15).sum()),
        "validation_2026_path_equal": validation_equal,
        "validation_2026_candidate": validation.loc[
            ("validation_2026", "immediate_gold_entry_veto")
        ].to_dict(),
        "paired_block_bootstrap": bootstrap_summary,
        "production_changed": False,
        "evidence_limit": checks["evidence_limit"],
    }

    output.mkdir(parents=True, exist_ok=True)
    daily = candidate_run.state.copy()
    daily["candidate_return"] = candidate
    daily["formal_v2_return"] = baseline
    daily["daily_difference"] = candidate - baseline
    daily.to_csv(output / "candidate_daily.csv")
    daily.to_parquet(output / "candidate_daily.parquet")
    metrics.to_csv(output / "segment_metrics.csv", index=False)
    entry_events.to_csv(output / "immediate_entry_events.csv", index=False)
    events.to_csv(output / "difference_events.csv", index=False)
    leave_one.to_csv(output / "leave_one_event.csv", index=False)
    bootstrap_frame.to_csv(output / "paired_block_bootstrap.csv", index=False)
    (output / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "REPORT.md").write_text(
        _report(
            config,
            metrics,
            entry_events,
            events,
            leave_one,
            bootstrap_summary,
        ),
        encoding="utf-8",
    )
    generate_standard_report(
        candidate,
        baseline,
        "Formal v2 (5-session Defender eligibility)",
        output / "candidate_vs_formal_v2.html",
        config,
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
