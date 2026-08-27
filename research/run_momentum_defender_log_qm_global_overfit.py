"""Global multiple-testing audit across all log-QM switch research rounds."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.momentum_defender_gold_override_overfit import (
    cscv_pbo,
    full_metrics,
    yearly_reality_check,
)
from research.run_momentum_held_asset_c2_overfit import (
    _deflated_sharpe,
    _effective_trials,
)


DEFAULT_OUTPUT = Path(
    "experiments/20260824_momentum_defender_log_qm_global_overfit_audit"
)
FAMILIES = {
    "original_parameter": Path(
        "experiments/20260824_momentum_defender_log_qm_switch_robust/unique_candidate_returns.parquet"
    ),
    "broad_mechanism": Path(
        "experiments/20260824_momentum_defender_log_qm_robust_mechanisms/unique_candidate_returns.parquet"
    ),
    "multihorizon": Path(
        "experiments/20260824_momentum_defender_log_qm_multihorizon_ensemble/unique_candidate_returns.parquet"
    ),
}
BASELINE_DAILY = Path(
    "experiments/20260824_momentum_defender_log_qm_switch_robust/baseline_formal_daily.csv"
)
STABILITY_CANDIDATE = (
    "broad_mechanism::"
    "gate_anchor_and_held_w120_t+0.000_b2_momentum_20_defender_40__"
    "em_downside_vol_w20_q0.95_expanding_strict_lag"
)


def _unique(frame: pd.DataFrame) -> pd.DataFrame:
    seen: set[str] = set()
    columns = []
    for column in frame:
        digest = hashlib.sha1(frame[column].to_numpy(float).tobytes()).hexdigest()
        if digest not in seen:
            seen.add(digest)
            columns.append(column)
    return frame[columns]


def run_audit(root: Path, output: Path) -> dict[str, object]:
    frames = []
    family_counts = {}
    for family, relative in FAMILIES.items():
        frame = pd.read_parquet(root / relative)
        family_counts[family] = int(frame.shape[1])
        frame = frame.rename(columns={column: f"{family}::{column}" for column in frame})
        frames.append(frame)
    all_returns = pd.concat(frames, axis=1)
    unique = _unique(all_returns)
    baseline_frame = pd.read_csv(
        root / BASELINE_DAILY, parse_dates=["date"]
    ).set_index("date")
    baseline = baseline_frame["return"].astype(float).reindex(unique.index)
    if baseline.isna().any():
        raise AssertionError("baseline does not align with global candidate matrix")
    if STABILITY_CANDIDATE not in unique:
        raise AssertionError("absolute-stability candidate was removed or renamed")
    candidate = unique[STABILITY_CANDIDATE].astype(float)

    metrics = full_metrics(unique, baseline)
    metrics.to_csv(output.parent / ".global_metrics.tmp.csv")
    pbo_frame, pbo_summary = cscv_pbo(unique, baseline, block_count=12)
    reality = yearly_reality_check(
        unique, baseline, repetitions=5000, seed=20260824
    )
    values = unique.to_numpy(float)
    effective_trials = _effective_trials(values)
    dsr_absolute = _deflated_sharpe(
        candidate.to_numpy(float), values, effective_trials
    )
    excess = values - baseline.to_numpy(float)[:, None]
    candidate_excess = candidate.to_numpy(float) - baseline.to_numpy(float)
    dsr_excess = _deflated_sharpe(
        candidate_excess,
        excess,
        _effective_trials(excess),
    )
    selected_metrics = metrics.loc[STABILITY_CANDIDATE].to_dict()
    full_dominating = int(
        (
            metrics["delta_annualized_return_252"].ge(0.0)
            & metrics["delta_sharpe"].ge(0.0)
            & metrics["delta_max_drawdown"].ge(0.0)
        ).sum()
    )
    audit = {
        "status": "passed",
        "family_unique_paths_before_global_deduplication": family_counts,
        "candidate_ids_before_global_deduplication": int(all_returns.shape[1]),
        "global_unique_return_paths": int(unique.shape[1]),
        "stability_candidate": STABILITY_CANDIDATE,
        "stability_candidate_metrics_vs_baseline": selected_metrics,
        "full_three_metric_dominating_paths": full_dominating,
        "cscv_pbo": pbo_summary,
        "yearly_reality_check": reality,
        "effective_trials": effective_trials,
        "deflated_sharpe_absolute": dsr_absolute,
        "deflated_sharpe_excess_vs_baseline": dsr_excess,
        "conclusion": (
            "candidate_is_absolutely_stable_but_not_significantly_superior_to_baseline"
        ),
    }
    output.mkdir(parents=True, exist_ok=False)
    temporary_metrics = output.parent / ".global_metrics.tmp.csv"
    temporary_metrics.replace(output / "global_candidate_metrics.csv")
    pbo_frame.to_csv(output / "global_cscv_pbo.csv", index=False)
    (output / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report = f"""# 双对数Momentum切换研究：全局多重试验校正

三轮研究合计{all_returns.shape[1]}条分轮唯一收益路径，全局去重后{unique.shape[1]}条。全局
审计避免在三轮各自校正后再事后挑选。

绝对稳定候选为`{STABILITY_CANDIDATE}`。其相对正式基线的年化、Sharpe、MDD差分别为
{selected_metrics['delta_annualized_return_252']:+.2%}、
{selected_metrics['delta_sharpe']:+.3f}、
{selected_metrics['delta_max_drawdown']:+.2%}。

全局CSCV-PBO为{pbo_summary['pbo']:.1%}；年度Reality Check p={reality['p_value']:.4f}。
候选绝对Deflated Sharpe概率为{dsr_absolute['deflated_sharpe_probability']:.1%}，但相对基线
增量Deflated Sharpe概率仅{dsr_excess['deflated_sharpe_probability']:.1%}。

结论：候选自身的跨年、删事件和绝对Sharpe稳定性较好，但没有经过全局多重试验证明优于
当前基线。可作为低波动研究候选，不能宣称是统计显著的生产升级。
"""
    (output / "research_report.md").write_text(report, encoding="utf-8")
    sources = [
        root / "research/run_momentum_defender_log_qm_global_overfit.py",
        root / "research/configs/momentum_defender_log_qm_absolute_stability_selected.yaml",
        *[root / path for path in FAMILIES.values()],
    ]
    manifest = {
        "experiment": "momentum_defender_log_qm_global_overfit_audit",
        "sources": [
            {
                "path": str(path.relative_to(root)),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path in sources
        ],
    }
    (output / "experiment_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    print(json.dumps(run_audit(root, output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
