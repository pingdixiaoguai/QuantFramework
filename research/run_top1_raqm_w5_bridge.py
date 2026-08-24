"""Research the emergency-safe RAQM-W5 bridge to any Momentum Top-1 ETF."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from research.gold_min5_risk_adjusted_momentum_w5 import (
    GoldRAQMW5Params,
    run_gold_raqm_w5,
)
from research.momentum_defender_gold_override import build_gold_override_context
from research.momentum_defender_gold_override_overfit import (
    cscv_pbo,
    expanding_walk_forward,
    full_metrics,
    paired_block_bootstrap,
    yearly_reality_check,
)
from research.momentum_defender_occam import HELD_RETURN, performance
from research.standard_report import generate_standard_report
from research.top1_raqm_w5_bridge import (
    FORMAL_GOLD_ENTRY_DIFFERENCE,
    FORMAL_GOLD_EXIT_DIFFERENCE,
    Top1RAQMW5BridgeParams,
    collect_grid,
    registered_raqm_w5_at_open,
    run_top1_raqm_w5_bridge,
)


DEFAULT_CONFIG = Path("research/configs/top1_raqm_w5_bridge.yaml")
DEFAULT_OUTPUT = Path("experiments/20260823_top1_raqm_w5_bridge")


def _params(values: pd.Series | dict[str, object]) -> Top1RAQMW5BridgeParams:
    difference = values.get("minimum_difference")
    if pd.isna(difference):
        difference = None
    return Top1RAQMW5BridgeParams(
        entry_minimum=float(values["entry_minimum"]),
        confirmation_days=int(values["confirmation_days"]),
        minimum_difference=(None if difference is None else float(difference)),
    )


def _json_record(values: pd.Series) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in values.items():
        if pd.isna(value):
            result[str(key)] = None
        elif isinstance(value, np.generic):
            result[str(key)] = value.item()
        else:
            result[str(key)] = value
    return result


def _annual(strategies: dict[str, pd.Series]) -> pd.DataFrame:
    rows = []
    for name, returns in strategies.items():
        for year, sample in returns.groupby(returns.index.year):
            rows.append(
                {
                    "strategy": name,
                    "year": int(year),
                    "observations": int(len(sample)),
                    "total_return": float((1.0 + sample).prod() - 1.0),
                }
            )
    return pd.DataFrame(rows)


def _unique_paths(returns: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    representatives: dict[str, str] = {}
    for candidate in returns.columns:
        digest = hashlib.sha256(
            returns[candidate].to_numpy(dtype="<f8").tobytes()
        ).hexdigest()
        representative = representatives.setdefault(digest, str(candidate))
        rows.append(
            {
                "candidate_id": str(candidate),
                "return_path_sha256": digest,
                "representative_candidate_id": representative,
                "is_representative": representative == str(candidate),
            }
        )
    mapping = pd.DataFrame(rows).set_index("candidate_id")
    unique = returns.loc[:, mapping["is_representative"].to_numpy(bool)]
    return mapping, unique


def _bridge_episodes(
    state: pd.DataFrame,
    candidate: pd.Series,
    baseline: pd.Series,
) -> pd.DataFrame:
    active = state["top1_bridge_active"].astype(bool)
    groups = active.ne(active.shift()).cumsum()
    rows = []
    for episode, (_, sample) in enumerate(
        state.loc[active].groupby(groups.loc[active]), start=1
    ):
        start_position = state.index.get_loc(sample.index.min())
        end_position = min(
            state.index.get_loc(sample.index.max()) + 1,
            len(state.index) - 1,
        )
        interval = state.index[start_position : end_position + 1]
        candidate_return = float((1.0 + candidate.loc[interval]).prod() - 1.0)
        baseline_return = float((1.0 + baseline.loc[interval]).prod() - 1.0)
        relative = (1.0 + candidate_return) / (1.0 + baseline_return) - 1.0
        rows.append(
            {
                "episode": episode,
                "start": interval.min().date().isoformat(),
                "end_including_exit": interval.max().date().isoformat(),
                "observations": int(len(interval)),
                "entry_asset": str(sample.iloc[0]["target_candidate"]),
                "candidate_return": candidate_return,
                "formal_baseline_return": baseline_return,
                "relative_return": relative,
                "entry_top1_metric": float(
                    sample.iloc[0]["top1_metric_at_open"]
                ),
                "entry_metric_difference": float(
                    sample.iloc[0]["metric_difference_at_open"]
                ),
            }
        )
    return pd.DataFrame(rows)


def _leave_one_event(
    episodes: pd.DataFrame,
    candidate: pd.Series,
    baseline: pd.Series,
) -> pd.DataFrame:
    rows = []
    for event in episodes.itertuples(index=False):
        counterfactual = candidate.copy()
        interval = counterfactual.loc[
            pd.Timestamp(event.start) : pd.Timestamp(event.end_including_exit)
        ].index
        counterfactual.loc[interval] = baseline.loc[interval]
        rows.append(
            {
                "removed_episode": int(event.episode),
                **performance(counterfactual),
            }
        )
    return pd.DataFrame(rows)


def _cost_stress(
    candidate: pd.Series,
    candidate_cost: pd.Series,
    baseline: pd.Series,
    baseline_cost: pd.Series,
    multipliers: list[int],
) -> pd.DataFrame:
    rows = []
    for multiplier in multipliers:
        extra = float(multiplier - 1)
        candidate_stressed = (
            (1.0 + candidate) * (1.0 - candidate_cost).pow(extra) - 1.0
        )
        baseline_stressed = (
            (1.0 + baseline) * (1.0 - baseline_cost).pow(extra) - 1.0
        )
        candidate_metrics = performance(candidate_stressed)
        baseline_metrics = performance(baseline_stressed)
        rows.append(
            {
                "cost_multiplier": int(multiplier),
                "candidate_annualized_return_252": candidate_metrics[
                    "annualized_return_252"
                ],
                "formal_annualized_return_252": baseline_metrics[
                    "annualized_return_252"
                ],
                "annualized_return_delta": candidate_metrics[
                    "annualized_return_252"
                ]
                - baseline_metrics["annualized_return_252"],
                "candidate_sharpe": candidate_metrics["sharpe"],
                "formal_sharpe": baseline_metrics["sharpe"],
                "sharpe_delta": candidate_metrics["sharpe"]
                - baseline_metrics["sharpe"],
                "candidate_max_drawdown": candidate_metrics["max_drawdown"],
                "formal_max_drawdown": baseline_metrics["max_drawdown"],
            }
        )
    return pd.DataFrame(rows)


def run_experiment(root: Path, config_path: Path, output: Path) -> dict[str, object]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    context = build_gold_override_context(root)
    formal_run = run_gold_raqm_w5(
        context,
        GoldRAQMW5Params(
            FORMAL_GOLD_ENTRY_DIFFERENCE,
            FORMAL_GOLD_EXIT_DIFFERENCE,
        ),
    )
    formal = formal_run.daily["return"].astype(float)
    current_c2 = context.integrated.result.simulated["return"].astype(float)
    original_momentum = context.integrated.result.inputs.momentum[
        HELD_RETURN
    ].astype(float)

    grid = config["grid"]
    metadata, returns = collect_grid(
        context,
        grid["entry_minimums"],
        grid["confirmation_days"],
        grid["minimum_differences"],
    )
    metrics = full_metrics(returns, formal).join(metadata)
    periods = {
        label: (pd.Timestamp(values[0]), pd.Timestamp(values[1]))
        for label, values in config["periods"].items()
    }
    for label, (start, end) in periods.items():
        period_metrics = full_metrics(returns.loc[start:end], formal.loc[start:end])
        for field in (
            "annualized_return_252",
            "sharpe",
            "max_drawdown",
            "delta_annualized_return_252",
            "delta_sharpe",
            "delta_max_drawdown",
        ):
            metrics[f"{label}_{field}"] = period_metrics[field]
    metrics["worst_split_sharpe"] = metrics[
        ["development_sharpe", "validation_sharpe", "recent_sharpe"]
    ].min(axis=1)

    direct_params = _params(config["pre_registered_direct_extension"])
    direct_id = direct_params.candidate_id()
    if direct_id not in metrics.index:
        raise AssertionError("direct extension is missing from the grid")

    selection = config["selection"]
    eligible = metrics.loc[
        metrics["bridge_entries"].ge(int(selection["minimum_repeated_entries"]))
        & metrics["bridge_days"].ge(int(selection["minimum_bridge_days"]))
    ].copy()
    core = selection["core_metrics"]
    improved = eligible.loc[
        eligible["delta_annualized_return_252"].gt(
            float(core["annualized_return_delta_minimum"])
        )
        & eligible["delta_sharpe"].gt(float(core["sharpe_delta_minimum"]))
        & eligible["delta_max_drawdown"].ge(
            float(core["max_drawdown_delta_minimum"])
        )
    ].copy()
    selection_pool = improved if not improved.empty else eligible
    if selection_pool.empty:
        selection_pool = metrics.loc[metrics["bridge_entries"].gt(0)].copy()
    observed = selection_pool.sort_values(
        [
            "annualized_return_252",
            "sharpe",
            "entry_minimum",
            "confirmation_days",
        ],
        ascending=[False, False, False, False],
    ).iloc[0]
    observed_id = str(observed.name)
    observed_run = run_top1_raqm_w5_bridge(
        context,
        _params(observed),
        metrics=registered_raqm_w5_at_open(context.curves),
        formal_run=formal_run,
    )
    direct_run = run_top1_raqm_w5_bridge(
        context,
        direct_params,
        metrics=observed_run.metrics_at_open,
        formal_run=formal_run,
    )

    path_mapping, unique_returns = _unique_paths(returns)
    cscv, pbo = cscv_pbo(
        unique_returns,
        formal,
        block_count=int(config["robustness"]["cscv_blocks"]),
    )
    walk = expanding_walk_forward(unique_returns, formal)
    bootstrap, bootstrap_summary = paired_block_bootstrap(
        returns[observed_id],
        formal,
        repetitions=int(config["robustness"]["bootstrap_repetitions"]),
    )
    reality = yearly_reality_check(
        unique_returns,
        formal,
        repetitions=int(config["robustness"]["reality_check_repetitions"]),
    )
    episodes = _bridge_episodes(observed_run.state, observed_run.daily["return"], formal)
    leave_one = _leave_one_event(episodes, observed_run.daily["return"], formal)
    cost_stress = _cost_stress(
        observed_run.daily["return"],
        observed_run.daily["cost_rate_at_open"],
        formal,
        formal_run.daily["cost_rate_at_open"],
        [int(value) for value in config["robustness"]["cost_multipliers"]],
    )

    positive_event_log = np.log1p(
        episodes.loc[episodes["relative_return"].gt(0), "relative_return"]
    )
    total_positive_event_log = float(positive_event_log.sum())
    top_event_share = (
        float(positive_event_log.max() / total_positive_event_log)
        if total_positive_event_log > 0.0
        else np.nan
    )
    improvement_counts = {
        "annualized_return": int(
            metrics["delta_annualized_return_252"].gt(0).sum()
        ),
        "sharpe": int(metrics["delta_sharpe"].gt(0).sum()),
        "max_drawdown_not_worse": int(
            metrics["delta_max_drawdown"].ge(-1e-12).sum()
        ),
        "all_three": int(
            (
                metrics["delta_annualized_return_252"].gt(0)
                & metrics["delta_sharpe"].gt(0)
                & metrics["delta_max_drawdown"].ge(-1e-12)
            ).sum()
        ),
    }
    overfit_flags = {
        "fewer_than_five_bridge_events": int(len(episodes)) < 5,
        "single_event_positive_share_above_half": bool(
            pd.notna(top_event_share) and top_event_share > 0.5
        ),
        "pbo_above_half": float(pbo["pbo"]) >= 0.5,
        "reality_check_not_significant_10pct": float(reality["p_value"]) >= 0.10,
        "bootstrap_return_probability_below_90pct": float(
            bootstrap_summary["annualized_return_delta_positive_probability"]
        )
        < 0.90,
        "walk_forward_return_win_below_half": float(
            walk["test_return_delta"].gt(0).mean()
        )
        < 0.5,
    }
    overfit_assessment = (
        "high"
        if sum(overfit_flags.values()) >= 3
        else "moderate" if any(overfit_flags.values()) else "low"
    )

    output.mkdir(parents=True, exist_ok=True)
    metrics.sort_values(
        ["annualized_return_252", "sharpe"], ascending=False
    ).to_csv(output / "candidate_metrics.csv")
    path_mapping.to_csv(output / "return_path_mapping.csv")
    observed_run.state.join(
        observed_run.daily, rsuffix="_execution"
    ).to_csv(output / "daily_observed_candidate.csv")
    direct_run.state.join(direct_run.daily, rsuffix="_execution").to_csv(
        output / "daily_direct_extension.csv"
    )
    episodes.to_csv(output / "bridge_episodes.csv", index=False)
    leave_one.to_csv(output / "leave_one_bridge_event.csv", index=False)
    cost_stress.to_csv(output / "cost_stress.csv", index=False)
    cscv.to_csv(output / "cscv_pbo.csv", index=False)
    walk.to_csv(output / "expanding_walk_forward.csv", index=False)
    bootstrap.to_csv(output / "paired_block_bootstrap.csv", index=False)
    _annual(
        {
            "observed_bridge": observed_run.daily["return"],
            "direct_extension": direct_run.daily["return"],
            "formal_gold_raqm_w5": formal,
            "current_c2": current_c2,
            "original_momentum": original_momentum,
        }
    ).to_csv(output / "calendar_year_returns.csv", index=False)
    pd.DataFrame(
        [
            {"strategy": name, **performance(values)}
            for name, values in {
                "observed_bridge": observed_run.daily["return"],
                "direct_extension": direct_run.daily["return"],
                "formal_gold_raqm_w5": formal,
                "current_c2": current_c2,
                "original_momentum": original_momentum,
            }.items()
        ]
    ).to_csv(output / "strategy_metrics.csv", index=False)
    (output / "strategy_config.yaml").write_text(
        config_path.read_text(encoding="utf-8"), encoding="utf-8"
    )

    report_config = {
        "experiment_id": config["experiment"]["id"],
        "candidate_id": observed_id,
        "params": observed_run.params.__dict__,
        "formal_baseline": "momentum_defender_c2_gold_raqm_w5_v1",
        "evidence_status": config["experiment"]["evidence_status"],
        "production_replacement": False,
    }
    generate_standard_report(
        observed_run.daily["return"],
        formal,
        "Formal Gold RAQM-W5",
        output / "observed_vs_formal_strategy.html",
        report_config,
    )
    generate_standard_report(
        observed_run.daily["return"],
        original_momentum,
        "Original Momentum",
        output / "observed_vs_original_momentum.html",
        report_config,
    )
    generate_standard_report(
        observed_run.daily["return"],
        current_c2,
        "Current Integrated C2",
        output / "observed_vs_current_c2.html",
        report_config,
    )

    observed_metrics = metrics.loc[observed_id]
    direct_metrics = metrics.loc[direct_id]
    summary = {
        "experiment_id": config["experiment"]["id"],
        "candidate_count": int(len(metrics)),
        "unique_return_path_count": int(len(unique_returns.columns)),
        "eligible_repeated_candidate_count": int(len(eligible)),
        "all_three_improved_repeated_candidate_count": int(len(improved)),
        "formal_baseline": performance(formal),
        "direct_extension_id": direct_id,
        "direct_extension": _json_record(direct_metrics),
        "direct_extension_audit": direct_run.audit,
        "observed_candidate_id": observed_id,
        "observed_candidate": _json_record(observed_metrics),
        "observed_candidate_audit": observed_run.audit,
        "improvement_counts": improvement_counts,
        "bridge_event_count": int(len(episodes)),
        "top_positive_event_share": top_event_share,
        "leave_one_event_minimum_annualized_return": (
            float(leave_one["annualized_return_252"].min())
            if not leave_one.empty
            else None
        ),
        "leave_one_event_minimum_sharpe": (
            float(leave_one["sharpe"].min()) if not leave_one.empty else None
        ),
        "pbo": pbo,
        "walk_forward_return_win_rate": float(
            walk["test_return_delta"].gt(0).mean()
        ),
        "walk_forward_sharpe_win_rate": float(
            walk["test_sharpe_delta"].gt(0).mean()
        ),
        "bootstrap": bootstrap_summary,
        "reality_check": reality,
        "overfit_flags": overfit_flags,
        "overfit_assessment": overfit_assessment,
        "production_replacement": False,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )

    formal_metrics = performance(formal)
    report = f"""# RAQM-W5 Momentum Top1 极强桥接研究

## 机制

注册因子口径保持不变：`risk_adjusted_quality_momentum(window=5,
vol_floor_annual=0.08)`，上一收盘计算、下一开盘执行。正式Gold RAQM-W5规则优先。
只有正式策略原本要持有Defender、510300慢门已经转为Momentum、且emergency未触发时，
当前Momentum Top1连续满足极强阈值才允许桥接；桥接后跟随原Momentum Top1正常轮动。

## 直接扩展（未调参）

沿用2.20阈值、单日确认：年化{float(direct_metrics['annualized_return_252']):.2%}、
Sharpe {float(direct_metrics['sharpe']):.3f}、MDD {float(direct_metrics['max_drawdown']):.2%}；
相对正式策略年化{float(direct_metrics['delta_annualized_return_252']):+.2%}、
Sharpe {float(direct_metrics['delta_sharpe']):+.3f}。直接扩展没有改善核心指标。

## 回溯网格中观察到的候选

`{observed_id}`：年化{float(observed_metrics['annualized_return_252']):.2%}、
Sharpe {float(observed_metrics['sharpe']):.3f}、MDD {float(observed_metrics['max_drawdown']):.2%}；
相对正式策略（年化{float(formal_metrics['annualized_return_252']):.2%}、
Sharpe {float(formal_metrics['sharpe']):.3f}、MDD {float(formal_metrics['max_drawdown']):.2%}）分别为
{float(observed_metrics['delta_annualized_return_252']):+.2%}、
{float(observed_metrics['delta_sharpe']):+.3f}、
{float(observed_metrics['delta_max_drawdown']):+.2%}。

桥接{int(observed_metrics['bridge_entries'])}次、{int(observed_metrics['bridge_days'])}日。
共测试{len(metrics)}组参数、{len(unique_returns.columns)}条唯一收益路径；同时改善年化、
Sharpe且MDD不恶化的全网格候选有{improvement_counts['all_three']}组。

## 过拟合审计

- 桥接事件仅{len(episodes)}个，最大正向事件占比{top_event_share:.1%}。
- PBO {float(pbo['pbo']):.1%}；年度块多重试验校正p={float(reality['p_value']):.3f}。
- 扩展walk-forward收益/Sharpe胜率分别为
  {summary['walk_forward_return_win_rate']:.1%}/{summary['walk_forward_sharpe_win_rate']:.1%}。
- 分块bootstrap年化差为正概率
  {bootstrap_summary['annualized_return_delta_positive_probability']:.1%}，95%区间
  [{bootstrap_summary['annualized_return_delta_ci_lower']:+.2%},
  {bootstrap_summary['annualized_return_delta_ci_upper']:+.2%}]。
- 综合过拟合风险：**{overfit_assessment.upper()}**。

## 决策

该机制能够在回溯网格中提高三项核心指标，但直接、未调参的2.20单日规则反而退化；
观察到的改善依赖极少数桥接事件，不能据此替换正式策略。保留为研究/前瞻shadow候选，
正式策略继续使用Gold RAQM-W5。
"""
    (output / "report.md").write_text(report, encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    root = args.root.resolve()
    config = args.config if args.config.is_absolute() else root / args.config
    output = args.output if args.output.is_absolute() else root / args.output
    summary = run_experiment(root, config, output)
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
