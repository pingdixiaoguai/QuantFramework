"""Robustness audit for the 2019-start Defender selector convention."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from datetime import date
from itertools import product
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
import yaml

from defender.relative_defender_rotation import DEFENSIVE_ASSET
from defender.w40_reversal_full_equity import (
    FORMAL_DIVIDEND_ASSETS,
    _load_formal_market,
)
from research.audit_current_strategy_occam_robustness import (
    _metric_row,
    _periods,
)
from research.audit_momentum_hold_2019_followup import (
    _calendar_year_comparison,
    _fixed_leave_one_year,
    _rolling_comparison,
    _scaled_cost_context,
)
from research.defender_curve_momentum import DEFENDER_CANDIDATE
from research.momentum_defender_gold_override_overfit import (
    cscv_pbo,
    expanding_walk_forward,
    leave_one_year_selection,
    paired_block_bootstrap,
    yearly_reality_check,
)
from research.momentum_defender_occam import performance
from research.momentum_defender_occam_defender import (
    MonthlySelectionSpec,
    build_portfolio_switch_interface,
    monthly_top1_selection,
    score_at_open,
    selected_asset_targets,
)
from research.momentum_defender_w40_asset_specific_escape import (
    run_asset_specific_w40_escape,
)
from research.momentum_defender_w40_top1_escape import quality_metrics_at_open
from strategy.momentum_defender_w40_gold_escape import (
    formal_policies,
    run_formal_strategy,
)


DEFAULT_CONFIG = Path(
    "research/configs/current_strategy_occam_robustness_audit_2019.yaml"
)
DEFAULT_OUTPUT = Path(
    "experiments/20260826_current_strategy_occam_robustness_audit_2019"
)


def _selection_and_targets(
    market: Mapping[str, pd.DataFrame],
    calendar: pd.DatetimeIndex,
    spec: MonthlySelectionSpec,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    scores = score_at_open(
        market,
        FORMAL_DIVIDEND_ASSETS,
        calendar,
        spec,
    )
    selection = monthly_top1_selection(
        market,
        FORMAL_DIVIDEND_ASSETS,
        calendar,
        scores,
        spec,
    )
    targets = selected_asset_targets(
        selection["selected_asset"].astype(str),
        FORMAL_DIVIDEND_ASSETS,
        selected_weight=1.0,
        residual_asset=DEFENSIVE_ASSET,
    )
    return selection, targets


def _run_selector(
    formal,
    base_context,
    market: Mapping[str, pd.DataFrame],
    spec: MonthlySelectionSpec,
    cost_multiplier: float,
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame, int]:
    selection, targets = _selection_and_targets(
        market, formal.context.calendar, spec
    )
    cost_rates = {
        **{
            asset: 0.0001 * cost_multiplier
            for asset in FORMAL_DIVIDEND_ASSETS
        },
        DEFENSIVE_ASSET: 0.00001 * cost_multiplier,
    }
    defender = build_portfolio_switch_interface(
        market,
        targets,
        cost_rates,
    )
    interfaces = dict(base_context.interfaces)
    interfaces[DEFENDER_CANDIDATE] = defender
    curves = base_context.curves.copy()
    curves[DEFENDER_CANDIDATE] = defender["nav_if_held"].astype(float)
    context = replace(
        base_context,
        interfaces=interfaces,
        curves=curves,
    )
    metrics = quality_metrics_at_open(context)
    run = run_asset_specific_w40_escape(
        context,
        formal.state,
        formal_policies(),
        metrics=metrics,
        immediate_entry_veto=True,
    )
    return (
        run.daily["return"].astype(float),
        selection,
        targets,
        int(run.audit["escape_entries"]),
    )


def _difference_events(
    candidate: pd.Series,
    baseline: pd.Series,
) -> pd.DataFrame:
    active = candidate.sub(baseline).abs().gt(1e-15)
    rows: list[dict[str, object]] = []
    start_position: int | None = None
    for position, is_active in enumerate(active.to_numpy(bool)):
        if is_active and start_position is None:
            start_position = position
        is_last = position == len(active) - 1
        if start_position is not None and (not is_active or is_last):
            end_position = position if is_active and is_last else position - 1
            candidate_sample = candidate.iloc[start_position : end_position + 1]
            baseline_sample = baseline.reindex(candidate_sample.index)
            candidate_total = float((1.0 + candidate_sample).prod() - 1.0)
            baseline_total = float((1.0 + baseline_sample).prod() - 1.0)
            rows.append(
                {
                    "event_id": len(rows) + 1,
                    "start": candidate_sample.index[0],
                    "end": candidate_sample.index[-1],
                    "observations": int(len(candidate_sample)),
                    "candidate_total_return": candidate_total,
                    "baseline_total_return": baseline_total,
                    "return_delta": candidate_total - baseline_total,
                    "log_excess": float(
                        np.log1p(candidate_sample).sum()
                        - np.log1p(baseline_sample).sum()
                    ),
                }
            )
            start_position = None
    return pd.DataFrame(rows)


def _leave_one_event(
    candidate: pd.Series,
    baseline: pd.Series,
    events: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for event in events.itertuples(index=False):
        adjusted = candidate.copy()
        adjusted.loc[pd.Timestamp(event.start) : pd.Timestamp(event.end)] = (
            baseline.loc[pd.Timestamp(event.start) : pd.Timestamp(event.end)]
        )
        measured = performance(adjusted)
        rows.append(
            {
                "event_id": int(event.event_id),
                "deleted_start": event.start,
                "deleted_end": event.end,
                "annualized_return_252": measured["annualized_return_252"],
                "sharpe": measured["sharpe"],
                "max_drawdown": measured["max_drawdown"],
            }
        )
    return pd.DataFrame(rows)


def run_audit(root: Path, config_path: Path, output: Path) -> dict[str, object]:
    applied = config_path if config_path.is_absolute() else root / config_path
    config = yaml.safe_load(applied.read_text(encoding="utf-8"))
    experiment = config["experiment"]
    spec_config = config["defender_selector_followup"]
    checks = config["overfit_checks"]
    start = date.fromisoformat(str(experiment["evaluation_start"]))
    end = date.fromisoformat(str(experiment["evidence_cutoff"]))
    formal = run_formal_strategy(root, start=start, end=end)
    market = _load_formal_market(end)
    periods = _periods(config)

    rows: list[dict[str, object]] = []
    returns: dict[str, pd.Series] = {}
    selections: dict[str, pd.DataFrame] = {}
    selector_specs: dict[str, MonthlySelectionSpec] = {}
    for method, direction, window in product(
        spec_config["score_methods"],
        spec_config["directions"],
        spec_config["windows"],
    ):
        selector = MonthlySelectionSpec(
            int(window), str(method), str(direction)
        )
        candidate_id = f"{method}_{direction}_w{int(window)}"
        candidate_returns, selection, _, escape_entries = _run_selector(
            formal,
            formal.context,
            market,
            selector,
            1.0,
        )
        returns[candidate_id] = candidate_returns
        selections[candidate_id] = selection
        selector_specs[candidate_id] = selector
        row = _metric_row(
            candidate_id,
            "defender_selector_followup",
            candidate_returns,
            periods,
        )
        row.update(
            {
                "score_method": str(method),
                "direction": str(direction),
                "window": int(window),
                "selection_switches": int(
                    selection["selection_changed"].sum()
                ),
                "escape_entries": escape_entries,
            }
        )
        rows.append(row)
    surface = pd.DataFrame(rows)
    baseline_id = str(spec_config["baseline_candidate"])
    selected_id = str(spec_config["selected_point_candidate"])
    baseline = returns[baseline_id]
    selected = returns[selected_id]
    parity = float(
        (baseline - formal.daily["return"].astype(float)).abs().max()
    )
    if parity > 1e-14:
        raise AssertionError(
            f"formal Defender selector parity failed: {parity:.3e}"
        )

    panel = pd.DataFrame(returns, index=formal.context.calendar)
    cscv_frame, cscv = cscv_pbo(
        panel,
        baseline,
        block_count=int(checks["cscv_blocks"]),
    )
    reality = yearly_reality_check(
        panel,
        baseline,
        repetitions=int(checks["yearly_reality_check_repetitions"]),
        seed=int(checks["random_seed"]),
    )
    walk_forward = expanding_walk_forward(panel, baseline)
    leave_year_selection = leave_one_year_selection(panel, baseline)
    bootstrap_frame, bootstrap = paired_block_bootstrap(
        selected,
        baseline,
        block_size=int(checks["paired_block_bootstrap_block"]),
        repetitions=int(checks["paired_block_bootstrap_repetitions"]),
        seed=int(checks["random_seed"]),
    )
    fixed_leave_year = _fixed_leave_one_year(selected, baseline)
    annual = _calendar_year_comparison(selected, baseline)
    rolling = _rolling_comparison(
        selected,
        baseline,
        [int(value) for value in spec_config["rolling_windows"]],
    )
    events = _difference_events(selected, baseline)
    leave_event = _leave_one_event(selected, baseline, events)

    cost_rows: list[dict[str, object]] = []
    for multiplier in spec_config["transaction_cost_multipliers"]:
        base_context = _scaled_cost_context(
            formal, float(multiplier), end
        )
        for candidate_id in (baseline_id, selected_id):
            candidate_returns, selection, _, escape_entries = _run_selector(
                formal,
                base_context,
                market,
                selector_specs[candidate_id],
                float(multiplier),
            )
            measured = performance(candidate_returns)
            cost_rows.append(
                {
                    "cost_multiplier": float(multiplier),
                    "candidate_id": candidate_id,
                    "annualized_return_252": measured[
                        "annualized_return_252"
                    ],
                    "sharpe": measured["sharpe"],
                    "max_drawdown": measured["max_drawdown"],
                    "selection_switches": int(
                        selection["selection_changed"].sum()
                    ),
                    "escape_entries": escape_entries,
                }
            )
    cost_stress = pd.DataFrame(cost_rows)

    baseline_row = surface.set_index("candidate_id").loc[baseline_id]
    selected_row = surface.set_index("candidate_id").loc[selected_id]
    dual = surface.loc[
        surface["annualized_return_252"].gt(
            baseline_row["annualized_return_252"]
        )
        & surface["sharpe"].gt(baseline_row["sharpe"])
    ]
    rolling_summary = {
        str(window): {
            "observations": int(len(group)),
            "annualized_return_win_rate": float(
                group["annualized_return_delta"].gt(0).mean()
            ),
            "sharpe_win_rate": float(group["sharpe_delta"].gt(0).mean()),
            "dual_win_rate": float(
                (
                    group["annualized_return_delta"].gt(0)
                    & group["sharpe_delta"].gt(0)
                ).mean()
            ),
            "shallower_drawdown_rate": float(
                group["max_drawdown_delta"].gt(0).mean()
            ),
        }
        for window, group in rolling.groupby("window")
    }
    positive = events.loc[events["log_excess"].gt(0), "log_excess"].sort_values(
        ascending=False
    )
    top_two_share = float(
        positive.head(2).sum() / positive.sum()
    ) if positive.sum() > 0 else 0.0
    audit: dict[str, object] = {
        "research_id": "current_strategy_occam_robustness_audit_2019_defender_selector_v1",
        "status": "rejected_small_retrospective_quality_score_gain",
        "evidence_status": spec_config["evidence_status"],
        "evaluation_start": start.isoformat(),
        "evidence_cutoff": end.isoformat(),
        "baseline_candidate": baseline_id,
        "selected_point_candidate": selected_id,
        "formal_parity_max_abs_error": parity,
        "candidate_ids": int(len(surface)),
        "unique_paths": int(surface["return_hash"].nunique()),
        "dual_improvement_candidates": dual["candidate_id"].tolist(),
        "selected_point_metrics": {
            key: float(selected_row[key])
            for key in (
                "annualized_return_252",
                "sharpe",
                "max_drawdown",
                "minimum_segment_sharpe",
            )
        },
        "baseline_metrics": {
            key: float(baseline_row[key])
            for key in (
                "annualized_return_252",
                "sharpe",
                "max_drawdown",
                "minimum_segment_sharpe",
            )
        },
        "paired_block_bootstrap": bootstrap,
        "cscv": cscv,
        "reality_check": reality,
        "walk_forward_dual_win_rate": float(
            (
                walk_forward["test_return_delta"].gt(0)
                & walk_forward["test_sharpe_delta"].gt(0)
            ).mean()
        ),
        "leave_one_year_selection_dual_win_rate": float(
            (
                leave_year_selection["test_return_delta"].gt(0)
                & leave_year_selection["test_sharpe_delta"].gt(0)
            ).mean()
        ),
        "fixed_candidate_delete_year_dual_pass_rate": float(
            (
                fixed_leave_year["annualized_return_delta"].gt(0)
                & fixed_leave_year["sharpe_delta"].gt(0)
            ).mean()
        ),
        "calendar_year_dual_win_rate": float(
            (
                annual["total_return_delta"].gt(0)
                & annual["sharpe_delta"].gt(0)
            ).mean()
        ),
        "rolling": rolling_summary,
        "difference_events": {
            "events": int(len(events)),
            "positive": int(events["log_excess"].gt(0).sum()),
            "negative": int(events["log_excess"].lt(0).sum()),
            "top_two_positive_share": top_two_share,
            "leave_one_min_annualized_return_252": float(
                leave_event["annualized_return_252"].min()
            ),
            "leave_one_min_sharpe": float(leave_event["sharpe"].min()),
        },
        "decision": {
            "production_selector": baseline_id,
            "quality_score_promoted": False,
            "reason": (
                "The quality-score selector is the only full-sample dual "
                "winner, but its small retrospective gain is not significant "
                "after bootstrap or multiple-testing correction and the "
                "method/window pair is not a broad family plateau."
            ),
        },
    }

    output.mkdir(parents=True, exist_ok=True)
    surface.to_csv(output / "defender_selector_surface.csv", index=False)
    cscv_frame.to_csv(output / "defender_selector_cscv.csv", index=False)
    walk_forward.to_csv(
        output / "defender_selector_walk_forward.csv", index=False
    )
    leave_year_selection.to_csv(
        output / "defender_selector_leave_year_selection.csv", index=False
    )
    fixed_leave_year.to_csv(
        output / "defender_selector_fixed_leave_year.csv", index=False
    )
    annual.to_csv(output / "defender_selector_annual.csv", index=False)
    rolling.to_csv(output / "defender_selector_rolling.csv", index=False)
    events.to_csv(output / "defender_selector_events.csv", index=False)
    leave_event.to_csv(
        output / "defender_selector_leave_one_event.csv", index=False
    )
    cost_stress.to_csv(
        output / "defender_selector_cost_stress.csv", index=False
    )
    bootstrap_frame.to_csv(
        output / "defender_selector_bootstrap.csv", index=False
    )
    pd.DataFrame(
        {
            "baseline_selected_asset": selections[baseline_id]["selected_asset"],
            "candidate_selected_asset": selections[selected_id]["selected_asset"],
        }
    ).to_csv(output / "defender_selector_targets.csv")
    (output / "defender_selector_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    report = f"""# Defender排序指标2019样本跟进审计

证据状态：2019主审计后展开的回溯跟进，不是独立样本外。  
结论：保留月度40日纯收益最弱反转，不把40日质量动量排序晋升生产。

## 点估计

当前`{baseline_id}`为{float(baseline_row['annualized_return_252']):.2%}年化、
{float(baseline_row['sharpe']):.3f} Sharpe、MDD
{float(baseline_row['max_drawdown']):.2%}；候选`{selected_id}`为
{float(selected_row['annualized_return_252']):.2%}/
{float(selected_row['sharpe']):.3f}/
{float(selected_row['max_drawdown']):.2%}。候选把同一个40日对数收益乘以路径效率，36条
方法/方向/窗口路径中只有它同时提高完整年化和Sharpe。

## 稳健性

- 20日配对Bootstrap年化差95%区间
  `[{float(bootstrap['annualized_return_delta_ci_lower']):.2%},
  {float(bootstrap['annualized_return_delta_ci_upper']):.2%}]`，Sharpe差区间
  `[{float(bootstrap['sharpe_delta_ci_lower']):.3f},
  {float(bootstrap['sharpe_delta_ci_upper']):.3f}]`。
- 年度Reality Check `p={float(reality['p_value']):.4f}`；CSCV-PBO
  {float(cscv['pbo']):.1%}，训练冠军在测试段击败当前排序的比例
  {float(cscv['selected_beats_baseline_rate']):.1%}。
- 扩展walk-forward双指标胜率
  {audit['walk_forward_dual_win_rate']:.1%}，留一年重选胜率
  {audit['leave_one_year_selection_dual_win_rate']:.1%}；固定候选删除任一年仍双指标领先比例
  {audit['fixed_candidate_delete_year_dual_pass_rate']:.1%}。
- 逐年双指标胜率{audit['calendar_year_dual_win_rate']:.1%}，252/504日滚动双指标胜率
  {rolling_summary['252']['dual_win_rate']:.1%}/
  {rolling_summary['504']['dual_win_rate']:.1%}。
- 共{len(events)}段差异事件，{int(events['log_excess'].gt(0).sum())}正/
  {int(events['log_excess'].lt(0).sum())}负；前两大正事件占正向log excess
  {top_two_share:.1%}。删除任一事件后最低年化/Sharpe为
  {float(leave_event['annualized_return_252'].min()):.2%}/
  {float(leave_event['sharpe'].min()):.3f}。

费用提高到3倍和10倍不改变点估计方向，但无法消除样本复用、方法选择和40日窗口精确峰值。
因此把该方案保留为研究证据，不修改正式策略。
"""
    (output / "defender_selector_REPORT.md").write_text(
        report, encoding="utf-8"
    )
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output if args.output.is_absolute() else root / args.output
    result = run_audit(root, args.config, output)
    print(json.dumps(result["decision"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
