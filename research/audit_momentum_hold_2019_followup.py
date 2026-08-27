"""Post-scan robustness audit for the 2019-start Momentum hold-day surface."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from datetime import date
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
    _momentum_market,
    _momentum_schedule,
    _periods,
    _return_hash,
)
from research.defender_curve_momentum import (
    DEFENDER_CANDIDATE,
    _single_etf_interface,
)
from research.momentum_defender_gold_override_overfit import (
    cscv_pbo,
    expanding_walk_forward,
    leave_one_year_selection,
    paired_block_bootstrap,
    yearly_reality_check,
)
from research.momentum_defender_occam import (
    HELD_RETURN,
    MOMENTUM_ASSETS,
    performance,
)
from research.momentum_defender_occam_defender import (
    build_portfolio_switch_interface,
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


def _target_for_hold(
    formal,
    market: Mapping[str, pd.DataFrame],
    master: pd.DatetimeIndex,
    hold_days: int,
) -> tuple[pd.Series, str]:
    target_all = _momentum_schedule(
        market,
        master,
        window=20,
        hold_days=hold_days,
    )
    target = target_all.reindex(formal.context.calendar)
    prior = target_all.loc[
        target_all.index < formal.context.calendar[0]
    ].dropna()
    previous = str(prior.iloc[-1] if not prior.empty else target.iloc[0])
    return target, previous


def _context_with_target(context, target: pd.Series, previous: str):
    return replace(
        context,
        momentum_target=target,
        baseline_target=target.rename("baseline_target_at_open"),
        initial_previous_candidate=previous,
    )


def _scaled_cost_context(formal, multiplier: float, end: date):
    calendar = formal.context.calendar
    interfaces: dict[str, pd.DataFrame] = {}
    curves: dict[str, pd.Series] = {}
    for asset in MOMENTUM_ASSETS:
        interface, close = _single_etf_interface(
            asset,
            calendar,
            end,
            cost_rate=0.0001 * multiplier,
        )
        interfaces[asset] = interface
        curves[asset] = close
    market = _load_formal_market(end)
    cost_rates = {
        **{
            asset: 0.0001 * multiplier
            for asset in FORMAL_DIVIDEND_ASSETS
        },
        DEFENSIVE_ASSET: 0.00001 * multiplier,
    }
    defender = build_portfolio_switch_interface(
        market,
        formal.base.defender.targets,
        cost_rates,
    )
    interfaces[DEFENDER_CANDIDATE] = defender
    curves[DEFENDER_CANDIDATE] = defender["nav_if_held"].astype(float)
    return replace(
        formal.context,
        interfaces=interfaces,
        curves=pd.DataFrame(curves, index=calendar),
    )


def _run_hold(
    formal,
    base_context,
    metrics: pd.DataFrame,
    target: pd.Series,
    previous: str,
) -> tuple[pd.Series, int]:
    context = _context_with_target(base_context, target, previous)
    run = run_asset_specific_w40_escape(
        context,
        formal.state,
        formal_policies(),
        metrics=metrics,
        immediate_entry_veto=True,
    )
    return run.daily["return"].astype(float), int(run.daily["switched"].sum())


def _fixed_leave_one_year(
    candidate: pd.Series,
    baseline: pd.Series,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for year in sorted(candidate.index.year.unique()):
        keep = candidate.index.year != year
        candidate_metrics = performance(candidate.loc[keep])
        baseline_metrics = performance(baseline.loc[keep])
        rows.append(
            {
                "deleted_year": int(year),
                "candidate_annualized_return_252": candidate_metrics[
                    "annualized_return_252"
                ],
                "baseline_annualized_return_252": baseline_metrics[
                    "annualized_return_252"
                ],
                "annualized_return_delta": candidate_metrics[
                    "annualized_return_252"
                ]
                - baseline_metrics["annualized_return_252"],
                "candidate_sharpe": candidate_metrics["sharpe"],
                "baseline_sharpe": baseline_metrics["sharpe"],
                "sharpe_delta": candidate_metrics["sharpe"]
                - baseline_metrics["sharpe"],
            }
        )
    return pd.DataFrame(rows)


def _calendar_year_comparison(
    candidate: pd.Series,
    baseline: pd.Series,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for year in sorted(candidate.index.year.unique()):
        mask = candidate.index.year == year
        candidate_metrics = performance(candidate.loc[mask])
        baseline_metrics = performance(baseline.loc[mask])
        rows.append(
            {
                "year": int(year),
                "candidate_total_return": candidate_metrics["total_return"],
                "baseline_total_return": baseline_metrics["total_return"],
                "total_return_delta": candidate_metrics["total_return"]
                - baseline_metrics["total_return"],
                "candidate_sharpe": candidate_metrics["sharpe"],
                "baseline_sharpe": baseline_metrics["sharpe"],
                "sharpe_delta": candidate_metrics["sharpe"]
                - baseline_metrics["sharpe"],
            }
        )
    return pd.DataFrame(rows)


def _rolling_comparison(
    candidate: pd.Series,
    baseline: pd.Series,
    windows: list[int],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for window in windows:
        for end_position in range(window - 1, len(candidate)):
            candidate_sample = candidate.iloc[
                end_position - window + 1 : end_position + 1
            ]
            baseline_sample = baseline.reindex(candidate_sample.index)
            candidate_metrics = performance(candidate_sample)
            baseline_metrics = performance(baseline_sample)
            rows.append(
                {
                    "window": int(window),
                    "end": candidate_sample.index[-1],
                    "annualized_return_delta": candidate_metrics[
                        "annualized_return_252"
                    ]
                    - baseline_metrics["annualized_return_252"],
                    "sharpe_delta": candidate_metrics["sharpe"]
                    - baseline_metrics["sharpe"],
                    "max_drawdown_delta": candidate_metrics["max_drawdown"]
                    - baseline_metrics["max_drawdown"],
                }
            )
    return pd.DataFrame(rows)


def run_followup(root: Path, config_path: Path, output: Path) -> dict[str, object]:
    applied = config_path if config_path.is_absolute() else root / config_path
    config = yaml.safe_load(applied.read_text(encoding="utf-8"))
    experiment = config["experiment"]
    spec = config["momentum_hold_followup"]
    checks = config["overfit_checks"]
    start = date.fromisoformat(str(experiment["evaluation_start"]))
    end = date.fromisoformat(str(experiment["evidence_cutoff"]))
    formal = run_formal_strategy(root, start=start, end=end)
    market, master = _momentum_market(end)
    periods = _periods(config)
    metrics = quality_metrics_at_open(formal.context)

    rows: list[dict[str, object]] = []
    returns: dict[str, pd.Series] = {}
    targets: dict[str, tuple[pd.Series, str]] = {}
    for hold_days in spec["rebalance_days"]:
        candidate_id = f"momentum_hold_{int(hold_days)}"
        target, previous = _target_for_hold(
            formal, market, master, int(hold_days)
        )
        targets[candidate_id] = (target, previous)
        candidate_returns, switches = _run_hold(
            formal,
            formal.context,
            metrics,
            target,
            previous,
        )
        returns[candidate_id] = candidate_returns
        row = _metric_row(
            candidate_id,
            "momentum_hold_followup",
            candidate_returns,
            periods,
        )
        row["rebalance_days"] = int(hold_days)
        row["candidate_switches"] = switches
        rows.append(row)
    surface = pd.DataFrame(rows)
    baseline_id = f"momentum_hold_{int(spec['baseline_rebalance_days'])}"
    selected_id = str(spec["selected_point_candidate"])
    baseline = returns[baseline_id]
    selected = returns[selected_id]
    parity = float(
        (baseline - formal.daily["return"].astype(float)).abs().max()
    )
    if parity > 1e-14:
        raise AssertionError(f"rd5 formal parity failed: {parity:.3e}")

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
        [int(value) for value in spec["rolling_windows"]],
    )

    cost_rows: list[dict[str, object]] = []
    for multiplier in spec["transaction_cost_multipliers"]:
        context = _scaled_cost_context(formal, float(multiplier), end)
        applied_metrics = quality_metrics_at_open(context)
        for candidate_id in (baseline_id, selected_id):
            target, previous = targets[candidate_id]
            candidate_returns, switches = _run_hold(
                formal,
                context,
                applied_metrics,
                target,
                previous,
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
                    "candidate_switches": switches,
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
    adjacent = surface.loc[
        surface["rebalance_days"].isin(
            [
                int(selected_row["rebalance_days"]) - 1,
                int(selected_row["rebalance_days"]) + 1,
            ]
        )
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
    audit: dict[str, object] = {
        "research_id": "current_strategy_occam_robustness_audit_2019_hold_followup_v1",
        "status": "rejected_isolated_rd7_point",
        "evidence_status": spec["evidence_status"],
        "evaluation_start": start.isoformat(),
        "evidence_cutoff": end.isoformat(),
        "baseline_candidate": baseline_id,
        "selected_point_candidate": selected_id,
        "rd5_formal_parity_max_abs_error": parity,
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
        "adjacent_points": adjacent[
            [
                "candidate_id",
                "annualized_return_252",
                "sharpe",
                "max_drawdown",
                "minimum_segment_sharpe",
            ]
        ].to_dict("records"),
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
        "fixed_rd7_delete_year_dual_pass_rate": float(
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
        "decision": {
            "production_rebalance_days": 5,
            "rd7_promoted": False,
            "reason": (
                "rd7 is the only full-sample dual winner in rd1-15, but it is "
                "an isolated point, worsens maximum drawdown and minimum "
                "segment Sharpe, and fails multiplicity, bootstrap, rolling, "
                "walk-forward, and leave-year robustness requirements."
            ),
        },
    }

    output.mkdir(parents=True, exist_ok=True)
    surface.to_csv(output / "momentum_hold_followup_surface.csv", index=False)
    cscv_frame.to_csv(output / "momentum_hold_followup_cscv.csv", index=False)
    walk_forward.to_csv(
        output / "momentum_hold_followup_walk_forward.csv", index=False
    )
    leave_year_selection.to_csv(
        output / "momentum_hold_followup_leave_year_selection.csv", index=False
    )
    fixed_leave_year.to_csv(
        output / "momentum_hold_followup_fixed_leave_year.csv", index=False
    )
    annual.to_csv(output / "momentum_hold_followup_annual.csv", index=False)
    rolling.to_csv(output / "momentum_hold_followup_rolling.csv", index=False)
    cost_stress.to_csv(
        output / "momentum_hold_followup_cost_stress.csv", index=False
    )
    bootstrap_frame.to_csv(
        output / "momentum_hold_followup_bootstrap.csv", index=False
    )
    (output / "momentum_hold_followup_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report = f"""# Momentum持有期2019样本跟进审计

证据状态：初始5点扫描后展开的回溯跟进，不是独立样本外。  
结论：保留`rebalance_days=5`，拒绝表面双指标领先的`rebalance_days=7`。

## 点估计

rd5为{float(baseline_row['annualized_return_252']):.2%}年化、
{float(baseline_row['sharpe']):.3f} Sharpe、MDD
{float(baseline_row['max_drawdown']):.2%}；rd7为
{float(selected_row['annualized_return_252']):.2%}/
{float(selected_row['sharpe']):.3f}/
{float(selected_row['max_drawdown']):.2%}。rd7提高年化和Sharpe，但最弱分段Sharpe由
{float(baseline_row['minimum_segment_sharpe']):.3f}降至
{float(selected_row['minimum_segment_sharpe']):.3f}。

rd1–15中只有rd7同时超过rd5，邻近rd6和rd8分别为
{float(adjacent.iloc[0]['annualized_return_252']):.2%}/
{float(adjacent.iloc[0]['sharpe']):.3f}与
{float(adjacent.iloc[1]['annualized_return_252']):.2%}/
{float(adjacent.iloc[1]['sharpe']):.3f}，是典型孤立峰。

## 过拟合与稳健性

- 20日配对Bootstrap：年化差95%区间
  `[{float(bootstrap['annualized_return_delta_ci_lower']):.2%},
  {float(bootstrap['annualized_return_delta_ci_upper']):.2%}]`；Sharpe差区间
  `[{float(bootstrap['sharpe_delta_ci_lower']):.3f},
  {float(bootstrap['sharpe_delta_ci_upper']):.3f}]`。
- rd1–15年度Reality Check `p={float(reality['p_value']):.4f}`；CSCV-PBO
  {float(cscv['pbo']):.1%}，训练冠军测试段击败rd5比例
  {float(cscv['selected_beats_baseline_rate']):.1%}。
- 扩展walk-forward双指标胜率
  {audit['walk_forward_dual_win_rate']:.1%}，留一年重选双指标胜率
  {audit['leave_one_year_selection_dual_win_rate']:.1%}；固定rd7删除任一年仍双指标领先的比例
  {audit['fixed_rd7_delete_year_dual_pass_rate']:.1%}。
- 逐年双指标胜率{audit['calendar_year_dual_win_rate']:.1%}；252/504日滚动双指标胜率分别
  {rolling_summary['252']['dual_win_rate']:.1%}/
  {rolling_summary['504']['dual_win_rate']:.1%}。

费用压力没有改变“rd7点估计较高”的方向，但不能修复时间与参数邻域不稳定。MDD由
{float(baseline_row['max_drawdown']):.2%}恶化到
{float(selected_row['max_drawdown']):.2%}，也不满足稳健优解要求。

因此不把2019完整样本上的单点冠军晋升为正式参数。
"""
    (output / "momentum_hold_followup_REPORT.md").write_text(
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
    result = run_followup(root, args.config, output)
    print(json.dumps(result["decision"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
