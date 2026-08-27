"""Search the absolute QM40 threshold for v4 Defender recovery."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date
from pathlib import Path

import pandas as pd
import yaml

from research.audit_current_strategy_occam_robustness import _metric_row
from research.audit_defender_selector_2019 import (
    _difference_events,
    _leave_one_event,
)
from research.audit_momentum_hold_2019_followup import (
    _calendar_year_comparison,
    _fixed_leave_one_year,
    _rolling_comparison,
    _scaled_cost_context,
)
from research.momentum_defender_gold_override_overfit import (
    cscv_pbo,
    expanding_walk_forward,
    leave_one_year_selection,
    paired_block_bootstrap,
    yearly_reality_check,
)
from research.momentum_defender_occam import performance
from research.momentum_defender_w40_asset_specific_escape import (
    run_asset_specific_w40_escape,
)
from research.momentum_defender_w40_top1_escape import quality_metrics_at_open
from strategy.momentum_defender_w40_qm40_signed_exit import (
    formal_policies,
    qm40_recovery_state_schedule,
    run_formal_strategy,
)


DEFAULT_CONFIG = Path(
    "research/configs/qm40_recovery_threshold_search_2019.yaml"
)
DEFAULT_OUTPUT = Path(
    "experiments/20260826_qm40_recovery_threshold_search_2019"
)


def _hash(returns: pd.Series) -> str:
    return hashlib.sha256(
        returns.to_numpy(dtype="<f8").tobytes()
    ).hexdigest()


def _candidate_id(threshold: float) -> str:
    return f"qm40_threshold_{threshold:+.5f}"


def _run_threshold(formal, context, metrics, threshold: float):
    state = qm40_recovery_state_schedule(
        formal.score_at_open,
        formal.anchor_qm40_at_open,
        qm40_recovery_threshold=threshold,
    )
    run = run_asset_specific_w40_escape(
        context,
        state,
        formal_policies(),
        metrics=metrics,
        immediate_entry_veto=True,
    )
    return run.daily["return"].astype(float), state, run


def run_search(root: Path, config_path: Path, output: Path) -> dict[str, object]:
    applied = config_path if config_path.is_absolute() else root / config_path
    config = yaml.safe_load(applied.read_text(encoding="utf-8"))
    experiment = config["experiment"]
    start = date.fromisoformat(str(experiment["evaluation_start"]))
    end = date.fromisoformat(str(experiment["evidence_cutoff"]))
    formal = run_formal_strategy(root, start=start, end=end)
    baseline = formal.daily["return"].astype(float)
    if _hash(baseline) != str(experiment["expected_baseline_return_hash"]):
        raise AssertionError("formal v4 2019 checkpoint changed")
    metrics = quality_metrics_at_open(formal.context)
    periods = {
        str(name): (str(bounds[0]), str(bounds[1]))
        for name, bounds in config["periods"].items()
    }
    thresholds = [float(value) for value in config["threshold_grid"]["values"]]

    rows: list[dict[str, object]] = []
    returns: dict[str, pd.Series] = {}
    states: dict[str, pd.DataFrame] = {}
    runs: dict[str, object] = {}
    for position, threshold in enumerate(thresholds):
        candidate_id = _candidate_id(threshold)
        candidate_returns, state, run = _run_threshold(
            formal, formal.context, metrics, threshold
        )
        returns[candidate_id] = candidate_returns
        states[candidate_id] = state
        runs[candidate_id] = run
        row = _metric_row(
            candidate_id,
            "qm40_recovery_threshold",
            candidate_returns,
            periods,
        )
        row.update(
            {
                "grid_position": position,
                "threshold": threshold,
                "early_recoveries": int(
                    state["state_reason"].eq(
                        "qm40_recovery_to_momentum"
                    ).sum()
                ),
                "base_defender_entries": int(
                    (state["state_changed"] & ~state["risk_on"]).sum()
                ),
                "gold_escape_entries": int(run.audit["escape_entries"]),
                "candidate_switches": int(run.daily["switched"].sum()),
            }
        )
        rows.append(row)
    surface = pd.DataFrame(rows).sort_values("grid_position").reset_index(drop=True)
    baseline_id = _candidate_id(float(config["threshold_grid"]["baseline_threshold"]))
    baseline_parity = float((returns[baseline_id] - baseline).abs().max())
    if baseline_parity > 1e-14:
        raise AssertionError(f"threshold-zero parity failed: {baseline_parity:.3e}")

    radius = int(
        config["robust_selection"]["neighborhood_radius_in_grid_steps"]
    )
    for row_position in range(len(surface)):
        left = max(0, row_position - radius)
        right = min(len(surface), row_position + radius + 1)
        neighborhood = surface.iloc[left:right]
        surface.at[row_position, "neighborhood_annualized_q25"] = float(
            neighborhood["annualized_return_252"].quantile(0.25)
        )
        surface.at[row_position, "neighborhood_sharpe_q25"] = float(
            neighborhood["sharpe"].quantile(0.25)
        )
        surface.at[row_position, "neighborhood_mdd_worst"] = float(
            neighborhood["max_drawdown"].min()
        )

    baseline_row = surface.set_index("candidate_id").loc[baseline_id]
    selection = config["robust_selection"]
    eligible = surface.loc[
        surface["max_drawdown"].ge(
            float(baseline_row["max_drawdown"])
            - float(selection["maximum_mdd_worsening"])
        )
        & surface["minimum_segment_sharpe"].ge(
            float(baseline_row["minimum_segment_sharpe"])
            - float(selection["maximum_minimum_segment_sharpe_drop"])
        )
        & surface["annualized_return_252"].ge(
            float(baseline_row["annualized_return_252"])
            if bool(selection["require_full_annualized_not_below_baseline"])
            else float("-inf")
        )
        & surface["sharpe"].ge(
            float(baseline_row["sharpe"])
            if bool(selection["require_full_sharpe_not_below_baseline"])
            else float("-inf")
        )
    ].copy()
    ranking_columns = {
        "full_annualized_return_252": "annualized_return_252",
        "full_sharpe": "sharpe",
        "minimum_segment_sharpe": "minimum_segment_sharpe",
        "complete_pool_sharpe": "complete_pool_sharpe",
        "neighborhood_annualized_q25": "neighborhood_annualized_q25",
        "neighborhood_sharpe_q25": "neighborhood_sharpe_q25",
    }
    rank_fields: list[str] = []
    for declared in selection["ranking_fields"]:
        column = ranking_columns[str(declared)]
        rank_column = f"rank_pct_{column}"
        eligible[rank_column] = eligible[column].rank(
            method="average", pct=True
        )
        rank_fields.append(rank_column)
    eligible["robust_score"] = eligible[rank_fields].mean(axis=1)
    best_score = float(eligible["robust_score"].max())
    finalists = eligible.loc[
        eligible["robust_score"].ge(best_score - 1e-12)
    ].copy()
    finalists["plateau_center_distance"] = (
        finalists["threshold"] - finalists["threshold"].median()
    ).abs()
    selected_id = str(
        finalists.sort_values(
            [
                "neighborhood_sharpe_q25",
                "neighborhood_annualized_q25",
                "plateau_center_distance",
            ],
            ascending=[False, False, True],
        ).iloc[0]["candidate_id"]
    )
    selected = returns[selected_id]
    selected_row = surface.set_index("candidate_id").loc[selected_id]

    unique: dict[str, pd.Series] = {}
    seen: set[str] = set()
    for candidate_id, candidate_returns in returns.items():
        digest = _hash(candidate_returns)
        if digest not in seen:
            unique[candidate_id] = candidate_returns
            seen.add(digest)
    panel = pd.DataFrame(unique, index=formal.context.calendar)
    checks = config["overfit_checks"]
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
        [int(value) for value in checks["rolling_windows"]],
    )
    events = _difference_events(selected, baseline)
    leave_event = _leave_one_event(selected, baseline, events)
    if events.empty:
        events = pd.DataFrame(
            columns=[
                "event_id",
                "start",
                "end",
                "observations",
                "candidate_total_return",
                "baseline_total_return",
                "return_delta",
                "log_excess",
            ]
        )
        leave_event = pd.DataFrame(
            columns=[
                "event_id",
                "deleted_start",
                "deleted_end",
                "annualized_return_252",
                "sharpe",
                "max_drawdown",
            ]
        )

    cost_rows: list[dict[str, object]] = []
    selected_threshold = float(selected_row["threshold"])
    for multiplier in checks["transaction_cost_multipliers"]:
        context = _scaled_cost_context(formal, float(multiplier), end)
        applied_metrics = quality_metrics_at_open(context)
        for candidate_id, threshold in (
            (baseline_id, 0.0),
            (selected_id, selected_threshold),
        ):
            candidate_returns, state, run = _run_threshold(
                formal,
                context,
                applied_metrics,
                threshold,
            )
            measured = performance(candidate_returns)
            cost_rows.append(
                {
                    "cost_multiplier": float(multiplier),
                    "candidate_id": candidate_id,
                    "threshold": threshold,
                    "annualized_return_252": measured[
                        "annualized_return_252"
                    ],
                    "sharpe": measured["sharpe"],
                    "max_drawdown": measured["max_drawdown"],
                    "early_recoveries": int(
                        state["state_reason"].eq(
                            "qm40_recovery_to_momentum"
                        ).sum()
                    ),
                    "gold_escape_entries": int(run.audit["escape_entries"]),
                }
            )
    cost_stress = pd.DataFrame(cost_rows)
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
        "research_id": experiment["id"],
        "status": "completed_research_only",
        "evidence_status": experiment["evidence_status"],
        "evaluation_start": start.isoformat(),
        "evidence_cutoff": end.isoformat(),
        "baseline_candidate": baseline_id,
        "selected_candidate": selected_id,
        "selected_threshold": selected_threshold,
        "baseline_parity_max_abs_error": baseline_parity,
        "candidate_ids": int(len(surface)),
        "unique_paths": int(len(panel.columns)),
        "eligible_candidates": int(len(eligible)),
        "baseline_metrics": {
            key: float(baseline_row[key])
            for key in (
                "annualized_return_252",
                "sharpe",
                "max_drawdown",
                "minimum_segment_sharpe",
            )
        },
        "selected_metrics": {
            key: float(selected_row[key])
            for key in (
                "annualized_return_252",
                "sharpe",
                "max_drawdown",
                "minimum_segment_sharpe",
                "complete_pool_annualized_return_252",
                "complete_pool_sharpe",
                "neighborhood_annualized_q25",
                "neighborhood_sharpe_q25",
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
        "fixed_selected_delete_year_dual_pass_rate": float(
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
            "leave_one_min_annualized_return_252": float(
                leave_event["annualized_return_252"].min()
                if not leave_event.empty
                else baseline_row["annualized_return_252"]
            ),
            "leave_one_min_sharpe": float(
                leave_event["sharpe"].min()
                if not leave_event.empty
                else baseline_row["sharpe"]
            ),
        },
        "production_changed": False,
        "decision": {
            "formal_threshold": 0.0,
            "selected_threshold_promoted": False,
            "reason": (
                "Threshold selection is retrospective and requires explicit "
                "promotion after reviewing robustness and multiplicity evidence."
            ),
        },
    }

    output.mkdir(parents=True, exist_ok=True)
    surface.to_csv(output / "threshold_surface.csv", index=False)
    eligible.to_csv(output / "eligible_candidates.csv", index=False)
    pd.DataFrame(returns).to_parquet(output / "daily_returns.parquet")
    states[selected_id].to_csv(output / "selected_state.csv")
    runs[selected_id].daily.to_csv(output / "selected_daily.csv")
    events.to_csv(output / "difference_events.csv", index=False)
    leave_event.to_csv(output / "leave_one_event.csv", index=False)
    cscv_frame.to_csv(output / "cscv.csv", index=False)
    walk_forward.to_csv(output / "walk_forward.csv", index=False)
    leave_year_selection.to_csv(
        output / "leave_one_year_selection.csv", index=False
    )
    fixed_leave_year.to_csv(
        output / "fixed_selected_leave_one_year.csv", index=False
    )
    annual.to_csv(output / "calendar_year_comparison.csv", index=False)
    rolling.to_csv(output / "rolling_comparison.csv", index=False)
    cost_stress.to_csv(output / "cost_stress.csv", index=False)
    bootstrap_frame.to_csv(output / "paired_block_bootstrap.csv", index=False)
    (output / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "research_config.yaml").write_text(
        applied.read_text(encoding="utf-8"), encoding="utf-8"
    )

    report = f"""# QM40基础Defender恢复阈值寻参（2019主样本）

其他正式v4层全部冻结，仅搜索早退条件\\(QM40>\\theta\\)中的绝对阈值\\(\\theta\\)。基础Defender
最低5日、连续10日确认和30日35%分位保底保持不变。研究是回溯寻参，不是独立样本外。

当前正式\\(\\theta=0\\)为{float(baseline_row['annualized_return_252']):.2%}年化、
{float(baseline_row['sharpe']):.3f} Sharpe、MDD
{float(baseline_row['max_drawdown']):.2%}。稳健排序选中
\\(\\theta={selected_threshold:+.5f}\\)，结果为
{float(selected_row['annualized_return_252']):.2%}/
{float(selected_row['sharpe']):.3f}/
{float(selected_row['max_drawdown']):.2%}；邻域Q25为
{float(selected_row['neighborhood_annualized_q25']):.2%}/
{float(selected_row['neighborhood_sharpe_q25']):.3f}。

- Bootstrap年化差区间
  `[{float(bootstrap['annualized_return_delta_ci_lower']):.2%},
  {float(bootstrap['annualized_return_delta_ci_upper']):.2%}]`，Sharpe差区间
  `[{float(bootstrap['sharpe_delta_ci_lower']):.3f},
  {float(bootstrap['sharpe_delta_ci_upper']):.3f}]`；
- Reality Check `p={float(reality['p_value']):.4f}`，CSCV-PBO
  {float(cscv['pbo']):.1%}；
- walk-forward/留一年重选双指标胜率
  {audit['walk_forward_dual_win_rate']:.1%}/
  {audit['leave_one_year_selection_dual_win_rate']:.1%}；
- 252/504日滚动双指标胜率
  {rolling_summary['252']['dual_win_rate']:.1%}/
  {rolling_summary['504']['dual_win_rate']:.1%}。

本报告先给出稳健候选与证据，不自动修改正式阈值。
"""
    (output / "REPORT.md").write_text(report, encoding="utf-8")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output if args.output.is_absolute() else root / args.output
    result = run_search(root, args.config, output)
    print(json.dumps(result["decision"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
