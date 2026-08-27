"""Multiple-testing and event audit for the 2019-start Gold X/Y surface."""

from __future__ import annotations

import argparse
import json
from datetime import date
from itertools import product
from pathlib import Path

import pandas as pd
import yaml

from research.audit_current_strategy_occam_robustness import (
    _metric_row,
    _periods,
)
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
from research.momentum_defender_occam import MOMENTUM_ASSETS, performance
from research.momentum_defender_w40_asset_specific_escape import (
    AssetXYPolicy,
    run_asset_specific_w40_escape,
)
from research.momentum_defender_w40_top1_escape import quality_metrics_at_open
from strategy.momentum_defender_w40_gold_escape import run_formal_strategy


DEFAULT_CONFIG = Path(
    "research/configs/current_strategy_occam_robustness_audit_2019.yaml"
)
DEFAULT_OUTPUT = Path(
    "experiments/20260826_current_strategy_occam_robustness_audit_2019"
)


def _policies(entry: float, recovery: float):
    result = {asset: None for asset in MOMENTUM_ASSETS}
    result["518880.SH"] = AssetXYPolicy(entry, recovery)
    return result


def _run_candidate(
    formal,
    context,
    entry: float,
    recovery: float,
    immediate: bool,
) -> tuple[pd.Series, dict[str, object]]:
    metrics = quality_metrics_at_open(context)
    run = run_asset_specific_w40_escape(
        context,
        formal.state,
        _policies(entry, recovery),
        metrics=metrics,
        immediate_entry_veto=immediate,
    )
    return run.daily["return"].astype(float), dict(run.audit)


def run_audit(root: Path, config_path: Path, output: Path) -> dict[str, object]:
    applied = config_path if config_path.is_absolute() else root / config_path
    config = yaml.safe_load(applied.read_text(encoding="utf-8"))
    experiment = config["experiment"]
    grid = config["gold_threshold_scan"]
    spec = config["gold_threshold_followup"]
    checks = config["overfit_checks"]
    start = date.fromisoformat(str(experiment["evaluation_start"]))
    end = date.fromisoformat(str(experiment["evidence_cutoff"]))
    formal = run_formal_strategy(root, start=start, end=end)
    periods = _periods(config)

    rows: list[dict[str, object]] = []
    returns: dict[str, pd.Series] = {}
    params: dict[str, tuple[float, float, bool]] = {}
    for entry, recovery, immediate in product(
        grid["entry_x"],
        grid["exit_y"],
        grid["immediate_entry_veto"],
    ):
        if float(recovery) > float(entry):
            continue
        candidate_id = (
            f"gold_x{float(entry):+.3f}_y{float(recovery):+.3f}_"
            f"iv{int(bool(immediate))}"
        )
        candidate_returns, run_audit = _run_candidate(
            formal,
            formal.context,
            float(entry),
            float(recovery),
            bool(immediate),
        )
        returns[candidate_id] = candidate_returns
        params[candidate_id] = (
            float(entry),
            float(recovery),
            bool(immediate),
        )
        row = _metric_row(
            candidate_id,
            "gold_threshold_followup",
            candidate_returns,
            periods,
        )
        row.update(
            {
                "entry_x": float(entry),
                "exit_y": float(recovery),
                "immediate_entry_veto": bool(immediate),
                "escape_entries": int(run_audit["escape_entries"]),
                "immediate_entries": int(
                    run_audit["immediate_entry_veto_entries"]
                ),
            }
        )
        rows.append(row)
    surface = pd.DataFrame(rows)
    baseline_id = str(spec["baseline_candidate"])
    selected_id = str(spec["selected_point_candidate"])
    baseline = returns[baseline_id]
    selected = returns[selected_id]
    parity = float(
        (baseline - formal.daily["return"].astype(float)).abs().max()
    )
    if parity > 1e-14:
        raise AssertionError(f"formal Gold parity failed: {parity:.3e}")

    unique: dict[str, pd.Series] = {}
    hashes: set[bytes] = set()
    for candidate_id, candidate_returns in returns.items():
        raw = candidate_returns.to_numpy(dtype="<f8").tobytes()
        if raw not in hashes:
            unique[candidate_id] = candidate_returns
            hashes.add(raw)
    panel = pd.DataFrame(unique, index=formal.context.calendar)
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
    events = _difference_events(selected, baseline)
    leave_event = _leave_one_event(selected, baseline, events)

    cost_rows: list[dict[str, object]] = []
    for multiplier in spec["transaction_cost_multipliers"]:
        context = _scaled_cost_context(formal, float(multiplier), end)
        for candidate_id in (baseline_id, selected_id):
            entry, recovery, immediate = params[candidate_id]
            candidate_returns, run_audit = _run_candidate(
                formal,
                context,
                entry,
                recovery,
                immediate,
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
                    "escape_entries": int(run_audit["escape_entries"]),
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
        "research_id": "current_strategy_occam_robustness_audit_2019_gold_threshold_v1",
        "status": "rejected_small_zero_exit_gain",
        "evidence_status": spec["evidence_status"],
        "evaluation_start": start.isoformat(),
        "evidence_cutoff": end.isoformat(),
        "baseline_candidate": baseline_id,
        "selected_point_candidate": selected_id,
        "formal_parity_max_abs_error": parity,
        "candidate_ids": int(len(surface)),
        "unique_paths": int(len(panel.columns)),
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
            "production_gold_exit_y": -0.020,
            "zero_exit_promoted": False,
            "reason": (
                "The zero exit line produces a small point improvement but "
                "fails bootstrap, multiplicity, temporal selection, and event "
                "breadth requirements."
            ),
        },
    }

    output.mkdir(parents=True, exist_ok=True)
    surface.to_csv(output / "gold_threshold_followup_surface.csv", index=False)
    cscv_frame.to_csv(output / "gold_threshold_followup_cscv.csv", index=False)
    walk_forward.to_csv(
        output / "gold_threshold_followup_walk_forward.csv", index=False
    )
    leave_year_selection.to_csv(
        output / "gold_threshold_followup_leave_year_selection.csv", index=False
    )
    fixed_leave_year.to_csv(
        output / "gold_threshold_followup_fixed_leave_year.csv", index=False
    )
    annual.to_csv(output / "gold_threshold_followup_annual.csv", index=False)
    rolling.to_csv(output / "gold_threshold_followup_rolling.csv", index=False)
    events.to_csv(output / "gold_threshold_followup_events.csv", index=False)
    leave_event.to_csv(
        output / "gold_threshold_followup_leave_one_event.csv", index=False
    )
    cost_stress.to_csv(
        output / "gold_threshold_followup_cost_stress.csv", index=False
    )
    bootstrap_frame.to_csv(
        output / "gold_threshold_followup_bootstrap.csv", index=False
    )
    (output / "gold_threshold_followup_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    report = f"""# 黄金X/Y参数2019样本跟进审计

证据状态：2019主网格的回溯跟进，不是独立样本外。  
结论：保留黄金退出线-0.020，不把自然零退出线晋升生产。

当前`{baseline_id}`为{float(baseline_row['annualized_return_252']):.2%}年化、
{float(baseline_row['sharpe']):.3f} Sharpe；零退出候选`{selected_id}`为
{float(selected_row['annualized_return_252']):.2%}/
{float(selected_row['sharpe']):.3f}，MDD均为
{float(selected_row['max_drawdown']):.2%}。

- 20日Bootstrap年化差区间
  `[{float(bootstrap['annualized_return_delta_ci_lower']):.2%},
  {float(bootstrap['annualized_return_delta_ci_upper']):.2%}]`，Sharpe差区间
  `[{float(bootstrap['sharpe_delta_ci_lower']):.3f},
  {float(bootstrap['sharpe_delta_ci_upper']):.3f}]`。
- 16条唯一黄金路径Reality Check `p={float(reality['p_value']):.4f}`，CSCV-PBO
  {float(cscv['pbo']):.1%}，训练冠军测试段击败当前黄金参数比例
  {float(cscv['selected_beats_baseline_rate']):.1%}。
- walk-forward/留一年重选双指标胜率分别
  {audit['walk_forward_dual_win_rate']:.1%}/
  {audit['leave_one_year_selection_dual_win_rate']:.1%}；252/504日滚动双指标胜率
  {rolling_summary['252']['dual_win_rate']:.1%}/
  {rolling_summary['504']['dual_win_rate']:.1%}。
- 共{len(events)}段差异事件，前两大正事件占正向log excess
  {top_two_share:.1%}。

费用压力不改变微小点估计方向，但统计和事件广度均不足，因此不调整正式X/Y。
"""
    (output / "gold_threshold_followup_REPORT.md").write_text(
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
