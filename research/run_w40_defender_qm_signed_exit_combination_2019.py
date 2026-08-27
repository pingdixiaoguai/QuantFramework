"""Backtest the user-requested W40 + Defender-QM + QM40-exit combination."""

from __future__ import annotations

import argparse
import hashlib
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
from factors.quality_momentum import compute as quality_momentum
from research.audit_current_strategy_occam_robustness import _metric_row
from research.audit_defender_exit_mechanism_2019 import (
    ExitPolicy,
    exit_state_schedule,
)
from research.audit_defender_selector_2019 import (
    _difference_events,
    _leave_one_event,
)
from research.audit_defender_signed_recovery_2019 import (
    signed_recovery_state_schedule,
)
from research.audit_momentum_hold_2019_followup import (
    _calendar_year_comparison,
    _fixed_leave_one_year,
    _rolling_comparison,
    _scaled_cost_context,
)
from research.defender_curve_momentum import DEFENDER_CANDIDATE
from research.momentum_defender_downside_raqm import strict_lag_percentile
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
from research.momentum_defender_w40_loss_gate import downside_log_loss
from research.momentum_defender_w40_top1_escape import quality_metrics_at_open
from research.momentum_volatility import asof_previous_close, load_ohlc
from strategy.momentum_defender_w40_gold_escape import (
    formal_policies,
    run_formal_strategy,
)


DEFAULT_CONFIG = Path(
    "research/configs/w40_defender_qm_signed_exit_combination_2019.yaml"
)
DEFAULT_OUTPUT = Path(
    "experiments/20260826_w40_defender_qm_signed_exit_combination_2019"
)


def _hash(returns: pd.Series) -> str:
    return hashlib.sha256(
        returns.to_numpy(dtype="<f8").tobytes()
    ).hexdigest()


def _w40_score_at_open(
    formal,
    end: date,
    *,
    history: int,
    min_history: int,
) -> pd.Series:
    close = load_ohlc("510300.SH", end)["close"].astype(float)
    loss = downside_log_loss(close, 40)
    percentile = strict_lag_percentile(
        loss,
        history_window=history,
        min_history=min_history,
    )
    return asof_previous_close(percentile, formal.context.calendar)


def _anchor_qm40_and_return_at_open(
    formal,
    end: date,
) -> tuple[pd.Series, pd.Series, int]:
    prices = load_ohlc("510300.SH", end)
    factor_input = prices[["close"]].reset_index()
    qm_close = quality_momentum(factor_input, {"window": 40})
    log_return_close = np.log(prices["close"]).diff(40)
    qm_open = asof_previous_close(qm_close, formal.context.calendar)
    return_open = asof_previous_close(
        log_return_close, formal.context.calendar
    )
    finite = qm_open.notna() & return_open.notna()
    sign_mismatches = int(
        (qm_open.loc[finite].gt(0.0) != return_open.loc[finite].gt(0.0)).sum()
    )
    return qm_open, return_open, sign_mismatches


def _defender_qm_context(
    formal,
    base_context,
    market: Mapping[str, pd.DataFrame],
    *,
    cost_multiplier: float,
):
    selector = MonthlySelectionSpec(40, "quality", "lowest")
    scores = score_at_open(
        market,
        FORMAL_DIVIDEND_ASSETS,
        formal.context.calendar,
        selector,
    )
    selection = monthly_top1_selection(
        market,
        FORMAL_DIVIDEND_ASSETS,
        formal.context.calendar,
        scores,
        selector,
    )
    targets = selected_asset_targets(
        selection["selected_asset"].astype(str),
        FORMAL_DIVIDEND_ASSETS,
        selected_weight=1.0,
        residual_asset=DEFENSIVE_ASSET,
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
    return context, selection, targets


def _state(
    score: pd.Series,
    qm40_open: pd.Series,
    *,
    use_w40_champion: bool,
    use_early_exit: bool,
) -> pd.DataFrame:
    entry = 0.60 if use_w40_champion else 0.55
    recovery = 0.35 if use_w40_champion else 0.40
    if use_early_exit:
        evidence = qm40_open.gt(0.0) & qm40_open.notna()
        return signed_recovery_state_schedule(
            score,
            evidence,
            qm40_open,
            confirmation_days=10,
            minimum_lock_days=5,
            fallback_day=30,
            entry_percentile=entry,
            recovery_percentile=recovery,
            momentum_lock_days=30,
        )
    return exit_state_schedule(
        score,
        ExitPolicy("fixed_lock", 30, 1),
        entry_percentile=entry,
        recovery_percentile=recovery,
        momentum_lock_days=30,
    )


def _run_variant(
    formal,
    base_context,
    current_score: pd.Series,
    champion_score: pd.Series,
    qm40_open: pd.Series,
    market: Mapping[str, pd.DataFrame],
    *,
    use_w40_champion: bool,
    use_defender_qm: bool,
    use_early_exit: bool,
    cost_multiplier: float,
):
    if use_defender_qm:
        context, selection, targets = _defender_qm_context(
            formal,
            base_context,
            market,
            cost_multiplier=cost_multiplier,
        )
    else:
        context = base_context
        selection = formal.base.defender.selection
        targets = formal.base.defender.targets
    score = champion_score if use_w40_champion else current_score
    state = _state(
        score,
        qm40_open,
        use_w40_champion=use_w40_champion,
        use_early_exit=use_early_exit,
    )
    metrics = quality_metrics_at_open(context)
    run = run_asset_specific_w40_escape(
        context,
        state,
        formal_policies(),
        metrics=metrics,
        immediate_entry_veto=True,
    )
    return run.daily["return"].astype(float), state, run, selection, targets


def run_combination(root: Path, config_path: Path, output: Path) -> dict[str, object]:
    applied = config_path if config_path.is_absolute() else root / config_path
    config = yaml.safe_load(applied.read_text(encoding="utf-8"))
    experiment = config["experiment"]
    start = date.fromisoformat(str(experiment["evaluation_start"]))
    end = date.fromisoformat(str(experiment["evidence_cutoff"]))
    formal = run_formal_strategy(root, start=start, end=end)
    baseline = formal.daily["return"].astype(float)
    if _hash(baseline) != str(experiment["expected_baseline_return_hash"]):
        raise AssertionError("2019 formal checkpoint changed")

    request = config["requested_changes"]
    w40 = request["w40_gate"]
    champion_score = _w40_score_at_open(
        formal,
        end,
        history=int(w40["percentile_history"]),
        min_history=int(w40["percentile_min_history"]),
    )
    current_score = formal.score_at_open.astype(float)
    qm40_open, return40_open, sign_mismatches = (
        _anchor_qm40_and_return_at_open(formal, end)
    )
    market = _load_formal_market(end)
    periods = {
        str(name): (str(bounds[0]), str(bounds[1]))
        for name, bounds in config["periods"].items()
    }

    variants = config["factorial_variants"]
    rows: list[dict[str, object]] = []
    returns: dict[str, pd.Series] = {}
    states: dict[str, pd.DataFrame] = {}
    runs: dict[str, object] = {}
    selections: dict[str, pd.DataFrame] = {}
    targets: dict[str, pd.DataFrame] = {}
    for candidate_id, enabled in variants.items():
        enabled_set = set(enabled)
        candidate_returns, state, run, selection, target = _run_variant(
            formal,
            formal.context,
            current_score,
            champion_score,
            qm40_open,
            market,
            use_w40_champion="w40_gate" in enabled_set,
            use_defender_qm="defender_selector" in enabled_set,
            use_early_exit="early_defender_exit" in enabled_set,
            cost_multiplier=1.0,
        )
        returns[candidate_id] = candidate_returns
        states[candidate_id] = state
        runs[candidate_id] = run
        selections[candidate_id] = selection
        targets[candidate_id] = target
        row = _metric_row(
            candidate_id,
            "requested_factorial_combination",
            candidate_returns,
            periods,
        )
        row.update(
            {
                "w40_champion": "w40_gate" in enabled_set,
                "defender_qm40": "defender_selector" in enabled_set,
                "qm40_early_exit": "early_defender_exit" in enabled_set,
                "base_defender_entries": int(
                    (state["state_changed"] & ~state["risk_on"]).sum()
                ),
                "base_momentum_recoveries": int(
                    (state["state_changed"] & state["risk_on"]).sum()
                ),
                "early_recoveries": int(
                    state["state_reason"].eq(
                        "to_momentum_signed_early"
                    ).sum()
                ),
                "gold_escape_entries": int(run.audit["escape_entries"]),
                "candidate_switches": int(run.daily["switched"].sum()),
            }
        )
        rows.append(row)
    surface = pd.DataFrame(rows)
    baseline_parity = float((returns["baseline"] - baseline).abs().max())
    baseline_state_parity = bool(
        states["baseline"]["risk_on"].equals(formal.state["risk_on"])
    )
    if baseline_parity > 1e-14 or not baseline_state_parity:
        raise AssertionError(
            f"baseline parity failed: returns={baseline_parity:.3e}, state={baseline_state_parity}"
        )
    requested_id = "requested_all_three"
    requested = returns[requested_id]

    panel = pd.DataFrame(returns, index=formal.context.calendar)
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
        requested,
        baseline,
        block_size=int(checks["paired_block_bootstrap_block"]),
        repetitions=int(checks["paired_block_bootstrap_repetitions"]),
        seed=int(checks["random_seed"]),
    )
    fixed_leave_year = _fixed_leave_one_year(requested, baseline)
    annual = _calendar_year_comparison(requested, baseline)
    rolling = _rolling_comparison(
        requested,
        baseline,
        [int(value) for value in checks["rolling_windows"]],
    )
    events = _difference_events(requested, baseline)
    leave_event = _leave_one_event(requested, baseline, events)

    cost_rows: list[dict[str, object]] = []
    for multiplier in checks["transaction_cost_multipliers"]:
        base_context = _scaled_cost_context(formal, float(multiplier), end)
        for candidate_id in ("baseline", requested_id):
            enabled_set = set(variants[candidate_id])
            candidate_returns, state, run, _, _ = _run_variant(
                formal,
                base_context,
                current_score,
                champion_score,
                qm40_open,
                market,
                use_w40_champion="w40_gate" in enabled_set,
                use_defender_qm="defender_selector" in enabled_set,
                use_early_exit="early_defender_exit" in enabled_set,
                cost_multiplier=float(multiplier),
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
                    "base_defender_entries": int(
                        (state["state_changed"] & ~state["risk_on"]).sum()
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
    baseline_row = surface.set_index("candidate_id").loc["baseline"]
    requested_row = surface.set_index("candidate_id").loc[requested_id]
    positive = events.loc[events["log_excess"].gt(0), "log_excess"].sort_values(
        ascending=False
    )
    top_two_share = float(
        positive.head(2).sum() / positive.sum()
    ) if positive.sum() > 0 else 0.0
    audit: dict[str, object] = {
        "research_id": experiment["id"],
        "status": "completed_user_requested_combination_backtest",
        "evidence_status": experiment["evidence_status"],
        "evaluation_start": start.isoformat(),
        "evidence_cutoff": end.isoformat(),
        "baseline_return_hash": _hash(baseline),
        "requested_return_hash": _hash(requested),
        "baseline_return_parity_max_abs_error": baseline_parity,
        "baseline_state_parity": baseline_state_parity,
        "qm40_positive_vs_r40_positive_mismatches": sign_mismatches,
        "qm40_zero_threshold_is_behaviorally_equivalent_to_r40_zero": bool(
            sign_mismatches == 0
        ),
        "factorial_candidates": int(len(surface)),
        "baseline_metrics": {
            key: float(baseline_row[key])
            for key in (
                "annualized_return_252",
                "sharpe",
                "max_drawdown",
                "minimum_segment_sharpe",
            )
        },
        "requested_metrics": {
            key: float(requested_row[key])
            for key in (
                "annualized_return_252",
                "sharpe",
                "max_drawdown",
                "minimum_segment_sharpe",
                "complete_pool_annualized_return_252",
                "complete_pool_sharpe",
                "complete_pool_max_drawdown",
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
        "fixed_requested_delete_year_dual_pass_rate": float(
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
        "production_changed": False,
        "decision": {
            "requested_combination_promoted": False,
            "reason": (
                "This is a user-requested post-selection combination. Report "
                "the backtest and robustness evidence, but do not change "
                "production without an explicit promotion decision."
            ),
        },
    }

    output.mkdir(parents=True, exist_ok=True)
    surface.to_csv(output / "factorial_metrics.csv", index=False)
    pd.DataFrame(returns).to_parquet(output / "factorial_daily_returns.parquet")
    pd.DataFrame(returns).to_csv(output / "factorial_daily_returns.csv")
    states[requested_id].to_csv(output / "requested_state.csv")
    runs[requested_id].daily.to_csv(output / "requested_daily.csv")
    selections[requested_id].to_csv(output / "requested_defender_selection.csv")
    targets[requested_id].to_csv(output / "requested_defender_targets.csv")
    events.to_csv(output / "difference_events.csv", index=False)
    leave_event.to_csv(output / "leave_one_event.csv", index=False)
    cscv_frame.to_csv(output / "cscv.csv", index=False)
    walk_forward.to_csv(output / "walk_forward.csv", index=False)
    leave_year_selection.to_csv(
        output / "leave_one_year_selection.csv", index=False
    )
    fixed_leave_year.to_csv(
        output / "fixed_requested_leave_one_year.csv", index=False
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

    report = f"""# W40冠军＋Defender QM40＋QM40恢复早退组合回测

主样本：2019-01-18至2026-08-26，状态重新初始化。  
证据状态：用户在观察单项结果后指定的组合，属于回溯事后组合，不是独立样本外。  
生产状态：只读研究，未修改正式v3。

## 请求方案

- W40：40日下跌幅度，严格滞后756日分位，60%进入Defender、35%恢复Momentum，30/30锁；
- Defender：每月选择\\(QM40=R40\\times ER40\\)最低的合格红利ETF；
- 早退：实际Defender至少5日，510300的QM40严格大于0连续10日则提前恢复Momentum；
- 30日后仍以W40分位≤35%作为保底恢复；Momentum和黄金覆盖保持正式v3不变。

QM40>0与R40>0的逐日触发差异为{sign_mismatches}，因此零阈值下路径效率调整不改变早退日期。

## 结果

当前v3为{float(baseline_row['annualized_return_252']):.2%}年化、
{float(baseline_row['sharpe']):.3f} Sharpe、MDD
{float(baseline_row['max_drawdown']):.2%}；请求组合为
{float(requested_row['annualized_return_252']):.2%}/
{float(requested_row['sharpe']):.3f}/
{float(requested_row['max_drawdown']):.2%}。完整六只池阶段为
{float(requested_row['complete_pool_annualized_return_252']):.2%}年化、
{float(requested_row['complete_pool_sharpe']):.3f} Sharpe、MDD
{float(requested_row['complete_pool_max_drawdown']):.2%}。

## 稳健性

- 20日Bootstrap年化差区间
  `[{float(bootstrap['annualized_return_delta_ci_lower']):.2%},
  {float(bootstrap['annualized_return_delta_ci_upper']):.2%}]`，Sharpe差区间
  `[{float(bootstrap['sharpe_delta_ci_lower']):.3f},
  {float(bootstrap['sharpe_delta_ci_upper']):.3f}]`。
- 八格组合Reality Check `p={float(reality['p_value']):.4f}`，CSCV-PBO
  {float(cscv['pbo']):.1%}，训练冠军测试段击败正式v3比例
  {float(cscv['selected_beats_baseline_rate']):.1%}。
- walk-forward/留一年重选双指标胜率
  {audit['walk_forward_dual_win_rate']:.1%}/
  {audit['leave_one_year_selection_dual_win_rate']:.1%}；固定请求组合删除任一年仍双指标领先比例
  {audit['fixed_requested_delete_year_dual_pass_rate']:.1%}。
- 252/504日滚动双指标胜率
  {rolling_summary['252']['dual_win_rate']:.1%}/
  {rolling_summary['504']['dual_win_rate']:.1%}；差异事件
  {len(events)}段，{int(events['log_excess'].gt(0).sum())}正/
  {int(events['log_excess'].lt(0).sum())}负。

本报告只回答组合后的历史效果。是否晋升必须另行决定，并接受三个已观察单项赢家被组合后的
多重选择风险。
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
    result = run_combination(root, args.config, output)
    print(json.dumps(result["decision"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
