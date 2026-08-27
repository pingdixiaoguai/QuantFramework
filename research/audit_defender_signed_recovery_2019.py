"""Audit signed and relative recovery signals for flexible Defender exits."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import date
from itertools import product
from pathlib import Path

import numpy as np
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
from research.momentum_volatility import asof_previous_close, load_ohlc
from strategy.momentum_defender_w40_gold_escape import (
    formal_policies,
    run_formal_strategy,
)


DEFAULT_CONFIG = Path("research/configs/defender_exit_mechanism_audit_2019.yaml")
DEFAULT_OUTPUT = Path(
    "experiments/20260826_defender_signed_recovery_audit_2019"
)


@dataclass(frozen=True)
class RecoverySignalSpec:
    family: str
    confirmation_days: int
    horizon: int | None = None

    @property
    def candidate_id(self) -> str:
        horizon = "none" if self.horizon is None else str(self.horizon)
        return f"{self.family}_h{horizon}_c{self.confirmation_days}"


def _specs(config: dict) -> list[RecoverySignalSpec]:
    raw = config["signed_recovery_followup"]
    anchor = raw["anchor_signed_log_return"]
    result = [
        RecoverySignalSpec("anchor_signed", int(confirm), int(horizon))
        for horizon, confirm in product(
            anchor["horizons"], anchor["confirmation_days"]
        )
    ]
    relative = raw["top1_minus_defender_qm20"]
    result.extend(
        RecoverySignalSpec("relative_qm", int(confirm))
        for confirm in relative["confirmation_days"]
    )
    dual = raw["anchor20_and_relative_qm"]
    result.extend(
        RecoverySignalSpec("anchor20_and_relative", int(confirm), 20)
        for confirm in dual["confirmation_days"]
    )
    return result


def _anchor_signed_at_open(
    formal,
    end: date,
    horizon: int,
) -> pd.Series:
    close = load_ohlc("510300.SH", end)["close"].astype(float)
    signed = np.log(close).diff(horizon)
    return asof_previous_close(signed, formal.context.calendar)


def _relative_qm_at_open(context) -> pd.Series:
    metrics = quality_metrics_at_open(context)
    values = [
        metrics.at[timestamp, str(context.momentum_target.loc[timestamp])]
        - metrics.at[timestamp, "DEFENDER"]
        for timestamp in context.calendar
    ]
    return pd.Series(
        values,
        index=context.calendar,
        name="top1_minus_defender_qm20_at_open",
        dtype=float,
    )


def _evidence(
    formal,
    context,
    end: date,
    spec: RecoverySignalSpec,
    anchor_cache: dict[int, pd.Series],
) -> tuple[pd.Series, pd.Series]:
    relative = _relative_qm_at_open(context)
    if spec.family == "relative_qm":
        raw = relative
        return raw.gt(0.0) & raw.notna(), raw
    horizon = int(spec.horizon or 20)
    if horizon not in anchor_cache:
        anchor_cache[horizon] = _anchor_signed_at_open(
            formal, end, horizon
        )
    anchor = anchor_cache[horizon]
    if spec.family == "anchor_signed":
        return anchor.gt(0.0) & anchor.notna(), anchor
    evidence = (
        anchor.gt(0.0)
        & anchor.notna()
        & relative.gt(0.0)
        & relative.notna()
    )
    raw = pd.concat(
        [anchor.rename("anchor"), relative.rename("relative")], axis=1
    ).min(axis=1)
    return evidence, raw


def signed_recovery_state_schedule(
    score_at_open: pd.Series,
    recovery_evidence: pd.Series,
    recovery_signal: pd.Series,
    *,
    confirmation_days: int,
    minimum_lock_days: int = 5,
    fallback_day: int = 30,
    entry_percentile: float = 0.55,
    recovery_percentile: float = 0.40,
    momentum_lock_days: int = 30,
) -> pd.DataFrame:
    risk_on = True
    held_days = 10**9
    recovery_streak = 0
    rows: list[dict[str, object]] = []
    for timestamp, raw_score in score_at_open.items():
        score = float(raw_score) if pd.notna(raw_score) else np.nan
        previous = risk_on
        reason = "hold"
        early_qualified = False
        fallback_qualified = False
        if not np.isfinite(score):
            recovery_streak = 0
            reason = "insufficient_factor_history"
        elif risk_on:
            recovery_streak = 0
            if score >= entry_percentile:
                if held_days >= momentum_lock_days:
                    risk_on = False
                    held_days = 0
                    reason = "to_defender"
                else:
                    reason = "defender_entry_blocked_by_momentum_lock"
        else:
            evidence = bool(recovery_evidence.loc[timestamp])
            recovery_streak = recovery_streak + 1 if evidence else 0
            early_qualified = bool(
                held_days >= minimum_lock_days
                and held_days < fallback_day
                and recovery_streak >= confirmation_days
            )
            fallback_qualified = bool(
                held_days >= fallback_day and score <= recovery_percentile
            )
            if early_qualified or fallback_qualified:
                risk_on = True
                held_days = 0
                recovery_streak = 0
                reason = (
                    "to_momentum_signed_early"
                    if early_qualified
                    else "to_momentum_fallback_day30"
                )
            elif score <= 0.40:
                reason = "w40_recovery_observed_but_exit_not_qualified"
        rows.append(
            {
                "date": timestamp,
                "risk_on": risk_on,
                "state_changed": risk_on != previous,
                "state_reason": reason,
                "downside_raqm_percentile_at_open": score,
                "entry_confirmation_streak": 0,
                "recovery_confirmation_streak": recovery_streak,
                "held_days_at_open": held_days,
                "recovery_signal_at_open": recovery_signal.loc[timestamp],
                "recovery_evidence": bool(recovery_evidence.loc[timestamp]),
                "early_exit_qualified": early_qualified,
                "fallback_exit_qualified": fallback_qualified,
            }
        )
        held_days += 1
    return pd.DataFrame(rows).set_index("date")


def _run_candidate(formal, context, metrics, state):
    run = run_asset_specific_w40_escape(
        context,
        state,
        formal_policies(),
        metrics=metrics,
        immediate_entry_veto=True,
    )
    return run.daily["return"].astype(float), dict(run.audit)


def _hash(returns: pd.Series) -> str:
    return hashlib.sha256(
        returns.to_numpy(dtype="<f8").tobytes()
    ).hexdigest()


def run_audit(root: Path, config_path: Path, output: Path) -> dict[str, object]:
    applied = config_path if config_path.is_absolute() else root / config_path
    config = yaml.safe_load(applied.read_text(encoding="utf-8"))
    experiment = config["experiment"]
    followup = config["signed_recovery_followup"]
    checks = config["overfit_checks"]
    start = date.fromisoformat(str(experiment["evaluation_start"]))
    end = date.fromisoformat(str(experiment["evidence_cutoff"]))
    formal = run_formal_strategy(root, start=start, end=end)
    baseline = formal.daily["return"].astype(float)
    if _hash(baseline) != str(experiment["expected_formal_return_hash"]):
        raise AssertionError("2019 formal checkpoint changed")
    periods = {
        str(name): (str(bounds[0]), str(bounds[1]))
        for name, bounds in config["periods"].items()
    }
    base_metrics = quality_metrics_at_open(formal.context)
    anchor_cache: dict[int, pd.Series] = {}
    specs = _specs(config)
    rows: list[dict[str, object]] = []
    returns: dict[str, pd.Series] = {"fixed_lock_30": baseline}
    states: dict[str, pd.DataFrame] = {"fixed_lock_30": formal.state}
    evidence_cache: dict[str, tuple[pd.Series, pd.Series]] = {}
    specs_by_id: dict[str, RecoverySignalSpec] = {}

    baseline_row = _metric_row(
        "fixed_lock_30",
        "signed_recovery",
        baseline,
        periods,
    )
    baseline_row.update(
        {
            "signal_family": "fixed_lock",
            "signal_horizon": np.nan,
            "confirmation_days": 1,
            "early_recoveries": 0,
            "escape_entries": int(formal.audit["escape_entries"]),
        }
    )
    rows.append(baseline_row)
    for spec in specs:
        evidence, raw_signal = _evidence(
            formal,
            formal.context,
            end,
            spec,
            anchor_cache,
        )
        state = signed_recovery_state_schedule(
            formal.score_at_open,
            evidence,
            raw_signal,
            confirmation_days=spec.confirmation_days,
            minimum_lock_days=int(followup["minimum_lock_days"]),
            fallback_day=int(followup["fallback_to_current_at_day"]),
        )
        candidate_returns, run_audit = _run_candidate(
            formal,
            formal.context,
            base_metrics,
            state,
        )
        candidate_id = spec.candidate_id
        returns[candidate_id] = candidate_returns
        states[candidate_id] = state
        evidence_cache[candidate_id] = (evidence, raw_signal)
        specs_by_id[candidate_id] = spec
        row = _metric_row(
            candidate_id,
            "signed_recovery",
            candidate_returns,
            periods,
        )
        row.update(
            {
                "signal_family": spec.family,
                "signal_horizon": spec.horizon,
                "confirmation_days": spec.confirmation_days,
                "early_recoveries": int(
                    state["state_reason"].eq(
                        "to_momentum_signed_early"
                    ).sum()
                ),
                "escape_entries": int(run_audit["escape_entries"]),
            }
        )
        rows.append(row)
    surface = pd.DataFrame(rows)
    baseline_metrics = surface.set_index("candidate_id").loc[
        "fixed_lock_30"
    ]
    dual = surface.loc[
        surface["annualized_return_252"].gt(
            baseline_metrics["annualized_return_252"]
        )
        & surface["sharpe"].gt(baseline_metrics["sharpe"])
    ].copy()
    strict = dual.loc[
        dual["max_drawdown"].ge(
            baseline_metrics["max_drawdown"] - 1e-12
        )
        & dual["minimum_segment_sharpe"].ge(
            baseline_metrics["minimum_segment_sharpe"] - 1e-12
        )
    ].copy()
    pool = strict if not strict.empty else dual
    if pool.empty:
        selected_id = str(
            surface.loc[~surface["candidate_id"].eq("fixed_lock_30")]
            .sort_values(
                ["sharpe", "annualized_return_252"], ascending=False
            )
            .iloc[0]["candidate_id"]
        )
    else:
        selected_id = str(
            pool.sort_values(
                [
                    "minimum_segment_sharpe",
                    "sharpe",
                    "annualized_return_252",
                ],
                ascending=False,
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
    for multiplier in checks["transaction_cost_multipliers"]:
        context = _scaled_cost_context(formal, float(multiplier), end)
        metrics = quality_metrics_at_open(context)
        for candidate_id in ("fixed_lock_30", selected_id):
            if candidate_id == "fixed_lock_30":
                state = formal.state
            else:
                spec = specs_by_id[candidate_id]
                evidence, raw_signal = _evidence(
                    formal,
                    context,
                    end,
                    spec,
                    anchor_cache,
                )
                state = signed_recovery_state_schedule(
                    formal.score_at_open,
                    evidence,
                    raw_signal,
                    confirmation_days=spec.confirmation_days,
                    minimum_lock_days=int(followup["minimum_lock_days"]),
                    fallback_day=int(followup["fallback_to_current_at_day"]),
                )
            candidate_returns, run_audit = _run_candidate(
                formal,
                context,
                metrics,
                state,
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
    strong_support = bool(
        selected_id in set(dual["candidate_id"])
        and bootstrap["annualized_return_delta_ci_lower"] > 0
        and bootstrap["sharpe_delta_ci_lower"] > 0
        and reality["p_value"] < 0.05
    )
    audit: dict[str, object] = {
        "research_id": "defender_signed_recovery_audit_2019_v1",
        "status": (
            "candidate_supported_pending_user_promotion"
            if strong_support
            else "keep_current_lock_no_robust_signed_recovery"
        ),
        "evidence_status": followup["evidence_status"],
        "evaluation_start": start.isoformat(),
        "evidence_cutoff": end.isoformat(),
        "baseline_candidate": "fixed_lock_30",
        "selected_point_candidate": selected_id,
        "candidate_ids": int(len(surface)),
        "unique_paths": int(len(panel.columns)),
        "dual_improvement_candidates": dual["candidate_id"].tolist(),
        "strict_dual_candidates": strict["candidate_id"].tolist(),
        "baseline_metrics": {
            key: float(baseline_metrics[key])
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
            "leave_one_min_annualized_return_252": float(
                leave_event["annualized_return_252"].min()
                if not leave_event.empty
                else baseline_metrics["annualized_return_252"]
            ),
            "leave_one_min_sharpe": float(
                leave_event["sharpe"].min()
                if not leave_event.empty
                else baseline_metrics["sharpe"]
            ),
        },
        "decision": {
            "production_defender_lock_days": 30,
            "signed_recovery_promoted": False,
            "strong_statistical_support": strong_support,
            "reason": (
                "No signed or relative recovery candidate clears the joint "
                "performance, drawdown, bootstrap, and multiple-testing gates."
            ),
        },
    }

    output.mkdir(parents=True, exist_ok=True)
    surface.to_csv(output / "candidate_surface.csv", index=False)
    states[selected_id].to_csv(output / "selected_state.csv")
    events.to_csv(output / "selected_difference_events.csv", index=False)
    leave_event.to_csv(output / "selected_leave_one_event.csv", index=False)
    cscv_frame.to_csv(output / "cscv.csv", index=False)
    walk_forward.to_csv(output / "walk_forward.csv", index=False)
    leave_year_selection.to_csv(
        output / "leave_one_year_selection.csv", index=False
    )
    fixed_leave_year.to_csv(
        output / "fixed_candidate_leave_one_year.csv", index=False
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

    report = f"""# Defender有符号恢复退出审计（2019主样本）

证据状态：诊断下跌分位的0点质量后展开的回溯研究，不是独立样本外。  
正式决定：保持30日锁；本轮不修改生产策略。

进入Defender继续使用当前510300 40日下跌幅度分位≥55%。早退只使用自然零阈值：
510300有符号10/20/40/60日对数收益转正、Momentum Top1相对连续Defender的QM20转正，或
20日收益与相对QM同时转正；实际Defender至少持有5日，并要求1/3/5/10日连续确认。第30日
仍回到当前P40≤0.40的单日恢复规则。

共{len(surface)}个参数ID、{len(panel.columns)}条唯一路径。当前30日锁为
{float(baseline_metrics['annualized_return_252']):.2%}年化、
{float(baseline_metrics['sharpe']):.3f} Sharpe、MDD
{float(baseline_metrics['max_drawdown']):.2%}。预设稳健排序选中的`{selected_id}`为
{float(selected_row['annualized_return_252']):.2%}/
{float(selected_row['sharpe']):.3f}/
{float(selected_row['max_drawdown']):.2%}。

- Bootstrap年化差区间
  `[{float(bootstrap['annualized_return_delta_ci_lower']):.2%},
  {float(bootstrap['annualized_return_delta_ci_upper']):.2%}]`，Sharpe差区间
  `[{float(bootstrap['sharpe_delta_ci_lower']):.3f},
  {float(bootstrap['sharpe_delta_ci_upper']):.3f}]`。
- Reality Check `p={float(reality['p_value']):.4f}`，CSCV-PBO
  {float(cscv['pbo']):.1%}，训练冠军测试段击败当前锁比例
  {float(cscv['selected_beats_baseline_rate']):.1%}。
- walk-forward/留一年重选双指标胜率
  {audit['walk_forward_dual_win_rate']:.1%}/
  {audit['leave_one_year_selection_dual_win_rate']:.1%}；252/504日滚动双指标胜率
  {rolling_summary['252']['dual_win_rate']:.1%}/
  {rolling_summary['504']['dual_win_rate']:.1%}。

把进入与退出指标解耦在机制上更优雅，也消除了“所有正40日收益都挤在P=0”的信息损失；但
历史结果仍未形成可晋升的稳健平台。建议把最简单的自然零阈值候选保存为前瞻影子，而不是用
同一历史继续选择 horizon、确认日或复合条件。
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
    result = run_audit(root, args.config, output)
    print(json.dumps(result["decision"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
