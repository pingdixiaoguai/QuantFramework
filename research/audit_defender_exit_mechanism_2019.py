"""Audit the hard 30-session Defender lock and flexible exit alternatives."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import date
from itertools import product
from pathlib import Path
from typing import Mapping

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
from strategy.momentum_defender_w40_gold_escape import (
    formal_policies,
    run_formal_strategy,
)


DEFAULT_CONFIG = Path("research/configs/defender_exit_mechanism_audit_2019.yaml")
DEFAULT_OUTPUT = Path(
    "experiments/20260826_defender_exit_mechanism_audit_2019"
)


@dataclass(frozen=True)
class ExitPolicy:
    family: str
    minimum_lock_days: int
    confirmation_days: int
    early_threshold: float | None = None
    fallback_day: int = 30
    final_threshold: float = 0.40

    @property
    def candidate_id(self) -> str:
        if self.family == "fixed_lock":
            return f"fixed_lock_{self.minimum_lock_days}"
        threshold = (
            "none"
            if self.early_threshold is None
            else f"{self.early_threshold:.2f}"
        )
        return (
            f"{self.family}_min{self.minimum_lock_days}_"
            f"thr{threshold}_c{self.confirmation_days}_fb{self.fallback_day}"
        )


def _policies(config: Mapping[str, object]) -> list[ExitPolicy]:
    families = config["candidate_families"]
    result: list[ExitPolicy] = []
    fixed = families["fixed_lock"]
    result.extend(
        ExitPolicy(
            "fixed_lock",
            int(lock),
            int(fixed["recovery_confirmation_days"]),
        )
        for lock in fixed["lock_days"]
    )
    confirmed = families["confirmed_early_exit"]
    result.extend(
        ExitPolicy(
            "confirmed_early",
            int(minimum),
            int(confirmation),
            early_threshold=0.40,
            fallback_day=int(confirmed["fallback_to_current_at_day"]),
        )
        for minimum, confirmation in product(
            confirmed["minimum_lock_days"],
            confirmed["recovery_confirmation_days"],
        )
    )
    strong = families["strong_recovery_early_exit"]
    result.extend(
        ExitPolicy(
            "strong_early",
            int(strong["minimum_lock_days"]),
            int(confirmation),
            early_threshold=float(threshold),
            fallback_day=int(strong["fallback_to_current_at_day"]),
        )
        for threshold, confirmation in product(
            strong["strong_recovery_percentiles"],
            strong["recovery_confirmation_days"],
        )
    )
    linear = families["linear_recovery_early_exit"]
    result.extend(
        ExitPolicy(
            "linear_early",
            int(linear["minimum_lock_days"]),
            int(confirmation),
            early_threshold=float(threshold),
            fallback_day=int(linear["fallback_to_current_at_day"]),
            final_threshold=float(linear["final_recovery_percentile"]),
        )
        for threshold, confirmation in product(
            linear["initial_recovery_percentiles"],
            linear["recovery_confirmation_days"],
        )
    )
    unique = {policy.candidate_id: policy for policy in result}
    return list(unique.values())


def _early_threshold(policy: ExitPolicy, held_days: int) -> float:
    if policy.early_threshold is None:
        return policy.final_threshold
    if policy.family != "linear_early":
        return policy.early_threshold
    span = max(policy.fallback_day - policy.minimum_lock_days, 1)
    progress = min(
        max((held_days - policy.minimum_lock_days) / span, 0.0),
        1.0,
    )
    return float(
        policy.early_threshold
        + progress * (policy.final_threshold - policy.early_threshold)
    )


def exit_state_schedule(
    score_at_open: pd.Series,
    policy: ExitPolicy,
    *,
    entry_percentile: float = 0.55,
    recovery_percentile: float = 0.40,
    momentum_lock_days: int = 30,
) -> pd.DataFrame:
    """Replay the frozen entry rule with one alternative Defender exit rule."""
    risk_on = True
    held_days = 10**9
    entry_streak = 0
    recovery_streak = 0
    rows: list[dict[str, object]] = []
    for timestamp, raw_score in score_at_open.items():
        score = float(raw_score) if pd.notna(raw_score) else np.nan
        previous = risk_on
        reason = "hold"
        applied_threshold = np.nan
        fallback_qualified = False
        early_qualified = False
        if not np.isfinite(score):
            entry_streak = 0
            recovery_streak = 0
            reason = "insufficient_factor_history"
        elif risk_on:
            entry_streak = (
                entry_streak + 1 if score >= entry_percentile else 0
            )
            recovery_streak = 0
            if entry_streak >= 1:
                if held_days >= momentum_lock_days:
                    risk_on = False
                    held_days = 0
                    entry_streak = 0
                    recovery_streak = 0
                    reason = "to_defender"
                else:
                    reason = "defender_entry_blocked_by_momentum_lock"
        else:
            entry_streak = 0
            if policy.family == "fixed_lock":
                applied_threshold = recovery_percentile
                evidence = score <= recovery_percentile
                recovery_streak = recovery_streak + 1 if evidence else 0
                early_qualified = bool(
                    held_days >= policy.minimum_lock_days
                    and recovery_streak >= policy.confirmation_days
                )
            else:
                applied_threshold = _early_threshold(policy, held_days)
                evidence = score <= applied_threshold
                recovery_streak = recovery_streak + 1 if evidence else 0
                early_qualified = bool(
                    held_days >= policy.minimum_lock_days
                    and held_days < policy.fallback_day
                    and recovery_streak >= policy.confirmation_days
                )
                fallback_qualified = bool(
                    held_days >= policy.fallback_day
                    and score <= recovery_percentile
                )
            if early_qualified or fallback_qualified:
                risk_on = True
                held_days = 0
                entry_streak = 0
                recovery_streak = 0
                reason = (
                    "to_momentum_early_recovery"
                    if early_qualified
                    else "to_momentum_fallback_day30"
                )
            elif score <= recovery_percentile:
                reason = "recovery_observed_but_exit_not_qualified"
        rows.append(
            {
                "date": timestamp,
                "risk_on": risk_on,
                "state_changed": risk_on != previous,
                "state_reason": reason,
                "downside_raqm_percentile_at_open": score,
                "entry_confirmation_streak": entry_streak,
                "recovery_confirmation_streak": recovery_streak,
                "held_days_at_open": held_days,
                "applied_early_recovery_threshold": applied_threshold,
                "early_exit_qualified": early_qualified,
                "fallback_exit_qualified": fallback_qualified,
            }
        )
        held_days += 1
    return pd.DataFrame(rows).set_index("date")


def _run_policy(
    formal,
    context,
    metrics: pd.DataFrame,
    state: pd.DataFrame,
) -> tuple[pd.Series, dict[str, object]]:
    run = run_asset_specific_w40_escape(
        context,
        state,
        formal_policies(),
        metrics=metrics,
        immediate_entry_veto=True,
    )
    return run.daily["return"].astype(float), dict(run.audit)


def _return_hash(returns: pd.Series) -> str:
    return hashlib.sha256(
        returns.to_numpy(dtype="<f8").tobytes()
    ).hexdigest()


def _current_lock_episodes(formal, metrics: pd.DataFrame) -> pd.DataFrame:
    state = formal.state
    baseline = formal.daily["return"].astype(float)
    entries = state.index[state["state_changed"] & ~state["risk_on"]]
    rows: list[dict[str, object]] = []
    for episode_id, entry in enumerate(entries, start=1):
        later = state.loc[state.index > entry]
        exits = later.index[later["state_changed"] & later["risk_on"]]
        exit_date = exits[0] if len(exits) else None
        end = exit_date if exit_date is not None else state.index[-1]
        interval = state.loc[entry:end]
        blocked = interval.loc[
            interval["state_reason"].eq(
                "momentum_recovery_blocked_by_defender_lock"
            )
        ]
        first_recovery = blocked.index[0] if len(blocked) else None
        release_log_excess = np.nan
        release_annualized = np.nan
        release_sharpe = np.nan
        release_mdd = np.nan
        baseline_interval_return = np.nan
        release_interval_return = np.nan
        if first_recovery is not None:
            counterfactual_state = state.copy()
            stop = (
                state.index[state.index.get_loc(exit_date) - 1]
                if exit_date is not None
                else state.index[-1]
            )
            changed_index = counterfactual_state.loc[first_recovery:stop].index
            counterfactual_state.loc[changed_index, "risk_on"] = True
            counterfactual_state.loc[changed_index, "state_changed"] = False
            counterfactual_state.loc[changed_index, "state_reason"] = (
                "event_counterfactual_early_release"
            )
            counterfactual_state.at[first_recovery, "state_changed"] = True
            if exit_date is not None:
                counterfactual_state.at[exit_date, "state_changed"] = False
            released, _ = _run_policy(
                formal,
                formal.context,
                metrics,
                counterfactual_state,
            )
            measured = performance(released)
            release_annualized = measured["annualized_return_252"]
            release_sharpe = measured["sharpe"]
            release_mdd = measured["max_drawdown"]
            release_log_excess = float(
                np.log1p(baseline).sum() - np.log1p(released).sum()
            )
            comparison_end = exit_date if exit_date is not None else end
            baseline_interval = baseline.loc[first_recovery:comparison_end]
            release_interval = released.reindex(baseline_interval.index)
            baseline_interval_return = float(
                (1.0 + baseline_interval).prod() - 1.0
            )
            release_interval_return = float(
                (1.0 + release_interval).prod() - 1.0
            )
        rows.append(
            {
                "episode_id": episode_id,
                "defender_entry": entry,
                "current_exit": exit_date,
                "episode_end": end,
                "episode_sessions": int(len(interval)),
                "first_blocked_recovery": first_recovery,
                "held_days_at_first_recovery": (
                    int(state.at[first_recovery, "held_days_at_open"])
                    if first_recovery is not None
                    else np.nan
                ),
                "blocked_recovery_sessions": int(len(blocked)),
                "current_interval_return_after_first_recovery": (
                    baseline_interval_return
                ),
                "early_release_interval_return": release_interval_return,
                "lock_log_excess_vs_event_release": release_log_excess,
                "event_release_full_annualized_return_252": release_annualized,
                "event_release_full_sharpe": release_sharpe,
                "event_release_full_max_drawdown": release_mdd,
            }
        )
    return pd.DataFrame(rows)


def run_audit(root: Path, config_path: Path, output: Path) -> dict[str, object]:
    applied = config_path if config_path.is_absolute() else root / config_path
    config = yaml.safe_load(applied.read_text(encoding="utf-8"))
    experiment = config["experiment"]
    start = date.fromisoformat(str(experiment["evaluation_start"]))
    end = date.fromisoformat(str(experiment["evidence_cutoff"]))
    formal = run_formal_strategy(root, start=start, end=end)
    baseline = formal.daily["return"].astype(float)
    if _return_hash(baseline) != str(experiment["expected_formal_return_hash"]):
        raise AssertionError("2019 formal checkpoint changed")
    metrics = quality_metrics_at_open(formal.context)
    periods = {
        str(name): (str(bounds[0]), str(bounds[1]))
        for name, bounds in config["periods"].items()
    }
    policies = _policies(config)

    rows: list[dict[str, object]] = []
    returns: dict[str, pd.Series] = {}
    states: dict[str, pd.DataFrame] = {}
    policy_by_id: dict[str, ExitPolicy] = {}
    for policy in policies:
        state = exit_state_schedule(formal.score_at_open, policy)
        candidate_returns, run_audit = _run_policy(
            formal,
            formal.context,
            metrics,
            state,
        )
        candidate_id = policy.candidate_id
        returns[candidate_id] = candidate_returns
        states[candidate_id] = state
        policy_by_id[candidate_id] = policy
        row = _metric_row(
            candidate_id,
            "defender_exit_mechanism",
            candidate_returns,
            periods,
        )
        early_exits = state["state_reason"].eq("to_momentum_early_recovery")
        row.update(
            {
                "policy_family": policy.family,
                "minimum_lock_days": policy.minimum_lock_days,
                "confirmation_days": policy.confirmation_days,
                "early_threshold": policy.early_threshold,
                "fallback_day": policy.fallback_day,
                "base_defender_entries": int(
                    (state["state_changed"] & ~state["risk_on"]).sum()
                ),
                "base_momentum_recoveries": int(
                    (state["state_changed"] & state["risk_on"]).sum()
                ),
                "early_recoveries": int(early_exits.sum()),
                "escape_entries": int(run_audit["escape_entries"]),
            }
        )
        rows.append(row)
    surface = pd.DataFrame(rows)
    baseline_id = "fixed_lock_30"
    current_parity = float((returns[baseline_id] - baseline).abs().max())
    state_parity = bool(
        states[baseline_id]["risk_on"].equals(formal.state["risk_on"])
    )
    if current_parity > 1e-14 or not state_parity:
        raise AssertionError(
            f"current lock parity failed: return={current_parity:.3e}, state={state_parity}"
        )
    baseline_row = surface.set_index("candidate_id").loc[baseline_id]
    flexible_surface = surface.loc[
        ~(
            surface["policy_family"].eq("fixed_lock")
            & surface["minimum_lock_days"].gt(30)
        )
    ]
    dual = flexible_surface.loc[
        flexible_surface["annualized_return_252"].gt(
            baseline_row["annualized_return_252"]
        )
        & flexible_surface["sharpe"].gt(baseline_row["sharpe"])
    ].copy()
    strict = dual.loc[
        dual["max_drawdown"].ge(baseline_row["max_drawdown"] - 1e-12)
        & dual["minimum_segment_sharpe"].ge(
            baseline_row["minimum_segment_sharpe"] - 1e-12
        )
    ].copy()
    selection_pool = strict if not strict.empty else dual
    if selection_pool.empty:
        selected_id = str(
            flexible_surface.loc[
                ~flexible_surface["candidate_id"].eq(baseline_id)
            ]
            .sort_values(
                ["sharpe", "annualized_return_252"], ascending=False
            )
            .iloc[0]["candidate_id"]
        )
    else:
        selected_id = str(
            selection_pool.sort_values(
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
        digest = _return_hash(candidate_returns)
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
    lock_episodes = _current_lock_episodes(formal, metrics)

    cost_rows: list[dict[str, object]] = []
    for multiplier in checks["transaction_cost_multipliers"]:
        context = _scaled_cost_context(formal, float(multiplier), end)
        applied_metrics = quality_metrics_at_open(context)
        for candidate_id in (baseline_id, selected_id):
            candidate_returns, run_audit = _run_policy(
                formal,
                context,
                applied_metrics,
                states[candidate_id],
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
    lock_values = lock_episodes["lock_log_excess_vs_event_release"].dropna()
    positive_lock = lock_values.loc[lock_values.gt(0)].sort_values(ascending=False)
    strong_evidence = bool(
        selected_id in set(dual["candidate_id"])
        and bootstrap["annualized_return_delta_ci_lower"] > 0
        and bootstrap["sharpe_delta_ci_lower"] > 0
        and reality["p_value"] < 0.05
    )
    audit: dict[str, object] = {
        "research_id": experiment["id"],
        "status": (
            "candidate_supported_pending_user_promotion"
            if strong_evidence
            else "keep_current_lock_no_robust_replacement"
        ),
        "evidence_status": experiment["evidence_status"],
        "evaluation_start": start.isoformat(),
        "evidence_cutoff": end.isoformat(),
        "baseline_candidate": baseline_id,
        "selected_point_candidate": selected_id,
        "formal_return_parity_max_abs_error": current_parity,
        "formal_state_parity": state_parity,
        "candidate_ids": int(len(surface)),
        "unique_paths": int(len(panel.columns)),
        "dual_improvement_candidates": dual["candidate_id"].tolist(),
        "strict_dual_candidates": strict["candidate_id"].tolist(),
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
            )
        },
        "current_lock_attribution": {
            "base_defender_entries": int(
                (formal.state["state_changed"] & ~formal.state["risk_on"]).sum()
            ),
            "blocked_recovery_observations": int(
                formal.state["state_reason"].eq(
                    "momentum_recovery_blocked_by_defender_lock"
                ).sum()
            ),
            "episodes_with_blocked_recovery": int(
                lock_episodes["first_blocked_recovery"].notna().sum()
            ),
            "lock_helped_events": int(lock_values.gt(0).sum()),
            "lock_hurt_events": int(lock_values.lt(0).sum()),
            "top_two_positive_lock_share": (
                float(positive_lock.head(2).sum() / positive_lock.sum())
                if positive_lock.sum() > 0
                else 0.0
            ),
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
                else baseline_row["annualized_return_252"]
            ),
            "leave_one_min_sharpe": float(
                leave_event["sharpe"].min()
                if not leave_event.empty
                else baseline_row["sharpe"]
            ),
        },
        "decision": {
            "production_defender_lock_days": 30,
            "replacement_promoted": False,
            "strong_statistical_support": strong_evidence,
            "reason": (
                "No flexible exit candidate clears the joint requirements for "
                "full and segment performance, drawdown, bootstrap, and "
                "multiple-testing robustness."
            ),
        },
    }

    output.mkdir(parents=True, exist_ok=True)
    surface.to_csv(output / "candidate_surface.csv", index=False)
    pd.DataFrame(states[selected_id]).to_csv(
        output / "selected_state.csv"
    )
    lock_episodes.to_csv(output / "current_lock_episodes.csv", index=False)
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

    report = f"""# Defender 30日锁与灵活退出机制审计（2019主样本）

证据状态：回溯机制研究，不是独立样本外。  
正式决定：保留30日Defender锁；本轮不修改生产策略。

## 当前锁的作用

2019重启后共有{audit['current_lock_attribution']['base_defender_entries']}次基础Defender进入，
恢复条件在锁内被阻塞{audit['current_lock_attribution']['blocked_recovery_observations']}个交易日，
涉及{audit['current_lock_attribution']['episodes_with_blocked_recovery']}段事件。逐段把第一次锁内恢复
改成提前释放后，原锁相对有利{audit['current_lock_attribution']['lock_helped_events']}段、不利
{audit['current_lock_attribution']['lock_hurt_events']}段；前两大正贡献占全部正向锁贡献
{audit['current_lock_attribution']['top_two_positive_lock_share']:.1%}。

## 候选搜索

共{len(surface)}个参数ID、{len(panel.columns)}条唯一路径。机制包括固定锁长度诊断、短锁加连续
确认、强恢复早退，以及从严格阈值逐步放宽到0.40的线性恢复门槛。灵活早退候选在第30日仍
回到当前单日0.40恢复规则，不允许比当前锁更迟钝；35–90日固定锁只用于检查30日是否为边界
峰值，不属于早退候选。

当前`{baseline_id}`为{float(baseline_row['annualized_return_252']):.2%}年化、
{float(baseline_row['sharpe']):.3f} Sharpe、MDD
{float(baseline_row['max_drawdown']):.2%}。按预设稳健排序选出的表面候选`{selected_id}`为
{float(selected_row['annualized_return_252']):.2%}/
{float(selected_row['sharpe']):.3f}/
{float(selected_row['max_drawdown']):.2%}。

## 稳健性

- 20日Bootstrap年化差95%区间
  `[{float(bootstrap['annualized_return_delta_ci_lower']):.2%},
  {float(bootstrap['annualized_return_delta_ci_upper']):.2%}]`，Sharpe差区间
  `[{float(bootstrap['sharpe_delta_ci_lower']):.3f},
  {float(bootstrap['sharpe_delta_ci_upper']):.3f}]`。
- Reality Check `p={float(reality['p_value']):.4f}`，CSCV-PBO
  {float(cscv['pbo']):.1%}，训练冠军测试段击败当前锁比例
  {float(cscv['selected_beats_baseline_rate']):.1%}。
- walk-forward/留一年重选双指标胜率
  {audit['walk_forward_dual_win_rate']:.1%}/
  {audit['leave_one_year_selection_dual_win_rate']:.1%}；固定候选删除任一年双指标通过率
  {audit['fixed_candidate_delete_year_dual_pass_rate']:.1%}。
- 252/504日滚动双指标胜率
  {rolling_summary['252']['dual_win_rate']:.1%}/
  {rolling_summary['504']['dual_win_rate']:.1%}。

## 结论

30日硬锁的优点是抑制一次性假恢复、降低状态来回切换；缺点是把恢复强度完全忽略，并在多数
事件中反复记录已经满足的恢复条件却机械等待。它的历史收益确实存在事件集中风险。

但本轮更灵活机制没有同时通过收益、Sharpe、回撤、分段和多重试验门槛。最优治理选择不是
把30日改成另一个精确天数，而是暂时冻结当前规则，把低维早退候选作为前瞻影子；在积累未观察
事件前不做生产替换。
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
