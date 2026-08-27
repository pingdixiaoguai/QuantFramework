"""Run the preregistered Occam peak-escape study for the formal W40 path."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from research.generate_strategy_drawdown_badcases import (
    distinct_drawdown_episodes,
)
from research.momentum_defender_gold_override import simulate_candidate_schedule
from research.momentum_defender_gold_override_overfit import (
    cscv_pbo,
    expanding_walk_forward,
    leave_one_year_selection,
    paired_block_bootstrap,
    yearly_reality_check,
)
from research.momentum_defender_occam import (
    ENTER_RETURN,
    ENTRY_COST,
    EXIT_COST,
    EXIT_RETURN,
    HELD_RETURN,
    INTERNAL_COST,
    performance,
)
from research.momentum_defender_peak_escape_occam import (
    PeakEscapeParams,
    build_peak_escape_features,
    collect_peak_escape_returns,
    same_window_top_drawdowns,
    top_drawdown_summary,
)
from research.standard_report import generate_standard_report
from strategy.momentum_defender_w40_loss import (
    FORMAL_STRATEGY_ID,
    run_formal_strategy,
)


DEFAULT_CONFIG = Path(
    "research/configs/momentum_defender_peak_escape_occam.yaml"
)
DEFAULT_OUTPUT = Path(
    "experiments/20260825_momentum_defender_peak_escape_occam"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _return_path_hash(values: pd.Series) -> str:
    return hashlib.sha256(values.to_numpy(np.float64).tobytes()).hexdigest()


def _params(row: pd.Series | dict[str, object]) -> PeakEscapeParams:
    return PeakEscapeParams(
        policy=str(row["policy"]),
        price_return_threshold=float(row["price_return_threshold"]),
        volume_ratio_threshold=float(row["volume_ratio_threshold"]),
        fund_share_flow_threshold=float(row["fund_share_flow_threshold"]),
        min_escape_hold_days=int(row["min_escape_hold_days"]),
    )


def _period_slice(
    values: pd.Series | pd.DataFrame,
    bounds: list[str],
) -> pd.Series | pd.DataFrame:
    return values.loc[pd.Timestamp(bounds[0]) : pd.Timestamp(bounds[1])]


def _metrics_for_returns(
    candidate: pd.Series,
    baseline: pd.Series,
    *,
    top_n: int,
) -> dict[str, float | int]:
    candidate_performance = performance(candidate.astype(float))
    baseline_performance = performance(baseline.astype(float))
    candidate_top, _ = top_drawdown_summary(candidate, top_n=top_n)
    baseline_top, _ = top_drawdown_summary(baseline, top_n=top_n)
    return {
        "observations": int(len(candidate)),
        "annualized_return_252": float(candidate_performance["annualized_return_252"]),
        "sharpe": float(candidate_performance["sharpe"]),
        "max_drawdown": float(candidate_performance["max_drawdown"]),
        "top_drawdown_count": int(candidate_top["top_drawdown_count"]),
        "top20_mean_drawdown": float(candidate_top["top_mean_drawdown"]),
        "top20_worst_drawdown": float(candidate_top["top_worst_drawdown"]),
        "delta_annualized_return_252": float(
            candidate_performance["annualized_return_252"]
            - baseline_performance["annualized_return_252"]
        ),
        "delta_sharpe": float(
            candidate_performance["sharpe"] - baseline_performance["sharpe"]
        ),
        "delta_max_drawdown": float(
            candidate_performance["max_drawdown"]
            - baseline_performance["max_drawdown"]
        ),
        "delta_top20_mean_drawdown": float(
            candidate_top["top_mean_drawdown"]
            - baseline_top["top_mean_drawdown"]
        ),
        "delta_top20_worst_drawdown": float(
            candidate_top["top_worst_drawdown"]
            - baseline_top["top_worst_drawdown"]
        ),
    }


def _candidate_metrics(
    returns: pd.DataFrame,
    baseline: pd.Series,
    metadata: pd.DataFrame,
    periods: dict[str, list[str]],
    *,
    top_n: int,
) -> pd.DataFrame:
    rows = []
    baseline_full_top, baseline_episodes = top_drawdown_summary(
        baseline, top_n=top_n
    )
    for candidate_id in returns.columns:
        row: dict[str, object] = {
            "candidate_id": candidate_id,
            **metadata.loc[candidate_id].to_dict(),
        }
        for label, bounds in periods.items():
            candidate_period = _period_slice(returns[candidate_id], bounds)
            baseline_period = _period_slice(baseline, bounds)
            metrics = _metrics_for_returns(
                candidate_period,
                baseline_period,
                top_n=top_n,
            )
            prefix = "" if label == "full" else f"{label}_"
            row.update({f"{prefix}{key}": value for key, value in metrics.items()})
        same_window = same_window_top_drawdowns(
            returns[candidate_id], baseline_episodes
        )
        row["baseline_top20_same_window_wins"] = int(
            same_window["candidate_improved"].sum()
        )
        row["baseline_top20_same_window_mean_improvement"] = float(
            same_window["improvement"].mean()
        )
        row["baseline_top20_mean_drawdown"] = float(
            baseline_full_top["top_mean_drawdown"]
        )
        rows.append(row)
    return pd.DataFrame(rows).set_index("candidate_id")


def _select_candidate(
    metrics: pd.DataFrame,
    returns: pd.DataFrame,
    baseline: pd.Series,
    config: dict,
) -> tuple[pd.Series, pd.DataFrame, str]:
    selection = config["selection"]
    ranked_source = metrics.copy()
    development_start = config["periods"]["development"][0]
    validation_end = config["periods"]["validation"][1]
    dv_baseline = baseline.loc[
        pd.Timestamp(development_start) : pd.Timestamp(validation_end)
    ]
    dv_base_top, _ = top_drawdown_summary(
        dv_baseline, top_n=int(config["objective"]["top_n"])
    )
    for candidate_id in ranked_source.index:
        dv_candidate = returns.loc[dv_baseline.index, candidate_id]
        dv_top, _ = top_drawdown_summary(
            dv_candidate, top_n=int(config["objective"]["top_n"])
        )
        ranked_source.at[
            candidate_id, "development_validation_top20_mean_improvement"
        ] = (
            float(dv_top["top_mean_drawdown"])
            - float(dv_base_top["top_mean_drawdown"])
        )
    ranked_source["worst_development_validation_top20_improvement"] = ranked_source[
        [
            "development_delta_top20_mean_drawdown",
            "validation_delta_top20_mean_drawdown",
        ]
    ].min(axis=1)
    ranked_source["worst_development_validation_sharpe_delta"] = ranked_source[
        ["development_delta_sharpe", "validation_delta_sharpe"]
    ].min(axis=1)
    ranked_source["policy_complexity"] = ranked_source["policy"].map(
        {"price_volume": 0, "price_crowding": 1}
    )
    ranked_source["candidate_id_tiebreak"] = ranked_source.index.astype(str)
    eligible = ranked_source.loc[
        metrics["escape_entries"].ge(int(selection["minimum_escape_entries"]))
        & metrics["development_delta_annualized_return_252"].ge(
            float(selection["annualized_return_delta_floor"])
        )
        & metrics["validation_delta_annualized_return_252"].ge(
            float(selection["annualized_return_delta_floor"])
        )
        & metrics["development_delta_sharpe"].ge(
            float(selection["sharpe_delta_floor"])
        )
        & metrics["validation_delta_sharpe"].ge(
            float(selection["sharpe_delta_floor"])
        )
        & metrics["development_delta_top20_mean_drawdown"].ge(
            float(selection["top20_mean_drawdown_delta_floor"])
        )
        & metrics["validation_delta_top20_mean_drawdown"].ge(
            float(selection["top20_mean_drawdown_delta_floor"])
        )
    ].copy()
    selection_status = (
        "eligible_selected"
        if not eligible.empty
        else "no_eligible_candidate_diagnostic_leader_only"
    )
    pool = eligible if not eligible.empty else ranked_source
    ranked = pool.sort_values(
        [
            "development_validation_top20_mean_improvement",
            "worst_development_validation_top20_improvement",
            "worst_development_validation_sharpe_delta",
            "escape_days",
            "policy_complexity",
            "candidate_id_tiebreak",
        ],
        ascending=[False, False, False, True, True, True],
    )
    return ranked.iloc[0], eligible.sort_values(
        "development_validation_top20_mean_improvement", ascending=False
    ), selection_status


def _escape_event_windows(state: pd.DataFrame) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    active = state["peak_escape_active"].astype(bool)
    groups = active.ne(active.shift(fill_value=False)).cumsum()
    calendar = pd.DatetimeIndex(state.index)
    windows = []
    for _, sample in state.loc[active].groupby(groups.loc[active]):
        start = pd.Timestamp(sample.index[0])
        last = pd.Timestamp(sample.index[-1])
        position = calendar.get_loc(last)
        end = pd.Timestamp(
            calendar[position if position == len(calendar) - 1 else position + 1]
        )
        windows.append((start, end))
    return windows


def _event_attribution(
    selected_run,
    baseline: pd.Series,
) -> pd.DataFrame:
    rows = []
    for event_id, (start, end) in enumerate(
        _escape_event_windows(selected_run.state), start=1
    ):
        interval = selected_run.daily.loc[start:end].index
        candidate_return = float(
            (1.0 + selected_run.daily.loc[interval, "return"]).prod() - 1.0
        )
        baseline_return = float((1.0 + baseline.loc[interval]).prod() - 1.0)
        entry = selected_run.state.loc[start]
        rows.append(
            {
                "event_id": event_id,
                "start": start,
                "end_including_exit_open": end,
                "observations": int(len(interval)),
                "formal_candidate_at_entry": entry["formal_requested_candidate"],
                "price_breakout_at_open": entry["price_breakout_at_open"],
                "price_return20_at_open": entry["price_return20_at_open"],
                "volume_ratio20_at_open": entry["volume_ratio20_at_open"],
                "adjusted_share_flow20_at_open": entry[
                    "adjusted_share_flow20_at_open"
                ],
                "price_flag": bool(entry["price_flag"]),
                "volume_flag": bool(entry["volume_flag"]),
                "scale_flag": bool(entry["scale_flag"]),
                "candidate_return": candidate_return,
                "baseline_return": baseline_return,
                "arithmetic_improvement": candidate_return - baseline_return,
                "log_excess": float(
                    np.log1p(candidate_return) - np.log1p(baseline_return)
                ),
            }
        )
    return pd.DataFrame(rows)


def _leave_one_event(
    selected: pd.Series,
    baseline: pd.Series,
    events: pd.DataFrame,
    *,
    top_n: int,
) -> pd.DataFrame:
    rows = []
    for _, event in events.iterrows():
        counterfactual = selected.copy()
        start = pd.Timestamp(event["start"])
        end = pd.Timestamp(event["end_including_exit_open"])
        counterfactual.loc[start:end] = baseline.loc[start:end]
        metrics = _metrics_for_returns(counterfactual, baseline, top_n=top_n)
        rows.append(
            {
                "removed_event_id": int(event["event_id"]),
                "removed_start": start,
                "removed_end": end,
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def _top_mean_numpy(values: np.ndarray, top_n: int) -> float:
    nav = 1.0
    peak = 1.0
    trough = 0.0
    active = False
    depths = []
    for value in values:
        nav *= 1.0 + float(value)
        if nav >= peak:
            if active:
                depths.append(trough)
                active = False
            peak = nav
            trough = 0.0
        else:
            drawdown = nav / peak - 1.0
            if not active:
                active = True
                trough = drawdown
            elif drawdown < trough:
                trough = drawdown
    if active:
        depths.append(trough)
    return float(np.mean(sorted(depths)[:top_n])) if depths else 0.0


def _paired_top20_bootstrap(
    candidate: pd.Series,
    baseline: pd.Series,
    *,
    top_n: int,
    block_size: int,
    repetitions: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    rng = np.random.default_rng(seed)
    candidate_values = candidate.to_numpy(float)
    baseline_values = baseline.to_numpy(float)
    observations = len(candidate_values)
    block_count = int(np.ceil(observations / block_size))
    deltas = np.empty(repetitions, dtype=float)
    for repetition in range(repetitions):
        starts = rng.integers(0, observations, size=block_count)
        indices = np.concatenate(
            [
                np.arange(start, start + block_size) % observations
                for start in starts
            ]
        )[:observations]
        deltas[repetition] = _top_mean_numpy(
            candidate_values[indices], top_n
        ) - _top_mean_numpy(baseline_values[indices], top_n)
    frame = pd.DataFrame(
        {
            "repetition": np.arange(1, repetitions + 1),
            "top20_mean_drawdown_delta": deltas,
        }
    )
    return frame, {
        "top_n": top_n,
        "block_size": block_size,
        "repetitions": repetitions,
        "seed": seed,
        "mean": float(frame["top20_mean_drawdown_delta"].mean()),
        "ci_lower": float(frame["top20_mean_drawdown_delta"].quantile(0.025)),
        "ci_upper": float(frame["top20_mean_drawdown_delta"].quantile(0.975)),
        "positive_probability": float(
            frame["top20_mean_drawdown_delta"].gt(0.0).mean()
        ),
    }


def _primary_walk_forward(
    returns: pd.DataFrame,
    baseline: pd.Series,
    *,
    top_n: int,
) -> pd.DataFrame:
    years = sorted(returns.index.year.unique())
    rows = []
    for position in range(3, len(years)):
        train_years = years[:position]
        test_year = years[position]
        train = returns.loc[returns.index.year.isin(train_years)]
        train_scores = {
            candidate: top_drawdown_summary(train[candidate], top_n=top_n)[0][
                "top_mean_drawdown"
            ]
            for candidate in train.columns
        }
        winner = max(train_scores, key=train_scores.get)
        test = returns.loc[returns.index.year == test_year, winner]
        base_test = baseline.loc[baseline.index.year == test_year]
        metrics = _metrics_for_returns(test, base_test, top_n=top_n)
        rows.append(
            {
                "test_year": int(test_year),
                "train_years": ",".join(map(str, train_years)),
                "selected_candidate": winner,
                "train_top20_mean_drawdown": float(train_scores[winner]),
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def _primary_leave_one_year(
    returns: pd.DataFrame,
    baseline: pd.Series,
    *,
    top_n: int,
) -> pd.DataFrame:
    rows = []
    for held_year in sorted(returns.index.year.unique()):
        train = returns.loc[returns.index.year != held_year]
        scores = {
            candidate: top_drawdown_summary(train[candidate], top_n=top_n)[0][
                "top_mean_drawdown"
            ]
            for candidate in train.columns
        }
        winner = max(scores, key=scores.get)
        test = returns.loc[returns.index.year == held_year, winner]
        base_test = baseline.loc[baseline.index.year == held_year]
        metrics = _metrics_for_returns(test, base_test, top_n=top_n)
        rows.append(
            {
                "held_year": int(held_year),
                "selected_candidate": winner,
                "train_top20_mean_drawdown": float(scores[winner]),
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def _friction_stress(
    context,
    selected_target: pd.Series,
    baseline_target: pd.Series,
    *,
    top_n: int,
) -> pd.DataFrame:
    rows = []
    for multiplier in (1.0, 3.0, 5.0):
        interfaces = {
            candidate: _scale_interface_net_costs(frame, multiplier)
            for candidate, frame in context.interfaces.items()
        }
        candidate = simulate_candidate_schedule(
            selected_target, interfaces, context.initial_previous_candidate
        )["return"].astype(float)
        baseline = simulate_candidate_schedule(
            baseline_target, interfaces, context.initial_previous_candidate
        )["return"].astype(float)
        rows.append(
            {
                "cost_multiplier": multiplier,
                **_metrics_for_returns(candidate, baseline, top_n=top_n),
            }
        )
    return pd.DataFrame(rows)


def _scale_interface_net_costs(
    frame: pd.DataFrame,
    multiplier: float,
) -> pd.DataFrame:
    """Scale exact net candidate legs without requiring stored gross columns."""

    if multiplier < 0.0:
        raise ValueError("cost multiplier cannot be negative")
    required = {
        HELD_RETURN,
        ENTER_RETURN,
        EXIT_RETURN,
        INTERNAL_COST,
        ENTRY_COST,
        EXIT_COST,
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"candidate interface missing net cost fields: {sorted(missing)}")
    stressed = frame.copy()
    for return_column, cost_column in (
        (HELD_RETURN, INTERNAL_COST),
        (ENTER_RETURN, ENTRY_COST),
        (EXIT_RETURN, EXIT_COST),
    ):
        returns = frame[return_column].astype(float)
        base_cost = frame[cost_column].astype(float)
        stressed_cost = base_cost * multiplier
        if stressed_cost.dropna().ge(1.0).any():
            raise ValueError("stressed cost rate must remain below 100%")
        gross_factor = (1.0 + returns) / (1.0 - base_cost)
        stressed[return_column] = gross_factor * (1.0 - stressed_cost) - 1.0
        stressed[cost_column] = stressed_cost
    return stressed


def _delay_stress(
    context,
    selected_state: pd.DataFrame,
    baseline_target: pd.Series,
    baseline_return: pd.Series,
    *,
    top_n: int,
) -> pd.DataFrame:
    rows = []
    for delay in (0, 1, 2):
        active = (
            selected_state["peak_escape_active"]
            .shift(delay)
            .fillna(False)
            .astype(bool)
        )
        target = baseline_target.copy()
        target.loc[active] = "DEFENDER"
        candidate = simulate_candidate_schedule(
            target, context.interfaces, context.initial_previous_candidate
        )["return"].astype(float)
        rows.append(
            {
                "additional_execution_delay_sessions": delay,
                **_metrics_for_returns(candidate, baseline_return, top_n=top_n),
            }
        )
    return pd.DataFrame(rows)


def _selected_neighborhood(
    selected: pd.Series,
    metrics: pd.DataFrame,
) -> pd.DataFrame:
    policy = str(selected["policy"])
    center_r = float(selected["price_return_threshold"])
    center_v = float(selected["volume_ratio_threshold"])
    center_h = int(selected["min_escape_hold_days"])
    neighborhood = metrics.loc[
        metrics["policy"].eq(policy)
        & metrics["price_return_threshold"].sub(center_r).abs().le(0.05 + 1e-12)
        & metrics["volume_ratio_threshold"].sub(center_v).abs().le(0.50 + 1e-12)
        & metrics["min_escape_hold_days"].sub(center_h).abs().le(5)
    ].copy()
    neighborhood["full_gate_direction"] = (
        neighborhood["delta_top20_mean_drawdown"].gt(0.0)
        & neighborhood["delta_max_drawdown"].ge(0.0)
        & neighborhood["delta_annualized_return_252"].ge(-0.03)
        & neighborhood["delta_sharpe"].ge(-0.05)
    )
    return neighborhood


def _render_report(
    config: dict,
    baseline_metrics: dict,
    selected: pd.Series,
    selected_id: str,
    events: pd.DataFrame,
    same_window: pd.DataFrame,
    neighborhood: pd.DataFrame,
    bootstrap_summary: dict,
    top_bootstrap_summary: dict,
    cscv_summary: dict,
    reality: dict,
    primary_walk: pd.DataFrame,
    primary_leave_year: pd.DataFrame,
    friction: pd.DataFrame,
    delay: pd.DataFrame,
    gates: dict[str, bool],
    unique_paths: int,
    selection_status: str,
) -> str:
    passed = sum(gates.values())
    all_passed = all(gates.values())
    if selection_status != "eligible_selected":
        decision = (
            "预注册development/validation资格池为空。下列路径只是按原排序规则展示的诊断"
            "领先者，不是可选择候选，不建立shadow或生产替换建议。"
        )
    else:
        decision = (
            "全部机械Gate通过，但证据仍是看过Top回撤后的回溯研究，只允许冻结shadow观察。"
            if all_passed
            else "至少一个机械Gate失败，保留为研究证据，不建立生产替换建议。"
        )
    return f"""# 价格×量能×基金份额：奥卡姆逃顶研究

- 日期：{config['experiment']['created_on']}
- 基线：`{config['experiment']['baseline_strategy']}`
- 证据：{config['experiment']['evidence_status']}
- 候选：{len(config['grid']['policy']) * len(config['grid']['price_return_threshold']) * len(config['grid']['volume_ratio_threshold']) * len(config['grid']['fund_share_flow_threshold']) * len(config['grid']['min_escape_hold_days'])}个ID，{unique_paths}条唯一收益路径

## 结论

选择状态：`{selection_status}`。诊断路径为`{selected_id}`：价格必须突破严格滞后200日高点且20日涨幅不低于
{float(selected['price_return_threshold']):.0%}，成交量不低于此前20日中位数的
{float(selected['volume_ratio_threshold']):.2f}倍；触发后下一开盘临时切Defender，至少持有
{int(selected['min_escape_hold_days'])}日，原条件消失后恢复正式路径。

完整样本Top20平均回撤从{float(baseline_metrics['top20_mean_drawdown']):.2%}改善至
{float(selected['top20_mean_drawdown']):.2%}，收窄{float(selected['delta_top20_mean_drawdown']) * 100:.2f}个百分点；
最大回撤从{float(baseline_metrics['max_drawdown']):.2%}改善至{float(selected['max_drawdown']):.2%}。
年化从{float(baseline_metrics['annualized_return_252']):.2%}变为{float(selected['annualized_return_252']):.2%}，
Sharpe从{float(baseline_metrics['sharpe']):.3f}变为{float(selected['sharpe']):.3f}。

{decision}

## 奥卡姆结构与基金规模结论

只测试两个布尔规则族、3个价格门槛、2个量比、固定5%拆分调整后份额增长门槛和2个持有期，
没有搜索因子权重、资产专用阈值、机器学习模型或逐事件例外。`price_volume`只要求价格+量能；
`price_crowding`允许基金份额增长替代量能证据。诊断路径的规则族为`{selected['policy']}`。

诊断路径全样本逃顶{int(selected['escape_entries'])}次、{int(selected['escape_days'])}日，其中份额增长在量能未达标时
独立增加的入场为{int(selected['entry_scale_without_volume_count'])}次。若该数字为0，说明基金份额在本候选族中
没有提供增量，应按奥卡姆原则从最终规则删除，而不是为了“使用三类数据”强行保留。

## 分段结果

|区间|年化|年化Δ|Sharpe|Sharpe Δ|MDD|Top20均值|Top20改善|
|---|---:|---:|---:|---:|---:|---:|---:|
|development|{selected['development_annualized_return_252']:.2%}|{selected['development_delta_annualized_return_252']:+.2%}|{selected['development_sharpe']:.3f}|{selected['development_delta_sharpe']:+.3f}|{selected['development_max_drawdown']:.2%}|{selected['development_top20_mean_drawdown']:.2%}|{selected['development_delta_top20_mean_drawdown']:+.2%}|
|validation|{selected['validation_annualized_return_252']:.2%}|{selected['validation_delta_annualized_return_252']:+.2%}|{selected['validation_sharpe']:.3f}|{selected['validation_delta_sharpe']:+.3f}|{selected['validation_max_drawdown']:.2%}|{selected['validation_top20_mean_drawdown']:.2%}|{selected['validation_delta_top20_mean_drawdown']:+.2%}|
|recent|{selected['recent_annualized_return_252']:.2%}|{selected['recent_delta_annualized_return_252']:+.2%}|{selected['recent_sharpe']:.3f}|{selected['recent_delta_sharpe']:+.3f}|{selected['recent_max_drawdown']:.2%}|{selected['recent_top20_mean_drawdown']:.2%}|{selected['recent_delta_top20_mean_drawdown']:+.2%}|
|full|{selected['annualized_return_252']:.2%}|{selected['delta_annualized_return_252']:+.2%}|{selected['sharpe']:.3f}|{selected['delta_sharpe']:+.3f}|{selected['max_drawdown']:.2%}|{selected['top20_mean_drawdown']:.2%}|{selected['delta_top20_mean_drawdown']:+.2%}|

## 稳健性与过拟合边界

- 诊断规则共有{len(events)}个逃顶事件，正贡献{int(events['arithmetic_improvement'].gt(0).sum())}个、负贡献{int(events['arithmetic_improvement'].lt(0).sum())}个；基准Top20同窗口改善{int(same_window['candidate_improved'].sum())}/20。
- 参数邻域{len(neighborhood)}个，维持Top20改善、MDD不恶化且收益/Sharpe不越过容忍线的比例为{float(neighborhood['full_gate_direction'].mean()):.1%}。
- 20日配对bootstrap：Sharpe差95%区间[{float(bootstrap_summary['sharpe_delta_ci_lower']):+.3f}, {float(bootstrap_summary['sharpe_delta_ci_upper']):+.3f}]，为正概率{float(bootstrap_summary['sharpe_delta_positive_probability']):.1%}；Top20均值改善95%区间[{float(top_bootstrap_summary['ci_lower']):+.2%}, {float(top_bootstrap_summary['ci_upper']):+.2%}]，为正概率{float(top_bootstrap_summary['positive_probability']):.1%}。
- CSCV-PBO={float(cscv_summary['pbo']):.1%}，训练选中者测试段击败基线Sharpe比例{float(cscv_summary['selected_beats_baseline_rate']):.1%}；年度Reality Check `p={float(reality['p_value']):.4f}`。
- 按Top20目标的扩展walk-forward测试年改善率{float(primary_walk['delta_top20_mean_drawdown'].gt(0).mean()):.1%}；leave-one-year改善率{float(primary_leave_year['delta_top20_mean_drawdown'].gt(0).mean()):.1%}。
- 删除任一事件后最低Top20改善、最低年化和最低Sharpe另见机器表；若优势依赖单一事件，不能晋升。
- 费用1/3/5倍与信号额外延迟0/1/2日结果分别保存；这些压力测试不参与重新选参。

## Gate

|Gate|通过|
|---|---:|
{chr(10).join(f'|{name}|{value}|' for name, value in gates.items())}

通过{passed}/{len(gates)}。本实验从2026-08-25已有Top回撤诊断出发，`recent`和全样本都不是独立OOS；
即使全部Gate通过也不能直接修改正式策略。机器证据位于
`experiments/20260825_momentum_defender_peak_escape_occam/`。
"""


def run_experiment(root: Path, config_path: Path, output: Path) -> dict[str, object]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config["experiment"]["baseline_strategy"] != FORMAL_STRATEGY_ID:
        raise AssertionError("peak escape baseline is not the formal W40 strategy")
    cutoff = date.fromisoformat(config["data"]["end"])
    formal = run_formal_strategy(root, end=cutoff)
    context = formal.context
    if formal.daily.index.max().date() != cutoff:
        raise AssertionError("formal cutoff does not match peak escape data cutoff")
    baseline = formal.daily["return"].astype(float)
    baseline_target = formal.daily["requested_candidate"].astype(str)
    baseline_replay = simulate_candidate_schedule(
        baseline_target, context.interfaces, context.initial_previous_candidate
    )
    baseline_parity = float(
        (baseline_replay["return"].astype(float) - baseline).abs().max()
    )
    if baseline_parity > 1e-14:
        raise AssertionError(f"formal baseline replay failed: {baseline_parity:.3e}")

    features = build_peak_escape_features(root, context.calendar, end=cutoff)
    metadata, returns, runs = collect_peak_escape_returns(
        context, baseline_target, features, config["grid"]
    )
    top_n = int(config["objective"]["top_n"])
    metrics = _candidate_metrics(
        returns,
        baseline,
        metadata,
        config["periods"],
        top_n=top_n,
    )
    selected, eligible, selection_status = _select_candidate(
        metrics, returns, baseline, config
    )
    selected_id = str(selected.name)
    selected_run = runs[selected_id]
    selected_returns = returns[selected_id]
    baseline_performance = performance(baseline)
    baseline_top, baseline_episodes = top_drawdown_summary(
        baseline, top_n=top_n
    )
    baseline_metrics = {
        **baseline_performance,
        "top20_mean_drawdown": baseline_top["top_mean_drawdown"],
        "top20_worst_drawdown": baseline_top["top_worst_drawdown"],
    }

    events = _event_attribution(selected_run, baseline)
    leave_event = _leave_one_event(
        selected_returns, baseline, events, top_n=top_n
    )
    same_window = same_window_top_drawdowns(
        selected_returns, baseline_episodes
    )
    neighborhood = _selected_neighborhood(selected, metrics)
    validation = config["validation"]
    bootstrap, bootstrap_summary = paired_block_bootstrap(
        selected_returns,
        baseline,
        block_size=int(validation["bootstrap_block_size"]),
        repetitions=int(validation["bootstrap_repetitions"]),
        seed=int(validation["seed"]),
    )
    top_bootstrap, top_bootstrap_summary = _paired_top20_bootstrap(
        selected_returns,
        baseline,
        top_n=top_n,
        block_size=int(validation["bootstrap_block_size"]),
        repetitions=int(validation["bootstrap_repetitions"]),
        seed=int(validation["seed"]),
    )
    cscv, cscv_summary = cscv_pbo(
        returns,
        baseline,
        block_count=int(validation["cscv_blocks"]),
    )
    standard_walk = expanding_walk_forward(returns, baseline)
    standard_leave_year = leave_one_year_selection(returns, baseline)
    primary_walk = _primary_walk_forward(returns, baseline, top_n=top_n)
    primary_leave_year = _primary_leave_one_year(
        returns, baseline, top_n=top_n
    )
    reality = yearly_reality_check(
        returns,
        baseline,
        repetitions=int(validation["reality_check_repetitions"]),
        seed=int(validation["seed"]),
    )
    friction = _friction_stress(
        context,
        selected_run.state["target_candidate"],
        baseline_target,
        top_n=top_n,
    )
    delay = _delay_stress(
        context,
        selected_run.state,
        baseline_target,
        baseline,
        top_n=top_n,
    )

    unique_paths = len({_return_path_hash(returns[column]) for column in returns})
    gates_config = config["final_gates"]
    gates = {
        "preregistered_development_validation_eligibility": selection_status
        == "eligible_selected",
        "full_top20_mean": float(selected["delta_top20_mean_drawdown"])
        >= float(gates_config["full_top20_mean_drawdown_improvement_minimum"]),
        "full_max_drawdown": float(selected["delta_max_drawdown"])
        >= float(gates_config["full_max_drawdown_delta_minimum"]),
        "full_annualized_return": float(selected["delta_annualized_return_252"])
        >= float(gates_config["full_annualized_return_delta_minimum"]),
        "full_sharpe": float(selected["delta_sharpe"])
        >= float(gates_config["full_sharpe_delta_minimum"]),
        "recent_top20_mean": float(selected["recent_delta_top20_mean_drawdown"])
        >= float(gates_config["recent_top20_mean_drawdown_delta_minimum"]),
        "baseline_top20_same_window": int(
            same_window["candidate_improved"].sum()
        )
        >= int(gates_config["baseline_top20_same_window_wins_minimum"]),
        "bootstrap_top20_probability": float(
            top_bootstrap_summary["positive_probability"]
        )
        >= float(gates_config["bootstrap_top20_positive_probability_minimum"]),
    }

    output.mkdir(parents=True, exist_ok=True)
    metrics.sort_values(
        ["delta_top20_mean_drawdown", "sharpe"], ascending=False
    ).to_csv(output / "candidate_metrics.csv")
    eligible.to_csv(output / "eligible_ranked.csv")
    features.coverage.to_csv(output / "feature_coverage.csv")
    selected_run.state.join(
        selected_run.daily, rsuffix="_execution"
    ).to_csv(output / "daily_diagnostic_leader.csv")
    events.to_csv(output / "diagnostic_events.csv", index=False)
    leave_event.to_csv(output / "leave_one_event.csv", index=False)
    same_window.to_csv(output / "baseline_top20_same_window.csv", index=False)
    neighborhood.to_csv(output / "diagnostic_neighborhood.csv")
    bootstrap.to_csv(output / "paired_block_bootstrap.csv", index=False)
    top_bootstrap.to_csv(output / "paired_top20_bootstrap.csv", index=False)
    cscv.to_csv(output / "cscv_pbo.csv", index=False)
    standard_walk.to_csv(output / "standard_expanding_walk_forward.csv", index=False)
    standard_leave_year.to_csv(output / "standard_leave_one_year.csv", index=False)
    primary_walk.to_csv(output / "primary_expanding_walk_forward.csv", index=False)
    primary_leave_year.to_csv(output / "primary_leave_one_year.csv", index=False)
    friction.to_csv(output / "friction_stress.csv", index=False)
    delay.to_csv(output / "execution_delay_stress.csv", index=False)
    baseline_episodes.to_csv(output / "baseline_top20_episodes.csv", index=False)
    selected_top, selected_episodes = top_drawdown_summary(
        selected_returns, top_n=top_n
    )
    selected_episodes.to_csv(
        output / "diagnostic_top20_episodes.csv", index=False
    )
    (output / "strategy_config.yaml").write_text(
        config_path.read_text(encoding="utf-8"), encoding="utf-8"
    )
    generate_standard_report(
        selected_returns,
        baseline,
        "Formal W40 baseline",
        output / "diagnostic_vs_formal.html",
        {
            "strategy_name": selected_id,
            "strategy_mode": "research_peak_escape_occam",
            "evidence_status": config["experiment"]["evidence_status"],
        },
    )

    audit = {
        "status": "passed",
        "experiment": config["experiment"],
        "calendar": {
            "start": baseline.index.min().date().isoformat(),
            "end": baseline.index.max().date().isoformat(),
            "observations": int(len(baseline)),
        },
        "baseline_parity_max_abs_error": baseline_parity,
        "candidate_ids": int(len(returns.columns)),
        "unique_return_paths": unique_paths,
        "baseline": baseline_metrics,
        "selected_candidate": (
            selected_id if selection_status == "eligible_selected" else None
        ),
        "diagnostic_candidate": selected_id,
        "selection_status": selection_status,
        "eligible_candidate_count": int(len(eligible)),
        "diagnostic": selected.to_dict(),
        "diagnostic_top20": selected_top,
        "diagnostic_audit": selected_run.audit,
        "event_summary": {
            "count": int(len(events)),
            "positive": int(events["arithmetic_improvement"].gt(0).sum()),
            "negative": int(events["arithmetic_improvement"].lt(0).sum()),
            "scale_without_volume_entries": int(
                selected["entry_scale_without_volume_count"]
            ),
        },
        "leave_one_event": {
            "minimum_top20_improvement": float(
                leave_event["delta_top20_mean_drawdown"].min()
            ),
            "minimum_annualized_return_delta": float(
                leave_event["delta_annualized_return_252"].min()
            ),
            "minimum_sharpe_delta": float(leave_event["delta_sharpe"].min()),
        },
        "same_window_wins": int(same_window["candidate_improved"].sum()),
        "neighborhood": {
            "count": int(len(neighborhood)),
            "full_gate_direction_rate": float(
                neighborhood["full_gate_direction"].mean()
            ),
        },
        "bootstrap": bootstrap_summary,
        "top20_bootstrap": top_bootstrap_summary,
        "cscv": cscv_summary,
        "reality_check": reality,
        "primary_walk_forward_top20_win_rate": float(
            primary_walk["delta_top20_mean_drawdown"].gt(0).mean()
        ),
        "primary_leave_one_year_top20_win_rate": float(
            primary_leave_year["delta_top20_mean_drawdown"].gt(0).mean()
        ),
        "gates": gates,
        "all_gates_passed": bool(all(gates.values())),
        "evidence_limit": "designed_after_reviewing_formal_top_drawdowns_not_oos",
        "production_decision": "research_only_no_formal_change",
    }
    (output / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    report = _render_report(
        config,
        baseline_metrics,
        selected,
        selected_id,
        events,
        same_window,
        neighborhood,
        bootstrap_summary,
        top_bootstrap_summary,
        cscv_summary,
        reality,
        primary_walk,
        primary_leave_year,
        friction,
        delay,
        gates,
        unique_paths,
        selection_status,
    )
    (output / "research_report.md").write_text(report, encoding="utf-8")

    source_paths = [
        config_path,
        root / "research/momentum_defender_peak_escape_occam.py",
        root / "research/run_momentum_defender_peak_escape_occam.py",
        root / "strategy/momentum_defender_w40_loss.py",
        root / config["data"]["fund_share_source"],
    ]
    manifest = {
        "experiment_id": config["experiment"]["id"],
        "created_on": config["experiment"]["created_on"],
        "files": [
            {
                "path": str(path.relative_to(root)),
                "sha256": _sha256(path),
                "size": path.stat().st_size,
            }
            for path in source_paths
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
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    root = args.root.resolve()
    config_path = args.config if args.config.is_absolute() else root / args.config
    output = args.output if args.output.is_absolute() else root / args.output
    audit = run_experiment(root, config_path, output)
    print(
        f"wrote {output}: diagnostic={audit['diagnostic_candidate']}, "
        f"selection_status={audit['selection_status']}, "
        f"gates={sum(audit['gates'].values())}/{len(audit['gates'])}"
    )


if __name__ == "__main__":
    main()
