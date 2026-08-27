"""Generate the versioned Defender-underperformance badcase ledger."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
import yaml

from defender.relative_defender_rotation import DEFENSIVE_ASSET
from defender.w40_reversal_full_equity import FORMAL_DIVIDEND_ASSETS
from research.defender_curve_momentum import DEFENDER_CANDIDATE
from research.formal_strategy_holdings import build_formal_target_schedule
from research.momentum_defender_gold_override import simulate_candidate_schedule
from research.momentum_defender_occam import HELD_RETURN, MOMENTUM_ASSETS
from strategy.momentum_defender_w40_qm40_signed_exit import (
    DEFENDER_ENTRY_PERCENTILE,
    W40_PERCENTILE_HISTORY,
)
from strategy.momentum_defender_w40_qm40_threshold import (
    FORMAL_STRATEGY_ID,
    run_formal_strategy,
)
from strategy.momentum_defender_w40_gold_escape import GOLD_ASSET


DEFAULT_CONTEXT = Path(
    "research/configs/momentum_defender_badcase_context.yaml"
)
DEFAULT_OUTPUT = Path("docs/research/momentum_defender_badcases.md")
FORMAL_HISTORY_START = date(2013, 1, 1)
DEFAULT_RESTART_START = date(2019, 1, 18)


def defender_episode_windows(
    candidate: pd.Series,
) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, bool]]:
    """Return actual Defender runs and an attribution end including exit open."""

    active = candidate.eq(DEFENDER_CANDIDATE)
    groups = active.ne(active.shift()).cumsum()
    calendar = pd.DatetimeIndex(candidate.index)
    windows = []
    for _, sample in candidate.loc[active].groupby(groups.loc[active]):
        start = pd.Timestamp(sample.index.min())
        last = pd.Timestamp(sample.index.max())
        last_position = calendar.get_loc(last)
        open_ended = last_position == len(calendar) - 1
        end = calendar[last_position if open_ended else last_position + 1]
        windows.append((start, last, pd.Timestamp(end), open_ended))
    return windows


def build_gold_lock_break_evidence(formal_run) -> pd.DataFrame:
    """Compare every Gold lock break with continuous Defender execution."""

    state = formal_run.escape.state
    daily = formal_run.escape.daily
    entries = state["escape_entry"].astype(bool)
    lock_breaks = entries & state["base_w40_held_days_at_open"].lt(30)
    rows: list[dict[str, object]] = []
    for start in state.index[lock_breaks]:
        start_position = state.index.get_loc(start)
        end_position = start_position
        while (
            end_position + 1 < len(state)
            and bool(state.iloc[end_position + 1]["escape_active"])
        ):
            end_position += 1
        interval = state.index[start_position : end_position + 1]
        previous_candidate = (
            str(formal_run.context.initial_previous_candidate)
            if start_position == 0
            else str(daily.iloc[start_position - 1]["candidate"])
        )
        defender_target = pd.Series(
            DEFENDER_CANDIDATE,
            index=interval,
            name="counterfactual_defender_target",
        )
        defender_daily = simulate_candidate_schedule(
            defender_target,
            formal_run.context.interfaces,
            previous_candidate,
        )
        escape_candidates = set(daily.loc[interval, "candidate"].astype(str))
        if escape_candidates != {GOLD_ASSET}:
            raise AssertionError(
                f"Gold lock-break event {start.date()} contains "
                f"non-Gold targets: {sorted(escape_candidates)}"
            )
        gold_return = float(
            (1.0 + daily.loc[interval, "return"].astype(float)).prod() - 1.0
        )
        defender_return = float(
            (1.0 + defender_daily["return"].astype(float)).prod() - 1.0
        )
        held_days = int(state.at[start, "base_w40_held_days_at_open"])
        immediate_veto = bool(
            state.at[start, "state_reason"]
            == "asset_escape_veto_defender_entry"
        )
        if immediate_veto != (held_days == 0):
            raise AssertionError(
                f"Gold lock-break classification mismatch on {start.date()}"
            )
        rows.append(
            {
                "start": pd.Timestamp(start),
                "end": pd.Timestamp(interval[-1]),
                "open_ended": bool(end_position == len(state) - 1),
                "observations": int(len(interval)),
                "base_defender_held_days_at_open": held_days,
                "event_type": (
                    "入场当日否决" if immediate_veto else "持有后突破"
                ),
                "gold_return": gold_return,
                "defender_return": defender_return,
                "gold_excess": gold_return - defender_return,
                "gold_outperformed": bool(gold_return > defender_return),
            }
        )
    evidence = pd.DataFrame(rows)
    expected = int(formal_run.audit["lock_break_entries"])
    if len(evidence) != expected:
        raise AssertionError(
            f"Gold lock-break ledger has {len(evidence)} events; audit has {expected}"
        )
    return evidence


def _gold_lock_break_counts(evidence: pd.DataFrame) -> dict[str, int | float]:
    immediate = evidence["event_type"].eq("入场当日否决")
    held = ~immediate
    total = int(len(evidence))
    wins = int(evidence["gold_outperformed"].astype(bool).sum())
    return {
        "total": total,
        "wins": wins,
        "win_rate": wins / total if total else np.nan,
        "held_total": int(held.sum()),
        "held_wins": int(evidence.loc[held, "gold_outperformed"].sum()),
        "veto_total": int(immediate.sum()),
        "veto_wins": int(evidence.loc[immediate, "gold_outperformed"].sum()),
        "open_events": int(evidence["open_ended"].astype(bool).sum()),
    }


def _portfolio_key(row: pd.Series, assets: tuple[str, ...]) -> tuple[float, ...]:
    return tuple(round(float(row.get(asset, 0.0)), 12) for asset in assets)


def _format_portfolio(
    row: pd.Series,
    assets: tuple[str, ...],
    asset_names: Mapping[str, str],
) -> str:
    parts = []
    for asset in assets:
        weight = float(row.get(asset, 0.0))
        if weight > 1e-12:
            parts.append(f"{asset}（{asset_names.get(asset, asset)}）{weight:.0%}")
    cash = float(row.get("target_cash_weight", 0.0))
    if cash > 1e-12:
        parts.append(f"现金{cash:.0%}")
    return " + ".join(parts) if parts else "无可执行目标"


def _defender_holding_runs(
    targets: pd.DataFrame,
    start: pd.Timestamp,
    last: pd.Timestamp,
    asset_names: Mapping[str, str],
) -> list[dict[str, object]]:
    assets = (*FORMAL_DIVIDEND_ASSETS, DEFENSIVE_ASSET)
    sample = targets.loc[start:last]
    keys = sample.apply(lambda row: _portfolio_key(row, assets), axis=1)
    groups = keys.ne(keys.shift()).cumsum()
    rows = []
    for _, run in sample.groupby(groups):
        rows.append(
            {
                "start": pd.Timestamp(run.index.min()),
                "end": pd.Timestamp(run.index.max()),
                "portfolio": _format_portfolio(
                    run.iloc[0], assets, asset_names
                ),
            }
        )
    return rows


def _momentum_holding_runs(
    target: pd.Series,
    returns: pd.Series,
    start: pd.Timestamp,
    end: pd.Timestamp,
    asset_names: Mapping[str, str],
) -> list[dict[str, object]]:
    selected = target.loc[start:end].astype(str)
    groups = selected.ne(selected.shift()).cumsum()
    rows = []
    for _, run in selected.groupby(groups):
        interval = run.index
        asset = str(run.iloc[0])
        rows.append(
            {
                "start": pd.Timestamp(interval.min()),
                "end": pd.Timestamp(interval.max()),
                "asset": asset,
                "asset_name": asset_names.get(asset, asset),
                "observations": int(len(interval)),
                "return": float((1.0 + returns.loc[interval]).prod() - 1.0),
            }
        )
    return rows


def _dominant_momentum_asset(
    runs: list[dict[str, object]],
) -> tuple[str, float, int]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for run in runs:
        grouped.setdefault(str(run["asset"]), []).append(run)
    rows = []
    for asset, samples in grouped.items():
        compounded = float(
            np.prod([1.0 + float(sample["return"]) for sample in samples]) - 1.0
        )
        observations = int(sum(int(sample["observations"]) for sample in samples))
        rows.append((asset, compounded, observations))
    return max(rows, key=lambda row: (row[1], row[2]))


def build_badcase_evidence(
    context,
    formal_run,
    *,
    threshold: float,
    asset_names: Mapping[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build all Defender episodes and the subset exceeding the badcase gap."""

    formal_daily = formal_run.daily
    momentum_returns = context.integrated.result.inputs.momentum[
        HELD_RETURN
    ].astype(float)
    base_state = formal_run.state
    raw_loss = formal_run.raw_loss_at_open
    score = formal_run.score_at_open
    formal_targets = build_formal_target_schedule(formal_run)
    rows = []
    details: dict[str, dict[str, object]] = {}
    for start, last, end, open_ended in defender_episode_windows(
        formal_daily["candidate"]
    ):
        interval = context.calendar[
            context.calendar.get_loc(start) : context.calendar.get_loc(end) + 1
        ]
        strategy_return = float(
            (1.0 + formal_daily.loc[interval, "return"]).prod() - 1.0
        )
        momentum_return = float(
            (1.0 + momentum_returns.loc[interval]).prod() - 1.0
        )
        absolute_gap = momentum_return - strategy_return
        relative_gap = (1.0 + strategy_return) / (1.0 + momentum_return) - 1.0
        prior = base_state.loc[:start]
        entries = prior.loc[
            prior["state_changed"].astype(bool)
            & ~prior["risk_on"].astype(bool)
        ]
        if entries.empty:
            raise AssertionError(f"no base Defender entry found before {start.date()}")
        base_entry = pd.Timestamp(entries.index[-1])
        base_reason = str(base_state.at[base_entry, "state_reason"])
        base_entry_position = context.calendar.get_loc(base_entry)
        prior_held_days = (
            int(
                base_state.iloc[base_entry_position - 1][
                    "held_days_at_open"
                ]
            )
            + 1
            if base_entry_position > 0
            else 10**9
        )
        current_base_reason = str(base_state.at[start, "state_reason"])
        immediate_reason = (
            current_base_reason
            if current_base_reason != "hold"
            else base_reason
        )
        momentum_runs = _momentum_holding_runs(
            context.momentum_target,
            momentum_returns,
            start,
            end,
            asset_names,
        )
        dominant_asset, dominant_return, dominant_days = _dominant_momentum_asset(
            momentum_runs
        )
        case_id = start.date().isoformat()
        row = {
            "case_start": case_id,
            "last_defender_date": last.date().isoformat(),
            "attribution_end": end.date().isoformat(),
            "open_ended": bool(open_ended),
            "defender_days": int(
                formal_daily.loc[start:last, "candidate"].eq(
                    DEFENDER_CANDIDATE
                ).sum()
            ),
            "attribution_days": int(len(interval)),
            "strategy_return": strategy_return,
            "momentum_return": momentum_return,
            "absolute_return_gap": absolute_gap,
            "relative_return_gap": relative_gap,
            "immediate_reason": immediate_reason,
            "base_defender_entry": base_entry.date().isoformat(),
            "base_entry_reason": base_reason,
            "base_entry_w40_downside_log_loss": float(raw_loss.at[base_entry]),
            "base_entry_w40_loss_percentile": float(score.at[base_entry]),
            "base_entry_prior_sleeve_held_days": prior_held_days,
            "base_entry_momentum_asset": str(
                context.momentum_target.at[base_entry]
            ),
            "dominant_momentum_asset": dominant_asset,
            "dominant_momentum_return": dominant_return,
            "dominant_momentum_days": dominant_days,
        }
        rows.append(row)
        details[case_id] = {
            "defender_runs": _defender_holding_runs(
                formal_targets,
                start,
                last,
                asset_names,
            ),
            "momentum_runs": momentum_runs,
        }
    episodes = pd.DataFrame(rows)
    badcases = episodes.loc[
        episodes["absolute_return_gap"].gt(threshold)
    ].copy()
    badcases.attrs["total_defender_episodes"] = int(len(episodes))
    badcases.attrs["details"] = {
        case_id: details[case_id] for case_id in badcases["case_start"]
    }
    return episodes, badcases


def _reason_text(row: pd.Series, asset_names: Mapping[str, str]) -> str:
    asset = str(row["base_entry_momentum_asset"])
    asset_label = f"{asset}（{asset_names.get(asset, asset)}）"
    if row["base_entry_reason"] in {
        "downside_raqm_to_defender",
        "w40_to_defender",
    }:
        return (
            f"正式状态于{row['base_defender_entry']}在Momentum已连续持有"
            f"{int(row['base_entry_prior_sleeve_held_days'])}日后切入Defender。当时Momentum "
            f"Top-1为{asset_label}；510300的40日对数下跌幅度为"
            f"{row['base_entry_w40_downside_log_loss']:.2%}，其严格滞后{W40_PERCENTILE_HISTORY}日分位为"
            f"{row['base_entry_w40_loss_percentile']:.2%}，已不低于{DEFENDER_ENTRY_PERCENTILE:.0%}。正式规则只需1日"
            "确认，且此前Momentum已满足30日锁。"
        )
    return (
        f"基础W40于{row['base_defender_entry']}以"
        f"`{row['base_entry_reason']}`进入Defender。"
    )


def _immediate_reason_text(row: pd.Series) -> str:
    if row["immediate_reason"] in {
        "downside_raqm_to_defender",
        "w40_to_defender",
    }:
        return "本段实际Defender与正式W40状态切换同步开始。"
    return "本段实际Defender在W40基础状态为Defender时开始。"


def _date_range(start: object, end: object) -> str:
    return f"{pd.Timestamp(start).date().isoformat()}—{pd.Timestamp(end).date().isoformat()}"


def _render_document(
    badcases: pd.DataFrame,
    context_config: dict,
    current_observation: Mapping[str, object] | None = None,
    gold_lock_break_ledgers: Mapping[str, pd.DataFrame] | None = None,
) -> str:
    asset_names = context_config["asset_names"]
    contexts = context_config["cases"]
    details = badcases.attrs["details"]
    configured = set(map(str, contexts))
    observed = set(badcases["case_start"].astype(str))
    if configured != observed:
        raise AssertionError(
            "badcase context coverage mismatch; "
            f"missing={sorted(observed - configured)}, stale={sorted(configured - observed)}"
        )
    total_defender_episodes = int(badcases.attrs["total_defender_episodes"])
    badcase_count = int(len(badcases))
    non_badcase_count = total_defender_episodes - badcase_count
    badcase_share = (
        badcase_count / total_defender_episodes
        if total_defender_episodes
        else np.nan
    )
    non_badcase_share = (
        non_badcase_count / total_defender_episodes
        if total_defender_episodes
        else np.nan
    )
    cause_counts = Counter(badcases["base_entry_reason"])
    dominant_counts = Counter(badcases["dominant_momentum_asset"])
    lines = [
        "# Momentum × Defender 正式策略 Badcase 合集",
        "",
        f"- 正式策略：`{context_config['strategy_id']}`",
        f"- 证据截止：{context_config['evidence_cutoff']}",
        "- Badcase阈值：同一归因区间内，原Momentum累计收益减正式策略累计收益严格大于1个百分点。",
        f"- 全部实际Defender持仓：{total_defender_episodes}段。",
        f"- 当前识别Badcase：{badcase_count}段，占全部Defender持仓段的{badcase_share:.2%}。",
        "",
        "## 口径",
        "",
        "一段实际Defender持仓从正式执行账本的`candidate=DEFENDER`开始，到最后一个完整",
        "Defender交易日结束。收益归因额外包含下一次切出的开盘日，因为该日仍包含Defender",
        "的“昨收→今开”退出腿；若证据截止日仍未切出，则标成开放事件。原Momentum使用同一",
        "交易日、同一费用和开盘执行口径。阈值按绝对收益百分点差判断，同时报告复合相对差。",
        "",
        "市场背景是事后解释，不是模型输入；低置信度案例明确标注，不能用叙事代替统计证据。",
        "",
        "## Badcase占比",
        "",
        "占比的分母是正式执行账本中全部连续`candidate=DEFENDER`持仓段；分子是按上述自定义",
        "口径，原Momentum累计收益比正式策略高出严格超过1个百分点的持仓段。",
        "",
        "|分类|段数|占全部Defender持仓段|",
        "|---|---:|---:|",
        f"|Badcase（跑输>1个百分点）|{badcase_count}|{badcase_share:.2%}|",
        f"|非Badcase|{non_badcase_count}|{non_badcase_share:.2%}|",
        f"|全部Defender持仓段|{total_defender_episodes}|100.00%|",
        "",
        "## 总览",
        "",
        "|ID|Defender完整持有期|归因至|Defender/正式策略|原Momentum|收益差|复合相对差|基础切入原因|Momentum最大贡献标的|",
        "|---|---|---|---:|---:|---:|---:|---|---|",
    ]
    for number, (_, row) in enumerate(badcases.iterrows(), start=1):
        open_mark = "（开放）" if bool(row["open_ended"]) else ""
        dominant = str(row["dominant_momentum_asset"])
        lines.append(
            f"|BC-{number:02d}|{row['case_start']}—{row['last_defender_date']}{open_mark}|"
            f"{row['attribution_end']}|{row['strategy_return']:.2%}|"
            f"{row['momentum_return']:.2%}|{row['absolute_return_gap']:+.2%}|"
            f"{row['relative_return_gap']:+.2%}|`{row['base_entry_reason']}`|"
            f"{dominant}（{asset_names.get(dominant, dominant)}）|"
        )
    if gold_lock_break_ledgers:
        restart_label = str(context_config["gold_lock_break_restart"])
        if restart_label not in gold_lock_break_ledgers:
            raise AssertionError("Gold lock-break restart ledger is missing")
        lines.extend(
            [
                "",
                "## 黄金打破Defender锁胜负台账",
                "",
                "正式审计把`escape_entry=true`且基础Defender锁龄小于30个交易日定义为一次黄金",
                "打破Defender锁，其中既包括已经实际持有Defender后的突破，也包括基础Defender",
                "入场当日被黄金直接否决。每个事件从触发日开盘计至最后一个完整黄金逃生日收盘；",
                "黄金路径与“保持同一前序持仓、但该事件全程改为连续Defender”的反事实路径使用",
                "完全相同的开盘执行、切换腿和费用口径。收益严格更高才记为黄金跑赢；截止日仍",
                "未结束的事件计入触发与暂时胜负，并标为开放，后续重建台账时自动更新。",
                "",
                "|重启口径|触发|黄金跑赢|胜率|持有后突破（赢）|入场当日否决（赢）|开放事件|",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for label, evidence in gold_lock_break_ledgers.items():
            counts = _gold_lock_break_counts(evidence)
            lines.append(
                f"|{label}—{context_config['evidence_cutoff']}|{counts['total']}|"
                f"{counts['wins']}|{counts['win_rate']:.2%}|"
                f"{counts['held_total']}（{counts['held_wins']}）|"
                f"{counts['veto_total']}（{counts['veto_wins']}）|"
                f"{counts['open_events']}|"
            )
        restart = gold_lock_break_ledgers[restart_label]
        lines.extend(
            [
                "",
                f"### {restart_label}重启口径逐事件",
                "",
                "|黄金逃生区间|状态|类型|触发时基础锁龄|黄金|持续Defender|黄金超额|结果|",
                "|---|---|---|---:|---:|---:|---:|---|",
            ]
        )
        for _, row in restart.iterrows():
            status = "开放" if bool(row["open_ended"]) else "已结束"
            result = "跑赢" if bool(row["gold_outperformed"]) else "未跑赢"
            lines.append(
                f"|{_date_range(row['start'], row['end'])}|{status}|"
                f"{row['event_type']}|{int(row['base_defender_held_days_at_open'])}日|"
                f"{row['gold_return']:+.2%}|{row['defender_return']:+.2%}|"
                f"{row['gold_excess']:+.2%}|{result}|"
            )
    lines.extend(
        [
            "",
            "## 横向结论",
            "",
            f"- 全部Badcase均来自510300单一40日下跌幅度分位达到{DEFENDER_ENTRY_PERCENTILE:.0%}后的正式切换；"
            f"共{cause_counts.get('w40_to_defender', 0)}段。",
            f"- Momentum最大贡献标的分布：黄金{dominant_counts.get('518880.SH', 0)}段、"
            f"纳指{dominant_counts.get('513100.SH', 0)}段、创业板"
            f"{dominant_counts.get('159915.SZ', 0)}段、沪深300"
            f"{dominant_counts.get('510300.SH', 0)}段。",
            "- 当前正式策略以510300 W40为基础风险状态，允许QM40连续恢复基础Momentum，"
            "并保留黄金QM20专用逃生。"
            "剩余薄弱环节主要是纳指、创业板等独立趋势仍无破锁政策，以及黄金逃生前5日资格"
            "与退出后返回100%红利造成的短片段错配。",
            "",
            "## 逐案记录",
            "",
        ]
    )
    for number, (_, row) in enumerate(badcases.iterrows(), start=1):
        case_id = str(row["case_start"])
        explanation = contexts[case_id]
        open_mark = "；该事件截至截止日仍开放" if bool(row["open_ended"]) else ""
        lines.extend(
            [
                f"### BC-{number:02d}：{row['case_start']}—{row['last_defender_date']}",
                "",
                f"置信度：**{str(explanation['confidence']).upper()}**{open_mark}。",
                "",
                "|项目|结果|",
                "|---|---:|",
                f"|Defender完整持有日|{int(row['defender_days'])}|",
                f"|含切出腿归因日|{int(row['attribution_days'])}|",
                f"|正式策略收益|{row['strategy_return']:.2%}|",
                f"|原Momentum收益|{row['momentum_return']:.2%}|",
                f"|绝对收益差|{row['absolute_return_gap']:+.2%}|",
                f"|正式策略相对Momentum|{row['relative_return_gap']:+.2%}|",
                "",
                "**为什么进入或回到Defender**",
                "",
                _reason_text(row, asset_names),
                "",
                _immediate_reason_text(row),
                "",
                "**Defender实际持仓**",
                "",
            ]
        )
        for run in details[case_id]["defender_runs"]:
            lines.append(
                f"- {_date_range(run['start'], run['end'])}：{run['portfolio']}。"
            )
        lines.extend(["", "**原Momentum同期持仓**", ""])
        for run in details[case_id]["momentum_runs"]:
            lines.append(
                f"- {_date_range(run['start'], run['end'])}：{run['asset']}"
                f"（{run['asset_name']}），该子段{float(run['return']):+.2%}。"
            )
        dominant = str(row["dominant_momentum_asset"])
        lines.extend(
            [
                "",
                "**为什么Momentum显著更强**",
                "",
                f"机械证据：贡献最大的Momentum标的是{dominant}"
                f"（{asset_names.get(dominant, dominant)}），合计"
                f"{int(row['dominant_momentum_days'])}日、复合收益"
                f"{row['dominant_momentum_return']:+.2%}。",
                "",
                str(explanation["market_context"]),
                "",
                str(explanation["why_momentum_outpaced"]),
                "",
                f"历史薄弱环节：**{explanation['weakness']}**",
            ]
        )
        sources = explanation.get("sources", [])
        if sources:
            lines.extend(["", "参考背景："])
            for source in sources:
                lines.append(f"- [{source['title']}]({source['url']})")
        lines.extend(["", "---", ""])
    if current_observation is not None:
        observation_config = context_config["current_observation"]
        lines.extend(
            [
                "## 当前开放观察段（人工要求保留）",
                "",
                f"- 观察区间：{current_observation['start']}—{current_observation['end']}（开放）。",
                f"- W40基础状态：Defender；实际顶层候选：`{current_observation['candidate']}`。",
                f"- 正式策略收益：{float(current_observation['strategy_return']):+.2%}。",
                f"- 原Momentum收益：{float(current_observation['momentum_return']):+.2%}。",
                f"- Momentum领先：{float(current_observation['absolute_return_gap']):+.2%}。",
                "",
                "本段按用户要求保留在Defender跑输Momentum台账中。它发生在基础W40仍为"
                "Defender的黄金逃生期间，但截至证据日的领先幅度尚未严格超过1个百分点，"
                "因此不计入上方正式Badcase数量和占比。",
                "",
                str(observation_config["market_context"]),
                "",
                str(observation_config["why_momentum_outpaced"]),
                "",
                f"当前薄弱环节：**{observation_config['weakness']}**",
                "",
                "---",
                "",
            ]
        )
    lines.extend(
        [
            "## 更新规则",
            "",
            "每次正式策略的状态机、阈值、标的池、Defender实现、执行费用、回测截止日或数据发生",
            "变化时，必须重新运行：",
            "",
            "```bash",
            "uv run python -m research.generate_momentum_defender_badcases",
            "uv run python -m research.generate_momentum_defender_badcases --check",
            "```",
            "",
            "生成器会重新识别全部Defender阶段，并要求上下文配置恰好覆盖所有收益差超过1个百分点",
            "的badcase。出现新案例、旧案例消失或起点变化时，配置覆盖校验会失败，必须人工补充或",
            "修订市场背景、因果解释、薄弱环节和来源后才能通过。不得只更新汇总数字。",
            "生成器还必须同时重算2013完整历史与2019-01-18重启口径下的黄金破锁触发数、黄金",
            "跑赢连续Defender的次数和胜率，并拆分持有后突破与入场当日否决；开放事件保留在",
            "分母中且随截止日滚动更新。",
            "",
            "本文件记录历史弱点，不等于要求消灭所有跑输。Defender主动放弃部分上涨以控制尾部风险",
            "是允许的；任何修复仍须经过完整的因果、成本、多重试验和过拟合检验。",
            "",
        ]
    )
    return "\n".join(lines)


def generate(root: Path, context_path: Path) -> tuple[str, pd.DataFrame, pd.DataFrame]:
    config = yaml.safe_load(context_path.read_text(encoding="utf-8"))
    if config["strategy_id"] != FORMAL_STRATEGY_ID:
        raise AssertionError("badcase context strategy ID is not the formal strategy")
    threshold = float(config["badcase_threshold_absolute_return_gap"])
    formal_run = run_formal_strategy(
        root,
        end=pd.Timestamp(config["evidence_cutoff"]).date(),
    )
    context = formal_run.context
    actual_cutoff = context.calendar.max().date().isoformat()
    if str(config["evidence_cutoff"]) != actual_cutoff:
        raise AssertionError(
            f"context cutoff {config['evidence_cutoff']} != data cutoff {actual_cutoff}"
        )
    episodes, badcases = build_badcase_evidence(
        context,
        formal_run,
        threshold=threshold,
        asset_names=config["asset_names"],
    )
    current_observation = None
    if "current_observation" in config:
        start = pd.Timestamp(config["current_observation"]["start"])
        end = pd.Timestamp(config["evidence_cutoff"])
        interval = formal_run.daily.loc[start:end].index
        if interval.empty or interval[0] != start or interval[-1] != end:
            raise AssertionError("current observation does not cover the configured interval")
        if formal_run.state.loc[interval, "risk_on"].astype(bool).any():
            raise AssertionError("current observation is not wholly inside base Defender state")
        candidates = formal_run.daily.loc[interval, "candidate"].astype(str).unique()
        if len(candidates) != 1:
            raise AssertionError("current observation must have one executable candidate")
        momentum_returns = formal_run.context.integrated.result.inputs.momentum[
            HELD_RETURN
        ].astype(float)
        strategy_return = float(
            (1.0 + formal_run.daily.loc[interval, "return"]).prod() - 1.0
        )
        momentum_return = float(
            (1.0 + momentum_returns.loc[interval]).prod() - 1.0
        )
        current_observation = {
            "start": start.date().isoformat(),
            "end": end.date().isoformat(),
            "candidate": str(candidates[0]),
            "strategy_return": strategy_return,
            "momentum_return": momentum_return,
            "absolute_return_gap": momentum_return - strategy_return,
        }
    restart_start = pd.Timestamp(
        config.get("gold_lock_break_restart", DEFAULT_RESTART_START)
    ).date()
    restart_run = run_formal_strategy(
        root,
        end=pd.Timestamp(config["evidence_cutoff"]).date(),
        start=restart_start,
    )
    gold_lock_break_ledgers = {
        FORMAL_HISTORY_START.isoformat(): build_gold_lock_break_evidence(
            formal_run
        ),
        restart_start.isoformat(): build_gold_lock_break_evidence(restart_run),
    }
    document = _render_document(
        badcases,
        config,
        current_observation,
        gold_lock_break_ledgers,
    )
    return document, episodes, badcases


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--context", type=Path, default=DEFAULT_CONTEXT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    context_path = args.context if args.context.is_absolute() else root / args.context
    output_path = args.output if args.output.is_absolute() else root / args.output
    document, episodes, badcases = generate(root, context_path)
    if args.check:
        if not output_path.exists():
            raise SystemExit(f"badcase document missing: {output_path}")
        if output_path.read_text(encoding="utf-8") != document:
            raise SystemExit(
                "badcase document is stale; regenerate and review contextual explanations"
            )
        print(
            f"badcase document current: {len(episodes)} Defender episodes, "
            f"{len(badcases)} badcases"
        )
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(document, encoding="utf-8")
    print(
        f"wrote {output_path}: {len(episodes)} Defender episodes, "
        f"{len(badcases)} badcases"
    )


if __name__ == "__main__":
    main()
