"""Generate the versioned Defender-underperformance badcase ledger."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
import yaml

from defender.relative_defender_rotation import DEFENSIVE_ASSET, ROTATION_ASSETS
from research.defender_curve_momentum import DEFENDER_CANDIDATE
from research.gold_min5_risk_adjusted_momentum_w5 import (
    GoldRAQMW5Params,
    run_gold_raqm_w5,
)
from research.momentum_defender_gold_override import build_gold_override_context
from research.momentum_defender_occam import HELD_RETURN, MOMENTUM_ASSETS
from strategy.momentum_defender_gold_raqm import (
    ENTRY_DIFFERENCE,
    EXIT_DIFFERENCE,
    FORMAL_STRATEGY_ID,
)


DEFAULT_CONTEXT = Path(
    "research/configs/momentum_defender_badcase_context.yaml"
)
DEFAULT_OUTPUT = Path("docs/research/momentum_defender_badcases.md")


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
    assets = (*ROTATION_ASSETS, DEFENSIVE_ASSET)
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


def _slow_return_at_open(context) -> pd.Series:
    result = context.integrated.result
    close = result.inputs.risk_close.astype(float).sort_index()
    trailing = close / close.shift(result.config.slow_lookback) - 1.0
    return trailing.shift(1).reindex(context.calendar).ffill()


def build_badcase_evidence(
    context,
    formal_run,
    *,
    threshold: float,
    asset_names: Mapping[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build all Defender episodes and the subset exceeding the badcase gap."""

    formal_daily = formal_run.daily
    formal_state = formal_run.state
    momentum_returns = context.integrated.result.inputs.momentum[
        HELD_RETURN
    ].astype(float)
    base_state = context.integrated.result.state
    base_daily = context.integrated.result.daily
    slow_return = _slow_return_at_open(context)
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
        immediate_reason = str(formal_state.at[start, "state_reason"])
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
            "base_entry_slow_return_40d": float(slow_return.loc[base_entry]),
            "base_entry_emergency_cap": float(
                base_daily.at[base_entry, "selected_cap"]
            ),
            "base_entry_momentum_asset": str(
                base_daily.at[base_entry, "momentum_asset_at_previous_close"]
            ),
            "formal_entry_metric_difference": float(
                formal_state.at[start, "metric_difference_at_open"]
            )
            if pd.notna(formal_state.at[start, "metric_difference_at_open"])
            else np.nan,
            "dominant_momentum_asset": dominant_asset,
            "dominant_momentum_return": dominant_return,
            "dominant_momentum_days": dominant_days,
        }
        rows.append(row)
        details[case_id] = {
            "defender_runs": _defender_holding_runs(
                context.integrated.targets,
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
    if row["base_entry_reason"] == "slow_regime_switch":
        return (
            f"基础C2于{row['base_defender_entry']}因慢门切入Defender：上一收盘可知的"
            f"510300.SH 40日收益为{row['base_entry_slow_return_40d']:.2%}，未超过"
            "+2.50%风险开启线。"
        )
    if row["base_entry_reason"] == "emergency_exit":
        return (
            f"基础C2于{row['base_defender_entry']}触发紧急退出：当时Momentum持有"
            f"{asset_label}，波动率cap={row['base_entry_emergency_cap']:.2f}，达到"
            "≤0.80条件。"
        )
    return (
        f"基础C2于{row['base_defender_entry']}以"
        f"`{row['base_entry_reason']}`进入Defender。"
    )


def _immediate_reason_text(row: pd.Series) -> str:
    if row["immediate_reason"] == "gold_to_defender_after_min_hold":
        return (
            "本段实际Defender从正式黄金覆盖退出开始：Gold-Defender的RAQM5差值为"
            f"{row['formal_entry_metric_difference']:.3f}，已≤0.60；基础C2当时仍是"
            "Defender，因此没有交回Momentum。"
        )
    return "本段实际Defender与基础C2的风险关闭状态同步开始。"


def _date_range(start: object, end: object) -> str:
    return f"{pd.Timestamp(start).date().isoformat()}—{pd.Timestamp(end).date().isoformat()}"


def _render_document(
    badcases: pd.DataFrame,
    context_config: dict,
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
    immediate_gold_exits = int(
        badcases["immediate_reason"].eq("gold_to_defender_after_min_hold").sum()
    )
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
    lines.extend(
        [
            "",
            "## 横向结论",
            "",
            f"- 基础C2切入原因：慢门切换{cause_counts.get('slow_regime_switch', 0)}段，"
            f"emergency退出{cause_counts.get('emergency_exit', 0)}段。",
            f"- {immediate_gold_exits}段实际Defender开始于黄金覆盖退出，而不是新的C2切换；"
            "这类案例集中反映短窗口黄金退出与中期趋势错位。",
            f"- Momentum最大贡献标的分布：黄金{dominant_counts.get('518880.SH', 0)}段、"
            f"纳指{dominant_counts.get('513100.SH', 0)}段、创业板"
            f"{dominant_counts.get('159915.SZ', 0)}段、沪深300"
            f"{dominant_counts.get('510300.SH', 0)}段。",
            "- 重复出现的薄弱环节有三类：慢门/30日锁错过V形反转；无方向波动率cap退出"
            "仍在上涨的高波动资产；Gold RAQM5短线冷却后回Defender但原Momentum仍持有黄金。",
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
    context = build_gold_override_context(root)
    formal_run = run_gold_raqm_w5(
        context,
        GoldRAQMW5Params(ENTRY_DIFFERENCE, EXIT_DIFFERENCE),
    )
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
    document = _render_document(badcases, config)
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
