"""Daily runner for the formal W40/QM40 Defender strategy with Gold escape.

The production default uses the frozen 756-day W40 gate, monthly lowest-QM40
Defender, signed-QM40 recovery, and Gold-only QM20 escape. Older W40,
weighted-DRAQM, C2, and raw-Gold configs remain explicitly dispatchable
rollback paths.
"""

from __future__ import annotations

import argparse
import math
from datetime import date
from pathlib import Path
from typing import Callable

from execution.interfaces import Order, diff
from execution.position import read_position
from notification.dingtalk import DingTalkNotifier
from notification.formatter import ASSET_NAMES
from notification.peak_warning import PeakWarning, evaluate_peak_warning
from run_daily import (
    _backfill_open_prices,
    _is_sse_trading_day,
    _latest_common_data_date,
    _load_config,
    _next_entry_date,
    _priced_state_as_of,
    _save_or_update_rebalance_target,
    _sync_and_check,
)
from strategy.momentum_defender import (
    IntegratedNextOpenSignal,
    build_integrated_next_open_signal,
)
from strategy.momentum_defender_gold_raqm import (
    FormalGoldNextOpenSignal,
    build_formal_gold_next_open_signal,
)
from strategy.momentum_defender_downside_raqm import (
    FormalDownsideRAQMNextOpenSignal,
    build_next_open_signal as build_downside_raqm_next_open_signal,
)
from strategy.momentum_defender_w40_loss import (
    FormalW40LossNextOpenSignal,
    build_next_open_signal as build_w40_loss_next_open_signal,
)
from strategy.momentum_defender_w40_full_equity import (
    build_next_open_signal as build_w40_full_equity_next_open_signal,
)
from strategy.momentum_defender_w40_gold_escape import (
    FormalW40GoldEscapeNextOpenSignal,
    build_next_open_signal as build_w40_gold_escape_next_open_signal,
)
from strategy.momentum_defender_w40_qm40_signed_exit import (
    FormalW40QM40NextOpenSignal,
    build_next_open_signal as build_w40_qm40_signed_exit_next_open_signal,
)
from strategy.momentum_defender_w40_qm40_threshold import (
    build_next_open_signal as build_w40_qm40_threshold_next_open_signal,
)
from strategy.prospective_ledger import append_signal_record


DEFENDER_ASSET_NAMES = {
    "512890.SH": "红利低波ETF",
    "159545.SZ": "恒生红利低波ETF",
    "513530.SH": "港股通红利ETF",
    "515080.SH": "中证红利ETF",
    "510880.SH": "红利ETF",
    "563020.SH": "低波红利ETF",
    "515450.SH": "红利低波50ETF",
    "513630.SH": "港股低波红利ETF",
    "511260.SH": "十年国债ETF",
}
ALL_ASSET_NAMES = {**ASSET_NAMES, **DEFENDER_ASSET_NAMES}
MOMENTUM_ASSET_ORDER = (
    "510300.SH",
    "159915.SZ",
    "513100.SH",
    "518880.SH",
)

W40_SLEEVE_LABELS = {
    "momentum": "动量",
    "defender": "防守",
    "gold_escape": "黄金逃生",
}
W40_STATE_REASON_LABELS = {
    "base_w40_momentum": "W40基础状态保持动量",
    "base_w40_defender": "W40基础状态保持防守",
    "asset_escape_break_defender_lock": "黄金满足逃生条件，打破防守锁",
    "asset_escape_veto_defender_entry": "黄金已满足条件，否决实际进入防守",
    "asset_escape_hard_hold": "黄金逃生硬持有期",
    "asset_escape_momentum_hold": "黄金逃生继续持有",
    "asset_escape_return_disabled_top1": "Momentum Top1不再是黄金，返回防守",
    "asset_escape_return_below_y": "黄金相对Defender指标跌破退出线，返回防守",
    "base_w40_recovered_to_momentum": "W40基础状态恢复动量，结束黄金逃生",
    "downside_raqm_to_defender": "W40达到进入线，切换防守",
    "downside_raqm_to_momentum": "W40达到恢复线，切换动量",
    "hold": "保持当前W40基础状态",
    "w40_to_defender": "W40达到60%进入线，切换防守",
    "qm40_recovery_to_momentum": "QM40转正连续10日，提前恢复动量",
    "w40_fallback_to_momentum": "Defender满30日且W40低于35%，恢复动量",
    "w40_recovery_blocked_by_defender_policy": "W40已恢复但尚未满足QM40早退或30日保底",
}


def _asset_label(asset: str) -> str:
    return f"{asset.split('.')[0]} {ALL_ASSET_NAMES.get(asset, asset)}"


def _allocation(weights: dict[str, float], cash: float = 0.0) -> str:
    parts = [
        f"{_asset_label(asset)} {weight:.0%}"
        for asset, weight in weights.items()
        if weight > 1e-14
    ]
    if cash > 1e-14:
        parts.append(f"现金 {cash:.0%}")
    return "、".join(parts) if parts else "空仓"


def _order_lines(orders: list[Order]) -> list[str]:
    actionable = [order for order in orders if order.action != "hold"]
    if not actionable:
        return ["• 无调仓，继续持有"]
    lines: list[str] = []
    for order in actionable:
        action = "买入" if order.action == "buy" else "卖出"
        lines.append(
            f"• {action} {_asset_label(order.asset)} "
            f"{abs(order.weight_delta):.0%}"
        )
    return lines


def _w40_sleeve_text(signal) -> str:
    current = W40_SLEEVE_LABELS.get(
        signal.current_model_sleeve, signal.current_model_sleeve
    )
    target = W40_SLEEVE_LABELS.get(signal.target_sleeve, signal.target_sleeve)
    return f"{current} → {target}" if current != target else f"{target}（保持）"


def _w40_reason_text(reason: str) -> str:
    return W40_STATE_REASON_LABELS.get(reason, f"未知状态原因：{reason}")


def _w40_model_instruction_lines(signal) -> list[str]:
    current_candidate = getattr(signal, "current_candidate", None)
    if current_candidate is None:
        current_candidate = (
            "DEFENDER"
            if signal.current_model_sleeve == "defender"
            else signal.momentum.held_asset
        )
    if current_candidate == "DEFENDER":
        current_weights = dict(signal.defender.current_weights)
        current_cash = max(0.0, 1.0 - sum(current_weights.values()))
    else:
        current_weights = {str(current_candidate): 1.0}
        current_cash = 0.0
    target_weights = dict(signal.target_weights)
    target_cash = float(signal.target_cash_weight)
    assets = set(current_weights) | set(target_weights)
    unchanged = all(
        abs(current_weights.get(asset, 0.0) - target_weights.get(asset, 0.0))
        <= 1e-12
        for asset in assets
    ) and abs(current_cash - target_cash) <= 1e-12
    if unchanged:
        return [f"• 继续持有 {_allocation(target_weights, target_cash)}"]
    return [
        f"• 调整：{_allocation(current_weights, current_cash)} → "
        f"{_allocation(target_weights, target_cash)}"
    ]


def _performance_difference_text(value: float, held: float) -> str:
    difference = value - held
    if abs(difference) <= 5e-12:
        return "与持仓持平"
    if difference > 0.0:
        return f"领先持仓 {difference:+.2%} ↑"
    return f"落后持仓 {difference:+.2%} ↓"


def _format_signal_performance_lines(
    signal: FormalW40GoldEscapeNextOpenSignal,
) -> list[str]:
    snapshot = signal.performance_snapshot
    if snapshot is None:
        return [
            "**同期表现**",
            f"• 数据不可用（{signal.performance_error or 'UnknownError'}）",
            "",
            "**周期表现**",
            f"• 数据不可用（{signal.performance_error or 'UnknownError'}）",
        ]
    held_return = float(snapshot.current_holding_return)
    held_label = (
        _asset_label(snapshot.current_holding_label)
        if snapshot.current_holding_label != "CURRENT_PORTFOLIO"
        else "当前组合"
    )
    lines = [
        f"**同期表现（自 {snapshot.since_date.isoformat()} 开盘起）**",
        f"• {held_label}　{held_return:+.2%}　|　当前持仓",
    ]
    comparison_order = ["MOMENTUM", "DEFENDER", *MOMENTUM_ASSET_ORDER]
    comparison_labels = {
        "MOMENTUM": "Momentum",
        "DEFENDER": "Defender",
    }
    for key in comparison_order:
        if key not in snapshot.concurrent_returns:
            continue
        value = float(snapshot.concurrent_returns[key])
        label = comparison_labels.get(key, _asset_label(key))
        lines.append(
            f"• {label}　{value:+.2%}　|　"
            f"{_performance_difference_text(value, held_return)}"
        )
    period_labels = {
        "FORMAL": "当前完整策略",
        "LEGACY_MOMENTUM": "原非对数Momentum（模型回放）",
        "PURE_MOMENTUM": "纯Momentum",
        "PURE_DEFENDER": "纯Defender",
    }
    lines.extend(["", "**周期表现**"])
    for key in (
        "FORMAL",
        "LEGACY_MOMENTUM",
        "PURE_MOMENTUM",
        "PURE_DEFENDER",
    ):
        values = snapshot.period_returns[key]
        lines.append(
            f"• {period_labels[key]}　本月 {values.month:+.2%}　|　"
            f"本季度 {values.quarter:+.2%}　|　本年 {values.year:+.2%}"
        )
    return lines


def format_integrated_notification(
    signal: IntegratedNextOpenSignal | FormalGoldNextOpenSignal,
    current_weights: dict[str, float],
    orders: list[Order],
) -> str:
    """Format one auditable composite next-open signal."""
    current_cash = max(0.0, 1.0 - sum(current_weights.values()))
    target = dict(signal.target_weights)
    sleeve_change = (
        f"{signal.current_model_sleeve} → {signal.target_sleeve}"
        if signal.current_model_sleeve != signal.target_sleeve
        else f"{signal.target_sleeve}（保持）"
    )
    if isinstance(signal, FormalGoldNextOpenSignal):
        emergency_text = (
            f"{'触发' if signal.emergency_alert else '未触发'}（"
            f"{_asset_label(signal.held_momentum_asset)}，5日趋势="
            f"{signal.emergency_log_return_5:+.2%}，20日下行波动="
            f"{signal.emergency_downside_volatility_20:.2%}，历史q95="
            f"{signal.emergency_downside_threshold_q95:.2%}）"
        )
        state_counter_label = "基础状态持续（仅诊断，不限制切换）"
        diagnostics = [
            (
                f"• 沪深300 120日对数收益 "
                f"{signal.anchor_log_return_120:+.2%}"
            ),
            (
                f"• 当前Momentum持仓 {_asset_label(signal.held_momentum_asset)} "
                f"120日对数收益 {signal.held_asset_log_return_120:+.2%}"
            ),
            (
                f"• 双趋势确认基础状态="
                f"{'Momentum' if signal.base_target_sleeve == 'momentum' else 'Defender'}；"
                f"恢复确认 {signal.risk_on_confirmation_streak}/"
                f"{signal.risk_on_confirmation_days}，风险关闭确认 "
                f"{signal.risk_off_confirmation_streak}/"
                f"{signal.risk_off_confirmation_days}"
            ),
            (
                f"• Top1快速反转桥接="
                f"{'激活' if signal.rapid_reversal_target_active else '未激活'}；"
                f"{_asset_label(signal.rapid_reversal_asset)} Raw RAQM5−Defender="
                f"{signal.rapid_reversal_difference:+.3f}（>"
                f"{signal.rapid_reversal_entry_threshold:.2f}入场，≤"
                f"{signal.rapid_reversal_exit_threshold:.2f}退出，无持有锁）"
            ),
            f"• 方向敏感紧急退出：{emergency_text}",
        ]
    else:
        emergency_text = (
            f"触发（{_asset_label(signal.emergency_asset)} cap="
            f"{signal.emergency_cap:.0%}）"
            if signal.emergency_alert
            else (
                f"未触发（{_asset_label(signal.emergency_asset)} cap="
                f"{signal.emergency_cap:.0%}）"
            )
        )
        state_counter_label = "30日状态锁计数"
        diagnostics = [
            (
                f"• 沪深300 40日收益 {signal.slow_gate_return:+.2%}，"
                f"慢门控={'Momentum' if signal.slow_gate_risk_on else 'Defender'}"
            ),
            f"• 紧急波动 cap：{emergency_text}",
        ]
    momentum_target = _allocation(dict(signal.momentum.effective_weights))
    defender_target = _allocation(
        dict(signal.defender.target_weights),
        signal.defender.target_cash_weight,
    )
    lines = [
        f"## 📊 {signal.strategy_id} 信号",
        "",
        f"**信号收盘：** {signal.signal_date.isoformat()}",
        f"**执行开盘：** {signal.execution_date.isoformat()}",
        "",
        f"**袖套状态：** {sleeve_change}",
        f"**状态原因：** {signal.state_reason}",
        f"**{state_counter_label}：** {signal.held_days_at_open} 个交易日",
        "",
        f"**当前实盘持仓：** {_allocation(current_weights, current_cash)}",
        f"**实际目标：** {_allocation(target, signal.target_cash_weight)}",
        "",
        "**调仓指令**",
        *_order_lines(orders),
        "",
        "**模型诊断**",
        *diagnostics,
        f"• Momentum 下一开盘目标：{momentum_target}",
        f"• Defender 下一开盘目标：{defender_target}",
        (
            f"• Defender 月度选择：{_asset_label(signal.defender.target_selected_asset)}"
            f"（{signal.defender.selection_reason}）"
        ),
        "",
        "信号仅使用截至信号日收盘的数据，目标于下一交易日开盘生效。",
    ]
    return "\n\n".join(lines)


def format_downside_raqm_notification(
    signal: FormalDownsideRAQMNextOpenSignal,
    current_weights: dict[str, float],
    orders: list[Order],
) -> str:
    """Format the frozen one-state-machine downside-RAQM signal."""
    current_cash = max(0.0, 1.0 - sum(current_weights.values()))
    sleeve_change = (
        f"{signal.current_model_sleeve} → {signal.target_sleeve}"
        if signal.current_model_sleeve != signal.target_sleeve
        else f"{signal.target_sleeve}（保持）"
    )
    momentum_target = _allocation(dict(signal.momentum.effective_weights))
    defender_target = _allocation(
        dict(signal.defender.target_weights), signal.defender.target_cash_weight
    )
    lines = [
        f"## 📊 {signal.strategy_id} 信号",
        "",
        f"**信号收盘：** {signal.signal_date.isoformat()}",
        f"**执行开盘：** {signal.execution_date.isoformat()}",
        "",
        f"**袖套状态：** {sleeve_change}",
        f"**状态原因：** {signal.state_reason}",
        f"**当前袖套持有计数：** {signal.held_days_at_open} 个交易日",
        "",
        f"**当前实盘持仓：** {_allocation(current_weights, current_cash)}",
        f"**实际目标：** {_allocation(dict(signal.target_weights), signal.target_cash_weight)}",
        "",
        "**调仓指令**",
        *_order_lines(orders),
        "",
        "**冻结510300下行DRAQM诊断**",
        f"• DRAQM30原值：{signal.downside_raqm_30:.4f}",
        f"• DRAQM40原值：{signal.downside_raqm_40:.4f}",
        (
            f"• 组合分位：{signal.downside_raqm_percentile:.2%}="
            "25%×P30+75%×P40"
        ),
        (
            f"• Defender进入：≥{signal.defender_entry_percentile:.0%}，连续 "
            f"{signal.entry_confirmation_streak}/"
            f"{signal.defender_entry_confirmation_days} 日"
        ),
        (
            f"• Momentum恢复：≤{signal.momentum_recovery_percentile:.0%}，连续 "
            f"{signal.recovery_confirmation_streak}/"
            f"{signal.momentum_recovery_confirmation_days} 日"
        ),
        (
            f"• 状态锁：Momentum {signal.momentum_lock_days}日 / "
            f"Defender {signal.defender_lock_days}日"
        ),
        f"• Momentum 下一开盘目标：{momentum_target}",
        f"• Defender 下一开盘目标：{defender_target}",
        (
            f"• Defender 月度选择：{_asset_label(signal.defender.target_selected_asset)}"
            f"（{signal.defender.selection_reason}）"
        ),
        "",
        "信号仅使用截至信号日收盘的数据，目标于下一交易日开盘生效。",
    ]
    return "\n\n".join(lines)


def format_w40_loss_notification(
    signal: FormalW40LossNextOpenSignal | FormalW40GoldEscapeNextOpenSignal,
    current_weights: dict[str, float],
    orders: list[Order],
    peak_warning: PeakWarning | None = None,
) -> str:
    """Format the frozen single-W40 downside-loss signal."""
    if peak_warning is None:
        peak_warning_lines = [
            "**价格×量能顶部预警（只读）**",
            "• 预警计算不可用。",
        ]
    else:
        status = "⚠️ 已触发" if peak_warning.triggered else "未触发"
        price_values_available = all(
            math.isfinite(value)
            for value in (
                peak_warning.current_close,
                peak_warning.prior_high200,
                peak_warning.price_breakout,
            )
        )
        if price_values_available:
            price_met = peak_warning.price_breakout > 0.0
            if price_met:
                price_condition = (
                    "已满足；严格滞后前高 "
                    f"{peak_warning.prior_high200:.3f}，收盘 "
                    f"{peak_warning.current_close:.3f}（高出"
                    f"{peak_warning.price_breakout:.2%}）"
                )
            else:
                price_gap = max(
                    0.0,
                    peak_warning.prior_high200 / peak_warning.current_close - 1.0,
                )
                price_condition = (
                    "不满足；严格滞后前高 "
                    f"{peak_warning.prior_high200:.3f}，收盘 "
                    f"{peak_warning.current_close:.3f}，尚差{price_gap:.2%}"
                )
        else:
            price_condition = "无法评估"

        return_values_available = all(
            math.isfinite(value)
            for value in (
                peak_warning.current_close,
                peak_warning.close20ago,
                peak_warning.price_return20,
            )
        )
        if return_values_available:
            return_met = peak_warning.price_return20 >= 0.15
            required_close = peak_warning.close20ago * 1.15
            return_gap_pp = max(0.0, 0.15 - peak_warning.price_return20) * 100.0
            return_condition = (
                f"{'已满足' if return_met else '不满足'}；当前"
                f"{peak_warning.price_return20:.2%}"
                + (
                    ""
                    if return_met
                    else f"，还差{return_gap_pp:.2f}个百分点"
                )
                + f"（收盘门槛≥{required_close:.3f}）"
            )
        else:
            return_condition = "无法评估"

        volume_values_available = all(
            math.isfinite(value)
            for value in (
                peak_warning.current_volume,
                peak_warning.prior_volume_median20,
                peak_warning.volume_ratio20,
            )
        )
        if volume_values_available:
            volume_met = peak_warning.volume_ratio20 >= 1.50
            required_volume = peak_warning.prior_volume_median20 * 1.50
            volume_gap = max(0.0, 1.50 - peak_warning.volume_ratio20)
            volume_condition = (
                f"{'已满足' if volume_met else '不满足'}；当前"
                f"{peak_warning.volume_ratio20:.2f}倍"
                + ("" if volume_met else f"，还差{volume_gap:.2f}倍")
                + f"（成交量门槛≥{required_volume:,.0f}）"
            )
        else:
            volume_condition = "无法评估"

        share_line = []
        if peak_warning.share_filter_required:
            if (
                peak_warning.share_data_available
                and peak_warning.share_flow20 is not None
            ):
                share_met = peak_warning.share_flow20 > 0.0
                share_condition = (
                    f"{'已满足' if share_met else '不满足'}；当前"
                    f"{peak_warning.share_flow20:+.2%}（要求严格>0）"
                )
            else:
                share_condition = "无法评估；保守地不预警"
            share_line = [
                f"• 创业板20日基金份额增长：{share_condition}"
            ]
        peak_warning_lines = [
            "**价格×量能顶部预警（只读）**",
            f"• 评估目标：{_asset_label(peak_warning.asset)}",
            f"• 突破200日前高：{price_condition}",
            f"• 20日涨幅≥15%：{return_condition}",
            f"• 成交量/此前20日中位数≥1.50倍：{volume_condition}",
            *share_line,
            f"• 预警状态：{status}",
        ]
    percentile_history = int(getattr(signal, "w40_percentile_history", 504))
    is_qm40_v4 = isinstance(signal, FormalW40QM40NextOpenSignal)
    if is_qm40_v4:
        recovery_line = (
            f"• W40保底恢复：基础Defender满{signal.defender_lock_days}日后，"
            f"分位≤{signal.momentum_recovery_percentile:.0%}"
        )
        lock_line = (
            f"• 状态约束：Momentum锁{signal.momentum_lock_days}日；"
            f"基础Defender最低{signal.base_defender_minimum_days}日；"
            f"QM40连续{signal.qm40_recovery_confirmation_days}日可早退；"
            f"{signal.defender_lock_days}日保底"
        )
    else:
        recovery_line = (
            f"• Momentum恢复：≤{signal.momentum_recovery_percentile:.0%}，连续"
            f"{signal.recovery_confirmation_streak}日"
        )
        lock_line = (
            f"• 状态锁：Momentum {signal.momentum_lock_days}日 / "
            f"Defender {signal.defender_lock_days}日"
        )
    lines = [
        f"## 📊 {signal.strategy_id} 信号",
        "",
        f"**信号收盘：** {signal.signal_date.isoformat()}",
        f"**执行开盘：** {signal.execution_date.isoformat()}",
        "",
        f"**袖套状态：** {_w40_sleeve_text(signal)}",
        f"**状态原因：** {_w40_reason_text(signal.state_reason)}",
        f"**当前袖套持有计数：** {signal.held_days_at_open} 个交易日",
        "",
        f"**实际目标：** {_allocation(dict(signal.target_weights), signal.target_cash_weight)}",
        "",
        "**调仓指令**",
        *_w40_model_instruction_lines(signal),
        "",
        "**510300 W40风险门控诊断**",
        f"• 40日对数下跌幅度：{signal.w40_downside_log_loss:.2%}",
        f"• 严格滞后{percentile_history}日分位：{signal.w40_loss_percentile:.2%}",
        (
            f"• Defender进入：≥{signal.defender_entry_percentile:.0%}，连续"
            f"{signal.entry_confirmation_streak}日"
        ),
        recovery_line,
        lock_line,
        f"• Momentum 下一开盘目标：{_allocation(dict(signal.momentum.effective_weights))}",
        (
            f"• Defender 下一开盘目标："
            f"{_allocation(dict(signal.defender.target_weights), signal.defender.target_cash_weight)}"
        ),
        (
            f"• Defender 月度选择：{_asset_label(signal.defender.target_selected_asset)}"
            f"（{signal.defender.selection_reason}）"
        ),
        "",
        *peak_warning_lines,
    ]
    return "\n\n".join(lines)


def format_w40_gold_escape_notification(
    signal: FormalW40GoldEscapeNextOpenSignal,
    current_weights: dict[str, float],
    orders: list[Order],
    peak_warning: PeakWarning | None,
) -> str:
    message = format_w40_loss_notification(
        signal, current_weights, orders, peak_warning
    )
    block = [
        "---",
        "**黄金QM20专用Defender破锁**",
        (
            f"• 状态：{'激活' if signal.escape_active else '未激活'}；"
            f"本次入场={'是' if signal.escape_entry else '否'}；"
            f"回Defender={'是' if signal.escape_return_to_defender else '否'}"
        ),
        (
            f"• 实际Defender持有：{signal.actual_defender_held_days_at_open}/"
            f"{signal.defender_eligibility_days}日"
        ),
        (
            "• Defender入口黄金否决："
            f"{'已触发' if signal.immediate_defender_entry_gold_veto_triggered else '未触发'}"
            f"（{'启用' if signal.immediate_defender_entry_gold_veto_enabled else '禁用'}）"
        ),
        (
            f"• 黄金逃生硬持有：{signal.escape_held_days_at_open}/"
            f"{signal.gold_hard_hold_days}日"
        ),
        f"• 当前Momentum Top1：{_asset_label(signal.momentum_top1)}",
        f"• Top1 QM20：{signal.top1_metric_at_open:+.4f}",
        f"• Defender QM20：{signal.defender_metric_at_open:+.4f}",
        f"• Top1−Defender：{signal.metric_difference_at_open:+.4f}",
        (
            f"• 黄金入场/退出线：>{signal.gold_entry_x:+.4f} / "
            f"<{signal.gold_exit_y:+.4f}"
        ),
    ]
    if isinstance(signal, FormalW40QM40NextOpenSignal):
        if abs(signal.anchor_log_return40_at_open) > 1e-14:
            efficiency = abs(
                signal.anchor_qm40_at_open
                / signal.anchor_log_return40_at_open
            )
            efficiency_text = f"{efficiency:.3f}"
        else:
            efficiency_text = "N/A"
        recovery_block = [
            "---",
            "**QM40基础Defender恢复机制**",
            f"• 基础状态原因：{_w40_reason_text(signal.base_state_reason)}",
            (
                "• 基础Defender决策前计数："
                f"{signal.base_defender_held_days_before_decision}日；"
                f"早退最低{signal.base_defender_minimum_days}日"
            ),
            f"• 510300 R40：{signal.anchor_log_return40_at_open:+.4f}",
            f"• 510300 ER40：{efficiency_text}",
            (
                f"• 510300 QM40：{signal.anchor_qm40_at_open:+.4f}"
                f"（要求严格>{signal.qm40_recovery_threshold:+.4f}）"
            ),
            (
                "• QM40连续恢复："
                f"{signal.qm40_recovery_streak_before_decision}/"
                f"{signal.qm40_recovery_confirmation_days}日"
            ),
            (
                "• 本次QM40早退："
                f"{'已触发' if signal.qm40_early_exit_qualified else '未触发'}"
            ),
            (
                "• 本次30日保底恢复："
                f"{'已触发' if signal.fallback_recovery_qualified else '未触发'}"
            ),
        ]
        message = message + "\n\n" + "\n\n".join(recovery_block)
    performance = _format_signal_performance_lines(signal)
    return (
        message
        + "\n\n"
        + "\n\n".join(block)
        + "\n\n---\n\n"
        + "\n\n".join(performance)
    )


def format_gold_notification(
    signal: FormalGoldNextOpenSignal,
    current_weights: dict[str, float],
    orders: list[Order],
) -> str:
    message = format_integrated_notification(signal, current_weights, orders)
    override_state = (
        "激活" if signal.gold_target_active else "未激活"
    )
    block = [
        "---",
        "**Raw Gold RAQM-W5 正式覆盖层（无地板、无剪裁）**",
        f"• 双趋势确认基础状态：{signal.base_target_sleeve}",
        f"• 基础状态原因：{signal.base_state_reason}",
        (
            f"• Top1快速反转桥接："
            f"{'激活' if signal.rapid_reversal_target_active else '未激活'}"
            "（无最短持有期）"
        ),
        f"• Gold指标：{signal.gold_metric:+.3f}",
        f"• Defender指标：{signal.defender_metric:+.3f}",
        f"• Gold−Defender：{signal.metric_difference:+.3f}",
        (
            f"• 入场/退出线：>{signal.gold_entry_threshold:.2f} / "
            f"≤{signal.gold_exit_threshold:.2f}"
        ),
        (
            f"• Gold状态：{override_state}；持有计数 "
            f"{signal.gold_held_days_at_open}/{signal.gold_min_hold_days}"
        ),
    ]
    return message + "\n\n" + "\n\n".join(block)


def _build_signal(
    config: dict,
    signal_date: date,
    execution_date: date,
) -> (
    IntegratedNextOpenSignal
    | FormalGoldNextOpenSignal
    | FormalDownsideRAQMNextOpenSignal
    | FormalW40LossNextOpenSignal
    | FormalW40GoldEscapeNextOpenSignal
):
    if config.get("strategy_mode") == "w40_qm40_threshold":
        return build_w40_qm40_threshold_next_open_signal(
            Path.cwd(), signal_date, execution_date
        )
    if config.get("strategy_mode") == "w40_qm40_signed_exit":
        return build_w40_qm40_signed_exit_next_open_signal(
            Path.cwd(), signal_date, execution_date
        )
    if config.get("strategy_mode") == "w40_gold_qm20_escape":
        return build_w40_gold_escape_next_open_signal(
            Path.cwd(), signal_date, execution_date
        )
    if config.get("strategy_mode") == "w40_reversal_full_equity":
        return build_w40_full_equity_next_open_signal(
            Path.cwd(), signal_date, execution_date
        )
    if config.get("strategy_mode") == "w40_loss":
        return build_w40_loss_next_open_signal(
            Path.cwd(), signal_date, execution_date
        )
    if config.get("strategy_mode") == "downside_raqm":
        return build_downside_raqm_next_open_signal(
            Path.cwd(), signal_date, execution_date
        )
    if config.get("strategy_mode") in {
        "gold_raqm_w5",
        "absolute_stability_raw_gold",
        "confirmation_bridge_raw_gold",
    }:
        return build_formal_gold_next_open_signal(
            Path.cwd(), signal_date, execution_date
        )
    return build_integrated_next_open_signal(
        Path.cwd(), signal_date, execution_date
    )


def _prospective_record(
    signal: (
        FormalGoldNextOpenSignal
        | FormalDownsideRAQMNextOpenSignal
        | FormalW40LossNextOpenSignal
        | FormalW40GoldEscapeNextOpenSignal
    ),
    target_weights: dict[str, float],
) -> dict[str, object]:
    if isinstance(signal, FormalW40GoldEscapeNextOpenSignal):
        record = {
            "strategy_id": signal.strategy_id,
            "signal_date": signal.signal_date.isoformat(),
            "execution_date": signal.execution_date.isoformat(),
            "target_sleeve": signal.target_sleeve,
            "target_candidate": signal.target_candidate,
            "target_weights": target_weights,
            "target_cash_weight": signal.target_cash_weight,
            "state_reason": signal.state_reason,
            "w40_downside_log_loss": signal.w40_downside_log_loss,
            "w40_loss_percentile": signal.w40_loss_percentile,
            "escape_active": signal.escape_active,
            "escape_entry": signal.escape_entry,
            "escape_return_to_defender": signal.escape_return_to_defender,
            "escape_entry_asset": signal.escape_entry_asset,
            "escape_held_days_at_open": signal.escape_held_days_at_open,
            "actual_defender_held_days_at_open": (
                signal.actual_defender_held_days_at_open
            ),
            "momentum_top1": signal.momentum_top1,
            "top1_metric_at_open": signal.top1_metric_at_open,
            "defender_metric_at_open": signal.defender_metric_at_open,
            "metric_difference_at_open": signal.metric_difference_at_open,
            "gold_entry_x": signal.gold_entry_x,
            "gold_exit_y": signal.gold_exit_y,
            "immediate_defender_entry_gold_veto_enabled": (
                signal.immediate_defender_entry_gold_veto_enabled
            ),
            "immediate_defender_entry_gold_veto_triggered": (
                signal.immediate_defender_entry_gold_veto_triggered
            ),
        }
        if isinstance(signal, FormalW40QM40NextOpenSignal):
            record.update(
                {
                    "w40_percentile_history": signal.w40_percentile_history,
                    "base_state_reason": signal.base_state_reason,
                    "base_defender_held_days_before_decision": (
                        signal.base_defender_held_days_before_decision
                    ),
                    "anchor_qm40_at_open": signal.anchor_qm40_at_open,
                    "anchor_log_return40_at_open": (
                        signal.anchor_log_return40_at_open
                    ),
                    "qm40_recovery_threshold": signal.qm40_recovery_threshold,
                    "qm40_recovery_streak_before_decision": (
                        signal.qm40_recovery_streak_before_decision
                    ),
                    "qm40_early_exit_qualified": (
                        signal.qm40_early_exit_qualified
                    ),
                    "fallback_recovery_qualified": (
                        signal.fallback_recovery_qualified
                    ),
                }
            )
        return record
    if isinstance(signal, FormalW40LossNextOpenSignal):
        return {
            "strategy_id": signal.strategy_id,
            "signal_date": signal.signal_date.isoformat(),
            "execution_date": signal.execution_date.isoformat(),
            "target_sleeve": signal.target_sleeve,
            "target_weights": target_weights,
            "target_cash_weight": signal.target_cash_weight,
            "state_reason": signal.state_reason,
            "held_days_at_open": signal.held_days_at_open,
            "w40_downside_log_loss": signal.w40_downside_log_loss,
            "w40_loss_percentile": signal.w40_loss_percentile,
            "entry_confirmation_streak": signal.entry_confirmation_streak,
            "recovery_confirmation_streak": signal.recovery_confirmation_streak,
            "momentum_lock_days": signal.momentum_lock_days,
            "defender_lock_days": signal.defender_lock_days,
        }
    if isinstance(signal, FormalDownsideRAQMNextOpenSignal):
        return {
            "strategy_id": signal.strategy_id,
            "signal_date": signal.signal_date.isoformat(),
            "execution_date": signal.execution_date.isoformat(),
            "target_sleeve": signal.target_sleeve,
            "target_weights": target_weights,
            "target_cash_weight": signal.target_cash_weight,
            "state_reason": signal.state_reason,
            "held_days_at_open": signal.held_days_at_open,
            "downside_raqm_percentile": signal.downside_raqm_percentile,
            "downside_raqm_30": signal.downside_raqm_30,
            "downside_raqm_40": signal.downside_raqm_40,
            "entry_confirmation_streak": signal.entry_confirmation_streak,
            "recovery_confirmation_streak": signal.recovery_confirmation_streak,
            "momentum_lock_days": signal.momentum_lock_days,
            "defender_lock_days": signal.defender_lock_days,
        }
    return {
        "strategy_id": signal.strategy_id,
        "signal_date": signal.signal_date.isoformat(),
        "execution_date": signal.execution_date.isoformat(),
        "target_sleeve": signal.target_sleeve,
        "target_weights": target_weights,
        "target_cash_weight": signal.target_cash_weight,
        "state_reason": signal.state_reason,
        "base_target_sleeve": signal.base_target_sleeve,
        "base_state_reason": signal.base_state_reason,
        "risk_on_confirmation_streak": signal.risk_on_confirmation_streak,
        "risk_off_confirmation_streak": signal.risk_off_confirmation_streak,
        "risk_on_confirmation_days": signal.risk_on_confirmation_days,
        "risk_off_confirmation_days": signal.risk_off_confirmation_days,
        "anchor_log_return_120": signal.anchor_log_return_120,
        "held_momentum_asset": signal.held_momentum_asset,
        "held_asset_log_return_120": signal.held_asset_log_return_120,
        "emergency_log_return_5": signal.emergency_log_return_5,
        "emergency_downside_volatility_20": (
            signal.emergency_downside_volatility_20
        ),
        "emergency_downside_threshold_q95": (
            signal.emergency_downside_threshold_q95
        ),
        "rapid_reversal_current_active": signal.rapid_reversal_current_active,
        "rapid_reversal_target_active": signal.rapid_reversal_target_active,
        "rapid_reversal_asset": signal.rapid_reversal_asset,
        "rapid_reversal_metric": signal.rapid_reversal_metric,
        "rapid_reversal_defender_metric": signal.rapid_reversal_defender_metric,
        "rapid_reversal_difference": signal.rapid_reversal_difference,
        "rapid_reversal_entry_threshold": (
            signal.rapid_reversal_entry_threshold
        ),
        "rapid_reversal_exit_threshold": (
            signal.rapid_reversal_exit_threshold
        ),
        "gold_current_active": signal.gold_current_active,
        "gold_target_active": signal.gold_target_active,
        "gold_held_days_at_open": signal.gold_held_days_at_open,
        "gold_metric": signal.gold_metric,
        "defender_metric": signal.defender_metric,
        "metric_difference": signal.metric_difference,
        "gold_entry_threshold": signal.gold_entry_threshold,
        "gold_exit_threshold": signal.gold_exit_threshold,
    }


def run(
    config: dict,
    *,
    dry_run: bool = False,
    notification_only: bool = False,
    skip_sync: bool = False,
    notifier_factory: Callable[[], DingTalkNotifier] = DingTalkNotifier,
    peak_warning_evaluator: Callable[[str, date], PeakWarning] = evaluate_peak_warning,
) -> tuple[
    IntegratedNextOpenSignal
    | FormalGoldNextOpenSignal
    | FormalDownsideRAQMNextOpenSignal
    | FormalW40LossNextOpenSignal
    | FormalW40GoldEscapeNextOpenSignal,
    str,
]:
    if dry_run and notification_only:
        raise ValueError("dry_run and notification_only are mutually exclusive")
    today = date.today()
    production = not dry_run and not notification_only
    if production and not _is_sse_trading_day(today):
        raise RuntimeError(f"{today} is not an SSE trading day; no signal sent")

    asset_pool = list(config["asset_pool"])
    if not skip_sync:
        _sync_and_check(asset_pool, today)
    signal_date = _latest_common_data_date(asset_pool, today)
    execution_date = _next_entry_date(signal_date)
    strategy_name = str(config["strategy_name"])

    current_state = read_position(strategy_name)
    if production:
        current_state = _backfill_open_prices(current_state, today, strategy_name)
    priced_state = _priced_state_as_of(current_state, signal_date)
    signal = _build_signal(config, signal_date, execution_date)
    target_weights = {
        asset: float(weight)
        for asset, weight in signal.target_weights.items()
        if float(weight) > 1e-14
    }
    orders = diff(target_weights, priced_state.weights)
    if isinstance(signal, FormalGoldNextOpenSignal):
        message = format_gold_notification(signal, priced_state.weights, orders)
    elif isinstance(signal, FormalDownsideRAQMNextOpenSignal):
        message = format_downside_raqm_notification(
            signal, priced_state.weights, orders
        )
    elif isinstance(signal, FormalW40GoldEscapeNextOpenSignal):
        warning_asset = signal.momentum_top1
        try:
            peak_warning = peak_warning_evaluator(
                warning_asset, signal.signal_date
            )
        except Exception as exc:
            peak_warning = PeakWarning(
                asset=warning_asset,
                signal_date=signal.signal_date,
                triggered=False,
                current_close=float("nan"),
                prior_high200=float("nan"),
                close20ago=float("nan"),
                current_volume=float("nan"),
                prior_volume_median20=float("nan"),
                price_breakout=float("nan"),
                price_return20=float("nan"),
                volume_ratio20=float("nan"),
                share_filter_required=warning_asset == "159915.SZ",
                share_data_available=False,
                share_flow20=None,
                reason=f"预警计算不可用（{type(exc).__name__}）",
            )
        message = format_w40_gold_escape_notification(
            signal, priced_state.weights, orders, peak_warning
        )
    elif isinstance(signal, FormalW40LossNextOpenSignal):
        momentum_assets = [
            asset
            for asset, weight in signal.momentum.effective_weights.items()
            if float(weight) > 1e-14
        ]
        if len(momentum_assets) != 1:
            raise AssertionError(
                "W40 peak warning requires exactly one Momentum target"
            )
        warning_asset = momentum_assets[0]
        try:
            peak_warning = peak_warning_evaluator(
                warning_asset, signal.signal_date
            )
        except Exception as exc:
            peak_warning = PeakWarning(
                asset=warning_asset,
                signal_date=signal.signal_date,
                triggered=False,
                current_close=float("nan"),
                prior_high200=float("nan"),
                close20ago=float("nan"),
                current_volume=float("nan"),
                prior_volume_median20=float("nan"),
                price_breakout=float("nan"),
                price_return20=float("nan"),
                volume_ratio20=float("nan"),
                share_filter_required=warning_asset == "159915.SZ",
                share_data_available=False,
                share_flow20=None,
                reason=f"预警计算不可用（{type(exc).__name__}）",
            )
        message = format_w40_loss_notification(
            signal, priced_state.weights, orders, peak_warning
        )
    else:
        message = format_integrated_notification(signal, priced_state.weights, orders)
    if dry_run:
        message = "## 🧪 本地演练（未发送、未写持仓）\n\n" + message
    elif notification_only:
        message = "## 🧪 通知测试（未写持仓）\n\n" + message
    print(message)

    if bool(config.get("enable_dingtalk", True)) and not dry_run:
        notifier = notifier_factory()
        if notification_only:
            notifier.send(
                message,
                title="Momentum × Defender 通知测试",
                alert_text="Momentum × Defender 信号链路测试，请核对，暂不执行。",
            )
        else:
            title = (
                "W40 × QM40 Defender × 黄金调仓信号"
                if isinstance(signal, FormalW40QM40NextOpenSignal)
                else "W40 × 黄金QM20破锁调仓信号"
                if isinstance(signal, FormalW40GoldEscapeNextOpenSignal)
                else "冻结510300单一40日下跌幅度调仓信号"
                if isinstance(signal, FormalW40LossNextOpenSignal)
                else (
                    "冻结510300下行DRAQM调仓信号"
                    if isinstance(signal, FormalDownsideRAQMNextOpenSignal)
                    else (
                        "无锁确认 × 快速反转 × Raw Gold 调仓信号"
                        if isinstance(signal, FormalGoldNextOpenSignal)
                        else "Momentum × Defender 调仓信号"
                    )
                )
            )
            notifier.send(message, title=title)
        print("DingTalk notification sent.")
    elif dry_run:
        print("Dry run — DingTalk not called.")
    else:
        print("DingTalk disabled by config.")

    if not production:
        print("Position state unchanged.")
        return signal, message

    is_rebalance = any(order.action in {"buy", "sell"} for order in orders)
    if is_rebalance:
        updated = _save_or_update_rebalance_target(
            current_state,
            target_weights,
            execution_date,
            signal_date,
            strategy_name,
        )
        action = "Pending position updated" if updated else "Position saved"
        print(f"{action}: {target_weights}, next_entry_date={execution_date}")
    else:
        print(f"Hold signal — position unchanged: {priced_state.weights}")
    ledger_path = config.get("prospective_ledger_path")
    if ledger_path and isinstance(
        signal,
        (
            FormalGoldNextOpenSignal,
            FormalDownsideRAQMNextOpenSignal,
            FormalW40LossNextOpenSignal,
            FormalW40GoldEscapeNextOpenSignal,
        ),
    ):
        appended = append_signal_record(
            Path(str(ledger_path)),
            _prospective_record(signal, target_weights),
        )
        print(
            "Prospective signal ledger appended."
            if appended
            else "Prospective signal ledger already contained this signal."
        )
    return signal, message


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run integrated Momentum/Defender daily signal"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "strategy/configs/momentum_defender_w40_gold_escape.yaml"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute and print the real signal without sending or writing state",
    )
    parser.add_argument(
        "--notification-only",
        action="store_true",
        help="Send a labelled test notification without writing state",
    )
    parser.add_argument(
        "--skip-sync",
        action="store_true",
        help="Use already-synchronized local data (intended for local verification)",
    )
    args = parser.parse_args()
    run(
        _load_config(args.config),
        dry_run=args.dry_run,
        notification_only=args.notification_only,
        skip_sync=args.skip_sync,
    )


if __name__ == "__main__":
    main()
