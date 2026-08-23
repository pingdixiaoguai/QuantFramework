"""Daily live runner for the formal C2 plus Gold RAQM-W5 strategy.

The runner synchronizes the combined 11-ETF universe, replays both sleeves
through the latest close, advances C2 and the frozen Gold override exactly one
market open, sends the resulting allocation, and persists the formal ledger.
Passing the base-C2 config explicitly retains the rollback path.
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path
from typing import Callable

from execution.interfaces import Order, diff
from execution.position import read_position
from notification.dingtalk import DingTalkNotifier
from notification.formatter import ASSET_NAMES
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
from strategy.prospective_ledger import append_signal_record


DEFENDER_ASSET_NAMES = {
    "512890.SH": "红利低波ETF",
    "159545.SZ": "恒生红利低波ETF",
    "513530.SH": "港股通红利ETF",
    "515080.SH": "中证红利ETF",
    "510880.SH": "红利ETF",
    "563020.SH": "低波红利ETF",
    "511260.SH": "十年国债ETF",
}
ALL_ASSET_NAMES = {**ASSET_NAMES, **DEFENDER_ASSET_NAMES}


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
    emergency_text = (
        f"触发（{_asset_label(signal.emergency_asset)} cap="
        f"{signal.emergency_cap:.0%}）"
        if signal.emergency_alert
        else (
            f"未触发（{_asset_label(signal.emergency_asset)} cap="
            f"{signal.emergency_cap:.0%}）"
        )
    )
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
        f"**30日状态锁计数：** {signal.held_days_at_open} 个交易日",
        "",
        f"**当前实盘持仓：** {_allocation(current_weights, current_cash)}",
        f"**实际目标：** {_allocation(target, signal.target_cash_weight)}",
        "",
        "**调仓指令**",
        *_order_lines(orders),
        "",
        "**模型诊断**",
        (
            f"• 沪深300 40日收益 {signal.slow_gate_return:+.2%}，"
            f"慢门控={'Momentum' if signal.slow_gate_risk_on else 'Defender'}"
        ),
        f"• 紧急波动 cap：{emergency_text}",
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
        "**Gold RAQM-W5 正式覆盖层**",
        f"• 基础C2下一开盘：{signal.base_c2_target_sleeve}",
        f"• 基础C2原因：{signal.base_c2_state_reason}",
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
) -> IntegratedNextOpenSignal | FormalGoldNextOpenSignal:
    if config.get("strategy_mode") == "gold_raqm_w5":
        return build_formal_gold_next_open_signal(
            Path.cwd(), signal_date, execution_date
        )
    return build_integrated_next_open_signal(
        Path.cwd(), signal_date, execution_date
    )


def _prospective_record(
    signal: FormalGoldNextOpenSignal,
    target_weights: dict[str, float],
) -> dict[str, object]:
    return {
        "strategy_id": signal.strategy_id,
        "signal_date": signal.signal_date.isoformat(),
        "execution_date": signal.execution_date.isoformat(),
        "target_sleeve": signal.target_sleeve,
        "target_weights": target_weights,
        "target_cash_weight": signal.target_cash_weight,
        "state_reason": signal.state_reason,
        "base_c2_target_sleeve": signal.base_c2_target_sleeve,
        "base_c2_state_reason": signal.base_c2_state_reason,
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
) -> tuple[IntegratedNextOpenSignal | FormalGoldNextOpenSignal, str]:
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
    message = (
        format_gold_notification(signal, priced_state.weights, orders)
        if isinstance(signal, FormalGoldNextOpenSignal)
        else format_integrated_notification(signal, priced_state.weights, orders)
    )
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
                "C2 × Gold RAQM-W5 调仓信号"
                if isinstance(signal, FormalGoldNextOpenSignal)
                else "Momentum × Defender 调仓信号"
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
    if ledger_path and isinstance(signal, FormalGoldNextOpenSignal):
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
            "strategy/configs/momentum_defender_c2_gold_raqm_w5.yaml"
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
