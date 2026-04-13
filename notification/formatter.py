"""Format orders into readable messages."""

from __future__ import annotations

from datetime import date

from execution.interfaces import Order


def format_orders(
    orders: list[Order],
    target_weights: dict[str, float],
    run_date: date | None = None,
) -> str:
    """Format orders as a markdown message for notification.

    Returns a markdown string with a summary header and order table.
    """
    if run_date is None:
        run_date = date.today()

    if not orders:
        return f"**{run_date} 调仓信号**\n\n无需调仓"

    buys = [o for o in orders if o.action == "buy"]
    sells = [o for o in orders if o.action == "sell"]
    holds = [o for o in orders if o.action == "hold"]

    lines = [
        f"**{run_date} 调仓信号**\n",
        f"买入: {len(buys)} | 卖出: {len(sells)} | 持有: {len(holds)}\n",
        "| 标的 | 操作 | 变动 | 目标权重 |",
        "|------|------|------|----------|",
    ]

    action_map = {"buy": "买入", "sell": "卖出", "hold": "持有"}

    for o in sorted(orders, key=lambda x: x.asset):
        action_cn = action_map[o.action]
        delta_str = f"{o.weight_delta:+.2%}"
        target = target_weights.get(o.asset, 0.0)
        target_str = f"{target:.2%}"
        lines.append(f"| {o.asset} | {action_cn} | {delta_str} | {target_str} |")

    return "\n".join(lines)
