"""Format notification context into DingTalk markdown messages."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from execution.interfaces import Order

ASSET_NAMES: dict[str, str] = {
    "510300.SH": "沪深300",
    "159915.SZ": "创业板",
    "513100.SH": "纳指ETF",
    "518880.SH": "黄金ETF",
}


@dataclass
class NotificationContext:
    strategy_name: str
    signal_date: date
    orders: list[Order]
    target_weights: dict[str, float]
    current_weights: dict[str, float]
    entry_date: date | None           # actual entry day (signal+1 trading day)
    holding_days: int | None          # trading days held
    position_return: float | None     # current position weighted return
    benchmark_returns: dict[str, float]  # asset → same-period return
    ytd_return: float | None          # YTD cumulative return
    asset_names: dict[str, str]       # asset → Chinese name
    asset_factor_values: dict[str, dict[str, float]] | None = None  # asset → factor → value


def _asset_label(asset: str, asset_names: dict[str, str]) -> str:
    """Return ticker prefix + Chinese name, or just ticker if unknown."""
    ticker = asset.split(".")[0]
    name = asset_names.get(asset, asset)
    return f"{ticker} {name}"


def _fmt_pct(value: float) -> str:
    """Format a float as a signed percentage string, e.g. +0.89%."""
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.2%}"


def _build_position_section(ctx: NotificationContext) -> str:
    """Build the 当前持仓 block for the hold case."""
    lines = ["**当前持仓**"]
    for asset, weight in ctx.current_weights.items():
        label = _asset_label(asset, ctx.asset_names)
        weight_str = f"{weight:.0%}"

        if ctx.holding_days is None or ctx.entry_date is None:
            days_str = "—"
            ret_str = "—"
        elif ctx.position_return is None:
            days_str = str(ctx.holding_days)
            ret_str = "待更新"
        else:
            days_str = str(ctx.holding_days)
            ret_str = _fmt_pct(ctx.position_return)

        lines.append(
            f"• {label}　{weight_str}　|　已持有 {days_str} 交易日　|　收益 **{ret_str}**"
        )
    return "\n\n".join(lines)


def _build_rebalance_section(ctx: NotificationContext) -> str:
    """Build the 调仓指令 block."""
    lines = ["**调仓指令**"]
    for order in ctx.orders:
        if order.action == "hold":
            continue
        label = _asset_label(order.asset, ctx.asset_names)
        current = ctx.current_weights.get(order.asset, 0.0)
        target = ctx.target_weights.get(order.asset, 0.0)
        action_cn = "卖出" if order.action == "sell" else "买入"
        lines.append(
            f"• {action_cn}：{label}　{current:.0%} → {target:.0%}"
        )
    return "\n\n".join(lines)


def _build_alpha_section(ctx: NotificationContext) -> str:
    """Build the 调仓超额 block showing factor scores, benchmark returns, and alpha for all candidates."""
    if not ctx.asset_factor_values and not ctx.benchmark_returns:
        return ""

    lines = ["**调仓依据（标的对比）**"]

    # Collect all assets from factor values or benchmark returns
    all_assets = list(
        dict.fromkeys(
            list(ctx.asset_factor_values or {})
            + list(ctx.benchmark_returns or {})
        )
    )
    if not all_assets:
        return ""

    # Determine factor names from the first asset
    factor_names: list[str] = []
    if ctx.asset_factor_values:
        first = next(iter(ctx.asset_factor_values.values()))
        factor_names = list(first.keys())

    FACTOR_DISPLAY = {"momentum": "动量", "volatility": "波动率"}

    # Compute weighted return of selected targets for alpha calculation
    target_return: float | None = None
    if ctx.benchmark_returns:
        total_w = 0.0
        weighted_ret = 0.0
        for asset, w in ctx.target_weights.items():
            if w > 0 and asset in ctx.benchmark_returns:
                weighted_ret += w * ctx.benchmark_returns[asset]
                total_w += w
        if total_w > 0:
            target_return = weighted_ret / total_w

    for asset in all_assets:
        label = _asset_label(asset, ctx.asset_names)
        target_w = ctx.target_weights.get(asset)
        is_target = target_w is not None and target_w > 0
        marker = "★" if is_target else "　"

        parts = [f"{marker} {label}"]

        # Factor values
        if ctx.asset_factor_values and asset in ctx.asset_factor_values:
            for fname in factor_names:
                val = ctx.asset_factor_values[asset].get(fname)
                if val is not None:
                    display_name = FACTOR_DISPLAY.get(fname, fname)
                    parts.append(f"{display_name} {_fmt_pct(val)}")

        # Same-period benchmark return + alpha vs target
        if asset in ctx.benchmark_returns:
            ret = ctx.benchmark_returns[asset]
            parts.append(f"同期 {_fmt_pct(ret)}")
            if target_return is not None and not is_target:
                alpha = ret - target_return
                parts.append(f"超额 {_fmt_pct(alpha)}")

        lines.append("• " + "　|　".join(parts))

    lines.append("")
    lines.append("★ = 调仓目标　|　超额 = 该标的同期收益 − 目标同期收益")

    return "\n\n".join(lines)


def _build_benchmark_section(ctx: NotificationContext) -> str:
    """Build the 同期对比 block."""
    if not ctx.benchmark_returns:
        return ""

    # Determine comparison start date
    if ctx.entry_date is not None:
        since = ctx.entry_date.isoformat()
    elif ctx.signal_date is not None:
        since = ctx.signal_date.isoformat()
    else:
        since = "—"

    lines = [f"**同期对比**（自 {since} 开盘起）"]

    position_ret = ctx.position_return or 0.0

    for asset, bench_ret in ctx.benchmark_returns.items():
        # Skip the asset that is the current holding (already shown in position)
        if asset in ctx.current_weights and len(ctx.current_weights) == 1:
            continue
        label = _asset_label(asset, ctx.asset_names)
        bench_str = _fmt_pct(bench_ret)
        diff = bench_ret - position_ret
        if diff > 0:
            diff_str = f"超出持仓 {_fmt_pct(diff)} ↑"
        else:
            diff_str = f"落后持仓 {_fmt_pct(diff)} ↓"
        lines.append(f"• {label}　{bench_str}　|　{diff_str}")

    return "\n\n".join(lines)


def _build_ytd_line(ctx: NotificationContext) -> str:
    if ctx.ytd_return is None:
        return "**年初至今：** 数据不足"
    return f"**年初至今：** {_fmt_pct(ctx.ytd_return)}"


def format_notification(ctx: NotificationContext) -> str:
    """Format a rich DingTalk markdown message from a NotificationContext.

    Uses the rebalance layout when orders contain buy/sell; otherwise the hold layout.
    """
    is_rebalance = any(o.action in ("buy", "sell") for o in ctx.orders)

    header = (
        f"## 📊 {ctx.strategy_name} 信号\n\n"
        f"**信号日期：** {ctx.signal_date.isoformat()}\n"
        "基于当日收盘价，建议次日开盘调仓"
    )

    if is_rebalance:
        middle = _build_rebalance_section(ctx)
    else:
        middle = _build_position_section(ctx)

    benchmark = _build_benchmark_section(ctx)
    alpha = _build_alpha_section(ctx) if is_rebalance else ""
    ytd = _build_ytd_line(ctx)

    parts = [header, "---", middle]
    if alpha:
        parts += ["---", alpha]
    if benchmark:
        parts += ["---", benchmark]
    parts += ["---", ytd]

    return "\n\n".join(parts)
