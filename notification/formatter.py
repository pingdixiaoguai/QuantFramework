"""Format notification context into DingTalk markdown messages."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from execution.interfaces import Order

ASSET_NAMES: dict[str, str] = {
    # quality_momentum_top1 pool
    "510300.SH": "沪深300",
    "159915.SZ": "创业板",
    "513100.SH": "纳指ETF",
    "518880.SH": "黄金ETF",
    # industry_quality_momentum_top5 pool — Financials / Real Estate
    "512880.SH": "证券",
    "512800.SH": "银行",
    "512200.SH": "房地产",
    # Consumer / Home Appliance
    "512690.SH": "酒",
    "159928.SZ": "消费",
    "159996.SZ": "家电",
    # Healthcare
    "512010.SH": "医药",
    "512170.SH": "医疗",
    # TMT
    "512480.SH": "半导体",
    "515880.SH": "通信",
    "512720.SH": "计算机",
    "159939.SZ": "信息技术",
    "512980.SH": "传媒",
    # Advanced Manufacturing
    "512660.SH": "军工",
    "515030.SH": "新能源车",
    "516110.SH": "汽车",
    "515790.SH": "光伏",
    "562800.SH": "风电",
    # Cyclicals
    "515220.SH": "煤炭",
    "512400.SH": "有色金属",
    "515210.SH": "钢铁",
    "159870.SZ": "化工",
    # Utilities / Agri / Infra
    "159611.SZ": "电力",
    "159825.SZ": "农业",
    "516970.SH": "基建",
}


@dataclass
class StrategySignalView:
    """One strategy's read-only cross-sectional signal diagnostics."""

    strategy_name: str
    target: str | None
    scores: dict[str, float]
    confidence: dict[str, float]
    expected_assets: list[str]


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
    rebalance_days: int | None = None
    production_signal: StrategySignalView | None = None
    shadow_signal: StrategySignalView | None = None


def _asset_label(asset: str, asset_names: dict[str, str]) -> str:
    """Return ticker prefix + Chinese name, or just ticker if unknown."""
    ticker = asset.split(".")[0]
    name = asset_names.get(asset, asset)
    return f"{ticker} {name}"


def _fmt_pct(value: float) -> str:
    """Format a float as a signed percentage string, e.g. +0.89%."""
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.2%}"


def _target_from_weights(weights: dict[str, float]) -> str | None:
    positive = {asset: weight for asset, weight in weights.items() if weight > 0}
    return max(positive, key=positive.get) if positive else None


def _weights_label(weights: dict[str, float], asset_names: dict[str, str]) -> str:
    if not weights:
        return "空仓"
    return "、".join(
        f"{_asset_label(asset, asset_names)} {weight:.0%}"
        for asset, weight in weights.items()
        if weight > 0
    ) or "空仓"


def _build_execution_section(ctx: NotificationContext, is_rebalance: bool) -> str:
    """Separate the raw production signal from the executable target."""
    production_target = (
        ctx.production_signal.target if ctx.production_signal is not None else None
    )
    actual_target = _target_from_weights(ctx.target_weights)
    lines = [f"**实际执行（{'调仓' if is_rebalance else '继续持有'}）**"]
    lines.append(f"• 当前持仓：{_weights_label(ctx.current_weights, ctx.asset_names)}")
    lines.append(
        "• 生产策略原始目标："
        + (
            _asset_label(production_target, ctx.asset_names)
            if production_target is not None
            else "数据不足"
        )
    )
    lines.append(
        "• 实际交易目标："
        + (
            _asset_label(actual_target, ctx.asset_names)
            if actual_target is not None
            else "空仓"
        )
    )

    if (
        ctx.holding_days is not None
        and ctx.rebalance_days is not None
        and ctx.current_weights
    ):
        lines.append(
            f"• 持有期：{ctx.holding_days}/{ctx.rebalance_days} 交易日"
        )
    if production_target is not None and production_target != actual_target:
        lines.append("• 持有期约束：原始目标暂未执行")

    if is_rebalance:
        for order in ctx.orders:
            if order.action == "hold":
                continue
            label = _asset_label(order.asset, ctx.asset_names)
            current = ctx.current_weights.get(order.asset, 0.0)
            target = ctx.target_weights.get(order.asset, 0.0)
            action_cn = "卖出" if order.action == "sell" else "买入"
            lines.append(f"• {action_cn}：{label}　{current:.0%} → {target:.0%}")
    elif ctx.current_weights:
        if ctx.position_return is None:
            return_text = "待更新"
        else:
            return_text = _fmt_pct(ctx.position_return)
        lines.append(f"• 当前持仓收益：**{return_text}**")
    else:
        lines.append("• 今日操作：无")

    return "\n\n".join(lines)


def _ranked_assets(signal: StrategySignalView) -> list[str]:
    return sorted(signal.scores, key=signal.scores.get, reverse=True)


def _build_strategy_block(
    title: str,
    signal: StrategySignalView,
    asset_names: dict[str, str],
) -> list[str]:
    ranked = _ranked_assets(signal)
    lines = [f"**{title} · {signal.strategy_name}**"]
    if not ranked:
        lines.append("• 无有效信号数据")
    else:
        for rank, asset in enumerate(ranked[:2], start=1):
            confidence = signal.confidence.get(asset)
            confidence_text = (
                f"　|　相对强度 {confidence:.1%}"
                if confidence is not None
                else ""
            )
            lines.append(
                f"• #{rank} {_asset_label(asset, asset_names)}"
                f"　|　得分 {_fmt_pct(signal.scores[asset])}{confidence_text}"
            )
        if len(ranked) >= 2:
            first_confidence = signal.confidence.get(ranked[0])
            second_confidence = signal.confidence.get(ranked[1])
            if first_confidence is not None and second_confidence is not None:
                lead = first_confidence - second_confidence
                lines.append(f"• Top1 领先：{lead * 100:.1f} 个百分点")

    missing = [asset for asset in signal.expected_assets if asset not in signal.scores]
    if missing:
        missing_labels = "、".join(_asset_label(a, asset_names) for a in missing)
        lines.append(
            f"• ⚠️ 数据完整性：{len(signal.scores)}/{len(signal.expected_assets)}"
            f"，缺少 {missing_labels}"
        )
    else:
        lines.append(
            f"• 数据完整性：{len(signal.scores)}/{len(signal.expected_assets)}"
        )
    return lines


def _build_signal_comparison(ctx: NotificationContext) -> str:
    """Build a symmetric production-versus-shadow ranking comparison."""
    if ctx.production_signal is None and ctx.shadow_signal is None:
        return ""

    lines = ["**策略对照**"]
    if ctx.production_signal is not None:
        lines.extend(
            _build_strategy_block("生产策略", ctx.production_signal, ctx.asset_names)
        )
    if ctx.shadow_signal is not None:
        lines.extend(
            _build_strategy_block("影子策略", ctx.shadow_signal, ctx.asset_names)
        )

    if ctx.production_signal is not None and ctx.shadow_signal is not None:
        production = ctx.production_signal
        shadow = ctx.shadow_signal
        lines.append("**核心差异**")
        if production.target == shadow.target and production.target is not None:
            lines.append(
                f"• 两个策略原始目标一致：{_asset_label(production.target, ctx.asset_names)}"
            )
        else:
            production_label = (
                _asset_label(production.target, ctx.asset_names)
                if production.target is not None
                else "数据不足"
            )
            shadow_label = (
                _asset_label(shadow.target, ctx.asset_names)
                if shadow.target is not None
                else "数据不足"
            )
            lines.append(
                f"• 原始目标不同：生产 {production_label}　|　影子 {shadow_label}"
            )

        production_ranks = {
            asset: rank
            for rank, asset in enumerate(_ranked_assets(production), start=1)
        }
        shadow_ranks = {
            asset: rank
            for rank, asset in enumerate(_ranked_assets(shadow), start=1)
        }
        common_assets = [
            asset
            for asset in production.expected_assets
            if asset in production_ranks and asset in shadow_ranks
        ]
        changed = [
            asset
            for asset in common_assets
            if production_ranks[asset] != shadow_ranks[asset]
        ]
        if changed:
            for asset in changed:
                lines.append(
                    f"• {_asset_label(asset, ctx.asset_names)}："
                    f"生产 #{production_ranks[asset]} → 影子 #{shadow_ranks[asset]}"
                )
        elif common_assets:
            lines.append("• 共同标的排序一致")

        lines.append("影子信号仅作观察，不改变生产策略交易")

    lines.append("相对强度为当日横截面 softmax，不代表上涨概率")
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

    lines = [f"**同期表现**（自 {since} 开盘起）"]

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

    execution = _build_execution_section(ctx, is_rebalance)
    comparison = _build_signal_comparison(ctx)
    benchmark = _build_benchmark_section(ctx)
    ytd = _build_ytd_line(ctx)

    parts = [header, "---", execution]
    if comparison:
        parts += ["---", comparison]
    if benchmark:
        parts += ["---", benchmark]
    parts += ["---", ytd]

    return "\n\n".join(parts)
