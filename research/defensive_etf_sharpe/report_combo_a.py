"""QuantStats HTML report for combo A (risk-only universe, rank tilt 0.70).

Benchmark is the fixed-weight buy-and-hold base: monthly deposits invested in
fixed 35/40/15/10 weights, never rebalanced, 1-share lot to avoid pathological
under-allocation from 100-share lots on small monthly deposits.

Run with: uv run python -m research.defensive_etf_sharpe.report_combo_a
"""

from __future__ import annotations

import html
import io
import re
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .calendar_ablation import (
    BASELINE_TARGET,
    CALENDAR_TRIGGER,
    EXTRA_POOL_ASSETS,
    _extended_metrics,
    _risk_only_tilt_builder,
)
from .engine import load_market_data
from .rebalance_timing import simulate_fixed_buyandhold
from .strategy import CASH_ASSET, load_confirmed_market
from .threshold_rebalance import simulate_threshold_rebalance


ROOT = Path(__file__).parent
OUTPUT = ROOT / "combo_a_report"

COST_RATE = 0.0005
MONTHLY_DEPOSIT = 20_000.0
MIN_REBALANCE_NOTIONAL = 10_000.0
LAMBDA = 0.70

SHORT_NAMES = {
    "512890.SH": "红利低波ETF",
    "511260.SH": "10年国债ETF",
    "511360.SH": "短融ETF",
    "511880.SH": "银华日利",
}


def _run_strategy(data, targets):
    return simulate_threshold_rebalance(
        data,
        targets,
        CALENDAR_TRIGGER,
        initial_target=dict(BASELINE_TARGET),
        cash_asset=CASH_ASSET,
        monthly_deposit=MONTHLY_DEPOSIT,
        cost_rate=COST_RATE,
        min_rebalance_notional=MIN_REBALANCE_NOTIONAL,
    )


def _fmt_positions(positions: dict[str, float]) -> str:
    if not positions:
        return "空仓"
    parts = [
        f"{SHORT_NAMES.get(asset, asset)} {weight:.1%}"
        for asset, weight in sorted(positions.items(), key=lambda item: -item[1])
        if weight > 0.0005
    ]
    return " / ".join(parts)


def _max_dd_date(result) -> str:
    returns = result.daily["return"].dropna().astype(float)
    curve = (1.0 + returns).cumprod()
    peak = curve.expanding().max()
    dd = curve / peak - 1.0
    return dd.idxmin().date().isoformat()


def _drawdown_table(result, top: int = 10) -> pd.DataFrame:
    """Top drawdown periods with holdings and trade context."""
    returns = result.daily["return"].dropna().astype(float)
    curve = (1.0 + returns).cumprod()
    peak = curve.expanding().max()
    dd_series = curve / peak - 1.0

    underwater = dd_series < -1e-6
    if not underwater.any():
        return pd.DataFrame()
    group = (underwater != underwater.shift()).cumsum()
    periods: list[dict[str, object]] = []
    for _, sub in dd_series[underwater].groupby(group[underwater]):
        first_under = sub.index[0]
        peak_value = peak.loc[first_under]
        candidates = curve.loc[curve.index <= first_under]
        start = candidates[candidates >= peak_value * (1 - 1e-9)].index[-1]
        valley = sub.idxmin()
        after_valley = curve.loc[curve.index > valley]
        recovered = after_valley[after_valley >= peak_value * (1 - 1e-9)]
        if not recovered.empty:
            end = recovered.index[0]
            recovered_flag = True
        else:
            end = curve.index[-1]
            recovered_flag = False
        periods.append({
            "start": start,
            "valley": valley,
            "end": end,
            "recovered": recovered_flag,
            "depth_pct": float(dd_series.loc[valley]) * 100.0,
            "days": int((end - start).days),
        })
    details = pd.DataFrame(periods).sort_values("depth_pct").head(top)

    daily = result.daily
    rows: list[dict[str, object]] = []
    for rank, (_, period) in enumerate(details.iterrows(), start=1):
        start = pd.Timestamp(period["start"])
        valley = pd.Timestamp(period["valley"])
        end = pd.Timestamp(period["end"])
        recovered = bool(period["recovered"])

        at_peak = daily.loc[daily.index <= start, "positions"]
        at_valley = daily.loc[daily.index <= valley, "positions"]
        peak_pos = at_peak.iloc[-1] if len(at_peak) else {}
        valley_pos = at_valley.iloc[-1] if len(at_valley) else {}

        if not result.trades.empty:
            trade_dates = pd.to_datetime(result.trades["date"])
            window_trades = result.trades.loc[(trade_dates >= start) & (trade_dates <= end)]
        else:
            window_trades = result.trades
        rebalances = window_trades.loc[window_trades["reason"] == "threshold_rebalance"]
        rebalance_desc = []
        for trade_date, group in rebalances.groupby(pd.to_datetime(rebalances["date"]).dt.date):
            legs = "; ".join(
                f"{SHORT_NAMES.get(t.asset, t.asset)}{'卖' if t.side == 'sell' else '买'}{t.notional / 10_000:.1f}万"
                for t in group.itertuples()
            )
            rebalance_desc.append(f"{trade_date}: {legs}")

        rows.append({
            "rank": rank,
            "start": start.date().isoformat(),
            "valley": valley.date().isoformat(),
            "end": end.date().isoformat() if recovered else "未收复",
            "days": int(period["days"]),
            "depth_pct": float(period["depth_pct"]),
            "holdings_at_peak": _fmt_positions(peak_pos),
            "holdings_at_valley": _fmt_positions(valley_pos),
            "rebalance_trades": len(rebalances),
            "rebalance_detail": "<br>".join(rebalance_desc) if rebalance_desc else "无",
        })
    return pd.DataFrame(rows)


def _drawdown_html(drawdowns: pd.DataFrame) -> str:
    rows = []
    for row in drawdowns.itertuples():
        rows.append(
            "<tr>"
            f"<td>{row.rank}</td>"
            f"<td>{row.start}</td>"
            f"<td>{row.valley}</td>"
            f"<td>{row.end}</td>"
            f"<td>{row.days}</td>"
            f"<td>{row.depth_pct:.2f}%</td>"
            f"<td class='pos'>{html.escape(row.holdings_at_peak)}</td>"
            f"<td class='pos'>{html.escape(row.holdings_at_valley)}</td>"
            f"<td class='pos'>{row.rebalance_detail}</td>"
            "</tr>"
        )
    return (
        "<h2>Top 10 回撤明细</h2>"
        "<table class='drawdown'><thead><tr>"
        "<th>#</th><th>峰值日</th><th>谷底日</th><th>收复日</th><th>持续天数</th><th>回撤深度</th>"
        "<th>峰值日持仓</th><th>谷底日持仓</th><th>期间再平衡</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def _eoy_chart_svg(strategy_returns: pd.Series, benchmark_returns: pd.Series) -> str:
    """Generate a clean grouped-bar EOY returns chart as inline SVG."""
    strategy_yearly = (1.0 + strategy_returns.dropna()).groupby(strategy_returns.dropna().index.year).prod() - 1.0
    benchmark_yearly = (1.0 + benchmark_returns.dropna()).groupby(benchmark_returns.dropna().index.year).prod() - 1.0
    frame = pd.DataFrame({"strategy": strategy_yearly, "base": benchmark_yearly}) * 100.0

    fig, ax = plt.subplots(figsize=(10, 4))
    width = 0.35
    years = frame.index.astype(str)
    x = np.arange(len(years))
    ax.bar(x - width / 2, frame["base"], width, label="Fixed Buy&Hold Base", color="#ffbb78")
    ax.bar(x + width / 2, frame["strategy"], width, label="Strategy", color="#1f77b4")
    ax.axhline(0, color="#333333", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(years, rotation=45, ha="right")
    ax.set_ylabel("Return (%)")
    ax.set_title("EOY Returns vs Fixed Buy&Hold Base")
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    buf = io.StringIO()
    fig.savefig(buf, format="svg", bbox_inches="tight")
    plt.close(fig)
    svg = buf.getvalue()
    # Strip xml header so it can sit inline inside HTML
    svg = re.sub(r"<\?xml[^?]*\?>\s*<!DOCTYPE[^>]*>", "", svg)
    return f'<div id="eoy_returns">{svg}</div>'


_EXTRA_CSS = """
    #left{width:620px;margin-right:18px;margin-top:-1.2rem;float:left}
    #right{width:320px;margin-top:0;float:right}
    .container:after{content:'';display:table;clear:both}
    .drawdown{border-collapse:collapse;font-size:12px;width:100%;margin-top:12px}
    .drawdown th,.drawdown td{border-bottom:1px solid #ddd;padding:6px;text-align:left;vertical-align:top}
    .drawdown th{background:#eee;white-space:nowrap}
    .drawdown td.pos{max-width:260px}
    @media screen and (max-width:980px){
      .container{max-width:100%}#left,#right{width:100%;float:none;margin:0}
      .drawdown{display:block;overflow-x:auto}
    }
"""


def _write_html(
    strategy: object,
    baseline: object,
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    metrics: dict[str, float | int | bool],
    drawdowns: pd.DataFrame,
    backtest_start: str,
    backtest_end: str,
) -> Path:
    import quantstats as qs

    OUTPUT.mkdir(parents=True, exist_ok=True)
    output = OUTPUT / "backtest_report.html"
    temporary = OUTPUT / "_quantstats_report.html"
    clean_strategy = strategy_returns.dropna()
    qs.reports.html(
        clean_strategy,
        benchmark=benchmark_returns.reindex(clean_strategy.index).dropna(),
        benchmark_title="Fixed Buy&Hold Base",
        title="组合A（风险宇宙倾斜 λ=0.70）回测报告",
        output=str(temporary),
        match_dates=False,
    )
    report = temporary.read_text(encoding="utf-8")
    temporary.unlink()

    report = report.replace('<html lang="en">', '<html lang="zh-CN">')
    report = report.replace(
        "<title>Tearsheet (generated by QuantStats)</title>",
        "<title>组合A（风险宇宙倾斜 λ=0.70）回测报告</title>",
    )
    report = re.sub(
        r"<h1>.*?</h1>",
        "<h1>组合A：3 风险资产 20 日反转排名倾斜 λ=0.70＋货币固定 10% "
        f"<dt>{html.escape(backtest_start)} - {html.escape(backtest_end)}</dt></h1>",
        report,
        count=1,
        flags=re.DOTALL,
    )
    report = re.sub(
        r"<h4>.*?</h4>",
        "<h4>四只防守ETF &bull; 固定权重不主动再平衡基准 &bull; 每月首日入金20,000元 "
        "&bull; 月初收盘信号次日开盘执行 &bull; 单笔门槛10,000元 &bull; 单边成本0.05%</h4>",
        report,
        count=1,
        flags=re.DOTALL,
    )
    report = report.replace("    </style>\n</head>", _EXTRA_CSS + "    </style>\n</head>")

    strategy_metrics = metrics
    # Replace QuantStats' metrics with repo-standard values.
    baseline_metrics = _extended_metrics(baseline)
    metric_replacements = {
        "Start Period": (
            strategy.daily.index.min().date().isoformat(),
            baseline.daily.index.min().date().isoformat(),
        ),
        "CAGR﹪": (
            f"{strategy_metrics['annualized_return']:.2%}",
            f"{baseline_metrics['annualized_return']:.2%}",
        ),
        "Sharpe": (
            f"{strategy_metrics['sharpe']:.2f}",
            f"{baseline_metrics['sharpe']:.2f}",
        ),
        "Sortino": (
            f"{strategy_metrics['sortino']:.2f}",
            f"{baseline_metrics['sortino']:.2f}",
        ),
        "Calmar": (
            f"{strategy_metrics['calmar']:.2f}",
            f"{baseline_metrics['calmar']:.2f}",
        ),
        "Max Drawdown": (
            f"{strategy_metrics['max_drawdown']:.2%}",
            f"{baseline_metrics['max_drawdown']:.2%}",
        ),
        "Max DD Date": (
            _max_dd_date(strategy),
            _max_dd_date(baseline),
        ),
    }
    for label, (strategy_value, baseline_value) in metric_replacements.items():
        pattern = re.compile(
            rf"<tr><td>{re.escape(label)}</td><td>([^<]*)</td><td>([^<]*)</td></tr>"
        )
        report = pattern.sub(
            f"<tr><td>{label}</td><td>{html.escape(baseline_value)}</td><td>{html.escape(strategy_value)}</td></tr>",
            report,
        )

    # Replace the QuantStats EOY chart with a clean matplotlib SVG.
    report = re.sub(
        r'<div id="eoy_returns">.*?</div>',
        _eoy_chart_svg(strategy_returns, benchmark_returns),
        report,
        count=1,
        flags=re.DOTALL,
    )

    # Inject drawdown table at the end of the report body.
    marker = "    </div>\n    <style>*{white-space:auto !important;}</style>"
    if marker not in report:
        raise RuntimeError("QuantStats report structure changed; insertion point missing")
    report = report.replace(
        marker,
        _drawdown_html(drawdowns)
        + "\n    </div>\n    <style>*{white-space:auto !important;}</style>",
        1,
    )
    output.write_text(report, encoding="utf-8")
    return output


def build() -> Path:
    universe, _ = load_confirmed_market()
    market = load_market_data(
        sorted(set(universe) | set(EXTRA_POOL_ASSETS)), date(2013, 1, 1), date.today()
    )

    combo_targets = _risk_only_tilt_builder(BASELINE_TARGET, LAMBDA)(market)
    strategy = _run_strategy(market, combo_targets)
    baseline = simulate_fixed_buyandhold(
        market,
        BASELINE_TARGET,
        cash_asset=CASH_ASSET,
        monthly_deposit=MONTHLY_DEPOSIT,
        cost_rate=COST_RATE,
        lot_size=1,
    )

    metrics = _extended_metrics(strategy)
    drawdowns = _drawdown_table(strategy)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    strategy.daily.to_csv(OUTPUT / "daily_performance.csv")
    strategy.trades.to_csv(OUTPUT / "strategy_trades.csv", index=False)
    baseline.daily.to_csv(OUTPUT / "baseline_daily.csv")
    drawdowns.to_csv(OUTPUT / "top10_drawdowns.csv", index=False)

    return _write_html(
        strategy,
        baseline,
        strategy.daily["return"],
        baseline.daily["return"],
        metrics,
        drawdowns,
        strategy.daily.index.min().date().isoformat(),
        strategy.daily.index.max().date().isoformat(),
    )


if __name__ == "__main__":
    print(build())
