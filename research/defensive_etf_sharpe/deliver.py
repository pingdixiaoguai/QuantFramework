"""Build the standard delivery package for the current defensive ETF strategy."""

from __future__ import annotations

import html
from pathlib import Path
import re

import pandas as pd

from .factor_research import _extended_metrics
from .rebalance_timing import daily_reversal_targets
from .reduced_pool_research import REDUCED_TARGET
from .strategy import CASH_ASSET, load_confirmed_market, metrics_for_daily
from .threshold_rebalance import TriggerSpec, simulate_threshold_rebalance


ROOT = Path(__file__).parent
OUTPUT = ROOT / "deliverable"
COST_RATE = 0.0005
MIN_REBALANCE_NOTIONAL = 10_000.0
MONTHLY_DEPOSIT = 20_000.0
CURRENT_TRIGGER = TriggerSpec(
    "portfolio_drift_10_monthly_cap1",
    "portfolio_drift",
    0.10,
    description="组合单边偏离达到10%，每月最多再平衡一次",
    max_rebalances_per_month=1,
)
CALENDAR_BENCHMARK = TriggerSpec(
    "calendar_monthly_reference",
    "calendar_monthly",
    0.5,
    description="每月第一个交易日收盘检查并于下一交易日开盘再平衡",
)


def _pct(value: float) -> str:
    return f"{value:.2%}"


def _result_metrics(result) -> dict[str, float | int | bool]:
    metrics = _extended_metrics(result)
    rebalance_trades = result.trades.loc[result.trades["reason"] == "threshold_rebalance"]
    metrics.update({
        "trigger_signals": len(result.signals),
        "rebalance_dates": (
            int(pd.to_datetime(rebalance_trades["date"]).nunique())
            if not rebalance_trades.empty
            else 0
        ),
        "deposit_trade_count": int((result.trades["reason"] == "deposit_invest").sum()),
        "rebalance_trade_count": int(
            (result.trades["reason"] == "threshold_rebalance").sum()
        ),
        "average_cash_weight": float(result.daily["cash_weight"].mean()),
    })
    return metrics


def _annual_metrics(portfolios: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for portfolio, frame in portfolios.items():
        for year, annual in frame.groupby(frame.index.year):
            returns = annual["return"].dropna().astype(float)
            if returns.empty:
                continue
            curve = (1.0 + returns).cumprod()
            drawdown = curve / curve.cummax() - 1.0
            volatility = float(returns.std(ddof=1) * (252.0 ** 0.5))
            rows.append({
                "portfolio": portfolio,
                "year": int(year),
                "calendar_return": float(curve.iloc[-1] - 1.0),
                "volatility": volatility,
                "sharpe": (
                    float(returns.mean() / returns.std(ddof=1) * (252.0 ** 0.5))
                    if returns.std(ddof=1) > 0
                    else 0.0
                ),
                "max_drawdown": float(drawdown.min()),
                "deposits": float(annual["deposit"].sum()),
            })
    return pd.DataFrame(rows)


def _comparison_table(metrics: dict[str, dict[str, float | int | bool]]) -> str:
    labels = {
        "strategy": "当前策略：10%偏离触发",
        "calendar": "主基线：月初固定检查",
        "static": "次基线：固定35/40/15/10",
    }
    rows = []
    for key in ("strategy", "calendar", "static"):
        values = metrics[key]
        rows.append(
            "<tr>"
            f"<td>{labels[key]}</td>"
            f"<td>{float(values['annualized_return']):.2%}</td>"
            f"<td>{float(values['volatility']):.2%}</td>"
            f"<td>{float(values['sharpe']):.3f}</td>"
            f"<td>{float(values['max_drawdown']):.2%}</td>"
            f"<td>{float(values['sortino']):.3f}</td>"
            f"<td>{float(values['final_nav']) / 10_000:.2f}万</td>"
            f"<td>{int(values['rebalance_dates'])}</td>"
            f"<td>{float(values['estimated_transaction_cost']) / 10_000:.2f}万</td>"
            "</tr>"
        )
    return (
        "<table class='comparison'><thead><tr>"
        "<th>组合</th><th>年化收益</th><th>年化波动</th><th>Sharpe</th>"
        "<th>最大回撤</th><th>Sortino</th><th>期末资产</th><th>再平衡日</th><th>估算成本</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
    )


def _methodology_html(
    metrics: dict[str, dict[str, float | int | bool]],
    backtest_start: str,
    backtest_end: str,
) -> str:
    strategy = metrics["strategy"]
    comparison = _comparison_table(metrics)
    return f"""
        <section class="methodology">
            <h2>策略与回测口径</h2>
            <div class="method-grid">
                <article>
                    <h3>1. 标的与基准权重</h3>
                    <table class="universe-table">
                        <thead><tr><th>代码</th><th>名称</th><th>资产类型</th><th>基准权重</th></tr></thead>
                        <tbody>
                            <tr><td>512890.SH</td><td>华泰柏瑞中证红利低波动ETF</td><td>红利低波股票</td><td>35%</td></tr>
                            <tr><td>511260.SH</td><td>国泰上证10年期国债ETF</td><td>长期国债</td><td>40%</td></tr>
                            <tr><td>511360.SH</td><td>海富通中证短融ETF</td><td>短期信用债</td><td>15%</td></tr>
                            <tr><td>511880.SH</td><td>银华日利ETF</td><td>货币市场</td><td>10%</td></tr>
                        </tbody>
                    </table>
                </article>
                <article>
                    <h3>2. 因子与目标权重</h3>
                    <p>每个交易日收盘后，使用后复权收盘价计算四只ETF的20日反转因子：</p>
                    <p class="formula">Rᵢ,t = −(Cᵢ,t / Cᵢ,t−20 − 1)</p>
                    <p>对有效因子值做全池横截面排名。平均名次居中后除以最大绝对偏差，得到 <span class="mono">rᵢ,t ∈ [−1, 1]</span>；并列名次取平均。历史不足21个交易日时，该标的排名分量记为0。</p>
                    <p class="formula">uᵢ,t = bᵢ × (1 + 0.5rᵢ,t)　　wᵢ,t = uᵢ,t / Σⱼuⱼ,t</p>
                    <p>其中 <span class="mono">bᵢ</span> 为35%/40%/15%/10%的基准权重。“全池50%倾斜”只改变相对权重，不会产生空头或杠杆，目标权重之和始终为100%。</p>
                </article>
                <article>
                    <h3>3. 入金与偏离触发</h3>
                    <ol>
                        <li>初始资金为0；每月第一个交易日开盘前注入20,000元。</li>
                        <li>新增资金按上月最后一个交易日收盘计算的目标权重，只买不卖地补足各标的缺口。</li>
                        <li>每日收盘计算最新目标权重及组合单边偏离：</li>
                    </ol>
                    <p class="formula">Dₜ = ½ × (Σᵢ|wᵢ,t(actual) − wᵢ,t(target)| + wₜ(cash excess))</p>
                    <p>当 <span class="mono">Dₜ ≥ 10.0%</span> 时生成信号，在下一交易日开盘再平衡；按实际执行月份统计，每个自然月最多成功再平衡一次。</p>
                </article>
                <article>
                    <h3>4. 成交、成本与约束</h3>
                    <ul>
                        <li>再平衡以开盘价计算目标股数，ETF按100股整数手成交。</li>
                        <li>计划单笔金额不足10,000元时删除该笔；合格卖单先执行，买单仅使用已有现金及卖出所得。</li>
                        <li>若部分卖单被门槛过滤，对应买单会因现金约束缩减或取消；实际买单仍须达到10,000元。</li>
                        <li>买卖双边均按成交额0.05%计成本；不另计冲击成本、申赎费用、税费或额外滑点。</li>
                        <li>仅做多，不使用杠杆、融资或融券；未成交资金保留为现金。</li>
                    </ul>
                </article>
            </div>

            <h2>核心指标对比</h2>
            {comparison}
            <p class="note">主基线使用相同的反转因子、权重公式、入金、成本、整数手和1万元门槛，但每月第一个交易日收盘固定检查并在下一交易日开盘再平衡，不采用10%偏离触发和每月一次上限。次基线使用固定35%/40%/15%/10%权重，并按同样的月初固定检查规则执行。</p>

            <h2>回测范围与解释</h2>
            <ul>
                <li>数据区间：{html.escape(backtest_start)}至{html.escape(backtest_end)}；使用本地后复权开盘价与收盘价，标的仅从真实上市交易日起可交易，不使用上市前指数代理。</li>
                <li>收益指标采用剔除每月外部入金后的日收益序列计算；期末资产包含累计入金。累计入金为{float(strategy['total_deposits']) / 10_000:.0f}万元。</li>
                <li>当前策略共发出{int(strategy['trigger_signals'])}个触发信号，在{int(strategy['rebalance_dates'])}个交易日实际再平衡；总交易笔数{int(strategy['trades'])}笔，平均现金权重{float(strategy['average_cash_weight']):.2%}。</li>
                <li>本报告为全样本历史研究。因子、倾斜幅度、10%阈值和成交门槛均经过历史比较，结果存在参数选择和样本内过拟合风险，不构成未来收益保证。</li>
            </ul>
        </section>
    """


def _write_html(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    metrics: dict[str, dict[str, float | int | bool]],
    backtest_start: str,
    backtest_end: str,
) -> Path:
    """Generate QuantStats charts and append the audited strategy definition."""
    import quantstats as qs

    OUTPUT.mkdir(parents=True, exist_ok=True)
    output = OUTPUT / "backtest_report.html"
    temporary = OUTPUT / "_quantstats_report.html"
    clean_strategy = strategy_returns.dropna()
    qs.reports.html(
        clean_strategy,
        benchmark=benchmark_returns.reindex(clean_strategy.index).dropna(),
        # QuantStats hard-codes Arial in SVG charts, so keep the plotted label
        # ASCII-only; the surrounding report defines the Chinese benchmark name.
        benchmark_title="Monthly Calendar Rebalance",
        title="防守型ETF反转倾斜策略回测报告",
        output=str(temporary),
    )
    report = temporary.read_text(encoding="utf-8")
    temporary.unlink()
    report = report.replace('<html lang="en">', '<html lang="zh-CN">')
    report = report.replace(
        "<title>Tearsheet (generated by QuantStats)</title>",
        "<title>防守型ETF反转倾斜策略回测报告</title>",
    )
    report = re.sub(
        r"<h1>.*?</h1>",
        "<h1>防守型ETF 20日反转＋全池50%倾斜＋10%偏离触发 "
        f"<dt>{html.escape(backtest_start)} - {html.escape(backtest_end)}</dt></h1>",
        report,
        count=1,
        flags=re.DOTALL,
    )
    report = re.sub(
        r"<h4>.*?</h4>",
        "<h4>四只防守ETF &bull; 每月首日入金20,000元 &bull; 每日收盘检查10%组合单边偏离 "
        "&bull; 次日开盘执行 &bull; 每月最多再平衡一次 &bull; 单笔门槛10,000元 "
        "&bull; 单边成本0.05%</h4>",
        report,
        count=1,
        flags=re.DOTALL,
    )
    extra_css = """
    #left{width:620px;margin-right:18px;margin-top:-1.2rem;float:left}
    #right{width:320px;margin-top:0;float:right}
    .container:after{content:'';display:table;clear:both}
    .methodology{clear:both;padding-top:28px;border-top:1px solid #ccc;color:#202124}
    .methodology h2{font-size:20px;font-weight:700;margin:28px 0 14px}
    .methodology h3{font-size:14px;margin:0 0 10px}
    .methodology p,.methodology li{font-size:13px;line-height:1.65}
    .methodology ol,.methodology ul{margin:6px 0 10px;padding-left:22px}
    .method-grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}
    .method-grid article{border:1px solid #ddd;border-radius:6px;padding:16px;background:#fafafa}
    .formula{font-family:Georgia,'Times New Roman',serif;background:#fff;border-left:3px solid #555;padding:8px 10px}
    .mono{font-family:Menlo,Consolas,monospace}
    .comparison,.universe-table{margin-bottom:12px;border-collapse:collapse}
    .comparison th,.comparison td,.universe-table th,.universe-table td{border-bottom:1px solid #ddd;padding:7px 6px}
    .comparison th,.universe-table th{background:#eee;font-weight:700;white-space:nowrap}
    .note{color:#555;background:#f6f6f6;padding:10px 12px;border-radius:4px}
    @media screen and (max-width:980px){
      .container{max-width:100%}#left,#right{width:100%;float:none;margin:0}.method-grid{grid-template-columns:1fr}
      .comparison{font-size:11px;display:block;overflow-x:auto}
    }
    """
    report = report.replace("    </style>\n</head>", extra_css + "    </style>\n</head>")
    marker = "    </div>\n    <style>*{white-space:auto !important;}</style>"
    if marker not in report:
        raise RuntimeError("QuantStats report structure changed; methodology insertion point missing")
    report = report.replace(
        marker,
        _methodology_html(metrics, backtest_start, backtest_end)
        + "\n    </div>\n    <style>*{white-space:auto !important;}</style>",
        1,
    )
    output.write_text(report, encoding="utf-8")
    return output


def _delivery_markdown(metrics: dict[str, dict[str, float | int | bool]]) -> str:
    strategy = metrics["strategy"]
    return f"""# 防守型ETF策略标准交付包

## 当前正式策略

正式版本固定为 `defensive_etf_reversal20_global_tilt50_drift10_monthly_cap1`。四只ETF基准权重为512890.SH 35%、511260.SH 40%、511360.SH 15%、511880.SH 10%；每日使用20日反转因子做全池50%排名倾斜。每月首个交易日新增20,000元按上月末目标权重买入；组合单边偏离达到10.0%后，下一交易日开盘再平衡，每月最多一次，单笔不足10,000元不执行。

## 核心结果

全期时间加权年化收益{_pct(float(strategy['annualized_return']))}，年化波动{_pct(float(strategy['volatility']))}，Sharpe {float(strategy['sharpe']):.3f}，最大回撤{_pct(float(strategy['max_drawdown']))}，期末资产{float(strategy['final_nav']):,.0f}元，估算交易成本{float(strategy['estimated_transaction_cost']):,.0f}元。

完整策略公式、执行顺序、基线定义、图表和限制见 `backtest_report.html`。明细数据见 `core_metrics.csv`、`annual_metrics.csv`、`daily_performance.csv`、`daily_target_weights.csv`、`strategy_signals.csv` 与 `strategy_trades.csv`。
"""


def build() -> Path:
    universe, market = load_confirmed_market()
    daily_targets = daily_reversal_targets(market, REDUCED_TARGET)
    strategy = simulate_threshold_rebalance(
        market,
        daily_targets,
        CURRENT_TRIGGER,
        initial_target=REDUCED_TARGET,
        cash_asset=CASH_ASSET,
        monthly_deposit=MONTHLY_DEPOSIT,
        cost_rate=COST_RATE,
        min_rebalance_notional=MIN_REBALANCE_NOTIONAL,
    )
    calendar = simulate_threshold_rebalance(
        market,
        daily_targets,
        CALENDAR_BENCHMARK,
        initial_target=REDUCED_TARGET,
        cash_asset=CASH_ASSET,
        monthly_deposit=MONTHLY_DEPOSIT,
        cost_rate=COST_RATE,
        min_rebalance_notional=MIN_REBALANCE_NOTIONAL,
    )
    static_targets = {timestamp: dict(REDUCED_TARGET) for timestamp in market.dates}
    static = simulate_threshold_rebalance(
        market,
        static_targets,
        CALENDAR_BENCHMARK,
        initial_target=REDUCED_TARGET,
        cash_asset=CASH_ASSET,
        monthly_deposit=MONTHLY_DEPOSIT,
        cost_rate=COST_RATE,
        min_rebalance_notional=MIN_REBALANCE_NOTIONAL,
    )
    metrics = {
        "strategy": _result_metrics(strategy),
        "calendar": _result_metrics(calendar),
        "static": _result_metrics(static),
    }

    OUTPUT.mkdir(parents=True, exist_ok=True)
    pool_rows = []
    for asset, weight in REDUCED_TARGET.items():
        metadata = universe[asset]
        close = market.closes[asset]
        pool_rows.append({
            "asset": asset,
            "name": metadata["name"],
            "short_name": metadata["short_name"],
            "sleeve": metadata["sleeve"],
            "baseline_weight": weight,
            "first_date": close.index.min().date().isoformat(),
            "last_date": close.index.max().date().isoformat(),
            "rows": len(close),
        })
    pd.DataFrame(pool_rows).to_csv(OUTPUT / "candidate_pool.csv", index=False)

    frames = {"strategy": strategy.daily, "calendar": calendar.daily, "static": static.daily}
    daily = pd.DataFrame(index=strategy.daily.index)
    for label, frame in frames.items():
        daily[f"{label}_nav"] = frame["nav"]
        daily[f"{label}_return"] = frame["return"]
        daily[f"{label}_cash_weight"] = frame["cash_weight"]
    daily["deposit"] = strategy.daily["deposit"]
    daily.to_csv(OUTPUT / "daily_performance.csv")

    strategy.trades.to_csv(OUTPUT / "strategy_trades.csv", index=False)
    strategy.signals.to_csv(OUTPUT / "strategy_signals.csv", index=False)
    _annual_metrics(frames).to_csv(OUTPUT / "annual_metrics.csv", index=False)
    pd.DataFrame([
        {"portfolio": name, **values} for name, values in metrics.items()
    ]).to_csv(OUTPUT / "core_metrics.csv", index=False)
    pd.DataFrame([
        {"date": timestamp.date().isoformat(), "asset": asset, "target_weight": weight}
        for timestamp, weights in daily_targets.items()
        for asset, weight in weights.items()
    ]).to_csv(OUTPUT / "daily_target_weights.csv", index=False)

    _write_html(
        strategy.daily["return"],
        calendar.daily["return"],
        metrics,
        strategy.daily.index.min().date().isoformat(),
        strategy.daily.index.max().date().isoformat(),
    )
    (OUTPUT / "DELIVERY.md").write_text(_delivery_markdown(metrics), encoding="utf-8")
    return OUTPUT


if __name__ == "__main__":
    print(build())
