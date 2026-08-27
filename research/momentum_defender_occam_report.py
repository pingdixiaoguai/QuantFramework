"""Self-contained HTML report for the frozen Momentum/Defender experiment."""

from __future__ import annotations

import html
from datetime import date
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


COLORS = {
    "candidate": "#0f766e",
    "momentum": "#2563eb",
    "defender": "#7c3aed",
    "positive": "#15803d",
    "negative": "#c2410c",
    "muted": "#64748b",
    "grid": "#dbe3ee",
}


def _pct(value: object, digits: int = 2, *, signed: bool = False) -> str:
    number = float(value)
    prefix = "+" if signed and number > 0 else ""
    return f"{prefix}{number:.{digits}%}"


def _number(value: object, digits: int = 3, *, signed: bool = False) -> str:
    number = float(value)
    prefix = "+" if signed and number > 0 else ""
    return f"{prefix}{number:.{digits}f}"


def _truthy(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() == "true"
    return bool(value)


def _drawdown(nav: pd.Series) -> pd.Series:
    values = nav.astype(float)
    return values / values.cummax() - 1.0


def _series_path(
    index: pd.DatetimeIndex,
    values: np.ndarray,
    *,
    left: float,
    top: float,
    width: float,
    height: float,
    start_ns: int,
    span_ns: int,
    y_min: float,
    y_span: float,
    log_y: bool,
) -> str:
    transformed = np.log(values) if log_y else values
    points: list[str] = []
    for timestamp, value in zip(index, transformed):
        if not np.isfinite(value):
            continue
        x = left + (int(timestamp.value) - start_ns) / span_ns * width
        y = top + (y_min + y_span - value) / y_span * height
        points.append(f"{x:.2f},{y:.2f}")
    if not points:
        return ""
    return "M" + " L".join(points)


def _line_chart(
    lines: dict[str, tuple[pd.Series, str]],
    *,
    title: str,
    y_mode: str,
    log_y: bool = False,
    markers: Iterable[tuple[pd.Timestamp, str, str]] = (),
    shaded_ranges: Iterable[tuple[pd.Timestamp, pd.Timestamp, str]] = (),
) -> str:
    clean: dict[str, tuple[pd.Series, str]] = {}
    for label, (series, color) in lines.items():
        values = series.astype(float).dropna().sort_index()
        if not values.empty:
            clean[label] = (values, color)
    if not clean:
        raise ValueError(f"chart {title!r} has no finite data")

    start = min(series.index.min() for series, _ in clean.values())
    end = max(series.index.max() for series, _ in clean.values())
    start_ns = int(start.value)
    span_ns = max(int(end.value) - start_ns, 1)
    raw_values = np.concatenate(
        [series.to_numpy(float) for series, _ in clean.values()]
    )
    if log_y and np.any(raw_values <= 0.0):
        raise ValueError(f"chart {title!r} cannot log non-positive values")
    transformed = np.log(raw_values) if log_y else raw_values
    y_min = float(np.nanmin(transformed))
    y_max = float(np.nanmax(transformed))
    if y_mode == "percent":
        y_max = max(y_max, 0.0)
        y_min = min(y_min, 0.0)
    y_span = max(y_max - y_min, 1e-9)
    padding = y_span * 0.08
    y_min -= padding
    y_max += padding
    y_span = y_max - y_min

    view_width, view_height = 980.0, 390.0
    left, top, plot_width, plot_height = 72.0, 48.0, 876.0, 270.0
    parts = [
        f'<figure class="chart-card"><figcaption>{html.escape(title)}</figcaption>',
        f'<svg viewBox="0 0 {view_width:.0f} {view_height:.0f}" role="img" '
        f'aria-label="{html.escape(title)}">',
    ]

    for range_start, range_end, color in shaded_ranges:
        x1 = left + (int(range_start.value) - start_ns) / span_ns * plot_width
        x2 = left + (int(range_end.value) - start_ns) / span_ns * plot_width
        parts.append(
            f'<rect x="{x1:.2f}" y="{top:.2f}" width="{max(x2-x1, 1):.2f}" '
            f'height="{plot_height:.2f}" fill="{color}" opacity="0.08" />'
        )

    for step in range(5):
        ratio = step / 4
        transformed_tick = y_max - ratio * y_span
        raw_tick = float(np.exp(transformed_tick)) if log_y else transformed_tick
        y = top + ratio * plot_height
        if y_mode == "percent":
            label = f"{raw_tick:.0%}"
        elif y_mode == "nav":
            label = f"{raw_tick:.1f}×"
        else:
            label = f"{raw_tick:.2f}"
        parts.extend(
            [
                f'<line x1="{left}" y1="{y:.2f}" x2="{left+plot_width}" '
                f'y2="{y:.2f}" stroke="{COLORS["grid"]}" stroke-width="1" />',
                f'<text x="{left-10}" y="{y+4:.2f}" text-anchor="end" '
                f'class="axis-label">{html.escape(label)}</text>',
            ]
        )

    tick_dates = pd.date_range(start=start, end=end, periods=5)
    for timestamp in tick_dates:
        x = left + (int(timestamp.value) - start_ns) / span_ns * plot_width
        parts.extend(
            [
                f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" '
                f'y2="{top+plot_height}" stroke="{COLORS["grid"]}" stroke-width="1" />',
                f'<text x="{x:.2f}" y="{top+plot_height+24}" text-anchor="middle" '
                f'class="axis-label">{timestamp:%Y-%m}</text>',
            ]
        )

    for label, (series, color) in clean.items():
        path = _series_path(
            pd.DatetimeIndex(series.index),
            series.to_numpy(float),
            left=left,
            top=top,
            width=plot_width,
            height=plot_height,
            start_ns=start_ns,
            span_ns=span_ns,
            y_min=y_min,
            y_span=y_span,
            log_y=log_y,
        )
        parts.append(
            f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2.4" '
            f'stroke-linejoin="round" stroke-linecap="round" />'
        )

    for number, (timestamp, label, color) in enumerate(markers):
        x = left + (int(timestamp.value) - start_ns) / span_ns * plot_width
        text_y = 22 + (number % 2) * 15
        parts.extend(
            [
                f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" '
                f'y2="{top+plot_height}" stroke="{color}" stroke-width="1.2" '
                'stroke-dasharray="5 4" />',
                f'<text x="{x+4:.2f}" y="{text_y}" class="marker-label" '
                f'fill="{color}">{html.escape(label)}</text>',
            ]
        )

    legend_x = left
    legend_y = view_height - 20
    for label, (_, color) in clean.items():
        parts.extend(
            [
                f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x+24}" '
                f'y2="{legend_y}" stroke="{color}" stroke-width="3" />',
                f'<text x="{legend_x+31}" y="{legend_y+4}" class="legend-label">'
                f'{html.escape(label)}</text>',
            ]
        )
        legend_x += 31 + max(84, len(label) * 13)
    parts.append("</svg></figure>")
    return "".join(parts)


def _episode_bar_chart(episodes: pd.DataFrame) -> str:
    values = episodes["arithmetic_excess_return"].astype(float).to_numpy()
    count = len(values)
    view_width, view_height = 980.0, 340.0
    left, top, plot_width, plot_height = 62.0, 36.0, 884.0, 242.0
    y_min = min(float(values.min()), 0.0)
    y_max = max(float(values.max()), 0.0)
    span = max(y_max - y_min, 1e-9)
    zero_y = top + (y_max / span) * plot_height
    slot = plot_width / count
    bar_width = max(slot * 0.62, 3.0)
    parts = [
        '<figure class="chart-card"><figcaption>各Defender窗口的累计收益差（融合 − Momentum）</figcaption>',
        f'<svg viewBox="0 0 {view_width:.0f} {view_height:.0f}" role="img" '
        'aria-label="22个Defender窗口累计收益差">',
        f'<line x1="{left}" y1="{zero_y:.2f}" x2="{left+plot_width}" '
        f'y2="{zero_y:.2f}" stroke="#94a3b8" stroke-width="1.2" />',
    ]
    for index, value in enumerate(values):
        x = left + index * slot + (slot - bar_width) / 2
        value_y = top + (y_max - value) / span * plot_height
        y = min(value_y, zero_y)
        height = max(abs(zero_y - value_y), 1.0)
        color = COLORS["positive"] if value >= 0.0 else COLORS["negative"]
        css_class = " critical-bar" if int(episodes.iloc[index]["episode"]) == 17 else ""
        parts.extend(
            [
                f'<rect class="episode-bar{css_class}" x="{x:.2f}" y="{y:.2f}" '
                f'width="{bar_width:.2f}" height="{height:.2f}" fill="{color}">',
                f'<title>第{index+1}段：{value:+.2%}</title></rect>',
                f'<text x="{x+bar_width/2:.2f}" y="{top+plot_height+18}" '
                f'text-anchor="middle" class="axis-label">{index+1}</text>',
            ]
        )
    parts.extend(
        [
            f'<text x="{left}" y="{view_height-14}" class="legend-label">'
            '绿色表示融合胜出；橙色表示Momentum胜出；第17段加粗描边。</text>',
            "</svg></figure>",
        ]
    )
    return "".join(parts)


def _performance_table(frame: pd.DataFrame) -> str:
    labels = {
        "momentum_official_runner": "正式Momentum",
        "momentum_exact_adapter": "精确分段Momentum",
        "defender_continuous": "连续持有Defender",
        "selected_fusion": "融合候选",
    }
    rows: list[str] = []
    for record in frame.to_dict("records"):
        name = str(record["strategy"])
        css_class = " class=\"selected-row\"" if name == "selected_fusion" else ""
        rows.append(
            f"<tr{css_class}><th>{html.escape(labels.get(name, name))}</th>"
            f"<td>{_pct(record['total_return'])}</td>"
            f"<td>{_pct(record['cagr_calendar'])}</td>"
            f"<td>{_pct(record['annualized_return_252'])}</td>"
            f"<td>{_pct(record['annualized_volatility'])}</td>"
            f"<td>{_number(record['sharpe'])}</td>"
            f"<td>{_pct(record['max_drawdown'])}</td></tr>"
        )
    return (
        '<div class="table-scroll"><table><thead><tr><th>策略</th><th>累计收益</th>'
        '<th>日历CAGR</th><th>252日年化</th><th>年化波动</th><th>Sharpe</th>'
        f'<th>最大回撤</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div>'
    )


def _annual_table(frame: pd.DataFrame) -> str:
    candidate = frame.loc[frame["strategy"].eq("selected_fusion")].set_index("period")
    momentum = frame.loc[frame["strategy"].eq("momentum_exact_adapter")].set_index(
        "period"
    )
    rows: list[str] = []
    for period in candidate.index:
        candidate_row = candidate.loc[period]
        momentum_row = momentum.loc[period]
        excess = float(candidate_row["total_return"]) - float(momentum_row["total_return"])
        rows.append(
            "<tr>"
            f"<th>{html.escape(str(period))}</th>"
            f"<td>{_pct(candidate_row['total_return'])}</td>"
            f"<td>{_pct(momentum_row['total_return'])}</td>"
            f"<td class={'positive' if excess >= 0 else 'negative'}>{_pct(excess, signed=True)}</td>"
            f"<td>{_number(candidate_row['sharpe'])}</td>"
            f"<td>{_number(momentum_row['sharpe'])}</td>"
            f"<td>{_pct(candidate_row['max_drawdown'])}</td>"
            f"<td>{_pct(momentum_row['max_drawdown'])}</td>"
            "</tr>"
        )
    return (
        '<div class="table-scroll"><table><thead><tr><th>年度</th><th>融合收益</th>'
        '<th>Momentum收益</th><th>收益差</th><th>融合Sharpe</th>'
        '<th>Momentum Sharpe</th><th>融合MDD</th><th>Momentum MDD</th>'
        f'</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'
    )


def _entry_reason_label(value: object) -> str:
    labels = {
        "emergency_exit": "紧急cap",
        "slow_regime_switch": "慢门控",
    }
    return labels.get(str(value), str(value))


def _episode_table(episodes: pd.DataFrame) -> str:
    rows: list[str] = []
    for record in episodes.to_dict("records"):
        number = int(record["episode"])
        excess = float(record["arithmetic_excess_return"])
        exit_text = (
            f"{record['window_end']}（未退出）"
            if str(record["exit_date"]) == "open_at_cutoff"
            else str(record["exit_date"])
        )
        row_class = "critical-episode" if number == 17 else ""
        rows.append(
            f'<tr class="{row_class}" data-positive="{str(excess >= 0).lower()}">'
            f"<th>{number}</th>"
            f"<td>{html.escape(str(record['entry_date']))}</td>"
            f"<td>{html.escape(exit_text)}</td>"
            f"<td>{html.escape(_entry_reason_label(record['entry_reason']))}</td>"
            f"<td>{int(record['window_observations'])}/{int(record['defender_days'])}</td>"
            f"<td>{_pct(record['candidate_return'])}</td>"
            f"<td>{_pct(record['candidate_annualized_return_252'])}</td>"
            f"<td>{_pct(record['candidate_annualized_volatility'])}</td>"
            f"<td>{_number(record['candidate_sharpe'], 2)}</td>"
            f"<td>{_pct(record['candidate_max_drawdown'])}</td>"
            f"<td>{_pct(record['momentum_return'])}</td>"
            f"<td>{_pct(record['momentum_annualized_return_252'])}</td>"
            f"<td>{_pct(record['momentum_annualized_volatility'])}</td>"
            f"<td>{_number(record['momentum_sharpe'], 2)}</td>"
            f"<td>{_pct(record['momentum_max_drawdown'])}</td>"
            f"<td class={'positive' if excess >= 0 else 'negative'}>{_pct(excess, signed=True)}</td>"
            "</tr>"
        )
    return (
        '<div class="episode-controls">'
        '<label>筛选 <input id="episode-search" type="search" placeholder="日期、段号或触发方式"></label>'
        '<label class="check-label"><input id="positive-only" type="checkbox">只看融合胜出的窗口</label>'
        '</div><div class="table-scroll episode-scroll"><table id="episode-table">'
        '<thead><tr><th rowspan="2">段</th><th rowspan="2">切入开盘</th>'
        '<th rowspan="2">切出开盘/截止</th><th rowspan="2">触发</th>'
        '<th rowspan="2">窗口/持有日</th><th colspan="5">融合候选</th>'
        '<th colspan="5">精确Momentum</th><th rowspan="2">累计收益差</th></tr>'
        '<tr><th>累计</th><th>252年化</th><th>波动</th><th>Sharpe</th><th>MDD</th>'
        '<th>累计</th><th>252年化</th><th>波动</th><th>Sharpe</th><th>MDD</th>'
        f'</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'
    )


def _lookup(frame: pd.DataFrame, column: str, value: str) -> pd.Series:
    selected = frame.loc[frame[column].astype(str).eq(value)]
    if len(selected) != 1:
        raise ValueError(f"expected exactly one row where {column}={value!r}")
    return selected.iloc[0]


def generate_html_report(
    output_dir: Path,
    output_path: Path | None = None,
) -> Path:
    """Create the audited, self-contained HTML report from final experiment CSVs."""
    output_dir = Path(output_dir)
    output_path = output_dir / "backtest_report.html" if output_path is None else output_path

    performance = pd.read_csv(output_dir / "performance_summary.csv")
    daily = pd.read_csv(
        output_dir / "selected_strategy_daily.csv",
        parse_dates=[
            "date",
            "defender_signal_row_date",
            "defender_signal_observation_date",
        ],
    ).set_index("date")
    episodes = pd.read_csv(output_dir / "defender_episode_metrics.csv")
    annual = pd.read_csv(output_dir / "annual_metrics.csv")
    ablation = pd.read_csv(output_dir / "ablation_metrics.csv")
    robustness = pd.read_csv(output_dir / "robustness_summary.csv")
    selection = pd.read_csv(output_dir / "selection_bias_checks.csv")
    bootstrap = pd.read_csv(output_dir / "paired_block_bootstrap_summary.csv")
    placebo = pd.read_csv(output_dir / "cap_timing_placebo.csv")
    leave_one = pd.read_csv(output_dir / "leave_one_defender_episode_out.csv")
    checks = pd.read_csv(output_dir / "reproduction_checks.csv")

    if len(daily) != 1837 or not daily.index.is_unique or not daily.index.is_monotonic_increasing:
        raise ValueError("HTML report requires the audited 1,837-row unique daily sample")
    if len(episodes) != 22:
        raise ValueError("HTML report requires all 22 Defender episodes")
    if not checks["passed"].map(_truthy).all():
        raise ValueError("HTML report will not render with failed reproduction checks")

    candidate = _lookup(performance, "strategy", "selected_fusion")
    exact_momentum = _lookup(performance, "strategy", "momentum_exact_adapter")
    selected_max_t = _lookup(
        selection, "check", "selected_candidate_familywise_maxT"
    )
    omnibus_max_t = _lookup(
        selection, "check", "local_family_studentized_block_maxT_omnibus"
    )
    critical_leave_one = leave_one.sort_values("max_drawdown_improvement").iloc[0]
    local_strict = _lookup(robustness, "check", "local_45_strict_triple")
    local_material = _lookup(robustness, "check", "local_45_material_triple")
    placebo_strict = _lookup(
        robustness, "check", "cap_timing_placebo_strict_triple"
    )
    placebo_dominates = _lookup(
        robustness, "check", "cap_timing_placebo_dominates_selected"
    )

    full_nav_chart = _line_chart(
        {
            "融合候选": (daily["nav"], COLORS["candidate"]),
            "精确Momentum": (daily["momentum_exact_nav"], COLORS["momentum"]),
            "连续Defender": (daily["defender_continuous_nav"], COLORS["defender"]),
        },
        title="全样本累计净值（对数刻度）",
        y_mode="nav",
        log_y=True,
    )
    drawdown_chart = _line_chart(
        {
            "融合候选": (_drawdown(daily["nav"]), COLORS["candidate"]),
            "精确Momentum": (
                _drawdown(daily["momentum_exact_nav"]),
                COLORS["momentum"],
            ),
        },
        title="全样本回撤路径",
        y_mode="percent",
    )

    entry_date = pd.Timestamp("2024-10-08")
    exit_date = pd.Timestamp("2024-11-19")
    trough_date = pd.Timestamp("2024-10-17")
    cap_clear_date = pd.Timestamp("2024-11-08")
    event = daily.loc[entry_date:exit_date]
    prior_date = daily.index[daily.index < entry_date].max()
    event_candidate = pd.concat(
        [pd.Series([1.0], index=[prior_date]), (1.0 + event["return"]).cumprod()]
    )
    event_momentum = pd.concat(
        [
            pd.Series([1.0], index=[prior_date]),
            (1.0 + event["momentum_exact_return"]).cumprod(),
        ]
    )
    event_chart = _line_chart(
        {
            "融合候选": (event_candidate, COLORS["candidate"]),
            "精确Momentum": (event_momentum, COLORS["momentum"]),
        },
        title="2024关键防守窗口：切入前净值归一为1",
        y_mode="nav",
        markers=(
            (entry_date, "10/08紧急切入", COLORS["negative"]),
            (trough_date, "10/17 Momentum谷底", COLORS["momentum"]),
            (cap_clear_date, "11/08 cap解除", COLORS["muted"]),
            (exit_date, "11/19切回Momentum", COLORS["positive"]),
        ),
        shaded_ranges=((entry_date, exit_date, COLORS["candidate"]),),
    )
    episode_chart = _episode_bar_chart(episodes)

    trigger = daily.loc[entry_date]
    episode_17 = episodes.loc[episodes["episode"].eq(17)].iloc[0]
    previous_reentry = daily.index[
        (daily.index < entry_date) & daily["transition"].eq("defender_to_momentum")
    ].max()
    momentum_days_before_emergency = int(
        daily.loc[previous_reentry:prior_date, "sleeve"].eq("momentum").sum()
    )
    volatility = float(trigger["signal_realized_volatility_20_asof_previous_close"])
    threshold = float(trigger["signal_cap_volatility_threshold_asof_previous_close"])
    raw_cap = min(1.0, threshold / volatility)
    cap_value = float(trigger["signal_volatility_cap_asof_previous_close"])
    shock_after_entry = event.loc[
        (event.index > entry_date) & (event.index <= trough_date)
    ]
    shock_candidate_return = float((1.0 + shock_after_entry["return"]).prod() - 1.0)
    shock_momentum_return = float(
        (1.0 + shock_after_entry["momentum_exact_return"]).prod() - 1.0
    )
    slow_min = float(event["slow_return_40_asof_previous_close"].min())

    slow_only = _lookup(ablation, "strategy", "slow_gate_only")
    cap_locked = _lookup(ablation, "strategy", "cap_respects_min_hold")
    selected_ablation = _lookup(ablation, "strategy", "selected_fusion")
    placebo_winner = placebo.loc[
        placebo["dominates_selected_all_three"].map(_truthy)
    ].iloc[0]

    bootstrap_cards = "".join(
        f'<div class="mini-card"><span>{int(row.block_days)}日块</span>'
        f'<strong>{float(row.strict_triple_rate):.1%}</strong>'
        '<small>三项目标同时为正</small></div>'
        for row in bootstrap.itertuples(index=False)
    )

    nav_delta = float(candidate["cagr_calendar"]) - float(
        exact_momentum["cagr_calendar"]
    )
    sharpe_delta = float(candidate["sharpe"]) - float(exact_momentum["sharpe"])
    mdd_delta = float(candidate["max_drawdown"]) - float(
        exact_momentum["max_drawdown"]
    )

    report = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Momentum × Defender 融合策略回测报告</title>
  <style>
    :root{{--ink:#122033;--muted:#5b6b7f;--line:#dbe3ee;--paper:#fff;--bg:#f3f6fa;
      --candidate:{COLORS['candidate']};--momentum:{COLORS['momentum']};--defender:{COLORS['defender']};
      --good:#15803d;--warn:#b45309;--bad:#b42318;--shadow:0 10px 28px rgba(30,50,80,.08)}}
    *{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--ink);font-family:-apple-system,BlinkMacSystemFont,
      "Segoe UI","PingFang SC","Microsoft YaHei",sans-serif;line-height:1.62}}
    a{{color:#0f5f9e}} .page{{max-width:1240px;margin:0 auto;padding:28px 24px 64px}}
    .hero{{background:linear-gradient(135deg,#102a43,#163b5c 62%,#0f766e);color:#fff;border-radius:22px;
      padding:34px 38px;box-shadow:var(--shadow)}} .eyebrow{{font-size:12px;letter-spacing:.12em;text-transform:uppercase;opacity:.72}}
    h1{{font-size:34px;line-height:1.2;margin:8px 0 12px}} .hero p{{max-width:850px;margin:0;color:#dce8f2}}
    .hero-meta{{display:flex;gap:10px;flex-wrap:wrap;margin-top:20px}} .badge{{display:inline-flex;align-items:center;
      padding:6px 11px;border-radius:999px;background:rgba(255,255,255,.12);font-size:13px}}
    .badge.warn{{background:#f59e0b;color:#241a02;font-weight:700}} .print-button{{margin-left:auto;border:1px solid rgba(255,255,255,.4);
      background:transparent;color:#fff;border-radius:9px;padding:7px 12px;cursor:pointer}}
    nav{{display:flex;gap:8px;flex-wrap:wrap;padding:14px 0 4px}} nav a{{text-decoration:none;color:#334155;background:#fff;
      border:1px solid var(--line);padding:6px 10px;border-radius:8px;font-size:13px}}
    section{{margin-top:28px}} h2{{font-size:23px;margin:0 0 12px}} h3{{font-size:17px;margin:0 0 8px}}
    .lede{{font-size:17px;color:#26384d;max-width:980px}} .grid{{display:grid;gap:14px}}
    .kpis{{grid-template-columns:repeat(4,minmax(0,1fr));margin-top:18px}} .kpi{{background:var(--paper);border:1px solid var(--line);
      border-radius:15px;padding:16px 18px;box-shadow:0 5px 16px rgba(30,50,80,.04)}} .kpi span{{display:block;color:var(--muted);font-size:13px}}
    .kpi strong{{display:block;font-size:26px;line-height:1.2;margin:5px 0}} .kpi small{{color:var(--muted)}}
    .good{{color:var(--good)}} .bad{{color:var(--bad)}} .positive{{color:var(--good);font-weight:650}}
    .negative{{color:var(--bad);font-weight:650}} .callout{{background:#fff;border:1px solid var(--line);border-left:5px solid var(--warn);
      border-radius:14px;padding:17px 19px;margin:14px 0}} .callout.good{{border-left-color:var(--good);color:var(--ink)}}
    .callout.bad{{border-left-color:var(--bad);color:var(--ink)}} .two-col{{grid-template-columns:repeat(2,minmax(0,1fr))}}
    .card{{background:var(--paper);border:1px solid var(--line);border-radius:16px;padding:20px;box-shadow:0 5px 16px rgba(30,50,80,.04)}}
    .chart-card{{background:#fff;border:1px solid var(--line);border-radius:16px;margin:14px 0;padding:12px 14px 6px;
      overflow:hidden}} .chart-card figcaption{{font-weight:750;margin:3px 0 4px}} .chart-card svg{{width:100%;height:auto;display:block}}
    .axis-label,.legend-label{{font-size:11px;fill:#5b6b7f}} .marker-label{{font-size:10px;font-weight:700}}
    .critical-bar{{stroke:#111827;stroke-width:2.2}} .formula{{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
      background:#0f172a;color:#e2e8f0;border-radius:12px;padding:15px 17px;overflow-x:auto;font-size:13px}}
    .timeline{{display:grid;grid-template-columns:repeat(5,1fr);gap:10px;margin:14px 0}} .timeline article{{position:relative;background:#fff;
      border:1px solid var(--line);border-radius:13px;padding:14px}} .timeline time{{font-weight:800;color:#0f5f9e}}
    .timeline p{{font-size:13px;margin:6px 0 0;color:#44556b}}
    .mini-grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}} .mini-card{{background:#f8fafc;border:1px solid var(--line);
      border-radius:12px;padding:13px}} .mini-card span,.mini-card small{{display:block;color:var(--muted);font-size:12px}}
    .mini-card strong{{font-size:21px}} .table-scroll{{overflow-x:auto;background:#fff;border:1px solid var(--line);border-radius:14px}}
    table{{border-collapse:collapse;width:100%;font-size:13px}} th,td{{padding:9px 10px;border-bottom:1px solid #e8edf4;text-align:right;white-space:nowrap}}
    th:first-child,td:first-child{{text-align:left}} thead th{{position:sticky;top:0;background:#eef3f8;color:#334155;z-index:2}}
    tbody tr:hover{{background:#f7fafc}} .selected-row{{background:#ecfdf5}} .critical-episode{{background:#fff7ed;outline:2px solid #fdba74;outline-offset:-2px}}
    .episode-controls{{display:flex;gap:18px;align-items:center;flex-wrap:wrap;margin:10px 0}}
    .episode-controls input[type=search]{{min-width:260px;border:1px solid #cbd5e1;border-radius:8px;padding:8px 10px}}
    .check-label{{font-size:13px;color:var(--muted)}} .episode-scroll{{max-height:660px}} .episode-scroll thead th{{top:0}}
    .note{{font-size:13px;color:var(--muted)}} details{{background:#fff;border:1px solid var(--line);border-radius:13px;padding:12px 15px}}
    summary{{font-weight:700;cursor:pointer}} .source-list{{columns:2;column-gap:28px}} footer{{color:var(--muted);font-size:12px;margin-top:34px}}
    @media(max-width:900px){{.kpis,.two-col,.timeline{{grid-template-columns:1fr 1fr}}.mini-grid{{grid-template-columns:1fr}}h1{{font-size:28px}}}}
    @media(max-width:560px){{.page{{padding:14px 12px 44px}}.hero{{padding:24px 20px}}.kpis,.two-col,.timeline{{grid-template-columns:1fr}}
      .print-button{{margin-left:0}}.source-list{{columns:1}}}}
    @media print{{body{{background:#fff}}.page{{max-width:none;padding:0}}.hero,.card,.chart-card,.kpi{{box-shadow:none}}nav,.print-button,.episode-controls{{display:none}}
      .episode-scroll{{max-height:none;overflow:visible}}section{{break-inside:avoid}}}}
  </style>
</head>
<body>
<main class="page">
  <header class="hero">
    <div class="eyebrow">Audited backtest · Frozen rules · Cutoff 2026-08-17</div>
    <h1>Momentum × Defender 融合策略回测报告</h1>
    <p>40日慢门控决定常态风险方向，30日最短持有抑制反复切换；Defender的波动率cap作为唯一紧急旁路，在下一开盘强制从Momentum转入Defender。</p>
    <div class="hero-meta">
      <span class="badge warn">结论：仅建议 Shadow</span><span class="badge">2019-01-18—2026-08-17</span>
      <span class="badge">1,837个交易日</span><span class="badge">43次切换</span><span class="badge">Defender 63.96%</span>
      <button class="print-button" type="button" onclick="window.print()">打印 / 存为PDF</button>
    </div>
  </header>
  <nav aria-label="报告目录"><a href="#conclusion">结论</a><a href="#performance">业绩</a><a href="#rules">规则</a>
    <a href="#event-2024">2024防守</a><a href="#episodes">22段明细</a><a href="#robustness">稳健性</a><a href="#audit">审计</a></nav>

  <section id="conclusion">
    <h2>结论先行</h2>
    <p class="lede">全样本上，融合候选同时提高收益和Sharpe，并把最大回撤从 {_pct(exact_momentum['max_drawdown'])} 收窄到 {_pct(candidate['max_drawdown'])}。但familywise maxT未通过5%门槛，且完整MDD优势依赖2024年一次防守，因此尚不能认定为生产替代。</p>
    <div class="grid kpis">
      <article class="kpi"><span>融合日历CAGR</span><strong>{_pct(candidate['cagr_calendar'])}</strong><small class="good">较Momentum {_pct(nav_delta, signed=True)}</small></article>
      <article class="kpi"><span>融合Sharpe</span><strong>{_number(candidate['sharpe'])}</strong><small class="good">较Momentum {_number(sharpe_delta, signed=True)}</small></article>
      <article class="kpi"><span>融合最大回撤</span><strong>{_pct(candidate['max_drawdown'])}</strong><small class="good">收窄 {_pct(mdd_delta, signed=True)}</small></article>
      <article class="kpi"><span>多重检验</span><strong class="bad">p={float(selected_max_t['value']):.3f}</strong><small>未通过5%正式门槛</small></article>
    </div>
    <div class="callout bad"><strong>最重要的限制：</strong>中和第17段（2024-10-08—2024-11-19）后，CAGR仍高出 {_pct(critical_leave_one['cagr_delta'], signed=True)}、Sharpe仍高 {_number(critical_leave_one['sharpe_delta'], signed=True)}，但最大回撤改善约为0。收益增强并非单事件驱动，完整样本MDD改善却是事件依赖的。</div>
  </section>

  <section id="performance">
    <h2>核心业绩与净值路径</h2>
    {_performance_table(performance)}
    <div class="grid two-col"><div>{full_nav_chart}</div><div>{drawdown_chart}</div></div>
    <p class="note">日历CAGR按首尾日期均计的日历天数年化；252日年化按交易日数；Sharpe使用零无风险利率和样本标准差；最大回撤从初始净值1锚定。图中累计净值使用精确分段Momentum作为主要基线。</p>
  </section>

  <section id="rules">
    <h2>冻结规则与 `signal_volatility_cap` 口径</h2>
    <div class="grid two-col">
      <article class="card"><h3>融合状态机</h3><ol>
        <li>每个收盘计算510300.SH的40交易日收益；高于2.5%为Momentum，否则为Defender。</li>
        <li>常态切换只在下一开盘执行，并要求当前策略已持有30个完整交易日。</li>
        <li><code>signal_volatility_cap &lt; 1</code>时，下一开盘可立即Momentum→Defender，不受30日锁限制。</li>
        <li>Defender→Momentum仍要求cap解除、慢门控risk-on、且Defender已满30日。</li>
      </ol></article>
      <article class="card"><h3>cap不是“净值跌了多少”</h3><p>它原本是Defender内部对锚定ETF <code>512890.SH</code> 的波动率限仓值。融合策略把“限仓开始生效”复用为二元紧急警报。它不看组合回撤，也不使用当日未来收盘后的价格执行。</p>
      <p><strong>行情口径：</strong>512890.SH固定基准后复权OHLC；20日Rogers–Satchell实现波动率；252日年化。</p></article>
    </div>
    <div class="formula">RSᵢ = max[ ln(Hᵢ/Cᵢ)·ln(Hᵢ/Oᵢ) + ln(Lᵢ/Cᵢ)·ln(Lᵢ/Oᵢ), 0 ]<br>
σ₂₀,t = √(252 × 最近20个锚交易日RSᵢ的均值)<br>
Q₈₀,t = 当前收盘之前全部有限σ₂₀历史的扩展80%分位（至少20个历史值，严格shift(1)）<br>
raw_cap = min(1, Q₈₀,t / σ₂₀,t)<br>
signal_volatility_cap = 0.2 × floor(raw_cap / 0.2 + ε)<br>
融合紧急信号 = signal_volatility_cap &lt; 1；收盘生成，按 signal_effective_next_open_date 在下一有效开盘执行。</div>
    <p class="note">cap取值为0、0.2、0.4、0.6、0.8、1.0；波动率或阈值无效时为1。这里的80%是“严格滞后的扩展历史分位”，不是滚动80日窗口。信号锚停牌时延用最近唯一观测，扩展分布不会重复计入填充值。</p>
  </section>

  <section id="event-2024">
    <h2>2024年成功防守：到底是什么起作用</h2>
    <div class="timeline">
      <article><time>09-27 开盘</time><p>慢门控risk-on，Defender→Momentum。</p></article>
      <article><time>09-30 收盘</time><p>σ₂₀={_pct(volatility, 3)} 高于Q₈₀={_pct(threshold, 3)}，cap降至{cap_value:.1f}。</p></article>
      <article><time>10-08 开盘</time><p>国庆后首个开盘，紧急旁路在仅持有Momentum {momentum_days_before_emergency}日时强制切Defender。</p></article>
      <article><time>10-09—10-17</time><p>Momentum从10/08收盘峰值回撤 {_pct(shock_momentum_return)}；融合同期 {_pct(shock_candidate_return, signed=True)}。</p></article>
      <article><time>11-19 开盘</time><p>cap已解除且Defender满30日，切回Momentum。</p></article>
    </div>
    <div class="grid two-col">
      <article class="card"><h3>触发计算</h3><p>2024-09-30收盘观测，国庆休市后于10月8日开盘生效：</p>
        <div class="formula">raw_cap = {_pct(threshold,3)} / {_pct(volatility,3)} = {raw_cap:.6f}<br>20%档位向下量化 → cap = {cap_value:.1f} &lt; 1</div>
        <p>当时40日慢门控整个事件窗口的最低收益仍为 {_pct(slow_min)}，远高于2.5%阈值，因此慢门控始终要求持有Momentum。若cap也遵守30日锁，这次切换会被阻断，并在锁期结束前已经解除。</p>
      </article>
      <article class="card"><h3>切换日收益链</h3><p>10月8日并没有提前放弃隔夜上涨：旧Momentum先承担隔夜段，再在开盘卖出；随后买入Defender并承担日内段。</p>
        <div class="formula">(1 + {_pct(trigger['exit_return_leg_used'],3)}) × (1 {_pct(trigger['enter_return_leg_used'],3, signed=True)}) − 1<br>= {_pct(trigger['return'],3, signed=True)}</div>
        <p>当日融合反而落后Momentum约 {_pct(float(trigger['return'])-float(trigger['momentum_exact_return']), signed=True)}；真正的保护来自10月9日之后已经处于Defender。</p>
      </article>
    </div>
    {event_chart}
    <div class="grid kpis">
      <article class="kpi"><span>第17段融合累计</span><strong>{_pct(episode_17['candidate_return'])}</strong><small>MDD {_pct(episode_17['candidate_max_drawdown'])}</small></article>
      <article class="kpi"><span>同期Momentum累计</span><strong>{_pct(episode_17['momentum_return'])}</strong><small>MDD {_pct(episode_17['momentum_max_drawdown'])}</small></article>
      <article class="kpi"><span>累计收益差</span><strong class="good">{_pct(episode_17['arithmetic_excess_return'], signed=True)}</strong><small>窗口31日，Defender状态30日</small></article>
      <article class="kpi"><span>关键参数角色</span><strong>cap入场</strong><small>30日锁负责延迟回归与降换手</small></article>
    </div>
    <div class="callout"><strong>重要机制细节：</strong>9月30日cap虽然为0.8，但Defender自身的40日区间网格目标已经是0.4，因此cap在Defender内部并未进一步压低目标。它在融合策略中主要充当“切换触发器”。这说明2024效果来自对cap字段的二元复用，而不是cap数值0.8直接决定了组合风险仓位。</div>
    <h3>机制消融</h3>
    <div class="table-scroll"><table><thead><tr><th>规则</th><th>CAGR</th><th>Sharpe</th><th>最大回撤</th><th>解释</th></tr></thead><tbody>
      <tr><th>仅40日慢门控</th><td>{_pct(slow_only['cagr_calendar'])}</td><td>{_number(slow_only['sharpe'])}</td><td>{_pct(slow_only['max_drawdown'])}</td><td>没有躲过2024回撤</td></tr>
      <tr><th>cap仍受30日锁</th><td>{_pct(cap_locked['cagr_calendar'])}</td><td>{_number(cap_locked['sharpe'])}</td><td>{_pct(cap_locked['max_drawdown'])}</td><td>10月8日被锁定阻断</td></tr>
      <tr class="selected-row"><th>cap紧急旁路</th><td>{_pct(selected_ablation['cagr_calendar'])}</td><td>{_number(selected_ablation['sharpe'])}</td><td>{_pct(selected_ablation['max_drawdown'])}</td><td>唯一实质修复MDD的版本</td></tr>
    </tbody></table></div>
  </section>

  <section id="episodes">
    <h2>全部22段Defender持有窗口</h2>
    <p>窗口从切入Defender的开盘日开始，并包含切出Defender、切回Momentum的开盘日整日收益；因此已结束窗口的“窗口日数”通常比实际Defender状态日数多1。最后一段截至cutoff仍未退出。累计收益是主要观察指标；30—50日短窗的年化收益和Sharpe会被显著放大，只作辅助。</p>
    {episode_chart}
    {_episode_table(episodes)}
  </section>

  <section id="robustness">
    <h2>年度表现、稳健性与过拟合风险</h2>
    {_annual_table(annual)}
    <div class="grid two-col" style="margin-top:14px">
      <article class="card"><h3>支持性证据</h3><ul>
        <li>局部45组参数严格通过 {int(local_strict['passed_count'])}/45，实质通过 {int(local_material['passed_count'])}/45。</li>
        <li>cap分位60%—95%共7/7实质通过；80%仅因它复现冻结信号而保留。</li>
        <li>成本0—10倍共5/5通过；慢信号和cap额外延迟0—2日共9/9通过。</li>
        <li>52个约36个月窗口全部通过，但相邻窗口重叠97.22%，不是52份独立样本。</li>
      </ul><div class="mini-grid">{bootstrap_cards}</div><p class="note">Bootstrap固定历史状态路径并按块重排，不代表未来成功概率。</p></article>
      <article class="card"><h3>不能忽略的负面证据</h3><ul>
        <li>45点族omnibus maxT：p={float(omnibus_max_t['value']):.4f}；最终参数familywise maxT：p={float(selected_max_t['value']):.4f}；均未过5%。</li>
        <li>cap时点安慰剂有 {int(placebo_strict['passed_count'])}/82 仍严格三项改善，{int(placebo_dominates['passed_count'])}/82 同时支配正式候选。</li>
        <li>支配候选的不可交易循环平移版本：平移{int(placebo_winner['circular_shift_days'])}日，CAGR {_pct(placebo_winner['cagr_calendar'])}、Sharpe {_number(placebo_winner['sharpe'])}、MDD {_pct(placebo_winner['max_drawdown'])}。</li>
        <li>leave-one-episode-out为21/22通过；唯一例外是第17段，完整样本MDD优势依赖该事件。</li>
      </ul></article>
    </div>
    <div class="callout bad"><strong>决策：</strong>参数、成本和延迟稳定性较好，但没有真正未参与研究的样本；多重检验失败且MDD事件依赖。因此当前定位是规则冻结后的前瞻shadow，不是生产替换。</div>
  </section>

  <section id="audit">
    <h2>数据审计与使用边界</h2>
    <div class="grid two-col">
      <article class="card"><h3>机械口径</h3><ul>
        <li>样本固定为2019-01-18—2026-08-17，共1,837行；日期唯一递增。</li>
        <li>切换日使用旧策略退出腿 × 新策略进入腿，不重复使用连续持有收益。</li>
        <li>2021-10-22保留；512890停牌时不再被误用为唯一收益日历。</li>
        <li>{len(checks)}/{len(checks)}项最终复现与接口检查通过。</li>
      </ul></article>
      <article class="card"><h3>上线前仍未覆盖</h3><ul>
        <li>成本压力是线性费用放大，不含开盘冲击成本、容量、部分成交和极端价差。</li>
        <li>cap只能在收盘确认、下一有效开盘执行，无法防守收盘后的瞬时隔夜跳空。</li>
        <li>所有稳健性切片仍来自同一历史；未来shadow期间若修改规则，应重置验证起点。</li>
      </ul></article>
    </div>
    <details><summary>报告数据文件</summary><ul class="source-list">
      <li><a href="performance_summary.csv">performance_summary.csv</a></li>
      <li><a href="selected_strategy_daily.csv">selected_strategy_daily.csv</a></li>
      <li><a href="defender_episode_metrics.csv">defender_episode_metrics.csv</a></li>
      <li><a href="switch_events.csv">switch_events.csv</a></li>
      <li><a href="annual_metrics.csv">annual_metrics.csv</a></li>
      <li><a href="ablation_metrics.csv">ablation_metrics.csv</a></li>
      <li><a href="selection_bias_checks.csv">selection_bias_checks.csv</a></li>
      <li><a href="reproduction_checks.csv">reproduction_checks.csv</a></li>
      <li><a href="research_report.md">research_report.md</a></li>
    </ul></details>
  </section>
  <footer>生成日期：{date.today().isoformat()}。本报告是历史研究记录，不构成未来收益保证；页面所有图表均内嵌，可离线打开。</footer>
</main>
<script>
  const search = document.getElementById('episode-search');
  const positiveOnly = document.getElementById('positive-only');
  const episodeRows = [...document.querySelectorAll('#episode-table tbody tr')];
  function filterEpisodes() {{
    const term = search.value.trim().toLowerCase();
    episodeRows.forEach(row => {{
      const matchesText = !term || row.textContent.toLowerCase().includes(term);
      const matchesSign = !positiveOnly.checked || row.dataset.positive === 'true';
      row.hidden = !(matchesText && matchesSign);
    }});
  }}
  search.addEventListener('input', filterEpisodes);
  positiveOnly.addEventListener('change', filterEpisodes);
</script>
</body>
</html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    return output_path
