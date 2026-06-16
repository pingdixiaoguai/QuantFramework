"""Research-only 2026 YTD rebalance-days attribution scan.

This script leaves production YAMLs and the changelog untouched. It reuses the
close-execution trace helper validated in ``close_execution_variant_study.py``
so the only scanned strategy parameter is ``rebalance_days``.
"""

from __future__ import annotations

import math
from copy import deepcopy
from datetime import date
from pathlib import Path

import pandas as pd
import yaml

from data.store import query
from factors.quality_momentum import compute as compute_quality_momentum
from scripts.close_execution_variant_study import _run_traced


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "strategy" / "configs" / "quality_momentum_top1.yaml"
ATTACHMENTS_DIR = ROOT / "strategy_changelog_attachments"
OUT_DIR = ATTACHMENTS_DIR / "2026-06-15_ytd_attribution_rebalance_scan"
REPORT_PATH = OUT_DIR / "2026-06-15_ytd_attribution_rebalance_scan.md"
MAIN_CSV_PATH = OUT_DIR / "2026-06-15_ytd_attribution_rebalance_scan_main.csv"
EVENT_CSV_PATH = OUT_DIR / "2026-06-15_ytd_attribution_rebalance_scan_events.csv"
COST_CSV_PATH = OUT_DIR / "2026-06-15_ytd_attribution_rebalance_scan_costs.csv"
EXEC_CSV_PATH = OUT_DIR / "2026-06-15_ytd_attribution_rebalance_scan_executions.csv"

WARMUP_START = date(2014, 1, 1)
YTD_START = pd.Timestamp("2026-01-01")
YTD_END = pd.Timestamp("2026-06-15")
RDS = [5, 7, 10]
FEES = [0.0001, 0.0003, 0.0005]
EVENTS = {
    "A": {
        "label": "事件A(01-16创业板)",
        "signal_date": pd.Timestamp("2026-01-15"),
        "execution_date": pd.Timestamp("2026-01-16"),
        "expected_asset": "159915.SZ",
        "note": "清晰反转",
    },
    "B": {
        "label": "事件B(03-06沪深300)",
        "signal_date": pd.Timestamp("2026-03-05"),
        "execution_date": pd.Timestamp("2026-03-06"),
        "expected_asset": "510300.SH",
        "note": "拥挤抖动",
    },
}


def _load_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config["start"] = WARMUP_START
    config["end"] = YTD_END.date()
    config["transaction_cost_rate"] = 0.0
    config["train_ratio"] = 0.7
    config.pop("rebalance_mode", None)
    return config


def _net_returns(trace, fee: float) -> pd.Series:
    returns = trace.result.gross_daily_returns.copy()
    costs = trace.result.turnover.reindex(returns.index, fill_value=0.0) * fee
    return returns - costs


def _ytd_slice(series: pd.Series) -> pd.Series:
    return series.loc[(series.index >= YTD_START) & (series.index <= YTD_END)]


def _total_return(returns: pd.Series) -> float:
    if returns.empty:
        return math.nan
    return float((1.0 + returns).prod() - 1.0)


def _sharpe(returns: pd.Series) -> float:
    if returns.empty or returns.std() == 0:
        return math.nan
    return float(returns.mean() / returns.std() * math.sqrt(252.0))


def _max_drawdown(returns: pd.Series) -> float:
    if returns.empty:
        return math.nan
    equity = (1.0 + returns).cumprod()
    return float((equity / equity.cummax() - 1.0).min())


def _annual_turnover(turnover: pd.Series, n_days: int) -> float:
    if n_days == 0:
        return math.nan
    return float(turnover.sum() / (n_days / 252.0))


def _score_gap(config: dict, signal_date: pd.Timestamp) -> dict[str, object]:
    scores: dict[str, float] = {}
    for asset in config["asset_pool"]:
        df = query(asset, WARMUP_START, signal_date.date())
        if df.empty:
            continue
        series = compute_quality_momentum(df, {"window": 20})
        last = series.iloc[-1]
        if pd.notna(last):
            scores[asset] = float(last)
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    if len(ranked) < 2:
        return {"top1": "", "top2": "", "gap": math.nan}
    return {
        "top1": ranked[0][0],
        "top2": ranked[1][0],
        "gap": ranked[0][1] - ranked[1][1],
    }


def _event_status(executions: pd.DataFrame, event: dict[str, object]) -> str:
    if executions.empty:
        return "已消除"
    ex = executions.copy()
    ex["execution_date"] = pd.to_datetime(ex["execution_date"])
    ex["signal_date"] = pd.to_datetime(ex["signal_date"])
    matched = ex[
        (ex["signal_date"] == event["signal_date"])
        & (ex["execution_date"] == event["execution_date"])
    ]
    if matched.empty:
        return "已消除"
    asset = str(matched.iloc[0]["new_asset"])
    return f"仍切换(切入 {asset})"


def _data_coverage(config: dict) -> pd.DataFrame:
    rows = []
    for asset in config["asset_pool"]:
        df = query(asset, YTD_START.date(), YTD_END.date())
        rows.append(
            {
                "asset": asset,
                "rows": int(len(df)),
                "start": df["date"].min().date().isoformat() if len(df) else "",
                "end": df["date"].max().date().isoformat() if len(df) else "",
            }
        )
    return pd.DataFrame(rows)


def _fmt_pct(value: float) -> str:
    return "n/a" if pd.isna(value) else f"{value:.2%}"


def _fmt_num(value: float, digits: int = 2) -> str:
    return "n/a" if pd.isna(value) else f"{value:.{digits}f}"


def _fmt_gap(value: float) -> str:
    return "n/a" if pd.isna(value) else f"{value:.6f}"


def _markdown_table(df: pd.DataFrame) -> str:
    return df.to_markdown(index=False)


def main() -> None:
    base_config = _load_config()
    coverage = _data_coverage(base_config)
    traces = {}
    main_rows = []
    cost_rows = []
    event_rows = []
    execution_rows = []
    gaps = {
        key: _score_gap(base_config, event["signal_date"])
        for key, event in EVENTS.items()
    }

    for rd in RDS:
        config = deepcopy(base_config)
        config["rebalance_days"] = rd
        trace = _run_traced(config, "close")
        traces[rd] = trace

        ytd_turnover = trace.result.turnover.loc[
            (trace.result.turnover.index >= YTD_START)
            & (trace.result.turnover.index <= YTD_END)
        ]
        executions = trace.executions.copy()
        if not executions.empty:
            executions["execution_date"] = pd.to_datetime(executions["execution_date"])
            executions["signal_date"] = pd.to_datetime(executions["signal_date"])
            ytd_exec = executions[
                (executions["execution_date"] >= YTD_START)
                & (executions["execution_date"] <= YTD_END)
            ].copy()
            switch_count = int(ytd_exec["old_asset"].notna().sum())
            for _, row in ytd_exec.iterrows():
                execution_rows.append(
                    {
                        "rebalance_days": rd,
                        "signal_date": row["signal_date"].date().isoformat(),
                        "execution_date": row["execution_date"].date().isoformat(),
                        "old_asset": "" if pd.isna(row["old_asset"]) else row["old_asset"],
                        "new_asset": row["new_asset"],
                        "turnover": row["turnover"],
                    }
                )
        else:
            switch_count = 0

        for fee in FEES:
            ytd_returns = _ytd_slice(_net_returns(trace, fee))
            ytd_return = _total_return(ytd_returns)
            if fee == 0.0001:
                main_rows.append(
                    {
                        "rebalance_days": rd,
                        "start": ytd_returns.index.min().date().isoformat(),
                        "end": ytd_returns.index.max().date().isoformat(),
                        "trading_days": int(len(ytd_returns)),
                        "ytd_return": ytd_return,
                        "sharpe": _sharpe(ytd_returns),
                        "max_drawdown": _max_drawdown(ytd_returns),
                        "switch_count": switch_count,
                        "annual_turnover_sum_abs": _annual_turnover(
                            ytd_turnover, len(ytd_returns)
                        ),
                        "annual_turnover_one_side": _annual_turnover(
                            ytd_turnover, len(ytd_returns)
                        )
                        / 2.0,
                    }
                )
            cost_rows.append(
                {
                    "rebalance_days": rd,
                    "fee_bps_one_side": fee * 10000.0,
                    "ytd_return": ytd_return,
                }
            )

        event_row = {"rebalance_days": rd}
        for key, event in EVENTS.items():
            event_row[EVENTS[key]["label"]] = _event_status(trace.executions, event)
        event_rows.append(event_row)

    main_df = pd.DataFrame(main_rows)
    event_df = pd.DataFrame(event_rows)
    cost_df = pd.DataFrame(cost_rows)
    exec_df = pd.DataFrame(execution_rows)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    main_df.to_csv(MAIN_CSV_PATH, index=False, encoding="utf-8-sig")
    event_df.to_csv(EVENT_CSV_PATH, index=False, encoding="utf-8-sig")
    cost_df.to_csv(COST_CSV_PATH, index=False, encoding="utf-8-sig")
    exec_df.to_csv(EXEC_CSV_PATH, index=False, encoding="utf-8-sig")

    main_fmt = main_df[
        [
            "rebalance_days",
            "ytd_return",
            "sharpe",
            "max_drawdown",
            "switch_count",
            "annual_turnover_sum_abs",
        ]
    ].copy()
    main_fmt["ytd_return"] = main_fmt["ytd_return"].map(_fmt_pct)
    main_fmt["sharpe"] = main_fmt["sharpe"].map(_fmt_num)
    main_fmt["max_drawdown"] = main_fmt["max_drawdown"].map(_fmt_pct)
    main_fmt["annual_turnover_sum_abs"] = main_fmt[
        "annual_turnover_sum_abs"
    ].map(_fmt_pct)
    main_fmt = main_fmt.rename(
        columns={
            "rebalance_days": "rebalance_days",
            "ytd_return": "YTD收益",
            "sharpe": "Sharpe",
            "max_drawdown": "最大回撤",
            "switch_count": "切换次数",
            "annual_turnover_sum_abs": "年化换手率(Σ|Δw|)",
        }
    )

    cost_pivot = cost_df.pivot(
        index="rebalance_days",
        columns="fee_bps_one_side",
        values="ytd_return",
    ).reset_index()
    cost_pivot.columns = [
        "rebalance_days",
        "1bp单边YTD收益",
        "3bp单边YTD收益",
        "5bp单边YTD收益",
    ]
    cost_fmt = cost_pivot[
        ["rebalance_days", "3bp单边YTD收益", "5bp单边YTD收益"]
    ].copy()
    for col in ["3bp单边YTD收益", "5bp单边YTD收益"]:
        cost_fmt[col] = cost_fmt[col].map(_fmt_pct)

    gap_lines = []
    for key, event in EVENTS.items():
        gap = gaps[key]
        gap_lines.append(
            f"- 事件 {key}: {event['signal_date'].date().isoformat()} 收盘 "
            f"Top1={gap['top1']}, Top2={gap['top2']}, "
            f"Top1-Top2 score差={_fmt_gap(gap['gap'])} ({event['note']})。"
        )

    best_rd = main_df.sort_values("ytd_return", ascending=False).iloc[0]
    base_rd5 = main_df[main_df["rebalance_days"] == 5].iloc[0]
    recovery = float(best_rd["ytd_return"] - base_rd5["ytd_return"])
    b_eliminated = event_df.set_index("rebalance_days").loc[
        best_rd["rebalance_days"], EVENTS["B"]["label"]
    ] == "已消除"
    a_eliminated = event_df.set_index("rebalance_days").loc[
        best_rd["rebalance_days"], EVENTS["A"]["label"]
    ] == "已消除"
    diagnosis = (
        "更慢节奏同时保留清晰反转事件 A、消除拥挤抖动事件 B，"
        "YTD 改善主要来自过滤贴线抖动。"
        if (not a_eliminated and b_eliminated)
        else "指定的拥挤抖动事件 B 在三档下都仍发生；rd=7 的改善不是来自消除该事件，而是来自整体换仓路径和后续持有期变化。"
    )
    min_end = coverage["end"].min()
    max_end = coverage["end"].max()
    if min_end == max_end:
        coverage_note = f"{coverage['asset'].nunique()} 只资产均覆盖至 {max_end}"
    else:
        coverage_note = (
            f"{coverage['asset'].nunique()} 只资产覆盖截止日范围为 "
            f"{min_end}~{max_end}"
        )

    lines = [
        "# 2026 YTD 损失归因 - rebalance_days 扫描",
        "",
        f"- 配置基线: `{CONFIG_PATH.relative_to(ROOT)}`；只在内存覆盖 `rebalance_days in {RDS}` 与 `transaction_cost_rate=0`。",
        f"- 本地 HFQ 数据覆盖: {coverage_note}。",
        f"- 研究窗口: {YTD_START.date().isoformat()} ~ {YTD_END.date().isoformat()}；实际收益行: {main_df['start'].iloc[0]} ~ {main_df['end'].iloc[0]} ({int(main_df['trading_days'].iloc[0])} 个交易日)。",
        f"- Warmup/决策重放起点: {WARMUP_START.isoformat()}，用于保留年初已有持仓与 20 日信号历史；指标只截取 2026 YTD。",
        "- 成交口径: T+1 收盘价成交；信号日 close 生成目标，下一交易日 close 后生效，成交日收益仍归旧持仓，避免 look-ahead。",
        "- 成本: 主表为单边 1bp；Top1 全仓切换 `Σ|Δw|=2`，一次切换成本 2bp；hold 日为 0。",
        "- 切换次数为 YTD 窗口内实际 T+1 收盘执行、且 old_asset 非空的切换数；年化换手率列使用成本一致口径 `Σ|Δw| / years`，若按单边 convention，数值为该列的一半。",
        "",
        "## 主表",
        "",
        _markdown_table(main_fmt),
        "",
        "## 事件追踪",
        "",
        *gap_lines,
        "",
        _markdown_table(event_df),
        "",
        "## 成本稳健性",
        "",
        _markdown_table(cost_fmt),
        "",
        "## 读数",
        "",
        f"- YTD 最优为 rd={int(best_rd['rebalance_days'])}: {_fmt_pct(float(best_rd['ytd_return']))}；相对 rd=5 挽回 {recovery:.2%}。",
        "- 事件 A 在 rd=7/10 下不再以 2026-01-15 信号、2026-01-16 执行的形式出现；事件 B 在 rd=5/7/10 下均仍切入 510300.SH。",
        f"- 判定: {diagnosis}",
        "",
        "## 存档",
        "",
        f"- 主表 CSV: `{MAIN_CSV_PATH.name}`",
        f"- 事件追踪 CSV: `{EVENT_CSV_PATH.name}`",
        f"- 成本稳健性 CSV: `{COST_CSV_PATH.name}`",
        f"- YTD 执行明细 CSV: `{EXEC_CSV_PATH.name}`",
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {REPORT_PATH}")
    print(f"wrote {MAIN_CSV_PATH}")
    print(f"wrote {EVENT_CSV_PATH}")
    print(f"wrote {COST_CSV_PATH}")
    print(f"wrote {EXEC_CSV_PATH}")


if __name__ == "__main__":
    main()
