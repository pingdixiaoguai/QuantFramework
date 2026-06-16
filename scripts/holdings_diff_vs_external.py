"""Research-only holdings diff between quality_momentum_top1 and an external path.

No returns are computed here. Strategy X is replayed with the close-execution
trace helper so its daily holding path aligns with the external tail-close
holding convention. Strategy Y is the manual holding sequence from the research
request and is not reverse engineered.
"""

from __future__ import annotations

from collections import Counter
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
OUT_DIR = ATTACHMENTS_DIR / "2026-06-15_holdings_diff_vs_external"
REPORT_PATH = OUT_DIR / "2026-06-15_holdings_diff_vs_external.md"
DAILY_CSV_PATH = OUT_DIR / "2026-06-15_holdings_diff_vs_external_daily.csv"
DIFF_CSV_PATH = OUT_DIR / "2026-06-15_holdings_diff_vs_external_diff_intervals.csv"

WARMUP_START = date(2014, 1, 1)
START = pd.Timestamp("2025-10-16")
END = pd.Timestamp("2026-06-15")
ASSET_NAMES = {
    "510300.SH": "沪深300",
    "159915.SZ": "创业板",
    "513100.SH": "纳指",
    "518880.SH": "黄金",
}
Y_SEGMENTS = [
    ("2025-10-16", "2025-10-29", "518880.SH"),
    ("2025-10-30", "2025-11-19", "513100.SH"),
    ("2025-11-20", "2025-11-26", "159915.SZ"),
    ("2025-11-27", "2025-12-10", "518880.SH"),
    ("2025-12-11", "2025-12-17", "159915.SZ"),
    ("2025-12-18", "2026-01-04", "518880.SH"),
    ("2026-01-05", "2026-01-11", "159915.SZ"),
    ("2026-01-12", "2026-03-16", "518880.SH"),
    ("2026-03-17", "2026-04-07", "159915.SZ"),
    ("2026-04-08", "2026-04-14", "513100.SH"),
    ("2026-04-15", "2026-04-28", "159915.SZ"),
    ("2026-04-29", "2026-05-10", "513100.SH"),
    ("2026-05-11", "2026-05-17", "159915.SZ"),
    ("2026-05-18", "2026-06-15", "513100.SH"),
]


def _load_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config["start"] = WARMUP_START
    config["end"] = END.date()
    config["rebalance_days"] = 5
    config["transaction_cost_rate"] = 0.0
    config["train_ratio"] = 0.7
    config.pop("rebalance_mode", None)
    return config


def _trading_calendar(config: dict) -> pd.DatetimeIndex:
    dates: set[pd.Timestamp] = set()
    for asset in config["asset_pool"]:
        df = query(asset, START.date(), END.date())
        dates.update(pd.Timestamp(dt) for dt in df["date"])
    return pd.DatetimeIndex(sorted(dates))


def _asset_from_row(row: pd.Series) -> str:
    weights = {
        str(asset): float(value)
        for asset, value in row.items()
        if pd.notna(value) and float(value) != 0.0
    }
    if not weights:
        return ""
    return max(weights, key=weights.get)


def _x_daily_holdings(trace, calendar: pd.DatetimeIndex) -> pd.Series:
    positions = trace.result.positions.copy()
    positions.index = pd.to_datetime(positions.index)
    positions = positions.fillna(0.0)
    aligned_index = positions.index.union(calendar).sort_values()
    positions = positions.reindex(aligned_index).ffill().loc[calendar]
    return positions.apply(_asset_from_row, axis=1)


def _y_daily_holdings(calendar: pd.DatetimeIndex) -> pd.Series:
    y = pd.Series(index=calendar, dtype=object)
    for start, end, asset in Y_SEGMENTS:
        mask = (calendar >= pd.Timestamp(start)) & (calendar <= pd.Timestamp(end))
        y.loc[mask] = asset
    if y.isna().any():
        missing = [dt.date().isoformat() for dt in y[y.isna()].index[:5]]
        raise RuntimeError(f"Y holding path has uncovered trading days: {missing}")
    return y


def _compact_path(values: pd.Series) -> str:
    out: list[str] = []
    for value in values:
        if not out or out[-1] != value:
            out.append(value)
    return " -> ".join(out)


def _asset_label(asset: str) -> str:
    return f"{asset} {ASSET_NAMES.get(asset, '')}".strip()


def _score_snapshot(config: dict, score_date: pd.Timestamp) -> dict[str, object]:
    scores: dict[str, float] = {}
    for asset in config["asset_pool"]:
        df = query(asset, WARMUP_START, score_date.date())
        if df.empty:
            continue
        series = compute_quality_momentum(df, {"window": 20})
        last = series.iloc[-1]
        if pd.notna(last):
            scores[asset] = float(last)
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    if len(ranked) < 2:
        return {
            "score_date": score_date.date().isoformat(),
            "score_ranking": "",
            "top1": "",
            "top2": "",
            "top1_top2_gap": pd.NA,
            "crowded": "",
        }
    ranking = "; ".join(
        f"{idx + 1}.{asset}={score:.6f}" for idx, (asset, score) in enumerate(ranked)
    )
    gap = ranked[0][1] - ranked[1][1]
    return {
        "score_date": score_date.date().isoformat(),
        "score_ranking": ranking,
        "top1": ranked[0][0],
        "top2": ranked[1][0],
        "top1_top2_gap": gap,
        "crowded": "Y" if gap < 0.001 else "N",
    }


def _x_switches(trace) -> pd.DataFrame:
    executions = trace.executions.copy()
    if executions.empty:
        return pd.DataFrame(
            columns=["execution_date", "signal_date", "old_asset", "new_asset"]
        )
    executions["execution_date"] = pd.to_datetime(executions["execution_date"])
    executions["signal_date"] = pd.to_datetime(executions["signal_date"])
    switches = executions[executions["old_asset"].notna()].copy()
    return switches[["execution_date", "signal_date", "old_asset", "new_asset"]]


def _score_date_for_interval(
    interval_start: pd.Timestamp,
    interval_end: pd.Timestamp,
    switches: pd.DataFrame,
) -> tuple[pd.Timestamp, str]:
    inside = switches[
        (switches["execution_date"] >= interval_start)
        & (switches["execution_date"] <= interval_end)
    ].sort_values("execution_date")
    if not inside.empty:
        return pd.Timestamp(inside.iloc[0]["signal_date"]), "first_x_switch_in_interval"
    exact = switches[switches["execution_date"] == interval_start]
    if not exact.empty:
        return pd.Timestamp(exact.iloc[0]["signal_date"]), "x_switch_at_interval_start"
    return interval_start, "interval_start_no_x_switch"


def _classify_interval(
    daily: pd.DataFrame,
    start_pos: int,
    end_pos: int,
) -> tuple[str, str]:
    rows = daily.iloc[start_pos : end_pos + 1]
    y_path = _compact_path(rows["Y持仓"])
    x_path = _compact_path(rows["X持仓"])
    prev_same = start_pos > 0 and daily.iloc[start_pos - 1]["X持仓"] == daily.iloc[start_pos - 1]["Y持仓"]
    next_same = (
        end_pos + 1 < len(daily)
        and daily.iloc[end_pos + 1]["X持仓"] == daily.iloc[end_pos + 1]["Y持仓"]
    )
    prev_asset = daily.iloc[start_pos - 1]["X持仓"] if start_pos > 0 else ""
    next_asset = daily.iloc[end_pos + 1]["X持仓"] if end_pos + 1 < len(daily) else ""
    y_unique = list(dict.fromkeys(rows["Y持仓"].tolist()))
    x_unique = list(dict.fromkeys(rows["X持仓"].tolist()))

    if (
        len(y_unique) == 1
        and prev_same
        and next_same
        and prev_asset == next_asset == y_unique[0]
    ):
        return "whipsaw", f"X 离开并回到 {_asset_label(y_unique[0])}; Y 全程持有。"

    boundary_assets = {prev_asset, next_asset} - {""}
    interval_assets = set(y_unique) | set(x_unique)
    if prev_same and next_same and len(boundary_assets) == 2 and interval_assets <= boundary_assets:
        return "相位错位", "前后持仓一致，区间内只是在同两只资产之间切换时点不同。"

    return "选择差异", f"X路径 {x_path}; Y路径 {y_path}。"


def _diff_intervals(
    daily: pd.DataFrame,
    switches: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    rows = []
    diff = daily["是否分歧"].tolist()
    i = 0
    while i < len(diff):
        if not diff[i]:
            i += 1
            continue
        start_pos = i
        while i + 1 < len(diff) and diff[i + 1]:
            i += 1
        end_pos = i
        start = pd.Timestamp(daily.iloc[start_pos]["date"])
        end = pd.Timestamp(daily.iloc[end_pos]["date"])
        interval = daily.iloc[start_pos : end_pos + 1]
        x_switch_count = int(
            (
                (switches["execution_date"] >= start)
                & (switches["execution_date"] <= end)
            ).sum()
        )
        classification, note = _classify_interval(daily, start_pos, end_pos)
        score_date, score_basis = _score_date_for_interval(start, end, switches)
        score = _score_snapshot(config, score_date)
        rows.append(
            {
                "start": start.date().isoformat(),
                "end": end.date().isoformat(),
                "trading_days": int(len(interval)),
                "X持仓": _compact_path(interval["X持仓"]),
                "Y持仓": _compact_path(interval["Y持仓"]),
                "whipsaw标志": "Y" if classification == "whipsaw" else "N",
                "X切换次数": x_switch_count,
                "分类": classification,
                "分类说明": note,
                "score_basis": score_basis,
                **score,
            }
        )
        i += 1
    return pd.DataFrame(rows)


def _y_switch_fingerprint(calendar: pd.DatetimeIndex) -> pd.DataFrame:
    starts = [pd.Timestamp(start) for start, _, _ in Y_SEGMENTS]
    rows = []
    for idx in range(1, len(starts)):
        prev = starts[idx - 1]
        curr = starts[idx]
        if prev not in calendar or curr not in calendar:
            raise RuntimeError(f"Y switch date not in trading calendar: {prev} -> {curr}")
        prev_idx = int(calendar.get_loc(prev))
        curr_idx = int(calendar.get_loc(curr))
        interval = curr_idx - prev_idx
        rows.append(
            {
                "switch_date": curr.date().isoformat(),
                "from_asset": Y_SEGMENTS[idx - 1][2],
                "to_asset": Y_SEGMENTS[idx][2],
                "trading_day_interval": interval,
                "multiple_of_5": "Y" if interval % 5 == 0 else "N",
            }
        )
    return pd.DataFrame(rows)


def _format_daily_for_report(daily: pd.DataFrame) -> pd.DataFrame:
    show = daily.copy()
    show["date"] = show["date"].map(lambda dt: pd.Timestamp(dt).date().isoformat())
    show["是否分歧"] = show["是否分歧"].map(lambda value: "Y" if value else "N")
    return show


def _format_diff_for_report(diff_df: pd.DataFrame) -> pd.DataFrame:
    show = diff_df.copy()
    if show.empty:
        return show
    show["top1_top2_gap"] = show["top1_top2_gap"].map(
        lambda value: "" if pd.isna(value) else f"gap={float(value):.6f}"
    )
    return show


def _pair_summary(diff_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in diff_df[diff_df["分类"] == "选择差异"].iterrows():
        xs = [part.strip() for part in str(row["X持仓"]).split("->")]
        ys = [part.strip() for part in str(row["Y持仓"]).split("->")]
        pair = f"X {xs[0]} / Y {ys[0]}" if len(xs) == len(ys) == 1 else f"X {row['X持仓']} / Y {row['Y持仓']}"
        rows.append({"资产对": pair, "交易日": int(row["trading_days"])})
    if not rows:
        return pd.DataFrame(columns=["资产对", "交易日"])
    df = pd.DataFrame(rows)
    return df.groupby("资产对", as_index=False)["交易日"].sum().sort_values("交易日", ascending=False)


def _summary_table(diff_df: pd.DataFrame) -> pd.DataFrame:
    order = ["whipsaw", "选择差异", "相位错位"]
    if diff_df.empty:
        grouped = pd.DataFrame(columns=["分类", "区间数", "交易日"])
    else:
        grouped = diff_df.groupby("分类", as_index=False).agg(
            区间数=("分类", "size"),
            交易日=("trading_days", "sum"),
        )
    rows = []
    indexed = grouped.set_index("分类") if len(grouped) else grouped
    for label in order:
        if len(grouped) and label in indexed.index:
            rows.append(
                {
                    "分类": label,
                    "区间数": int(indexed.loc[label, "区间数"]),
                    "交易日": int(indexed.loc[label, "交易日"]),
                }
            )
        else:
            rows.append({"分类": label, "区间数": 0, "交易日": 0})
    return pd.DataFrame(rows)


def _write_report(
    daily: pd.DataFrame,
    diff_df: pd.DataFrame,
    y_fp: pd.DataFrame,
) -> None:
    daily_show = _format_daily_for_report(daily)
    diff_show = _format_diff_for_report(diff_df)
    summary = _summary_table(diff_df)
    pair_summary = _pair_summary(diff_df)
    multiples = int((y_fp["multiple_of_5"] == "Y").sum())
    non_multiples = int((y_fp["multiple_of_5"] == "N").sum())
    intervals = ", ".join(str(v) for v in y_fp["trading_day_interval"].tolist())
    min_interval = int(y_fp["trading_day_interval"].min()) if len(y_fp) else 0
    cadence_read = (
        "存在非 5 倍数间隔，证据更偏向 min_hold 或其他非固定 5 日网格；不下定论。"
        if non_multiples
        else "所有间隔均为 5 的整数倍；该证据更接近 fixed_cycle 指纹，但仍不反推 Y 信号。"
    )
    selection_days = int(
        diff_df.loc[diff_df["分类"] == "选择差异", "trading_days"].sum()
    )
    lines = [
        "# 实盘动量策略 vs 外部对标策略 - 逐日持仓 diff 归因",
        "",
        f"- 策略 X: `{CONFIG_PATH.relative_to(ROOT)}`，内存固定 `rebalance_days=5`，T+1 收盘成交重放。",
        f"- 策略 Y: 人工调仓记录展开；不反推信号，不计算收益。",
        f"- 区间: {START.date().isoformat()} ~ {END.date().isoformat()}；交易日 {len(daily)} 天。",
        f"- Warmup: {WARMUP_START.isoformat()} 起重放 X，以保留 20 日信号历史与既有持仓状态。",
        "- 结论仅作方向假设，不作为任何参数或信号变更依据。",
        "- 存档按附件 README 规范建目录；未修改 `strategy_changelog.md`。",
        "",
        "## 分歧汇总",
        "",
        summary.to_markdown(index=False),
        "",
        f"- 选择差异型合计覆盖 {selection_days} 个交易日。",
        "",
        pair_summary.to_markdown(index=False) if len(pair_summary) else "无选择差异型资产对。",
        "",
        "## Y 换仓节奏指纹",
        "",
        f"- Y 切换间隔完整列表: {intervals}",
        f"- 5 的整数倍: {multiples} 次；非 5 倍数: {non_multiples} 次；最短间隔: {min_interval} 个交易日。",
        f"- 证据读数: {cadence_read}",
        "",
        y_fp.to_markdown(index=False),
        "",
        "## 分歧区间表",
        "",
        diff_show.to_markdown(index=False),
        "",
        "## 逐日对照表",
        "",
        daily_show.to_markdown(index=False),
        "",
        "## 存档",
        "",
        f"- 逐日对照 CSV: `{DAILY_CSV_PATH.name}`",
        f"- 分歧区间 CSV: `{DIFF_CSV_PATH.name}`",
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    config = _load_config()
    trace = _run_traced(deepcopy(config), "close")
    calendar = _trading_calendar(config)
    x = _x_daily_holdings(trace, calendar)
    y = _y_daily_holdings(calendar)
    daily = pd.DataFrame(
        {
            "date": calendar,
            "X持仓": x.values,
            "Y持仓": y.values,
        }
    )
    daily["是否分歧"] = daily["X持仓"] != daily["Y持仓"]

    switches = _x_switches(trace)
    diff_df = _diff_intervals(daily, switches, config)
    y_fp = _y_switch_fingerprint(calendar)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    daily_out = _format_daily_for_report(daily)
    daily_out.to_csv(DAILY_CSV_PATH, index=False, encoding="utf-8-sig")
    diff_df.to_csv(DIFF_CSV_PATH, index=False, encoding="utf-8-sig")
    _write_report(daily, diff_df, y_fp)
    print(f"wrote {REPORT_PATH}")
    print(f"wrote {DAILY_CSV_PATH}")
    print(f"wrote {DIFF_CSV_PATH}")


if __name__ == "__main__":
    main()
