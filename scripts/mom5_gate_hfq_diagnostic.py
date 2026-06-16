"""Read-only mom5 x COM gate diagnostic on local HFQ prices.

This script intentionally avoids importing runner/strategy/backtest/configs.
It reads local Parquet data through data.store.query()/read_storage(), computes
only point-in-time features, and writes four diagnostic CSVs plus one markdown.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.store import query, read_storage
from factors.quality_momentum import compute as compute_quality_momentum


ASSETS = ["510300.SH", "159915.SZ", "513100.SH", "518880.SH"]
START = date(2014, 1, 1)
END = date(2026, 6, 4)
WINDOW = 20
MOM5_WINDOW = 5
FWD_START_OFFSET = 1
FWD_END_OFFSET = 6

OUTPUT_PREFIX = "2026-06-16_mom5_gate_hfq"
OUTPUT_DIR = REPO_ROOT / "strategy_changelog_attachments" / OUTPUT_PREFIX
MARKDOWN_PATH = OUTPUT_DIR / f"{OUTPUT_PREFIX}.md"

GROUP_LABELS = ["low", "mid", "high"]
METRIC_COLUMNS = ["mean", "median", "count"]


@dataclass(frozen=True)
class OutputBundle:
    frame: pd.DataFrame
    tables: dict[str, pd.DataFrame]
    hfq_checks: pd.DataFrame
    factor_checks: pd.DataFrame
    alignment_checks: pd.DataFrame
    elapsed_seconds: float


def _fmt_date(value: object) -> str:
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _fmt_pct(value: object) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.4%}"


def _fmt_float(value: object, digits: int = 8) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.{digits}f}"


def _kaufman_er(close: pd.Series) -> pd.Series:
    displacement = (close - close.shift(WINDOW)).abs()
    path_length = close.diff().abs().rolling(WINDOW).sum()
    return displacement / path_length.replace(0, np.nan)


def _com(close: pd.Series) -> pd.Series:
    returns = close.pct_change()
    values: list[float] = [np.nan] * len(close)
    positions = np.arange(WINDOW, dtype=float)

    for end_pos in range(WINDOW, len(close)):
        start_pos = end_pos - WINDOW
        window_abs_returns = returns.iloc[start_pos + 1 : end_pos + 1].abs()
        denom = window_abs_returns.sum()
        if pd.isna(denom) or denom <= 0:
            continue
        values[end_pos] = float(
            (positions * window_abs_returns.to_numpy()).sum()
            / ((WINDOW - 1) * denom)
        )

    return pd.Series(values, index=close.index)


def _assert_hfq_projection(asset: str, projected: pd.DataFrame) -> pd.DataFrame:
    stored = read_storage(asset)
    if stored is None or stored.empty:
        raise RuntimeError(f"No local storage found for {asset}")

    required = {"date", "raw_open", "raw_close", "adj_factor"}
    if not required.issubset(stored.columns):
        raise RuntimeError(
            f"{asset} local Parquet is not raw+adj_factor HFQ storage; "
            f"columns={list(stored.columns)}"
        )

    stored = stored.sort_values("date").reset_index(drop=True)
    baseline = float(stored["adj_factor"].iloc[0])
    if baseline == 0:
        raise RuntimeError(f"{asset} first adj_factor is zero")

    sample_positions = sorted({0, len(stored) // 2, len(stored) - 1})
    rows: list[dict[str, object]] = []
    projected_by_date = projected.set_index("date")
    for pos in sample_positions:
        stored_row = stored.iloc[pos]
        sample_date = pd.Timestamp(stored_row["date"])
        if sample_date not in projected_by_date.index:
            continue
        query_row = projected_by_date.loc[sample_date]
        expected_open = (
            float(stored_row["raw_open"]) * float(stored_row["adj_factor"]) / baseline
        )
        expected_close = (
            float(stored_row["raw_close"]) * float(stored_row["adj_factor"]) / baseline
        )
        rows.append(
            {
                "asset": asset,
                "date": sample_date,
                "raw_open": float(stored_row["raw_open"]),
                "raw_close": float(stored_row["raw_close"]),
                "adj_factor": float(stored_row["adj_factor"]),
                "baseline_adj_factor": baseline,
                "query_open": float(query_row["open"]),
                "expected_hfq_open": expected_open,
                "open_abs_diff": abs(float(query_row["open"]) - expected_open),
                "query_close": float(query_row["close"]),
                "expected_hfq_close": expected_close,
                "close_abs_diff": abs(float(query_row["close"]) - expected_close),
            }
        )

    checks = pd.DataFrame(rows)
    if checks.empty:
        raise RuntimeError(f"No HFQ projection samples could be checked for {asset}")
    max_diff = checks[["open_abs_diff", "close_abs_diff"]].max().max()
    if max_diff > 1e-10:
        raise RuntimeError(f"{asset} HFQ projection check failed: max_diff={max_diff}")
    return checks


def _asset_frame(asset: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw = query(asset, date(1900, 1, 1), END)
    if raw.empty:
        raise RuntimeError(f"No local data found for {asset}")

    raw = raw.sort_values("date").reset_index(drop=True)
    hfq_checks = _assert_hfq_projection(asset, raw)

    er = _kaufman_er(raw["close"])
    mom20 = raw["close"].pct_change(WINDOW)
    mom5 = raw["close"].pct_change(MOM5_WINDOW)
    com = _com(raw["close"])
    quality_momentum = compute_quality_momentum(raw, {"window": WINDOW}).reindex(
        raw["date"]
    )
    qmom_from_parts = pd.Series((mom20 * er).to_numpy(), index=raw["date"])

    out = raw[["date", "open", "close"]].copy()
    out["asset"] = asset
    out["row_in_asset"] = np.arange(len(out))
    out["window_start"] = out["date"].shift(WINDOW)
    out["window_end"] = out["date"]
    out["t_plus_1"] = out["date"].shift(-FWD_START_OFFSET)
    out["t_plus_6"] = out["date"].shift(-FWD_END_OFFSET)
    out["open_t_plus_1"] = raw["open"].shift(-FWD_START_OFFSET)
    out["open_t_plus_6"] = raw["open"].shift(-FWD_END_OFFSET)
    out["er"] = er.to_numpy()
    out["mom20"] = mom20.to_numpy()
    out["mom5"] = mom5.to_numpy()
    out["com"] = com.to_numpy()
    out["quality_momentum"] = quality_momentum.to_numpy()
    out["quality_momentum_from_parts"] = qmom_from_parts.reindex(raw["date"]).to_numpy()
    out["qmom_abs_diff"] = (
        out["quality_momentum"] - out["quality_momentum_from_parts"]
    ).abs()
    out["fwd"] = out["open_t_plus_6"] / out["open_t_plus_1"] - 1.0

    mask = (out["date"] >= pd.Timestamp(START)) & (out["date"] <= pd.Timestamp(END))
    required = [
        "er",
        "mom20",
        "mom5",
        "com",
        "fwd",
        "window_start",
        "t_plus_1",
        "t_plus_6",
        "open_t_plus_1",
        "open_t_plus_6",
    ]
    frame = out.loc[mask].dropna(subset=required).reset_index(drop=True)

    valid_factor_rows = frame.loc[frame["qmom_abs_diff"].notna()].sort_values(
        ["asset", "date"]
    )
    if valid_factor_rows.empty:
        raise RuntimeError(f"{asset} has no valid quality_momentum parity rows")
    sample_positions = sorted({0, min(50, len(valid_factor_rows) - 1), len(valid_factor_rows) - 1})
    factor_sample = valid_factor_rows.iloc[sample_positions].loc[
        :,
        [
            "asset",
            "date",
            "mom20",
            "er",
            "quality_momentum_from_parts",
            "quality_momentum",
            "qmom_abs_diff",
        ],
    ]
    max_factor_diff = frame["qmom_abs_diff"].max()
    if pd.notna(max_factor_diff) and max_factor_diff > 1e-12:
        raise RuntimeError(
            f"{asset} quality_momentum parity failed: max_diff={max_factor_diff}"
        )

    return frame, hfq_checks, factor_sample


def _asset_rank_standardize(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for column in ["er", "mom20", "mom5", "com"]:
        out[f"{column}_rank"] = out.groupby("asset")[column].rank(
            method="average", pct=True
        )
        out[f"{column}_group"] = pd.cut(
            out[f"{column}_rank"],
            bins=[0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0],
            labels=GROUP_LABELS,
            include_lowest=True,
        )
    return out.dropna(
        subset=["er_group", "mom20_group", "mom5_group", "com_group"]
    ).reset_index(drop=True)


def _sample_every_five(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["eligible_order"] = out.groupby("asset").cumcount()
    return out.loc[out["eligible_order"].mod(5).eq(0)].drop(
        columns=["eligible_order"]
    )


def _complete_index(columns: list[str]) -> pd.MultiIndex:
    if len(columns) == 1:
        return pd.Index(GROUP_LABELS, name=columns[0])
    return pd.MultiIndex.from_product(
        [GROUP_LABELS for _ in columns], names=columns
    )


def _sort_table(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    grouped = (
        frame.groupby(group_columns, observed=False)["fwd"]
        .agg(mean="mean", median="median", count="count")
        .reindex(_complete_index(group_columns))
        .reset_index()
    )
    grouped["count"] = grouped["count"].fillna(0).astype(int)
    return grouped


def _make_tables(full: pd.DataFrame, every_5: pd.DataFrame) -> dict[str, pd.DataFrame]:
    samples = {"full": full, "every_5": every_5}
    tables: dict[str, pd.DataFrame] = {}
    for sample_name, frame in samples.items():
        high_er = frame.loc[frame["er_group"].eq("high")]
        tables[f"a_{sample_name}"] = _sort_table(high_er, ["com_group"])
        tables[f"b_{sample_name}"] = _sort_table(high_er, ["mom5_group", "com_group"])
    return tables


def _write_tables(tables: dict[str, pd.DataFrame]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(OUTPUT_DIR / f"{OUTPUT_PREFIX}_{name}.csv", index=False)


def _alignment_checks(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.Series] = []
    sample_specs = [
        ("510300.SH", 0),
        ("159915.SZ", 50),
        ("513100.SH", -1),
    ]
    for asset, pos in sample_specs:
        part = frame.loc[frame["asset"].eq(asset)].sort_values("date")
        if part.empty:
            continue
        rows.append(part.iloc[pos])
    sample = pd.DataFrame(rows)
    return sample.loc[
        :,
        [
            "asset",
            "window_start",
            "window_end",
            "t_plus_1",
            "t_plus_6",
            "open_t_plus_1",
            "open_t_plus_6",
            "fwd",
            "mom20",
            "er",
            "quality_momentum_from_parts",
            "quality_momentum",
        ],
    ]


def _spread_note(table: pd.DataFrame) -> str:
    low = table.loc[table["com_group"].eq("low")]
    high = table.loc[table["com_group"].eq("high")]
    if low.empty or high.empty:
        return "missing low/high cells"
    mean_spread = high["mean"].iloc[0] - low["mean"].iloc[0]
    median_spread = high["median"].iloc[0] - low["median"].iloc[0]
    min_count = int(min(high["count"].iloc[0], low["count"].iloc[0]))
    return (
        f"high-low mean {mean_spread:.4%}, median {median_spread:.4%}, "
        f"min count {min_count}"
    )


def _gate_spread_frame(table: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for mom5_group in GROUP_LABELS:
        part = table.loc[table["mom5_group"].eq(mom5_group)]
        low = part.loc[part["com_group"].eq("low")]
        high = part.loc[part["com_group"].eq("high")]
        if low.empty or high.empty:
            continue
        rows.append(
            {
                "mom5_group": mom5_group,
                "com_high_minus_low_mean": high["mean"].iloc[0] - low["mean"].iloc[0],
                "com_high_minus_low_median": high["median"].iloc[0]
                - low["median"].iloc[0],
                "min_count": int(min(high["count"].iloc[0], low["count"].iloc[0])),
            }
        )
    return pd.DataFrame(rows)


def _count_range(table: pd.DataFrame) -> str:
    populated = table.loc[table["count"].gt(0), "count"]
    if populated.empty:
        return "0 populated cells"
    return f"{int(populated.min())}-{int(populated.max())}"


def _cell_extremes(table: pd.DataFrame) -> str:
    populated = table.loc[table["count"].gt(0)].copy()
    if populated.empty:
        return "no populated cells"
    best = populated.loc[populated["mean"].idxmax()]
    worst = populated.loc[populated["mean"].idxmin()]
    return (
        f"highest mean: mom5={best['mom5_group']}, com={best['com_group']}, "
        f"mean {best['mean']:.4%}, median {best['median']:.4%}, count {int(best['count'])}; "
        f"lowest mean: mom5={worst['mom5_group']}, com={worst['com_group']}, "
        f"mean {worst['mean']:.4%}, median {worst['median']:.4%}, count {int(worst['count'])}"
    )


def _markdown_table(table: pd.DataFrame) -> str:
    view = table.copy()
    for column in ["date", "window_start", "window_end", "t_plus_1", "t_plus_6"]:
        if column in view.columns:
            view[column] = view[column].map(_fmt_date)
    for column in [
        "mean",
        "median",
        "fwd",
        "mom20",
        "er",
        "quality_momentum_from_parts",
        "quality_momentum",
        "qmom_abs_diff",
        "open_abs_diff",
        "close_abs_diff",
        "com_high_minus_low_mean",
        "com_high_minus_low_median",
    ]:
        if column in view.columns:
            if column in {
                "mean",
                "median",
                "fwd",
                "mom20",
                "er",
                "com_high_minus_low_mean",
                "com_high_minus_low_median",
            }:
                view[column] = view[column].map(_fmt_pct)
            else:
                view[column] = view[column].map(lambda value: _fmt_float(value, 12))
    for column in [
        "raw_open",
        "raw_close",
        "adj_factor",
        "baseline_adj_factor",
        "query_open",
        "expected_hfq_open",
        "query_close",
        "expected_hfq_close",
        "open_t_plus_1",
        "open_t_plus_6",
    ]:
        if column in view.columns:
            view[column] = view[column].map(lambda value: _fmt_float(value, 6))
    view = view.fillna("")

    headers = [str(column) for column in view.columns]
    rows = [[str(value) for value in row] for row in view.to_numpy()]
    widths = [
        max(len(headers[pos]), *(len(row[pos]) for row in rows))
        if rows
        else len(headers[pos])
        for pos in range(len(headers))
    ]

    def render_row(values: list[str]) -> str:
        cells = [values[pos].ljust(widths[pos]) for pos in range(len(values))]
        return "| " + " | ".join(cells) + " |"

    separator = "| " + " | ".join("-" * width for width in widths) + " |"
    return "\n".join([render_row(headers), separator, *(render_row(row) for row in rows)])


def _format_factor_checks(checks: pd.DataFrame) -> pd.DataFrame:
    return checks.loc[
        :,
        [
            "asset",
            "date",
            "mom20",
            "er",
            "quality_momentum_from_parts",
            "quality_momentum",
            "qmom_abs_diff",
        ],
    ]


def _write_markdown(bundle: OutputBundle) -> None:
    full = bundle.frame.loc[bundle.frame["sample"].eq("full")]
    every_5 = bundle.frame.loc[bundle.frame["sample"].eq("every_5")]
    tables = bundle.tables
    a_full = tables["a_full"]
    a_every_5 = tables["a_every_5"]
    b_full = tables["b_full"]
    b_every_5 = tables["b_every_5"]
    b_full_spreads = _gate_spread_frame(b_full)
    b_every_5_spreads = _gate_spread_frame(b_every_5)

    lines = [
        "# Mom5 x COM gate HFQ diagnostic",
        "",
        f"- Run date: 2026-06-16",
        f"- Assets: {', '.join(ASSETS)}",
        f"- Signal dates: {START.isoformat()} to {END.isoformat()}; rows needing `open[t+6]` are naturally truncated.",
        "- Scope: read-only local Parquet diagnostic; no strategy, engine, YAML, or production entry point code is imported or modified.",
        "- Prices: HFQ open/close from `data.store.query()`. This was verified against `read_storage()` raw prices as `raw_price * adj_factor / first_adj_factor`; this is not qfq.",
        "- Features: ER, mom20, mom5, and COM use only `[t-20, t]` or shorter lookback data; forward return is `open[t+6] / open[t+1] - 1`.",
        "- Standardization: ER, mom20, mom5, and COM are percentile-ranked within each asset before tercile grouping.",
        "- Grouping: all reported tables first gate to the high ER tercile, then sort by COM or by mom5 x COM.",
        "- Caveat: overlapping daily windows have strong autocorrelation; the every-5-trading-day sample is included as a second view. Cells with single-digit counts are not interpreted.",
        f"- Runtime: {bundle.elapsed_seconds:.2f}s",
        "",
        "## HFQ口径确认",
        "",
        _markdown_table(bundle.hfq_checks.head(12)),
        "",
        "## 因子口径与对齐手验",
        "",
        "- `mom20 * ER` matched `factors.quality_momentum.compute(window=20)` on the eligible panel within floating-point tolerance.",
        "- The alignment rows show feature window end `t`, forward start `t+1`, and forward end `t+6`, so the signal window and forward return window do not overlap.",
        "",
        _markdown_table(_format_factor_checks(bundle.factor_checks)),
        "",
        _markdown_table(bundle.alignment_checks),
        "",
        "## 样本量",
        "",
        f"- Full overlapping high-ER rows: {int(full['er_group'].eq('high').sum())}; total eligible rows before ER gate: {len(full)}.",
        f"- Every-5 high-ER rows: {int(every_5['er_group'].eq('high').sum())}; total eligible rows before ER gate: {len(every_5)}.",
        f"- (a) full COM cell count range: {_count_range(a_full)}.",
        f"- (a) every-5 COM cell count range: {_count_range(a_every_5)}.",
        f"- (b) full mom5 x COM cell count range: {_count_range(b_full)}.",
        f"- (b) every-5 mom5 x COM cell count range: {_count_range(b_every_5)}.",
        "",
        "## (a) COM复现检查",
        "",
        f"- Full sample COM spread: {_spread_note(a_full)}.",
        f"- Every-5 sample COM spread: {_spread_note(a_every_5)}.",
        "",
        "Full sample:",
        "",
        _markdown_table(a_full),
        "",
        "Every-5 sample:",
        "",
        _markdown_table(a_every_5),
        "",
        "## (b) mom5 x COM gate",
        "",
        f"- Full sample cell extremes: {_cell_extremes(b_full)}.",
        f"- Every-5 sample cell extremes: {_cell_extremes(b_every_5)}.",
        "- COM high-minus-low spreads by mom5 tercile:",
        "",
        "Full sample:",
        "",
        _markdown_table(b_full_spreads),
        "",
        "Every-5 sample:",
        "",
        _markdown_table(b_every_5_spreads),
        "",
        "Full 3x3 table:",
        "",
        _markdown_table(b_full),
        "",
        "Every-5 3x3 table:",
        "",
        _markdown_table(b_every_5),
        "",
        "## CSV outputs",
        "",
    ]

    for name in ["a_full", "a_every_5", "b_full", "b_every_5"]:
        path = OUTPUT_DIR / f"{OUTPUT_PREFIX}_{name}.csv"
        lines.append(f"- `{path.name}`")

    lines.extend(
        [
            "",
            "## 判读结论",
            "",
            "- (a) COM 在干净 HFQ 口径上复现，但强度比重叠全样本显示得更弱。全样本 COM low/mid/high 的 mean 为 0.3917%/0.5929%/0.7792%，median 为 0.3020%/0.4356%/0.6679%，high-low spread 为 +0.3875%/+0.3660%。Every-5 非重叠样本仍为正，但 mean 0.5671%/0.7349%/0.7590% 中到高基本走平，median 0.4679%/0.4209%/0.6803% 非单调，high-low spread 缩到 +0.1919%/+0.2124%。",
            "- (b) 控住 mom5 后，COM 的增量信号在全样本和 every-5 两版之间不一致。全样本三个 mom5 桶内 COM high-low mean spread 为 +0.1579%/+0.3807%/+0.3513%，但 every-5 为 +0.3898%/+0.0360%/-0.3165%；其中 mom5 高桶从全样本 +0.3513% 翻为 every-5 -0.3165%。按事先规则，重叠窗口自相关严重，两版冲突时信非重叠版。",
            "- 判定：控住 mom5 后，COM 没有稳定的增量前向收益排序；DTW/路径形状这一族关闭。此前 conv 已在稳健抽样下失效，COM 是最后保留的形状候选，本次 mom5 gate 没有给更复杂形状工具留下触发条件。",
            "- 不转向短窗口动量项。表格没有把 mom5 本身立成稳定前向预测变量；高 ER 组内，无论用 COM 还是 mom5 切 20 日路径的内部时序结构，都没有拿出抽样稳定的排序。Every-5 中局部活跃的单格不作为新方向依据。",
            "- 治理归位：本次为 Mode C 只读诊断，无部署、无策略/引擎/YAML 修改；不进入 `strategy_changelog.md` 正文，不触发对现因子或历史认知条目的修正。",
        ]
    )

    MARKDOWN_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run() -> OutputBundle:
    started = perf_counter()
    asset_results = [_asset_frame(asset) for asset in ASSETS]
    asset_frames = [result[0] for result in asset_results]
    hfq_checks = pd.concat([result[1] for result in asset_results], ignore_index=True)
    factor_checks = pd.concat([result[2] for result in asset_results], ignore_index=True)

    full = _asset_rank_standardize(pd.concat(asset_frames, ignore_index=True))
    full["sample"] = "full"

    every_5 = _sample_every_five(full.drop(columns=["sample"]))
    every_5["sample"] = "every_5"

    combined = pd.concat([full, every_5], ignore_index=True)
    tables = _make_tables(full, every_5)
    elapsed = perf_counter() - started
    alignment = _alignment_checks(full)

    bundle = OutputBundle(
        frame=combined,
        tables=tables,
        hfq_checks=hfq_checks,
        factor_checks=factor_checks,
        alignment_checks=alignment,
        elapsed_seconds=elapsed,
    )
    _write_tables(tables)
    _write_markdown(bundle)
    return bundle


def main() -> None:
    bundle = run()
    print(f"Wrote markdown: {MARKDOWN_PATH}")
    print(f"Wrote {len(bundle.tables)} CSV tables under: {OUTPUT_DIR}")
    print(f"Elapsed seconds: {bundle.elapsed_seconds:.2f}")
    print("HFQ projection samples max abs diff:")
    print(
        bundle.hfq_checks[["open_abs_diff", "close_abs_diff"]]
        .max()
        .to_string()
    )
    print("Alignment samples:")
    samples = bundle.alignment_checks.copy()
    for column in ["window_start", "window_end", "t_plus_1", "t_plus_6"]:
        samples[column] = samples[column].map(_fmt_date)
    print(samples.to_string(index=False))


if __name__ == "__main__":
    main()
