"""Read-only diagnostic for 20-day path shape vs. forward open returns.

This script intentionally avoids importing runner/strategy/backtest/configs.
It reads local qfq Parquet data through data.store.query(), reuses the
quality_momentum factor implementation to recover the existing ER series, and
writes diagnostic tables plus a short markdown note.
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

from data.store import query
from factors.quality_momentum import compute as compute_quality_momentum


ASSETS = ["510300.SH", "159915.SZ", "513100.SH", "518880.SH"]
START = date(2014, 1, 1)
END = date(2026, 5, 19)
WINDOW = 20
FWD_START_OFFSET = 1
FWD_END_OFFSET = 6

OUTPUT_DIR = REPO_ROOT / "strategy_changelog_attachments"
MARKDOWN_PATH = OUTPUT_DIR / "2026-05-20_shape_signal_diagnostic.md"

GROUP_LABELS = ["low", "mid", "high"]
METRIC_COLUMNS = ["mean", "median", "count"]


@dataclass(frozen=True)
class OutputBundle:
    frame: pd.DataFrame
    tables: dict[str, pd.DataFrame]
    alignment_samples: pd.DataFrame
    elapsed_seconds: float


def _fmt_date(value: object) -> str:
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _recover_er_from_quality_momentum(df: pd.DataFrame) -> pd.Series:
    """Reuse quality_momentum.compute() and divide out same-window momentum."""
    momentum = pd.Series(df["close"].pct_change(WINDOW).to_numpy(), index=df["date"])
    qmom = compute_quality_momentum(df, {"window": WINDOW}).reindex(df["date"])

    er = qmom / momentum.replace(0, np.nan)

    # In the quality_momentum implementation, zero same-window momentum implies
    # zero displacement; if the path length is non-zero, ER is exactly zero.
    er = er.mask(momentum.eq(0), 0.0)
    return er.clip(lower=0.0, upper=1.0)


def _shape_descriptors(close: pd.Series) -> tuple[pd.Series, pd.Series]:
    log_close = np.log(close.astype(float))
    returns = close.pct_change()

    x = np.arange(WINDOW + 1, dtype=float)
    x = x - x.mean()

    conv_values: list[float] = [np.nan] * len(close)
    com_values: list[float] = [np.nan] * len(close)

    for end_pos in range(WINDOW, len(close)):
        start_pos = end_pos - WINDOW
        window_log = log_close.iloc[start_pos : end_pos + 1]
        if window_log.isna().any():
            continue

        _, _, c = np.polyfit(x, window_log.to_numpy(), deg=2)
        conv_values[end_pos] = float(c)

        window_returns = returns.iloc[start_pos + 1 : end_pos + 1].abs()
        denom = window_returns.sum()
        if pd.notna(denom) and denom > 0:
            weights = np.arange(1, WINDOW + 1, dtype=float)
            center = float((weights * window_returns.to_numpy()).sum() / denom)
            com_values[end_pos] = (center - 1.0) / (WINDOW - 1.0)

    index = close.index
    return pd.Series(conv_values, index=index), pd.Series(com_values, index=index)


def _asset_frame(asset: str) -> pd.DataFrame:
    raw = query(asset, date(1900, 1, 1), END)
    if raw.empty:
        raise RuntimeError(f"No local data found for {asset}")

    raw = raw.sort_values("date").reset_index(drop=True)
    er = _recover_er_from_quality_momentum(raw)
    conv, com = _shape_descriptors(raw["close"])

    out = raw[["date", "open", "close"]].copy()
    out["asset"] = asset
    out["row_in_asset"] = np.arange(len(out))
    out["window_start"] = out["date"].shift(WINDOW)
    out["window_end"] = out["date"]
    out["t_plus_1"] = out["date"].shift(-FWD_START_OFFSET)
    out["t_plus_6"] = out["date"].shift(-FWD_END_OFFSET)
    out["er"] = er.to_numpy()
    out["mom"] = raw["close"].pct_change(WINDOW).to_numpy()
    out["conv"] = conv.to_numpy()
    out["com"] = com.to_numpy()
    out["fwd"] = (
        raw["open"].shift(-FWD_END_OFFSET) / raw["open"].shift(-FWD_START_OFFSET)
        - 1.0
    ).to_numpy()

    mask = (out["date"] >= pd.Timestamp(START)) & (out["date"] <= pd.Timestamp(END))
    required = ["er", "mom", "conv", "com", "fwd", "window_start", "t_plus_1", "t_plus_6"]
    return out.loc[mask].dropna(subset=required).reset_index(drop=True)


def _asset_rank_standardize(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for column in ["er", "mom", "conv", "com"]:
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
        subset=["er_group", "mom_group", "conv_group", "com_group"]
    ).reset_index(drop=True)


def _sample_every_five(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["eligible_order"] = out.groupby("asset").cumcount()
    return out.loc[out["eligible_order"].mod(5).eq(0)].drop(
        columns=["eligible_order"]
    )


def _complete_index(columns: list[str]) -> pd.MultiIndex:
    levels = [GROUP_LABELS for _ in columns]
    return pd.MultiIndex.from_product(levels, names=columns)


def _sort_table(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    grouped = (
        frame.groupby(group_columns, observed=False)["fwd"]
        .agg(mean="mean", median="median", count="count")
        .reindex(_complete_index(group_columns))
        .reset_index()
    )
    grouped["count"] = grouped["count"].fillna(0).astype(int)
    return grouped


def _make_tables(frame: pd.DataFrame, sample_name: str) -> dict[str, pd.DataFrame]:
    specs = {
        f"{sample_name}_er_conv": ["er_group", "conv_group"],
        f"{sample_name}_er_com": ["er_group", "com_group"],
        f"{sample_name}_er_mom_conv": ["er_group", "mom_group", "conv_group"],
        f"{sample_name}_er_mom_com": ["er_group", "mom_group", "com_group"],
    }
    return {name: _sort_table(frame, columns) for name, columns in specs.items()}


def _write_tables(tables: dict[str, pd.DataFrame]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(OUTPUT_DIR / f"2026-05-20_shape_signal_diagnostic_{name}.csv", index=False)


def _alignment_samples(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.Series] = []
    positions = [0, 50, -1]
    for asset, pos in zip(ASSETS[:3], positions, strict=True):
        part = frame.loc[frame["asset"].eq(asset)].sort_values("date")
        if part.empty:
            continue
        rows.append(part.iloc[pos])
    sample = pd.DataFrame(rows)
    columns = ["asset", "window_start", "window_end", "t_plus_1", "t_plus_6", "fwd"]
    return sample.loc[:, columns]


def _spread_note(table: pd.DataFrame, shape_group_column: str) -> str:
    pieces: list[str] = []
    for er_group in GROUP_LABELS:
        part = table.loc[table["er_group"].eq(er_group)].copy()
        part = part.loc[part["count"].gt(0)]
        if part.empty:
            pieces.append(f"ER={er_group}: no populated cells")
            continue
        low = part.loc[part[shape_group_column].eq("low")]
        high = part.loc[part[shape_group_column].eq("high")]
        if low.empty or high.empty:
            pieces.append(f"ER={er_group}: missing low/high cells")
            continue
        mean_spread = high["mean"].iloc[0] - low["mean"].iloc[0]
        median_spread = high["median"].iloc[0] - low["median"].iloc[0]
        count_min = int(min(high["count"].iloc[0], low["count"].iloc[0]))
        pieces.append(
            f"ER={er_group}: high-low mean {mean_spread:.4%}, "
            f"median {median_spread:.4%}, min count {count_min}"
        )
    return "; ".join(pieces)


def _count_range(table: pd.DataFrame) -> str:
    populated = table.loc[table["count"].gt(0), "count"]
    if populated.empty:
        return "0 populated cells"
    return f"{int(populated.min())}-{int(populated.max())}"


def _controlled_spread_note(table: pd.DataFrame, shape_group_column: str) -> str:
    rows: list[dict[str, object]] = []
    for er_group in GROUP_LABELS:
        for mom_group in GROUP_LABELS:
            part = table.loc[
                table["er_group"].eq(er_group) & table["mom_group"].eq(mom_group)
            ]
            low = part.loc[part[shape_group_column].eq("low")]
            high = part.loc[part[shape_group_column].eq("high")]
            if low.empty or high.empty:
                continue
            rows.append(
                {
                    "er_group": er_group,
                    "mom_group": mom_group,
                    "mean_spread": high["mean"].iloc[0] - low["mean"].iloc[0],
                    "median_spread": high["median"].iloc[0] - low["median"].iloc[0],
                    "min_count": int(min(high["count"].iloc[0], low["count"].iloc[0])),
                }
            )
    if not rows:
        return "no populated low/high comparisons"

    spread_frame = pd.DataFrame(rows)
    spread_frame["abs_mean_spread"] = spread_frame["mean_spread"].abs()
    top = spread_frame.sort_values("abs_mean_spread", ascending=False).head(3)
    return "; ".join(
        f"ER={row.er_group}, mom={row.mom_group}: high-low mean "
        f"{row.mean_spread:.4%}, median {row.median_spread:.4%}, "
        f"min count {int(row.min_count)}"
        for row in top.itertuples(index=False)
    )


def _markdown_table(table: pd.DataFrame, max_rows: int | None = None) -> str:
    view = table.copy()
    if max_rows is not None:
        view = view.head(max_rows)
    for column in ["mean", "median"]:
        if column in view.columns:
            view[column] = view[column].map(
                lambda value: "" if pd.isna(value) else f"{value:.4%}"
            )
    view = view.fillna("")
    headers = [str(column) for column in view.columns]
    rows = [[str(value) for value in row] for row in view.to_numpy()]
    widths = [
        max(len(headers[pos]), *(len(row[pos]) for row in rows)) if rows else len(headers[pos])
        for pos in range(len(headers))
    ]

    def render_row(values: list[str]) -> str:
        cells = [values[pos].ljust(widths[pos]) for pos in range(len(values))]
        return "| " + " | ".join(cells) + " |"

    separator = "| " + " | ".join("-" * width for width in widths) + " |"
    return "\n".join([render_row(headers), separator, *(render_row(row) for row in rows)])


def _write_markdown(bundle: OutputBundle) -> None:
    full = bundle.frame.loc[bundle.frame["sample"] == "full"]
    sampled = bundle.frame.loc[bundle.frame["sample"] == "every_5"]
    tables = bundle.tables

    lines = [
        "# 20-day shape signal diagnostic",
        "",
        f"- Run date: 2026-05-20",
        f"- Assets: {', '.join(ASSETS)}",
        f"- Signal dates: {START.isoformat()} to {END.isoformat()}",
        "- Prices: local qfq-adjusted open/close from `data.store.query()`.",
        "- ER: recovered from `factors.quality_momentum.compute(window=20)` divided by the same 20-day close momentum.",
        "- Forward return: `open[t+6] / open[t+1] - 1`.",
        "- Standardization: percentile rank within each asset for ER, momentum, conv, and com before tercile grouping.",
        f"- Runtime: {bundle.elapsed_seconds:.2f}s",
        "",
        "## Alignment samples",
        "",
        _markdown_table(_format_alignment_samples(bundle.alignment_samples)),
        "",
        "## Sample counts",
        "",
        f"- Full overlapping sample rows: {len(full)}",
        f"- Every-5-trading-day sample rows: {len(sampled)}",
        f"- Full ER x conv cell count range: {_count_range(tables['full_er_conv'])}",
        f"- Full ER x com cell count range: {_count_range(tables['full_er_com'])}",
        f"- Full ER x mom x conv cell count range: {_count_range(tables['full_er_mom_conv'])}",
        f"- Full ER x mom x com cell count range: {_count_range(tables['full_er_mom_com'])}",
        f"- Every-5 ER x conv cell count range: {_count_range(tables['every_5_er_conv'])}",
        f"- Every-5 ER x com cell count range: {_count_range(tables['every_5_er_com'])}",
        f"- Every-5 ER x mom x conv cell count range: {_count_range(tables['every_5_er_mom_conv'])}",
        f"- Every-5 ER x mom x com cell count range: {_count_range(tables['every_5_er_mom_com'])}",
        "",
        "## Objective observations",
        "",
        f"- Full ER x conv high-minus-low spreads: {_spread_note(tables['full_er_conv'], 'conv_group')}.",
        f"- Full ER x com high-minus-low spreads: {_spread_note(tables['full_er_com'], 'com_group')}.",
        f"- Every-5 ER x conv high-minus-low spreads: {_spread_note(tables['every_5_er_conv'], 'conv_group')}.",
        f"- Every-5 ER x com high-minus-low spreads: {_spread_note(tables['every_5_er_com'], 'com_group')}.",
        f"- Full ER x mom x conv largest absolute high-minus-low mean spreads: {_controlled_spread_note(tables['full_er_mom_conv'], 'conv_group')}.",
        f"- Full ER x mom x com largest absolute high-minus-low mean spreads: {_controlled_spread_note(tables['full_er_mom_com'], 'com_group')}.",
        f"- Every-5 ER x mom x conv largest absolute high-minus-low mean spreads: {_controlled_spread_note(tables['every_5_er_mom_conv'], 'conv_group')}.",
        f"- Every-5 ER x mom x com largest absolute high-minus-low mean spreads: {_controlled_spread_note(tables['every_5_er_mom_com'], 'com_group')}.",
        "",
        "## Full sample: ER then conv",
        "",
        _markdown_table(tables["full_er_conv"]),
        "",
        "## Full sample: ER then com",
        "",
        _markdown_table(tables["full_er_com"]),
        "",
        "## Every-5 sample: ER then conv",
        "",
        _markdown_table(tables["every_5_er_conv"]),
        "",
        "## Every-5 sample: ER then com",
        "",
        _markdown_table(tables["every_5_er_com"]),
        "",
        "## ER + momentum + shape tables",
        "",
        "CSV outputs:",
    ]

    for name in sorted(tables):
        path = OUTPUT_DIR / f"2026-05-20_shape_signal_diagnostic_{name}.csv"
        lines.append(f"- `{path.name}`")

    MARKDOWN_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _format_alignment_samples(samples: pd.DataFrame) -> pd.DataFrame:
    out = samples.copy()
    for column in ["window_start", "window_end", "t_plus_1", "t_plus_6"]:
        out[column] = out[column].map(_fmt_date)
    out["fwd"] = out["fwd"].map(lambda value: f"{value:.4%}")
    return out


def run() -> OutputBundle:
    started = perf_counter()
    asset_frames = [_asset_frame(asset) for asset in ASSETS]
    full = _asset_rank_standardize(pd.concat(asset_frames, ignore_index=True))
    full["sample"] = "full"

    every_5 = _sample_every_five(full.drop(columns=["sample"]))
    every_5["sample"] = "every_5"

    combined = pd.concat([full, every_5], ignore_index=True)
    tables = _make_tables(full, "full") | _make_tables(every_5, "every_5")
    elapsed = perf_counter() - started
    samples = _alignment_samples(full)

    bundle = OutputBundle(
        frame=combined,
        tables=tables,
        alignment_samples=samples,
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
    print("Alignment samples:")
    samples = bundle.alignment_samples.copy()
    for column in ["window_start", "window_end", "t_plus_1", "t_plus_6"]:
        samples[column] = samples[column].map(_fmt_date)
    print(samples.to_string(index=False))


if __name__ == "__main__":
    main()
