"""Build the point-in-time monthly ETF universe from cached market snapshots."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT_CACHE = ROOT / "data/db/historical_etf_month_end_size.parquet"
BASIC_CACHE = ROOT / "data/db/historical_etf_basic.parquet"
OUT = Path(__file__).resolve().parent
PREFIX = "2026-08-17_dynamic_etf_pool_phase3"

CORE_CODES = {"510300.SH", "159915.SZ", "513100.SH", "518880.SH"}
CORE_INDEX_NAMES = {"沪深300", "创业板指", "纳斯达克100指数", "黄金9999"}
PHASE1_SATELLITES = {
    "510210.SH", "510500.SH", "512100.SH", "588000.SH", "563360.SH",
    "513500.SH", "513180.SH", "513050.SH", "512880.SH", "512690.SH",
    "513120.SH", "515880.SH", "588200.SH", "159819.SZ", "562500.SH",
    "159326.SZ", "512400.SH", "515220.SH", "159870.SZ", "159611.SZ",
    "515790.SH", "512660.SH",
}

FORBIDDEN = re.compile(
    r"货币|现金添益|收益宝|保证金|短融|(?:国|地方政府|政策性金融|公司|信用|城投|科创)债|"
    r"债券|转债|同业存单|红利|股息|低波|自由现金流|现金流|黄金|商品|原油|豆粕|期货|"
    r"上海金|REIT|基础设施",
    re.IGNORECASE,
)


def build_membership() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not SNAPSHOT_CACHE.exists() or not BASIC_CACHE.exists():
        raise RuntimeError("run fetch_historical_etf_universe_audit.py first")
    snapshots = pd.read_parquet(SNAPSHOT_CACHE)
    basic = pd.read_parquet(BASIC_CACHE)
    snapshots["month_end"] = pd.to_datetime(snapshots["month_end"])
    basic["list_date_parsed"] = pd.to_datetime(basic["list_date"], format="%Y%m%d", errors="coerce")
    frame = snapshots.loc[snapshots["estimated_size_yi"] >= 50.0].merge(
        basic, on="ts_code", how="left", validate="many_to_one"
    )
    frame["classification_text"] = frame[["extname", "index_name", "cname"]].fillna("").agg("|".join, axis=1)
    frame["excluded_category"] = frame["classification_text"].str.contains(FORBIDDEN)
    frame["excluded_core_duplicate"] = frame["ts_code"].isin(CORE_CODES) | frame["index_name"].isin(CORE_INDEX_NAMES)
    frame["old_enough"] = (frame["month_end"] - frame["list_date_parsed"]).dt.days >= 365
    eligible = frame.loc[
        ~frame["excluded_category"] & ~frame["excluded_core_duplicate"] & frame["old_enough"]
    ].copy()
    eligible["exposure_key"] = eligible["index_name"].fillna(eligible["extname"])
    # Avoid duplicate ETFs tracking the same index: keep the largest at each month end.
    membership = (
        eligible.sort_values(
            ["month_end", "exposure_key", "estimated_size_yi", "amount_yi"],
            ascending=[True, True, False, False],
        )
        .drop_duplicates(["month_end", "exposure_key"], keep="first")
        .sort_values(["month_end", "estimated_size_yi"], ascending=[True, False])
    )
    membership["covered_by_phase1_pool"] = membership["ts_code"].isin(PHASE1_SATELLITES)

    union = (
        membership.groupby(["ts_code", "extname", "index_name", "exposure_key"], dropna=False)
        .agg(
            first_month=("month_end", "min"),
            last_month=("month_end", "max"),
            eligible_months=("month_end", "size"),
            max_estimated_size_yi=("estimated_size_yi", "max"),
            max_amount_yi=("amount_yi", "max"),
            list_date=("list_date_parsed", "first"),
            list_status=("list_status", "first"),
        )
        .reset_index()
        .sort_values(["eligible_months", "max_estimated_size_yi"], ascending=False)
    )
    coverage = membership.groupby("month_end").agg(
        eligible_exposures=("exposure_key", "size"),
        phase1_code_matches=("covered_by_phase1_pool", "sum"),
    )
    coverage["phase1_code_coverage"] = coverage["phase1_code_matches"] / coverage["eligible_exposures"]
    return membership, union, coverage.reset_index()


def main() -> None:
    membership, union, coverage = build_membership()
    membership[[
        "month_end", "ts_code", "extname", "index_name", "exposure_key",
        "estimated_size_yi", "amount_yi", "covered_by_phase1_pool",
    ]].to_csv(OUT / f"{PREFIX}_monthly_membership.csv", index=False)
    union.to_csv(OUT / f"{PREFIX}_union.csv", index=False)
    coverage.to_csv(OUT / f"{PREFIX}_coverage.csv", index=False)
    print(
        f"month ends={membership.month_end.nunique()} rows={len(membership)} "
        f"union_codes={membership.ts_code.nunique()} exposures={membership.exposure_key.nunique()} "
        f"phase1_row_coverage={membership.covered_by_phase1_pool.mean():.2%}"
    )


if __name__ == "__main__":
    main()
