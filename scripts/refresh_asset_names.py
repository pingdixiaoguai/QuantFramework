"""Refresh ASSET_NAMES suggestions from Tushare fund_basic.

Usage:
    uv run python scripts/refresh_asset_names.py

Reads every YAML under strategy/configs/, gathers the union of `asset_pool`
codes, queries `pro.fund_basic(market="E")` to fetch official ETF metadata,
and prints suggested `ASSET_NAMES` entries with shortened Chinese names.

Human review and paste into notification/formatter.py is required — this
script does not modify any source file.
"""

from __future__ import annotations

import re
from pathlib import Path

import tushare as ts
import yaml

from data.config import get_tushare_token

_REPO_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_DIR = _REPO_ROOT / "strategy" / "configs"

_FUND_PREFIXES = (
    "易方达", "华夏", "南方", "嘉实", "广发", "博时", "鹏华", "工银瑞信",
    "国泰", "招商", "建信", "汇添富", "富国", "华泰柏瑞", "天弘", "银华",
    "中欧", "万家", "国投瑞银", "民生加银", "永赢", "兴业", "海富通",
    "平安", "前海开源", "信诚", "上投摩根", "诺安", "申万菱信",
    "中海", "中融", "华安",
)

_INDEX_PREFIXES = (
    "中证", "国证", "上证", "深证", "沪深", "申万",
)


def _shorten(raw_name: str) -> str:
    """Best-effort short Chinese name for an ETF.

    Strips fund-company prefix, index family prefix, and trailing 'ETF'/'交易型...'.
    Falls back to the input if nothing matches.
    """
    name = raw_name
    for pref in _FUND_PREFIXES:
        if name.startswith(pref):
            name = name[len(pref):]
            break
    for pref in _INDEX_PREFIXES:
        if name.startswith(pref):
            name = name[len(pref):]
            break
    # Strip trailing ETF / 交易型开放式指数证券投资基金 / 联接 etc.
    name = re.sub(r"(交易型.*?基金|ETF.*$|联接.*$)", "", name).strip()
    if not name:
        name = raw_name
    return name


def _gather_asset_pool() -> set[str]:
    pool: set[str] = set()
    for cfg_path in sorted(_CONFIG_DIR.glob("*.yaml")):
        with open(cfg_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        for code in cfg.get("asset_pool", []):
            pool.add(code)
    return pool


def main() -> None:
    pool = _gather_asset_pool()
    if not pool:
        print(f"No asset_pool codes found under {_CONFIG_DIR}")
        return

    print(f"Found {len(pool)} unique codes across strategy configs.")
    print("Querying Tushare fund_basic(market='E') ...")

    pro = ts.pro_api(get_tushare_token())
    df = pro.fund_basic(market="E")
    if df is None or df.empty:
        print("fund_basic returned no rows.")
        return

    by_code = {row["ts_code"]: row for _, row in df.iterrows()}

    print()
    print("# Paste into notification/formatter.py ASSET_NAMES dict")
    print("# Verify the short names manually before commit.")
    print()
    missing: list[str] = []
    for code in sorted(pool):
        row = by_code.get(code)
        if row is None:
            missing.append(code)
            continue
        raw = str(row.get("name", ""))
        short = _shorten(raw)
        print(f'    "{code}": "{short}",  # {raw}')

    if missing:
        print()
        print("# WARNING: codes not found in fund_basic — verify or drop:")
        for code in missing:
            print(f"#   {code}")


if __name__ == "__main__":
    main()
