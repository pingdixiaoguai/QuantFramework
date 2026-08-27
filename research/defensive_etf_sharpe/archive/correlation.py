"""Daily HFQ-close-return correlation study for the defensive ETF pool.

The study deliberately uses only market-traded ETFs whose local HFQ closes
represent an investable total-return price series.  Correlations are Pearson
statistics calculated pairwise over each two ETFs' actual overlapping days.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

from data.store import query


@dataclass(frozen=True)
class Asset:
    code: str
    name: str
    sleeve: str


PROPOSED_POOL = (
    Asset("510880.SH", "华泰柏瑞上证红利ETF", "红利"),
    Asset("512890.SH", "华泰柏瑞中证红利低波动ETF", "红利低波"),
    Asset("515450.SH", "南方标普中国A股大盘红利低波50ETF", "红利低波"),
    Asset("511010.SH", "国泰上证5年期国债ETF", "5年国债"),
    Asset("511260.SH", "国泰上证10年期国债ETF", "10年国债"),
    Asset("511090.SH", "鹏扬中债-30年期国债ETF", "30年国债"),
    Asset("511360.SH", "海富通中证短融ETF", "短融信用债"),
    Asset("511880.SH", "银华日利ETF", "货币现金"),
)


def _local_hfq_close(code: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    frame = query(code, start.date(), end.date())
    if frame.empty:
        raise RuntimeError(f"no local HFQ price data for {code}")
    series = pd.Series(
        frame["close"].astype(float).to_numpy(),
        index=pd.DatetimeIndex(frame["date"]),
        name=code,
    )
    return series[~series.index.duplicated(keep="last")].sort_index().dropna()


def daily_hfq_returns(start: pd.Timestamp, end: pd.Timestamp) -> dict[str, pd.Series]:
    return {
        asset.code: _local_hfq_close(asset.code, start, end)
        .pct_change()
        .dropna()
        .rename(asset.code)
        for asset in PROPOSED_POOL
    }


def pairwise_correlation(
    returns: dict[str, pd.Series],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    codes = list(returns)
    correlation = pd.DataFrame(index=codes, columns=codes, dtype=float)
    observations = pd.DataFrame(index=codes, columns=codes, dtype=int)
    windows = []
    for left in codes:
        for right in codes:
            paired = pd.concat([returns[left], returns[right]], axis=1, join="inner").dropna()
            observations.loc[left, right] = len(paired)
            correlation.loc[left, right] = (
                1.0 if left == right else paired.iloc[:, 0].corr(paired.iloc[:, 1])
            )
            windows.append(
                {
                    "left": left,
                    "right": right,
                    "observations": len(paired),
                    "first_date": paired.index.min().date().isoformat() if not paired.empty else None,
                    "last_date": paired.index.max().date().isoformat() if not paired.empty else None,
                }
            )
    return correlation, observations, pd.DataFrame(windows)


def write_outputs(output_dir: Path, start: pd.Timestamp, end: pd.Timestamp) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    returns = daily_hfq_returns(start, end)
    correlation, observations, windows = pairwise_correlation(returns)

    correlation.to_csv(output_dir / "defensive_pool_daily_hfq_correlation.csv", float_format="%.6f")
    observations.to_csv(output_dir / "defensive_pool_daily_hfq_overlap.csv")
    windows.to_csv(output_dir / "defensive_pool_daily_hfq_windows.csv", index=False)

    metadata = {
        "as_of": end.date().isoformat(),
        "frequency": "daily",
        "price_basis": "local HFQ close",
        "return_definition": "close_t / close_t-1 - 1",
        "correlation_method": "Pearson; every matrix cell uses all actual overlapping daily returns for that pair",
        "assets": [asset.__dict__ for asset in PROPOSED_POOL],
        "price_coverage": {
            code: {
                "start": series.index.min().date().isoformat(),
                "end": series.index.max().date().isoformat(),
                "daily_return_observations": len(series),
            }
            for code, series in returns.items()
        },
    }
    (output_dir / "defensive_pool_daily_hfq_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {
        "daily_correlation": correlation.round(6).to_dict(),
        "daily_observations": observations.astype(int).to_dict(),
        "metadata": metadata,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2013-01-01")
    parser.add_argument("--end", default=date.today().isoformat())
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "outputs")
    args = parser.parse_args()
    result = write_outputs(args.output_dir, pd.Timestamp(args.start), pd.Timestamp(args.end))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
