"""Pool leg swap diagnostic: 510300 -> dividend-low-vol / free-cash-flow leg.

Research-only (Mode C). Preregistered design: 2026-07-14_pool_leg_swap_design.md
in this directory. Does not modify production configs.

Run from repo root:
    uv run python strategy_changelog_attachments/2026-07-14_pool_leg_swap_dividend_cashflow/exp_pool_leg_swap.py

Engine runs at zero cost; fee levels are applied arithmetically from gross
returns and executed turnover (same approach as scripts/cost_tau_scan.py).
Index series are written as pseudo-assets into data/db (gitignored,
adj_factor=1.0, filename = index code).
"""

from __future__ import annotations

import math
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from backtest.runner import BacktestResult, run  # noqa: E402
from data import store  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent
PREFIX = "2026-07-14_pool_leg_swap"

GROWTH_LEG = "159915.SZ"
BASE_POOL = ["510300.SH", "159915.SZ", "513100.SH", "518880.SH"]
INDEX_CODES = ["H20269.CSI", "H30269.CSI", "932365.CSI"]
FEES = [0.0, 0.0001, 0.0005]

BRANCHES: dict[str, dict] = {
    "E": {
        "start": date(2019, 1, 18),
        "arms": {
            "E-A0": BASE_POOL,
            "E-B1": ["512890.SH", "159915.SZ", "513100.SH", "518880.SH"],
        },
    },
    "P": {
        "start": date(2013, 12, 31),
        "arms": {
            "P-A0": BASE_POOL,
            "P-B1": ["H20269.CSI", "159915.SZ", "513100.SH", "518880.SH"],
            "P-B1p": ["H30269.CSI", "159915.SZ", "513100.SH", "518880.SH"],
            "P-B2": ["932365.CSI", "159915.SZ", "513100.SH", "518880.SH"],
        },
    },
    "O": {
        "start": date(2025, 2, 27),
        "arms": {
            "O-A0": BASE_POOL,
            "O-B1": ["512890.SH", "159915.SZ", "513100.SH", "518880.SH"],
            "O-B2a": ["159201.SZ", "159915.SZ", "513100.SH", "518880.SH"],
            "O-B2b": ["159399.SZ", "159915.SZ", "513100.SH", "518880.SH"],
        },
    },
}


# ---------------------------------------------------------------- index data


def fetch_and_store_indices() -> None:
    """Fetch index daily bars from Tushare and write pseudo-asset parquets."""
    import tushare as ts

    from data.config import get_tushare_token

    pro = ts.pro_api(get_tushare_token())
    for code in INDEX_CODES:
        df = pro.index_daily(ts_code=code, start_date="20130101", end_date="20991231")
        if df is None or df.empty:
            raise RuntimeError(f"index_daily returned no rows for {code}")
        out = pd.DataFrame(
            {
                "date": pd.to_datetime(df["trade_date"], format="%Y%m%d"),
                "raw_open": pd.to_numeric(df["open"], errors="coerce"),
                "raw_high": pd.to_numeric(df["high"], errors="coerce"),
                "raw_low": pd.to_numeric(df["low"], errors="coerce"),
                "raw_close": pd.to_numeric(df["close"], errors="coerce"),
                "volume": pd.to_numeric(df.get("vol"), errors="coerce").fillna(0.0),
                "adj_factor": 1.0,
            }
        ).sort_values("date").reset_index(drop=True)
        if out["raw_close"].isna().any():
            raise RuntimeError(f"{code}: NaN close in fetched index bars")
        store.DB_DIR.mkdir(parents=True, exist_ok=True)
        out.to_parquet(store.DB_DIR / f"{code}.parquet", index=False)
        print(f"stored {code}: {len(out)} rows {out['date'].iloc[0].date()} -> {out['date'].iloc[-1].date()}")


def common_end_date(codes: list[str]) -> date:
    """Last trading date available across all given series."""
    last_dates = []
    for code in codes:
        df = store.read_local(code)
        if df is None or df.empty:
            raise RuntimeError(f"no local data for {code}")
        last_dates.append(df["date"].max())
    return min(last_dates).date()


# ---------------------------------------------------------------- gate D


def daily_returns_of(code: str, start: date, end: date) -> pd.Series:
    df = store.query(code, start, end)
    ser = pd.Series(df["close"].values, index=df["date"])
    return ser.pct_change().dropna()

def gate_d(end: date) -> pd.DataFrame:
    rows = []
    # D1: 512890 HFQ vs H20269 total-return index, overlap correlation
    etf = daily_returns_of("512890.SH", date(2019, 1, 18), end)
    idx = daily_returns_of("H20269.CSI", date(2019, 1, 18), end)
    joined = pd.concat([etf.rename("etf"), idx.rename("index")], axis=1).dropna()
    corr = float(joined["etf"].corr(joined["index"]))
    rows.append(
        {
            "gate": "D1",
            "check": "512890 HFQ vs H20269 total-return daily-return corr",
            "value": round(corr, 6),
            "threshold": ">=0.97",
            "n_overlap_days": len(joined),
            "passed": corr >= 0.97,
        }
    )
    # D2: sanity on every new series
    for code in ["512890.SH", "159201.SZ", "159399.SZ", *INDEX_CODES]:
        df = store.read_local(code)
        ok = (
            df is not None
            and not df.empty
            and not df["close"].isna().any()
            and df["date"].is_monotonic_increasing
            and df["date"].is_unique
        )
        rows.append(
            {
                "gate": "D2",
                "check": f"{code} no-NaN close, strictly increasing unique dates",
                "value": len(df) if df is not None else 0,
                "threshold": "sanity",
                "n_overlap_days": "",
                "passed": bool(ok),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- engine runs


def run_arm(pool: list[str], start: date, end: date) -> BacktestResult:
    config = {
        "strategy_name": "pool_leg_swap_diagnostic",
        "strategy_class": "strategy.top1.Top1",
        "asset_pool": list(pool),
        "start": start,
        "end": end,
        "factors": [{"name": "quality_momentum", "weight": 1.0, "params": {"window": 20}}],
        "train_ratio": 0.7,
        "rebalance_days": 5,
        "transaction_cost_rate": 0.0,
    }
    return run(config)


def net_returns(result: BacktestResult, fee: float) -> pd.Series:
    gross = result.gross_daily_returns
    if gross is None or gross.empty or fee == 0:
        return (gross if gross is not None else pd.Series(dtype=float)).copy()
    turnover = result.turnover
    costs = turnover.reindex(gross.index, fill_value=0.0) * fee
    return gross - costs


# ---------------------------------------------------------------- metrics


def _sharpe(returns: pd.Series) -> float:
    std = returns.std()
    return float(returns.mean() / std * math.sqrt(252.0)) if std and std > 0 else 0.0


def _annual(returns: pd.Series) -> float:
    if returns.empty:
        return 0.0
    total = float((1.0 + returns).prod() - 1.0)
    return float((1.0 + total) ** (252.0 / len(returns)) - 1.0)


def _max_drawdown(returns: pd.Series) -> float:
    if returns.empty:
        return 0.0
    cumulative = (1.0 + returns).cumprod()
    return float((cumulative / cumulative.cummax() - 1.0).min())


def metrics_row(arm: str, fee: float, result: BacktestResult) -> dict[str, object]:
    returns = net_returns(result, fee)
    n_days = len(returns)
    train_end = pd.Timestamp(result.train_end)
    is_ret = returns[returns.index <= train_end]
    oos_ret = returns[returns.index > train_end]
    years = n_days / 252.0 if n_days else 0.0
    turnover_sum = float(result.turnover.sum()) if result.turnover is not None else 0.0
    n_positions = len(result.positions)
    return {
        "arm": arm,
        "fee_bps_one_side": fee * 10000,
        "start": returns.index.min().date().isoformat() if n_days else "",
        "end": returns.index.max().date().isoformat() if n_days else "",
        "trading_days": n_days,
        "total_return": float((1.0 + returns).prod() - 1.0) if n_days else 0.0,
        "annual_return": _annual(returns),
        "sharpe": _sharpe(returns),
        "max_drawdown": _max_drawdown(returns),
        "annual_turnover_sum_abs": turnover_sum / years if years else 0.0,
        "avg_holding_days": n_days / n_positions if n_positions else 0.0,
        "switch_count": max(n_positions - 1, 0),
        "train_end": result.train_end.isoformat(),
        "is_annual_return": _annual(is_ret),
        "is_sharpe": _sharpe(is_ret),
        "oos_annual_return": _annual(oos_ret),
        "oos_sharpe": _sharpe(oos_ret),
    }


def rolling_36m_table(a0: pd.Series, b: pd.Series, label_b: str) -> pd.DataFrame:
    """Rolling 756-trading-day Sharpe comparison, stepping 21 days."""
    joined = pd.concat([a0.rename("a0"), b.rename("b")], axis=1).dropna()
    window, step = 756, 21
    rows = []
    for end_idx in range(window, len(joined) + 1, step):
        chunk = joined.iloc[end_idx - window : end_idx]
        rows.append(
            {
                "window_end": chunk.index[-1].date().isoformat(),
                "sharpe_a0": _sharpe(chunk["a0"]),
                f"sharpe_{label_b}": _sharpe(chunk["b"]),
                "b_leads": _sharpe(chunk["b"]) > _sharpe(chunk["a0"]),
            }
        )
    return pd.DataFrame(rows)


def yearly_table(arm_returns: dict[str, pd.Series]) -> pd.DataFrame:
    frames = {}
    for arm, returns in arm_returns.items():
        yearly = (1.0 + returns).groupby(returns.index.year).prod() - 1.0
        frames[arm] = yearly
    return pd.DataFrame(frames).round(6)


def holdings_share(result: BacktestResult, calendar: pd.DatetimeIndex) -> dict[str, float]:
    positions = result.positions
    if positions.empty:
        return {}
    daily = positions.fillna(0.0).reindex(calendar).ffill().fillna(0.0)
    held = daily[daily.sum(axis=1) > 0]
    if held.empty:
        return {}
    top = held.idxmax(axis=1)
    return (top.value_counts() / len(held)).round(4).to_dict()


def episode_concentration(a0: pd.Series, b: pd.Series, b_result: BacktestResult) -> pd.DataFrame:
    """Max single-episode share of cumulative log excess (B vs A0).

    Episodes are B's holding segments delimited by its execution dates.
    """
    joined = pd.concat([a0.rename("a0"), b.rename("b")], axis=1).dropna()
    excess = np.log1p(joined["b"]) - np.log1p(joined["a0"])
    exec_dates = [d for d in b_result.turnover.index if d in excess.index]
    boundaries = sorted(set(exec_dates) | {excess.index[0]})
    segment_ids = pd.Series(0, index=excess.index)
    for i, boundary in enumerate(boundaries):
        segment_ids.loc[excess.index >= boundary] = i
    seg_sums = excess.groupby(segment_ids).sum()
    total = float(excess.sum())
    max_seg = float(seg_sums.max()) if len(seg_sums) else 0.0
    return pd.DataFrame(
        [
            {
                "total_log_excess": round(total, 6),
                "n_episodes": len(seg_sums),
                "max_episode_log_excess": round(max_seg, 6),
                "max_episode_share_of_total": round(max_seg / total, 4) if total > 0 else np.nan,
                "note": "share only meaningful when total_log_excess > 0",
            }
        ]
    )


def leg_correlations(end: date) -> pd.DataFrame:
    rows = []
    specs = [
        ("E", date(2019, 1, 18), ["510300.SH", "512890.SH"]),
        ("P", date(2013, 12, 31), ["510300.SH", "H20269.CSI", "H30269.CSI", "932365.CSI"]),
        ("O", date(2025, 2, 27), ["510300.SH", "512890.SH", "159201.SZ", "159399.SZ"]),
    ]
    for branch, start, legs in specs:
        growth = daily_returns_of(GROWTH_LEG, start, end)
        for leg in legs:
            leg_ret = daily_returns_of(leg, start, end)
            joined = pd.concat([growth.rename("g"), leg_ret.rename("l")], axis=1).dropna()
            rows.append(
                {
                    "branch": branch,
                    "leg": leg,
                    "vs": GROWTH_LEG,
                    "corr_daily_returns": round(float(joined["g"].corr(joined["l"])), 4),
                    "n_days": len(joined),
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- main


def main() -> None:
    fetch_and_store_indices()

    all_codes = sorted({code for branch in BRANCHES.values() for pool in branch["arms"].values() for code in pool})
    end = common_end_date(all_codes)
    print(f"common end date: {end}")

    gate_d_df = gate_d(end)
    gate_d_df.to_csv(OUT_DIR / f"{PREFIX}_data_gate.csv", index=False)
    print(gate_d_df.to_string(index=False))

    for branch_name, branch in BRANCHES.items():
        results: dict[str, BacktestResult] = {}
        for arm, pool in branch["arms"].items():
            results[arm] = run_arm(pool, branch["start"], end)

        rows = [metrics_row(arm, fee, result) for arm, result in results.items() for fee in FEES]
        metrics_df = pd.DataFrame(rows)
        metrics_df.to_csv(OUT_DIR / f"{PREFIX}_metrics_branch_{branch_name}.csv", index=False)

        net_1bp = {arm: net_returns(result, 0.0001) for arm, result in results.items()}
        yearly_table(net_1bp).to_csv(OUT_DIR / f"{PREFIX}_yearly_{branch_name}.csv")

        share_rows = []
        for arm, result in results.items():
            calendar = net_1bp[arm].index
            for asset, share in holdings_share(result, calendar).items():
                share_rows.append({"branch": branch_name, "arm": arm, "asset": asset, "held_share": share})
        pd.DataFrame(share_rows).to_csv(
            OUT_DIR / f"{PREFIX}_holdings_share_{branch_name}.csv", index=False
        )

        a0_key = f"{branch_name}-A0"
        if branch_name in ("E", "P"):
            for arm in branch["arms"]:
                if arm == a0_key:
                    continue
                table = rolling_36m_table(net_1bp[a0_key], net_1bp[arm], arm)
                table.to_csv(OUT_DIR / f"{PREFIX}_rolling36m_{arm}.csv", index=False)
        if branch_name == "E":
            episode_concentration(net_1bp[a0_key], net_1bp["E-B1"], results["E-B1"]).to_csv(
                OUT_DIR / f"{PREFIX}_episode_concentration_E-B1.csv", index=False
            )
        print(f"branch {branch_name} done: {list(branch['arms'])}")

    leg_correlations(end).to_csv(OUT_DIR / f"{PREFIX}_leg_correlation.csv", index=False)
    print("all outputs written to", OUT_DIR)


if __name__ == "__main__":
    main()
