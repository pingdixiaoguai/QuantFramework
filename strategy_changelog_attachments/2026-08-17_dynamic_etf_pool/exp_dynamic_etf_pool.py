"""Dynamic ETF universe diagnostic (research only).

Run from repository root:
    uv run python strategy_changelog_attachments/2026-08-17_dynamic_etf_pool/exp_dynamic_etf_pool.py

The script reads HFQ prices from data/db and point-in-time fund shares from
data/db/dynamic_etf_fund_share.parquet. It does not modify production config.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from data import store  # noqa: E402

OUT = Path(__file__).resolve().parent
PREFIX = "2026-08-17_dynamic_etf_pool"
SHARE_CACHE = store.DB_DIR / "dynamic_etf_fund_share.parquet"

CORE = ["510300.SH", "159915.SZ", "513100.SH", "518880.SH"]
CORE_NAMES = {
    "510300.SH": "沪深300ETF",
    "159915.SZ": "创业板ETF",
    "513100.SH": "纳指ETF",
    "518880.SH": "黄金ETF",
}
SATELLITES: dict[str, tuple[str, str]] = {
    "510210.SH": ("上证综指ETF", "broad_cn"),
    "510500.SH": ("中证500ETF", "broad_cn"),
    "512100.SH": ("中证1000ETF", "broad_cn"),
    "588000.SH": ("科创50ETF", "broad_cn"),
    "563360.SH": ("中证A500ETF", "broad_cn"),
    "513500.SH": ("标普500ETF", "broad_overseas"),
    "513180.SH": ("恒生科技ETF", "style_overseas"),
    "513050.SH": ("中概互联网ETF", "style_overseas"),
    "512880.SH": ("证券ETF", "industry_financial"),
    "512690.SH": ("酒ETF", "industry_consumer"),
    "513120.SH": ("港股创新药ETF", "industry_healthcare"),
    "515880.SH": ("通信ETF", "industry_tmt"),
    "588200.SH": ("科创芯片ETF", "industry_tmt"),
    "159819.SZ": ("人工智能ETF", "industry_tmt"),
    "562500.SH": ("机器人ETF", "industry_manufacturing"),
    "159326.SZ": ("电网设备ETF", "industry_manufacturing"),
    "512400.SH": ("有色金属ETF", "industry_cyclical"),
    "515220.SH": ("煤炭ETF", "industry_cyclical"),
    "159870.SZ": ("化工ETF", "industry_cyclical"),
    "159611.SZ": ("电力ETF", "industry_utility"),
    "515790.SH": ("光伏ETF", "industry_energy"),
    "512660.SH": ("军工ETF", "industry_defense"),
}
ALL = CORE + list(SATELLITES)

WARMUP_START = pd.Timestamp("2013-01-01")
EVAL_START = pd.Timestamp("2014-01-02")
TRAIN_END = pd.Timestamp("2022-12-30")
REBALANCE_DAYS = 5
SIZE_FLOOR_YI = 50.0
SIZE_PERSISTENCE_DAYS = 20
MIN_LISTING_DAYS = 252
FEES = (0.0001, 0.0005)


@dataclass(frozen=True)
class Arm:
    name: str
    use_size: bool
    trend: str | None = None
    slots: int | None = None
    static_current: bool = False


ARMS = [
    Arm("B0_core4", use_size=False),
    Arm("S0_static_current", use_size=False, static_current=True),
    Arm("D0_size_only", use_size=True),
    Arm("D1_positive_top5", use_size=True, trend="positive", slots=5),
    Arm("D2_median_core_top3", use_size=True, trend="median", slots=3),
    Arm("D3_best_core_top2", use_size=True, trend="best", slots=2),
]


def _native_series(code: str, column: str) -> pd.Series:
    df = store.read_local(code)
    if df is None or df.empty:
        raise RuntimeError(f"missing local price data: {code}")
    df = df.sort_values("date")
    return pd.Series(df[column].to_numpy(dtype=float), index=pd.DatetimeIndex(df["date"]), name=code)


def _raw_close(code: str) -> pd.Series:
    df = store.read_storage(code)
    if df is None or df.empty or "raw_close" not in df:
        raise RuntimeError(f"missing raw-close storage data: {code}")
    df = df.sort_values("date")
    return pd.Series(df["raw_close"].to_numpy(dtype=float), index=pd.DatetimeIndex(df["date"]), name=code)


def load_panels() -> dict[str, pd.DataFrame]:
    calendar = _native_series(CORE[0], "close").loc[WARMUP_START:].index
    closes = pd.DataFrame(index=calendar)
    opens = pd.DataFrame(index=calendar)
    scores = pd.DataFrame(index=calendar)
    mom60 = pd.DataFrame(index=calendar)
    above_ma120 = pd.DataFrame(index=calendar)
    age = pd.DataFrame(index=calendar)
    raw_closes: dict[str, pd.Series] = {}

    for code in ALL:
        close = _native_series(code, "close")
        open_ = _native_series(code, "open")
        momentum20 = close.pct_change(20)
        displacement = (close - close.shift(20)).abs()
        path = close.diff().abs().rolling(20).sum()
        quality = momentum20 * displacement / path.replace(0.0, np.nan)
        closes[code] = close.reindex(calendar)
        opens[code] = open_.reindex(calendar)
        scores[code] = quality.reindex(calendar)
        mom60[code] = close.pct_change(60).reindex(calendar)
        above_ma120[code] = (close > close.rolling(120).mean()).reindex(calendar)
        age[code] = pd.Series(np.arange(1, len(close) + 1), index=close.index).reindex(calendar).ffill()
        raw_closes[code] = _raw_close(code)

    if not SHARE_CACHE.exists():
        raise RuntimeError(f"missing {SHARE_CACHE}; refresh point-in-time fund_share data first")
    shares = pd.read_parquet(SHARE_CACHE)
    shares["trade_date"] = pd.to_datetime(shares["trade_date"])
    size = pd.DataFrame(index=calendar, columns=ALL, dtype=float)
    for code in ALL:
        one = shares.loc[shares["ts_code"] == code, ["trade_date", "fd_share"]].drop_duplicates("trade_date")
        share = pd.Series(one["fd_share"].to_numpy(dtype=float), index=pd.DatetimeIndex(one["trade_date"]))
        observed = (share * raw_closes[code].reindex(share.index) / 10000.0).dropna()
        size[code] = observed.reindex(calendar).ffill()

    return {
        "close": closes,
        "open": opens,
        "score": scores,
        "mom60": mom60,
        "above_ma120": above_ma120.astype("boolean"),
        "age": age,
        "size": size,
    }


def current_snapshot(panels: dict[str, pd.DataFrame]) -> pd.DataFrame:
    size = panels["size"]
    age = panels["age"]
    rows = []
    for code in ALL:
        valid = size[code].dropna()
        asof = valid.index[-1]
        name, sleeve = ((CORE_NAMES[code], "core") if code in CORE else SATELLITES[code])
        rows.append(
            {
                "asset": code,
                "name": name,
                "sleeve": sleeve,
                "core": code in CORE,
                "size_asof": asof.date().isoformat(),
                "estimated_size_yi": float(valid.iloc[-1]),
                "listing_observations": int(age[code].dropna().iloc[-1]),
                "passes_current_50yi": bool(valid.iloc[-1] >= SIZE_FLOOR_YI),
            }
        )
    return pd.DataFrame(rows)


def eligibility_for(arm: Arm, p: dict[str, pd.DataFrame]) -> pd.DataFrame:
    idx = p["close"].index
    eligible = pd.DataFrame(False, index=idx, columns=ALL)
    eligible.loc[:, CORE] = True
    if arm.name == "B0_core4":
        return eligible

    sat = list(SATELLITES)
    base = p["age"][sat].ge(MIN_LISTING_DAYS)
    if arm.use_size:
        # A t-close signal only sees size observations through t-1.
        lagged_size = p["size"][sat].shift(1)
        persistent = lagged_size.rolling(SIZE_PERSISTENCE_DAYS, min_periods=SIZE_PERSISTENCE_DAYS).min()
        base &= persistent.ge(SIZE_FLOOR_YI)
    elif arm.static_current:
        current_ok = p["size"][sat].ffill().iloc[-1].ge(SIZE_FLOOR_YI)
        base &= pd.DataFrame(
            np.broadcast_to(current_ok.to_numpy(), base.shape),
            index=base.index,
            columns=base.columns,
        )

    if arm.trend is not None:
        base &= p["mom60"][sat].gt(0.0)
        base &= p["above_ma120"][sat].fillna(False).astype(bool)
        if arm.trend == "median":
            reference = p["mom60"][CORE].median(axis=1)
            base &= p["mom60"][sat].gt(reference, axis=0)
        elif arm.trend == "best":
            reference = p["mom60"][CORE].max(axis=1)
            base &= p["mom60"][sat].gt(reference, axis=0)
        elif arm.trend != "positive":
            raise ValueError(f"unknown trend rule: {arm.trend}")

    if arm.slots is not None:
        rank = p["mom60"][sat].where(base).rank(axis=1, ascending=False, method="first")
        base &= rank.le(arm.slots)
    eligible.loc[:, sat] = base
    return eligible


def signal_targets(scores: pd.DataFrame, eligible: pd.DataFrame) -> pd.Series:
    masked = scores.where(eligible)
    has_score = masked.notna().any(axis=1)
    target = masked.fillna(-np.inf).idxmax(axis=1)
    target[~has_score] = None
    return target


def _safe_ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator == 0:
        return 1.0
    return float(numerator / denominator)


def simulate(targets: pd.Series, opens: pd.DataFrame, closes: pd.DataFrame) -> dict[str, pd.Series]:
    dates = closes.index
    gross = pd.Series(0.0, index=dates, dtype=float)
    turnover = pd.Series(0.0, index=dates, dtype=float)
    held = pd.Series(index=dates, dtype="object")
    current: str | None = None
    entry_idx: int | None = None
    pending: str | None = None
    pending_idx: int | None = None

    for i, t in enumerate(dates):
        if i > 0:
            prev = dates[i - 1]
            old = current
            if pending_idx == i and pending is not None:
                overnight = 0.0 if old is None else _safe_ratio(opens.at[t, old], closes.at[prev, old]) - 1.0
                current = pending
                entry_idx = i
                turnover.at[t] = 1.0 if old is None else (0.0 if old == current else 2.0)
                intraday = _safe_ratio(closes.at[t, current], opens.at[t, current]) - 1.0
                gross.at[t] = (1.0 + overnight) * (1.0 + intraday) - 1.0
                pending = None
                pending_idx = None
            elif current is not None:
                gross.at[t] = _safe_ratio(closes.at[t, current], closes.at[prev, current]) - 1.0
        held.at[t] = current

        holding_days = i - entry_idx + 1 if current is not None and entry_idx is not None else None
        should_signal = pending is None and (current is None or holding_days is None or holding_days >= REBALANCE_DAYS)
        if should_signal:
            new = targets.at[t]
            if isinstance(new, str) and new != current and i + 1 < len(dates):
                next_t = dates[i + 1]
                if np.isfinite(opens.at[next_t, new]) and np.isfinite(closes.at[next_t, new]):
                    pending = new
                    pending_idx = i + 1

    return {"gross": gross.loc[EVAL_START:], "turnover": turnover.loc[EVAL_START:], "held": held.loc[EVAL_START:]}


def sharpe(r: pd.Series) -> float:
    sd = float(r.std())
    return float(r.mean() / sd * math.sqrt(252.0)) if sd > 0 else 0.0


def annual_return(r: pd.Series) -> float:
    if r.empty:
        return 0.0
    return float((1.0 + r).prod() ** (252.0 / len(r)) - 1.0)


def max_drawdown(r: pd.Series) -> float:
    wealth = (1.0 + r).cumprod()
    return float((wealth / wealth.cummax() - 1.0).min())


def metrics(name: str, fee: float, sim: dict[str, pd.Series]) -> dict[str, object]:
    r = sim["gross"] - sim["turnover"] * fee
    ins = r.loc[:TRAIN_END]
    oos = r.loc[TRAIN_END + pd.Timedelta(days=1):]
    years = len(r) / 252.0
    return {
        "arm": name,
        "fee_bps_one_side": fee * 10000,
        "start": r.index.min().date().isoformat(),
        "end": r.index.max().date().isoformat(),
        "days": len(r),
        "annual_return": annual_return(r),
        "sharpe": sharpe(r),
        "max_drawdown": max_drawdown(r),
        "is_annual_return": annual_return(ins),
        "is_sharpe": sharpe(ins),
        "oos_annual_return": annual_return(oos),
        "oos_sharpe": sharpe(oos),
        "annual_turnover_sum_abs": float(sim["turnover"].sum() / years),
        "switches": int((sim["turnover"] >= 2.0).sum()),
    }


def rolling36(arm_returns: dict[str, pd.Series], baseline: pd.Series) -> pd.DataFrame:
    rows = []
    window, step = 756, 21
    for name, r in arm_returns.items():
        if name == "B0_core4":
            continue
        joined = pd.concat([baseline.rename("baseline"), r.rename("candidate")], axis=1).dropna()
        for end in range(window, len(joined) + 1, step):
            chunk = joined.iloc[end - window:end]
            rows.append(
                {
                    "arm": name,
                    "window_end": chunk.index[-1].date().isoformat(),
                    "baseline_sharpe": sharpe(chunk["baseline"]),
                    "candidate_sharpe": sharpe(chunk["candidate"]),
                    "candidate_leads": sharpe(chunk["candidate"]) > sharpe(chunk["baseline"]),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    p = load_panels()
    snapshot = current_snapshot(p)
    if not snapshot["passes_current_50yi"].all():
        failed = snapshot.loc[~snapshot["passes_current_50yi"], ["asset", "estimated_size_yi"]]
        raise RuntimeError(f"pre-registered candidate failed current 50yi screen:\n{failed}")

    sims: dict[str, dict[str, pd.Series]] = {}
    eligibility_rows = []
    for arm in ARMS:
        eligible = eligibility_for(arm, p)
        targets = signal_targets(p["score"], eligible)
        sims[arm.name] = simulate(targets, p["open"], p["close"])
        sat_count = eligible[list(SATELLITES)].sum(axis=1).loc[EVAL_START:]
        eligibility_rows.append(
            {
                "arm": arm.name,
                "mean_eligible_satellites": float(sat_count.mean()),
                "median_eligible_satellites": float(sat_count.median()),
                "max_eligible_satellites": int(sat_count.max()),
                "days_with_any_satellite": int(sat_count.gt(0).sum()),
                "share_days_with_any_satellite": float(sat_count.gt(0).mean()),
            }
        )

    metric_rows = [metrics(name, fee, sim) for name, sim in sims.items() for fee in FEES]
    metric_df = pd.DataFrame(metric_rows)
    one_bp = {
        name: sim["gross"] - sim["turnover"] * FEES[0]
        for name, sim in sims.items()
    }
    rolling = rolling36(one_bp, one_bp["B0_core4"])

    yearly = pd.DataFrame({name: (1.0 + r).groupby(r.index.year).prod() - 1.0 for name, r in one_bp.items()})
    yearly.index.name = "year"
    holdings = []
    for name, sim in sims.items():
        counts = sim["held"].value_counts(dropna=True)
        total = int(counts.sum())
        sat_total = int(counts[counts.index.isin(SATELLITES)].sum())
        for asset, days in counts.items():
            holdings.append(
                {
                    "arm": name,
                    "asset": asset,
                    "name": SATELLITES.get(asset, (CORE_NAMES.get(asset, asset), "core"))[0],
                    "sleeve": SATELLITES.get(asset, ("核心ETF", "core"))[1],
                    "holding_days": int(days),
                    "share_all_holding_days": float(days / total) if total else 0.0,
                    "share_satellite_holding_days": float(days / sat_total) if sat_total and asset in SATELLITES else 0.0,
                }
            )

    # Result-aware diagnostics are written separately and are not part of the
    # preregistered deployment gates. Additive log excess groups each arm's
    # opportunity cost versus B0 by the asset held in the expanded arm.
    attribution = []
    base_one_bp = one_bp["B0_core4"]
    for name, sim in sims.items():
        if name == "B0_core4":
            continue
        diff = np.log1p(one_bp[name]) - np.log1p(base_one_bp)
        frame = pd.DataFrame({"excess_log_return": diff, "held_asset": sim["held"]})
        frame["period"] = np.where(frame.index <= TRAIN_END, "IS", "OOS")
        frame["year"] = frame.index.year
        for (period, asset), group in frame.dropna(subset=["held_asset"]).groupby(["period", "held_asset"]):
            attribution.append(
                {
                    "arm": name,
                    "period": period,
                    "asset": asset,
                    "name": SATELLITES.get(asset, (CORE_NAMES.get(asset, asset), "core"))[0],
                    "holding_days": len(group),
                    "additive_log_excess_vs_B0": float(group["excess_log_return"].sum()),
                }
            )

    # Post-hoc mechanism check: retain an independent B0 core sleeve and give
    # the expanded arm only a capped satellite budget. These figures may seed
    # a later preregistration but must not be treated as OOS validation here.
    overlay_rows = []
    overlay_rolling_rows = []
    for name in ["D1_positive_top5", "D2_median_core_top3", "D3_best_core_top2"]:
        for satellite_weight in (0.25, 0.50, 0.75):
            gross = (1.0 - satellite_weight) * sims["B0_core4"]["gross"] + satellite_weight * sims[name]["gross"]
            turn = (1.0 - satellite_weight) * sims["B0_core4"]["turnover"] + satellite_weight * sims[name]["turnover"]
            for fee in FEES:
                blended = {"gross": gross, "turnover": turn, "held": sims[name]["held"]}
                row = metrics(f"overlay_{name}", fee, blended)
                row["satellite_weight"] = satellite_weight
                overlay_rows.append(row)
            candidate = gross - turn * FEES[0]
            roll = rolling36({"B0_core4": base_one_bp, f"overlay_{name}_{satellite_weight}": candidate}, base_one_bp)
            overlay_rolling_rows.append(
                {
                    "expanded_arm": name,
                    "satellite_weight": satellite_weight,
                    "rolling36_lead_share": float(roll["candidate_leads"].mean()),
                    "rolling36_windows": len(roll),
                }
            )

    dynamic_names = [a.name for a in ARMS if a.name.startswith("D") and a.name != "D0_size_only"]
    is_table = metric_df[(metric_df["fee_bps_one_side"] == 1.0) & metric_df["arm"].isin(dynamic_names)]
    selected = str(is_table.sort_values("is_sharpe", ascending=False).iloc[0]["arm"])
    base_metrics = metric_df[(metric_df.arm == "B0_core4") & (metric_df.fee_bps_one_side == 1.0)].iloc[0]
    selected_metrics = metric_df[(metric_df.arm == selected) & (metric_df.fee_bps_one_side == 1.0)].iloc[0]
    base_5 = metric_df[(metric_df.arm == "B0_core4") & (metric_df.fee_bps_one_side == 5.0)].iloc[0]
    selected_5 = metric_df[(metric_df.arm == selected) & (metric_df.fee_bps_one_side == 5.0)].iloc[0]
    selected_roll = rolling[rolling.arm == selected]
    lead_share = float(selected_roll.candidate_leads.mean())
    selected_hold = pd.DataFrame(holdings)
    selected_sat = selected_hold[(selected_hold.arm == selected) & (selected_hold.sleeve != "core")]
    max_sat_concentration = float(selected_sat.share_satellite_holding_days.max()) if not selected_sat.empty else 0.0
    gates = pd.DataFrame(
        [
            {"gate": "OOS Sharpe delta >= 0.10", "value": selected_metrics.oos_sharpe - base_metrics.oos_sharpe, "passed": selected_metrics.oos_sharpe - base_metrics.oos_sharpe >= 0.10},
            {"gate": "5bp OOS Sharpe direction positive", "value": selected_5.oos_sharpe - base_5.oos_sharpe, "passed": selected_5.oos_sharpe > base_5.oos_sharpe},
            {"gate": "maxDD deterioration <= 3pp", "value": selected_metrics.max_drawdown - base_metrics.max_drawdown, "passed": selected_metrics.max_drawdown - base_metrics.max_drawdown >= -0.03},
            {"gate": "rolling36 lead share >= 60%", "value": lead_share, "passed": lead_share >= 0.60},
            {"gate": "single satellite concentration <= 60%", "value": max_sat_concentration, "passed": max_sat_concentration <= 0.60},
        ]
    )
    gates.insert(0, "selected_by_is", selected)

    snapshot.to_csv(OUT / f"{PREFIX}_candidate_snapshot.csv", index=False)
    metric_df.to_csv(OUT / f"{PREFIX}_metrics.csv", index=False)
    pd.DataFrame(eligibility_rows).to_csv(OUT / f"{PREFIX}_eligibility.csv", index=False)
    rolling.to_csv(OUT / f"{PREFIX}_rolling36m.csv", index=False)
    yearly.to_csv(OUT / f"{PREFIX}_yearly.csv")
    pd.DataFrame(holdings).to_csv(OUT / f"{PREFIX}_holdings_share.csv", index=False)
    gates.to_csv(OUT / f"{PREFIX}_gates.csv", index=False)
    pd.DataFrame(attribution).to_csv(OUT / f"{PREFIX}_attribution.csv", index=False)
    pd.DataFrame(overlay_rows).to_csv(OUT / f"{PREFIX}_posthoc_overlay_metrics.csv", index=False)
    pd.DataFrame(overlay_rolling_rows).to_csv(OUT / f"{PREFIX}_posthoc_overlay_rolling36m.csv", index=False)

    print(metric_df.to_string(index=False))
    print(f"\nSelected by IS only: {selected}")
    print(gates.to_string(index=False))


if __name__ == "__main__":
    main()
