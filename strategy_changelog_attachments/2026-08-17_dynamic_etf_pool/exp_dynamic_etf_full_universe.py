"""Phase-3 full historical ETF universe survivorship audit."""

from __future__ import annotations

import importlib.util
import math
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from data import store  # noqa: E402

HERE = Path(__file__).resolve().parent
PHASE2_PATH = HERE / "exp_dynamic_etf_pool_phase2.py"
MEMBERSHIP_PATH = HERE / "2026-08-17_dynamic_etf_pool_phase3_monthly_membership.csv"
UNION_PATH = HERE / "2026-08-17_dynamic_etf_pool_phase3_union.csv"
PREFIX = "2026-08-17_dynamic_etf_pool_phase3"

CORE = ["510300.SH", "159915.SZ", "513100.SH", "518880.SH"]
CORE_NAMES = {
    "510300.SH": "沪深300ETF",
    "159915.SZ": "创业板ETF",
    "513100.SH": "纳指ETF",
    "518880.SH": "黄金ETF",
}
EVAL_START = pd.Timestamp("2014-01-02")
END = pd.Timestamp("2026-08-14")


def load_phase2():
    spec = importlib.util.spec_from_file_location("dynamic_etf_phase2_for_phase3", PHASE2_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {PHASE2_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def native(code: str, column: str) -> pd.Series:
    df = store.read_local(code)
    if df is None or df.empty:
        raise RuntimeError(f"missing local data for {code}")
    df = df.sort_values("date")
    return pd.Series(df[column].to_numpy(dtype=float), index=pd.DatetimeIndex(df["date"]), name=code)


def classify_sleeve(text: str) -> str:
    rules = [
        ("financial", r"证券|银行|金融|非银|财富管理"),
        ("consumer", r"消费|酒|食品|饮料|家电|旅游|养殖|畜牧"),
        ("healthcare", r"医药|医疗|创新药|生物|器械"),
        ("tmt", r"半导体|芯片|通信|5G|人工智能|软件|互联网|科技|游戏|动漫|传媒|电子|卫星通信"),
        ("manufacturing", r"机器人|电网|设备|汽车|高端装备"),
        ("cyclical", r"有色|煤炭|化工|稀土|稀有金属|钢铁"),
        ("energy", r"光伏|新能源|电池|风电"),
        ("utility", r"电力|公用事业"),
        ("defense", r"军工|国防"),
        ("overseas_broad", r"恒生|香港|港股|标普|日经|德国|法国|纳斯达克|MSCI"),
        ("domestic_broad", r"上证|深证|中证|创业板|科创|A50|A500|国证|央企|国企|价值|成长"),
    ]
    for sleeve, pattern in rules:
        if re.search(pattern, text, re.IGNORECASE):
            return sleeve
    return "other"


def load_market() -> tuple[SimpleNamespace, dict[str, pd.DataFrame], dict, pd.DataFrame, pd.DataFrame]:
    membership = pd.read_csv(MEMBERSHIP_PATH, parse_dates=["month_end"])
    union = pd.read_csv(UNION_PATH)
    satellite_codes = union["ts_code"].tolist()
    all_codes = CORE + satellite_codes
    names = dict(zip(union["ts_code"], union["extname"], strict=True))
    exposure = dict(zip(union["ts_code"], union["exposure_key"], strict=True))
    sleeves = {
        code: classify_sleeve(f"{names.get(code, '')}|{exposure.get(code, '')}")
        for code in satellite_codes
    }
    model = SimpleNamespace(
        ALL=all_codes,
        CORE=CORE,
        EVAL_START=EVAL_START,
        SATELLITES={code: (names[code], sleeves[code]) for code in satellite_codes},
        CORE_NAMES=CORE_NAMES,
    )

    calendar = native(CORE[0], "close").loc["2013-01-01":END].index
    opens = pd.DataFrame(index=calendar, columns=all_codes, dtype=float)
    closes = pd.DataFrame(index=calendar, columns=all_codes, dtype=float)
    qm20 = pd.DataFrame(index=calendar, columns=all_codes, dtype=float)
    momentum = {h: pd.DataFrame(index=calendar, columns=all_codes, dtype=float) for h in (20, 60, 120)}
    vol60 = pd.DataFrame(index=calendar, columns=all_codes, dtype=float)
    above_ma120 = pd.DataFrame(index=calendar, columns=all_codes, dtype=bool)

    for number, code in enumerate(all_codes, 1):
        close = native(code, "close")
        open_ = native(code, "open")
        opens[code] = open_.reindex(calendar)
        closes[code] = close.reindex(calendar)
        daily = close.pct_change(fill_method=None)
        vol60[code] = (daily.rolling(60).std() * math.sqrt(252.0)).reindex(calendar)
        for horizon in (20, 60, 120):
            momentum[horizon][code] = close.pct_change(horizon, fill_method=None).reindex(calendar)
        path20 = close.diff().abs().rolling(20).sum()
        er20 = (close - close.shift(20)).abs() / path20.replace(0.0, np.nan)
        qm20[code] = (close.pct_change(20, fill_method=None) * er20).reindex(calendar)
        above_ma120[code] = (close > close.rolling(120).mean()).reindex(calendar).fillna(False)
        if number % 25 == 0:
            print(f"loaded factors {number}/{len(all_codes)}", flush=True)

    month_dates = pd.DatetimeIndex(sorted(membership["month_end"].unique()))
    monthly = pd.DataFrame(False, index=month_dates, columns=satellite_codes)
    for date, group in membership.groupby("month_end"):
        monthly.loc[pd.Timestamp(date), group["ts_code"].tolist()] = True
    persistent = monthly & monthly.shift(1, fill_value=False)
    # Reindex then lag one trading day: month-end close membership becomes
    # available for signals starting on the next trading day.
    daily_membership = persistent.reindex(calendar, method="ffill").shift(1).fillna(False).astype(bool)

    panels = {"open": opens, "close": closes}
    factors = {
        "qm20": qm20,
        "momentum20": momentum[20],
        "momentum60": momentum[60],
        "momentum120": momentum[120],
        "vol60": vol60,
        "above_ma120": above_ma120,
    }
    return model, panels, factors, daily_membership, membership


def build_targets(model, factors: dict, membership: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    all_codes = model.ALL
    sat_codes = list(model.SATELLITES)
    core_idx = np.array([all_codes.index(code) for code in CORE], dtype=int)
    sat_idx = np.array([all_codes.index(code) for code in sat_codes], dtype=int)
    qm = factors["qm20"][all_codes].to_numpy(dtype=float)
    trend = (
        factors["momentum20"][sat_codes]
        + 0.5 * factors["momentum60"][sat_codes]
        + 0.25 * factors["momentum120"][sat_codes]
    )
    aligned = (
        membership[sat_codes]
        & factors["momentum20"][sat_codes].gt(0.0)
        & factors["momentum60"][sat_codes].gt(0.0)
        & factors["momentum120"][sat_codes].gt(0.0)
        & factors["above_ma120"][sat_codes]
    )
    trend_rank = trend.where(aligned).rank(axis=1, ascending=False, method="first")
    selector = aligned & trend_rank.le(3)
    selector_values = selector.to_numpy(dtype=bool)
    vol_values = factors["vol60"][sat_codes].to_numpy(dtype=float)
    sleeves = [model.SATELLITES[code][1] for code in sat_codes]
    targets = np.zeros((len(qm), len(all_codes)), dtype=float)

    for i in range(len(qm)):
        core_scores = qm[i, core_idx]
        if not np.isfinite(core_scores).any():
            continue
        best_local = int(np.nanargmax(core_scores))
        best_idx = int(core_idx[best_local])
        best_score = float(core_scores[best_local])
        targets[i, best_idx] = 1.0
        if all_codes[best_idx] == "518880.SH":
            continue
        sat_scores = qm[i, sat_idx]
        candidates = np.flatnonzero(selector_values[i] & np.isfinite(sat_scores) & (sat_scores > best_score))
        if not len(candidates):
            continue
        ordered = candidates[np.argsort(-sat_scores[candidates])]
        chosen = []
        used = set()
        for local in ordered:
            sleeve = sleeves[int(local)]
            if sleeve in used:
                continue
            chosen.append(int(local))
            used.add(sleeve)
            if len(chosen) == 2:
                break
        if not chosen:
            continue
        targets[i, best_idx] = 0.85
        if len(chosen) == 1:
            weights = np.array([0.15])
        else:
            vols = vol_values[i, chosen]
            if np.isfinite(vols).all() and (vols > 0).all():
                inv = 1.0 / vols
                weights = 0.15 * inv / inv.sum()
            else:
                weights = np.repeat(0.15 / len(chosen), len(chosen))
        for local, weight in zip(chosen, weights, strict=True):
            targets[i, int(sat_idx[local])] = float(weight)
    diagnostics = pd.DataFrame(
        {
            "eligible_monthly_size_assets": membership.sum(axis=1),
            "aligned_assets": aligned.sum(axis=1),
            "selector_assets": selector.sum(axis=1),
        },
        index=membership.index,
    )
    return targets, diagnostics


def baseline_targets(model, factors: dict) -> np.ndarray:
    scores = factors["qm20"][CORE].to_numpy(dtype=float)
    targets = np.zeros((len(scores), len(model.ALL)), dtype=float)
    for i, row in enumerate(scores):
        if np.isfinite(row).any():
            targets[i, int(np.nanargmax(row))] = 1.0
    return targets


def main() -> None:
    p2 = load_phase2()
    model, panels, factors, daily_membership, monthly_membership = load_market()
    candidate_target, diagnostics = build_targets(model, factors, daily_membership)
    baseline_target = baseline_targets(model, factors)
    baseline_sim = p2.simulate_weighted(model, panels, baseline_target, 5, record_positions=True)
    candidate_sim = p2.simulate_weighted(model, panels, candidate_target, 5, record_positions=True)
    baseline_1, baseline_5 = p2.net(baseline_sim, 0.0001), p2.net(baseline_sim, 0.0005)
    candidate_1, candidate_5 = p2.net(candidate_sim, 0.0001), p2.net(candidate_sim, 0.0005)

    # Baseline compatibility with the fixed-universe Phase-2 engine.
    m2 = p2.load_phase1()
    b2 = m2.load_panels()
    fp2 = p2.factor_panels(m2, b2)
    target2 = p2.build_target_matrix(m2, fp2["scores"]["QM20"], None, fp2, 0.0, 1, "beat_best", None)
    sim2 = p2.simulate_weighted(m2, b2, target2, 5)
    reference = p2.net(sim2, 0.0001)
    overlap = baseline_1.index.intersection(reference.index)
    max_baseline_diff = float((baseline_1.loc[overlap] - reference.loc[overlap]).abs().max())
    if max_baseline_diff > 1e-12:
        raise RuntimeError(f"baseline mismatch: {max_baseline_diff}")

    periods = {
        "D": (p2.D_START, p2.D_END),
        "V": (p2.V_START, p2.V_END),
        "T_pseudo_oos": (p2.T_START, p2.T_END),
        "FULL": (p2.D_START, p2.T_END),
    }
    rows = []
    for fee, baseline, candidate in [(1.0, baseline_1, candidate_1), (5.0, baseline_5, candidate_5)]:
        for period, (start, end) in periods.items():
            b = p2.metric_pack(p2.segment(baseline, start, end))
            c = p2.metric_pack(p2.segment(candidate, start, end))
            rows.append(
                {
                    "fee_bps_one_side": fee,
                    "period": period,
                    "baseline_annual_return": b["annual_return"],
                    "candidate_annual_return": c["annual_return"],
                    "baseline_sharpe": b["sharpe"],
                    "candidate_sharpe": c["sharpe"],
                    "sharpe_delta": c["sharpe"] - b["sharpe"],
                    "baseline_max_drawdown": b["max_drawdown"],
                    "candidate_max_drawdown": c["max_drawdown"],
                }
            )
    comparison = pd.DataFrame(rows)
    rolling = p2.rolling36(candidate_1, baseline_1)
    lead_share = float(rolling["candidate_leads"].mean())
    one = comparison[comparison.fee_bps_one_side == 1.0].set_index("period")
    five = comparison[comparison.fee_bps_one_side == 5.0].set_index("period")
    deltas = one.loc[["D", "V", "T_pseudo_oos"], "sharpe_delta"]
    satellite_days = int(candidate_sim["satellite_exposure"].gt(0).sum())
    gates = pd.DataFrame(
        [
            {"gate": "D/V/T Sharpe deltas nonnegative", "value": f"D={deltas['D']:.4f};V={deltas['V']:.4f};T={deltas['T_pseudo_oos']:.4f}", "passed": bool((deltas >= 0).all())},
            {"gate": "full 1bp Sharpe delta >= +0.05", "value": float(one.loc["FULL", "sharpe_delta"]), "passed": bool(one.loc["FULL", "sharpe_delta"] >= 0.05)},
            {"gate": "full 5bp Sharpe direction positive", "value": float(five.loc["FULL", "sharpe_delta"]), "passed": bool(five.loc["FULL", "sharpe_delta"] > 0)},
            {"gate": "full maxDD deterioration <= 3pp", "value": float(one.loc["FULL", "candidate_max_drawdown"] - one.loc["FULL", "baseline_max_drawdown"]), "passed": bool(one.loc["FULL", "candidate_max_drawdown"] - one.loc["FULL", "baseline_max_drawdown"] >= -0.03)},
            {"gate": "rolling36 lead share >= 60%", "value": lead_share, "passed": bool(lead_share >= 0.60)},
            {"gate": "satellite holding days >= 200", "value": satellite_days, "passed": bool(satellite_days >= 200)},
            {"gate": "baseline exact replication", "value": max_baseline_diff, "passed": bool(max_baseline_diff <= 1e-12)},
        ]
    )

    positions = candidate_sim["positions"]
    holdings = []
    for code in model.ALL:
        weight = positions[code]
        holdings.append(
            {
                "asset": code,
                "name": model.CORE_NAMES.get(code, model.SATELLITES.get(code, (code, ""))[0]),
                "sleeve": "core" if code in CORE else model.SATELLITES[code][1],
                "holding_days": int(weight.gt(0).sum()),
                "average_weight_when_held": float(weight[weight.gt(0)].mean()) if weight.gt(0).any() else 0.0,
                "average_weight_all_days": float(weight.mean()),
            }
        )
    holdings = pd.DataFrame(holdings).sort_values("average_weight_all_days", ascending=False)
    yearly = pd.DataFrame(
        {
            "baseline": (1.0 + baseline_1).groupby(baseline_1.index.year).prod() - 1.0,
            "candidate": (1.0 + candidate_1).groupby(candidate_1.index.year).prod() - 1.0,
        }
    )
    yearly["candidate_minus_baseline"] = yearly["candidate"] - yearly["baseline"]
    yearly.index.name = "year"
    latest = pd.DataFrame(
        {
            "asset": model.ALL,
            "name": [model.CORE_NAMES.get(code, model.SATELLITES.get(code, (code, ""))[0]) for code in model.ALL],
            "signal_target_weight": candidate_target[-1],
            "simulated_weight": positions.iloc[-1].to_numpy(),
        }
    )
    latest = latest[(latest.signal_target_weight > 0) | (latest.simulated_weight > 0)]

    comparison.to_csv(HERE / f"{PREFIX}_comparison.csv", index=False)
    gates.to_csv(HERE / f"{PREFIX}_gates.csv", index=False)
    rolling.to_csv(HERE / f"{PREFIX}_rolling36m.csv", index=False)
    holdings.to_csv(HERE / f"{PREFIX}_holdings.csv", index=False)
    yearly.to_csv(HERE / f"{PREFIX}_yearly.csv")
    latest.to_csv(HERE / f"{PREFIX}_latest_weights.csv", index=False)
    diagnostics.loc[EVAL_START:END].to_csv(HERE / f"{PREFIX}_daily_eligibility.csv")
    monthly_membership.to_csv(HERE / f"{PREFIX}_monthly_membership_source.csv", index=False)

    print(comparison.to_string(index=False))
    print("\nGates")
    print(gates.to_string(index=False))
    print(f"\nfull-universe codes={len(model.SATELLITES)} satellite_days={satellite_days}")


if __name__ == "__main__":
    main()
