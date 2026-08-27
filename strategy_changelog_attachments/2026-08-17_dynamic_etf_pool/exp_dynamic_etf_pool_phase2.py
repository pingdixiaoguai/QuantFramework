"""Phase-2 dynamic ETF universe mechanism search (research only).

Run from repository root:
    uv run python strategy_changelog_attachments/2026-08-17_dynamic_etf_pool/exp_dynamic_etf_pool_phase2.py

The design is frozen in 2026-08-17_dynamic_etf_pool_phase2_design.md.  The
pseudo-OOS period is never used to rank candidates.
"""

from __future__ import annotations

import importlib.util
import itertools
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PHASE1_PATH = HERE / "exp_dynamic_etf_pool.py"
PREFIX = "2026-08-17_dynamic_etf_pool_phase2"

D_START = pd.Timestamp("2014-01-02")
D_END = pd.Timestamp("2018-12-31")
V_START = pd.Timestamp("2019-01-01")
V_END = pd.Timestamp("2022-12-30")
T_START = pd.Timestamp("2023-01-01")
T_END = pd.Timestamp("2026-08-14")
FEE_MAIN = 0.0001
FEE_STRESS = 0.0005

SELECTOR_NAMES = [
    "P60_TOP5",
    "ALIGN_TOP3",
    "MEDIAN_TOP3",
    "BEST_TOP2",
    "RISK_MEDIAN_TOP3",
]
FACTOR_NAMES = ["QM20", "ER_FLOOR20", "QM60", "DUAL_RANK", "TRI_RANK"]


def load_phase1():
    spec = importlib.util.spec_from_file_location("dynamic_etf_phase1", PHASE1_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {PHASE1_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def factor_panels(m, base: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    calendar = base["close"].index
    momentum = {h: pd.DataFrame(index=calendar, columns=m.ALL, dtype=float) for h in (20, 60, 120)}
    er = {h: pd.DataFrame(index=calendar, columns=m.ALL, dtype=float) for h in (20, 60, 120)}
    qm = {h: pd.DataFrame(index=calendar, columns=m.ALL, dtype=float) for h in (20, 60, 120)}
    vol60 = pd.DataFrame(index=calendar, columns=m.ALL, dtype=float)

    for code in m.ALL:
        close = m._native_series(code, "close").sort_index()
        daily = close.pct_change(fill_method=None)
        vol60[code] = (daily.rolling(60).std() * math.sqrt(252.0)).reindex(calendar)
        for h in (20, 60, 120):
            mom = close.pct_change(h, fill_method=None)
            path = close.diff().abs().rolling(h).sum()
            efficiency = (close - close.shift(h)).abs() / path.replace(0.0, np.nan)
            momentum[h][code] = mom.reindex(calendar)
            er[h][code] = efficiency.reindex(calendar)
            qm[h][code] = (mom * efficiency).reindex(calendar)

    rank_qm = {h: qm[h].rank(axis=1, pct=True) for h in (20, 60, 120)}
    scores = {
        "QM20": qm[20],
        "ER_FLOOR20": momentum[20] * (0.5 + 0.5 * er[20]),
        "QM60": qm[60],
        "DUAL_RANK": 0.6 * rank_qm[20] + 0.4 * rank_qm[60],
        "TRI_RANK": 0.5 * rank_qm[20] + 0.3 * rank_qm[60] + 0.2 * rank_qm[120],
    }
    trend = momentum[20] + 0.5 * momentum[60] + 0.25 * momentum[120]
    risk_trend = trend / vol60.replace(0.0, np.nan)
    returns = base["close"].pct_change(fill_method=None)
    correlations: dict[str, pd.DataFrame] = {}
    for core in m.CORE:
        correlations[core] = pd.DataFrame(
            {
                sat: returns[sat].rolling(60, min_periods=40).corr(returns[core])
                for sat in m.SATELLITES
            },
            index=calendar,
        )
    return {
        "momentum20": momentum[20],
        "momentum60": momentum[60],
        "momentum120": momentum[120],
        "trend": trend,
        "risk_trend": risk_trend,
        "vol60": vol60,
        "scores": scores,
        "correlations": correlations,
    }


def _top_slots(mask: pd.DataFrame, ranking: pd.DataFrame, slots: int) -> pd.DataFrame:
    rank = ranking.where(mask).rank(axis=1, ascending=False, method="first")
    return mask & rank.le(slots)


def selector_panels(m, base: dict[str, pd.DataFrame], fp: dict) -> dict[str, pd.DataFrame]:
    sat = list(m.SATELLITES)
    size_arm = next(arm for arm in m.ARMS if arm.name == "D0_size_only")
    size_ok = m.eligibility_for(size_arm, base)[sat]
    p20, p60, p120 = fp["momentum20"][sat], fp["momentum60"][sat], fp["momentum120"][sat]
    above = base["above_ma120"][sat].fillna(False).astype(bool)
    trend = fp["trend"][sat]
    aligned = size_ok & p20.gt(0.0) & p60.gt(0.0) & p120.gt(0.0) & above

    p60_mask = size_ok & p60.gt(0.0) & above
    median_ref = fp["trend"][m.CORE].median(axis=1)
    best_ref = fp["trend"][m.CORE].max(axis=1)
    risk_ref = fp["risk_trend"][m.CORE].median(axis=1)
    panels = {
        "P60_TOP5": _top_slots(p60_mask, trend, 5),
        "ALIGN_TOP3": _top_slots(aligned, trend, 3),
        "MEDIAN_TOP3": _top_slots(aligned & trend.gt(median_ref, axis=0), trend, 3),
        "BEST_TOP2": _top_slots(aligned & trend.gt(best_ref, axis=0), trend, 2),
        "RISK_MEDIAN_TOP3": _top_slots(
            aligned & fp["risk_trend"][sat].gt(risk_ref, axis=0),
            fp["risk_trend"][sat],
            3,
        ),
    }
    return panels


def build_target_matrix(
    m,
    score: pd.DataFrame,
    selector: pd.DataFrame | None,
    fp: dict,
    satellite_weight: float,
    satellite_topn: int,
    activation: str,
    corr_cap: float | None,
) -> np.ndarray:
    all_codes = m.ALL
    core_idx = np.array([all_codes.index(code) for code in m.CORE], dtype=int)
    sat_codes = list(m.SATELLITES)
    sat_idx = np.array([all_codes.index(code) for code in sat_codes], dtype=int)
    sleeves = [m.SATELLITES[code][1] for code in sat_codes]
    score_values = score[all_codes].to_numpy(dtype=float)
    eligible = (
        np.zeros((len(score), len(sat_codes)), dtype=bool)
        if selector is None
        else selector[sat_codes].fillna(False).to_numpy(dtype=bool)
    )
    vol_values = fp["vol60"][sat_codes].to_numpy(dtype=float)
    corr_values = np.stack(
        [fp["correlations"][core][sat_codes].to_numpy(dtype=float) for core in m.CORE],
        axis=1,
    )
    targets = np.zeros((len(score), len(all_codes)), dtype=float)

    for i in range(len(score)):
        core_scores = score_values[i, core_idx]
        finite_core = np.isfinite(core_scores)
        if not finite_core.any():
            continue
        best_core_local = int(np.nanargmax(core_scores))
        best_core_idx = int(core_idx[best_core_local])
        best_core_score = float(core_scores[best_core_local])
        targets[i, best_core_idx] = 1.0
        if selector is None or satellite_weight <= 0:
            continue

        sat_scores = score_values[i, sat_idx]
        mask = eligible[i] & np.isfinite(sat_scores)
        if activation == "beat_best":
            mask &= sat_scores > best_core_score
        elif activation != "selector_only":
            raise ValueError(f"unknown activation: {activation}")
        if corr_cap is not None:
            corr = corr_values[i, best_core_local]
            mask &= np.isfinite(corr) & (corr <= corr_cap)
        candidates = np.flatnonzero(mask)
        if not len(candidates):
            continue

        ordered = candidates[np.argsort(-sat_scores[candidates])]
        chosen: list[int] = []
        used_sleeves: set[str] = set()
        for local_idx in ordered:
            sleeve = sleeves[int(local_idx)]
            if satellite_topn > 1 and sleeve in used_sleeves:
                continue
            chosen.append(int(local_idx))
            used_sleeves.add(sleeve)
            if len(chosen) >= satellite_topn:
                break
        if not chosen:
            continue

        targets[i, best_core_idx] = 1.0 - satellite_weight
        if len(chosen) == 1:
            allocations = np.array([satellite_weight])
        else:
            vols = vol_values[i, chosen]
            if np.isfinite(vols).all() and (vols > 0).all():
                inv = 1.0 / vols
                allocations = satellite_weight * inv / inv.sum()
            else:
                allocations = np.repeat(satellite_weight / len(chosen), len(chosen))
        for local_idx, weight in zip(chosen, allocations, strict=True):
            targets[i, int(sat_idx[local_idx])] = float(weight)
    return targets


def simulate_weighted(m, base: dict[str, pd.DataFrame], targets: np.ndarray, rebalance_days: int, record_positions: bool = False) -> dict:
    dates = base["close"].index
    opens = base["open"][m.ALL].to_numpy(dtype=float)
    closes = base["close"][m.ALL].to_numpy(dtype=float)
    n_assets = len(m.ALL)
    current = np.zeros(n_assets, dtype=float)
    pending: np.ndarray | None = None
    pending_idx: int | None = None
    entry_idx: int | None = None
    gross = np.zeros(len(dates), dtype=float)
    turnover = np.zeros(len(dates), dtype=float)
    satellite_exposure = np.zeros(len(dates), dtype=float)
    positions = np.zeros((len(dates), n_assets), dtype=float) if record_positions else None
    sat_start = len(m.CORE)

    def weighted_ratio(weights: np.ndarray, numerator: np.ndarray, denominator: np.ndarray) -> float:
        ratios = np.ones(n_assets, dtype=float)
        valid = np.isfinite(numerator) & np.isfinite(denominator) & (denominator != 0)
        ratios[valid] = numerator[valid] / denominator[valid]
        return float(np.dot(weights, ratios) - weights.sum())

    for i in range(len(dates)):
        if i > 0:
            if pending_idx == i and pending is not None:
                overnight = weighted_ratio(current, opens[i], closes[i - 1])
                old = current.copy()
                current = pending
                entry_idx = i
                turnover[i] = float(np.abs(current - old).sum())
                intraday = weighted_ratio(current, closes[i], opens[i])
                gross[i] = (1.0 + overnight) * (1.0 + intraday) - 1.0
                pending = None
                pending_idx = None
            elif current.sum() > 0:
                gross[i] = weighted_ratio(current, closes[i], closes[i - 1])

        satellite_exposure[i] = float(current[sat_start:].sum())
        if positions is not None:
            positions[i] = current
        holding = i - entry_idx + 1 if entry_idx is not None and current.sum() > 0 else None
        should_signal = pending is None and (current.sum() == 0 or holding is None or holding >= rebalance_days)
        if should_signal and i + 1 < len(dates):
            new = targets[i]
            if new.sum() > 0 and not np.allclose(new, current, atol=1e-12):
                needed = new > 0
                if np.isfinite(opens[i + 1, needed]).all() and np.isfinite(closes[i + 1, needed]).all():
                    pending = new.copy()
                    pending_idx = i + 1

    result = {
        "gross": pd.Series(gross, index=dates).loc[m.EVAL_START:T_END],
        "turnover": pd.Series(turnover, index=dates).loc[m.EVAL_START:T_END],
        "satellite_exposure": pd.Series(satellite_exposure, index=dates).loc[m.EVAL_START:T_END],
    }
    if positions is not None:
        result["positions"] = pd.DataFrame(positions, index=dates, columns=m.ALL).loc[m.EVAL_START:T_END]
    return result


def net(sim: dict, fee: float) -> pd.Series:
    return sim["gross"] - sim["turnover"] * fee


def sharpe(r: pd.Series) -> float:
    sd = float(r.std())
    return float(r.mean() / sd * math.sqrt(252.0)) if sd > 0 else 0.0


def annual_return(r: pd.Series) -> float:
    return float((1.0 + r).prod() ** (252.0 / len(r)) - 1.0) if len(r) else 0.0


def max_drawdown(r: pd.Series) -> float:
    wealth = (1.0 + r).cumprod()
    return float((wealth / wealth.cummax() - 1.0).min()) if len(r) else 0.0


def segment(r: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    return r.loc[start:end]


def metric_pack(r: pd.Series) -> dict[str, float]:
    return {"annual_return": annual_return(r), "sharpe": sharpe(r), "max_drawdown": max_drawdown(r)}


def rolling36(candidate: pd.Series, baseline: pd.Series) -> pd.DataFrame:
    joined = pd.concat([baseline.rename("baseline"), candidate.rename("candidate")], axis=1).dropna()
    rows = []
    for end in range(756, len(joined) + 1, 21):
        chunk = joined.iloc[end - 756:end]
        b, c = sharpe(chunk["baseline"]), sharpe(chunk["candidate"])
        rows.append(
            {
                "window_end": chunk.index[-1].date().isoformat(),
                "baseline_sharpe": b,
                "candidate_sharpe": c,
                "candidate_leads": c > b,
            }
        )
    return pd.DataFrame(rows)


def config_id(row: dict) -> str:
    corr = "none" if row["corr_cap"] is None else str(row["corr_cap"])
    return (
        f"{row['selector']}|{row['factor']}|w{row['satellite_weight']:.2f}|"
        f"n{row['satellite_topn']}|{row['activation']}|corr{corr}|rd{row['rebalance_days']}"
    )


def main() -> None:
    m = load_phase1()
    base = m.load_panels()
    fp = factor_panels(m, base)
    selectors = selector_panels(m, base, fp)

    baseline_targets = build_target_matrix(
        m, fp["scores"]["QM20"], None, fp, 0.0, 1, "beat_best", None
    )
    baseline_sim = simulate_weighted(m, base, baseline_targets, 5)
    baseline_1 = net(baseline_sim, FEE_MAIN)
    baseline_5 = net(baseline_sim, FEE_STRESS)

    # Exact compatibility gate against the phase-1 engine / production runner.
    phase1_b0 = next(arm for arm in m.ARMS if arm.name == "B0_core4")
    phase1_eligible = m.eligibility_for(phase1_b0, base)
    phase1_targets = m.signal_targets(base["score"], phase1_eligible)
    phase1_sim = m.simulate(phase1_targets, base["open"], base["close"])
    phase1_net = phase1_sim["gross"] - phase1_sim["turnover"] * FEE_MAIN
    validation_overlap = baseline_1.index.intersection(phase1_net.index)
    max_baseline_diff = float((baseline_1.loc[validation_overlap] - phase1_net.loc[validation_overlap]).abs().max())
    if max_baseline_diff > 1e-12:
        raise RuntimeError(f"weighted engine failed B0 replication: max diff {max_baseline_diff}")

    b_d = sharpe(segment(baseline_1, D_START, D_END))
    b_v = sharpe(segment(baseline_1, V_START, V_END))
    b_t = sharpe(segment(baseline_1, T_START, T_END))
    selector_rows = []
    for name in SELECTOR_NAMES:
        target = build_target_matrix(
            m, fp["scores"]["QM20"], selectors[name], fp, 0.25, 1, "beat_best", None
        )
        sim = simulate_weighted(m, base, target, 5)
        r = net(sim, FEE_MAIN)
        d, v, t = (
            sharpe(segment(r, D_START, D_END)),
            sharpe(segment(r, V_START, V_END)),
            sharpe(segment(r, T_START, T_END)),
        )
        selector_rows.append(
            {
                "selector": name,
                "sharpe_D": d,
                "delta_D": d - b_d,
                "sharpe_V": v,
                "delta_V": v - b_v,
                "min_DV_delta": min(d - b_d, v - b_v),
                "sharpe_DV": sharpe(segment(r, D_START, V_END)),
                "sharpe_T_not_used_for_selection": t,
                "delta_T_not_used_for_selection": t - b_t,
                "satellite_days": int(sim["satellite_exposure"].gt(0).sum()),
            }
        )
    selector_df = pd.DataFrame(selector_rows).sort_values(
        ["min_DV_delta", "sharpe_DV", "selector"], ascending=[False, False, True]
    )
    selected_selectors = selector_df.head(3)["selector"].tolist()
    print("Selected universe rules by D/V only:", selected_selectors, flush=True)

    rows = []
    combinations = list(
        itertools.product(
            selected_selectors,
            FACTOR_NAMES,
            (0.25, 0.40),
            (1, 2),
            ("beat_best", "selector_only"),
            (None, 0.85),
            (5, 10, 20),
        )
    )
    for number, (selector, factor, weight, topn, activation, corr_cap, rd) in enumerate(combinations, 1):
        target = build_target_matrix(
            m,
            fp["scores"][factor],
            selectors[selector],
            fp,
            weight,
            topn,
            activation,
            corr_cap,
        )
        sim = simulate_weighted(m, base, target, rd)
        r = net(sim, FEE_MAIN)
        d = sharpe(segment(r, D_START, D_END))
        v = sharpe(segment(r, V_START, V_END))
        t = sharpe(segment(r, T_START, T_END))
        row = {
            "selector": selector,
            "factor": factor,
            "satellite_weight": weight,
            "satellite_topn": topn,
            "activation": activation,
            "corr_cap": corr_cap,
            "rebalance_days": rd,
            "sharpe_D": d,
            "delta_D": d - b_d,
            "sharpe_V": v,
            "delta_V": v - b_v,
            "min_DV_delta": min(d - b_d, v - b_v),
            "sharpe_DV": sharpe(segment(r, D_START, V_END)),
            "sharpe_T_not_used_for_selection": t,
            "delta_T_not_used_for_selection": t - b_t,
            "full_sharpe": sharpe(r),
            "full_annual_return": annual_return(r),
            "full_max_drawdown": max_drawdown(r),
            "satellite_days": int(sim["satellite_exposure"].gt(0).sum()),
        }
        row["config_id"] = config_id(row)
        rows.append(row)
        if number % 120 == 0:
            print(f"searched {number}/{len(combinations)}", flush=True)

    search = pd.DataFrame(rows).sort_values(
        ["min_DV_delta", "sharpe_DV", "satellite_weight", "satellite_topn", "rebalance_days", "config_id"],
        ascending=[False, False, True, True, True, True],
    )
    winner = search.iloc[0].to_dict()
    winner_target = build_target_matrix(
        m,
        fp["scores"][str(winner["factor"])],
        selectors[str(winner["selector"])],
        fp,
        float(winner["satellite_weight"]),
        int(winner["satellite_topn"]),
        str(winner["activation"]),
        None if pd.isna(winner["corr_cap"]) else float(winner["corr_cap"]),
    )
    winner_sim = simulate_weighted(m, base, winner_target, int(winner["rebalance_days"]), record_positions=True)
    winner_1 = net(winner_sim, FEE_MAIN)
    winner_5 = net(winner_sim, FEE_STRESS)

    periods = {
        "D": (D_START, D_END),
        "V": (V_START, V_END),
        "T_pseudo_oos": (T_START, T_END),
        "FULL": (D_START, T_END),
    }
    comparison_rows = []
    for fee, base_r, candidate_r in [
        (1.0, baseline_1, winner_1),
        (5.0, baseline_5, winner_5),
    ]:
        for period, (start, end) in periods.items():
            b = metric_pack(segment(base_r, start, end))
            c = metric_pack(segment(candidate_r, start, end))
            comparison_rows.append(
                {
                    "config_id": winner["config_id"],
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
    comparison = pd.DataFrame(comparison_rows)
    roll = rolling36(winner_1, baseline_1)
    lead_share = float(roll["candidate_leads"].mean())
    position_df = winner_sim["positions"]
    holdings = []
    for code in m.ALL:
        weight = position_df[code]
        holdings.append(
            {
                "asset": code,
                "name": m.CORE_NAMES.get(code, m.SATELLITES.get(code, (code, ""))[0]),
                "sleeve": "core" if code in m.CORE else m.SATELLITES[code][1],
                "holding_days": int(weight.gt(0).sum()),
                "average_weight_when_held": float(weight[weight.gt(0)].mean()) if weight.gt(0).any() else 0.0,
                "average_weight_all_days": float(weight.mean()),
            }
        )
    holdings_df = pd.DataFrame(holdings)
    yearly = pd.DataFrame(
        {
            "baseline": (1.0 + baseline_1).groupby(baseline_1.index.year).prod() - 1.0,
            "candidate": (1.0 + winner_1).groupby(winner_1.index.year).prod() - 1.0,
        }
    )
    yearly["candidate_minus_baseline"] = yearly["candidate"] - yearly["baseline"]
    yearly.index.name = "year"

    one_bp = comparison[comparison.fee_bps_one_side == 1.0].set_index("period")
    five_bp = comparison[comparison.fee_bps_one_side == 5.0].set_index("period")
    segment_deltas = one_bp.loc[["D", "V", "T_pseudo_oos"], "sharpe_delta"]
    satellite_days = int(winner_sim["satellite_exposure"].gt(0).sum())
    gates = pd.DataFrame(
        [
            {
                "gate": "D/V/T Sharpe nonnegative and at least two >= +0.05",
                "value": f"D={segment_deltas['D']:.4f};V={segment_deltas['V']:.4f};T={segment_deltas['T_pseudo_oos']:.4f}",
                "passed": bool((segment_deltas >= 0).all() and (segment_deltas >= 0.05).sum() >= 2),
            },
            {
                "gate": "full 1bp Sharpe delta >= +0.05",
                "value": float(one_bp.loc["FULL", "sharpe_delta"]),
                "passed": bool(one_bp.loc["FULL", "sharpe_delta"] >= 0.05),
            },
            {
                "gate": "full 5bp Sharpe direction positive",
                "value": float(five_bp.loc["FULL", "sharpe_delta"]),
                "passed": bool(five_bp.loc["FULL", "sharpe_delta"] > 0),
            },
            {
                "gate": "full maxDD deterioration <= 3pp",
                "value": float(one_bp.loc["FULL", "candidate_max_drawdown"] - one_bp.loc["FULL", "baseline_max_drawdown"]),
                "passed": bool(one_bp.loc["FULL", "candidate_max_drawdown"] - one_bp.loc["FULL", "baseline_max_drawdown"] >= -0.03),
            },
            {
                "gate": "rolling36 lead share >= 60%",
                "value": lead_share,
                "passed": bool(lead_share >= 0.60),
            },
            {
                "gate": "satellite holding days >= 200",
                "value": satellite_days,
                "passed": bool(satellite_days >= 200),
            },
        ]
    )
    gates.insert(0, "selected_config", winner["config_id"])
    validation = pd.DataFrame(
        [
            {
                "check": "weighted B0 exact replication",
                "value": max_baseline_diff,
                "threshold": "<=1e-12",
                "passed": max_baseline_diff <= 1e-12,
            },
            {
                "check": "pseudo-OOS excluded from selection sort",
                "value": "sort=min_DV_delta,sharpe_DV,complexity",
                "threshold": "design",
                "passed": True,
            },
        ]
    )

    selector_df.to_csv(HERE / f"{PREFIX}_selector_screen.csv", index=False)
    search.to_csv(HERE / f"{PREFIX}_search.csv", index=False)
    comparison.to_csv(HERE / f"{PREFIX}_candidate_comparison.csv", index=False)
    roll.to_csv(HERE / f"{PREFIX}_rolling36m.csv", index=False)
    holdings_df.to_csv(HERE / f"{PREFIX}_holdings.csv", index=False)
    yearly.to_csv(HERE / f"{PREFIX}_yearly.csv")
    gates.to_csv(HERE / f"{PREFIX}_gates.csv", index=False)
    validation.to_csv(HERE / f"{PREFIX}_validation.csv", index=False)

    print("\nWinner selected by D/V only:", winner["config_id"])
    print(comparison.to_string(index=False))
    print("\nGates")
    print(gates.to_string(index=False))


if __name__ == "__main__":
    main()
