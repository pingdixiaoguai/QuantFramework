"""Phase-4 relative-strength selector audit on the full historical universe."""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PHASE3_PATH = HERE / "exp_dynamic_etf_full_universe.py"
PREFIX = "2026-08-17_dynamic_etf_pool_phase4"
RULES = ["ALIGN", "REL_TREND_WINNER", "REL_ALL_WINNER", "REL_TREND_BEST", "REL_ALL_BEST"]


def load_phase3():
    spec = importlib.util.spec_from_file_location("dynamic_etf_phase3_for_phase4", PHASE3_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {PHASE3_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def paired_block_bootstrap(p2, baseline: pd.Series, candidate: pd.Series) -> pd.DataFrame:
    joined = pd.concat([baseline.rename("baseline"), candidate.rename("candidate")], axis=1).dropna()
    values_array = joined.to_numpy(dtype=float)
    n = len(values_array)
    rng = np.random.default_rng(8172026)
    rows = []
    for block in (20, 60, 120):
        values = np.empty(3000, dtype=float)
        for replication in range(len(values)):
            starts = rng.integers(0, n - block + 1, size=math.ceil(n / block))
            indices = np.concatenate([np.arange(start, start + block) for start in starts])[:n]
            sample = values_array[indices]
            values[replication] = p2.sharpe(pd.Series(sample[:, 1])) - p2.sharpe(pd.Series(sample[:, 0]))
        rows.append(
            {
                "block_days": block,
                "replicates": len(values),
                "observed_sharpe_delta": p2.sharpe(candidate) - p2.sharpe(baseline),
                "bootstrap_mean": float(values.mean()),
                "probability_delta_positive": float((values > 0).mean()),
                "q025": float(np.quantile(values, 0.025)),
                "q05": float(np.quantile(values, 0.05)),
                "median": float(np.quantile(values, 0.50)),
                "q95": float(np.quantile(values, 0.95)),
                "q975": float(np.quantile(values, 0.975)),
            }
        )
    return pd.DataFrame(rows)


def build_targets(p3, model, panels, factors, membership, rule: str, corr_cap: float | None):
    all_codes = model.ALL
    sat_codes = list(model.SATELLITES)
    core_idx = np.array([all_codes.index(code) for code in model.CORE], dtype=int)
    sat_idx = np.array([all_codes.index(code) for code in sat_codes], dtype=int)
    qm = factors["qm20"][all_codes].to_numpy(dtype=float)
    p20 = factors["momentum20"][all_codes].to_numpy(dtype=float)
    p60 = factors["momentum60"][all_codes].to_numpy(dtype=float)
    p120 = factors["momentum120"][all_codes].to_numpy(dtype=float)
    trend = p20 + 0.5 * p60 + 0.25 * p120
    vol = factors["vol60"][sat_codes].to_numpy(dtype=float)
    aligned = (
        membership[sat_codes].to_numpy(dtype=bool)
        & (p20[:, sat_idx] > 0)
        & (p60[:, sat_idx] > 0)
        & (p120[:, sat_idx] > 0)
        & factors["above_ma120"][sat_codes].to_numpy(dtype=bool)
    )
    returns = panels["close"][all_codes].pct_change(fill_method=None)
    correlations = np.stack(
        [
            pd.DataFrame(
                {
                    sat: returns[sat].rolling(60, min_periods=40).corr(returns[core])
                    for sat in sat_codes
                },
                index=returns.index,
            ).to_numpy(dtype=float)
            for core in model.CORE
        ],
        axis=1,
    )
    sleeves = [model.SATELLITES[code][1] for code in sat_codes]
    targets = np.zeros((len(qm), len(all_codes)), dtype=float)
    eligible_counts = np.zeros(len(qm), dtype=int)

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
        mask = aligned[i].copy()
        if rule == "REL_TREND_WINNER":
            mask &= trend[i, sat_idx] > trend[i, best_idx]
        elif rule == "REL_ALL_WINNER":
            mask &= (
                (p20[i, sat_idx] > p20[i, best_idx])
                & (p60[i, sat_idx] > p60[i, best_idx])
                & (p120[i, sat_idx] > p120[i, best_idx])
            )
        elif rule == "REL_TREND_BEST":
            core_trend = trend[i, core_idx]
            if np.isfinite(core_trend).any():
                mask &= trend[i, sat_idx] > np.nanmax(core_trend)
            else:
                mask &= False
        elif rule == "REL_ALL_BEST":
            core_horizons = [p20[i, core_idx], p60[i, core_idx], p120[i, core_idx]]
            if all(np.isfinite(values).any() for values in core_horizons):
                mask &= (
                    (p20[i, sat_idx] > np.nanmax(core_horizons[0]))
                    & (p60[i, sat_idx] > np.nanmax(core_horizons[1]))
                    & (p120[i, sat_idx] > np.nanmax(core_horizons[2]))
                )
            else:
                mask &= False
        elif rule != "ALIGN":
            raise ValueError(rule)
        if corr_cap is not None:
            corr = correlations[i, best_local]
            mask &= np.isfinite(corr) & (corr <= corr_cap)
        eligible_counts[i] = int(mask.sum())
        candidates = np.flatnonzero(mask)
        if not len(candidates):
            continue
        # Dynamic pool has at most three mainline candidates by the unchanged
        # multi-horizon trend score; final activation remains QM20 > core.
        ordered_trend = candidates[np.argsort(-trend[i, sat_idx][candidates])][:3]
        active = ordered_trend[qm[i, sat_idx][ordered_trend] > best_score]
        if not len(active):
            continue
        ordered_qm = active[np.argsort(-qm[i, sat_idx][active])]
        chosen = []
        used = set()
        for local in ordered_qm:
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
            vols = vol[i, chosen]
            if np.isfinite(vols).all() and (vols > 0).all():
                inverse = 1.0 / vols
                weights = 0.15 * inverse / inverse.sum()
            else:
                weights = np.repeat(0.15 / len(chosen), len(chosen))
        for local, weight in zip(chosen, weights, strict=True):
            targets[i, int(sat_idx[local])] = float(weight)
    return targets, pd.Series(eligible_counts, index=panels["close"].index)


def main() -> None:
    p3 = load_phase3()
    p2 = p3.load_phase2()
    model, panels, factors, membership, _ = p3.load_market()
    baseline_target = p3.baseline_targets(model, factors)
    baseline_sim = p2.simulate_weighted(model, panels, baseline_target, 5)
    baseline_1, baseline_5 = p2.net(baseline_sim, 0.0001), p2.net(baseline_sim, 0.0005)
    b_d = p2.sharpe(p2.segment(baseline_1, p2.D_START, p2.D_END))
    b_v = p2.sharpe(p2.segment(baseline_1, p2.V_START, p2.V_END))
    b_t = p2.sharpe(p2.segment(baseline_1, p2.T_START, p2.T_END))

    results = []
    sims = {}
    targets_by_key = {}
    for rule in RULES:
        for corr_cap in (None, 0.85):
            target, eligible_counts = build_targets(p3, model, panels, factors, membership, rule, corr_cap)
            sim = p2.simulate_weighted(model, panels, target, 5)
            r = p2.net(sim, 0.0001)
            d = p2.sharpe(p2.segment(r, p2.D_START, p2.D_END))
            v = p2.sharpe(p2.segment(r, p2.V_START, p2.V_END))
            t = p2.sharpe(p2.segment(r, p2.T_START, p2.T_END))
            key = f"{rule}|corr={'none' if corr_cap is None else corr_cap}"
            results.append(
                {
                    "config": key,
                    "rule": rule,
                    "corr_cap": corr_cap,
                    "delta_D": d - b_d,
                    "delta_V": v - b_v,
                    "min_DV_delta": min(d - b_d, v - b_v),
                    "sharpe_DV": p2.sharpe(p2.segment(r, p2.D_START, p2.V_END)),
                    "delta_T_not_used_for_selection": t - b_t,
                    "full_sharpe": p2.sharpe(r),
                    "full_annual_return": p2.annual_return(r),
                    "full_max_drawdown": p2.max_drawdown(r),
                    "satellite_days": int(sim["satellite_exposure"].gt(0).sum()),
                    "mean_relative_eligible_assets": float(eligible_counts.loc[p3.EVAL_START:p3.END].mean()),
                }
            )
            sims[key] = sim
            targets_by_key[key] = target
    screen = pd.DataFrame(results).sort_values(
        ["min_DV_delta", "sharpe_DV", "config"], ascending=[False, False, True]
    )
    winner = screen.iloc[0]
    winner_key = str(winner["config"])
    winner_target = targets_by_key[winner_key]
    winner_sim = p2.simulate_weighted(model, panels, winner_target, 5, record_positions=True)
    winner_1, winner_5 = p2.net(winner_sim, 0.0001), p2.net(winner_sim, 0.0005)

    periods = {
        "D": (p2.D_START, p2.D_END),
        "V": (p2.V_START, p2.V_END),
        "T_pseudo_oos": (p2.T_START, p2.T_END),
        "FULL": (p2.D_START, p2.T_END),
    }
    rows = []
    for fee, baseline, candidate in [(1.0, baseline_1, winner_1), (5.0, baseline_5, winner_5)]:
        for period, (start, end) in periods.items():
            b = p2.metric_pack(p2.segment(baseline, start, end))
            c = p2.metric_pack(p2.segment(candidate, start, end))
            rows.append(
                {
                    "selected_config": winner_key,
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
    rolling = p2.rolling36(winner_1, baseline_1)
    one = comparison[comparison.fee_bps_one_side == 1.0].set_index("period")
    five = comparison[comparison.fee_bps_one_side == 5.0].set_index("period")
    deltas = one.loc[["D", "V", "T_pseudo_oos"], "sharpe_delta"]
    satellite_days = int(winner_sim["satellite_exposure"].gt(0).sum())
    gates = pd.DataFrame(
        [
            {"gate": "D/V/T Sharpe deltas nonnegative", "value": f"D={deltas['D']:.4f};V={deltas['V']:.4f};T={deltas['T_pseudo_oos']:.4f}", "passed": bool((deltas >= 0).all())},
            {"gate": "full 1bp Sharpe delta >= +0.05", "value": float(one.loc["FULL", "sharpe_delta"]), "passed": bool(one.loc["FULL", "sharpe_delta"] >= 0.05)},
            {"gate": "full 5bp Sharpe direction positive", "value": float(five.loc["FULL", "sharpe_delta"]), "passed": bool(five.loc["FULL", "sharpe_delta"] > 0)},
            {"gate": "full maxDD deterioration <= 3pp", "value": float(one.loc["FULL", "candidate_max_drawdown"] - one.loc["FULL", "baseline_max_drawdown"]), "passed": bool(one.loc["FULL", "candidate_max_drawdown"] - one.loc["FULL", "baseline_max_drawdown"] >= -0.03)},
            {"gate": "rolling36 lead share >= 60%", "value": float(rolling["candidate_leads"].mean()), "passed": bool(rolling["candidate_leads"].mean() >= 0.60)},
            {"gate": "satellite holding days >= 200", "value": satellite_days, "passed": bool(satellite_days >= 200)},
        ]
    )
    gates.insert(0, "selected_config", winner_key)

    positions = winner_sim["positions"]
    holdings = []
    for code in model.ALL:
        weight = positions[code]
        holdings.append(
            {
                "asset": code,
                "name": model.CORE_NAMES.get(code, model.SATELLITES.get(code, (code, ""))[0]),
                "sleeve": "core" if code in model.CORE else model.SATELLITES[code][1],
                "holding_days": int(weight.gt(0).sum()),
                "average_weight_when_held": float(weight[weight.gt(0)].mean()) if weight.gt(0).any() else 0.0,
                "average_weight_all_days": float(weight.mean()),
            }
        )
    holdings = pd.DataFrame(holdings).sort_values("average_weight_all_days", ascending=False)
    yearly = pd.DataFrame(
        {
            "baseline": (1.0 + baseline_1).groupby(baseline_1.index.year).prod() - 1.0,
            "candidate": (1.0 + winner_1).groupby(winner_1.index.year).prod() - 1.0,
        }
    )
    yearly["candidate_minus_baseline"] = yearly["candidate"] - yearly["baseline"]
    yearly.index.name = "year"

    # ALIGN with the 0.85 correlation cap is the economically motivated
    # full-sample audit candidate, even though the frozen min(D, V) selector
    # narrowly prefers uncapped ALIGN. Keep its uncertainty audit separate.
    correlation_audit_key = "ALIGN|corr=0.85"
    correlation_audit_1 = p2.net(sims[correlation_audit_key], 0.0001)
    correlation_bootstrap = paired_block_bootstrap(p2, baseline_1, correlation_audit_1)

    screen.to_csv(HERE / f"{PREFIX}_screen.csv", index=False)
    comparison.to_csv(HERE / f"{PREFIX}_comparison.csv", index=False)
    gates.to_csv(HERE / f"{PREFIX}_gates.csv", index=False)
    rolling.to_csv(HERE / f"{PREFIX}_rolling36m.csv", index=False)
    holdings.to_csv(HERE / f"{PREFIX}_holdings.csv", index=False)
    yearly.to_csv(HERE / f"{PREFIX}_yearly.csv")
    correlation_bootstrap.to_csv(HERE / f"{PREFIX}_corr085_bootstrap.csv", index=False)

    print(screen.to_string(index=False))
    print("\nSelected by D/V only:", winner_key)
    print(comparison.to_string(index=False))
    print("\nGates")
    print(gates.to_string(index=False))
    print("\nALIGN|corr=0.85 bootstrap")
    print(correlation_bootstrap.to_string(index=False))


if __name__ == "__main__":
    main()
