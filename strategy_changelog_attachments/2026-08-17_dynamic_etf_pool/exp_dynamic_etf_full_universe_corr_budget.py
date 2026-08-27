"""Phase-5 smooth correlation-budget audit."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PHASE3_PATH = HERE / "exp_dynamic_etf_full_universe.py"
PREFIX = "2026-08-17_dynamic_etf_pool_phase5"


def load_phase3():
    spec = importlib.util.spec_from_file_location("dynamic_etf_phase3_for_phase5", PHASE3_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(PHASE3_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def taper_targets(model, panels, raw_targets: np.ndarray) -> tuple[np.ndarray, pd.DataFrame]:
    targets = raw_targets.copy()
    returns = panels["close"][model.ALL].pct_change(fill_method=None)
    correlations = {
        core: pd.DataFrame(
            {
                sat: returns[sat].rolling(60, min_periods=40).corr(returns[core])
                for sat in model.SATELLITES
            },
            index=returns.index,
        )
        for core in model.CORE
    }
    sat_codes = list(model.SATELLITES)
    sat_idx = np.array([model.ALL.index(code) for code in sat_codes], dtype=int)
    rows = []
    for i, date in enumerate(returns.index):
        sat_weights = targets[i, sat_idx]
        total = float(sat_weights.sum())
        if total <= 0:
            continue
        core_candidates = [code for code in model.CORE if targets[i, model.ALL.index(code)] > 0]
        if len(core_candidates) != 1:
            raise RuntimeError(f"unexpected core targets on {date}")
        core = core_candidates[0]
        selected = np.flatnonzero(sat_weights > 0)
        corr = correlations[core].iloc[i, selected].to_numpy(dtype=float)
        normalized = sat_weights[selected] / total
        if not np.isfinite(corr).all():
            scale = 0.0
            weighted_corr = np.nan
        else:
            weighted_corr = float(np.dot(normalized, corr))
            scale = float(np.clip(2.0 * (1.0 - weighted_corr), 0.0, 1.0))
        new_total = total * scale
        targets[i, sat_idx[selected]] = sat_weights[selected] * scale
        core_idx = model.ALL.index(core)
        targets[i, core_idx] += total - new_total
        rows.append(
            {
                "date": date,
                "core": core,
                "raw_satellite_budget": total,
                "weighted_corr60": weighted_corr,
                "scale": scale,
                "actual_satellite_budget": new_total,
            }
        )
    return targets, pd.DataFrame(rows)


def main() -> None:
    p3 = load_phase3()
    p2 = p3.load_phase2()
    model, panels, factors, membership, _ = p3.load_market()
    raw_candidate, _ = p3.build_targets(model, factors, membership)
    candidate_target, budget = taper_targets(model, panels, raw_candidate)
    baseline_target = p3.baseline_targets(model, factors)
    baseline_sim = p2.simulate_weighted(model, panels, baseline_target, 5)
    candidate_sim = p2.simulate_weighted(model, panels, candidate_target, 5, record_positions=True)
    b1, b5 = p2.net(baseline_sim, 0.0001), p2.net(baseline_sim, 0.0005)
    c1, c5 = p2.net(candidate_sim, 0.0001), p2.net(candidate_sim, 0.0005)
    periods = {
        "D": (p2.D_START, p2.D_END),
        "V": (p2.V_START, p2.V_END),
        "T_pseudo_oos": (p2.T_START, p2.T_END),
        "FULL": (p2.D_START, p2.T_END),
    }
    rows = []
    for fee, baseline, candidate in [(1.0, b1, c1), (5.0, b5, c5)]:
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
    rolling = p2.rolling36(c1, b1)
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
            {"gate": "rolling36 lead share >= 60%", "value": float(rolling["candidate_leads"].mean()), "passed": bool(rolling["candidate_leads"].mean() >= 0.60)},
            {"gate": "satellite holding days >= 200", "value": satellite_days, "passed": bool(satellite_days >= 200)},
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
                "sleeve": "core" if code in model.CORE else model.SATELLITES[code][1],
                "holding_days": int(weight.gt(0).sum()),
                "average_weight_when_held": float(weight[weight.gt(0)].mean()) if weight.gt(0).any() else 0.0,
                "average_weight_all_days": float(weight.mean()),
            }
        )
    yearly = pd.DataFrame(
        {
            "baseline": (1.0 + b1).groupby(b1.index.year).prod() - 1.0,
            "candidate": (1.0 + c1).groupby(c1.index.year).prod() - 1.0,
        }
    )
    yearly["candidate_minus_baseline"] = yearly["candidate"] - yearly["baseline"]
    yearly.index.name = "year"

    comparison.to_csv(HERE / f"{PREFIX}_comparison.csv", index=False)
    gates.to_csv(HERE / f"{PREFIX}_gates.csv", index=False)
    rolling.to_csv(HERE / f"{PREFIX}_rolling36m.csv", index=False)
    budget.to_csv(HERE / f"{PREFIX}_budget_events.csv", index=False)
    pd.DataFrame(holdings).sort_values("average_weight_all_days", ascending=False).to_csv(
        HERE / f"{PREFIX}_holdings.csv", index=False
    )
    yearly.to_csv(HERE / f"{PREFIX}_yearly.csv")

    print(comparison.to_string(index=False))
    print("\nGates")
    print(gates.to_string(index=False))
    print("\nBudget summary")
    print(budget[["weighted_corr60", "scale", "actual_satellite_budget"]].describe().to_string())


if __name__ == "__main__":
    main()
