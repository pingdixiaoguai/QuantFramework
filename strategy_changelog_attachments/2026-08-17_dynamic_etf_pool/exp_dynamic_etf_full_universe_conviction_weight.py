"""Phase-7 conviction-proportional satellite budget audit."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PHASE3_PATH = HERE / "exp_dynamic_etf_full_universe.py"
PREFIX = "2026-08-17_dynamic_etf_pool_phase7"


def load_phase3():
    spec = importlib.util.spec_from_file_location("dynamic_etf_phase3_for_phase7", PHASE3_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {PHASE3_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def scale_targets(model, factors, raw_targets):
    targets = raw_targets.copy()
    qm = factors["qm20"][model.ALL].to_numpy(dtype=float)
    core_idx = np.array([model.ALL.index(code) for code in model.CORE], dtype=int)
    sat_idx = np.array([model.ALL.index(code) for code in model.SATELLITES], dtype=int)
    events = []
    for i in range(len(targets)):
        active_sat = sat_idx[targets[i, sat_idx] > 0]
        if not len(active_sat):
            continue
        active_core = core_idx[targets[i, core_idx] > 0]
        if len(active_core) != 1:
            raise RuntimeError("expected exactly one active core when satellites are present")
        core = int(active_core[0])
        best_sat_score = float(np.nanmax(qm[i, active_sat]))
        core_score = float(qm[i, core])
        if core_score <= 0:
            scale = 1.0
        else:
            scale = float(np.clip(best_sat_score / core_score - 1.0, 0.0, 1.0))
        old_budget = float(targets[i, active_sat].sum())
        targets[i, active_sat] *= scale
        new_budget = float(targets[i, active_sat].sum())
        targets[i, core] += old_budget - new_budget
        events.append(
            {
                "date": factors["qm20"].index[i],
                "core": model.ALL[core],
                "core_qm20": core_score,
                "best_satellite_qm20": best_sat_score,
                "scale": scale,
                "actual_satellite_budget": new_budget,
            }
        )
    return targets, pd.DataFrame(events)


def main() -> None:
    p3 = load_phase3()
    p2 = p3.load_phase2()
    model, panels, factors, membership, _ = p3.load_market()
    raw_target, _ = p3.build_targets(model, factors, membership)
    candidate_target, events = scale_targets(model, factors, raw_target)
    baseline_target = p3.baseline_targets(model, factors)
    baseline_sim = p2.simulate_weighted(model, panels, baseline_target, 5, record_positions=True)
    candidate_sim = p2.simulate_weighted(model, panels, candidate_target, 5, record_positions=True)
    baseline_1, baseline_5 = p2.net(baseline_sim, 0.0001), p2.net(baseline_sim, 0.0005)
    candidate_1, candidate_5 = p2.net(candidate_sim, 0.0001), p2.net(candidate_sim, 0.0005)

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
    one = comparison[comparison.fee_bps_one_side == 1.0].set_index("period")
    five = comparison[comparison.fee_bps_one_side == 5.0].set_index("period")
    deltas = one.loc[["D", "V", "T_pseudo_oos"], "sharpe_delta"]
    satellite_days = int(candidate_sim["satellite_exposure"].gt(0).sum())
    maxdd_change = float(one.loc["FULL", "candidate_max_drawdown"] - one.loc["FULL", "baseline_max_drawdown"])
    gates = pd.DataFrame(
        [
            {"gate": "D/V/T Sharpe deltas nonnegative", "value": f"D={deltas['D']:.4f};V={deltas['V']:.4f};T={deltas['T_pseudo_oos']:.4f}", "passed": bool((deltas >= 0).all())},
            {"gate": "full 1bp Sharpe delta >= +0.05", "value": float(one.loc["FULL", "sharpe_delta"]), "passed": bool(one.loc["FULL", "sharpe_delta"] >= 0.05)},
            {"gate": "full 5bp Sharpe direction positive", "value": float(five.loc["FULL", "sharpe_delta"]), "passed": bool(five.loc["FULL", "sharpe_delta"] > 0)},
            {"gate": "full maxDD deterioration <= 3pp", "value": maxdd_change, "passed": bool(maxdd_change >= -0.03)},
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
    holdings = pd.DataFrame(holdings).sort_values("average_weight_all_days", ascending=False)
    yearly = pd.DataFrame(
        {
            "baseline": (1.0 + baseline_1).groupby(baseline_1.index.year).prod() - 1.0,
            "candidate": (1.0 + candidate_1).groupby(candidate_1.index.year).prod() - 1.0,
        }
    )
    yearly["candidate_minus_baseline"] = yearly["candidate"] - yearly["baseline"]
    yearly.index.name = "year"

    comparison.to_csv(HERE / f"{PREFIX}_comparison.csv", index=False)
    gates.to_csv(HERE / f"{PREFIX}_gates.csv", index=False)
    rolling.to_csv(HERE / f"{PREFIX}_rolling36m.csv", index=False)
    holdings.to_csv(HERE / f"{PREFIX}_holdings.csv", index=False)
    yearly.to_csv(HERE / f"{PREFIX}_yearly.csv")
    events.to_csv(HERE / f"{PREFIX}_budget_events.csv", index=False)

    print(comparison.to_string(index=False))
    print("\nGates")
    print(gates.to_string(index=False))
    print("\nBudget summary")
    print(events[["scale", "actual_satellite_budget"]].describe().to_string())


if __name__ == "__main__":
    main()
