"""Narrow local audit of the post-hoc smooth-budget center."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PHASE5_PATH = HERE / "exp_four_etf_tail_factors_phase5_multi_mechanism.py"
PREFIX = "2026-08-17_four_etf_tail_factors_phase11"
CENTER = (0.825, 0.25, 0.25)


def load_phase5():
    spec = importlib.util.spec_from_file_location("phase5_for_phase11", PHASE5_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {PHASE5_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    p5 = load_phase5()
    p2 = p5.load_module("phase2_for_phase11", p5.PHASE2_PATH)
    p3 = p5.load_module("phase3_for_phase11", p5.PHASE3_PATH)
    p1 = p2.load_phase1()
    prices, fields = p1.load_panels()
    factors, _ = p1.build_factors(prices, fields)
    qm20 = factors["qm20"][p1.CORE]
    risk = p5.weighted_risk(factors["risk_ranks"], p5.load_fitted_weights())[p1.CORE]
    mom20 = prices["close"][p1.CORE].pct_change(20, fill_method=None)
    base_target = p5.base_target(p1, qm20)
    base_sim = p2.simulate_weighted(p1, base_target, prices["open"], prices["close"])
    baseline_1 = p1.net(base_sim, p1.FEE_MAIN)
    baseline_5 = p1.net(base_sim, p1.FEE_STRESS)
    periods = {"D": (p1.EVAL_START, p1.D_END), "V": (p1.V_START, p1.V_END), "T": (p1.T_START, p1.END), "FULL": (p1.EVAL_START, p1.END)}

    qm_values = qm20.to_numpy(dtype=float)
    risk_values = risk.to_numpy(dtype=float)
    mom_values = mom20.to_numpy(dtype=float)
    qm_fill = np.where(np.isfinite(qm_values), qm_values, -np.inf)
    winner = np.argmax(qm_fill, axis=1)
    rows_index = np.arange(len(qm20))
    valid_winner = np.isfinite(qm_values[rows_index, winner])
    winner_risk = risk_values[rows_index, winner]
    winner_momentum = mom_values[rows_index, winner]
    risk_max = np.nanmax(risk_values, axis=1)
    risk_is_highest = np.isfinite(winner_risk) & (winner_risk >= risk_max - 1e-12)
    alternative_score = np.where(np.isfinite(mom_values) & (mom_values > 0.0), risk_values, np.inf)
    alternative_score[rows_index, winner] = np.inf
    alternative = np.argmin(alternative_score, axis=1)
    valid_alternative = np.isfinite(alternative_score[rows_index, alternative])

    surface_rows = []
    center_artifacts = None
    local_id = 0
    for threshold in (0.815, 0.825, 0.835):
        for momentum_cap in (0.225, 0.250, 0.275):
            for maximum_budget in (0.20, 0.25, 0.30):
                local_id += 1
                budget = maximum_budget * np.clip((momentum_cap - winner_momentum) / momentum_cap, 0.0, 1.0)
                condition = (
                    valid_winner
                    & risk_is_highest
                    & valid_alternative
                    & (winner_risk >= threshold)
                    & np.isfinite(budget)
                    & (budget > 1e-12)
                )
                target_values = base_target.to_numpy(dtype=float).copy()
                active_rows = rows_index[condition]
                target_values[active_rows] = 0.0
                target_values[active_rows, winner[active_rows]] = 1.0 - budget[active_rows]
                target_values[active_rows, alternative[active_rows]] = budget[active_rows]
                target = pd.DataFrame(target_values, index=qm20.index, columns=p1.CORE)
                simulation = p2.simulate_weighted(p1, target, prices["open"], prices["close"])
                candidate_1 = p1.net(simulation, p1.FEE_MAIN)
                candidate_5 = p1.net(simulation, p1.FEE_STRESS)
                row = {
                    "local_id": local_id,
                    "threshold": threshold,
                    "momentum_cap": momentum_cap,
                    "maximum_budget": maximum_budget,
                    "signal_days": int(condition.sum()),
                }
                all_triple = True
                for period, (start, end) in periods.items():
                    candidate_segment = candidate_1.loc[start:end]
                    base_segment = baseline_1.loc[start:end]
                    deltas = {
                        "sharpe": p1.sharpe(candidate_segment) - p1.sharpe(base_segment),
                        "annual": p1.annual_return(candidate_segment) - p1.annual_return(base_segment),
                        "top10": p1.top10_summary(candidate_segment)["top10_mean_depth"] - p1.top10_summary(base_segment)["top10_mean_depth"],
                    }
                    for metric, value in deltas.items():
                        row[f"{period}_{metric}_delta"] = value
                    all_triple &= min(deltas.values()) > 0.0
                row["all_segments_triple"] = all_triple
                row["FULL_5bp_sharpe_delta"] = p1.sharpe(candidate_5) - p1.sharpe(baseline_5)
                row["FULL_5bp_annual_delta"] = p1.annual_return(candidate_5) - p1.annual_return(baseline_5)
                surface_rows.append(row)
                if all(np.isclose(value, center) for value, center in zip((threshold, momentum_cap, maximum_budget), CENTER)):
                    center_artifacts = (target, simulation, candidate_1, candidate_5)
    surface = pd.DataFrame(surface_rows).sort_values(["all_segments_triple", "FULL_annual_delta", "FULL_sharpe_delta"], ascending=[False, False, False])
    if center_artifacts is None:
        raise RuntimeError("center missing")
    target, simulation, center_1, center_5 = center_artifacts
    center_row = surface.loc[
        np.isclose(surface["threshold"], CENTER[0])
        & np.isclose(surface["momentum_cap"], CENTER[1])
        & np.isclose(surface["maximum_budget"], CENTER[2])
    ].iloc[0]
    local_pass_rate = float(surface["all_segments_triple"].mean())
    rolling = p1.rolling36(center_1, baseline_1)
    same_windows = p1.same_window_comparison(baseline_1, center_1)
    maxdd_delta = p1.max_drawdown(center_1) - p1.max_drawdown(baseline_1)
    active = target.sum(axis=1) > 0.0
    target_valid = bool(list(target.columns) == p1.CORE and (target.loc[active] >= -1e-12).all().all() and np.allclose(target.loc[active].sum(axis=1), 1.0, atol=1e-12))
    gates = pd.DataFrame(
        [
            {"gate": "local all-segment triple pass rate >=30%", "value": local_pass_rate, "passed": local_pass_rate >= 0.30},
            {"gate": "center all-segment triple", "value": bool(center_row["all_segments_triple"]), "passed": bool(center_row["all_segments_triple"])},
            {"gate": "center 5bp Sharpe positive", "value": center_row["FULL_5bp_sharpe_delta"], "passed": center_row["FULL_5bp_sharpe_delta"] > 0.0},
            {"gate": "center 5bp annual positive", "value": center_row["FULL_5bp_annual_delta"], "passed": center_row["FULL_5bp_annual_delta"] > 0.0},
            {"gate": "center maxDD no worse -1pp", "value": maxdd_delta, "passed": maxdd_delta >= -0.01},
            {"gate": "center rolling36 lead >=60%", "value": float(rolling["candidate_leads"].mean()), "passed": float(rolling["candidate_leads"].mean()) >= 0.60},
            {"gate": "center same-window wins >=7", "value": int(same_windows["candidate_improves"].sum()), "passed": int(same_windows["candidate_improves"].sum()) >= 7},
            {"gate": "target invariants", "value": target_valid, "passed": target_valid},
        ]
    )
    bootstrap = p5.paired_triple_bootstrap(p1, p3, baseline_1, center_1)
    summary = pd.DataFrame(
        [{
            "surface_points": len(surface),
            "all_segment_triple_points": int(surface["all_segments_triple"].sum()),
            "local_pass_rate": local_pass_rate,
            "center_all_gates_passed": bool(gates["passed"].all()),
            "center_full_sharpe_delta": center_row["FULL_sharpe_delta"],
            "center_full_annual_delta": center_row["FULL_annual_delta"],
            "center_full_top10_delta": center_row["FULL_top10_delta"],
        }]
    )
    surface.to_csv(HERE / f"{PREFIX}_surface.csv", index=False)
    gates.to_csv(HERE / f"{PREFIX}_gates.csv", index=False)
    bootstrap.to_csv(HERE / f"{PREFIX}_bootstrap.csv", index=False)
    rolling.to_csv(HERE / f"{PREFIX}_rolling36m.csv", index=False)
    same_windows.to_csv(HERE / f"{PREFIX}_same_window_top10.csv", index=False)
    summary.to_csv(HERE / f"{PREFIX}_summary.csv", index=False)
    print("Summary")
    print(summary.to_string(index=False))
    print("\nSurface")
    print(surface.to_string(index=False))
    print("\nGates")
    print(gates.to_string(index=False))
    print("\nBootstrap")
    print(bootstrap.to_string(index=False))


if __name__ == "__main__":
    main()
