"""Local robustness audit for the post-hoc RA_CLOSE center candidate."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PHASE5_PATH = HERE / "exp_four_etf_tail_factors_phase5_multi_mechanism.py"
PREFIX = "2026-08-17_four_etf_tail_factors_phase6"
CENTER = {
    "family": "RA_CLOSE",
    "threshold": 0.80,
    "confidence_cap": 1.00,
    "beta": 0.75,
    "budget": 0.20,
}


def load_phase5():
    spec = importlib.util.spec_from_file_location("phase5_for_phase6", PHASE5_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {PHASE5_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def same_params(left: dict[str, object], right: dict[str, object]) -> bool:
    return all(str(left.get(key)) == str(right.get(key)) for key in CENTER)


def main() -> None:
    p5 = load_phase5()
    p2 = p5.load_module("phase2_for_phase6", p5.PHASE2_PATH)
    p3 = p5.load_module("phase3_for_phase6", p5.PHASE3_PATH)
    p1 = p2.load_phase1()
    prices, fields = p1.load_panels()
    factors, _ = p1.build_factors(prices, fields)
    qm20 = factors["qm20"]
    risk = p5.weighted_risk(factors["risk_ranks"], p5.load_fitted_weights())
    qm_rank, mom20, confidence = p5.signal_inputs(p1, qm20, prices["close"])
    base_target = p5.base_target(p1, qm20)
    base_sim = p2.simulate_weighted(p1, base_target, prices["open"], prices["close"])
    baseline_1 = p1.net(base_sim, p1.FEE_MAIN)
    baseline_5 = p1.net(base_sim, p1.FEE_STRESS)
    periods = {
        "D": (p1.EVAL_START, p1.D_END),
        "V": (p1.V_START, p1.V_END),
        "T": (p1.T_START, p1.END),
        "FULL": (p1.EVAL_START, p1.END),
    }
    rows = []
    center_artifacts = None
    counter = 0
    for threshold in (0.775, 0.800, 0.825):
        for cap in (0.75, 1.00, 1.25):
            for beta in (0.50, 0.75, 1.00):
                for budget in (0.15, 0.20, 0.25):
                    counter += 1
                    params = {
                        "family": "RA_CLOSE",
                        "threshold": threshold,
                        "confidence_cap": cap,
                        "beta": beta,
                        "budget": budget,
                    }
                    target, triggers = p5.build_candidate_target(
                        p1, qm20, qm_rank, mom20, risk, confidence, params
                    )
                    simulation = p2.simulate_weighted(
                        p1, target, prices["open"], prices["close"]
                    )
                    candidate_1 = p1.net(simulation, p1.FEE_MAIN)
                    candidate_5 = p1.net(simulation, p1.FEE_STRESS)
                    row = {"local_id": counter, **params, "trigger_days": int(triggers["date"].nunique())}
                    triple_all = True
                    for period, (start, end) in periods.items():
                        candidate_segment = candidate_1.loc[start:end]
                        base_segment = baseline_1.loc[start:end]
                        deltas = {
                            "sharpe": p1.sharpe(candidate_segment) - p1.sharpe(base_segment),
                            "annual": p1.annual_return(candidate_segment) - p1.annual_return(base_segment),
                            "top10": p1.top10_summary(candidate_segment)["top10_mean_depth"]
                            - p1.top10_summary(base_segment)["top10_mean_depth"],
                        }
                        for name, value in deltas.items():
                            row[f"{period}_{name}_delta"] = value
                        triple_all &= min(deltas.values()) >= -1e-12
                    row["all_segments_triple"] = triple_all
                    row["FULL_5bp_sharpe_delta"] = p1.sharpe(candidate_5) - p1.sharpe(baseline_5)
                    row["FULL_5bp_annual_delta"] = p1.annual_return(candidate_5) - p1.annual_return(baseline_5)
                    rows.append(row)
                    if same_params(params, CENTER):
                        center_artifacts = (target, triggers, simulation, candidate_1, candidate_5)
    surface = pd.DataFrame(rows).sort_values(
        ["all_segments_triple", "FULL_annual_delta", "FULL_sharpe_delta", "FULL_top10_delta"],
        ascending=[False, False, False, False],
    )
    if center_artifacts is None:
        raise RuntimeError("center candidate missing")
    target, triggers, simulation, center_1, center_5 = center_artifacts
    rolling = p1.rolling36(center_1, baseline_1)
    same_windows = p1.same_window_comparison(baseline_1, center_1)
    rolling_lead = float(rolling["candidate_leads"].mean())
    same_window_wins = int(same_windows["candidate_improves"].sum())
    maxdd_delta = p1.max_drawdown(center_1) - p1.max_drawdown(baseline_1)
    local_pass_rate = float(surface["all_segments_triple"].mean())
    active = target.sum(axis=1) > 0.0
    target_valid = bool(
        list(target.columns) == p1.CORE
        and (target.loc[active] >= -1e-12).all().all()
        and np.allclose(target.loc[active].sum(axis=1), 1.0, atol=1e-12)
    )
    center_row = surface.loc[
        np.isclose(surface["threshold"], CENTER["threshold"])
        & np.isclose(surface["confidence_cap"], CENTER["confidence_cap"])
        & np.isclose(surface["beta"], CENTER["beta"])
        & np.isclose(surface["budget"], CENTER["budget"])
    ].iloc[0]
    gates = pd.DataFrame(
        [
            {"gate": "local all-segment triple pass rate >=30%", "value": local_pass_rate, "passed": local_pass_rate >= 0.30},
            {"gate": "center D/V/T/FULL triple", "value": bool(center_row["all_segments_triple"]), "passed": bool(center_row["all_segments_triple"])},
            {"gate": "center FULL 5bp Sharpe positive", "value": center_row["FULL_5bp_sharpe_delta"], "passed": center_row["FULL_5bp_sharpe_delta"] >= 0.0},
            {"gate": "center FULL 5bp annual positive", "value": center_row["FULL_5bp_annual_delta"], "passed": center_row["FULL_5bp_annual_delta"] >= 0.0},
            {"gate": "center maxDD no worse than -1pp", "value": maxdd_delta, "passed": maxdd_delta >= -0.01},
            {"gate": "center rolling36 Sharpe lead >=60%", "value": rolling_lead, "passed": rolling_lead >= 0.60},
            {"gate": "center baseline Top10 windows wins >=7", "value": same_window_wins, "passed": same_window_wins >= 7},
            {"gate": "target invariants", "value": target_valid, "passed": target_valid},
        ]
    )
    comparison_rows = []
    for fee, base_returns, candidate_returns in ((1.0, baseline_1, center_1), (5.0, baseline_5, center_5)):
        for period, (start, end) in periods.items():
            for name, returns in (("BASE_QM20", base_returns), ("RA_CENTER", candidate_returns)):
                one = returns.loc[start:end]
                comparison_rows.append(
                    {
                        "fee_bps_one_side": fee,
                        "period": period,
                        "strategy": name,
                        "annual_return": p1.annual_return(one),
                        "sharpe": p1.sharpe(one),
                        "max_drawdown": p1.max_drawdown(one),
                        **p1.top10_summary(one),
                    }
                )
    comparison = pd.DataFrame(comparison_rows)
    yearly_rows = []
    for year in sorted(set(baseline_1.index.year)):
        base_year = baseline_1.loc[baseline_1.index.year == year]
        center_year = center_1.loc[center_1.index.year == year]
        yearly_rows.append(
            {
                "year": year,
                "annual_delta": float((1.0 + center_year).prod() - (1.0 + base_year).prod()),
                "sharpe_delta": p1.sharpe(center_year) - p1.sharpe(base_year),
                "top10_delta": p1.top10_summary(center_year)["top10_mean_depth"]
                - p1.top10_summary(base_year)["top10_mean_depth"],
            }
        )
    yearly = pd.DataFrame(yearly_rows)
    yearly["triple_win"] = yearly[["annual_delta", "sharpe_delta", "top10_delta"]].ge(0.0).all(axis=1)
    bootstrap = p5.paired_triple_bootstrap(p1, p3, baseline_1, center_1)
    summary = pd.DataFrame(
        [
            {
                "surface_points": len(surface),
                "all_segment_triple_points": int(surface["all_segments_triple"].sum()),
                "local_pass_rate": local_pass_rate,
                "center_all_gates_passed": bool(gates["passed"].all()),
                "center_full_sharpe_delta": center_row["FULL_sharpe_delta"],
                "center_full_annual_delta": center_row["FULL_annual_delta"],
                "center_full_top10_delta": center_row["FULL_top10_delta"],
            }
        ]
    )
    surface.to_csv(HERE / f"{PREFIX}_ra_local_surface.csv", index=False)
    gates.to_csv(HERE / f"{PREFIX}_gates.csv", index=False)
    comparison.to_csv(HERE / f"{PREFIX}_center_comparison.csv", index=False)
    yearly.to_csv(HERE / f"{PREFIX}_center_yearly.csv", index=False)
    bootstrap.to_csv(HERE / f"{PREFIX}_center_bootstrap.csv", index=False)
    rolling.to_csv(HERE / f"{PREFIX}_center_rolling36m.csv", index=False)
    same_windows.to_csv(HERE / f"{PREFIX}_center_same_window_top10.csv", index=False)
    triggers.to_csv(HERE / f"{PREFIX}_center_triggers.csv", index=False)
    summary.to_csv(HERE / f"{PREFIX}_summary.csv", index=False)
    print("Local surface summary")
    print(summary.to_string(index=False))
    print("\nCenter comparison")
    print(comparison.to_string(index=False))
    print("\nGates")
    print(gates.to_string(index=False))
    print("\nYearly")
    print(yearly.to_string(index=False))
    print("\nBootstrap")
    print(bootstrap.to_string(index=False))


if __name__ == "__main__":
    main()
