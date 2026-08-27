"""Local robustness audit for the post-hoc risk-delay center."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
PHASE12_PATH = HERE / "exp_four_etf_tail_factors_phase12_risk_delay.py"
PREFIX = "2026-08-17_four_etf_tail_factors_phase13"
CENTER = (0.75, 0.05, 1.0)


def load_phase12():
    spec = importlib.util.spec_from_file_location("phase12_for_phase13", PHASE12_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {PHASE12_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    p12 = load_phase12()
    p5 = p12.load_phase5()
    p2 = p5.load_module("phase2_for_phase13", p5.PHASE2_PATH)
    p3 = p5.load_module("phase3_for_phase13", p5.PHASE3_PATH)
    p1 = p2.load_phase1()
    prices, fields = p1.load_panels()
    factors, _ = p1.build_factors(prices, fields)
    qm20 = factors["qm20"]
    targets = p1.targets_from_score(qm20)
    risk = p5.weighted_risk(factors["risk_ranks"], p5.load_fitted_weights())
    mom20 = prices["close"][p1.CORE].pct_change(20, fill_method=None)
    _, _, confidence = p5.signal_inputs(p1, qm20, prices["close"])
    base_sim = p12.simulate_delay(p1, targets, prices["open"], prices["close"], risk, mom20, confidence)
    baseline_1 = p1.net(base_sim, p1.FEE_MAIN)
    baseline_5 = p1.net(base_sim, p1.FEE_STRESS)
    official_difference = p1.official_baseline_check(baseline_1)
    periods = {"D": (p1.EVAL_START, p1.D_END), "V": (p1.V_START, p1.V_END), "T": (p1.T_START, p1.END), "FULL": (p1.EVAL_START, p1.END)}
    rows = []
    center_artifacts = None
    local_id = 0
    for threshold in (0.725, 0.750, 0.775):
        for floor in (0.025, 0.050, 0.075):
            for cap in (0.8, 1.0, 1.2):
                local_id += 1
                params = {"threshold": threshold, "delay_days": 3, "incumbent_momentum_floor": floor, "confidence_cap": cap, "minimum_risk_spread": 0.0}
                simulation = p12.simulate_delay(p1, targets, prices["open"], prices["close"], risk, mom20, confidence, params)
                candidate_1 = p1.net(simulation, p1.FEE_MAIN)
                candidate_5 = p1.net(simulation, p1.FEE_STRESS)
                row = {"local_id": local_id, "threshold": threshold, "incumbent_momentum_floor": floor, "confidence_cap": cap, "event_days": int(simulation["events"]["date"].nunique() if len(simulation["events"]) else 0)}
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
                rows.append(row)
                if (threshold, floor, cap) == CENTER:
                    center_artifacts = (simulation, candidate_1, candidate_5)
    surface = pd.DataFrame(rows).sort_values(["all_segments_triple", "FULL_annual_delta", "FULL_sharpe_delta"], ascending=[False, False, False])
    if center_artifacts is None:
        raise RuntimeError("center missing")
    simulation, center_1, center_5 = center_artifacts
    center_row = surface.loc[(surface["threshold"] == CENTER[0]) & (surface["incumbent_momentum_floor"] == CENTER[1]) & (surface["confidence_cap"] == CENTER[2])].iloc[0]
    local_pass_rate = float(surface["all_segments_triple"].mean())
    rolling = p1.rolling36(center_1, baseline_1)
    same_windows = p1.same_window_comparison(baseline_1, center_1)
    maxdd_delta = p1.max_drawdown(center_1) - p1.max_drawdown(baseline_1)
    gates = pd.DataFrame(
        [
            {"gate": "local all-segment triple pass rate >=30%", "value": local_pass_rate, "passed": local_pass_rate >= 0.30},
            {"gate": "center all-segment triple", "value": bool(center_row["all_segments_triple"]), "passed": bool(center_row["all_segments_triple"])},
            {"gate": "center 5bp Sharpe positive", "value": center_row["FULL_5bp_sharpe_delta"], "passed": center_row["FULL_5bp_sharpe_delta"] > 0.0},
            {"gate": "center 5bp annual positive", "value": center_row["FULL_5bp_annual_delta"], "passed": center_row["FULL_5bp_annual_delta"] > 0.0},
            {"gate": "center maxDD no worse -1pp", "value": maxdd_delta, "passed": maxdd_delta >= -0.01},
            {"gate": "center rolling36 lead >=60%", "value": float(rolling["candidate_leads"].mean()), "passed": float(rolling["candidate_leads"].mean()) >= 0.60},
            {"gate": "center same-window wins >=7", "value": int(same_windows["candidate_improves"].sum()), "passed": int(same_windows["candidate_improves"].sum()) >= 7},
            {"gate": "official baseline diff <=1e-12", "value": official_difference, "passed": official_difference <= 1e-12},
        ]
    )
    comparison_rows = []
    for fee, base_returns, candidate_returns in ((1.0, baseline_1, center_1), (5.0, baseline_5, center_5)):
        for period, (start, end) in periods.items():
            for strategy, returns in (("BASE_QM20", base_returns), ("RISK_DELAY_CENTER", candidate_returns)):
                one = returns.loc[start:end]
                comparison_rows.append({"fee_bps_one_side": fee, "period": period, "strategy": strategy, "annual_return": p1.annual_return(one), "sharpe": p1.sharpe(one), "max_drawdown": p1.max_drawdown(one), **p1.top10_summary(one)})
    comparison = pd.DataFrame(comparison_rows)
    yearly_rows = []
    for year in sorted(set(baseline_1.index.year)):
        base_year = baseline_1.loc[baseline_1.index.year == year]
        candidate_year = center_1.loc[center_1.index.year == year]
        yearly_rows.append({"year": year, "annual_delta": float((1.0 + candidate_year).prod() - (1.0 + base_year).prod()), "sharpe_delta": p1.sharpe(candidate_year) - p1.sharpe(base_year), "top10_delta": p1.top10_summary(candidate_year)["top10_mean_depth"] - p1.top10_summary(base_year)["top10_mean_depth"]})
    yearly = pd.DataFrame(yearly_rows)
    yearly["triple_win"] = yearly[["annual_delta", "sharpe_delta", "top10_delta"]].gt(0.0).all(axis=1)
    bootstrap = p5.paired_triple_bootstrap(p1, p3, baseline_1, center_1)
    summary = pd.DataFrame([{"surface_points": len(surface), "all_segment_triple_points": int(surface["all_segments_triple"].sum()), "local_pass_rate": local_pass_rate, "center_all_gates_passed": bool(gates["passed"].all()), "center_full_sharpe_delta": center_row["FULL_sharpe_delta"], "center_full_annual_delta": center_row["FULL_annual_delta"], "center_full_top10_delta": center_row["FULL_top10_delta"], "event_days": center_row["event_days"]}])
    surface.to_csv(HERE / f"{PREFIX}_surface.csv", index=False)
    gates.to_csv(HERE / f"{PREFIX}_gates.csv", index=False)
    comparison.to_csv(HERE / f"{PREFIX}_comparison.csv", index=False)
    yearly.to_csv(HERE / f"{PREFIX}_yearly.csv", index=False)
    bootstrap.to_csv(HERE / f"{PREFIX}_bootstrap.csv", index=False)
    rolling.to_csv(HERE / f"{PREFIX}_rolling36m.csv", index=False)
    same_windows.to_csv(HERE / f"{PREFIX}_same_window_top10.csv", index=False)
    simulation["events"].to_csv(HERE / f"{PREFIX}_events.csv", index=False)
    summary.to_csv(HERE / f"{PREFIX}_summary.csv", index=False)
    print("Summary")
    print(summary.to_string(index=False))
    print("\nSurface")
    print(surface.to_string(index=False))
    print("\nComparison")
    print(comparison.to_string(index=False))
    print("\nGates")
    print(gates.to_string(index=False))
    print("\nYearly")
    print(yearly.to_string(index=False))
    print("\nBootstrap")
    print(bootstrap.to_string(index=False))


if __name__ == "__main__":
    main()
