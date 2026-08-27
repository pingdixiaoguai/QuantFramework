"""Combine a risk-delay return sleeve with a SAFE_POS defensive sleeve."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PHASE12_PATH = HERE / "exp_four_etf_tail_factors_phase12_risk_delay.py"
PREFIX = "2026-08-17_four_etf_tail_factors_phase14"


def load_phase12():
    spec = importlib.util.spec_from_file_location("phase12_for_phase14", PHASE12_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {PHASE12_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def combine_sleeves(primary: pd.Series, defensive: pd.Series, defensive_weight: float) -> pd.Series:
    joined = pd.concat([primary.rename("primary"), defensive.rename("defensive")], axis=1).dropna()
    primary_wealth = (1.0 + joined["primary"]).cumprod()
    defensive_wealth = (1.0 + joined["defensive"]).cumprod()
    wealth = (1.0 - defensive_weight) * primary_wealth + defensive_weight * defensive_wealth
    returns = wealth.pct_change(fill_method=None)
    returns.iloc[0] = wealth.iloc[0] - 1.0
    return returns


def main() -> None:
    p12 = load_phase12()
    p5 = p12.load_phase5()
    p2 = p5.load_module("phase2_for_phase14", p5.PHASE2_PATH)
    p3 = p5.load_module("phase3_for_phase14", p5.PHASE3_PATH)
    p1 = p2.load_phase1()
    prices, fields = p1.load_panels()
    factors, _ = p1.build_factors(prices, fields)
    qm20 = factors["qm20"]
    targets = p1.targets_from_score(qm20)
    risk = p5.weighted_risk(factors["risk_ranks"], p5.load_fitted_weights())
    mom20 = prices["close"][p1.CORE].pct_change(20, fill_method=None)
    qm_rank, _, confidence = p5.signal_inputs(p1, qm20, prices["close"])
    base_sim = p12.simulate_delay(p1, targets, prices["open"], prices["close"], risk, mom20, confidence)
    baseline_1 = p1.net(base_sim, p1.FEE_MAIN)
    baseline_5 = p1.net(base_sim, p1.FEE_STRESS)
    delay_params = {"threshold": 0.75, "delay_days": 3, "incumbent_momentum_floor": 0.05, "confidence_cap": 1.0, "minimum_risk_spread": 0.0}
    delay_sim = p12.simulate_delay(p1, targets, prices["open"], prices["close"], risk, mom20, confidence, delay_params)
    delay_1 = p1.net(delay_sim, p1.FEE_MAIN)
    delay_5 = p1.net(delay_sim, p1.FEE_STRESS)
    defensive_streams = {}
    for budget in (0.20, 0.35):
        params = {"family": "SAFE_POS", "threshold": 0.70, "budget": budget}
        target, _ = p5.build_candidate_target(p1, qm20, qm_rank, mom20, risk, confidence, params)
        simulation = p2.simulate_weighted(p1, target, prices["open"], prices["close"])
        defensive_streams[budget] = (p1.net(simulation, p1.FEE_MAIN), p1.net(simulation, p1.FEE_STRESS))
    periods = {"D": (p1.EVAL_START, p1.D_END), "V": (p1.V_START, p1.V_END), "T": (p1.T_START, p1.END), "FULL": (p1.EVAL_START, p1.END)}
    rows = []
    artifacts = {}
    candidate_id = 0
    for defensive_budget, (defensive_1, defensive_5) in defensive_streams.items():
        for sleeve_weight in (0.10, 0.20, 0.30, 0.40):
            candidate_id += 1
            name = f"P14_{candidate_id:02d}"
            candidate_1 = combine_sleeves(delay_1, defensive_1, sleeve_weight)
            candidate_5 = combine_sleeves(delay_5, defensive_5, sleeve_weight)
            artifacts[name] = (candidate_1, candidate_5)
            row = {"candidate": name, "defensive_budget": defensive_budget, "defensive_sleeve_weight": sleeve_weight}
            improvements = []
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
                if period in {"D", "V"}:
                    improvements.extend([deltas["sharpe"] / 0.05, deltas["annual"] / 0.01, deltas["top10"] / 0.005])
            row["robust_score"] = min(improvements)
            row["eligible_DV_triple"] = min(improvements) > 1e-6
            row["all_segments_triple"] = all_triple
            row["FULL_5bp_sharpe_delta"] = p1.sharpe(candidate_5) - p1.sharpe(baseline_5)
            row["FULL_5bp_annual_delta"] = p1.annual_return(candidate_5) - p1.annual_return(baseline_5)
            rows.append(row)
    screen = pd.DataFrame(rows).sort_values(["eligible_DV_triple", "robust_score", "defensive_sleeve_weight"], ascending=[False, False, True])
    eligible = screen.loc[screen["eligible_DV_triple"]]
    selected = str(eligible.iloc[0]["candidate"]) if len(eligible) else None
    screen.to_csv(HERE / f"{PREFIX}_screen.csv", index=False)
    if selected is None:
        pd.DataFrame([{"selected": None}]).to_csv(HERE / f"{PREFIX}_summary.csv", index=False)
        print(screen.to_string(index=False))
        return
    selected_1, selected_5 = artifacts[selected]
    selected_row = screen.loc[screen["candidate"] == selected].iloc[0]
    rolling = p1.rolling36(selected_1, baseline_1)
    same_windows = p1.same_window_comparison(baseline_1, selected_1)
    maxdd_delta = p1.max_drawdown(selected_1) - p1.max_drawdown(baseline_1)
    gates = []
    for period in periods:
        for metric in ("sharpe", "annual", "top10"):
            value = selected_row[f"{period}_{metric}_delta"]
            gates.append({"gate": f"{period} {metric} positive", "value": value, "passed": value > 0.0})
    gates.extend(
        [
            {"gate": "5bp Sharpe positive", "value": selected_row["FULL_5bp_sharpe_delta"], "passed": selected_row["FULL_5bp_sharpe_delta"] > 0.0},
            {"gate": "5bp annual positive", "value": selected_row["FULL_5bp_annual_delta"], "passed": selected_row["FULL_5bp_annual_delta"] > 0.0},
            {"gate": "maxDD no worse -1pp", "value": maxdd_delta, "passed": maxdd_delta >= -0.01},
            {"gate": "rolling36 lead >=60%", "value": float(rolling["candidate_leads"].mean()), "passed": float(rolling["candidate_leads"].mean()) >= 0.60},
            {"gate": "same-window wins >=7", "value": int(same_windows["candidate_improves"].sum()), "passed": int(same_windows["candidate_improves"].sum()) >= 7},
        ]
    )
    gates = pd.DataFrame(gates)
    bootstrap = p5.paired_triple_bootstrap(p1, p3, baseline_1, selected_1)
    comparison_rows = []
    for fee, base_returns, candidate_returns in ((1.0, baseline_1, selected_1), (5.0, baseline_5, selected_5)):
        for period, (start, end) in periods.items():
            for strategy, returns in (("BASE_QM20", base_returns), (selected, candidate_returns)):
                one = returns.loc[start:end]
                comparison_rows.append({"fee_bps_one_side": fee, "period": period, "strategy": strategy, "annual_return": p1.annual_return(one), "sharpe": p1.sharpe(one), "max_drawdown": p1.max_drawdown(one), **p1.top10_summary(one)})
    comparison = pd.DataFrame(comparison_rows)
    summary = pd.DataFrame([{"selected": selected, "all_gates_passed": bool(gates["passed"].all()), "eligible_DV_count": len(eligible), "all_segment_triple_count": int(screen["all_segments_triple"].sum()), "full_sharpe_delta": selected_row["FULL_sharpe_delta"], "full_annual_delta": selected_row["FULL_annual_delta"], "full_top10_delta": selected_row["FULL_top10_delta"], "rolling36_lead": float(rolling["candidate_leads"].mean()), "same_window_wins": int(same_windows["candidate_improves"].sum())}])
    gates.to_csv(HERE / f"{PREFIX}_gates.csv", index=False)
    comparison.to_csv(HERE / f"{PREFIX}_comparison.csv", index=False)
    bootstrap.to_csv(HERE / f"{PREFIX}_bootstrap.csv", index=False)
    rolling.to_csv(HERE / f"{PREFIX}_rolling36m.csv", index=False)
    same_windows.to_csv(HERE / f"{PREFIX}_same_window_top10.csv", index=False)
    summary.to_csv(HERE / f"{PREFIX}_summary.csv", index=False)
    print("Screen")
    print(screen.to_string(index=False))
    print("\nSelected", selected)
    print("\nComparison")
    print(comparison.to_string(index=False))
    print("\nGates")
    print(gates.to_string(index=False))
    print("\nBootstrap")
    print(bootstrap.to_string(index=False))


if __name__ == "__main__":
    main()
