"""Smooth momentum-dependent risk budget."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PHASE5_PATH = HERE / "exp_four_etf_tail_factors_phase5_multi_mechanism.py"
PREFIX = "2026-08-17_four_etf_tail_factors_phase10"


def load_phase5():
    spec = importlib.util.spec_from_file_location("phase5_for_phase10", PHASE5_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {PHASE5_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_target(p1, p5, qm20, mom20, risk, threshold, momentum_cap, maximum_budget):
    target = p5.base_target(p1, qm20)
    triggers = []
    for timestamp in qm20.index:
        qm_row = qm20.loc[timestamp, p1.CORE].dropna()
        risk_row = risk.loc[timestamp, p1.CORE].dropna()
        if len(qm_row) < 2 or len(risk_row) < len(p1.CORE):
            continue
        winner = str(qm_row.idxmax())
        winner_risk = float(risk_row[winner])
        winner_momentum = mom20.at[timestamp, winner]
        if (
            winner_risk < threshold
            or winner_risk < float(risk_row.max()) - 1e-12
            or not np.isfinite(winner_momentum)
        ):
            continue
        budget = maximum_budget * np.clip((momentum_cap - winner_momentum) / momentum_cap, 0.0, 1.0)
        if budget <= 1e-12:
            continue
        positive = [
            code
            for code in p1.CORE
            if code != winner and np.isfinite(mom20.at[timestamp, code]) and mom20.at[timestamp, code] > 0.0
        ]
        if not positive:
            continue
        alternative = str(risk_row[positive].idxmin())
        target.loc[timestamp] = 0.0
        target.at[timestamp, winner] = 1.0 - budget
        target.at[timestamp, alternative] = budget
        triggers.append(
            {
                "date": timestamp,
                "winner": winner,
                "alternative": alternative,
                "winner_risk": winner_risk,
                "winner_momentum20": winner_momentum,
                "budget": budget,
            }
        )
    return target, pd.DataFrame(triggers)


def main() -> None:
    p5 = load_phase5()
    p2 = p5.load_module("phase2_for_phase10", p5.PHASE2_PATH)
    p3 = p5.load_module("phase3_for_phase10", p5.PHASE3_PATH)
    p1 = p2.load_phase1()
    prices, fields = p1.load_panels()
    factors, _ = p1.build_factors(prices, fields)
    qm20 = factors["qm20"]
    mom20 = prices["close"][p1.CORE].pct_change(20, fill_method=None)
    risk = p5.weighted_risk(factors["risk_ranks"], p5.load_fitted_weights())
    base_target = p5.base_target(p1, qm20)
    base_sim = p2.simulate_weighted(p1, base_target, prices["open"], prices["close"])
    baseline_1 = p1.net(base_sim, p1.FEE_MAIN)
    baseline_5 = p1.net(base_sim, p1.FEE_STRESS)
    periods = {"D": (p1.EVAL_START, p1.D_END), "V": (p1.V_START, p1.V_END), "T": (p1.T_START, p1.END), "FULL": (p1.EVAL_START, p1.END)}
    rows = []
    artifacts = {}
    candidate_id = 0
    for threshold in (0.775, 0.800, 0.825):
        for momentum_cap in (0.15, 0.20, 0.25):
            for maximum_budget in (0.25, 0.35, 0.45):
                candidate_id += 1
                name = f"P10_{candidate_id:02d}"
                target, triggers = build_target(p1, p5, qm20, mom20, risk, threshold, momentum_cap, maximum_budget)
                simulation = p2.simulate_weighted(p1, target, prices["open"], prices["close"])
                returns_1 = p1.net(simulation, p1.FEE_MAIN)
                returns_5 = p1.net(simulation, p1.FEE_STRESS)
                artifacts[name] = (target, triggers, simulation, returns_1, returns_5)
                row = {"candidate": name, "threshold": threshold, "momentum_cap": momentum_cap, "maximum_budget": maximum_budget, "trigger_days": int(triggers["date"].nunique() if len(triggers) else 0)}
                improvements = []
                for period in ("D", "V"):
                    start, end = periods[period]
                    candidate_segment = returns_1.loc[start:end]
                    base_segment = baseline_1.loc[start:end]
                    deltas = {
                        "sharpe": p1.sharpe(candidate_segment) - p1.sharpe(base_segment),
                        "annual": p1.annual_return(candidate_segment) - p1.annual_return(base_segment),
                        "top10": p1.top10_summary(candidate_segment)["top10_mean_depth"] - p1.top10_summary(base_segment)["top10_mean_depth"],
                    }
                    for metric, value in deltas.items():
                        row[f"{period}_{metric}_delta"] = value
                    row[f"{period}_trigger_days"] = int(triggers.loc[triggers["date"].between(start, end), "date"].nunique() if len(triggers) else 0)
                    improvements.extend([deltas["sharpe"] / 0.05, deltas["annual"] / 0.01, deltas["top10"] / 0.005])
                row["robust_score"] = min(improvements)
                row["eligible_DV_triple"] = bool(min(improvements) > 1e-6 and row["D_trigger_days"] >= 3 and row["V_trigger_days"] >= 3)
                rows.append(row)
    screen = pd.DataFrame(rows).sort_values(["eligible_DV_triple", "robust_score", "trigger_days", "maximum_budget"], ascending=[False, False, True, True])
    eligible = screen.loc[screen["eligible_DV_triple"]]
    selected = str(eligible.iloc[0]["candidate"]) if len(eligible) else None
    screen.to_csv(HERE / f"{PREFIX}_dv_screen.csv", index=False)
    if selected is None:
        pd.DataFrame([{"selected": None, "eligible_count": 0}]).to_csv(HERE / f"{PREFIX}_summary.csv", index=False)
        print("No D/V triple candidate")
        print(screen.to_string(index=False))
        return
    audit_rows = []
    for name in eligible["candidate"]:
        target, triggers, simulation, returns_1, returns_5 = artifacts[name]
        row = {"candidate": name}
        all_triple = True
        for period, (start, end) in periods.items():
            candidate_segment = returns_1.loc[start:end]
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
        row["FULL_5bp_sharpe_delta"] = p1.sharpe(returns_5) - p1.sharpe(baseline_5)
        row["FULL_5bp_annual_delta"] = p1.annual_return(returns_5) - p1.annual_return(baseline_5)
        audit_rows.append(row)
    audit = pd.DataFrame(audit_rows).merge(screen, on="candidate").sort_values(["all_segments_triple", "FULL_annual_delta", "FULL_sharpe_delta"], ascending=[False, False, False])
    audit.to_csv(HERE / f"{PREFIX}_eligible_audit.csv", index=False)
    target, triggers, simulation, selected_1, selected_5 = artifacts[selected]
    selected_audit = audit.loc[audit["candidate"] == selected].iloc[0]
    rolling = p1.rolling36(selected_1, baseline_1)
    same_windows = p1.same_window_comparison(baseline_1, selected_1)
    bootstrap = p5.paired_triple_bootstrap(p1, p3, baseline_1, selected_1)
    summary = pd.DataFrame([{"selected": selected, "eligible_DV_count": len(eligible), "total_candidates": len(screen), "selected_all_segments_triple": bool(selected_audit["all_segments_triple"]), "any_all_segments_triple": bool(audit["all_segments_triple"].any()), "full_sharpe_delta": selected_audit["FULL_sharpe_delta"], "full_annual_delta": selected_audit["FULL_annual_delta"], "full_top10_delta": selected_audit["FULL_top10_delta"], "rolling36_lead": float(rolling["candidate_leads"].mean()), "same_window_wins": int(same_windows["candidate_improves"].sum())}])
    summary.to_csv(HERE / f"{PREFIX}_summary.csv", index=False)
    bootstrap.to_csv(HERE / f"{PREFIX}_bootstrap.csv", index=False)
    print("Selected", selected)
    print(screen.loc[screen["candidate"] == selected].to_string(index=False))
    print("\nAudit")
    print(audit.to_string(index=False))
    print("\nSummary")
    print(summary.to_string(index=False))
    print("\nBootstrap")
    print(bootstrap.to_string(index=False))


if __name__ == "__main__":
    main()
