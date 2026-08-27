"""Post-selection audit of all phase-5 candidates that passed the D/V triple gate."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
PHASE5_PATH = HERE / "exp_four_etf_tail_factors_phase5_multi_mechanism.py"
SCREEN = HERE / "2026-08-17_four_etf_tail_factors_phase5_dv_screen.csv"
OUTPUT = HERE / "2026-08-17_four_etf_tail_factors_phase5_all_eligible_audit.csv"


def load_phase5():
    spec = importlib.util.spec_from_file_location("phase5_for_eligible_audit", PHASE5_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {PHASE5_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    p5 = load_phase5()
    p2 = p5.load_module("phase2_for_phase5_eligible", p5.PHASE2_PATH)
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
    screen = pd.read_csv(SCREEN)
    eligible = screen.loc[screen["eligible_DV_triple"].astype(bool)]
    parameter_map = {row["candidate"]: row for row in p5.candidate_parameters()}
    rows = []
    for candidate in eligible["candidate"]:
        params = parameter_map[candidate]
        target, triggers = p5.build_candidate_target(
            p1, qm20, qm_rank, mom20, risk, confidence, params
        )
        simulation = p2.simulate_weighted(p1, target, prices["open"], prices["close"])
        returns_1 = p1.net(simulation, p1.FEE_MAIN)
        returns_5 = p1.net(simulation, p1.FEE_STRESS)
        row = dict(params)
        passed = True
        for period, (start, end) in periods.items():
            candidate_segment = returns_1.loc[start:end]
            baseline_segment = baseline_1.loc[start:end]
            for metric, function in (
                ("sharpe", p1.sharpe),
                ("annual", p1.annual_return),
            ):
                delta = function(candidate_segment) - function(baseline_segment)
                row[f"{period}_{metric}_delta"] = delta
                passed &= delta >= -1e-12
            top10_delta = (
                p1.top10_summary(candidate_segment)["top10_mean_depth"]
                - p1.top10_summary(baseline_segment)["top10_mean_depth"]
            )
            row[f"{period}_top10_delta"] = top10_delta
            passed &= top10_delta >= -1e-12
        row["FULL_5bp_sharpe_delta"] = p1.sharpe(returns_5) - p1.sharpe(baseline_5)
        row["FULL_5bp_annual_delta"] = p1.annual_return(returns_5) - p1.annual_return(baseline_5)
        row["trigger_days"] = int(triggers["date"].nunique())
        row["passes_all_segments_triple"] = passed
        rows.append(row)
    audit = pd.DataFrame(rows).sort_values(
        ["passes_all_segments_triple", "FULL_annual_delta", "FULL_sharpe_delta"],
        ascending=[False, False, False],
    )
    audit.to_csv(OUTPUT, index=False)
    print(audit.to_string(index=False))


if __name__ == "__main__":
    main()
