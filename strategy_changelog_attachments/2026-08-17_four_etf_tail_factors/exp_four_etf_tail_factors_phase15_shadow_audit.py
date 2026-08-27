"""Final audit of the post-hoc two-sleeve shadow center P14_03."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
PHASE14_PATH = HERE / "exp_four_etf_tail_factors_phase14_two_sleeve.py"
PHASE14_SCREEN = HERE / "2026-08-17_four_etf_tail_factors_phase14_screen.csv"
PREFIX = "2026-08-17_four_etf_tail_factors_phase15"


def load_phase14():
    spec = importlib.util.spec_from_file_location("phase14_for_phase15", PHASE14_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {PHASE14_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    p14 = load_phase14()
    p12 = p14.load_phase12()
    p5 = p12.load_phase5()
    p2 = p5.load_module("phase2_for_phase15", p5.PHASE2_PATH)
    p3 = p5.load_module("phase3_for_phase15", p5.PHASE3_PATH)
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
    official_difference = p1.official_baseline_check(baseline_1)
    delay_params = {"threshold": 0.75, "delay_days": 3, "incumbent_momentum_floor": 0.05, "confidence_cap": 1.0, "minimum_risk_spread": 0.0}
    delay_sim = p12.simulate_delay(p1, targets, prices["open"], prices["close"], risk, mom20, confidence, delay_params)
    delay_1 = p1.net(delay_sim, p1.FEE_MAIN)
    delay_5 = p1.net(delay_sim, p1.FEE_STRESS)
    defensive_params = {"family": "SAFE_POS", "threshold": 0.70, "budget": 0.20}
    defensive_target, defensive_triggers = p5.build_candidate_target(p1, qm20, qm_rank, mom20, risk, confidence, defensive_params)
    defensive_sim = p2.simulate_weighted(p1, defensive_target, prices["open"], prices["close"])
    defensive_1 = p1.net(defensive_sim, p1.FEE_MAIN)
    defensive_5 = p1.net(defensive_sim, p1.FEE_STRESS)
    candidate_1 = p14.combine_sleeves(delay_1, defensive_1, 0.30)
    candidate_5 = p14.combine_sleeves(delay_5, defensive_5, 0.30)
    periods = {"D": (p1.EVAL_START, p1.D_END), "V": (p1.V_START, p1.V_END), "T": (p1.T_START, p1.END), "FULL": (p1.EVAL_START, p1.END)}
    comparison_rows = []
    gates = []
    for fee, base_returns, candidate_returns in ((1.0, baseline_1, candidate_1), (5.0, baseline_5, candidate_5)):
        for period, (start, end) in periods.items():
            for strategy, returns in (("BASE_QM20", base_returns), ("TWO_SLEEVE_SHADOW", candidate_returns)):
                one = returns.loc[start:end]
                comparison_rows.append({"fee_bps_one_side": fee, "period": period, "strategy": strategy, "annual_return": p1.annual_return(one), "sharpe": p1.sharpe(one), "max_drawdown": p1.max_drawdown(one), **p1.top10_summary(one)})
    comparison = pd.DataFrame(comparison_rows)
    one = comparison.loc[comparison["fee_bps_one_side"] == 1.0].set_index(["strategy", "period"])
    five = comparison.loc[comparison["fee_bps_one_side"] == 5.0].set_index(["strategy", "period"])
    for period in periods:
        for metric in ("sharpe", "annual_return", "top10_mean_depth"):
            value = one.at[("TWO_SLEEVE_SHADOW", period), metric] - one.at[("BASE_QM20", period), metric]
            gates.append({"gate": f"{period} {metric} positive", "value": value, "passed": value > 0.0})
    phase14_screen = pd.read_csv(PHASE14_SCREEN)
    platform_rate = float(phase14_screen["all_segments_triple"].astype(bool).mean())
    maxdd_delta = one.at[("TWO_SLEEVE_SHADOW", "FULL"), "max_drawdown"] - one.at[("BASE_QM20", "FULL"), "max_drawdown"]
    rolling = p1.rolling36(candidate_1, baseline_1)
    same_windows = p1.same_window_comparison(baseline_1, candidate_1)
    gates.extend(
        [
            {"gate": "phase14 platform pass rate >=50%", "value": platform_rate, "passed": platform_rate >= 0.50},
            {"gate": "FULL 5bp Sharpe positive", "value": five.at[("TWO_SLEEVE_SHADOW", "FULL"), "sharpe"] - five.at[("BASE_QM20", "FULL"), "sharpe"], "passed": five.at[("TWO_SLEEVE_SHADOW", "FULL"), "sharpe"] > five.at[("BASE_QM20", "FULL"), "sharpe"]},
            {"gate": "FULL 5bp annual positive", "value": five.at[("TWO_SLEEVE_SHADOW", "FULL"), "annual_return"] - five.at[("BASE_QM20", "FULL"), "annual_return"], "passed": five.at[("TWO_SLEEVE_SHADOW", "FULL"), "annual_return"] > five.at[("BASE_QM20", "FULL"), "annual_return"]},
            {"gate": "FULL maxDD no worse -1pp", "value": maxdd_delta, "passed": maxdd_delta >= -0.01},
            {"gate": "rolling36 Sharpe lead >=60%", "value": float(rolling["candidate_leads"].mean()), "passed": float(rolling["candidate_leads"].mean()) >= 0.60},
            {"gate": "same-window wins >=7", "value": int(same_windows["candidate_improves"].sum()), "passed": int(same_windows["candidate_improves"].sum()) >= 7},
            {"gate": "official baseline diff <=1e-12", "value": official_difference, "passed": official_difference <= 1e-12},
        ]
    )
    gates = pd.DataFrame(gates)
    yearly_rows = []
    for year in sorted(set(baseline_1.index.year)):
        base_year = baseline_1.loc[baseline_1.index.year == year]
        candidate_year = candidate_1.loc[candidate_1.index.year == year]
        yearly_rows.append({"year": year, "annual_delta": float((1.0 + candidate_year).prod() - (1.0 + base_year).prod()), "sharpe_delta": p1.sharpe(candidate_year) - p1.sharpe(base_year), "top10_delta": p1.top10_summary(candidate_year)["top10_mean_depth"] - p1.top10_summary(base_year)["top10_mean_depth"]})
    yearly = pd.DataFrame(yearly_rows)
    yearly["triple_win"] = yearly[["annual_delta", "sharpe_delta", "top10_delta"]].gt(0.0).all(axis=1)
    bootstrap = p5.paired_triple_bootstrap(p1, p3, baseline_1, candidate_1)
    summary = pd.DataFrame([{"candidate": "TWO_SLEEVE_SHADOW", "all_gates_passed": bool(gates["passed"].all()), "platform_rate": platform_rate, "full_sharpe_delta": one.at[("TWO_SLEEVE_SHADOW", "FULL"), "sharpe"] - one.at[("BASE_QM20", "FULL"), "sharpe"], "full_annual_delta": one.at[("TWO_SLEEVE_SHADOW", "FULL"), "annual_return"] - one.at[("BASE_QM20", "FULL"), "annual_return"], "full_top10_delta": one.at[("TWO_SLEEVE_SHADOW", "FULL"), "top10_mean_depth"] - one.at[("BASE_QM20", "FULL"), "top10_mean_depth"], "full_maxdd_delta": maxdd_delta, "rolling36_lead": float(rolling["candidate_leads"].mean()), "same_window_wins": int(same_windows["candidate_improves"].sum()), "delay_event_days": int(delay_sim["events"]["date"].nunique()), "defensive_trigger_days": int(defensive_triggers["date"].nunique())}])
    comparison.to_csv(HERE / f"{PREFIX}_comparison.csv", index=False)
    gates.to_csv(HERE / f"{PREFIX}_gates.csv", index=False)
    yearly.to_csv(HERE / f"{PREFIX}_yearly.csv", index=False)
    bootstrap.to_csv(HERE / f"{PREFIX}_bootstrap.csv", index=False)
    rolling.to_csv(HERE / f"{PREFIX}_rolling36m.csv", index=False)
    same_windows.to_csv(HERE / f"{PREFIX}_same_window_top10.csv", index=False)
    summary.to_csv(HERE / f"{PREFIX}_summary.csv", index=False)
    print("Summary")
    print(summary.to_string(index=False))
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
