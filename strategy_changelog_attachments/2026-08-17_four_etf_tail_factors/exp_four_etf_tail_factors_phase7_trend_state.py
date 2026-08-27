"""Risk intervention gated by the winner's absolute trend state."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PHASE5_PATH = HERE / "exp_four_etf_tail_factors_phase5_multi_mechanism.py"
PREFIX = "2026-08-17_four_etf_tail_factors_phase7"


def load_phase5():
    spec = importlib.util.spec_from_file_location("phase5_for_phase7", PHASE5_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {PHASE5_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parameters() -> list[dict[str, object]]:
    rows = []
    counter = 0

    def add(**kwargs):
        nonlocal counter
        counter += 1
        rows.append({"candidate": f"P7_{counter:03d}", **kwargs})

    for threshold in (0.70, 0.80):
        for mode in ("SAFE", "RA"):
            for budget in (0.15, 0.25, 0.35):
                for momentum_cap in (0.05, 0.10, 0.15, 0.20):
                    add(gate="MOMCAP", threshold=threshold, mode=mode, budget=budget, momentum_cap=momentum_cap)
                for minimum_drawdown in (0.02, 0.05, 0.08):
                    add(gate="DRAWDOWN", threshold=threshold, mode=mode, budget=budget, minimum_drawdown=minimum_drawdown)
            for budget in (0.20, 0.30):
                for momentum_cap in (0.10, 0.15):
                    for minimum_drawdown in (0.02, 0.05):
                        add(
                            gate="BOTH",
                            threshold=threshold,
                            mode=mode,
                            budget=budget,
                            momentum_cap=momentum_cap,
                            minimum_drawdown=minimum_drawdown,
                        )
    return rows


def build_target(p1, p5, qm20, qm_rank, mom20, drawdown60, risk, params):
    target = p5.base_target(p1, qm20)
    triggers = []
    for timestamp in qm20.index:
        qm_row = qm20.loc[timestamp, p1.CORE].dropna()
        risk_row = risk.loc[timestamp, p1.CORE].dropna()
        if len(qm_row) < 2 or len(risk_row) < len(p1.CORE):
            continue
        winner = str(qm_row.idxmax())
        winner_risk = float(risk_row[winner])
        if winner_risk < float(params["threshold"]) or winner_risk < float(risk_row.max()) - 1e-12:
            continue
        winner_momentum = mom20.at[timestamp, winner]
        winner_drawdown = drawdown60.at[timestamp, winner]
        if not np.isfinite(winner_momentum) or not np.isfinite(winner_drawdown):
            continue
        gate = str(params["gate"])
        if gate in {"MOMCAP", "BOTH"} and winner_momentum > float(params["momentum_cap"]):
            continue
        if gate in {"DRAWDOWN", "BOTH"} and winner_drawdown > -float(params["minimum_drawdown"]):
            continue
        positive = [
            code
            for code in p1.CORE
            if code != winner and np.isfinite(mom20.at[timestamp, code]) and mom20.at[timestamp, code] > 0.0
        ]
        if not positive:
            continue
        if params["mode"] == "SAFE":
            diversifier = str(risk_row[positive].idxmin())
        else:
            utility = qm_rank.loc[timestamp, positive] - 0.75 * risk_row[positive]
            diversifier = str(utility.idxmax())
        budget = float(params["budget"])
        target.loc[timestamp] = 0.0
        target.at[timestamp, winner] = 1.0 - budget
        target.at[timestamp, diversifier] = budget
        triggers.append(
            {
                "date": timestamp,
                "winner": winner,
                "diversifier": diversifier,
                "budget": budget,
                "winner_risk": winner_risk,
                "winner_momentum20": winner_momentum,
                "winner_drawdown60": winner_drawdown,
            }
        )
    return target, pd.DataFrame(triggers)


def main() -> None:
    p5 = load_phase5()
    p2 = p5.load_module("phase2_for_phase7", p5.PHASE2_PATH)
    p3 = p5.load_module("phase3_for_phase7", p5.PHASE3_PATH)
    p1 = p2.load_phase1()
    prices, fields = p1.load_panels()
    factors, _ = p1.build_factors(prices, fields)
    qm20 = factors["qm20"]
    risk = p5.weighted_risk(factors["risk_ranks"], p5.load_fitted_weights())
    qm_rank, mom20, _ = p5.signal_inputs(p1, qm20, prices["close"])
    drawdown60 = prices["close"][p1.CORE] / prices["close"][p1.CORE].rolling(60).max() - 1.0
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
    candidates = parameters()
    artifacts = {}
    rows = []
    for index, params in enumerate(candidates, start=1):
        target, triggers = build_target(p1, p5, qm20, qm_rank, mom20, drawdown60, risk, params)
        simulation = p2.simulate_weighted(p1, target, prices["open"], prices["close"])
        returns_1 = p1.net(simulation, p1.FEE_MAIN)
        returns_5 = p1.net(simulation, p1.FEE_STRESS)
        artifacts[params["candidate"]] = (target, triggers, simulation, returns_1, returns_5)
        row = {**params, "trigger_days": int(triggers["date"].nunique() if len(triggers) else 0)}
        improvements = []
        for period in ("D", "V"):
            start, end = periods[period]
            candidate_segment = returns_1.loc[start:end]
            base_segment = baseline_1.loc[start:end]
            deltas = {
                "sharpe": p1.sharpe(candidate_segment) - p1.sharpe(base_segment),
                "annual": p1.annual_return(candidate_segment) - p1.annual_return(base_segment),
                "top10": p1.top10_summary(candidate_segment)["top10_mean_depth"]
                - p1.top10_summary(base_segment)["top10_mean_depth"],
            }
            for name, value in deltas.items():
                row[f"{period}_{name}_delta"] = value
            improvements.extend([deltas["sharpe"] / 0.05, deltas["annual"] / 0.01, deltas["top10"] / 0.005])
        row["robust_score"] = min(improvements)
        row["eligible_DV_triple"] = min(improvements) >= -1e-12
        rows.append(row)
        if index % 40 == 0:
            print(f"evaluated {index}/{len(candidates)}", flush=True)
    screen = pd.DataFrame(rows).sort_values(
        ["eligible_DV_triple", "robust_score", "trigger_days", "budget"],
        ascending=[False, False, True, True],
    )
    eligible = screen.loc[screen["eligible_DV_triple"]]
    selected = str(eligible.iloc[0]["candidate"]) if len(eligible) else None
    screen.to_csv(HERE / f"{PREFIX}_dv_screen.csv", index=False)
    if selected is None:
        pd.DataFrame([{"selected": None, "eligible_count": 0}]).to_csv(HERE / f"{PREFIX}_summary.csv", index=False)
        print("No D/V triple candidate")
        print(screen.head(30).to_string(index=False))
        return

    eligible_audit_rows = []
    for name in eligible["candidate"]:
        _, triggers, simulation, returns_1, returns_5 = artifacts[name]
        row = {"candidate": name}
        all_triple = True
        for period, (start, end) in periods.items():
            candidate_segment = returns_1.loc[start:end]
            base_segment = baseline_1.loc[start:end]
            deltas = {
                "sharpe": p1.sharpe(candidate_segment) - p1.sharpe(base_segment),
                "annual": p1.annual_return(candidate_segment) - p1.annual_return(base_segment),
                "top10": p1.top10_summary(candidate_segment)["top10_mean_depth"]
                - p1.top10_summary(base_segment)["top10_mean_depth"],
            }
            for metric, value in deltas.items():
                row[f"{period}_{metric}_delta"] = value
            all_triple &= min(deltas.values()) >= -1e-12
        row["all_segments_triple"] = all_triple
        row["FULL_5bp_sharpe_delta"] = p1.sharpe(returns_5) - p1.sharpe(baseline_5)
        row["FULL_5bp_annual_delta"] = p1.annual_return(returns_5) - p1.annual_return(baseline_5)
        eligible_audit_rows.append(row)
    eligible_audit = pd.DataFrame(eligible_audit_rows).merge(
        screen, on="candidate", how="left", suffixes=("", "_screen")
    ).sort_values(["all_segments_triple", "FULL_annual_delta", "FULL_sharpe_delta"], ascending=[False, False, False])
    eligible_audit.to_csv(HERE / f"{PREFIX}_eligible_audit.csv", index=False)

    target, triggers, simulation, selected_1, selected_5 = artifacts[selected]
    selected_audit = eligible_audit.loc[eligible_audit["candidate"] == selected].iloc[0]
    rolling = p1.rolling36(selected_1, baseline_1)
    same_windows = p1.same_window_comparison(baseline_1, selected_1)
    maxdd_delta = p1.max_drawdown(selected_1) - p1.max_drawdown(baseline_1)
    active = target.sum(axis=1) > 0.0
    target_valid = bool(
        list(target.columns) == p1.CORE
        and (target.loc[active] >= -1e-12).all().all()
        and np.allclose(target.loc[active].sum(axis=1), 1.0, atol=1e-12)
    )
    gates = []
    for period in periods:
        for metric in ("sharpe", "annual", "top10"):
            value = selected_audit[f"{period}_{metric}_delta"]
            gates.append({"gate": f"{period} {metric} positive", "value": value, "passed": value >= -1e-12})
    gates.extend(
        [
            {"gate": "FULL 5bp Sharpe positive", "value": selected_audit["FULL_5bp_sharpe_delta"], "passed": selected_audit["FULL_5bp_sharpe_delta"] >= 0.0},
            {"gate": "FULL 5bp annual positive", "value": selected_audit["FULL_5bp_annual_delta"], "passed": selected_audit["FULL_5bp_annual_delta"] >= 0.0},
            {"gate": "FULL maxDD no worse -1pp", "value": maxdd_delta, "passed": maxdd_delta >= -0.01},
            {"gate": "rolling36 Sharpe lead >=60%", "value": float(rolling["candidate_leads"].mean()), "passed": float(rolling["candidate_leads"].mean()) >= 0.60},
            {"gate": "baseline Top10 windows wins >=7", "value": int(same_windows["candidate_improves"].sum()), "passed": int(same_windows["candidate_improves"].sum()) >= 7},
            {"gate": "target invariants", "value": target_valid, "passed": target_valid},
        ]
    )
    gates = pd.DataFrame(gates)
    bootstrap = p5.paired_triple_bootstrap(p1, p3, baseline_1, selected_1)
    comparison_rows = []
    for fee, base_returns, candidate_returns in ((1.0, baseline_1, selected_1), (5.0, baseline_5, selected_5)):
        for period, (start, end) in periods.items():
            for strategy, returns in (("BASE_QM20", base_returns), (selected, candidate_returns)):
                one = returns.loc[start:end]
                comparison_rows.append(
                    {
                        "fee_bps_one_side": fee,
                        "period": period,
                        "strategy": strategy,
                        "annual_return": p1.annual_return(one),
                        "sharpe": p1.sharpe(one),
                        "max_drawdown": p1.max_drawdown(one),
                        **p1.top10_summary(one),
                    }
                )
    comparison = pd.DataFrame(comparison_rows)
    summary = pd.DataFrame(
        [{"selected": selected, "eligible_DV_count": len(eligible), "total_candidates": len(screen), "all_gates_passed": bool(gates["passed"].all())}]
    )
    comparison.to_csv(HERE / f"{PREFIX}_selected_comparison.csv", index=False)
    gates.to_csv(HERE / f"{PREFIX}_gates.csv", index=False)
    bootstrap.to_csv(HERE / f"{PREFIX}_bootstrap.csv", index=False)
    rolling.to_csv(HERE / f"{PREFIX}_rolling36m.csv", index=False)
    same_windows.to_csv(HERE / f"{PREFIX}_same_window_top10.csv", index=False)
    triggers.to_csv(HERE / f"{PREFIX}_selected_triggers.csv", index=False)
    summary.to_csv(HERE / f"{PREFIX}_summary.csv", index=False)
    print("Selected without T", selected)
    print(screen.loc[screen["candidate"] == selected].to_string(index=False))
    print("\nEligible audit")
    print(eligible_audit.to_string(index=False))
    print("\nComparison")
    print(comparison.to_string(index=False))
    print("\nGates")
    print(gates.to_string(index=False))
    print("\nBootstrap")
    print(bootstrap.to_string(index=False))


if __name__ == "__main__":
    main()
