"""Delay switching into a high-risk new QM winner while the incumbent trend is positive."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PHASE5_PATH = HERE / "exp_four_etf_tail_factors_phase5_multi_mechanism.py"
PREFIX = "2026-08-17_four_etf_tail_factors_phase12"


def load_phase5():
    spec = importlib.util.spec_from_file_location("phase5_for_phase12", PHASE5_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {PHASE5_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def simulate_delay(p1, targets, opens, closes, risk, mom20, confidence, params=None):
    dates = closes.index
    gross = pd.Series(0.0, index=dates, dtype=float)
    turnover = pd.Series(0.0, index=dates, dtype=float)
    held = pd.Series(index=dates, dtype="object")
    current = None
    entry_idx = None
    pending = None
    pending_idx = None
    veto_proposal = None
    veto_count = 0
    events = []
    for i, timestamp in enumerate(dates):
        if i > 0:
            previous = dates[i - 1]
            old = current
            if pending_idx == i and pending is not None:
                overnight = 0.0 if old is None else p1.safe_ratio(opens.at[timestamp, old], closes.at[previous, old]) - 1.0
                current = pending
                entry_idx = i
                turnover.at[timestamp] = 1.0 if old is None else (0.0 if old == current else 2.0)
                intraday = p1.safe_ratio(closes.at[timestamp, current], opens.at[timestamp, current]) - 1.0
                gross.at[timestamp] = (1.0 + overnight) * (1.0 + intraday) - 1.0
                pending = None
                pending_idx = None
            elif current is not None:
                gross.at[timestamp] = p1.safe_ratio(closes.at[timestamp, current], closes.at[previous, current]) - 1.0
        held.at[timestamp] = current
        holding_days = i - entry_idx + 1 if current is not None and entry_idx is not None else None
        should_signal = pending is None and (current is None or holding_days is None or holding_days >= p1.REBALANCE_DAYS)
        if should_signal and i + 1 < len(dates):
            proposal = targets.at[timestamp]
            if not isinstance(proposal, str) or proposal == current:
                veto_proposal = None
                veto_count = 0
                continue
            veto = False
            if params is not None and current is not None:
                proposal_risk = risk.at[timestamp, proposal]
                current_risk = risk.at[timestamp, current]
                proposal_is_highest = np.isfinite(proposal_risk) and proposal_risk >= float(risk.loc[timestamp].max()) - 1e-12
                veto = bool(
                    proposal_is_highest
                    and proposal_risk >= params["threshold"]
                    and np.isfinite(current_risk)
                    and proposal_risk - current_risk >= params["minimum_risk_spread"]
                    and np.isfinite(mom20.at[timestamp, current])
                    and mom20.at[timestamp, current] >= params["incumbent_momentum_floor"]
                    and np.isfinite(confidence.at[timestamp])
                    and confidence.at[timestamp] <= params["confidence_cap"]
                )
            if veto:
                if veto_proposal != proposal:
                    veto_proposal = proposal
                    veto_count = 0
                if veto_count < params["delay_days"]:
                    veto_count += 1
                    events.append({"date": timestamp, "current": current, "proposal": proposal, "veto_day": veto_count})
                    continue
            veto_proposal = None
            veto_count = 0
            next_timestamp = dates[i + 1]
            if np.isfinite(opens.at[next_timestamp, proposal]) and np.isfinite(closes.at[next_timestamp, proposal]):
                pending = proposal
                pending_idx = i + 1
    return {
        "gross": gross.loc[p1.EVAL_START : p1.END],
        "turnover": turnover.loc[p1.EVAL_START : p1.END],
        "held": held.loc[p1.EVAL_START : p1.END],
        "events": pd.DataFrame(events),
    }


def main() -> None:
    p5 = load_phase5()
    p2 = p5.load_module("phase2_for_phase12", p5.PHASE2_PATH)
    p3 = p5.load_module("phase3_for_phase12", p5.PHASE3_PATH)
    p1 = p2.load_phase1()
    prices, fields = p1.load_panels()
    factors, _ = p1.build_factors(prices, fields)
    qm20 = factors["qm20"]
    targets = p1.targets_from_score(qm20)
    risk = p5.weighted_risk(factors["risk_ranks"], p5.load_fitted_weights())
    mom20 = prices["close"][p1.CORE].pct_change(20, fill_method=None)
    _, _, confidence = p5.signal_inputs(p1, qm20, prices["close"])
    base_sim = simulate_delay(p1, targets, prices["open"], prices["close"], risk, mom20, confidence)
    baseline_1 = p1.net(base_sim, p1.FEE_MAIN)
    baseline_5 = p1.net(base_sim, p1.FEE_STRESS)
    official_difference = p1.official_baseline_check(baseline_1)
    periods = {"D": (p1.EVAL_START, p1.D_END), "V": (p1.V_START, p1.V_END), "T": (p1.T_START, p1.END), "FULL": (p1.EVAL_START, p1.END)}
    rows = []
    artifacts = {}
    candidate_id = 0
    for threshold in (0.75, 0.80, 0.825):
        for delay_days in (1, 3, 5):
            for floor in (0.0, 0.05):
                for cap in (1.0, 1.5):
                    for spread in (0.0, 0.10):
                        candidate_id += 1
                        name = f"P12_{candidate_id:02d}"
                        params = {"threshold": threshold, "delay_days": delay_days, "incumbent_momentum_floor": floor, "confidence_cap": cap, "minimum_risk_spread": spread}
                        simulation = simulate_delay(p1, targets, prices["open"], prices["close"], risk, mom20, confidence, params)
                        returns_1 = p1.net(simulation, p1.FEE_MAIN)
                        returns_5 = p1.net(simulation, p1.FEE_STRESS)
                        artifacts[name] = (params, simulation, returns_1, returns_5)
                        row = {"candidate": name, **params, "event_days": int(simulation["events"]["date"].nunique() if len(simulation["events"]) else 0)}
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
                            events = simulation["events"]
                            row[f"{period}_event_days"] = int(events.loc[events["date"].between(start, end), "date"].nunique() if len(events) else 0)
                            improvements.extend([deltas["sharpe"] / 0.05, deltas["annual"] / 0.01, deltas["top10"] / 0.005])
                        row["robust_score"] = min(improvements)
                        row["eligible_DV_triple"] = bool(min(improvements) > 1e-6 and row["D_event_days"] >= 3 and row["V_event_days"] >= 3)
                        rows.append(row)
    screen = pd.DataFrame(rows).sort_values(["eligible_DV_triple", "robust_score", "event_days", "delay_days"], ascending=[False, False, True, True])
    eligible = screen.loc[screen["eligible_DV_triple"]]
    selected = str(eligible.iloc[0]["candidate"]) if len(eligible) else None
    screen.to_csv(HERE / f"{PREFIX}_dv_screen.csv", index=False)
    if selected is None:
        pd.DataFrame([{"selected": None, "eligible_count": 0, "official_difference": official_difference}]).to_csv(HERE / f"{PREFIX}_summary.csv", index=False)
        print("No D/V triple candidate")
        print(screen.head(30).to_string(index=False))
        return
    audit_rows = []
    for name in eligible["candidate"]:
        params, simulation, returns_1, returns_5 = artifacts[name]
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
    params, selected_sim, selected_1, selected_5 = artifacts[selected]
    selected_audit = audit.loc[audit["candidate"] == selected].iloc[0]
    rolling = p1.rolling36(selected_1, baseline_1)
    same_windows = p1.same_window_comparison(baseline_1, selected_1)
    bootstrap = p5.paired_triple_bootstrap(p1, p3, baseline_1, selected_1)
    summary = pd.DataFrame([{"selected": selected, "eligible_DV_count": len(eligible), "total_candidates": len(screen), "selected_all_segments_triple": bool(selected_audit["all_segments_triple"]), "any_all_segments_triple": bool(audit["all_segments_triple"].any()), "full_sharpe_delta": selected_audit["FULL_sharpe_delta"], "full_annual_delta": selected_audit["FULL_annual_delta"], "full_top10_delta": selected_audit["FULL_top10_delta"], "rolling36_lead": float(rolling["candidate_leads"].mean()), "same_window_wins": int(same_windows["candidate_improves"].sum()), "official_difference": official_difference}])
    summary.to_csv(HERE / f"{PREFIX}_summary.csv", index=False)
    bootstrap.to_csv(HERE / f"{PREFIX}_bootstrap.csv", index=False)
    selected_sim["events"].to_csv(HERE / f"{PREFIX}_selected_events.csv", index=False)
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
