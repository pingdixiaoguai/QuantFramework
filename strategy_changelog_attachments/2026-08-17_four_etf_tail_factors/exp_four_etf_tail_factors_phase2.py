"""Phase-2 conditional diversification for high-risk QM20 winners."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PHASE1_PATH = HERE / "exp_four_etf_tail_factors.py"
PREFIX = "2026-08-17_four_etf_tail_factors_phase2"


def load_phase1():
    spec = importlib.util.spec_from_file_location("four_etf_tail_phase1_for_phase2", PHASE1_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {PHASE1_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_targets(p1, qm20: pd.DataFrame, risk: pd.DataFrame, threshold: float, leg: str, budget: float):
    targets = pd.DataFrame(0.0, index=qm20.index, columns=p1.CORE)
    trigger_rows = []
    for timestamp in qm20.index:
        qm_row = qm20.loc[timestamp].dropna()
        if qm_row.empty:
            continue
        winner = str(qm_row.idxmax())
        targets.at[timestamp, winner] = 1.0
        risk_row = risk.loc[timestamp].dropna()
        if winner not in risk_row or risk_row.empty:
            continue
        winner_risk = float(risk_row[winner])
        is_worst = winner_risk >= float(risk_row.max()) - 1e-12
        if not is_worst or winner_risk < threshold:
            continue
        alternatives = [code for code in p1.CORE if code != winner and code in risk_row.index]
        if leg == "SAFE":
            diversifier = str(risk_row[alternatives].idxmin())
        elif leg == "QM2":
            alternative_qm = qm_row.reindex(alternatives).dropna()
            if alternative_qm.empty:
                continue
            diversifier = str(alternative_qm.idxmax())
        else:
            raise ValueError(leg)
        targets.at[timestamp, winner] = 1.0 - budget
        targets.at[timestamp, diversifier] = budget
        trigger_rows.append(
            {
                "date": timestamp,
                "threshold": threshold,
                "leg": leg,
                "budget": budget,
                "winner": winner,
                "winner_risk": winner_risk,
                "diversifier": diversifier,
                "diversifier_risk": float(risk_row[diversifier]),
            }
        )
    return targets, pd.DataFrame(trigger_rows)


def simulate_weighted(p1, targets: pd.DataFrame, opens: pd.DataFrame, closes: pd.DataFrame):
    dates = closes.index
    open_values = opens[p1.CORE].to_numpy(dtype=float)
    close_values = closes[p1.CORE].to_numpy(dtype=float)
    target_values = targets[p1.CORE].to_numpy(dtype=float)
    current = np.zeros(len(p1.CORE), dtype=float)
    pending = None
    pending_idx = None
    entry_idx = None
    gross = np.zeros(len(dates), dtype=float)
    turnover = np.zeros(len(dates), dtype=float)

    def weighted_ratio(weights, numerator, denominator):
        ratios = np.ones(len(weights), dtype=float)
        valid = np.isfinite(numerator) & np.isfinite(denominator) & (denominator != 0)
        ratios[valid] = numerator[valid] / denominator[valid]
        return float(np.dot(weights, ratios) - weights.sum())

    for i in range(len(dates)):
        if i > 0:
            if pending_idx == i and pending is not None:
                overnight = weighted_ratio(current, open_values[i], close_values[i - 1])
                old = current.copy()
                current = pending
                entry_idx = i
                turnover[i] = float(np.abs(current - old).sum())
                intraday = weighted_ratio(current, close_values[i], open_values[i])
                gross[i] = (1.0 + overnight) * (1.0 + intraday) - 1.0
                pending = None
                pending_idx = None
            elif current.sum() > 0:
                gross[i] = weighted_ratio(current, close_values[i], close_values[i - 1])
        holding_days = i - entry_idx + 1 if entry_idx is not None and current.sum() > 0 else None
        should_signal = pending is None and (
            current.sum() == 0 or holding_days is None or holding_days >= p1.REBALANCE_DAYS
        )
        if should_signal and i + 1 < len(dates):
            new = target_values[i]
            if new.sum() > 0 and not np.allclose(new, current, atol=1e-12):
                needed = new > 0
                if np.isfinite(open_values[i + 1, needed]).all() and np.isfinite(close_values[i + 1, needed]).all():
                    pending = new.copy()
                    pending_idx = i + 1
    index = dates[(dates >= p1.EVAL_START) & (dates <= p1.END)]
    mask = (dates >= p1.EVAL_START) & (dates <= p1.END)
    return {
        "gross": pd.Series(gross[mask], index=index),
        "turnover": pd.Series(turnover[mask], index=index),
    }


def main() -> None:
    p1 = load_phase1()
    prices, fields = p1.load_panels()
    factors, _ = p1.build_factors(prices, fields)
    qm20 = factors["qm20"]
    risk = factors["multifield_risk"]
    base_weights = pd.DataFrame(0.0, index=qm20.index, columns=p1.CORE)
    base_targets = p1.targets_from_score(qm20)
    for timestamp, code in base_targets.items():
        if isinstance(code, str):
            base_weights.at[timestamp, code] = 1.0

    target_map = {"BASE_QM20": base_weights}
    trigger_frames = []
    for threshold in (0.75, 0.80):
        for leg in ("QM2", "SAFE"):
            for budget in (0.25, 0.50):
                name = f"COND_{leg}_T{int(threshold * 100)}_W{int(budget * 100)}"
                target, triggers = build_targets(p1, qm20, risk, threshold, leg, budget)
                triggers.insert(0, "candidate", name)
                target_map[name] = target
                trigger_frames.append(triggers)
    triggers = pd.concat(trigger_frames, ignore_index=True)
    simulations = {
        name: simulate_weighted(p1, target, prices["open"], prices["close"])
        for name, target in target_map.items()
    }
    returns_1bp = {name: p1.net(sim, p1.FEE_MAIN) for name, sim in simulations.items()}
    returns_5bp = {name: p1.net(sim, p1.FEE_STRESS) for name, sim in simulations.items()}
    baseline = returns_1bp["BASE_QM20"]
    baseline_difference = p1.official_baseline_check(baseline)

    periods = {
        "D": (p1.EVAL_START, p1.D_END),
        "V": (p1.V_START, p1.V_END),
        "T_pseudo_oos": (p1.T_START, p1.END),
        "FULL": (p1.EVAL_START, p1.END),
    }
    metrics_rows = []
    for name, returns in returns_1bp.items():
        for period, (start, end) in periods.items():
            metrics_rows.append(
                p1.metric_row(
                    name,
                    period,
                    p1.segment(returns, start, end),
                    p1.segment(simulations[name]["turnover"], start, end),
                )
            )
    metrics = pd.DataFrame(metrics_rows)
    base_metrics = metrics.loc[metrics["candidate"] == "BASE_QM20"].set_index("period")
    base_dv = p1.segment(baseline, p1.EVAL_START, p1.V_END)
    base_dv_top10 = p1.top10_summary(base_dv)
    screen_rows = []
    for name, returns in returns_1bp.items():
        if name == "BASE_QM20":
            continue
        candidate_metrics = metrics.loc[metrics["candidate"] == name].set_index("period")
        d_delta = float(candidate_metrics.at["D", "sharpe"] - base_metrics.at["D", "sharpe"])
        v_delta = float(candidate_metrics.at["V", "sharpe"] - base_metrics.at["V", "sharpe"])
        candidate_dv_top10 = p1.top10_summary(p1.segment(returns, p1.EVAL_START, p1.V_END))
        screen_rows.append(
            {
                "candidate": name,
                "sharpe_delta_D": d_delta,
                "sharpe_delta_V": v_delta,
                "min_sharpe_delta_DV": min(d_delta, v_delta),
                "dv_top10_mean_improvement": candidate_dv_top10["top10_mean_depth"] - base_dv_top10["top10_mean_depth"],
                "dv_worst_drawdown_improvement": candidate_dv_top10["top10_worst_depth"] - base_dv_top10["top10_worst_depth"],
                "annual_turnover_DV": float(simulations[name]["turnover"].loc[p1.EVAL_START:p1.V_END].sum() / (len(base_dv) / 252.0)),
                "trigger_days_DV": int(triggers.loc[(triggers["candidate"] == name) & (triggers["date"] <= p1.V_END), "date"].nunique()),
                "eligible_DV": d_delta >= 0.0 and v_delta >= 0.0,
            }
        )
    screen = pd.DataFrame(screen_rows).sort_values(
        ["eligible_DV", "dv_top10_mean_improvement", "min_sharpe_delta_DV", "annual_turnover_DV", "candidate"],
        ascending=[False, False, False, True, True],
    )
    eligible = screen.loc[screen["eligible_DV"]]
    selected = str(eligible.iloc[0]["candidate"]) if len(eligible) else None

    comparison_rows = []
    if selected is not None:
        for fee, returns_map in ((1.0, returns_1bp), (5.0, returns_5bp)):
            for period, (start, end) in periods.items():
                base_segment = p1.segment(returns_map["BASE_QM20"], start, end)
                candidate_segment = p1.segment(returns_map[selected], start, end)
                comparison_rows.append(
                    {
                        "selected_candidate": selected,
                        "fee_bps_one_side": fee,
                        "period": period,
                        "baseline_annual_return": p1.annual_return(base_segment),
                        "candidate_annual_return": p1.annual_return(candidate_segment),
                        "baseline_sharpe": p1.sharpe(base_segment),
                        "candidate_sharpe": p1.sharpe(candidate_segment),
                        "sharpe_delta": p1.sharpe(candidate_segment) - p1.sharpe(base_segment),
                        "baseline_max_drawdown": p1.max_drawdown(base_segment),
                        "candidate_max_drawdown": p1.max_drawdown(candidate_segment),
                    }
                )
        comparison = pd.DataFrame(comparison_rows)
        one = comparison.loc[comparison["fee_bps_one_side"] == 1.0].set_index("period")
        five = comparison.loc[comparison["fee_bps_one_side"] == 5.0].set_index("period")
        rolling = p1.rolling36(returns_1bp[selected], baseline)
        same_windows = p1.same_window_comparison(baseline, returns_1bp[selected])
        base_top10 = p1.drawdown_episodes(baseline).head(10).assign(strategy="BASE_QM20")
        candidate_top10 = p1.drawdown_episodes(returns_1bp[selected]).head(10).assign(strategy=selected)
        top10 = pd.concat([base_top10, candidate_top10], ignore_index=True)
        base_summary = p1.top10_summary(baseline)
        candidate_summary = p1.top10_summary(returns_1bp[selected])
        sharpe_deltas = one.loc[["D", "V", "T_pseudo_oos", "FULL"], "sharpe_delta"]
        mean_improvement = candidate_summary["top10_mean_depth"] - base_summary["top10_mean_depth"]
        worst_improvement = candidate_summary["top10_worst_depth"] - base_summary["top10_worst_depth"]
        same_window_wins = int(same_windows["candidate_improves"].sum())
        rolling_lead = float(rolling["candidate_leads"].mean())
        gates = pd.DataFrame(
            [
                {"gate": "D/V/T/FULL 1bp Sharpe nonnegative", "value": ";".join(f"{key}={value:.4f}" for key, value in sharpe_deltas.items()), "passed": bool((sharpe_deltas >= 0.0).all())},
                {"gate": "FULL 5bp Sharpe nonnegative", "value": float(five.at["FULL", "sharpe_delta"]), "passed": bool(five.at["FULL", "sharpe_delta"] >= 0.0)},
                {"gate": "FULL top10 mean improves >=1pp", "value": mean_improvement, "passed": bool(mean_improvement >= 0.01)},
                {"gate": "FULL worst drawdown improves >=0.5pp", "value": worst_improvement, "passed": bool(worst_improvement >= 0.005)},
                {"gate": "baseline top10 same-window wins >=7", "value": same_window_wins, "passed": bool(same_window_wins >= 7)},
                {"gate": "candidate top10 no worse than baseline worst -2pp", "value": candidate_summary["top10_worst_depth"] - base_summary["top10_worst_depth"], "passed": bool(candidate_summary["top10_worst_depth"] >= base_summary["top10_worst_depth"] - 0.02)},
                {"gate": "rolling36 Sharpe lead >=60%", "value": rolling_lead, "passed": bool(rolling_lead >= 0.60)},
                {"gate": "official baseline max daily diff <=1e-12", "value": baseline_difference, "passed": bool(baseline_difference <= 1e-12)},
            ]
        )
        yearly = pd.DataFrame(
            {
                "baseline": (1.0 + baseline).groupby(baseline.index.year).prod() - 1.0,
                "candidate": (1.0 + returns_1bp[selected]).groupby(baseline.index.year).prod() - 1.0,
            }
        )
        yearly["candidate_minus_baseline"] = yearly["candidate"] - yearly["baseline"]
        yearly.index.name = "year"
    else:
        comparison = pd.DataFrame(comparison_rows)
        rolling = pd.DataFrame()
        same_windows = pd.DataFrame()
        top10 = pd.DataFrame()
        yearly = pd.DataFrame()
        gates = pd.DataFrame(
            [{"gate": "at least one D/V Sharpe nonnegative candidate", "value": 0, "passed": False}]
        )

    triggers.to_csv(HERE / f"{PREFIX}_triggers.csv", index=False)
    metrics.to_csv(HERE / f"{PREFIX}_all_metrics_1bp.csv", index=False)
    screen.to_csv(HERE / f"{PREFIX}_dv_screen.csv", index=False)
    comparison.to_csv(HERE / f"{PREFIX}_selected_comparison.csv", index=False)
    gates.to_csv(HERE / f"{PREFIX}_gates.csv", index=False)
    rolling.to_csv(HERE / f"{PREFIX}_rolling36m.csv", index=False)
    same_windows.to_csv(HERE / f"{PREFIX}_same_window_top10.csv", index=False)
    top10.to_csv(HERE / f"{PREFIX}_top10_episodes.csv", index=False)
    yearly.to_csv(HERE / f"{PREFIX}_yearly.csv")

    print(f"official baseline max daily difference: {baseline_difference:.3g}")
    print("\nD/V screen")
    print(screen.to_string(index=False))
    print(f"\nSelected without T: {selected}")
    if len(comparison):
        print("\nComparison")
        print(comparison.to_string(index=False))
    print("\nGates")
    print(gates.to_string(index=False))


if __name__ == "__main__":
    main()
