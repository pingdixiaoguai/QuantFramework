"""Multi-mechanism search for simultaneous Sharpe, CAGR, and Top10 improvement."""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PHASE2_PATH = HERE / "exp_four_etf_tail_factors_phase2.py"
PHASE3_PATH = HERE / "exp_four_etf_tail_factors_phase3.py"
WEIGHTS_PATH = HERE / "2026-08-17_four_etf_tail_factors_phase4_weights.csv"
PREFIX = "2026-08-17_four_etf_tail_factors_phase5"
FEATURES = [
    "downside_lpm20",
    "cvar20",
    "range20",
    "gap_tail20",
    "amihud20",
    "amount_shock20",
    "share_flow20",
    "premium_crowding",
]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_fitted_weights() -> np.ndarray:
    table = pd.read_csv(WEIGHTS_PATH).set_index("field")
    weights = table.loc[FEATURES, "fitted_weight"].to_numpy(dtype=float)
    if (weights < 0.0).any() or not np.isclose(weights.sum(), 1.0, atol=1e-10):
        raise RuntimeError("invalid frozen fitted weights")
    return weights


def weighted_risk(risk_ranks: dict[str, pd.DataFrame], weights: np.ndarray) -> pd.DataFrame:
    result = risk_ranks[FEATURES[0]] * weights[0]
    for name, weight in zip(FEATURES[1:], weights[1:]):
        result = result + risk_ranks[name] * weight
    return result


def base_target(p1, qm20: pd.DataFrame) -> pd.DataFrame:
    target = pd.DataFrame(0.0, index=qm20.index, columns=p1.CORE)
    for timestamp, code in p1.targets_from_score(qm20).items():
        if isinstance(code, str):
            target.at[timestamp, code] = 1.0
    return target


def signal_inputs(p1, qm20: pd.DataFrame, close: pd.DataFrame):
    qm_rank = qm20.rank(axis=1, pct=True, method="average")
    mom20 = close[p1.CORE].pct_change(20, fill_method=None)
    confidence = pd.Series(np.nan, index=qm20.index, dtype=float)
    for timestamp, row in qm20[p1.CORE].iterrows():
        valid = row.dropna().sort_values(ascending=False)
        if len(valid) < 2:
            continue
        scale = float(valid.std(ddof=0))
        if scale > 0.0:
            confidence.at[timestamp] = float((valid.iloc[0] - valid.iloc[1]) / scale)
    return qm_rank, mom20, confidence


def build_candidate_target(
    p1,
    qm20: pd.DataFrame,
    qm_rank: pd.DataFrame,
    mom20: pd.DataFrame,
    risk: pd.DataFrame,
    confidence: pd.Series,
    params: dict[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    target = base_target(p1, qm20)
    triggers = []
    family = str(params["family"])
    threshold = float(params["threshold"])
    cap = float(params["confidence_cap"]) if pd.notna(params.get("confidence_cap")) else np.inf
    for timestamp in qm20.index:
        qm_row = qm20.loc[timestamp, p1.CORE].dropna()
        risk_row = risk.loc[timestamp, p1.CORE].dropna()
        if len(qm_row) < 2 or len(risk_row) < len(p1.CORE):
            continue
        winner = str(qm_row.idxmax())
        winner_risk = float(risk_row[winner])
        if winner_risk < threshold or winner_risk < float(risk_row.max()) - 1e-12:
            continue
        conf = float(confidence.at[timestamp]) if np.isfinite(confidence.at[timestamp]) else np.inf
        if family != "SAFE_POS" and conf > cap:
            continue
        positive = [
            code
            for code in p1.CORE
            if code != winner and np.isfinite(mom20.at[timestamp, code]) and mom20.at[timestamp, code] > 0.0
        ]
        if not positive:
            continue

        if family in {"SAFE_POS", "SAFE_CLOSE", "DYNAMIC_CLOSE"}:
            diversifier = str(risk_row[positive].idxmin())
        elif family in {"RA_CLOSE", "SWITCH_CLOSE_RA"}:
            beta = float(params.get("beta", 0.50))
            utility = qm_rank.loc[timestamp, positive] - beta * risk_row[positive]
            diversifier = str(utility.idxmax())
        elif family == "SWITCH_CLOSE_SAFE":
            diversifier = str(risk_row[positive].idxmin())
        elif family == "TIEBREAK":
            scale = float(qm_row.std(ddof=0))
            close_assets = [winner]
            if scale > 0.0:
                close_assets.extend(
                    code
                    for code in positive
                    if float((qm_row[winner] - qm_row[code]) / scale) <= cap
                )
            diversifier = str(risk_row[close_assets].idxmin())
            if diversifier == winner:
                continue
        else:
            raise ValueError(family)

        if family.startswith("SWITCH_CLOSE") or family == "TIEBREAK":
            budget = 1.0
        elif family == "DYNAMIC_CLOSE":
            maximum = float(params["max_budget"])
            budget = maximum * (winner_risk - threshold) / (1.0 - threshold)
            budget = float(np.clip(budget, 0.0, maximum))
            if budget <= 1e-12:
                continue
        else:
            budget = float(params["budget"])
        target.loc[timestamp] = 0.0
        target.at[timestamp, winner] = 1.0 - budget
        target.at[timestamp, diversifier] = budget
        triggers.append(
            {
                "date": timestamp,
                "family": family,
                "winner": winner,
                "diversifier": diversifier,
                "winner_risk": winner_risk,
                "confidence": conf,
                "budget": budget,
            }
        )
    return target, pd.DataFrame(triggers)


def candidate_parameters() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    candidate_id = 0

    def add(**params):
        nonlocal candidate_id
        candidate_id += 1
        rows.append({"candidate": f"P5_{candidate_id:03d}_{params['family']}", **params})

    for threshold in (0.70, 0.75, 0.80):
        for budget in (0.20, 0.35, 0.50):
            add(family="SAFE_POS", threshold=threshold, budget=budget)
        for cap in (0.50, 1.00, 1.50):
            for budget in (0.20, 0.35, 0.50):
                add(
                    family="SAFE_CLOSE",
                    threshold=threshold,
                    confidence_cap=cap,
                    budget=budget,
                )
            for beta in (0.25, 0.50, 0.75):
                for budget in (0.20, 0.35, 0.50):
                    add(
                        family="RA_CLOSE",
                        threshold=threshold,
                        confidence_cap=cap,
                        beta=beta,
                        budget=budget,
                    )
            for family in ("SWITCH_CLOSE_SAFE", "SWITCH_CLOSE_RA"):
                add(family=family, threshold=threshold, confidence_cap=cap, beta=0.50)
            for maximum in (0.35, 0.50, 0.65):
                add(
                    family="DYNAMIC_CLOSE",
                    threshold=threshold,
                    confidence_cap=cap,
                    max_budget=maximum,
                )
            add(family="TIEBREAK", threshold=threshold, confidence_cap=cap)
    return rows


def period_metrics(p1, returns: pd.Series, turnover: pd.Series, start, end) -> dict[str, float]:
    one = returns.loc[start:end]
    summary = p1.top10_summary(one)
    years = len(one) / 252.0
    return {
        "sharpe": p1.sharpe(one),
        "annual_return": p1.annual_return(one),
        "top10_mean_depth": summary["top10_mean_depth"],
        "max_drawdown": p1.max_drawdown(one),
        "annual_turnover": float(turnover.loc[start:end].sum() / years) if years else 0.0,
    }


def paired_triple_bootstrap(p1, p3, baseline: pd.Series, candidate: pd.Series) -> pd.DataFrame:
    joined = pd.concat([baseline.rename("baseline"), candidate.rename("candidate")], axis=1).dropna()
    values = joined.to_numpy(dtype=float)
    n = len(values)
    rng = np.random.default_rng(8172027)
    rows = []
    for block in (20, 60, 120):
        replications = 2000
        sharpe_delta = np.empty(replications)
        annual_delta = np.empty(replications)
        top10_delta = np.empty(replications)
        for replication in range(replications):
            starts = rng.integers(0, n - block + 1, size=math.ceil(n / block))
            indices = np.concatenate([np.arange(start, start + block) for start in starts])[:n]
            base = values[indices, 0]
            candidate_values = values[indices, 1]
            sharpe_delta[replication] = (
                candidate_values.mean() / candidate_values.std(ddof=1)
                - base.mean() / base.std(ddof=1)
            ) * math.sqrt(252.0)
            annual_delta[replication] = (
                np.prod(1.0 + candidate_values) ** (252.0 / n)
                - np.prod(1.0 + base) ** (252.0 / n)
            )
            top10_delta[replication] = p3.top10_mean_numpy(candidate_values) - p3.top10_mean_numpy(base)
        rows.append(
            {
                "block_days": block,
                "replicates": replications,
                "prob_sharpe_positive": float((sharpe_delta > 0.0).mean()),
                "sharpe_q025": float(np.quantile(sharpe_delta, 0.025)),
                "sharpe_median": float(np.quantile(sharpe_delta, 0.50)),
                "sharpe_q975": float(np.quantile(sharpe_delta, 0.975)),
                "prob_annual_positive": float((annual_delta > 0.0).mean()),
                "annual_q025": float(np.quantile(annual_delta, 0.025)),
                "annual_median": float(np.quantile(annual_delta, 0.50)),
                "annual_q975": float(np.quantile(annual_delta, 0.975)),
                "prob_top10_positive": float((top10_delta > 0.0).mean()),
                "top10_q025": float(np.quantile(top10_delta, 0.025)),
                "top10_median": float(np.quantile(top10_delta, 0.50)),
                "top10_q975": float(np.quantile(top10_delta, 0.975)),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    p2 = load_module("phase2_for_phase5", PHASE2_PATH)
    p3 = load_module("phase3_for_phase5", PHASE3_PATH)
    p1 = p2.load_phase1()
    prices, fields = p1.load_panels()
    factors, _ = p1.build_factors(prices, fields)
    qm20 = factors["qm20"]
    fitted_risk = weighted_risk(factors["risk_ranks"], load_fitted_weights())
    qm_rank, mom20, confidence = signal_inputs(p1, qm20, prices["close"])

    params = candidate_parameters()
    base = base_target(p1, qm20)
    base_sim = p2.simulate_weighted(p1, base, prices["open"], prices["close"])
    baseline_1 = p1.net(base_sim, p1.FEE_MAIN)
    baseline_5 = p1.net(base_sim, p1.FEE_STRESS)
    baseline_difference = p1.official_baseline_check(baseline_1)
    periods = {
        "D": (p1.EVAL_START, p1.D_END),
        "V": (p1.V_START, p1.V_END),
        "T_pseudo_oos": (p1.T_START, p1.END),
        "FULL": (p1.EVAL_START, p1.END),
    }
    base_metrics = {
        period: period_metrics(p1, baseline_1, base_sim["turnover"], start, end)
        for period, (start, end) in periods.items()
    }

    simulations = {}
    returns_1 = {}
    returns_5 = {}
    trigger_frames = []
    screen_rows = []
    for index, candidate_params in enumerate(params, start=1):
        name = str(candidate_params["candidate"])
        target, triggers = build_candidate_target(
            p1, qm20, qm_rank, mom20, fitted_risk, confidence, candidate_params
        )
        simulation = p2.simulate_weighted(p1, target, prices["open"], prices["close"])
        candidate_1 = p1.net(simulation, p1.FEE_MAIN)
        candidate_5 = p1.net(simulation, p1.FEE_STRESS)
        simulations[name] = simulation
        returns_1[name] = candidate_1
        returns_5[name] = candidate_5
        if len(triggers):
            triggers.insert(0, "candidate", name)
            trigger_frames.append(triggers)
        row = dict(candidate_params)
        for period in ("D", "V"):
            start, end = periods[period]
            metrics = period_metrics(p1, candidate_1, simulation["turnover"], start, end)
            for metric in ("sharpe", "annual_return", "top10_mean_depth", "max_drawdown"):
                row[f"{period}_{metric}"] = metrics[metric]
                row[f"{period}_{metric}_delta"] = metrics[metric] - base_metrics[period][metric]
            row[f"{period}_annual_turnover"] = metrics["annual_turnover"]
        improvements = [
            row[f"{period}_{metric}_delta"] / scale
            for period in ("D", "V")
            for metric, scale in (
                ("sharpe", 0.05),
                ("annual_return", 0.01),
                ("top10_mean_depth", 0.005),
            )
        ]
        row["robust_score"] = min(improvements)
        row["eligible_DV_triple"] = bool(min(improvements) >= -1e-10)
        combined_candidate = candidate_1.loc[p1.EVAL_START : p1.V_END]
        combined_baseline = baseline_1.loc[p1.EVAL_START : p1.V_END]
        row["DV_sharpe_delta"] = p1.sharpe(combined_candidate) - p1.sharpe(combined_baseline)
        row["DV_annual_delta"] = p1.annual_return(combined_candidate) - p1.annual_return(combined_baseline)
        row["DV_top10_delta"] = (
            p1.top10_summary(combined_candidate)["top10_mean_depth"]
            - p1.top10_summary(combined_baseline)["top10_mean_depth"]
        )
        row["DV_trigger_days"] = int(
            triggers.loc[triggers["date"] <= p1.V_END, "date"].nunique() if len(triggers) else 0
        )
        screen_rows.append(row)
        if index % 40 == 0:
            print(f"evaluated {index}/{len(params)} candidates", flush=True)

    screen = pd.DataFrame(screen_rows).sort_values(
        [
            "eligible_DV_triple",
            "robust_score",
            "DV_sharpe_delta",
            "DV_annual_delta",
            "DV_top10_delta",
            "D_annual_turnover",
            "candidate",
        ],
        ascending=[False, False, False, False, False, True, True],
    )
    eligible = screen.loc[screen["eligible_DV_triple"]]
    selected = str(eligible.iloc[0]["candidate"]) if len(eligible) else None
    triggers_all = pd.concat(trigger_frames, ignore_index=True) if trigger_frames else pd.DataFrame()

    screen.to_csv(HERE / f"{PREFIX}_dv_screen.csv", index=False)
    if selected is None:
        pd.DataFrame([{"selected": None, "eligible_count": 0}]).to_csv(
            HERE / f"{PREFIX}_summary.csv", index=False
        )
        print("No candidate passed the D/V triple objective.")
        print(screen.head(30).to_string(index=False))
        return

    selected_row = screen.loc[screen["candidate"] == selected].iloc[0]
    selected_1 = returns_1[selected]
    selected_5 = returns_5[selected]
    selected_sim = simulations[selected]
    comparison_rows = []
    for fee, base_returns, candidate_returns in (
        (1.0, baseline_1, selected_1),
        (5.0, baseline_5, selected_5),
    ):
        for period, (start, end) in periods.items():
            for strategy, returns, simulation in (
                ("BASE_QM20", base_returns, base_sim),
                (selected, candidate_returns, selected_sim),
            ):
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
    one = comparison.loc[comparison["fee_bps_one_side"] == 1.0].set_index(["strategy", "period"])
    five = comparison.loc[comparison["fee_bps_one_side"] == 5.0].set_index(["strategy", "period"])

    segment_gates = []
    for period in periods:
        for metric in ("sharpe", "annual_return", "top10_mean_depth"):
            delta = one.at[(selected, period), metric] - one.at[("BASE_QM20", period), metric]
            segment_gates.append(
                {"gate": f"{period} {metric} nonnegative", "value": delta, "passed": bool(delta >= -1e-12)}
            )
    full_sharpe_delta = one.at[(selected, "FULL"), "sharpe"] - one.at[("BASE_QM20", "FULL"), "sharpe"]
    full_annual_delta = one.at[(selected, "FULL"), "annual_return"] - one.at[("BASE_QM20", "FULL"), "annual_return"]
    full_top10_delta = one.at[(selected, "FULL"), "top10_mean_depth"] - one.at[("BASE_QM20", "FULL"), "top10_mean_depth"]
    maxdd_delta = one.at[(selected, "FULL"), "max_drawdown"] - one.at[("BASE_QM20", "FULL"), "max_drawdown"]
    full_5_sharpe = five.at[(selected, "FULL"), "sharpe"] - five.at[("BASE_QM20", "FULL"), "sharpe"]
    full_5_annual = five.at[(selected, "FULL"), "annual_return"] - five.at[("BASE_QM20", "FULL"), "annual_return"]
    rolling = p1.rolling36(selected_1, baseline_1)
    same_windows = p1.same_window_comparison(baseline_1, selected_1)
    rolling_lead = float(rolling["candidate_leads"].mean())
    same_window_wins = int(same_windows["candidate_improves"].sum())
    target_valid = bool(
        list(selected_sim["gross"].index) == list(baseline_1.index)
        and set(base.columns) == set(p1.CORE)
    )
    gates = pd.DataFrame(
        segment_gates
        + [
            {"gate": "FULL Sharpe improves >=0.05", "value": full_sharpe_delta, "passed": full_sharpe_delta >= 0.05},
            {"gate": "FULL annual improves >=1pp", "value": full_annual_delta, "passed": full_annual_delta >= 0.01},
            {"gate": "FULL Top10 improves >=1pp", "value": full_top10_delta, "passed": full_top10_delta >= 0.01},
            {"gate": "FULL maxDD no worse than -1pp", "value": maxdd_delta, "passed": maxdd_delta >= -0.01},
            {"gate": "FULL 5bp Sharpe nonnegative", "value": full_5_sharpe, "passed": full_5_sharpe >= 0.0},
            {"gate": "FULL 5bp annual nonnegative", "value": full_5_annual, "passed": full_5_annual >= 0.0},
            {"gate": "rolling36 Sharpe lead >=60%", "value": rolling_lead, "passed": rolling_lead >= 0.60},
            {"gate": "baseline Top10 same-window wins >=7", "value": same_window_wins, "passed": same_window_wins >= 7},
            {"gate": "official baseline diff <=1e-12", "value": baseline_difference, "passed": baseline_difference <= 1e-12},
            {"gate": "fixed-core target/simulation invariant", "value": target_valid, "passed": target_valid},
        ]
    )

    yearly_rows = []
    for year in sorted(set(baseline_1.index.year)):
        base_year = baseline_1.loc[baseline_1.index.year == year]
        candidate_year = selected_1.loc[selected_1.index.year == year]
        for strategy, returns in (("BASE_QM20", base_year), (selected, candidate_year)):
            yearly_rows.append(
                {
                    "year": year,
                    "strategy": strategy,
                    "annual_return": float((1.0 + returns).prod() - 1.0),
                    "sharpe": p1.sharpe(returns),
                    "top10_mean_depth": p1.top10_summary(returns)["top10_mean_depth"],
                }
            )
    yearly = pd.DataFrame(yearly_rows)
    year_wide = yearly.pivot(index="year", columns="strategy")
    yearly_audit = pd.DataFrame(index=year_wide.index)
    for metric in ("annual_return", "sharpe", "top10_mean_depth"):
        yearly_audit[f"{metric}_delta"] = year_wide[(metric, selected)] - year_wide[(metric, "BASE_QM20")]
    yearly_audit["triple_win"] = (yearly_audit >= 0.0).all(axis=1)
    yearly_audit = yearly_audit.reset_index()

    selected_family = str(selected_row["family"])
    neighborhood = screen.loc[screen["family"] == selected_family].copy()
    for column, step in (("threshold", 0.05), ("confidence_cap", 0.50), ("budget", 0.15), ("beta", 0.25), ("max_budget", 0.15)):
        if column in neighborhood and pd.notna(selected_row.get(column)):
            neighborhood = neighborhood.loc[
                neighborhood[column].isna() | ((neighborhood[column] - float(selected_row[column])).abs() <= step + 1e-12)
            ]
    neighborhood_summary = pd.DataFrame(
        [
            {
                "selected": selected,
                "family": selected_family,
                "neighbor_count": len(neighborhood),
                "neighbor_DV_eligible_rate": float(neighborhood["eligible_DV_triple"].mean()),
                "neighbor_min_robust_score": float(neighborhood["robust_score"].min()),
                "neighbor_median_robust_score": float(neighborhood["robust_score"].median()),
            }
        ]
    )
    bootstrap = paired_triple_bootstrap(p1, p3, baseline_1, selected_1)
    selected_triggers = triggers_all.loc[triggers_all["candidate"] == selected].copy()
    summary = pd.DataFrame(
        [
            {
                "selected": selected,
                "family": selected_family,
                "eligible_DV_count": len(eligible),
                "total_candidates": len(screen),
                "all_final_gates_passed": bool(gates["passed"].all()),
                "full_sharpe_delta": full_sharpe_delta,
                "full_annual_delta": full_annual_delta,
                "full_top10_delta": full_top10_delta,
            }
        ]
    )

    comparison.to_csv(HERE / f"{PREFIX}_selected_comparison.csv", index=False)
    gates.to_csv(HERE / f"{PREFIX}_gates.csv", index=False)
    yearly_audit.to_csv(HERE / f"{PREFIX}_yearly_audit.csv", index=False)
    neighborhood.to_csv(HERE / f"{PREFIX}_selected_neighborhood.csv", index=False)
    neighborhood_summary.to_csv(HERE / f"{PREFIX}_neighborhood_summary.csv", index=False)
    bootstrap.to_csv(HERE / f"{PREFIX}_bootstrap.csv", index=False)
    rolling.to_csv(HERE / f"{PREFIX}_rolling36m.csv", index=False)
    same_windows.to_csv(HERE / f"{PREFIX}_same_window_top10.csv", index=False)
    selected_triggers.to_csv(HERE / f"{PREFIX}_selected_triggers.csv", index=False)
    summary.to_csv(HERE / f"{PREFIX}_summary.csv", index=False)

    print("\nSelected without T:", selected)
    print("\nSelected parameters")
    print(selected_row.to_string())
    print("\nComparison")
    print(comparison.to_string(index=False))
    print("\nGates")
    print(gates.to_string(index=False))
    print("\nYearly audit")
    print(yearly_audit.to_string(index=False))
    print("\nNeighborhood")
    print(neighborhood_summary.to_string(index=False))
    print("\nBootstrap")
    print(bootstrap.to_string(index=False))


if __name__ == "__main__":
    main()
