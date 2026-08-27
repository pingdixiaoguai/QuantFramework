"""Robustness audit for the post-hoc SAFE_T75_W50 candidate."""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PHASE2_PATH = HERE / "exp_four_etf_tail_factors_phase2.py"
PREFIX = "2026-08-17_four_etf_tail_factors_phase3"
CANDIDATE = "COND_SAFE_T75_W50"


def load_phase2():
    spec = importlib.util.spec_from_file_location("four_etf_tail_phase2_for_phase3", PHASE2_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {PHASE2_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def base_weights(p1, qm20: pd.DataFrame) -> pd.DataFrame:
    weights = pd.DataFrame(0.0, index=qm20.index, columns=p1.CORE)
    targets = p1.targets_from_score(qm20)
    for timestamp, code in targets.items():
        if isinstance(code, str):
            weights.at[timestamp, code] = 1.0
    return weights


def top10_mean_numpy(returns: np.ndarray) -> float:
    wealth = np.cumprod(1.0 + returns)
    if not len(wealth):
        return 0.0
    peak = float(wealth[0])
    trough = peak
    underwater = False
    depths = []
    for value_raw in wealth[1:]:
        value = float(value_raw)
        if value >= peak:
            if underwater:
                depths.append(trough / peak - 1.0)
                underwater = False
            peak = value
            trough = value
        else:
            if not underwater:
                underwater = True
                trough = value
            elif value < trough:
                trough = value
    if underwater:
        depths.append(trough / peak - 1.0)
    if not depths:
        return 0.0
    return float(np.mean(sorted(depths)[:10]))


def paired_block_bootstrap(p1, baseline: pd.Series, candidate: pd.Series) -> pd.DataFrame:
    joined = pd.concat([baseline.rename("baseline"), candidate.rename("candidate")], axis=1).dropna()
    values = joined.to_numpy(dtype=float)
    n = len(values)
    rng = np.random.default_rng(8172026)
    rows = []
    observed_sharpe = p1.sharpe(candidate) - p1.sharpe(baseline)
    observed_top10 = p1.top10_summary(candidate)["top10_mean_depth"] - p1.top10_summary(baseline)["top10_mean_depth"]
    for block in (20, 60, 120):
        sharpe_deltas = np.empty(2000, dtype=float)
        top10_deltas = np.empty(2000, dtype=float)
        for replication in range(2000):
            starts = rng.integers(0, n - block + 1, size=math.ceil(n / block))
            indices = np.concatenate([np.arange(start, start + block) for start in starts])[:n]
            sample = values[indices]
            baseline_sample = sample[:, 0]
            candidate_sample = sample[:, 1]
            sharpe_deltas[replication] = (
                candidate_sample.mean() / candidate_sample.std(ddof=1)
                - baseline_sample.mean() / baseline_sample.std(ddof=1)
            ) * math.sqrt(252.0)
            top10_deltas[replication] = top10_mean_numpy(candidate_sample) - top10_mean_numpy(baseline_sample)
        rows.append(
            {
                "block_days": block,
                "replicates": len(sharpe_deltas),
                "observed_sharpe_delta": observed_sharpe,
                "probability_sharpe_delta_positive": float((sharpe_deltas > 0.0).mean()),
                "sharpe_q025": float(np.quantile(sharpe_deltas, 0.025)),
                "sharpe_median": float(np.quantile(sharpe_deltas, 0.50)),
                "sharpe_q975": float(np.quantile(sharpe_deltas, 0.975)),
                "observed_top10_mean_improvement": observed_top10,
                "probability_top10_improvement_positive": float((top10_deltas > 0.0).mean()),
                "top10_q025": float(np.quantile(top10_deltas, 0.025)),
                "top10_median": float(np.quantile(top10_deltas, 0.50)),
                "top10_q975": float(np.quantile(top10_deltas, 0.975)),
            }
        )
    return pd.DataFrame(rows)


def audit_row(p1, p2, name, target, baseline_sim, baseline_1, baseline_5, prices):
    simulation = p2.simulate_weighted(p1, target, prices["open"], prices["close"])
    candidate_1 = p1.net(simulation, p1.FEE_MAIN)
    candidate_5 = p1.net(simulation, p1.FEE_STRESS)
    base_summary = p1.top10_summary(baseline_1)
    candidate_summary = p1.top10_summary(candidate_1)
    same_windows = p1.same_window_comparison(baseline_1, candidate_1)
    rolling = p1.rolling36(candidate_1, baseline_1)
    row = {
        "configuration": name,
        "full_sharpe": p1.sharpe(candidate_1),
        "full_sharpe_delta": p1.sharpe(candidate_1) - p1.sharpe(baseline_1),
        "full_5bp_sharpe_delta": p1.sharpe(candidate_5) - p1.sharpe(baseline_5),
        "top10_mean_depth": candidate_summary["top10_mean_depth"],
        "top10_mean_improvement": candidate_summary["top10_mean_depth"] - base_summary["top10_mean_depth"],
        "worst_drawdown_improvement": candidate_summary["top10_worst_depth"] - base_summary["top10_worst_depth"],
        "same_window_wins": int(same_windows["candidate_improves"].sum()),
        "rolling36_lead": float(rolling["candidate_leads"].mean()),
        "annual_turnover": float(simulation["turnover"].sum() / (len(candidate_1) / 252.0)),
    }
    for label, start, end in (
        ("D", p1.EVAL_START, p1.D_END),
        ("V", p1.V_START, p1.V_END),
        ("T", p1.T_START, p1.END),
    ):
        row[f"sharpe_delta_{label}"] = p1.sharpe(candidate_1.loc[start:end]) - p1.sharpe(baseline_1.loc[start:end])
    return row, simulation, candidate_1, candidate_5


def main() -> None:
    p2 = load_phase2()
    p1 = p2.load_phase1()
    prices, fields = p1.load_panels()
    factors, _ = p1.build_factors(prices, fields)
    qm20 = factors["qm20"]
    qm_rank = qm20.rank(axis=1, pct=True, method="average")
    scores = factors["scores"]
    downside = (qm_rank - scores["DOWNSIDE_25"]) / 0.25
    range_risk = (qm_rank - scores["RANGE_25"]) / 0.25
    liquidity = (qm_rank - scores["LIQUIDITY_25"]) / 0.25
    flow_premium = (qm_rank - scores["FLOW_PREMIUM_25"]) / 0.25
    risk_families = {
        "FULL": factors["multifield_risk"],
        "PRICE_ONLY": (downside + range_risk) / 2.0,
        "TUSHARE_ONLY": (liquidity + flow_premium) / 2.0,
        "LIQUIDITY_ONLY": liquidity,
        "FLOW_PREMIUM_ONLY": flow_premium,
    }

    baseline_target = base_weights(p1, qm20)
    baseline_sim = p2.simulate_weighted(p1, baseline_target, prices["open"], prices["close"])
    baseline_1 = p1.net(baseline_sim, p1.FEE_MAIN)
    baseline_5 = p1.net(baseline_sim, p1.FEE_STRESS)
    p1.official_baseline_check(baseline_1)

    configs = {}
    for threshold in (0.70, 0.75, 0.80, 0.85):
        name = f"THRESHOLD_{threshold:.2f}_BUDGET_0.50"
        configs[name], _ = p2.build_targets(p1, qm20, risk_families["FULL"], threshold, "SAFE", 0.50)
    for budget in (0.40, 0.50, 0.60):
        name = f"THRESHOLD_0.75_BUDGET_{budget:.2f}"
        configs[name], _ = p2.build_targets(p1, qm20, risk_families["FULL"], 0.75, "SAFE", budget)
    for family, risk in risk_families.items():
        name = f"ABLATION_{family}_T0.75_W0.50"
        configs[name], _ = p2.build_targets(p1, qm20, risk, 0.75, "SAFE", 0.50)

    invariant_rows = []
    for name, target in configs.items():
        columns_exact = list(target.columns) == p1.CORE
        nonnegative = bool((target >= 0.0).all().all())
        row_sums = target.sum(axis=1)
        fully_invested_when_active = bool(row_sums[row_sums > 0.0].sub(1.0).abs().le(1e-12).all())
        if not (columns_exact and nonnegative and fully_invested_when_active):
            raise RuntimeError(f"target invariant failed for {name}")
        invariant_rows.append(
            {
                "configuration": name,
                "asset_columns": "|".join(target.columns),
                "columns_exactly_fixed_core": columns_exact,
                "weights_nonnegative": nonnegative,
                "fully_invested_when_active": fully_invested_when_active,
            }
        )
    invariants = pd.DataFrame(invariant_rows)

    audit_rows = []
    simulations = {}
    returns_1 = {}
    returns_5 = {}
    for name, target in configs.items():
        row, simulation, candidate_1, candidate_5 = audit_row(
            p1, p2, name, target, baseline_sim, baseline_1, baseline_5, prices
        )
        audit_rows.append(row)
        simulations[name] = simulation
        returns_1[name] = candidate_1
        returns_5[name] = candidate_5
    audit = pd.DataFrame(audit_rows).drop_duplicates("configuration").sort_values(
        ["top10_mean_improvement", "full_sharpe_delta"], ascending=False
    )

    candidate_key = "THRESHOLD_0.75_BUDGET_0.50"
    candidate_1 = returns_1[candidate_key]
    candidate_5 = returns_5[candidate_key]
    candidate_sim = simulations[candidate_key]
    periods = {
        "D": (p1.EVAL_START, p1.D_END),
        "V": (p1.V_START, p1.V_END),
        "T_pseudo_oos": (p1.T_START, p1.END),
        "FULL": (p1.EVAL_START, p1.END),
    }
    comparison_rows = []
    for fee, baseline, candidate in ((1.0, baseline_1, candidate_1), (5.0, baseline_5, candidate_5)):
        for period, (start, end) in periods.items():
            base_segment = baseline.loc[start:end]
            candidate_segment = candidate.loc[start:end]
            comparison_rows.append(
                {
                    "candidate": CANDIDATE,
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
    base_summary = p1.top10_summary(baseline_1)
    candidate_summary = p1.top10_summary(candidate_1)
    same_windows = p1.same_window_comparison(baseline_1, candidate_1)
    rolling = p1.rolling36(candidate_1, baseline_1)
    sharpe_deltas = one.loc[["D", "V", "T_pseudo_oos", "FULL"], "sharpe_delta"]
    mean_improvement = candidate_summary["top10_mean_depth"] - base_summary["top10_mean_depth"]
    worst_improvement = candidate_summary["top10_worst_depth"] - base_summary["top10_worst_depth"]
    same_window_wins = int(same_windows["candidate_improves"].sum())
    neighborhood = audit.loc[
        audit["configuration"].str.startswith("THRESHOLD_")
        & ~audit["configuration"].duplicated()
    ].copy()
    neighborhood_passes = int(
        ((neighborhood["full_sharpe_delta"] >= 0.0) & (neighborhood["top10_mean_improvement"] >= 0.01)).sum()
    )
    neighborhood_required = math.ceil(len(neighborhood) / 2)
    full_ablation = audit.loc[audit["configuration"] == "ABLATION_FULL_T0.75_W0.50"].iloc[0]
    price_ablation = audit.loc[audit["configuration"] == "ABLATION_PRICE_ONLY_T0.75_W0.50"].iloc[0]
    tushare_ablation = audit.loc[audit["configuration"] == "ABLATION_TUSHARE_ONLY_T0.75_W0.50"].iloc[0]
    gates = pd.DataFrame(
        [
            {"gate": "D/V/T/FULL 1bp Sharpe nonnegative", "value": ";".join(f"{key}={value:.4f}" for key, value in sharpe_deltas.items()), "passed": bool((sharpe_deltas >= 0.0).all())},
            {"gate": "FULL 5bp Sharpe nonnegative", "value": float(five.at["FULL", "sharpe_delta"]), "passed": bool(five.at["FULL", "sharpe_delta"] >= 0.0)},
            {"gate": "FULL top10 mean improves >=1pp", "value": mean_improvement, "passed": bool(mean_improvement >= 0.01)},
            {"gate": "FULL worst drawdown improves >=0.5pp", "value": worst_improvement, "passed": bool(worst_improvement >= 0.005)},
            {"gate": "baseline top10 same-window wins >=7", "value": same_window_wins, "passed": bool(same_window_wins >= 7)},
            {"gate": "candidate top10 no worse than baseline worst -2pp", "value": candidate_summary["top10_worst_depth"] - base_summary["top10_worst_depth"], "passed": bool(candidate_summary["top10_worst_depth"] >= base_summary["top10_worst_depth"] - 0.02)},
            {"gate": "rolling36 Sharpe lead >=60%", "value": float(rolling["candidate_leads"].mean()), "passed": bool(rolling["candidate_leads"].mean() >= 0.60)},
            {"gate": "neighborhood half pass Sharpe+Top10", "value": f"{neighborhood_passes}/{len(neighborhood)}", "passed": bool(neighborhood_passes >= neighborhood_required)},
            {"gate": "FULL not dominated by PRICE_ONLY", "value": f"full_sh={full_ablation['full_sharpe_delta']:.4f};price_sh={price_ablation['full_sharpe_delta']:.4f};full_dd={full_ablation['top10_mean_improvement']:.4f};price_dd={price_ablation['top10_mean_improvement']:.4f}", "passed": bool(full_ablation["full_sharpe_delta"] >= price_ablation["full_sharpe_delta"] or full_ablation["top10_mean_improvement"] >= price_ablation["top10_mean_improvement"])},
            {"gate": "Tushare-only creates nonzero effect", "value": f"sharpe_delta={tushare_ablation['full_sharpe_delta']:.4f};top10_delta={tushare_ablation['top10_mean_improvement']:.4f}", "passed": bool(abs(tushare_ablation["full_sharpe_delta"]) > 1e-12 or abs(tushare_ablation["top10_mean_improvement"]) > 1e-12)},
        ]
    )

    top10 = pd.concat(
        [
            p1.drawdown_episodes(baseline_1).head(10).assign(strategy="BASE_QM20"),
            p1.drawdown_episodes(candidate_1).head(10).assign(strategy=CANDIDATE),
        ],
        ignore_index=True,
    )
    bootstrap = paired_block_bootstrap(p1, baseline_1, candidate_1)
    yearly = pd.DataFrame(
        {
            "baseline": (1.0 + baseline_1).groupby(baseline_1.index.year).prod() - 1.0,
            "candidate": (1.0 + candidate_1).groupby(candidate_1.index.year).prod() - 1.0,
        }
    )
    yearly["candidate_minus_baseline"] = yearly["candidate"] - yearly["baseline"]
    yearly.index.name = "year"
    latest_target = configs[candidate_key].loc[p1.END]
    latest = pd.DataFrame(
        {"asset": latest_target.index, "target_weight": latest_target.to_numpy(dtype=float)}
    ).loc[lambda frame: frame["target_weight"] > 0.0]

    audit.to_csv(HERE / f"{PREFIX}_sensitivity_ablation.csv", index=False)
    invariants.to_csv(HERE / f"{PREFIX}_target_invariants.csv", index=False)
    comparison.to_csv(HERE / f"{PREFIX}_comparison.csv", index=False)
    gates.to_csv(HERE / f"{PREFIX}_gates.csv", index=False)
    top10.to_csv(HERE / f"{PREFIX}_top10_episodes.csv", index=False)
    same_windows.to_csv(HERE / f"{PREFIX}_same_window_top10.csv", index=False)
    rolling.to_csv(HERE / f"{PREFIX}_rolling36m.csv", index=False)
    bootstrap.to_csv(HERE / f"{PREFIX}_bootstrap.csv", index=False)
    yearly.to_csv(HERE / f"{PREFIX}_yearly.csv")
    latest.to_csv(HERE / f"{PREFIX}_latest_weights.csv", index=False)

    print("Sensitivity and ablation")
    print(audit.to_string(index=False))
    print("\nCandidate comparison")
    print(comparison.to_string(index=False))
    print("\nGates")
    print(gates.to_string(index=False))
    print("\nBootstrap")
    print(bootstrap.to_string(index=False))
    print("\nLatest target")
    print(latest.to_string(index=False))


if __name__ == "__main__":
    main()
