"""Result-aware Phase-2B diagnostic for the dynamic ETF sleeve.

This is not fresh OOS evidence.  It codifies the single mechanism learned
from Phase 2: equity satellites must not dilute a winning gold core leg.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PHASE2_PATH = HERE / "exp_dynamic_etf_pool_phase2.py"
PREFIX = "2026-08-17_dynamic_etf_pool_phase2b"


def load_phase2():
    spec = importlib.util.spec_from_file_location("dynamic_etf_phase2", PHASE2_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {PHASE2_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def gold_veto(m, targets: np.ndarray) -> tuple[np.ndarray, int]:
    out = targets.copy()
    gold = m.ALL.index("518880.SH")
    satellite_idx = np.arange(len(m.CORE), len(m.ALL))
    mask = (out[:, gold] > 0) & (out[:, satellite_idx].sum(axis=1) > 0)
    out[mask] = 0.0
    out[mask, gold] = 1.0
    return out, int(mask.sum())


def candidate_targets(p2, m, fp, selectors, selector="ALIGN_TOP3", factor="QM20", weight=0.15, rd=5):
    raw = p2.build_target_matrix(
        m,
        fp["scores"][factor],
        selectors[selector],
        fp,
        weight,
        2,
        "beat_best",
        None,
    )
    target, veto_days = gold_veto(m, raw)
    return target, veto_days, rd


def comparison_table(p2, baseline_1, baseline_5, candidate_1, candidate_5) -> pd.DataFrame:
    periods = {
        "D": (p2.D_START, p2.D_END),
        "V": (p2.V_START, p2.V_END),
        "T_pseudo_oos": (p2.T_START, p2.T_END),
        "FULL": (p2.D_START, p2.T_END),
    }
    rows = []
    for fee, baseline, candidate in [(1.0, baseline_1, candidate_1), (5.0, baseline_5, candidate_5)]:
        for period, (start, end) in periods.items():
            b = p2.metric_pack(p2.segment(baseline, start, end))
            c = p2.metric_pack(p2.segment(candidate, start, end))
            rows.append(
                {
                    "fee_bps_one_side": fee,
                    "period": period,
                    "baseline_annual_return": b["annual_return"],
                    "candidate_annual_return": c["annual_return"],
                    "baseline_sharpe": b["sharpe"],
                    "candidate_sharpe": c["sharpe"],
                    "sharpe_delta": c["sharpe"] - b["sharpe"],
                    "baseline_max_drawdown": b["max_drawdown"],
                    "candidate_max_drawdown": c["max_drawdown"],
                }
            )
    return pd.DataFrame(rows)


def paired_block_bootstrap(p2, baseline: pd.Series, candidate: pd.Series) -> pd.DataFrame:
    b = baseline.to_numpy(dtype=float)
    c = candidate.to_numpy(dtype=float)
    n = len(b)
    rng = np.random.default_rng(8172026)
    rows = []
    observed = p2.sharpe(candidate) - p2.sharpe(baseline)
    for block in (20, 60, 120):
        values = []
        for _ in range(3000):
            starts = rng.integers(0, n - block + 1, size=math.ceil(n / block))
            idx = np.concatenate([np.arange(start, start + block) for start in starts])[:n]
            xb, xc = b[idx], c[idx]
            delta = (xc.mean() / xc.std(ddof=1) - xb.mean() / xb.std(ddof=1)) * math.sqrt(252.0)
            values.append(delta)
        values = np.asarray(values)
        rows.append(
            {
                "block_days": block,
                "replicates": len(values),
                "observed_sharpe_delta": observed,
                "bootstrap_mean": float(values.mean()),
                "probability_delta_positive": float((values > 0).mean()),
                "q025": float(np.quantile(values, 0.025)),
                "q05": float(np.quantile(values, 0.05)),
                "median": float(np.quantile(values, 0.5)),
                "q95": float(np.quantile(values, 0.95)),
                "q975": float(np.quantile(values, 0.975)),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    p2 = load_phase2()
    m = p2.load_phase1()
    base = m.load_panels()
    fp = p2.factor_panels(m, base)
    selectors = p2.selector_panels(m, base, fp)

    baseline_target = p2.build_target_matrix(m, fp["scores"]["QM20"], None, fp, 0.0, 1, "beat_best", None)
    baseline_sim = p2.simulate_weighted(m, base, baseline_target, 5, record_positions=True)
    target, veto_signal_days, rd = candidate_targets(p2, m, fp, selectors)
    candidate_sim = p2.simulate_weighted(m, base, target, rd, record_positions=True)
    baseline_1, baseline_5 = p2.net(baseline_sim, 0.0001), p2.net(baseline_sim, 0.0005)
    candidate_1, candidate_5 = p2.net(candidate_sim, 0.0001), p2.net(candidate_sim, 0.0005)

    comparison = comparison_table(p2, baseline_1, baseline_5, candidate_1, candidate_5)
    rolling = p2.rolling36(candidate_1, baseline_1)
    lead_share = float(rolling["candidate_leads"].mean())
    satellite_days = int(candidate_sim["satellite_exposure"].gt(0).sum())
    one = comparison[comparison.fee_bps_one_side == 1.0].set_index("period")
    five = comparison[comparison.fee_bps_one_side == 5.0].set_index("period")
    deltas = one.loc[["D", "V", "T_pseudo_oos"], "sharpe_delta"]
    config = "ALIGN_TOP3|QM20|w0.15|n2_ivol|beat_best|gold_veto|rd5"
    gates = pd.DataFrame(
        [
            {
                "gate": "D/V/T Sharpe nonnegative and at least two >= +0.05",
                "value": f"D={deltas['D']:.4f};V={deltas['V']:.4f};T={deltas['T_pseudo_oos']:.4f}",
                "passed": bool((deltas >= 0).all() and (deltas >= 0.05).sum() >= 2),
            },
            {"gate": "full 1bp Sharpe delta >= +0.05", "value": float(one.loc["FULL", "sharpe_delta"]), "passed": bool(one.loc["FULL", "sharpe_delta"] >= 0.05)},
            {"gate": "full 5bp Sharpe direction positive", "value": float(five.loc["FULL", "sharpe_delta"]), "passed": bool(five.loc["FULL", "sharpe_delta"] > 0)},
            {
                "gate": "full maxDD deterioration <= 3pp",
                "value": float(one.loc["FULL", "candidate_max_drawdown"] - one.loc["FULL", "baseline_max_drawdown"]),
                "passed": bool(one.loc["FULL", "candidate_max_drawdown"] - one.loc["FULL", "baseline_max_drawdown"] >= -0.03),
            },
            {"gate": "rolling36 lead share >= 60%", "value": lead_share, "passed": bool(lead_share >= 0.60)},
            {"gate": "satellite holding days >= 200", "value": satellite_days, "passed": bool(satellite_days >= 200)},
        ]
    )
    gates.insert(0, "posthoc_config", config)

    sensitivity_rows = []
    for selector in p2.SELECTOR_NAMES:
        for weight in (0.15, 0.25, 0.35):
            for hold_days in (5, 10, 20):
                raw = p2.build_target_matrix(
                    m, fp["scores"]["QM20"], selectors[selector], fp, weight, 2, "beat_best", None
                )
                candidate_target, _, _ = (*gold_veto(m, raw), hold_days)
                sim = p2.simulate_weighted(m, base, candidate_target, hold_days)
                r, r5 = p2.net(sim, 0.0001), p2.net(sim, 0.0005)
                segment_deltas = []
                for start, end in [(p2.D_START, p2.D_END), (p2.V_START, p2.V_END), (p2.T_START, p2.T_END)]:
                    segment_deltas.append(
                        p2.sharpe(p2.segment(r, start, end)) - p2.sharpe(p2.segment(baseline_1, start, end))
                    )
                roll = p2.rolling36(r, baseline_1)
                sensitivity_rows.append(
                    {
                        "selector": selector,
                        "satellite_weight": weight,
                        "rebalance_days": hold_days,
                        "delta_D": segment_deltas[0],
                        "delta_V": segment_deltas[1],
                        "delta_T_pseudo_oos": segment_deltas[2],
                        "min_segment_delta": min(segment_deltas),
                        "full_sharpe_delta": p2.sharpe(r) - p2.sharpe(baseline_1),
                        "full_5bp_sharpe_delta": p2.sharpe(r5) - p2.sharpe(baseline_5),
                        "max_drawdown_delta": p2.max_drawdown(r) - p2.max_drawdown(baseline_1),
                        "rolling36_lead_share": float(roll["candidate_leads"].mean()),
                        "satellite_days": int(sim["satellite_exposure"].gt(0).sum()),
                    }
                )
    sensitivity = pd.DataFrame(sensitivity_rows).sort_values(
        ["min_segment_delta", "full_sharpe_delta"], ascending=False
    )

    factor_rows = []
    for selector in ("ALIGN_TOP3", "MEDIAN_TOP3", "RISK_MEDIAN_TOP3"):
        for factor in p2.FACTOR_NAMES:
            raw = p2.build_target_matrix(
                m, fp["scores"][factor], selectors[selector], fp, 0.15, 2, "beat_best", None
            )
            candidate_target, _ = gold_veto(m, raw)
            sim = p2.simulate_weighted(m, base, candidate_target, 5)
            r, r5 = p2.net(sim, 0.0001), p2.net(sim, 0.0005)
            segment_deltas = []
            for start, end in [(p2.D_START, p2.D_END), (p2.V_START, p2.V_END), (p2.T_START, p2.T_END)]:
                segment_deltas.append(
                    p2.sharpe(p2.segment(r, start, end)) - p2.sharpe(p2.segment(baseline_1, start, end))
                )
            roll = p2.rolling36(r, baseline_1)
            factor_rows.append(
                {
                    "selector": selector,
                    "factor": factor,
                    "delta_D": segment_deltas[0],
                    "delta_V": segment_deltas[1],
                    "delta_T_pseudo_oos": segment_deltas[2],
                    "min_segment_delta": min(segment_deltas),
                    "full_sharpe_delta": p2.sharpe(r) - p2.sharpe(baseline_1),
                    "full_5bp_sharpe_delta": p2.sharpe(r5) - p2.sharpe(baseline_5),
                    "max_drawdown_delta": p2.max_drawdown(r) - p2.max_drawdown(baseline_1),
                    "rolling36_lead_share": float(roll["candidate_leads"].mean()),
                    "satellite_days": int(sim["satellite_exposure"].gt(0).sum()),
                }
            )
    factor_ablation = pd.DataFrame(factor_rows).sort_values(
        ["min_segment_delta", "full_sharpe_delta"], ascending=False
    )

    positions = candidate_sim["positions"]
    holding_rows = []
    for code in m.ALL:
        weight = positions[code]
        holding_rows.append(
            {
                "asset": code,
                "name": m.CORE_NAMES.get(code, m.SATELLITES.get(code, (code, ""))[0]),
                "sleeve": "core" if code in m.CORE else m.SATELLITES[code][1],
                "holding_days": int(weight.gt(0).sum()),
                "average_weight_when_held": float(weight[weight.gt(0)].mean()) if weight.gt(0).any() else 0.0,
                "average_weight_all_days": float(weight.mean()),
            }
        )
    holdings = pd.DataFrame(holding_rows)
    yearly = pd.DataFrame(
        {
            "baseline": (1.0 + baseline_1).groupby(baseline_1.index.year).prod() - 1.0,
            "candidate": (1.0 + candidate_1).groupby(candidate_1.index.year).prod() - 1.0,
        }
    )
    yearly["candidate_minus_baseline"] = yearly["candidate"] - yearly["baseline"]
    yearly.index.name = "year"
    bootstrap = paired_block_bootstrap(p2, baseline_1, candidate_1)

    latest = pd.DataFrame(
        {
            "asset": m.ALL,
            "name": [m.CORE_NAMES.get(code, m.SATELLITES.get(code, (code, ""))[0]) for code in m.ALL],
            "latest_signal_target_weight": target[-1],
            "latest_simulated_weight": positions.iloc[-1].to_numpy(),
        }
    )
    latest = latest[(latest.latest_signal_target_weight > 0) | (latest.latest_simulated_weight > 0)]

    comparison.to_csv(HERE / f"{PREFIX}_comparison.csv", index=False)
    gates.to_csv(HERE / f"{PREFIX}_gates.csv", index=False)
    rolling.to_csv(HERE / f"{PREFIX}_rolling36m.csv", index=False)
    sensitivity.to_csv(HERE / f"{PREFIX}_sensitivity.csv", index=False)
    factor_ablation.to_csv(HERE / f"{PREFIX}_factor_ablation.csv", index=False)
    holdings.to_csv(HERE / f"{PREFIX}_holdings.csv", index=False)
    yearly.to_csv(HERE / f"{PREFIX}_yearly.csv")
    bootstrap.to_csv(HERE / f"{PREFIX}_bootstrap.csv", index=False)
    latest.to_csv(HERE / f"{PREFIX}_latest_weights.csv", index=False)

    print(f"posthoc config: {config}")
    print(f"gold-veto signal days: {veto_signal_days}")
    print(comparison.to_string(index=False))
    print("\nGates")
    print(gates.to_string(index=False))
    print("\nBootstrap")
    print(bootstrap.to_string(index=False))


if __name__ == "__main__":
    main()
