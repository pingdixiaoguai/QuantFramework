"""Attribute universal 510300 gating versus current-asset DRAQM gating."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.momentum_defender_downside_raqm import (
    DownsideRAQMSpec,
    FactorProfile,
    build_downside_raqm_features,
    build_exact_execution_data,
    exact_candidate_schedule,
    run_downside_raqm_spec,
)
from research.momentum_defender_gold_override import build_gold_override_context
from research.momentum_defender_occam import performance
from research.momentum_volatility import load_ohlc


ROOT = Path(__file__).resolve().parents[1]
UNIVERSAL = ROOT / "experiments/20260824_momentum_defender_downside_raqm_final_selection"
ASSET = ROOT / "experiments/20260825_momentum_defender_dual_regime_final_selection"
OUTPUT = ROOT / "experiments/20260825_universal_vs_asset_gate_attribution"


def _compound(values: pd.Series) -> float:
    return float(np.expm1(np.log1p(values.astype(float)).sum()))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    universal = pd.read_parquet(UNIVERSAL / "selected_daily.parquet").add_prefix("u_")
    asset = pd.read_parquet(ASSET / "selected_daily.parquet").add_prefix("a_")
    daily = universal.join(asset)
    daily["log_excess_asset_vs_universal"] = np.log1p(daily["a_return"]) - np.log1p(
        daily["u_return"]
    )
    daily["category"] = np.select(
        [
            ~daily["u_risk_on"].astype(bool) & daily["a_risk_on"].astype(bool),
            daily["u_risk_on"].astype(bool) & ~daily["a_risk_on"].astype(bool),
            ~daily["u_risk_on"].astype(bool) & ~daily["a_risk_on"].astype(bool),
            daily["u_risk_on"].astype(bool) & daily["a_risk_on"].astype(bool),
        ],
        [
            "universal_defender_asset_momentum",
            "universal_momentum_asset_defender",
            "both_defender",
            "both_momentum",
        ],
        default="other",
    )
    category_rows = []
    for category, sample in daily.groupby("category"):
        category_rows.append(
            {
                "category": category,
                "days": len(sample),
                "asset_return": _compound(sample["a_return"]),
                "universal_return": _compound(sample["u_return"]),
                "log_excess": float(sample["log_excess_asset_vs_universal"].sum()),
            }
        )
    category = pd.DataFrame(category_rows).sort_values("log_excess")
    universal_off = daily.loc[
        daily["category"].eq("universal_defender_asset_momentum")
    ]
    top1 = (
        universal_off.groupby("a_momentum_top1_at_open")
        .agg(
            days=("log_excess_asset_vs_universal", "size"),
            log_excess=("log_excess_asset_vs_universal", "sum"),
            asset_return=("a_return", _compound),
            universal_return=("u_return", _compound),
        )
        .sort_values("log_excess")
    )
    universal_on = daily.loc[
        daily["category"].eq("universal_momentum_asset_defender")
    ]
    trigger = (
        universal_on.groupby("a_trigger_asset")
        .agg(
            days=("log_excess_asset_vs_universal", "size"),
            log_excess=("log_excess_asset_vs_universal", "sum"),
            asset_return=("a_return", _compound),
            universal_return=("u_return", _compound),
        )
        .sort_values("log_excess")
    )
    yearly = daily.groupby(daily.index.year).agg(
        log_excess=("log_excess_asset_vs_universal", "sum"),
        asset_return=("a_return", _compound),
        universal_return=("u_return", _compound),
    )

    events = pd.read_csv(ASSET / "events_vs_universal_anchor.csv")
    event_rows = []
    for _, event in events.iterrows():
        sample = daily.loc[event["start"] : event["end_including_exit"]]
        top_counts = sample["a_momentum_top1_at_open"].value_counts()
        event_rows.append(
            {
                **event.to_dict(),
                "top1_counts": ";".join(
                    f"{key}:{value}" for key, value in top_counts.items()
                ),
                "asset_targets": ";".join(
                    f"{key}:{value}"
                    for key, value in sample["a_actual_candidate"].value_counts().items()
                ),
                "universal_targets": ";".join(
                    f"{key}:{value}"
                    for key, value in sample["u_actual_candidate"].value_counts().items()
                ),
            }
        )
    event_detail = pd.DataFrame(event_rows).sort_values("log_excess")

    context = build_gold_override_context(ROOT, end=pd.Timestamp("2026-08-21").date())
    data = build_exact_execution_data(context)
    defender = context.integrated.result.inputs.defender.loc[
        "2022-08-05":"2022-11-28"
    ]
    assets = ["512890", "159545", "513530", "515080", "510880", "563020", "511260"]
    signatures = (
        defender[[f"policy_target_weight_{asset}" for asset in assets]]
        .round(4)
        .astype(str)
        .agg("|".join, axis=1)
    )
    groups = signatures.ne(signatures.shift()).cumsum()
    defender_rows = []
    for _, sample in defender.groupby(groups):
        row = {
            "start": sample.index.min().date().isoformat(),
            "end": sample.index.max().date().isoformat(),
            "days": len(sample),
            "return": _compound(sample["daily_net_return_if_held"]),
            "selected_asset": str(sample.iloc[0]["selected_asset"]),
            "selection_reason": str(sample.iloc[0]["selection_reason"]),
        }
        for asset_code in assets:
            weight = float(sample.iloc[0][f"policy_target_weight_{asset_code}"])
            if weight > 0.0:
                row[f"weight_{asset_code}"] = weight
        defender_rows.append(row)
    defender_intervals = pd.DataFrame(defender_rows).fillna(0.0)

    # Exact causal diagnostic: preserve universal state except for gold-specific
    # exemptions inferred from the already-frozen asset-specific state.
    base_target = universal["u_actual_candidate"].astype(str).copy()
    top = asset["a_momentum_top1_at_open"].astype(str)
    gold_exemption = (
        ~universal["u_risk_on"].astype(bool)
        & asset["a_risk_on"].astype(bool)
        & top.eq("518880.SH")
    )
    gold_extra_defender = (
        universal["u_risk_on"].astype(bool)
        & top.eq("518880.SH")
        & ~asset["a_risk_on"].astype(bool)
        & asset["a_trigger_asset"].astype(str).eq("518880.SH")
    )
    target_variants = {"universal": base_target.copy()}
    exemption_target = base_target.copy()
    exemption_target.loc[gold_exemption] = "518880.SH"
    target_variants["universal_plus_gold_exemption"] = exemption_target
    bidirectional = exemption_target.copy()
    bidirectional.loc[gold_extra_defender] = "DEFENDER"
    target_variants["universal_plus_gold_bidirectional"] = bidirectional
    counterfactual_rows = []
    for name, target in target_variants.items():
        requested = target.map(data.candidate_index).to_numpy(int)
        values, _, switches = exact_candidate_schedule(data, requested)
        returns = pd.Series(values, index=daily.index)
        counterfactual_rows.append(
            {
                "strategy": name,
                **performance(returns),
                "candidate_switches": switches,
                "gold_exemption_days": int(
                    (target.ne(base_target) & target.eq("518880.SH")).sum()
                ),
                "gold_extra_defender_days": int(
                    (target.ne(exemption_target) & target.eq("DEFENDER")).sum()
                    if name == "universal_plus_gold_bidirectional"
                    else 0
                ),
            }
        )
    counterfactual = pd.DataFrame(counterfactual_rows)

    # Universal parameter neighbors: does the decisive 2022 interval depend on
    # one exact threshold/hold choice?
    profile = FactorProfile("w30_40_25_75", (30, 40), (0.25, 0.75))
    features = build_downside_raqm_features(
        load_ohlc("510300.SH", pd.Timestamp("2026-08-21").date())["close"],
        data.calendar,
        {profile.profile_id: profile},
        {"rolling_504_strict_lag": 504},
        min_history=252,
        volatility_floor_annual=0.08,
        winsor_limit=3.0,
    )
    base = {"entry": 0.55, "exit": 0.20, "mh": 30, "dh": 30, "ec": 3, "rc": 1}
    variants = {"base": {}}
    for key, values in {
        "entry": [0.50, 0.60],
        "exit": [0.10, 0.30],
        "mh": [25],
        "dh": [25],
        "ec": [2, 4],
        "rc": [2],
    }.items():
        for value in values:
            variants[f"{key}={value}"] = {key: value}
    sensitivity_rows = []
    for name, change in variants.items():
        params = {**base, **change}
        run = run_downside_raqm_spec(
            data,
            features,
            DownsideRAQMSpec(
                profile,
                "rolling_504_strict_lag",
                float(params["entry"]),
                float(params["exit"]),
                int(params["mh"]),
                int(params["dh"]),
                int(params["ec"]),
                int(params["rc"]),
            ),
        )
        returns = pd.Series(run.returns, index=data.calendar)
        period = returns.loc["2022-08-05":"2022-11-28"]
        state = run.state.loc[period.index]
        sensitivity_rows.append(
            {
                "variant": name,
                **performance(returns),
                "period_return": _compound(period),
                "period_defender_days": int((~state["risk_on"]).sum()),
            }
        )
    sensitivity = pd.DataFrame(sensitivity_rows)

    # Remove or replace the largest cross-strategy difference events.
    worst = event_detail.sort_values("log_excess")
    counter_rows = []
    for count in range(4):
        replacement = asset["a_return"].copy()
        keep = pd.Series(True, index=daily.index)
        for _, event in worst.head(count).iterrows():
            interval = slice(event["start"], event["end_including_exit"])
            replacement.loc[interval] = universal.loc[interval, "u_return"]
            keep.loc[interval] = False
        replacement_metrics = performance(replacement)
        asset_removed = performance(asset.loc[keep, "a_return"])
        universal_removed = performance(universal.loc[keep, "u_return"])
        counter_rows.append(
            {
                "worst_intervals": count,
                "replacement_annualized": replacement_metrics[
                    "annualized_return_252"
                ],
                "replacement_sharpe": replacement_metrics["sharpe"],
                "asset_removed_annualized": asset_removed["annualized_return_252"],
                "universal_removed_annualized": universal_removed[
                    "annualized_return_252"
                ],
                "asset_removed_sharpe": asset_removed["sharpe"],
                "universal_removed_sharpe": universal_removed["sharpe"],
            }
        )
    removal = pd.DataFrame(counter_rows)

    daily.to_parquet(OUTPUT / "aligned_daily.parquet")
    category.to_csv(OUTPUT / "category_summary.csv", index=False)
    top1.to_csv(OUTPUT / "universal_defender_by_momentum_top1.csv")
    trigger.to_csv(OUTPUT / "universal_momentum_by_asset_trigger.csv")
    yearly.to_csv(OUTPUT / "yearly_comparison.csv")
    event_detail.to_csv(OUTPUT / "difference_event_detail.csv", index=False)
    defender_intervals.to_csv(OUTPUT / "defender_2022_decisive_interval.csv", index=False)
    counterfactual.to_csv(OUTPUT / "gold_exception_counterfactual.csv", index=False)
    sensitivity.to_csv(OUTPUT / "universal_parameter_sensitivity_2022.csv", index=False)
    removal.to_csv(OUTPUT / "difference_event_removal.csv", index=False)

    universal_audit = json.loads((UNIVERSAL / "audit.json").read_text())
    asset_audit = json.loads((ASSET / "audit.json").read_text())
    audit = {
        "total_log_excess_asset_vs_universal": float(
            daily["log_excess_asset_vs_universal"].sum()
        ),
        "category_summary": category.to_dict(orient="records"),
        "universal_defender_by_top1": top1.reset_index().to_dict(orient="records"),
        "universal_momentum_asset_defender_by_trigger": trigger.reset_index().to_dict(
            orient="records"
        ),
        "worst_difference_event": event_detail.iloc[0].to_dict(),
        "worst_event_negative_share": float(
            event_detail.iloc[0]["log_excess"]
            / event_detail.loc[event_detail["log_excess"].lt(0.0), "log_excess"].sum()
        ),
        "counterfactual": counterfactual.to_dict(orient="records"),
        "universal_parameter_sensitivity": sensitivity.to_dict(orient="records"),
        "event_removal": removal.to_dict(orient="records"),
        "universal_existing_robustness": {
            "parameter_stability": universal_audit["parameter_stability"],
            "global_reality_check": universal_audit["global_reality_check"],
            "events": universal_audit["events"],
            "fixed_leave_year_min_annualized_return_252": universal_audit[
                "fixed_leave_year_min_annualized_return_252"
            ],
            "three_x_cost_annualized_return_252": universal_audit[
                "three_x_cost_annualized_return_252"
            ],
        },
        "asset_strategy_metrics": asset_audit["candidate"]["full_metrics"],
        "conclusion": {
            "gold_hypothesis_supported": True,
            "primary_structural_loss": "ungated_chinext_and_nasdaq_while_universal_held_defender",
            "universal_absolute_performance_entirely_one_event": False,
            "universal_advantage_over_asset_strategy_concentrated_in_one_event": True,
            "recommended_mechanism_direction": "universal_anchor_gate_with_gold_exception",
        },
    }
    (OUTPUT / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    manifest_sources = [
        Path(__file__),
        UNIVERSAL / "selected_daily.parquet",
        ASSET / "selected_daily.parquet",
    ]
    manifest = {
        "sources": {str(path.relative_to(ROOT)): _sha(path) for path in manifest_sources},
        "artifacts": {
            path.name: _sha(path)
            for path in OUTPUT.iterdir()
            if path.is_file() and path.name != "experiment_manifest.json"
        },
    }
    (OUTPUT / "experiment_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
