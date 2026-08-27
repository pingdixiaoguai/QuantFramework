"""Finalize full-history and ordinary-regime searches after local stability audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from research.finalize_momentum_defender_selected_asset_draqm import (
    _fixed_leave_year,
    _period_metrics,
    _trigger_episodes,
)
from research.momentum_defender_common_score_trimmed import (
    ExtremeBlockSpec,
    build_extreme_block_mask,
)
from research.momentum_defender_downside_raqm import (
    DownsideRAQMSpec,
    FactorProfile,
    build_downside_raqm_features,
    build_exact_execution_data,
    exact_candidate_schedule,
    run_downside_raqm_spec,
)
from research.momentum_defender_gold_override import build_gold_override_context
from research.momentum_defender_gold_override_overfit import (
    cscv_pbo,
    expanding_walk_forward,
    paired_block_bootstrap,
    yearly_reality_check,
)
from research.momentum_defender_occam import performance
from research.momentum_defender_selected_asset_draqm import (
    STICKY_ENTRY_ASSET,
    AssetDRAQMPolicy,
    SelectedAssetDRAQMSpec,
    run_selected_asset_draqm_spec,
)
from research.momentum_volatility import load_ohlc
from research.run_momentum_defender_common_score_trimmed import _block_performance
from research.run_momentum_defender_log_qm_robust import (
    _event_stress,
    _friction,
    _selected_cost_schedule,
)
from research.standard_report import generate_standard_report


EXPERIMENTS = (
    Path("experiments/20260824_momentum_defender_selected_asset_draqm"),
    Path("experiments/20260824_momentum_defender_selected_asset_draqm_focused"),
    Path(
        "experiments/20260824_momentum_defender_selected_asset_draqm_final_neighborhood"
    ),
    Path("experiments/20260824_momentum_defender_common_score_trimmed"),
    Path("experiments/20260824_momentum_defender_common_score_raw_trim"),
    Path("experiments/20260824_momentum_defender_common_score_raw_trim_focused"),
    Path("experiments/20260825_momentum_defender_dual_regime_search"),
)
OUTPUT = Path("experiments/20260825_momentum_defender_dual_regime_final_selection")
END = pd.Timestamp("2026-08-21")


def _global_unique_returns(root: Path, local: pd.DataFrame):
    seen = set()
    frames = []
    counts = []
    sources = [
        (experiment.name, pd.read_parquet(root / experiment / "unique_candidate_returns.parquet"))
        for experiment in EXPERIMENTS
    ]
    sources.append(("final_dual_regime_local", local))
    for name, frame in sources:
        keep = []
        for column in frame:
            digest = hashlib.sha1(frame[column].to_numpy(float).tobytes()).hexdigest()
            if digest not in seen:
                seen.add(digest)
                keep.append(str(column))
        selected = frame.loc[:, keep].copy()
        selected.columns = [f"{name}::{column}" for column in keep]
        frames.append(selected)
        counts.append(
            {
                "experiment": name,
                "input_unique_paths": int(frame.shape[1]),
                "new_global_unique_paths": len(keep),
            }
        )
    return pd.concat(frames, axis=1), counts


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output = root / OUTPUT
    output.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    context = build_gold_override_context(root, end=END.date())
    data = build_exact_execution_data(context)
    profiles = {
        "csi_w40": FactorProfile("csi_w40", (40,), (1.0,)),
        "csi_w20_40": FactorProfile("csi_w20_40", (20, 40), (0.25, 0.75)),
        "csi_w30_40": FactorProfile("csi_w30_40", (30, 40), (0.25, 0.75)),
        "gold_w30_40": FactorProfile("gold_w30_40", (30, 40), (0.25, 0.75)),
        "gold_equal": FactorProfile("gold_equal", (30, 40), (0.50, 0.50)),
        "gold_w30": FactorProfile("gold_w30", (30,), (1.0,)),
        "gold_w40": FactorProfile("gold_w40", (40,), (1.0,)),
    }
    closes = {asset: load_ohlc(asset, END.date())["close"] for asset in ("510300.SH", "518880.SH")}
    features = {
        asset: build_downside_raqm_features(
            closes[asset],
            data.calendar,
            profiles,
            {"rolling_504_strict_lag": 504},
            min_history=252,
            volatility_floor_annual=0.08,
            winsor_limit=3.0,
        )
        for asset in closes
    }
    extreme = build_extreme_block_mask(
        closes,
        data.calendar,
        ExtremeBlockSpec(normalization_mode="raw_absolute_log_return"),
    )
    ordinary = extreme.selection_mask.astype(bool)
    base = {
        "csi_profile": "csi_w40",
        "csi_entry": 0.30,
        "csi_recovery": 0.20,
        "csi_ec": 1,
        "csi_rc": 1,
        "gold_profile": "gold_w30_40",
        "gold_entry": 0.20,
        "gold_recovery": 0.05,
        "gold_ec": 5,
        "gold_rc": 1,
        "momentum_hold": 25,
        "defender_hold": 25,
    }
    variants = {"selected": {}}
    for key, values in {
        "csi_profile": ["csi_w20_40", "csi_w30_40"],
        "csi_entry": [0.25, 0.35, 0.40],
        "csi_recovery": [0.10],
        "csi_ec": [2, 3, 4, 5],
        "csi_rc": [3],
        "gold_profile": ["gold_equal", "gold_w30", "gold_w40"],
        "gold_entry": [0.15, 0.25, 0.30],
        "gold_recovery": [0.00, 0.10],
        "gold_ec": [3, 4, 6, 7],
        "gold_rc": [2, 3],
        "momentum_hold": [0, 5, 10, 15, 20, 30],
        "defender_hold": [0, 5, 10, 15, 20, 30],
    }.items():
        for value in values:
            variants[f"{key}={value}"] = {key: value}
    rows = []
    return_map = {}
    run_map = {}
    spec_map = {}
    for name, change in variants.items():
        values = {**base, **change}
        policies = {
            "510300.SH": AssetDRAQMPolicy(
                "510300.SH",
                profiles[str(values["csi_profile"])],
                float(values["csi_entry"]),
                float(values["csi_recovery"]),
                int(values["csi_ec"]),
                int(values["csi_rc"]),
            ),
            "518880.SH": AssetDRAQMPolicy(
                "518880.SH",
                profiles[str(values["gold_profile"])],
                float(values["gold_entry"]),
                float(values["gold_recovery"]),
                int(values["gold_ec"]),
                int(values["gold_rc"]),
            ),
        }
        spec = SelectedAssetDRAQMSpec(
            policies,
            int(values["momentum_hold"]),
            int(values["defender_hold"]),
            STICKY_ENTRY_ASSET,
        )
        run = run_selected_asset_draqm_spec(
            data, context.momentum_target, features, spec
        )
        returns = pd.Series(run.returns, index=data.calendar, name=name)
        rows.append(
            {
                "variant": name,
                **values,
                **{f"full_{key}": value for key, value in performance(returns).items()},
                **{
                    f"ordinary_{key}": value
                    for key, value in performance(returns.loc[ordinary]).items()
                },
                "defender_entries": run.defender_entries,
                "defender_days": run.defender_days,
            }
        )
        return_map[name] = returns
        run_map[name] = run
        spec_map[name] = spec
    local = pd.DataFrame(rows).set_index("variant")
    selected_returns = return_map["selected"]
    selected_run = run_map["selected"]
    selected_spec = spec_map["selected"]
    selected_target = pd.Series(
        [data.candidates[value] for value in selected_run.actual_target],
        index=data.calendar,
    )

    momentum_values, momentum_actual, _ = exact_candidate_schedule(
        data, data.momentum_target
    )
    momentum_returns = pd.Series(
        momentum_values, index=data.calendar, name="log_qm_momentum"
    )
    momentum_target = pd.Series(
        [data.candidates[value] for value in momentum_actual], index=data.calendar
    )
    anchor_profile = FactorProfile(
        "universal_w30_40_25_75", (30, 40), (0.25, 0.75)
    )
    anchor_features = build_downside_raqm_features(
        closes["510300.SH"],
        data.calendar,
        {anchor_profile.profile_id: anchor_profile},
        {"rolling_504_strict_lag": 504},
        min_history=252,
        volatility_floor_annual=0.08,
        winsor_limit=3.0,
    )
    anchor_run = run_downside_raqm_spec(
        data,
        anchor_features,
        DownsideRAQMSpec(
            anchor_profile,
            "rolling_504_strict_lag",
            0.55,
            0.20,
            30,
            30,
            3,
            1,
        ),
    )
    anchor_returns = pd.Series(
        anchor_run.returns, index=data.calendar, name="universal_510300_draqm"
    )
    anchor_target = pd.Series(
        [data.candidates[value] for value in anchor_run.actual_target],
        index=data.calendar,
    )

    global_returns, family_counts = _global_unique_returns(
        root, pd.DataFrame(return_map, index=data.calendar)
    )
    pbo_full, pbo_full_summary = cscv_pbo(
        global_returns, momentum_returns, block_count=12
    )
    pbo_ordinary, pbo_ordinary_summary = cscv_pbo(
        global_returns.loc[ordinary], momentum_returns.loc[ordinary], block_count=12
    )
    reality_full = yearly_reality_check(
        global_returns, momentum_returns, repetitions=5000, seed=20260825
    )
    reality_ordinary = yearly_reality_check(
        global_returns.loc[ordinary],
        momentum_returns.loc[ordinary],
        repetitions=5000,
        seed=20260825,
    )
    reality_anchor = yearly_reality_check(
        global_returns, anchor_returns, repetitions=5000, seed=20260825
    )
    walk_momentum = expanding_walk_forward(global_returns, momentum_returns)
    walk_anchor = expanding_walk_forward(global_returns, anchor_returns)
    bootstrap_momentum, bootstrap_momentum_summary = paired_block_bootstrap(
        selected_returns,
        momentum_returns,
        block_size=20,
        repetitions=5000,
        seed=20260825,
    )
    bootstrap_anchor, bootstrap_anchor_summary = paired_block_bootstrap(
        selected_returns,
        anchor_returns,
        block_size=20,
        repetitions=5000,
        seed=20260825,
    )
    events_momentum, leave_momentum, top_momentum, event_momentum_summary = _event_stress(
        selected_returns,
        momentum_returns,
        selected_target,
        momentum_target,
        [1, 2, 3],
    )
    events_anchor, leave_anchor, top_anchor, event_anchor_summary = _event_stress(
        selected_returns,
        anchor_returns,
        selected_target,
        anchor_target,
        [1, 2, 3],
    )
    trigger_episodes, trigger_summary = _trigger_episodes(
        selected_run.state, selected_returns, momentum_returns
    )
    costs = _selected_cost_schedule(context, data, selected_run.actual_target)
    friction = _friction(selected_returns, costs, [1.0, 2.0, 3.0])
    fixed_leave = _fixed_leave_year(selected_returns)
    periods = _period_metrics(
        selected_returns,
        {"momentum": momentum_returns, "universal_anchor": anchor_returns},
    )
    blocks = _block_performance(
        extreme.blocks,
        {
            "selected": selected_returns,
            "momentum": momentum_returns,
            "universal_anchor": anchor_returns,
        },
    )

    local.to_csv(output / "final_parameter_neighborhood.csv")
    extreme.blocks.to_csv(output / "shock_blocks.csv")
    blocks.to_csv(output / "block_performance.csv", index=False)
    pbo_full.to_csv(output / "global_cscv_full.csv", index=False)
    pbo_ordinary.to_csv(output / "global_cscv_ordinary.csv", index=False)
    walk_momentum.to_csv(output / "global_walk_forward_vs_momentum.csv", index=False)
    walk_anchor.to_csv(output / "global_walk_forward_vs_universal_anchor.csv", index=False)
    bootstrap_momentum.to_csv(output / "bootstrap_vs_momentum.csv", index=False)
    bootstrap_anchor.to_csv(output / "bootstrap_vs_universal_anchor.csv", index=False)
    events_momentum.to_csv(output / "events_vs_momentum.csv", index=False)
    leave_momentum.to_csv(output / "leave_one_event_vs_momentum.csv", index=False)
    top_momentum.to_csv(output / "top_event_deletion_vs_momentum.csv", index=False)
    events_anchor.to_csv(output / "events_vs_universal_anchor.csv", index=False)
    leave_anchor.to_csv(output / "leave_one_event_vs_universal_anchor.csv", index=False)
    top_anchor.to_csv(output / "top_event_deletion_vs_universal_anchor.csv", index=False)
    trigger_episodes.to_csv(output / "defender_episodes_by_trigger_asset.csv", index=False)
    trigger_summary.to_csv(output / "trigger_asset_summary.csv", index=False)
    friction.to_csv(output / "friction_stress.csv", index=False)
    fixed_leave.to_csv(output / "fixed_candidate_leave_one_year.csv", index=False)
    periods.to_csv(output / "period_metrics.csv", index=False)

    daily = selected_run.state.copy()
    daily["ordinary_selection_day"] = ordinary
    daily["return"] = selected_returns
    daily["nav"] = (1.0 + selected_returns).cumprod()
    daily["requested_candidate"] = [
        data.candidates[value] for value in selected_run.requested_target
    ]
    daily["actual_candidate"] = selected_target
    daily["cost_rate_at_open"] = costs
    daily.to_csv(output / "selected_daily.csv")
    daily.to_parquet(output / "selected_daily.parquet")

    full = performance(selected_returns)
    ordinary_metrics = performance(selected_returns.loc[ordinary])
    shock_metrics = performance(selected_returns.loc[~ordinary])
    neighbors = local.drop(index="selected")
    exact_path = local.loc[
        local["full_annualized_return_252"].sub(full["annualized_return_252"]).abs().lt(1e-12)
        & local["full_sharpe"].sub(full["sharpe"]).abs().lt(1e-12)
    ]
    neighborhood = {
        "neighbors": int(len(neighbors)),
        "exact_path_equivalent_variants": int(len(exact_path)),
        "exact_path_equivalent_variant_names": exact_path.index.astype(str).tolist(),
        "full_annualized_q25": float(
            neighbors["full_annualized_return_252"].quantile(0.25)
        ),
        "full_sharpe_q25": float(neighbors["full_sharpe"].quantile(0.25)),
        "ordinary_annualized_q25": float(
            neighbors["ordinary_annualized_return_252"].quantile(0.25)
        ),
        "ordinary_sharpe_q25": float(
            neighbors["ordinary_sharpe"].quantile(0.25)
        ),
        "selected_is_ordinary_annualized_max": bool(
            ordinary_metrics["annualized_return_252"]
            >= local["ordinary_annualized_return_252"].max() - 1e-15
        ),
        "selected_is_ordinary_sharpe_max": bool(
            ordinary_metrics["sharpe"]
            >= local["ordinary_sharpe"].max() - 1e-15
        ),
    }
    three_x = friction.loc[friction["cost_multiplier"].eq(3.0)].iloc[0]
    candidate = {
        "policies": {
            "510300.SH": {
                "profile_id": "w40",
                "horizons": [40],
                "weights": [1.0],
                "entry_percentile": 0.30,
                "recovery_percentile": 0.20,
                "entry_confirmation_days": 1,
                "recovery_confirmation_days": 1,
            },
            "518880.SH": {
                "profile_id": "w30_40_25_75",
                "horizons": [30, 40],
                "weights": [0.25, 0.75],
                "entry_percentile": 0.20,
                "recovery_percentile": 0.05,
                "entry_confirmation_days": 5,
                "recovery_confirmation_days": 1,
            },
        },
        "state_policy": {
            "momentum_lock_days": 25,
            "defender_lock_days": 25,
            "recovery_mode": STICKY_ENTRY_ASSET,
        },
        "full_metrics": full,
        "ordinary_metrics": ordinary_metrics,
        "shock_block_metrics": shock_metrics,
        "defender_entries": selected_run.defender_entries,
        "defender_days": selected_run.defender_days,
        "sleeve_switches": selected_run.sleeve_switches,
        "candidate_switches": selected_run.candidate_switches,
        "parameter_neighborhood": neighborhood,
        "bootstrap_vs_momentum": bootstrap_momentum_summary,
        "bootstrap_vs_universal_anchor": bootstrap_anchor_summary,
        "events_vs_momentum": event_momentum_summary,
        "events_vs_universal_anchor": event_anchor_summary,
        "trigger_asset_summary": trigger_summary.to_dict(orient="records"),
        "fixed_leave_year_min_annualized_return_252": float(
            fixed_leave["annualized_return_252"].min()
        ),
        "fixed_leave_year_min_sharpe": float(fixed_leave["sharpe"].min()),
        "three_x_cost_annualized_return_252": float(
            three_x["annualized_return_252"]
        ),
        "three_x_cost_sharpe": float(three_x["sharpe"]),
        "daily_return_sha256_float64_le": hashlib.sha256(
            selected_returns.to_numpy(dtype="<f8").tobytes()
        ).hexdigest(),
        "latest_state": {
            "date": daily.index[-1].date().isoformat(),
            "risk_on": bool(daily.iloc[-1]["risk_on"]),
            "momentum_top1": str(daily.iloc[-1]["momentum_top1_at_open"]),
            "actual_candidate": str(daily.iloc[-1]["actual_candidate"]),
        },
    }
    audit = {
        "strategy_id": "momentum_defender_dual_regime_stable_v1",
        "selection_result": {
            "including_extremes": "converged_to_stable_candidate",
            "excluding_extremes": "converged_to_stable_candidate",
            "same_candidate": True,
            "rejected_full_sample_peak": {
                "rule": "510300 entry confirmation 3 days",
                "reason": "isolated across consecutive 1-5 day confirmation neighborhood",
                "annualized_return_252": 0.475841,
                "sharpe": 1.620657,
            },
        },
        "lock_grid": [0, 5, 10, 15, 20, 25, 30],
        "candidate": candidate,
        "search_scope": {
            "candidate_ids": 38607 + 16710 + len(local),
            "family_unique_paths": family_counts,
            "global_unique_paths": int(global_returns.shape[1]),
        },
        "global_audit": {
            "cscv_full": pbo_full_summary,
            "cscv_ordinary": pbo_ordinary_summary,
            "reality_full": reality_full,
            "reality_ordinary": reality_ordinary,
            "reality_vs_universal_anchor": reality_anchor,
            "walk_forward_vs_momentum_return_win_rate": float(
                walk_momentum["test_return_delta"].gt(0.0).mean()
            ),
            "walk_forward_vs_momentum_sharpe_win_rate": float(
                walk_momentum["test_sharpe_delta"].gt(0.0).mean()
            ),
            "walk_forward_vs_anchor_return_win_rate": float(
                walk_anchor["test_return_delta"].gt(0.0).mean()
            ),
            "walk_forward_vs_anchor_sharpe_win_rate": float(
                walk_anchor["test_sharpe_delta"].gt(0.0).mean()
            ),
        },
        "benchmarks": {
            "momentum": performance(momentum_returns),
            "universal_anchor": performance(anchor_returns),
        },
        "daily_return_sha256_float64_le": hashlib.sha256(
            selected_returns.to_numpy(dtype="<f8").tobytes()
        ).hexdigest(),
        "decision": {
            "beats_momentum": full["annualized_return_252"]
            > performance(momentum_returns)["annualized_return_252"]
            and full["sharpe"] > performance(momentum_returns)["sharpe"],
            "beats_universal_anchor": full["annualized_return_252"]
            > performance(anchor_returns)["annualized_return_252"]
            and full["sharpe"] > performance(anchor_returns)["sharpe"],
            "automatic_production_promotion": False,
        },
    }
    (output / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for objective in ("including_extremes", "excluding_extremes"):
        config = {
            "strategy_id": f"momentum_defender_dual_regime_{objective}_v1",
            "status": "research_candidate_not_production",
            "selection_objective": objective,
            "converged_with_other_objective": True,
            "policies": candidate["policies"],
            "state_policy": candidate["state_policy"],
            "checkpoint": {
                **full,
                "defender_entries": selected_run.defender_entries,
                "defender_days": selected_run.defender_days,
                "sleeve_switches": selected_run.sleeve_switches,
                "candidate_switches": selected_run.candidate_switches,
                "daily_return_sha256_float64_le": audit[
                    "daily_return_sha256_float64_le"
                ],
            },
            "ordinary_metrics": ordinary_metrics,
            "parameter_neighborhood": neighborhood,
        }
        (output / f"selected_{objective}_config.yaml").write_text(
            yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
    generate_standard_report(
        selected_returns,
        momentum_returns,
        "Log-QM Momentum",
        output / "selected_vs_momentum.html",
        {"strategy_id": audit["strategy_id"]},
    )
    generate_standard_report(
        selected_returns,
        anchor_returns,
        "Universal 510300 DRAQM",
        output / "selected_vs_universal_anchor.html",
        {"strategy_id": audit["strategy_id"]},
    )
    sources = [
        root / "research/momentum_defender_selected_asset_draqm.py",
        root / "research/run_momentum_defender_dual_regime_search.py",
        root / "research/finalize_momentum_defender_dual_regime_search.py",
        root / "research/configs/momentum_defender_dual_regime_research.yaml",
        root / "data/db/510300.SH.parquet",
        root / "data/db/518880.SH.parquet",
    ]
    manifest = {
        "strategy_id": audit["strategy_id"],
        "sources": {str(path.relative_to(root)): _sha(path) for path in sources},
        "artifacts": {
            path.name: _sha(path)
            for path in output.iterdir()
            if path.is_file() and path.name != "experiment_manifest.json"
        },
    }
    (output / "experiment_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
