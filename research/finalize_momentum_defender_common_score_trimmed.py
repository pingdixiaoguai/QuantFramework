"""Finalize common-score DRAQM selected on candidate-independent ordinary regimes."""

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
    validate_common_score_policies,
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
    leave_one_year_selection,
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
)
OUTPUT = Path(
    "experiments/20260824_momentum_defender_common_score_trimmed_final_selection"
)
END = pd.Timestamp("2026-08-21")


def _global_unique_returns(
    root: Path,
    local_returns: pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    seen: set[str] = set()
    frames = []
    counts = []
    sources = [
        (experiment.name, pd.read_parquet(root / experiment / "unique_candidate_returns.parquet"))
        for experiment in EXPERIMENTS
    ]
    sources.append(("final_local_one_at_a_time", local_returns))
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
    context = build_gold_override_context(root, end=END.date())
    data = build_exact_execution_data(context)
    profile = FactorProfile("w20_40_25_75", (20, 40), (0.25, 0.75))
    closes = {asset: load_ohlc(asset, END.date())["close"] for asset in ("510300.SH", "518880.SH")}
    features = {
        asset: build_downside_raqm_features(
            closes[asset],
            data.calendar,
            {profile.profile_id: profile},
            {"rolling_504_strict_lag": 504},
            min_history=252,
            volatility_floor_annual=0.08,
            winsor_limit=3.0,
        )
        for asset in closes
    }
    raw_trim = build_extreme_block_mask(
        closes,
        data.calendar,
        ExtremeBlockSpec(normalization_mode="raw_absolute_log_return"),
    )
    volatility_trim = build_extreme_block_mask(
        closes,
        data.calendar,
        ExtremeBlockSpec(normalization_mode="volatility_adjusted"),
    )
    base = {
        "csi_entry": 0.35,
        "csi_recovery": 0.25,
        "csi_ec": 1,
        "csi_rc": 1,
        "gold_entry": 0.50,
        "gold_recovery": 0.05,
        "gold_ec": 3,
        "gold_rc": 1,
        "momentum_hold": 25,
        "defender_hold": 23,
    }
    variants: dict[str, dict[str, float | int]] = {"selected": {}}
    for key, values in {
        "csi_entry": [0.30, 0.40],
        "csi_recovery": [0.20, 0.30],
        "csi_ec": [2],
        "csi_rc": [2],
        "gold_entry": [0.45, 0.55],
        "gold_recovery": [0.00, 0.10],
        "gold_ec": [2, 4],
        "gold_rc": [2, 3],
        "momentum_hold": [20, 30],
        "defender_hold": [21, 22, 24, 25, 26, 27, 30],
    }.items():
        for value in values:
            variants[f"{key}={value}"] = {key: value}
    local_rows = []
    local_return_map = {}
    local_runs = {}
    local_specs = {}
    for name, change in variants.items():
        values = {**base, **change}
        policies = {
            "510300.SH": AssetDRAQMPolicy(
                "510300.SH",
                profile,
                float(values["csi_entry"]),
                float(values["csi_recovery"]),
                int(values["csi_ec"]),
                int(values["csi_rc"]),
            ),
            "518880.SH": AssetDRAQMPolicy(
                "518880.SH",
                profile,
                float(values["gold_entry"]),
                float(values["gold_recovery"]),
                int(values["gold_ec"]),
                int(values["gold_rc"]),
            ),
        }
        validate_common_score_policies(policies)
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
        full = performance(returns)
        raw = performance(returns.loc[raw_trim.selection_mask])
        vol = performance(returns.loc[volatility_trim.selection_mask])
        local_rows.append(
            {
                "variant": name,
                **values,
                **{f"full_{key}": value for key, value in full.items()},
                **{f"raw_trim_{key}": value for key, value in raw.items()},
                **{f"volatility_trim_{key}": value for key, value in vol.items()},
                "defender_entries": run.defender_entries,
                "defender_days": run.defender_days,
            }
        )
        local_return_map[name] = returns
        local_runs[name] = run
        local_specs[name] = spec
    local_table = pd.DataFrame(local_rows).set_index("variant")
    selected_returns = local_return_map["selected"]
    selected_run = local_runs["selected"]
    selected_spec = local_specs["selected"]
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

    local_returns = pd.DataFrame(local_return_map, index=data.calendar)
    global_returns, family_counts = _global_unique_returns(root, local_returns)
    pbo_full, pbo_full_summary = cscv_pbo(
        global_returns, momentum_returns, block_count=12
    )
    pbo_raw, pbo_raw_summary = cscv_pbo(
        global_returns.loc[raw_trim.selection_mask],
        momentum_returns.loc[raw_trim.selection_mask],
        block_count=12,
    )
    pbo_volatility, pbo_volatility_summary = cscv_pbo(
        global_returns.loc[volatility_trim.selection_mask],
        momentum_returns.loc[volatility_trim.selection_mask],
        block_count=12,
    )
    reality_full = yearly_reality_check(
        global_returns, momentum_returns, repetitions=5000, seed=20260824
    )
    reality_raw = yearly_reality_check(
        global_returns.loc[raw_trim.selection_mask],
        momentum_returns.loc[raw_trim.selection_mask],
        repetitions=5000,
        seed=20260824,
    )
    reality_volatility = yearly_reality_check(
        global_returns.loc[volatility_trim.selection_mask],
        momentum_returns.loc[volatility_trim.selection_mask],
        repetitions=5000,
        seed=20260824,
    )
    reality_anchor = yearly_reality_check(
        global_returns, anchor_returns, repetitions=5000, seed=20260824
    )
    walk_momentum = expanding_walk_forward(global_returns, momentum_returns)
    walk_anchor = expanding_walk_forward(global_returns, anchor_returns)
    leave_selection = leave_one_year_selection(global_returns, momentum_returns)
    bootstrap_momentum, bootstrap_momentum_summary = paired_block_bootstrap(
        selected_returns,
        momentum_returns,
        block_size=20,
        repetitions=5000,
        seed=20260824,
    )
    bootstrap_anchor, bootstrap_anchor_summary = paired_block_bootstrap(
        selected_returns,
        anchor_returns,
        block_size=20,
        repetitions=5000,
        seed=20260824,
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
    raw_blocks = _block_performance(
        raw_trim.blocks,
        {
            "selected": selected_returns,
            "momentum": momentum_returns,
            "universal_anchor": anchor_returns,
        },
    )
    volatility_blocks = _block_performance(
        volatility_trim.blocks,
        {
            "selected": selected_returns,
            "momentum": momentum_returns,
            "universal_anchor": anchor_returns,
        },
    )

    local_table.to_csv(output / "one_at_a_time_parameter_neighborhood.csv")
    raw_trim.blocks.to_csv(output / "raw_shock_blocks.csv")
    volatility_trim.blocks.to_csv(output / "volatility_adjusted_shock_blocks.csv")
    raw_blocks.to_csv(output / "raw_block_performance.csv", index=False)
    volatility_blocks.to_csv(output / "volatility_adjusted_block_performance.csv", index=False)
    pbo_full.to_csv(output / "global_cscv_full.csv", index=False)
    pbo_raw.to_csv(output / "global_cscv_raw_trim.csv", index=False)
    pbo_volatility.to_csv(output / "global_cscv_volatility_trim.csv", index=False)
    walk_momentum.to_csv(output / "global_walk_forward_vs_momentum.csv", index=False)
    walk_anchor.to_csv(output / "global_walk_forward_vs_universal_anchor.csv", index=False)
    leave_selection.to_csv(output / "global_leave_one_year_selection.csv", index=False)
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

    selected_daily = selected_run.state.copy()
    selected_daily["raw_ordinary_selection_day"] = raw_trim.selection_mask
    selected_daily["volatility_adjusted_ordinary_selection_day"] = (
        volatility_trim.selection_mask
    )
    selected_daily["return"] = selected_returns
    selected_daily["nav"] = (1.0 + selected_returns).cumprod()
    selected_daily["requested_candidate"] = [
        data.candidates[value] for value in selected_run.requested_target
    ]
    selected_daily["actual_candidate"] = selected_target
    selected_daily["cost_rate_at_open"] = costs
    selected_daily.to_csv(output / "selected_daily.csv")
    selected_daily.to_parquet(output / "selected_daily.parquet")

    full_metrics = performance(selected_returns)
    raw_metrics = performance(selected_returns.loc[raw_trim.selection_mask])
    volatility_metrics = performance(
        selected_returns.loc[volatility_trim.selection_mask]
    )
    momentum_metrics = performance(momentum_returns)
    anchor_metrics = performance(anchor_returns)
    neighbor = local_table.drop(index="selected")
    neighborhood_summary = {
        "neighbors": int(len(neighbor)),
        "full_annualized_q25": float(
            neighbor["full_annualized_return_252"].quantile(0.25)
        ),
        "full_sharpe_q25": float(neighbor["full_sharpe"].quantile(0.25)),
        "raw_trim_annualized_q25": float(
            neighbor["raw_trim_annualized_return_252"].quantile(0.25)
        ),
        "raw_trim_sharpe_q25": float(
            neighbor["raw_trim_sharpe"].quantile(0.25)
        ),
        "volatility_trim_annualized_q25": float(
            neighbor["volatility_trim_annualized_return_252"].quantile(0.25)
        ),
        "volatility_trim_sharpe_q25": float(
            neighbor["volatility_trim_sharpe"].quantile(0.25)
        ),
        "selected_is_full_annualized_max": bool(
            full_metrics["annualized_return_252"]
            >= local_table["full_annualized_return_252"].max() - 1e-15
        ),
        "selected_is_raw_trim_sharpe_max": bool(
            raw_metrics["sharpe"] >= local_table["raw_trim_sharpe"].max() - 1e-15
        ),
        "selected_is_volatility_trim_sharpe_max": bool(
            volatility_metrics["sharpe"]
            >= local_table["volatility_trim_sharpe"].max() - 1e-15
        ),
    }
    three_x = friction.loc[friction["cost_multiplier"].eq(3.0)].iloc[0]
    audit = {
        "strategy_id": "momentum_defender_common_score_trimmed_v1",
        "selection_status": "post_v3_common_profile_dual_trim_local_plateau",
        "common_score": {
            "profile_id": profile.profile_id,
            "horizons": [20, 40],
            "weights": [0.25, 0.75],
            "formula": "downside_regularized_raqm",
            "percentile_history": "rolling_504_strict_lag",
        },
        "selected_policies": {
            "510300.SH": {
                "entry_percentile": 0.35,
                "recovery_percentile": 0.25,
                "entry_confirmation_days": 1,
                "recovery_confirmation_days": 1,
            },
            "518880.SH": {
                "entry_percentile": 0.50,
                "recovery_percentile": 0.05,
                "entry_confirmation_days": 3,
                "recovery_confirmation_days": 1,
            },
        },
        "state_policy": {
            "momentum_lock_days": 25,
            "defender_lock_days": 23,
            "recovery_mode": STICKY_ENTRY_ASSET,
            "other_momentum_assets_gated": False,
        },
        "trim": {
            "selection_mask_candidate_independent": True,
            "primary": "raw_absolute_5d_log_return_top_10pct_fixed_20d_blocks",
            "sensitivity": "volatility_adjusted_absolute_5d_return_top_10pct_fixed_20d_blocks",
            "primary_excluded_sessions": int((~raw_trim.selection_mask).sum()),
            "sensitivity_excluded_sessions": int(
                (~volatility_trim.selection_mask).sum()
            ),
        },
        "metrics": full_metrics,
        "raw_trim_metrics": raw_metrics,
        "volatility_adjusted_trim_metrics": volatility_metrics,
        "momentum_metrics": momentum_metrics,
        "universal_anchor_metrics": anchor_metrics,
        "parameter_neighborhood": neighborhood_summary,
        "search_scope": {
            "candidate_ids": 25101 + 4680 + 4680 + 4146 + len(local_table),
            "family_unique_paths": family_counts,
            "global_unique_paths": int(global_returns.shape[1]),
        },
        "global_cscv_full": pbo_full_summary,
        "global_cscv_raw_trim": pbo_raw_summary,
        "global_cscv_volatility_trim": pbo_volatility_summary,
        "global_reality_full": reality_full,
        "global_reality_raw_trim": reality_raw,
        "global_reality_volatility_trim": reality_volatility,
        "global_reality_vs_universal_anchor": reality_anchor,
        "global_walk_forward_vs_momentum_return_win_rate": float(
            walk_momentum["test_return_delta"].gt(0.0).mean()
        ),
        "global_walk_forward_vs_momentum_sharpe_win_rate": float(
            walk_momentum["test_sharpe_delta"].gt(0.0).mean()
        ),
        "global_walk_forward_vs_anchor_return_win_rate": float(
            walk_anchor["test_return_delta"].gt(0.0).mean()
        ),
        "global_walk_forward_vs_anchor_sharpe_win_rate": float(
            walk_anchor["test_sharpe_delta"].gt(0.0).mean()
        ),
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
            "date": selected_daily.index[-1].date().isoformat(),
            "risk_on": bool(selected_daily.iloc[-1]["risk_on"]),
            "momentum_top1": str(selected_daily.iloc[-1]["momentum_top1_at_open"]),
            "state_reason": str(selected_daily.iloc[-1]["state_reason"]),
            "actual_candidate": str(selected_daily.iloc[-1]["actual_candidate"]),
        },
        "decision": {
            "beats_log_qm_momentum": full_metrics["annualized_return_252"]
            > momentum_metrics["annualized_return_252"]
            and full_metrics["sharpe"] > momentum_metrics["sharpe"],
            "beats_universal_anchor": full_metrics["annualized_return_252"]
            > anchor_metrics["annualized_return_252"]
            and full_metrics["sharpe"] > anchor_metrics["sharpe"],
            "automatic_production_promotion": False,
        },
    }
    (output / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    config = {
        "strategy_id": audit["strategy_id"],
        "status": "research_candidate_not_production",
        "selected_on": "2026-08-24",
        "common_score": audit["common_score"],
        "asset_policies": audit["selected_policies"],
        "state_policy": audit["state_policy"],
        "selection_trim": audit["trim"],
        "checkpoint": {
            **full_metrics,
            "defender_entries": selected_run.defender_entries,
            "defender_days": selected_run.defender_days,
            "sleeve_switches": selected_run.sleeve_switches,
            "candidate_switches": selected_run.candidate_switches,
            "daily_return_sha256_float64_le": audit[
                "daily_return_sha256_float64_le"
            ],
        },
        "parameter_neighborhood": neighborhood_summary,
        "decision": audit["decision"],
    }
    (output / "selected_research_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    generate_standard_report(
        selected_returns,
        momentum_returns,
        "Log-QM Momentum",
        output / "selected_vs_momentum.html",
        config,
    )
    generate_standard_report(
        selected_returns,
        anchor_returns,
        "Universal 510300 DRAQM",
        output / "selected_vs_universal_anchor.html",
        config,
    )

    sources = [
        root / "research/momentum_defender_common_score_trimmed.py",
        root / "research/run_momentum_defender_common_score_trimmed.py",
        root / "research/finalize_momentum_defender_common_score_trimmed.py",
        root / "research/configs/momentum_defender_common_score_trimmed_search.yaml",
        root / "research/configs/momentum_defender_common_score_raw_trim_search.yaml",
        root / "research/configs/momentum_defender_common_score_raw_trim_focused.yaml",
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
