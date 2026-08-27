"""Finalize the forced 510300/518880 selected-asset DRAQM research."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

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
)
OUTPUT = Path(
    "experiments/20260824_momentum_defender_selected_asset_draqm_final_selection"
)
END = pd.Timestamp("2026-08-21")


def _global_unique_returns(root: Path) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    seen: set[str] = set()
    frames = []
    counts = []
    for experiment in EXPERIMENTS:
        frame = pd.read_parquet(root / experiment / "unique_candidate_returns.parquet")
        keep = []
        for column in frame:
            digest = hashlib.sha1(frame[column].to_numpy(float).tobytes()).hexdigest()
            if digest not in seen:
                seen.add(digest)
                keep.append(str(column))
        selected = frame.loc[:, keep].copy()
        selected.columns = [f"{experiment.name}::{column}" for column in keep]
        frames.append(selected)
        counts.append(
            {
                "experiment": experiment.name,
                "input_unique_paths": int(frame.shape[1]),
                "new_global_unique_paths": len(keep),
            }
        )
    return pd.concat(frames, axis=1), counts


def _fixed_leave_year(returns: pd.Series) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "removed_year": int(year),
                **performance(returns.loc[returns.index.year != year]),
            }
            for year in sorted(returns.index.year.unique())
        ]
    )


def _period_metrics(returns: pd.Series, benchmarks: dict[str, pd.Series]) -> pd.DataFrame:
    periods = {
        "development": ("2019-01-18", "2022-12-30"),
        "validation": ("2023-01-01", "2024-12-31"),
        "recent": ("2025-01-01", "2026-08-21"),
        "full": ("2019-01-18", "2026-08-21"),
    }
    rows = []
    for period, (start, end) in periods.items():
        row = {
            "period": period,
            **{
                f"selected_{key}": value
                for key, value in performance(returns.loc[start:end]).items()
            },
        }
        for name, benchmark in benchmarks.items():
            row.update(
                {
                    f"{name}_{key}": value
                    for key, value in performance(benchmark.loc[start:end]).items()
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _trigger_episodes(
    state: pd.DataFrame,
    returns: pd.Series,
    momentum: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    risk_off = ~state["risk_on"].astype(bool)
    groups = risk_off.ne(risk_off.shift()).cumsum()
    rows = []
    calendar = returns.index
    for event, (_, sample) in enumerate(state.loc[risk_off].groupby(groups.loc[risk_off]), 1):
        start = calendar.get_loc(sample.index.min())
        end = min(calendar.get_loc(sample.index.max()) + 1, len(calendar) - 1)
        interval = calendar[start : end + 1]
        candidate_total = float((1.0 + returns.loc[interval]).prod() - 1.0)
        momentum_total = float((1.0 + momentum.loc[interval]).prod() - 1.0)
        trigger = str(sample.iloc[0]["trigger_asset"])
        rows.append(
            {
                "event": event,
                "trigger_asset": trigger,
                "start": interval.min().date().isoformat(),
                "end_including_exit": interval.max().date().isoformat(),
                "observations": len(interval),
                "selected_return": candidate_total,
                "momentum_return": momentum_total,
                "log_excess": float(
                    np.log1p(candidate_total) - np.log1p(momentum_total)
                ),
            }
        )
    episodes = pd.DataFrame(rows)
    summary = (
        episodes.groupby("trigger_asset")
        .agg(
            events=("event", "count"),
            positive_events=("log_excess", lambda values: int((values > 0.0).sum())),
            negative_events=("log_excess", lambda values: int((values < 0.0).sum())),
            total_log_excess=("log_excess", "sum"),
            median_log_excess=("log_excess", "median"),
            max_log_excess=("log_excess", "max"),
            min_log_excess=("log_excess", "min"),
        )
        .reset_index()
    )
    return episodes, summary


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output = root / OUTPUT
    output.mkdir(parents=True, exist_ok=True)
    context = build_gold_override_context(root, end=END.date())
    data = build_exact_execution_data(context)
    csi_profile = FactorProfile("w30_40_25_75", (30, 40), (0.25, 0.75))
    gold_profile = FactorProfile("w20_40_25_75", (20, 40), (0.25, 0.75))
    csi_policy = AssetDRAQMPolicy(
        "510300.SH", csi_profile, 0.35, 0.25, 1, 1
    )
    gold_policy = AssetDRAQMPolicy(
        "518880.SH", gold_profile, 0.45, 0.00, 5, 1
    )
    selected_spec = SelectedAssetDRAQMSpec(
        {"510300.SH": csi_policy, "518880.SH": gold_policy},
        20,
        23,
        STICKY_ENTRY_ASSET,
    )
    features = {
        "510300.SH": build_downside_raqm_features(
            load_ohlc("510300.SH", END.date())["close"],
            data.calendar,
            {csi_profile.profile_id: csi_profile},
            {"rolling_504_strict_lag": 504},
            min_history=252,
            volatility_floor_annual=0.08,
            winsor_limit=3.0,
        ),
        "518880.SH": build_downside_raqm_features(
            load_ohlc("518880.SH", END.date())["close"],
            data.calendar,
            {gold_profile.profile_id: gold_profile},
            {"rolling_504_strict_lag": 504},
            min_history=252,
            volatility_floor_annual=0.08,
            winsor_limit=3.0,
        ),
    }
    selected_run = run_selected_asset_draqm_spec(
        data, context.momentum_target, features, selected_spec
    )
    selected_returns = pd.Series(
        selected_run.returns, index=data.calendar, name=selected_spec.candidate_id()
    )
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
    anchor_run = run_downside_raqm_spec(
        data,
        features["510300.SH"],
        DownsideRAQMSpec(
            csi_profile,
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

    ablations = {}
    ablation_runs = {}
    for name, policies in {
        "both_selected_assets": {
            "510300.SH": csi_policy,
            "518880.SH": gold_policy,
        },
        "510300_only": {"510300.SH": csi_policy, "518880.SH": None},
        "518880_only": {"510300.SH": None, "518880.SH": gold_policy},
    }.items():
        run = run_selected_asset_draqm_spec(
            data,
            context.momentum_target,
            features,
            SelectedAssetDRAQMSpec(policies, 20, 23, STICKY_ENTRY_ASSET),
        )
        returns = pd.Series(run.returns, index=data.calendar)
        ablations[name] = {**performance(returns), "defender_entries": run.defender_entries, "defender_days": run.defender_days}
        ablation_runs[name] = returns

    global_returns, family_counts = _global_unique_returns(root)
    pbo_momentum, pbo_momentum_summary = cscv_pbo(
        global_returns, momentum_returns, block_count=12
    )
    pbo_anchor, pbo_anchor_summary = cscv_pbo(
        global_returns, anchor_returns, block_count=12
    )
    reality_momentum = yearly_reality_check(
        global_returns, momentum_returns, repetitions=5000, seed=20260824
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
    events_momentum, leave_event_momentum, top_momentum, event_momentum_summary = _event_stress(
        selected_returns,
        momentum_returns,
        selected_target,
        momentum_target,
        [1, 2, 3],
    )
    events_anchor, leave_event_anchor, top_anchor, event_anchor_summary = _event_stress(
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

    pbo_momentum.to_csv(output / "global_cscv_vs_momentum.csv", index=False)
    pbo_anchor.to_csv(output / "global_cscv_vs_universal_anchor.csv", index=False)
    walk_momentum.to_csv(output / "global_walk_forward_vs_momentum.csv", index=False)
    walk_anchor.to_csv(output / "global_walk_forward_vs_universal_anchor.csv", index=False)
    leave_selection.to_csv(output / "global_leave_one_year_selection.csv", index=False)
    bootstrap_momentum.to_csv(output / "bootstrap_vs_momentum.csv", index=False)
    bootstrap_anchor.to_csv(output / "bootstrap_vs_universal_anchor.csv", index=False)
    events_momentum.to_csv(output / "event_attribution_vs_momentum.csv", index=False)
    leave_event_momentum.to_csv(output / "leave_one_event_vs_momentum.csv", index=False)
    top_momentum.to_csv(output / "top_positive_event_deletion_vs_momentum.csv", index=False)
    events_anchor.to_csv(output / "event_attribution_vs_universal_anchor.csv", index=False)
    leave_event_anchor.to_csv(output / "leave_one_event_vs_universal_anchor.csv", index=False)
    top_anchor.to_csv(output / "top_positive_event_deletion_vs_universal_anchor.csv", index=False)
    trigger_episodes.to_csv(output / "defender_episodes_by_trigger_asset.csv", index=False)
    trigger_summary.to_csv(output / "trigger_asset_summary.csv", index=False)
    friction.to_csv(output / "friction_stress.csv", index=False)
    fixed_leave.to_csv(output / "fixed_candidate_leave_one_year.csv", index=False)
    periods.to_csv(output / "period_metrics.csv", index=False)
    pd.DataFrame(
        [{"strategy": name, **metrics} for name, metrics in ablations.items()]
    ).to_csv(output / "asset_policy_ablation.csv", index=False)

    selected_daily = selected_run.state.copy()
    selected_daily["return"] = selected_returns
    selected_daily["nav"] = (1.0 + selected_returns).cumprod()
    selected_daily["requested_candidate"] = [
        data.candidates[value] for value in selected_run.requested_target
    ]
    selected_daily["actual_candidate"] = selected_target
    selected_daily["cost_rate_at_open"] = costs
    selected_daily.to_csv(output / "selected_daily.csv")
    selected_daily.to_parquet(output / "selected_daily.parquet")

    metrics = performance(selected_returns)
    momentum_metrics = performance(momentum_returns)
    anchor_metrics = performance(anchor_returns)
    three_x = friction.loc[friction["cost_multiplier"].eq(3.0)].iloc[0]
    audit = {
        "strategy_id": "momentum_defender_selected_asset_draqm_v1",
        "requested_asset_interpretation": "510330 interpreted as existing 510300.SH",
        "selection_status": "post_v3_final_local_plateau_representative",
        "selected_candidate": selected_spec.candidate_id(),
        "selected_policies": {
            "510300.SH": {
                "profile": csi_profile.profile_id,
                "horizons": [30, 40],
                "weights": [0.25, 0.75],
                "entry_percentile": 0.35,
                "recovery_percentile": 0.25,
                "entry_confirmation_days": 1,
                "recovery_confirmation_days": 1,
            },
            "518880.SH": {
                "profile": gold_profile.profile_id,
                "horizons": [20, 40],
                "weights": [0.25, 0.75],
                "entry_percentile": 0.45,
                "recovery_percentile": 0.00,
                "entry_confirmation_days": 5,
                "recovery_confirmation_days": 1,
            },
        },
        "state_policy": {
            "momentum_lock_days": 20,
            "defender_lock_days": 23,
            "recovery_mode": STICKY_ENTRY_ASSET,
            "other_momentum_assets_gated": False,
        },
        "metrics": metrics,
        "momentum_metrics": momentum_metrics,
        "universal_anchor_metrics": anchor_metrics,
        "asset_policy_ablation": ablations,
        "search_scope": {
            "candidate_ids": (1728 + 1458) + (6270 + 7098) + (1932 + 6615),
            "family_unique_paths": family_counts,
            "global_unique_paths": int(global_returns.shape[1]),
        },
        "global_cscv_vs_momentum": pbo_momentum_summary,
        "global_cscv_vs_universal_anchor": pbo_anchor_summary,
        "global_reality_vs_momentum": reality_momentum,
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
            "beats_log_qm_momentum": metrics["annualized_return_252"]
            > momentum_metrics["annualized_return_252"]
            and metrics["sharpe"] > momentum_metrics["sharpe"],
            "beats_universal_anchor": metrics["annualized_return_252"]
            > anchor_metrics["annualized_return_252"]
            and metrics["sharpe"] > anchor_metrics["sharpe"],
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
        "requested_asset_interpretation": audit["requested_asset_interpretation"],
        "factor": {
            "formula": "downside_regularized_raqm",
            "volatility_floor_annual": 0.08,
            "winsor_limit": 3.0,
            "percentile_history": "rolling_504_strict_lag",
            "percentile_min_history": 252,
            "asset_policies": audit["selected_policies"],
        },
        "state_policy": audit["state_policy"],
        "execution": {
            "signal_timing": "previous_close_to_next_open",
            "costs": "inherited_exact_asset_interfaces",
        },
        "checkpoint": {
            **metrics,
            "defender_entries": selected_run.defender_entries,
            "defender_days": selected_run.defender_days,
            "sleeve_switches": selected_run.sleeve_switches,
            "candidate_switches": selected_run.candidate_switches,
            "daily_return_sha256_float64_le": audit[
                "daily_return_sha256_float64_le"
            ],
        },
        "evidence": {
            "candidate_ids": audit["search_scope"]["candidate_ids"],
            "global_unique_paths": audit["search_scope"]["global_unique_paths"],
            "reality_check_p_vs_momentum": reality_momentum["p_value"],
            "reality_check_p_vs_universal_anchor": reality_anchor["p_value"],
            "three_x_cost_annualized_return_252": audit[
                "three_x_cost_annualized_return_252"
            ],
        },
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

    manifest_sources = [
        root / "research/momentum_defender_selected_asset_draqm.py",
        root / "research/run_momentum_defender_selected_asset_draqm.py",
        root / "research/finalize_momentum_defender_selected_asset_draqm.py",
        root / "research/configs/momentum_defender_selected_asset_draqm_search.yaml",
        root / "research/configs/momentum_defender_selected_asset_draqm_focused.yaml",
        root / "research/configs/momentum_defender_selected_asset_draqm_final_neighborhood.yaml",
        root / "data/db/510300.SH.parquet",
        root / "data/db/518880.SH.parquet",
    ]
    manifest = {
        "strategy_id": audit["strategy_id"],
        "sources": {
            str(path.relative_to(root)): _sha(path) for path in manifest_sources
        },
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
