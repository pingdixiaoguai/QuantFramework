"""Finalize the weighted downside-RAQM research candidate and global audit."""

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
from research.momentum_volatility import load_ohlc
from research.run_momentum_defender_log_qm_robust import (
    _event_stress,
    _friction,
    _selected_cost_schedule,
)
from research.standard_report import generate_standard_report


EXPERIMENTS = (
    Path("experiments/20260824_momentum_defender_downside_raqm"),
    Path("experiments/20260824_momentum_defender_downside_raqm_focused_stability"),
    Path("experiments/20260824_momentum_defender_downside_raqm_weighted_profiles"),
)
OUTPUT = Path("experiments/20260824_momentum_defender_downside_raqm_final_selection")
END = pd.Timestamp("2026-08-21")
SELECTED_ID = "draqm_w30_40_25_75_r504_en0.55_ex0.20_mh30_dh30_ec3_rc1"


def _unique_global_returns(root: Path) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    seen: set[str] = set()
    frames: list[pd.DataFrame] = []
    counts: list[dict[str, object]] = []
    for experiment in EXPERIMENTS:
        path = root / experiment / "unique_candidate_returns.parquet"
        frame = pd.read_parquet(path)
        keep: list[str] = []
        for column in frame:
            digest = hashlib.sha1(frame[column].to_numpy(float).tobytes()).hexdigest()
            if digest not in seen:
                seen.add(digest)
                keep.append(str(column))
        selected = frame.loc[:, keep].copy()
        prefix = experiment.name.removeprefix("20260824_")
        selected.columns = [f"{prefix}::{column}" for column in selected.columns]
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


def _calendar_year_metrics(returns: pd.Series, baseline: pd.Series) -> pd.DataFrame:
    rows = []
    for year in sorted(returns.index.year.unique()):
        candidate = performance(returns.loc[returns.index.year == year])
        base = performance(baseline.loc[baseline.index.year == year])
        rows.append(
            {
                "year": int(year),
                **{f"candidate_{key}": value for key, value in candidate.items()},
                **{f"momentum_{key}": value for key, value in base.items()},
            }
        )
    return pd.DataFrame(rows)


def _segment_metrics(returns: pd.Series, baseline: pd.Series) -> pd.DataFrame:
    periods = {
        "development": ("2019-01-18", "2022-12-30"),
        "validation": ("2023-01-01", "2024-12-31"),
        "recent": ("2025-01-01", "2026-08-21"),
        "full": ("2019-01-18", "2026-08-21"),
    }
    rows = []
    for period, (start, end) in periods.items():
        candidate = performance(returns.loc[start:end])
        base = performance(baseline.loc[start:end])
        rows.append(
            {
                "period": period,
                **{f"candidate_{key}": value for key, value in candidate.items()},
                **{f"momentum_{key}": value for key, value in base.items()},
            }
        )
    return pd.DataFrame(rows)


def _candidate_comparison(root: Path) -> pd.DataFrame:
    candidates = {
        "v1_stability_selection": (
            EXPERIMENTS[0],
            "draqm_w40_r504_en0.60_ex0.30_mh30_dh30_ec1_rc1",
        ),
        "v2_cross_window_fallback": (
            EXPERIMENTS[1],
            "draqm_w35_r504_en0.70_ex0.30_mh25_dh30_ec3_rc1",
        ),
        "v3_preregistered_selection": (
            EXPERIMENTS[2],
            "draqm_w40_r504_en0.55_ex0.20_mh30_dh30_ec3_rc1",
        ),
        "final_weighted_candidate": (EXPERIMENTS[2], SELECTED_ID),
    }
    rows = []
    for role, (experiment, candidate_id) in candidates.items():
        grid = pd.read_csv(root / experiment / "search_grid.csv").set_index(
            "candidate_id"
        )
        selected = grid.loc[candidate_id]
        rows.append({"role": role, "candidate_id": candidate_id, **selected.to_dict()})
    return pd.DataFrame(rows)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output = root / OUTPUT
    output.mkdir(parents=True, exist_ok=True)

    context = build_gold_override_context(root, end=END.date())
    data = build_exact_execution_data(context)
    profile = FactorProfile("w30_40_25_75", (30, 40), (0.25, 0.75))
    spec = DownsideRAQMSpec(
        profile,
        "rolling_504_strict_lag",
        0.55,
        0.20,
        30,
        30,
        3,
        1,
    )
    if spec.candidate_id() != SELECTED_ID:
        raise AssertionError("final candidate ID drift")
    features = build_downside_raqm_features(
        load_ohlc("510300.SH", END.date())["close"],
        data.calendar,
        {profile.profile_id: profile},
        {"rolling_504_strict_lag": 504},
        min_history=252,
        volatility_floor_annual=0.08,
        winsor_limit=3.0,
    )
    run = run_downside_raqm_spec(data, features, spec)
    selected_returns = pd.Series(run.returns, index=data.calendar, name=SELECTED_ID)
    selected_target = pd.Series(
        [data.candidates[value] for value in run.actual_target], index=data.calendar
    )
    baseline_values, baseline_actual, _ = exact_candidate_schedule(
        data, data.momentum_target
    )
    baseline_returns = pd.Series(
        baseline_values, index=data.calendar, name="log_qm_momentum"
    )
    baseline_target = pd.Series(
        [data.candidates[value] for value in baseline_actual], index=data.calendar
    )

    global_returns, family_counts = _unique_global_returns(root)
    if not global_returns.index.equals(data.calendar):
        raise AssertionError("global candidate returns calendar mismatch")
    checks = {
        "block_count": 12,
        "bootstrap_block": 20,
        "repetitions": 5000,
        "seed": 20260824,
    }
    pbo, pbo_summary = cscv_pbo(
        global_returns, baseline_returns, block_count=checks["block_count"]
    )
    reality = yearly_reality_check(
        global_returns,
        baseline_returns,
        repetitions=checks["repetitions"],
        seed=checks["seed"],
    )
    walk = expanding_walk_forward(global_returns, baseline_returns)
    leave_selection = leave_one_year_selection(global_returns, baseline_returns)
    bootstrap, bootstrap_summary = paired_block_bootstrap(
        selected_returns,
        baseline_returns,
        block_size=checks["bootstrap_block"],
        repetitions=checks["repetitions"],
        seed=checks["seed"],
    )
    events, leave_event, top_deletion, event_summary = _event_stress(
        selected_returns,
        baseline_returns,
        selected_target,
        baseline_target,
        [1, 2, 3],
    )
    costs = _selected_cost_schedule(context, data, run.actual_target)
    friction = _friction(selected_returns, costs, [1.0, 2.0, 3.0])
    fixed_leave = _fixed_leave_year(selected_returns)
    yearly = _calendar_year_metrics(selected_returns, baseline_returns)
    segments = _segment_metrics(selected_returns, baseline_returns)
    comparison = _candidate_comparison(root)

    selected_daily = run.state.copy()
    selected_daily["return"] = selected_returns
    selected_daily["nav"] = (1.0 + selected_returns).cumprod()
    selected_daily["requested_candidate"] = [
        data.candidates[value] for value in run.requested_target
    ]
    selected_daily["actual_candidate"] = selected_target
    selected_daily["cost_rate_at_open"] = costs
    for horizon, values in features.raw_at_open.items():
        selected_daily[f"downside_raqm_{horizon}_at_open"] = values

    pbo.to_csv(output / "global_cscv_pbo.csv", index=False)
    walk.to_csv(output / "global_expanding_walk_forward.csv", index=False)
    leave_selection.to_csv(output / "global_leave_one_year_selection.csv", index=False)
    bootstrap.to_csv(output / "paired_block_bootstrap.csv", index=False)
    events.to_csv(output / "event_attribution.csv", index=False)
    leave_event.to_csv(output / "leave_one_event.csv", index=False)
    top_deletion.to_csv(output / "top_positive_event_deletion.csv", index=False)
    friction.to_csv(output / "friction_stress.csv", index=False)
    fixed_leave.to_csv(output / "fixed_candidate_leave_one_year.csv", index=False)
    yearly.to_csv(output / "calendar_year_metrics.csv", index=False)
    segments.to_csv(output / "segment_metrics.csv", index=False)
    comparison.to_csv(output / "candidate_comparison.csv", index=False)
    selected_daily.to_csv(output / "selected_daily.csv")
    selected_daily.to_parquet(output / "selected_daily.parquet")

    metrics = performance(selected_returns)
    baseline_metrics = performance(baseline_returns)
    selected_grid = comparison.loc[
        comparison["role"].eq("final_weighted_candidate")
    ].iloc[0]
    three_x = friction.loc[friction["cost_multiplier"].eq(3.0)].iloc[0]
    audit = {
        "strategy_id": "momentum_defender_downside_raqm_weighted_v1",
        "selected_candidate": SELECTED_ID,
        "selection_status": "post_v3_near_tie_governance_selection",
        "selection_reason": (
            "Weighted candidate has the same 100% within-profile annualized "
            "neighborhood pass rate as the preregistered v3 winner, only "
            "0.001 lower neighborhood Sharpe q25, but higher full annualized "
            "return, full Sharpe, and minimum segment Sharpe."
        ),
        "requirements": {
            "all_horizons_at_least_20": min(profile.horizons) >= 20,
            "momentum_lock_in_20_30": 20 <= spec.momentum_lock_days <= 30,
            "defender_lock_in_20_30": 20 <= spec.defender_lock_days <= 30,
            "annualized_return_at_least_45pct": metrics[
                "annualized_return_252"
            ]
            >= 0.45,
            "gold_override_disabled": True,
            "emergency_override_disabled": True,
            "signal_timing": "strictly_previous_close_to_open",
        },
        "metrics": metrics,
        "momentum_baseline_metrics": baseline_metrics,
        "parameter_stability": {
            "neighborhood_count": int(selected_grid["neighborhood_count"]),
            "annualized_45pct_pass_rate": float(
                selected_grid["neighborhood_annualized_pass_rate"]
            ),
            "annualized_q25": float(selected_grid["neighborhood_annualized_q25"]),
            "annualized_median": float(
                selected_grid["neighborhood_annualized_median"]
            ),
            "sharpe_q25": float(selected_grid["neighborhood_sharpe_q25"]),
            "sharpe_median": float(selected_grid["neighborhood_sharpe_median"]),
        },
        "search_scope": {
            "candidate_ids": 24624 + 21600 + 25920,
            "family_unique_paths": family_counts,
            "global_unique_paths": int(global_returns.shape[1]),
        },
        "global_cscv": pbo_summary,
        "global_reality_check": reality,
        "global_walk_forward_return_win_rate": float(
            walk["test_return_delta"].gt(0.0).mean()
        ),
        "global_walk_forward_sharpe_win_rate": float(
            walk["test_sharpe_delta"].gt(0.0).mean()
        ),
        "global_leave_selection_return_win_rate": float(
            leave_selection["test_return_delta"].gt(0.0).mean()
        ),
        "global_leave_selection_sharpe_win_rate": float(
            leave_selection["test_sharpe_delta"].gt(0.0).mean()
        ),
        "paired_bootstrap": bootstrap_summary,
        "events": event_summary,
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
            "state_reason": str(selected_daily.iloc[-1]["state_reason"]),
            "downside_raqm_percentile_at_open": float(
                selected_daily.iloc[-1]["downside_raqm_percentile_at_open"]
            ),
            "actual_candidate": str(selected_daily.iloc[-1]["actual_candidate"]),
        },
    }
    (output / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    config = {
        "strategy_id": audit["strategy_id"],
        "status": "research_candidate_not_production",
        "selected_on": "2026-08-24",
        "factor": {
            "anchor_asset": "510300.SH",
            "formula": "downside_regularized_raqm",
            "horizons": [30, 40],
            "weights": [0.25, 0.75],
            "volatility_floor_annual": 0.08,
            "winsor_limit": 3.0,
            "percentile_history": "rolling_504_strict_lag",
            "percentile_min_history": 252,
        },
        "state_policy": {
            "defender_entry_percentile": 0.55,
            "defender_exit_percentile": 0.20,
            "momentum_lock_days": 30,
            "defender_lock_days": 30,
            "defender_entry_confirmation_days": 3,
            "momentum_recovery_confirmation_days": 1,
            "emergency_override": False,
        },
        "execution": {
            "signal_timing": "previous_close_to_next_open",
            "costs": "inherited_exact_asset_interfaces",
        },
        "checkpoint": {
            **metrics,
            "observations": len(selected_returns),
            "defender_entries": run.defender_entries,
            "defender_days": run.defender_days,
            "sleeve_switches": run.sleeve_switches,
            "candidate_switches": run.candidate_switches,
            "daily_return_sha256_float64_le": audit[
                "daily_return_sha256_float64_le"
            ],
        },
        "parameter_stability": audit["parameter_stability"],
        "overfit_audit": {
            "candidate_ids": audit["search_scope"]["candidate_ids"],
            "global_unique_paths": audit["search_scope"]["global_unique_paths"],
            "cscv_pbo": pbo_summary["pbo"],
            "reality_check_p_value": reality["p_value"],
            "bootstrap_sharpe_delta_ci": [
                bootstrap_summary["sharpe_delta_ci_lower"],
                bootstrap_summary["sharpe_delta_ci_upper"],
            ],
            "leave_one_event_min_annualized_return_252": event_summary[
                "leave_one_min_annualized_return_252"
            ],
            "fixed_leave_year_min_annualized_return_252": audit[
                "fixed_leave_year_min_annualized_return_252"
            ],
            "three_x_cost_annualized_return_252": audit[
                "three_x_cost_annualized_return_252"
            ],
        },
    }
    (output / "selected_research_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    generate_standard_report(
        selected_returns,
        baseline_returns,
        "Log-QM Momentum",
        output / "selected_vs_momentum.html",
        config,
    )

    manifest_sources = [
        root / "research/momentum_defender_downside_raqm.py",
        root / "research/finalize_momentum_defender_downside_raqm.py",
        root / "research/configs/momentum_defender_downside_raqm_search.yaml",
        root / "research/configs/momentum_defender_downside_raqm_focused_stability.yaml",
        root / "research/configs/momentum_defender_downside_raqm_weighted_profiles.yaml",
        root / "factors/quality_momentum.py",
        root / "data/db/510300.SH.parquet",
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
