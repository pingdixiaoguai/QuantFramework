"""Preregistered search for a relative Gold-versus-Defender overlay."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from factors.quality_momentum import METADATA as QUALITY_METADATA
from research.momentum_defender_common_score_trimmed import (
    ExtremeBlockSpec,
    build_extreme_block_mask,
)
from research.momentum_defender_downside_raqm import (
    FactorProfile,
    build_exact_execution_data,
    exact_candidate_schedule,
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
from research.momentum_defender_relative_gold_overlay import (
    RelativeGoldOverlaySpec,
    fast_relative_gold_state,
    run_relative_gold_overlay,
    signed_raqm_profiles_at_open,
)
from research.momentum_volatility import load_ohlc
from research.run_momentum_defender_log_qm_robust import (
    _event_stress,
    _friction,
    _selected_cost_schedule,
    _unique_paths,
)
from research.standard_report import generate_standard_report


DEFAULT_CONFIG = Path(
    "research/configs/momentum_defender_relative_gold_overlay_search.yaml"
)
DEFAULT_OUTPUT = Path(
    "experiments/20260825_momentum_defender_relative_gold_overlay_search"
)


def _load(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("relative Gold overlay config must be a mapping")
    return value


def _profiles(config: dict) -> dict[str, FactorProfile]:
    return {
        profile_id: FactorProfile(
            profile_id,
            tuple(map(int, values["horizons"])),
            tuple(map(float, values["weights"])),
        )
        for profile_id, values in config["factor"]["profiles"].items()
    }


def _specs(config: dict, profiles: dict[str, FactorProfile]):
    grid = config["relative_overlay_grid"]
    minimum_gap = float(grid["minimum_hysteresis_gap"])
    result: dict[str, RelativeGoldOverlaySpec] = {}
    for profile, entry, exit_, entry_c, exit_c, hold, mode in product(
        profiles.values(),
        grid["entry_differences"],
        grid["exit_differences"],
        grid["entry_confirmation_days"],
        grid["exit_confirmation_days"],
        grid["minimum_gold_hold_days"],
        grid["override_modes"],
    ):
        if float(entry) - float(exit_) + 1e-12 < minimum_gap:
            continue
        spec = RelativeGoldOverlaySpec(
            profile=profile,
            entry_difference=float(entry),
            exit_difference=float(exit_),
            entry_confirmation_days=int(entry_c),
            exit_confirmation_days=int(exit_c),
            minimum_gold_hold_days=int(hold),
            override_mode=str(mode),
        )
        result[spec.candidate_id()] = spec
    return result


def _matrix_stats(values: np.ndarray) -> dict[str, np.ndarray]:
    mean = values.mean(axis=0)
    std = values.std(axis=0, ddof=1)
    sharpe = np.divide(
        mean * np.sqrt(252.0),
        std,
        out=np.zeros_like(mean),
        where=std > 0.0,
    )
    curve = np.cumprod(1.0 + values, axis=0)
    annualized = np.power(curve[-1], 252.0 / len(values)) - 1.0
    drawdown = curve / np.maximum.accumulate(curve, axis=0) - 1.0
    return {
        "annualized_return_252": annualized,
        "annualized_volatility": std * np.sqrt(252.0),
        "sharpe": sharpe,
        "max_drawdown": drawdown.min(axis=0),
    }


def _metrics(
    metadata: pd.DataFrame,
    returns: pd.DataFrame,
    universal: pd.Series,
    ordinary: pd.Series,
    config: dict,
) -> pd.DataFrame:
    result = metadata.copy()
    ordinary = ordinary.reindex(returns.index).fillna(False).astype(bool)
    scopes = {
        "full": np.ones(len(returns), dtype=bool),
        "ordinary": ordinary.to_numpy(bool),
    }
    for label, mask in scopes.items():
        stats = _matrix_stats(returns.to_numpy(float)[mask])
        baseline = performance(universal.loc[mask])
        for field, values in stats.items():
            result[f"{label}_{field}"] = values
            if field in {
                "annualized_return_252",
                "sharpe",
                "max_drawdown",
            }:
                result[f"{label}_delta_{field}"] = values - float(
                    baseline[field]
                )
    for period in ("development", "validation", "recent"):
        start, end = map(pd.Timestamp, config["periods"][period])
        period_mask = returns.index.to_series().between(start, end).to_numpy()
        for sample, mask in (
            (period, period_mask),
            (f"ordinary_{period}", period_mask & ordinary.to_numpy(bool)),
        ):
            stats = _matrix_stats(returns.to_numpy(float)[mask])
            for field, values in stats.items():
                result[f"{sample}_{field}"] = values
    result["full_minimum_segment_sharpe"] = result[
        [
            "development_sharpe",
            "validation_sharpe",
            "recent_sharpe",
        ]
    ].min(axis=1)
    result["ordinary_minimum_segment_sharpe"] = result[
        [
            "ordinary_development_sharpe",
            "ordinary_validation_sharpe",
            "ordinary_recent_sharpe",
        ]
    ].min(axis=1)
    return result


def _add_neighborhood(table: pd.DataFrame, config: dict) -> pd.DataFrame:
    result = table.copy()
    grid = config["relative_overlay_grid"]
    dimensions = {
        "entry_difference": list(map(float, grid["entry_differences"])),
        "exit_difference": list(map(float, grid["exit_differences"])),
        "entry_confirmation_days": list(
            map(int, grid["entry_confirmation_days"])
        ),
        "exit_confirmation_days": list(map(int, grid["exit_confirmation_days"])),
        "minimum_gold_hold_days": list(
            map(int, grid["minimum_gold_hold_days"])
        ),
    }
    coordinate_columns = []
    for field, values in dimensions.items():
        column = f"_{field}"
        result[column] = result[field].map(
            {value: position for position, value in enumerate(values)}
        )
        coordinate_columns.append(column)
    rows: dict[str, dict[str, float | int]] = {}
    arrays = {
        "full_annualized": "full_annualized_return_252",
        "full_sharpe": "full_sharpe",
        "full_mdd": "full_max_drawdown",
        "ordinary_annualized": "ordinary_annualized_return_252",
        "ordinary_sharpe": "ordinary_sharpe",
        "ordinary_mdd": "ordinary_max_drawdown",
    }
    for _, group in result.groupby(["profile_id", "override_mode"], sort=False):
        coordinates = group[coordinate_columns].to_numpy(int)
        values = {
            label: group[field].to_numpy(float) for label, field in arrays.items()
        }
        for position, candidate_id in enumerate(group.index):
            members = np.all(np.abs(coordinates - coordinates[position]) <= 1, axis=1)
            rows[str(candidate_id)] = {
                "neighborhood_count": int(members.sum()),
                **{
                    f"neighborhood_{label}_q25": float(
                        np.quantile(array[members], 0.25)
                    )
                    for label, array in values.items()
                },
                **{
                    f"neighborhood_{label}_median": float(
                        np.median(array[members])
                    )
                    for label, array in values.items()
                },
            }
    return result.drop(columns=coordinate_columns).join(
        pd.DataFrame.from_dict(rows, orient="index")
    )


def _complexity(row: pd.Series, config: dict) -> float:
    values = config["occam_complexity"]
    weighted = len(str(row["horizons"]).split("|")) > 1
    score = values["weighted_profile"] if weighted else values["single_window_profile"]
    score += values[str(row["override_mode"])]
    score += values["confirmation_day_unit"] * (
        int(row["entry_confirmation_days"])
        + int(row["exit_confirmation_days"])
        - 2
    )
    score += values["hold_day_unit"] * int(row["minimum_gold_hold_days"])
    return float(score)


def _select(table: pd.DataFrame, config: dict, key: str, prefix: str):
    result = table.copy()
    fields = list(config[key]["ranking_fields"])
    percentiles = result[fields].rank(pct=True)
    result[f"{prefix}_rank_min"] = percentiles.min(axis=1)
    result[f"{prefix}_rank_mean"] = percentiles.mean(axis=1)
    best_min = float(result[f"{prefix}_rank_min"].max())
    stable = result.loc[result[f"{prefix}_rank_min"].ge(best_min - 0.03)].copy()
    best_mean = float(stable[f"{prefix}_rank_mean"].max())
    stable = stable.loc[stable[f"{prefix}_rank_mean"].ge(best_mean - 0.03)]
    annual_field, sharpe_field = fields[:2]
    best_annual = float(stable[annual_field].max())
    best_sharpe = float(stable[sharpe_field].max())
    near = stable.loc[
        stable[annual_field].ge(
            best_annual - float(config[key]["occam_annualized_tolerance"])
        )
        & stable[sharpe_field].ge(
            best_sharpe - float(config[key]["occam_sharpe_tolerance"])
        )
    ].copy()
    near["occam_complexity"] = near.apply(_complexity, axis=1, config=config)
    selected = near.sort_values(
        ["occam_complexity", annual_field, sharpe_field, f"{prefix}_rank_mean"],
        ascending=[True, False, False, False],
    ).iloc[0]
    result[f"{prefix}_occam_near"] = result.index.isin(near.index)
    return selected, result


def _pareto(table: pd.DataFrame, prefix: str) -> pd.DataFrame:
    fields = [
        f"{prefix}_annualized_return_252",
        f"{prefix}_sharpe",
        f"{prefix}_max_drawdown",
    ]
    values = table[fields].to_numpy(float)
    keep = np.ones(len(table), dtype=bool)
    for position in range(len(table)):
        if not keep[position]:
            continue
        dominated = np.all(values >= values[position], axis=1) & np.any(
            values > values[position], axis=1
        )
        if dominated.any():
            keep[position] = False
    return table.loc[keep].sort_values(fields[:2], ascending=False)


def _neighborhood_detail(table: pd.DataFrame, selected: pd.Series, config: dict):
    grid = config["relative_overlay_grid"]
    fields = [
        "entry_difference",
        "exit_difference",
        "entry_confirmation_days",
        "exit_confirmation_days",
        "minimum_gold_hold_days",
    ]
    values_by_field = {
        "entry_difference": list(map(float, grid["entry_differences"])),
        "exit_difference": list(map(float, grid["exit_differences"])),
        "entry_confirmation_days": list(map(int, grid["entry_confirmation_days"])),
        "exit_confirmation_days": list(map(int, grid["exit_confirmation_days"])),
        "minimum_gold_hold_days": list(map(int, grid["minimum_gold_hold_days"])),
    }
    sample = table.loc[
        table["profile_id"].eq(selected["profile_id"])
        & table["override_mode"].eq(selected["override_mode"])
    ].copy()
    member = np.ones(len(sample), dtype=bool)
    for field in fields:
        mapping = {value: position for position, value in enumerate(values_by_field[field])}
        selected_position = mapping[selected[field]]
        member &= sample[field].map(mapping).sub(selected_position).abs().le(1).to_numpy()
    neighborhood = sample.loc[member].copy()
    summary: dict[str, float | int] = {"count": int(len(neighborhood))}
    for scope in ("full", "ordinary"):
        for metric in ("annualized_return_252", "sharpe", "max_drawdown"):
            field = f"{scope}_{metric}"
            delta = neighborhood[field] - float(selected[field])
            summary[f"{scope}_{metric}_q25"] = float(neighborhood[field].quantile(0.25))
            summary[f"{scope}_{metric}_median"] = float(neighborhood[field].median())
            summary[f"{scope}_{metric}_improve_rate"] = float(delta.gt(0.0).mean())
    for scope in ("full", "ordinary"):
        three = (
            neighborhood[f"{scope}_annualized_return_252"].ge(
                float(selected[f"{scope}_annualized_return_252"])
            )
            & neighborhood[f"{scope}_sharpe"].ge(float(selected[f"{scope}_sharpe"]))
            & neighborhood[f"{scope}_max_drawdown"].ge(
                float(selected[f"{scope}_max_drawdown"])
            )
        )
        summary[f"{scope}_three_metric_nonworse_rate"] = float(three.mean())
    return neighborhood, summary


def _selected_detail(
    label: str,
    selected: pd.Series,
    spec: RelativeGoldOverlaySpec,
    data,
    context,
    base_risk_on,
    metrics,
    universal_daily,
    ordinary,
    output: Path,
    config: dict,
) -> dict:
    run = run_relative_gold_overlay(
        data,
        context.momentum_target,
        base_risk_on,
        metrics[spec.profile.profile_id],
        spec,
    )
    returns = pd.Series(run.returns, index=data.calendar, name=spec.candidate_id())
    actual = pd.Series(
        [data.candidates[value] for value in run.actual_target], index=data.calendar
    )
    daily = run.state.copy()
    daily["return"] = returns
    daily["nav"] = (1.0 + returns).cumprod()
    daily["requested_candidate"] = [
        data.candidates[value] for value in run.requested_target
    ]
    daily["actual_candidate"] = actual
    daily.to_parquet(output / f"selected_{label}_daily.parquet")
    daily.to_csv(output / f"selected_{label}_daily.csv")

    universal_returns = universal_daily["return"].astype(float)
    universal_target = universal_daily["actual_candidate"].astype(str)
    events, leave_events, deleted, event_summary = _event_stress(
        returns,
        universal_returns,
        actual,
        universal_target,
        list(map(int, config["overfit_checks"]["top_positive_event_deletions"])),
    )
    events.to_csv(output / f"selected_{label}_events.csv", index=False)
    leave_events.to_csv(output / f"selected_{label}_leave_event.csv", index=False)
    deleted.to_csv(output / f"selected_{label}_delete_top_events.csv", index=False)
    bootstrap, bootstrap_summary = paired_block_bootstrap(
        returns,
        universal_returns,
        block_size=int(config["overfit_checks"]["paired_block_bootstrap_block"]),
        repetitions=int(
            config["overfit_checks"]["paired_block_bootstrap_repetitions"]
        ),
        seed=int(config["overfit_checks"]["random_seed"]),
    )
    bootstrap.to_csv(output / f"selected_{label}_bootstrap.csv", index=False)
    costs = _selected_cost_schedule(context, data, run.actual_target)
    friction = _friction(
        returns,
        costs,
        list(map(float, config["overfit_checks"]["friction_cost_multipliers"])),
    )
    friction.to_csv(output / f"selected_{label}_friction.csv", index=False)
    generate_standard_report(
        returns,
        universal_returns,
        "frozen_universal_gate",
        output / f"selected_{label}_vs_universal.html",
        {"experiment": config["experiment"], "selected": asdict(spec)},
    )
    selected_config = {
        "strategy_id": f"momentum_defender_relative_gold_overlay_{label}_v1",
        "status": "research_candidate_not_production",
        "selected_on": config["experiment"]["created_on"],
        "evidence_status": config["experiment"]["evidence_status"],
        "base": config["frozen_layers"]["universal_gate_config"],
        "factor": {
            "formula": config["factor"]["formula"],
            "horizons": list(spec.profile.horizons),
            "weights": list(spec.profile.weights),
            "volatility_floor_annual": config["factor"]["volatility_floor_annual"],
            "winsor_limit": config["factor"]["winsor_limit"],
            "gold_health_threshold": 0.0,
            "timing": config["factor"]["timing"],
        },
        "overlay": {
            "entry_difference": spec.entry_difference,
            "exit_difference": spec.exit_difference,
            "entry_confirmation_days": spec.entry_confirmation_days,
            "exit_confirmation_days": spec.exit_confirmation_days,
            "minimum_gold_hold_days": spec.minimum_gold_hold_days,
            "override_mode": spec.override_mode,
            "reset_when_gold_leaves_top1": True,
            "weak_gold_bypasses_hold": True,
        },
        "checkpoint": {
            **performance(returns),
            "ordinary": performance(returns.loc[ordinary]),
            "universal": performance(universal_returns),
            "gold_entries": run.gold_entries,
            "gold_allowed_days": run.gold_allowed_days,
            "override_days": run.override_days,
            "candidate_switches": run.candidate_switches,
            "daily_return_sha256_float64_le": hashlib.sha256(
                returns.to_numpy(dtype="<f8").tobytes()
            ).hexdigest(),
        },
        "bootstrap_vs_universal": bootstrap_summary,
        "event_stress_vs_universal": event_summary,
        "three_x_cost": friction.loc[
            friction["cost_multiplier"].eq(3.0)
        ].iloc[0].to_dict(),
        "decision": {
            "automatic_production_promotion": False,
            "requires_explicit_user_promotion": True,
        },
    }
    (output / f"selected_{label}_config.yaml").write_text(
        yaml.safe_dump(selected_config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return {
        "candidate_id": spec.candidate_id(),
        "config": selected_config,
        "event_summary": event_summary,
        "bootstrap_summary": bootstrap_summary,
        "friction": friction.to_dict(orient="records"),
    }


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_experiment(root: Path, config_path: Path, output: Path) -> dict:
    config = _load(config_path)
    if QUALITY_METADATA["version"] != config["frozen_layers"][
        "momentum_factor_version"
    ]:
        raise AssertionError("Momentum factor version mismatch")
    output.mkdir(parents=True, exist_ok=True)
    end = pd.Timestamp(config["periods"]["full"][1])
    context = build_gold_override_context(root, end=end.date())
    data = build_exact_execution_data(context)
    profiles = _profiles(config)
    specs = _specs(config, profiles)
    metrics = signed_raqm_profiles_at_open(
        context.curves,
        profiles,
        volatility_floor_annual=float(config["factor"]["volatility_floor_annual"]),
        winsor_limit=float(config["factor"]["winsor_limit"]),
    )
    universal_path = root / config["frozen_layers"]["universal_gate_daily"]
    universal_daily = pd.read_parquet(universal_path).reindex(data.calendar)
    if universal_daily[["risk_on", "return", "actual_candidate"]].isna().any().any():
        raise ValueError("frozen universal daily file does not cover experiment calendar")
    base_risk_on = universal_daily["risk_on"].astype(bool)
    universal_returns = universal_daily["return"].astype(float)

    closes = {
        asset: load_ohlc(asset, end.date())["close"]
        for asset in ("510300.SH", "518880.SH")
    }
    trim = config["extreme_block_trim"]
    shock = build_extreme_block_mask(
        closes,
        data.calendar,
        ExtremeBlockSpec(
            shock_return_window=int(trim["shock_return_window"]),
            block_length_sessions=int(trim["block_length_sessions"]),
            excluded_block_fraction=float(trim["excluded_block_fraction"]),
            normalization_mode=str(trim["normalization_mode"]),
        ),
    )
    shock.blocks.to_csv(output / "shock_blocks.csv", index=False)
    ordinary = shock.selection_mask.astype(bool)

    ids = list(specs)
    matrix = np.empty((len(data.calendar), len(ids)), dtype=np.float32)
    records = []
    gold_top1 = context.momentum_target.eq("518880.SH").to_numpy(bool)
    base_values = base_risk_on.to_numpy(bool)
    defender_index = data.candidate_index["DEFENDER"]
    for position, candidate_id in enumerate(ids):
        spec = specs[candidate_id]
        metric = metrics[spec.profile.profile_id]
        state = fast_relative_gold_state(
            gold_top1,
            base_values,
            metric["518880.SH"].to_numpy(float),
            metric["DEFENDER"].to_numpy(float),
            metric["difference"].to_numpy(float),
            spec,
        )
        requested = np.where(
            state.effective_risk_on, data.momentum_target, defender_index
        ).astype(int)
        candidate_returns, _, candidate_switches = exact_candidate_schedule(
            data, requested
        )
        matrix[:, position] = candidate_returns
        entries = state.gold_overlay_changed & state.gold_overlay_active
        records.append(
            {
                "candidate_id": candidate_id,
                "profile_id": spec.profile.profile_id,
                "horizons": "|".join(map(str, spec.profile.horizons)),
                "weights": "|".join(f"{value:.3f}" for value in spec.profile.weights),
                "entry_difference": spec.entry_difference,
                "exit_difference": spec.exit_difference,
                "entry_confirmation_days": spec.entry_confirmation_days,
                "exit_confirmation_days": spec.exit_confirmation_days,
                "minimum_gold_hold_days": spec.minimum_gold_hold_days,
                "override_mode": spec.override_mode,
                "gold_entries": int(entries.sum()),
                "gold_allowed_days": int(state.gold_overlay_active.sum()),
                "override_days": int(state.gold_overrides_base.sum()),
                "candidate_switches": candidate_switches,
            }
        )
        if (position + 1) % 500 == 0 or position + 1 == len(ids):
            print(f"relative overlay: {position + 1}/{len(ids)}", flush=True)
    returns = pd.DataFrame(matrix, index=data.calendar, columns=ids)
    metadata = pd.DataFrame(records).set_index("candidate_id")
    table = _add_neighborhood(
        _metrics(metadata, returns, universal_returns, ordinary, config), config
    )
    selected_full, table = _select(
        table, config, "selection_including_extremes", "full"
    )
    selected_ordinary, table = _select(
        table, config, "selection_excluding_extremes", "ordinary"
    )
    exploratory_full_champion = table.sort_values(
        ["full_annualized_return_252", "full_sharpe"], ascending=False
    ).iloc[0]
    table.to_csv(output / "candidate_grid.csv")
    _pareto(table, "full").to_csv(output / "pareto_full.csv")
    _pareto(table, "ordinary").to_csv(output / "pareto_ordinary.csv")

    neighborhood_summaries = {}
    for label, selected in (
        ("including_extremes", selected_full),
        ("excluding_extremes", selected_ordinary),
        ("exploratory_full_champion", exploratory_full_champion),
    ):
        neighborhood, summary = _neighborhood_detail(table, selected, config)
        neighborhood.to_csv(output / f"selected_{label}_neighborhood.csv")
        neighborhood_summaries[label] = summary

    unique = _unique_paths(returns)
    unique.to_parquet(output / "unique_candidate_returns.parquet")
    cscv_full, cscv_full_summary = cscv_pbo(
        unique,
        universal_returns,
        block_count=int(config["overfit_checks"]["cscv_blocks"]),
    )
    cscv_ordinary, cscv_ordinary_summary = cscv_pbo(
        unique.loc[ordinary],
        universal_returns.loc[ordinary],
        block_count=int(config["overfit_checks"]["cscv_blocks"]),
    )
    cscv_full.to_csv(output / "cscv_full.csv", index=False)
    cscv_ordinary.to_csv(output / "cscv_ordinary.csv", index=False)
    reality_full = yearly_reality_check(
        unique,
        universal_returns,
        repetitions=int(config["overfit_checks"]["yearly_reality_check_repetitions"]),
        seed=int(config["overfit_checks"]["random_seed"]),
    )
    reality_ordinary = yearly_reality_check(
        unique.loc[ordinary],
        universal_returns.loc[ordinary],
        repetitions=int(config["overfit_checks"]["yearly_reality_check_repetitions"]),
        seed=int(config["overfit_checks"]["random_seed"]),
    )
    walk = expanding_walk_forward(unique, universal_returns)
    leave_year = leave_one_year_selection(unique, universal_returns)
    walk.to_csv(output / "walk_forward.csv", index=False)
    leave_year.to_csv(output / "leave_one_year.csv", index=False)

    global_audit = None
    if config["overfit_checks"].get(
        "include_prior_gold_exception_paths_in_global_audit", False
    ):
        prior_path = root / (
            "experiments/20260825_momentum_defender_gold_exception_search/"
            "unique_candidate_returns.parquet"
        )
        prior = pd.read_parquet(prior_path).reindex(data.calendar)
        global_paths = _unique_paths(
            pd.concat(
                [prior.add_prefix("prior::"), unique.add_prefix("relative::")],
                axis=1,
            )
        )
        global_audit = {
            "prior_paths": int(prior.shape[1]),
            "relative_paths": int(unique.shape[1]),
            "global_unique_paths": int(global_paths.shape[1]),
            "full": yearly_reality_check(
                global_paths,
                universal_returns,
                repetitions=int(
                    config["overfit_checks"]["yearly_reality_check_repetitions"]
                ),
                seed=int(config["overfit_checks"]["random_seed"]),
            ),
            "ordinary": yearly_reality_check(
                global_paths.loc[ordinary],
                universal_returns.loc[ordinary],
                repetitions=int(
                    config["overfit_checks"]["yearly_reality_check_repetitions"]
                ),
                seed=int(config["overfit_checks"]["random_seed"]),
            ),
        }

    details = {
        "including_extremes": _selected_detail(
            "including_extremes",
            selected_full,
            specs[str(selected_full.name)],
            data,
            context,
            base_risk_on,
            metrics,
            universal_daily,
            ordinary,
            output,
            config,
        ),
        "excluding_extremes": _selected_detail(
            "excluding_extremes",
            selected_ordinary,
            specs[str(selected_ordinary.name)],
            data,
            context,
            base_risk_on,
            metrics,
            universal_daily,
            ordinary,
            output,
            config,
        ),
        "exploratory_full_champion": _selected_detail(
            "exploratory_full_champion",
            exploratory_full_champion,
            specs[str(exploratory_full_champion.name)],
            data,
            context,
            base_risk_on,
            metrics,
            universal_daily,
            ordinary,
            output,
            config,
        ),
    }
    audit = {
        "experiment": config["experiment"],
        "calendar": {
            "start": data.calendar.min().date().isoformat(),
            "end": data.calendar.max().date().isoformat(),
            "observations": len(data.calendar),
            "ordinary_observations": int(ordinary.sum()),
        },
        "search": {
            "candidate_ids": len(specs),
            "unique_paths": int(unique.shape[1]),
            "selected_including_extremes": str(selected_full.name),
            "selected_excluding_extremes": str(selected_ordinary.name),
            "exploratory_full_champion": str(exploratory_full_champion.name),
        },
        "universal": performance(universal_returns),
        "neighborhoods": neighborhood_summaries,
        "cscv": {
            "full": cscv_full_summary,
            "ordinary": cscv_ordinary_summary,
        },
        "reality_check": {"full": reality_full, "ordinary": reality_ordinary},
        "global_prior_audit": global_audit,
        "walk_forward": {
            "observations": len(walk),
            "return_win_rate": float(walk["test_return_delta"].gt(0.0).mean()),
            "sharpe_win_rate": float(walk["test_sharpe_delta"].gt(0.0).mean()),
        },
        "leave_one_year": {
            "observations": len(leave_year),
            "return_win_rate": float(leave_year["test_return_delta"].gt(0.0).mean()),
            "sharpe_win_rate": float(leave_year["test_sharpe_delta"].gt(0.0).mean()),
        },
        "selected": details,
        "causality": {
            "signal_timing": "previous_close_to_next_open",
            "base_state_source": str(universal_path.relative_to(root)),
            "base_state_mutated": False,
            "outside_gold_top1_effective_equals_base": True,
            "exact_exit_and_entry_legs": True,
            "untradable_switch_retains_previous_candidate": True,
            "baseline_parity_max_abs_error": context.baseline_parity_max_abs_error,
        },
        "decision": config["decision"],
    }
    (output / "audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    if global_audit is not None:
        (output / "global_prior_audit.json").write_text(
            json.dumps(global_audit, indent=2, ensure_ascii=False, default=str)
            + "\n",
            encoding="utf-8",
        )
    sources = [
        config_path,
        Path(__file__),
        root / "research/momentum_defender_relative_gold_overlay.py",
        root / "research/tests/test_momentum_defender_relative_gold_overlay.py",
        root / "research/audit_relative_gold_global_paths.py",
        universal_path,
    ]
    manifest = {
        "experiment_id": config["experiment"]["id"],
        "files": {
            str(path.resolve().relative_to(root)): _sha(path.resolve()) for path in sources
        },
    }
    (output / "experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    audit = run_experiment(root, args.config.resolve(), args.output.resolve())
    print(json.dumps(audit["search"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
