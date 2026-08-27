"""Finalize the joint full/ordinary performance leader from the Occam search."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from defender.relative_defender_rotation import (
    ROTATION_COST_RATES,
    load_rotation_market,
)
from research.defender_curve_momentum import DEFENDER_CANDIDATE
from research.momentum_defender_common_score_trimmed import (
    ExtremeBlockSpec,
    build_extreme_block_mask,
)
from research.momentum_defender_downside_raqm import (
    DownsideRAQMSpec,
    build_exact_execution_data,
    run_downside_raqm_spec,
)
from research.momentum_defender_occam_defender import (
    build_portfolio_switch_interface,
)
from research.momentum_volatility import load_ohlc
from research.run_momentum_defender_occam_defender_search import (
    DEFAULT_CONFIG,
    DEFAULT_OUTPUT,
    _load,
    _occam_targets,
    _selected_detail,
)
from strategy.momentum_defender_downside_raqm import (
    formal_profile,
    run_formal_strategy,
)


METRICS = (
    "full_annualized_return_252",
    "full_sharpe",
    "ordinary_annualized_return_252",
    "ordinary_sharpe",
)


def _performance_leader(table: pd.DataFrame) -> pd.Series:
    pool = table.loc[
        table["eligible_full"].astype(bool)
        & table["eligible_ordinary"].astype(bool)
    ].copy()
    ranks = pool[list(METRICS)].rank(pct=True)
    pool["joint_performance_rank_min"] = ranks.min(axis=1)
    pool["joint_performance_rank_mean"] = ranks.mean(axis=1)
    return pool.sort_values(
        ["joint_performance_rank_min", "joint_performance_rank_mean"],
        ascending=False,
    ).iloc[0]


def _local_numeric_neighborhood(
    table: pd.DataFrame,
    selected: pd.Series,
    config: dict,
) -> pd.DataFrame:
    """Perturb one numeric parameter one grid step with confirmations frozen."""
    grid = config["state_grid"]
    dimensions = {
        "defender_window": list(map(int, config["defender_selector"]["windows"])),
        "entry_percentile": list(map(float, grid["entry_percentiles"])),
        "recovery_percentile": list(map(float, grid["recovery_percentiles"])),
        "momentum_lock_days": list(map(int, grid["momentum_lock_days"])),
        "defender_lock_days": list(map(int, grid["defender_lock_days"])),
    }
    valid = (
        table["entry_confirmation_days"].eq(
            selected["entry_confirmation_days"]
        )
        & table["recovery_confirmation_days"].eq(
            selected["recovery_confirmation_days"]
        )
    )
    distance = pd.Series(0, index=table.index, dtype=int)
    for field, values in dimensions.items():
        lookup = {value: position for position, value in enumerate(values)}
        current = lookup[selected[field]]
        delta = table[field].map(lookup).sub(current).abs()
        valid &= delta.notna() & delta.le(1)
        distance += delta.fillna(99).astype(int)
    result = table.loc[valid & distance.le(1)].copy()
    result["changed_parameter_count"] = distance.loc[result.index]
    return result.sort_values(
        ["changed_parameter_count", "full_annualized_return_252"],
        ascending=[True, False],
    )


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    config = _load((root / DEFAULT_CONFIG).resolve())
    output = (root / DEFAULT_OUTPUT).resolve()
    table = pd.read_csv(output / "candidate_grid.csv", index_col=0)
    selected = _performance_leader(table)
    end = pd.Timestamp(config["periods"]["full"][1]).date()
    formal = run_formal_strategy(root, end=end)
    context = formal.context
    market = load_rotation_market(end=end)
    window = int(selected["defender_window"])
    targets, selection = _occam_targets(market, context.calendar, window)
    interface = build_portfolio_switch_interface(
        market, targets, ROTATION_COST_RATES
    )
    selection.attrs["switch_interface"] = interface
    candidate_context = replace(
        context,
        interfaces={**context.interfaces, DEFENDER_CANDIDATE: interface},
    )
    data = build_exact_execution_data(candidate_context)
    spec = DownsideRAQMSpec(
        profile=formal_profile(),
        history_mode="rolling_504_strict_lag",
        entry_percentile=float(selected["entry_percentile"]),
        exit_percentile=float(selected["recovery_percentile"]),
        momentum_lock_days=int(selected["momentum_lock_days"]),
        defender_lock_days=int(selected["defender_lock_days"]),
        entry_confirmation_days=int(selected["entry_confirmation_days"]),
        recovery_confirmation_days=int(selected["recovery_confirmation_days"]),
    )
    run = run_downside_raqm_spec(data, formal.features, spec)
    candidate_id = str(selected.name)
    returns = pd.DataFrame(
        {candidate_id: run.returns}, index=context.calendar, dtype=np.float64
    )
    trim = config["extreme_block_trim"]
    extreme = build_extreme_block_mask(
        {
            asset: load_ohlc(asset, end)["close"]
            for asset in trim["shock_assets"]
        },
        context.calendar,
        ExtremeBlockSpec(
            shock_return_window=int(trim["shock_return_window"]),
            block_length_sessions=int(trim["block_length_sessions"]),
            excluded_block_fraction=float(trim["excluded_block_fraction"]),
            normalization_mode=str(trim["normalization_mode"]),
        ),
    )
    detail = _selected_detail(
        "performance_leader",
        selected,
        returns,
        {candidate_id: (data, run, selection)},
        formal.daily[["return", "candidate"]],
        extreme.selection_mask.astype(bool),
        context,
        output,
        config,
    )
    neighborhood = _local_numeric_neighborhood(table, selected, config)
    neighborhood.to_csv(output / "performance_leader_local_neighborhood.csv")
    summary = {
        "count": int(len(neighborhood)),
        **{
            f"{field}_q25": float(neighborhood[field].quantile(0.25))
            for field in METRICS
        },
        **{
            f"{field}_median": float(neighborhood[field].median())
            for field in METRICS
        },
        **{
            f"{field}_minimum": float(neighborhood[field].min())
            for field in METRICS
        },
    }
    audit = {
        "selection_rule": (
            "maximize the minimum percentile rank, then mean percentile rank, "
            "across full/ordinary annualized return and Sharpe among candidates "
            "eligible under both pre-existing selection screens"
        ),
        "selected": detail,
        "local_numeric_neighborhood": summary,
        "evidence_status": config["experiment"]["evidence_status"],
        "production_status": "research_candidate_not_production",
    }
    (output / "performance_leader_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
