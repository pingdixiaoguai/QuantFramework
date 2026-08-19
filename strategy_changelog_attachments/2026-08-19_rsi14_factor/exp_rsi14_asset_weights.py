"""Asset-specific RSI14 weight research with a training-only selection gate."""

from __future__ import annotations

import itertools
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


OUTPUT_DIR = Path(__file__).resolve().parent
REPO_ROOT = OUTPUT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import exp_rsi14_strategy as base
from backtest.runner import run as run_official_backtest


PREFIX = "2026-08-19_rsi14_asset_weights"
WEIGHT_GRID = [0.0, 0.3, 0.6, 0.9]
REBALANCE_GRID = [1, 2, 3, 5]
SELECTED_WEIGHTS = {
    "510300.SH": 0.9,
    "159915.SZ": 0.3,
    "513100.SH": 0.3,
    "518880.SH": 0.3,
}
SELECTED_REBALANCE_DAYS = 2
SELECTION_COST_RATE = 0.0005
UNIFORM_WEIGHTS = {asset: 0.6 for asset in base.ASSET_POOL}


@dataclass(frozen=True)
class AssetWeightSpec:
    asset_weights: tuple[float, ...]
    rebalance_days: int

    def as_mapping(self, asset_pool: list[str]) -> dict[str, float]:
        return dict(zip(asset_pool, self.asset_weights, strict=True))


def asset_weight_targets(
    engine: base.FastResearchEngine,
    asset_weights: dict[str, float],
) -> np.ndarray:
    """Build centered inverse-RSI composite Top-1 targets."""
    quality = engine.q_values[20]
    rsi = engine.rsi_values
    weights = np.asarray(
        [asset_weights[asset] for asset in engine.asset_pool], dtype=float
    )
    targets = np.full(
        (len(engine.dates), engine.asset_count), np.nan, dtype=float
    )

    for row in range(len(engine.dates)):
        valid = np.isfinite(quality[row]) & np.isfinite(rsi[row])
        if not valid.any():
            continue
        indices = np.flatnonzero(valid)
        quality_rank = engine._ranks(quality[row], valid)
        inverse_rsi_rank = engine._ranks(rsi[row], valid, flip=True)
        rank_center = (len(indices) + 1.0) / 2.0
        scores = quality_rank + weights * (inverse_rsi_rank - rank_center)
        winner = indices[np.argmax(scores[indices])]
        targets[row] = 0.0
        targets[row, winner] = 1.0
    return targets


def simulate_targets(
    engine: base.FastResearchEngine,
    targets: np.ndarray,
    rebalance_days: int,
    cost_rate: float = 0.0,
) -> base.SimulationResult:
    """Replay precomputed targets with production execution semantics."""
    current = np.zeros(engine.asset_count, dtype=float)
    has_position = False
    entry_index = -1
    pending: np.ndarray | None = None
    pending_index = -1
    returns = np.full(len(engine.dates), np.nan, dtype=float)
    turnover = np.zeros(len(engine.dates), dtype=float)

    for row in range(len(engine.dates)):
        if row > 0:
            old = current.copy()
            opened_today = pending is not None and pending_index == row
            if opened_today:
                overnight = (
                    engine._weighted_return(
                        old, engine.closes[row - 1], engine.opens[row]
                    )
                    if has_position
                    else math.nan
                )
                current = pending
                has_position = bool(np.any(current))
                entry_index = row
                pending = None
                pending_index = -1
                turnover[row] = float(np.abs(current - old).sum())
                intraday = engine._weighted_return(
                    current, engine.opens[row], engine.closes[row]
                )
                parts = [
                    value for value in (overnight, intraday) if np.isfinite(value)
                ]
                gross_return = (
                    float(np.prod(np.asarray(parts) + 1.0) - 1.0)
                    if parts
                    else math.nan
                )
            elif has_position:
                gross_return = engine._weighted_return(
                    current, engine.closes[row - 1], engine.closes[row]
                )
            else:
                gross_return = math.nan

            if np.isfinite(gross_return):
                returns[row] = gross_return - turnover[row] * cost_rate

        holding_days = row - entry_index + 1 if has_position else None
        should_signal = pending is None and (
            not has_position
            or rebalance_days <= 1
            or holding_days >= rebalance_days
        )
        target = targets[row]
        if (
            should_signal
            and np.isfinite(target).all()
            and not np.array_equal(target, current)
            and row + 1 < len(engine.dates)
        ):
            pending = target.copy()
            pending_index = row + 1

    return base.SimulationResult(
        returns=pd.Series(returns, index=engine.dates, dtype=float).dropna(),
        turnover=pd.Series(turnover, index=engine.dates, dtype=float),
    )


def official_config(asset_weights: dict[str, float], cost_rate: float = 0.0) -> dict:
    return {
        "strategy_name": "quality_momentum_rsi14_asset_weighted_top1_research",
        "strategy_class": "strategy.composite_top1.CompositeTop1",
        "asset_pool": list(base.ASSET_POOL),
        "start": base.START,
        "end": base.END,
        "factors": [
            {
                "name": "quality_momentum",
                "weight": 1.0,
                "params": {"window": 20},
            },
            {
                "name": "rsi",
                "weight": 0.0,
                "asset_weights": asset_weights,
                "direction_flip": True,
                "center_rank": True,
                "params": {"window": 14},
            },
        ],
        "train_ratio": base.TRAIN_RATIO,
        "rebalance_days": SELECTED_REBALANCE_DAYS,
        "transaction_cost_rate": cost_rate,
    }


def period_row(
    label: str,
    result: base.SimulationResult,
    split_date: pd.Timestamp,
    weights: dict[str, float] | None = None,
) -> dict:
    row: dict[str, object] = {"label": label}
    if weights is not None:
        row.update({f"weight_{asset}": weights[asset] for asset in base.ASSET_POOL})
    for period, values in base.period_metrics(result.returns, split_date).items():
        for name, value in values.items():
            row[f"{period}_{name}"] = value
    row["full_annual_turnover"] = float(
        result.turnover.sum() / (len(result.returns) / 252.0)
    )
    return row


def training_search(
    engine: base.FastResearchEngine,
    split_date: pd.Timestamp,
) -> tuple[pd.DataFrame, AssetWeightSpec]:
    baseline = engine.run(base.BASELINE, SELECTION_COST_RATE).returns
    train_dates = baseline.index[baseline.index <= split_date]
    blocks = [pd.DatetimeIndex(block) for block in np.array_split(train_dates, 4)]
    baseline_train = base.metrics(baseline.loc[train_dates])
    baseline_blocks = [base.metrics(baseline.loc[block]) for block in blocks]
    rows = []

    for weight_tuple in itertools.product(WEIGHT_GRID, repeat=len(base.ASSET_POOL)):
        spec_weights = dict(zip(base.ASSET_POOL, weight_tuple, strict=True))
        targets = asset_weight_targets(engine, spec_weights)
        for rebalance_days in REBALANCE_GRID:
            result = simulate_targets(
                engine, targets, rebalance_days, SELECTION_COST_RATE
            )
            train_returns = result.returns[result.returns.index <= split_date]
            train_metrics = base.metrics(train_returns)
            annual_delta = (
                train_metrics["annual_return"] - baseline_train["annual_return"]
            )
            sharpe_delta = train_metrics["sharpe"] - baseline_train["sharpe"]
            block_annual_deltas = []
            block_sharpe_deltas = []
            for block, baseline_metrics in zip(
                blocks, baseline_blocks, strict=True
            ):
                candidate_metrics = base.metrics(
                    result.returns.loc[result.returns.index.intersection(block)]
                )
                block_annual_deltas.append(
                    candidate_metrics["annual_return"]
                    - baseline_metrics["annual_return"]
                )
                block_sharpe_deltas.append(
                    candidate_metrics["sharpe"] - baseline_metrics["sharpe"]
                )
            dual_blocks = sum(
                annual > 0.0 and sharpe > 0.0
                for annual, sharpe in zip(
                    block_annual_deltas, block_sharpe_deltas, strict=True
                )
            )
            median_annual_delta = float(np.median(block_annual_deltas))
            median_sharpe_delta = float(np.median(block_sharpe_deltas))
            robust_score = min(
                median_annual_delta / 0.05,
                median_sharpe_delta / 0.20,
            )
            train_joint_improvement = min(
                annual_delta / max(abs(baseline_train["annual_return"]), 0.01),
                sharpe_delta / max(abs(baseline_train["sharpe"]), 0.01),
            )
            row = {
                **{
                    f"weight_{asset}": weight
                    for asset, weight in spec_weights.items()
                },
                "rebalance_days": rebalance_days,
                "selection_cost_rate": SELECTION_COST_RATE,
                "train_annual_return": train_metrics["annual_return"],
                "train_sharpe": train_metrics["sharpe"],
                "train_max_drawdown": train_metrics["max_drawdown"],
                "train_annual_return_delta": annual_delta,
                "train_sharpe_delta": sharpe_delta,
                "dual_improvement_blocks": dual_blocks,
                "median_block_annual_return_delta": median_annual_delta,
                "median_block_sharpe_delta": median_sharpe_delta,
                "robust_score": robust_score,
                "train_joint_improvement": train_joint_improvement,
                "weight_std": float(np.std(weight_tuple)),
            }
            for index, (annual, sharpe) in enumerate(
                zip(block_annual_deltas, block_sharpe_deltas, strict=True),
                start=1,
            ):
                row[f"block_{index}_annual_return_delta"] = annual
                row[f"block_{index}_sharpe_delta"] = sharpe
            rows.append(row)

    grid = pd.DataFrame(rows)
    eligible = grid[
        (grid["train_annual_return_delta"] > 0.0)
        & (grid["train_sharpe_delta"] > 0.0)
        & (grid["dual_improvement_blocks"] >= 3)
        & (grid["weight_std"] > 0.0)
    ].copy()
    ranked = eligible.sort_values(
        [
            "robust_score",
            "dual_improvement_blocks",
            "train_joint_improvement",
            "weight_std",
        ],
        ascending=[False, False, False, True],
    )
    winner = ranked.iloc[0]
    selected = AssetWeightSpec(
        tuple(float(winner[f"weight_{asset}"]) for asset in base.ASSET_POOL),
        int(winner["rebalance_days"]),
    )
    expected = AssetWeightSpec(
        tuple(SELECTED_WEIGHTS[asset] for asset in base.ASSET_POOL),
        SELECTED_REBALANCE_DAYS,
    )
    if selected != expected:
        raise RuntimeError(f"training-only selection changed: {selected} != {expected}")
    grid["eligible"] = grid.index.isin(eligible.index)
    grid["selected_on_train"] = grid.index == winner.name
    return grid, selected


def local_neighborhood(
    engine: base.FastResearchEngine,
    baseline: pd.Series,
    split_date: pd.Timestamp,
) -> pd.DataFrame:
    choices = []
    for asset in base.ASSET_POOL:
        selected = SELECTED_WEIGHTS[asset]
        choices.append([
            value
            for value in WEIGHT_GRID
            if abs(value - selected) <= 0.3000001
        ])
    baseline_periods = base.period_metrics(baseline, split_date)
    rows = []
    for weight_tuple in itertools.product(*choices):
        weights = dict(zip(base.ASSET_POOL, weight_tuple, strict=True))
        result = simulate_targets(
            engine,
            asset_weight_targets(engine, weights),
            SELECTED_REBALANCE_DAYS,
            SELECTION_COST_RATE,
        )
        row = period_row("neighbor", result, split_date, weights)
        row["is_selected"] = weights == SELECTED_WEIGHTS
        row["dual_improvement_train_and_test"] = all(
            row[f"{period}_annual_return"]
            > baseline_periods[period]["annual_return"]
            and row[f"{period}_sharpe"]
            > baseline_periods[period]["sharpe"]
            for period in ["train", "test"]
        )
        rows.append(row)
    return pd.DataFrame(rows)


def weight_permutations(
    engine: base.FastResearchEngine,
    baseline: pd.Series,
    split_date: pd.Timestamp,
) -> pd.DataFrame:
    baseline_full = base.metrics(baseline)
    rows = []
    for weight_tuple in sorted(set(itertools.permutations(SELECTED_WEIGHTS.values()))):
        weights = dict(zip(base.ASSET_POOL, weight_tuple, strict=True))
        result = simulate_targets(
            engine,
            asset_weight_targets(engine, weights),
            SELECTED_REBALANCE_DAYS,
            SELECTION_COST_RATE,
        )
        row = period_row("permutation", result, split_date, weights)
        row["is_selected_mapping"] = weights == SELECTED_WEIGHTS
        row["full_joint_improvement"] = min(
            row["full_annual_return"] / baseline_full["annual_return"] - 1.0,
            row["full_sharpe"] / baseline_full["sharpe"] - 1.0,
        )
        rows.append(row)
    frame = pd.DataFrame(rows).sort_values(
        ["full_joint_improvement", "full_sharpe"], ascending=False
    )
    frame["full_rank"] = np.arange(1, len(frame) + 1)
    return frame


def leave_one_asset_out() -> pd.DataFrame:
    rows = []
    for excluded in base.ASSET_POOL:
        pool = [asset for asset in base.ASSET_POOL if asset != excluded]
        engine = base.FastResearchEngine(pool, base.START, base.END, [20])
        baseline = engine.run(base.BASELINE, SELECTION_COST_RATE).returns
        selected_weights = {asset: SELECTED_WEIGHTS[asset] for asset in pool}
        selected = simulate_targets(
            engine,
            asset_weight_targets(engine, selected_weights),
            SELECTED_REBALANCE_DAYS,
            SELECTION_COST_RATE,
        ).returns
        baseline_metrics = base.metrics(baseline)
        selected_metrics = base.metrics(selected)
        rows.append({
            "excluded_asset": excluded,
            "asset_pool": "|".join(pool),
            "baseline_annual_return": baseline_metrics["annual_return"],
            "selected_annual_return": selected_metrics["annual_return"],
            "annual_return_delta": (
                selected_metrics["annual_return"]
                - baseline_metrics["annual_return"]
            ),
            "baseline_sharpe": baseline_metrics["sharpe"],
            "selected_sharpe": selected_metrics["sharpe"],
            "sharpe_delta": selected_metrics["sharpe"] - baseline_metrics["sharpe"],
            "dual_improvement": (
                selected_metrics["annual_return"]
                > baseline_metrics["annual_return"]
                and selected_metrics["sharpe"] > baseline_metrics["sharpe"]
            ),
        })
    return pd.DataFrame(rows)


def save_csv(frame: pd.DataFrame, slug: str) -> None:
    frame.to_csv(OUTPUT_DIR / f"{PREFIX}_{slug}.csv", index=False)


def main() -> None:
    engine = base.FastResearchEngine(
        base.ASSET_POOL, base.START, base.END, [20]
    )
    split_date = engine.dates[int(len(engine.dates) * base.TRAIN_RATIO)]
    grid, selected_spec = training_search(engine, split_date)
    save_csv(grid, "training_grid")

    baseline_result = engine.run(base.BASELINE)
    baseline_selection_cost_result = engine.run(
        base.BASELINE, SELECTION_COST_RATE
    )
    uniform_targets = asset_weight_targets(engine, UNIFORM_WEIGHTS)
    uniform_result = simulate_targets(engine, uniform_targets, 1)
    pd.testing.assert_series_equal(
        uniform_result.returns,
        engine.run(base.SELECTED).returns,
        check_names=False,
        rtol=0.0,
        atol=1e-14,
    )

    selected_weights = selected_spec.as_mapping(base.ASSET_POOL)
    selected_targets = asset_weight_targets(engine, selected_weights)
    selected_result = simulate_targets(
        engine, selected_targets, selected_spec.rebalance_days
    )
    selected_selection_cost_result = simulate_targets(
        engine,
        selected_targets,
        selected_spec.rebalance_days,
        SELECTION_COST_RATE,
    )
    official = run_official_backtest(official_config(selected_weights)).daily_returns
    pd.testing.assert_series_equal(
        selected_result.returns,
        official,
        check_names=False,
        rtol=0.0,
        atol=1e-14,
    )

    comparison = pd.DataFrame([
        period_row("baseline", baseline_result, split_date),
        period_row("uniform_rsi_0_6", uniform_result, split_date, UNIFORM_WEIGHTS),
        period_row(
            "asset_weighted_selected",
            selected_result,
            split_date,
            selected_weights,
        ),
    ])
    save_csv(comparison, "comparison")
    selection_cost_comparison = pd.DataFrame([
        period_row(
            "baseline_5bp",
            baseline_selection_cost_result,
            split_date,
        ),
        period_row(
            "asset_weighted_selected_5bp",
            selected_selection_cost_result,
            split_date,
            selected_weights,
        ),
    ])
    save_csv(selection_cost_comparison, "selection_cost_comparison")

    cost_rows = []
    for cost_rate in base.COST_RATES:
        strategies = [
            ("baseline", engine.run(base.BASELINE, cost_rate)),
            (
                "uniform_rsi_0_6",
                simulate_targets(engine, uniform_targets, 1, cost_rate),
            ),
            (
                "asset_weighted_selected",
                simulate_targets(
                    engine,
                    selected_targets,
                    selected_spec.rebalance_days,
                    cost_rate,
                ),
            ),
        ]
        for label, result in strategies:
            cost_rows.append({
                "strategy": label,
                "one_way_cost_rate": cost_rate,
                **base.metrics(result.returns),
            })
    save_csv(pd.DataFrame(cost_rows), "cost_stress")

    rolling_gross = base.rolling_comparison(
        baseline_result.returns, selected_result.returns
    )
    save_csv(rolling_gross, "rolling_36m_gross")
    rolling_costed = base.rolling_comparison(
        baseline_selection_cost_result.returns,
        selected_selection_cost_result.returns,
    )
    save_csv(rolling_costed, "rolling_36m_cost_5bp")
    yearly_gross = base.yearly_comparison(
        baseline_result.returns, selected_result.returns
    )
    save_csv(yearly_gross, "yearly_gross")
    yearly_costed = base.yearly_comparison(
        baseline_selection_cost_result.returns,
        selected_selection_cost_result.returns,
    )
    save_csv(yearly_costed, "yearly_cost_5bp")
    bootstrap_samples, bootstrap_summary = base.paired_block_bootstrap(
        baseline_selection_cost_result.returns,
        selected_selection_cost_result.returns,
    )
    save_csv(bootstrap_samples, "paired_block_bootstrap_samples")
    save_csv(bootstrap_summary, "paired_block_bootstrap_summary")

    neighborhood = local_neighborhood(
        engine, baseline_selection_cost_result.returns, split_date
    )
    save_csv(neighborhood, "local_neighborhood")
    permutations = weight_permutations(
        engine, baseline_selection_cost_result.returns, split_date
    )
    save_csv(permutations, "weight_permutations")
    leave_one_out = leave_one_asset_out()
    save_csv(leave_one_out, "leave_one_asset_out")

    selected_permutation_rank = int(
        permutations.loc[permutations["is_selected_mapping"], "full_rank"].iloc[0]
    )
    checks = pd.DataFrame([
        {
            "check": "official_engine_pointwise_reproduction",
            "value": True,
            "detail": f"selected matches {len(official)} official daily returns",
        },
        {
            "check": "training_grid_trial_count",
            "value": len(grid),
            "detail": "test metrics were not used for selection",
        },
        {
            "check": "training_eligible_count",
            "value": int(grid["eligible"].sum()),
            "detail": "non-uniform candidates passing the training gate",
        },
        {
            "check": "holdout_dual_improvement_at_5bp",
            "value": bool(
                selection_cost_comparison.loc[
                    selection_cost_comparison["label"]
                    == "asset_weighted_selected_5bp",
                    "test_annual_return",
                ].iloc[0]
                > selection_cost_comparison.loc[
                    selection_cost_comparison["label"] == "baseline_5bp",
                    "test_annual_return",
                ].iloc[0]
                and selection_cost_comparison.loc[
                    selection_cost_comparison["label"]
                    == "asset_weighted_selected_5bp",
                    "test_sharpe",
                ].iloc[0]
                > selection_cost_comparison.loc[
                    selection_cost_comparison["label"] == "baseline_5bp",
                    "test_sharpe",
                ].iloc[0]
            ),
            "detail": "2022-09-01 through 2026-08-17 at one-way 5bp",
        },
        {
            "check": "rolling_36m_dual_improvement_rate_at_5bp",
            "value": float(rolling_costed["dual_improvement"].mean()),
            "detail": (
                f"{int(rolling_costed['dual_improvement'].sum())}"
                f"/{len(rolling_costed)} windows"
            ),
        },
        {
            "check": "local_neighbors_dual_train_and_test_rate",
            "value": float(neighborhood["dual_improvement_train_and_test"].mean()),
            "detail": (
                f"{int(neighborhood['dual_improvement_train_and_test'].sum())}"
                f"/{len(neighborhood)} neighbors"
            ),
        },
        {
            "check": "selected_mapping_full_rank_among_permutations",
            "value": selected_permutation_rank,
            "detail": f"rank among {len(permutations)} assignments of the same weights",
        },
        {
            "check": "leave_one_asset_out_dual_improvement_rate",
            "value": float(leave_one_out["dual_improvement"].mean()),
            "detail": f"{int(leave_one_out['dual_improvement'].sum())}/4 pools",
        },
        {
            "check": "selection_status",
            "value": "cost_aware_quasi_oos_shadow_only",
            "detail": "the holdout was seen in the prior uniform-weight research",
        },
    ])
    save_csv(checks, "overfit_checks")


if __name__ == "__main__":
    main()
