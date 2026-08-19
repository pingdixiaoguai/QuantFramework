"""Research range-position percentiles, volume, and RSI as a QM20 overlay."""

from __future__ import annotations

import itertools
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


OUTPUT_DIR = Path(__file__).resolve().parent
REPO_ROOT = OUTPUT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import exp_rsi14_asset_weights as asset_research
import exp_rsi14_strategy as base
from backtest.runner import run as run_official_backtest
from run_backtest import _load_config_from_yaml


PREFIX = "2026-08-19_range_volume_rsi_overlay"
EVALUATION_START = pd.Timestamp("2016-01-01")
SELECTION_COST_RATE = 0.0005
X_WINDOWS = [10, 20, 40, 60]
HISTORY_WINDOWS = [126, 252, 504]
ALPHAS = [0.25, 0.50, 1.00]
REBALANCE_DAYS = [1, 2, 3, 5, 10]
SELECTED_X = 60
SELECTED_HISTORY = 504
SELECTED_DIRECTIONS = (1, -1, 1, -1)
SELECTED_MECHANISM = "pullback_high_volume_low_rsi"
SELECTED_ALPHA = 0.25
SELECTED_REBALANCE_DAYS = 10
SELECTED_CONFIG = (
    REPO_ROOT
    / "strategy/configs/quality_momentum_range_volume_rsi_top1_research.yaml"
)


@dataclass(frozen=True)
class FeatureSet:
    drawdown: np.ndarray
    rebound: np.ndarray
    volume: np.ndarray


def mechanism_specs() -> list[tuple[str, tuple[int, int, int, int]]]:
    specs = []
    for state_name, (drawdown, rebound) in [
        ("near_high", (-1, 1)),
        ("pullback", (1, -1)),
    ]:
        for volume_name, volume in [
            ("high_volume", 1),
            ("low_volume", -1),
        ]:
            for rsi_name, rsi in [("high_rsi", 1), ("low_rsi", -1)]:
                specs.append((
                    f"{state_name}_{volume_name}_{rsi_name}",
                    (drawdown, rebound, volume, rsi),
                ))
    return specs


def build_features(
    engine: base.FastResearchEngine,
    window: int,
    history: int,
) -> FeatureSet:
    columns: dict[str, list[np.ndarray]] = {
        "drawdown": [],
        "rebound": [],
        "volume": [],
    }
    for frame in engine.frames.values():
        close = frame["close"].astype(float)
        volume = frame["volume"].astype(float)
        rolling_high = close.rolling(window, min_periods=window).max()
        rolling_low = close.rolling(window, min_periods=window).min()
        raw_values = {
            "drawdown": (rolling_high - close) / rolling_high,
            "rebound": close / rolling_low - 1.0,
            "volume": volume / volume.rolling(window, min_periods=window).mean(),
        }
        for name, values in raw_values.items():
            percentile = values.rolling(
                history, min_periods=history
            ).rank(pct=True)
            columns[name].append(
                pd.Series(percentile.to_numpy(), index=frame["date"])
                .reindex(engine.dates)
                .ffill()
                .to_numpy(dtype=float)
            )
    return FeatureSet(
        drawdown=np.column_stack(columns["drawdown"]),
        rebound=np.column_stack(columns["rebound"]),
        volume=np.column_stack(columns["volume"]),
    )


def overlay_targets(
    engine: base.FastResearchEngine,
    features: FeatureSet,
    directions: tuple[int, int, int, int],
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    quality = engine.q_values[20]
    rsi = engine.rsi_values
    targets = np.full(
        (len(engine.dates), engine.asset_count), np.nan, dtype=float
    )
    scores = np.full_like(targets, np.nan)
    feature_arrays = [features.drawdown, features.rebound, features.volume]

    for row in range(len(engine.dates)):
        valid = np.isfinite(quality[row]) & np.isfinite(rsi[row])
        for values in feature_arrays:
            valid &= np.isfinite(values[row])
        if not valid.any():
            continue
        indices = np.flatnonzero(valid)
        quality_rank = engine._ranks(quality[row], valid)
        adjustment = alpha * (
            directions[0] * (features.drawdown[row] - 0.5) * 2.0
            + directions[1] * (features.rebound[row] - 0.5) * 2.0
            + directions[2] * (features.volume[row] - 0.5) * 2.0
            + directions[3] * (rsi[row] - 50.0) / 50.0
        )
        scores[row] = quality_rank + adjustment
        winner = indices[np.argmax(scores[row, indices])]
        targets[row] = 0.0
        targets[row, winner] = 1.0
    return targets, scores


def evaluation_returns(returns: pd.Series) -> pd.Series:
    return returns[returns.index >= EVALUATION_START]


def period_metrics(
    returns: pd.Series, split_date: pd.Timestamp
) -> dict[str, dict[str, float | int]]:
    evaluated = evaluation_returns(returns)
    return {
        "train": base.metrics(evaluated[evaluated.index <= split_date]),
        "test": base.metrics(evaluated[evaluated.index > split_date]),
        "full": base.metrics(evaluated),
    }


def period_row(
    label: str,
    result: base.SimulationResult,
    split_date: pd.Timestamp,
) -> dict:
    row: dict[str, object] = {"label": label}
    for period, values in period_metrics(result.returns, split_date).items():
        for name, value in values.items():
            row[f"{period}_{name}"] = value
    evaluated_turnover = result.turnover[result.turnover.index >= EVALUATION_START]
    evaluated_returns = evaluation_returns(result.returns)
    row["full_annual_turnover"] = float(
        evaluated_turnover.sum() / (len(evaluated_returns) / 252.0)
    )
    return row


def training_grid(
    engine: base.FastResearchEngine,
    feature_cache: dict[tuple[int, int], FeatureSet],
    split_date: pd.Timestamp,
) -> pd.DataFrame:
    current = engine.run(base.BASELINE, SELECTION_COST_RATE).returns
    train_dates = current.index[
        (current.index >= EVALUATION_START) & (current.index <= split_date)
    ]
    blocks = [pd.DatetimeIndex(block) for block in np.array_split(train_dates, 4)]
    current_metrics = base.metrics(current.loc[train_dates])
    current_blocks = [base.metrics(current.loc[block]) for block in blocks]
    matched_results = {
        rd: engine.run(base.Spec("qmom", q_window=20, rebalance_days=rd), SELECTION_COST_RATE)
        for rd in REBALANCE_DAYS
    }
    matched_metrics = {
        rd: base.metrics(
            result.returns.loc[result.returns.index.intersection(train_dates)]
        )
        for rd, result in matched_results.items()
    }
    matched_blocks = {
        rd: [
            base.metrics(
                result.returns.loc[result.returns.index.intersection(block)]
            )
            for block in blocks
        ]
        for rd, result in matched_results.items()
    }
    rows = []

    for (window, history), features in feature_cache.items():
        for mechanism, directions in mechanism_specs():
            for alpha in ALPHAS:
                targets, _ = overlay_targets(
                    engine, features, directions, alpha
                )
                for rebalance_days in REBALANCE_DAYS:
                    result = asset_research.simulate_targets(
                        engine,
                        targets,
                        rebalance_days,
                        SELECTION_COST_RATE,
                    )
                    candidate = base.metrics(
                        result.returns.loc[
                            result.returns.index.intersection(train_dates)
                        ]
                    )
                    annual_delta = (
                        candidate["annual_return"]
                        - current_metrics["annual_return"]
                    )
                    sharpe_delta = candidate["sharpe"] - current_metrics["sharpe"]
                    matched_annual_delta = (
                        candidate["annual_return"]
                        - matched_metrics[rebalance_days]["annual_return"]
                    )
                    matched_sharpe_delta = (
                        candidate["sharpe"]
                        - matched_metrics[rebalance_days]["sharpe"]
                    )
                    current_block_deltas = []
                    matched_block_deltas = []
                    for index, block in enumerate(blocks):
                        block_candidate = base.metrics(
                            result.returns.loc[
                                result.returns.index.intersection(block)
                            ]
                        )
                        current_block_deltas.append((
                            block_candidate["annual_return"]
                            - current_blocks[index]["annual_return"],
                            block_candidate["sharpe"]
                            - current_blocks[index]["sharpe"],
                        ))
                        matched_block_deltas.append((
                            block_candidate["annual_return"]
                            - matched_blocks[rebalance_days][index]["annual_return"],
                            block_candidate["sharpe"]
                            - matched_blocks[rebalance_days][index]["sharpe"],
                        ))
                    current_dual_blocks = sum(
                        annual > 0.0 and sharpe > 0.0
                        for annual, sharpe in current_block_deltas
                    )
                    matched_dual_blocks = sum(
                        annual > 0.0 and sharpe > 0.0
                        for annual, sharpe in matched_block_deltas
                    )
                    median_annual_delta = float(np.median([
                        delta[0] for delta in current_block_deltas
                    ]))
                    median_sharpe_delta = float(np.median([
                        delta[1] for delta in current_block_deltas
                    ]))
                    robust_score = min(
                        median_annual_delta / 0.05,
                        median_sharpe_delta / 0.20,
                    )
                    joint_improvement = min(
                        annual_delta
                        / max(abs(current_metrics["annual_return"]), 0.01),
                        sharpe_delta / max(abs(current_metrics["sharpe"]), 0.01),
                    )
                    row = {
                        "window": window,
                        "history": history,
                        "mechanism": mechanism,
                        "drawdown_direction": directions[0],
                        "rebound_direction": directions[1],
                        "volume_direction": directions[2],
                        "rsi_direction": directions[3],
                        "alpha": alpha,
                        "rebalance_days": rebalance_days,
                        "selection_cost_rate": SELECTION_COST_RATE,
                        "train_annual_return": candidate["annual_return"],
                        "train_sharpe": candidate["sharpe"],
                        "train_max_drawdown": candidate["max_drawdown"],
                        "current_annual_return_delta": annual_delta,
                        "current_sharpe_delta": sharpe_delta,
                        "matched_annual_return_delta": matched_annual_delta,
                        "matched_sharpe_delta": matched_sharpe_delta,
                        "current_dual_blocks": current_dual_blocks,
                        "matched_dual_blocks": matched_dual_blocks,
                        "median_current_annual_return_delta": median_annual_delta,
                        "median_current_sharpe_delta": median_sharpe_delta,
                        "robust_score": robust_score,
                        "current_joint_improvement": joint_improvement,
                    }
                    for index, (current_delta, matched_delta) in enumerate(
                        zip(
                            current_block_deltas,
                            matched_block_deltas,
                            strict=True,
                        ),
                        start=1,
                    ):
                        row[f"block_{index}_current_annual_delta"] = current_delta[0]
                        row[f"block_{index}_current_sharpe_delta"] = current_delta[1]
                        row[f"block_{index}_matched_annual_delta"] = matched_delta[0]
                        row[f"block_{index}_matched_sharpe_delta"] = matched_delta[1]
                    rows.append(row)

    grid = pd.DataFrame(rows)
    grid["current_gate"] = (
        (grid["current_annual_return_delta"] > 0.0)
        & (grid["current_sharpe_delta"] > 0.0)
        & (grid["current_dual_blocks"] >= 3)
    )
    grid["matched_gate"] = (
        (grid["matched_annual_return_delta"] > 0.0)
        & (grid["matched_sharpe_delta"] > 0.0)
        & (grid["matched_dual_blocks"] >= 3)
    )
    ranked = grid[grid["current_gate"]].sort_values(
        ["robust_score", "current_dual_blocks", "current_joint_improvement"],
        ascending=[False, False, False],
    )
    winner = ranked.iloc[0]
    expected = {
        "window": SELECTED_X,
        "history": SELECTED_HISTORY,
        "mechanism": SELECTED_MECHANISM,
        "alpha": SELECTED_ALPHA,
        "rebalance_days": SELECTED_REBALANCE_DAYS,
    }
    actual = {name: winner[name] for name in expected}
    if actual != expected:
        raise RuntimeError(f"training selection changed: {actual} != {expected}")
    if not bool(winner["matched_gate"]):
        raise RuntimeError("selected candidate failed the matched-rd attribution gate")
    grid["selected_on_train"] = grid.index == winner.name
    return grid


def local_neighborhood(
    engine: base.FastResearchEngine,
    feature_cache: dict[tuple[int, int], FeatureSet],
    split_date: pd.Timestamp,
) -> pd.DataFrame:
    current = engine.run(base.BASELINE, SELECTION_COST_RATE).returns
    rows = []
    for window, history, alpha, rebalance_days in itertools.product(
        [40, 60], [252, 504], [0.15, 0.25, 0.35, 0.50], [5, 10, 15]
    ):
        features = feature_cache.get((window, history))
        if features is None:
            features = build_features(engine, window, history)
        targets, _ = overlay_targets(
            engine, features, SELECTED_DIRECTIONS, alpha
        )
        result = asset_research.simulate_targets(
            engine, targets, rebalance_days, SELECTION_COST_RATE
        )
        matched = engine.run(
            base.Spec("qmom", q_window=20, rebalance_days=rebalance_days),
            SELECTION_COST_RATE,
        )
        row = {
            "window": window,
            "history": history,
            "alpha": alpha,
            "rebalance_days": rebalance_days,
            "is_selected": (
                window == SELECTED_X
                and history == SELECTED_HISTORY
                and alpha == SELECTED_ALPHA
                and rebalance_days == SELECTED_REBALANCE_DAYS
            ),
        }
        candidate_periods = period_metrics(result.returns, split_date)
        current_periods = period_metrics(current, split_date)
        matched_periods = period_metrics(matched.returns, split_date)
        for period in ["train", "test", "full"]:
            for name in ["annual_return", "sharpe", "max_drawdown"]:
                row[f"{period}_{name}"] = candidate_periods[period][name]
            row[f"{period}_dual_vs_current"] = (
                candidate_periods[period]["annual_return"]
                > current_periods[period]["annual_return"]
                and candidate_periods[period]["sharpe"]
                > current_periods[period]["sharpe"]
            )
            row[f"{period}_dual_vs_matched"] = (
                candidate_periods[period]["annual_return"]
                > matched_periods[period]["annual_return"]
                and candidate_periods[period]["sharpe"]
                > matched_periods[period]["sharpe"]
            )
        rows.append(row)
    return pd.DataFrame(rows)


def signal_ablation(
    engine: base.FastResearchEngine,
    features: FeatureSet,
    split_date: pd.Timestamp,
) -> pd.DataFrame:
    ablations = {
        "full": SELECTED_DIRECTIONS,
        "without_drawdown": (0, -1, 1, -1),
        "without_rebound": (1, 0, 1, -1),
        "without_volume": (1, -1, 0, -1),
        "without_rsi": (1, -1, 1, 0),
        "qmom_only": (0, 0, 0, 0),
    }
    matched_spec = base.Spec(
        "qmom", q_window=20, rebalance_days=SELECTED_REBALANCE_DAYS
    )
    matched = engine.run(matched_spec, SELECTION_COST_RATE)
    matched_targets = engine.targets(matched_spec)
    rows = []
    for name, directions in ablations.items():
        targets, _ = overlay_targets(
            engine, features, directions, SELECTED_ALPHA
        )
        result = asset_research.simulate_targets(
            engine,
            targets,
            SELECTED_REBALANCE_DAYS,
            SELECTION_COST_RATE,
        )
        row = period_row(name, result, split_date)
        valid = np.isfinite(targets).all(axis=1) & np.isfinite(matched_targets).all(axis=1)
        train = valid & (engine.dates >= EVALUATION_START) & (engine.dates <= split_date)
        test = valid & (engine.dates > split_date)
        row["train_target_diff_days"] = int(
            np.any(targets[train] != matched_targets[train], axis=1).sum()
        )
        row["test_target_diff_days"] = int(
            np.any(targets[test] != matched_targets[test], axis=1).sum()
        )
        rows.append(row)
    return pd.DataFrame(rows)


def target_difference_events(
    engine: base.FastResearchEngine,
    features: FeatureSet,
    targets: np.ndarray,
    scores: np.ndarray,
    split_date: pd.Timestamp,
) -> pd.DataFrame:
    matched_targets = engine.targets(
        base.Spec("qmom", q_window=20, rebalance_days=SELECTED_REBALANCE_DAYS)
    )
    valid = np.isfinite(targets).all(axis=1) & np.isfinite(matched_targets).all(axis=1)
    differs = valid & np.any(targets != matched_targets, axis=1)
    rows = []
    for row in np.flatnonzero(differs):
        matched_index = int(np.argmax(matched_targets[row]))
        candidate_index = int(np.argmax(targets[row]))
        record: dict[str, object] = {
            "date": engine.dates[row].date().isoformat(),
            "period": "train" if engine.dates[row] <= split_date else "test",
            "qmom_asset": engine.asset_pool[matched_index],
            "candidate_asset": engine.asset_pool[candidate_index],
            "qmom_asset_score": scores[row, matched_index],
            "candidate_asset_score": scores[row, candidate_index],
        }
        for label, index in [
            ("qmom", matched_index),
            ("candidate", candidate_index),
        ]:
            record[f"{label}_drawdown_percentile"] = features.drawdown[row, index]
            record[f"{label}_rebound_percentile"] = features.rebound[row, index]
            record[f"{label}_volume_percentile"] = features.volume[row, index]
            record[f"{label}_rsi"] = engine.rsi_values[row, index]
        rows.append(record)
    return pd.DataFrame(rows)


def leave_one_asset_out() -> pd.DataFrame:
    rows = []
    for excluded in base.ASSET_POOL:
        pool = [asset for asset in base.ASSET_POOL if asset != excluded]
        engine = base.FastResearchEngine(pool, base.START, base.END, [20])
        split_date = engine.dates[int(len(engine.dates) * base.TRAIN_RATIO)]
        features = build_features(engine, SELECTED_X, SELECTED_HISTORY)
        targets, _ = overlay_targets(
            engine, features, SELECTED_DIRECTIONS, SELECTED_ALPHA
        )
        candidate = asset_research.simulate_targets(
            engine,
            targets,
            SELECTED_REBALANCE_DAYS,
            SELECTION_COST_RATE,
        )
        current = engine.run(base.BASELINE, SELECTION_COST_RATE)
        matched = engine.run(
            base.Spec("qmom", q_window=20, rebalance_days=SELECTED_REBALANCE_DAYS),
            SELECTION_COST_RATE,
        )
        candidate_metrics = period_metrics(candidate.returns, split_date)["full"]
        current_metrics = period_metrics(current.returns, split_date)["full"]
        matched_metrics = period_metrics(matched.returns, split_date)["full"]
        rows.append({
            "excluded_asset": excluded,
            "asset_pool": "|".join(pool),
            "candidate_annual_return": candidate_metrics["annual_return"],
            "candidate_sharpe": candidate_metrics["sharpe"],
            "current_annual_return": current_metrics["annual_return"],
            "current_sharpe": current_metrics["sharpe"],
            "matched_annual_return": matched_metrics["annual_return"],
            "matched_sharpe": matched_metrics["sharpe"],
            "dual_vs_current": (
                candidate_metrics["annual_return"]
                > current_metrics["annual_return"]
                and candidate_metrics["sharpe"] > current_metrics["sharpe"]
            ),
            "dual_vs_matched": (
                candidate_metrics["annual_return"]
                > matched_metrics["annual_return"]
                and candidate_metrics["sharpe"] > matched_metrics["sharpe"]
            ),
        })
    return pd.DataFrame(rows)


def save_csv(frame: pd.DataFrame, slug: str) -> None:
    frame.to_csv(OUTPUT_DIR / f"{PREFIX}_{slug}.csv", index=False)


def main() -> None:
    engine = base.FastResearchEngine(base.ASSET_POOL, base.START, base.END, [20])
    split_date = engine.dates[int(len(engine.dates) * base.TRAIN_RATIO)]
    feature_cache = {
        (window, history): build_features(engine, window, history)
        for window, history in itertools.product(X_WINDOWS, HISTORY_WINDOWS)
    }
    grid = training_grid(engine, feature_cache, split_date)
    save_csv(grid, "training_grid")

    selected_features = feature_cache[(SELECTED_X, SELECTED_HISTORY)]
    selected_targets, selected_scores = overlay_targets(
        engine,
        selected_features,
        SELECTED_DIRECTIONS,
        SELECTED_ALPHA,
    )
    current_spec = base.BASELINE
    matched_spec = base.Spec(
        "qmom", q_window=20, rebalance_days=SELECTED_REBALANCE_DAYS
    )
    selected_costed = asset_research.simulate_targets(
        engine,
        selected_targets,
        SELECTED_REBALANCE_DAYS,
        SELECTION_COST_RATE,
    )
    official = run_official_backtest(
        _load_config_from_yaml(SELECTED_CONFIG)
    ).daily_returns
    pd.testing.assert_series_equal(
        selected_costed.returns,
        official,
        check_names=False,
        rtol=0.0,
        atol=1e-14,
    )

    comparison_rows = []
    cost_rows = []
    for cost_rate in base.COST_RATES:
        strategies = [
            ("current_qmom_rd5", engine.run(current_spec, cost_rate)),
            ("matched_qmom_rd10", engine.run(matched_spec, cost_rate)),
            (
                "range_volume_rsi_overlay",
                asset_research.simulate_targets(
                    engine,
                    selected_targets,
                    SELECTED_REBALANCE_DAYS,
                    cost_rate,
                ),
            ),
        ]
        for label, result in strategies:
            row = period_row(label, result, split_date)
            row["one_way_cost_rate"] = cost_rate
            cost_rows.append(row)
            if cost_rate in {0.0, SELECTION_COST_RATE}:
                comparison_rows.append(row)
    save_csv(pd.DataFrame(comparison_rows), "comparison")
    save_csv(pd.DataFrame(cost_rows), "cost_stress")

    current_costed = engine.run(current_spec, SELECTION_COST_RATE)
    matched_costed = engine.run(matched_spec, SELECTION_COST_RATE)
    evaluated_candidate = evaluation_returns(selected_costed.returns)
    evaluated_current = evaluation_returns(current_costed.returns)
    evaluated_matched = evaluation_returns(matched_costed.returns)

    rolling_current = base.rolling_comparison(
        evaluated_current, evaluated_candidate
    )
    rolling_current["benchmark"] = "current_qmom_rd5"
    rolling_matched = base.rolling_comparison(
        evaluated_matched, evaluated_candidate
    )
    rolling_matched["benchmark"] = "matched_qmom_rd10"
    rolling = pd.concat([rolling_current, rolling_matched], ignore_index=True)
    save_csv(rolling, "rolling_36m_cost_5bp")

    bootstrap_frames = []
    bootstrap_summaries = []
    for benchmark, returns in [
        ("current_qmom_rd5", evaluated_current),
        ("matched_qmom_rd10", evaluated_matched),
    ]:
        samples, summary = base.paired_block_bootstrap(
            returns, evaluated_candidate
        )
        samples["benchmark"] = benchmark
        summary["benchmark"] = benchmark
        bootstrap_frames.append(samples)
        bootstrap_summaries.append(summary)
    save_csv(pd.concat(bootstrap_frames, ignore_index=True), "bootstrap_samples")
    bootstrap_summary = pd.concat(bootstrap_summaries, ignore_index=True)
    save_csv(bootstrap_summary, "bootstrap_summary")

    neighborhood = local_neighborhood(
        engine, feature_cache, split_date
    )
    save_csv(neighborhood, "local_neighborhood")
    ablation = signal_ablation(engine, selected_features, split_date)
    save_csv(ablation, "signal_ablation")
    events = target_difference_events(
        engine,
        selected_features,
        selected_targets,
        selected_scores,
        split_date,
    )
    save_csv(events, "target_difference_events")
    leave_out = leave_one_asset_out()
    save_csv(leave_out, "leave_one_asset_out")

    selected_test = period_metrics(selected_costed.returns, split_date)["test"]
    matched_test = period_metrics(matched_costed.returns, split_date)["test"]
    current_rolling = rolling[rolling["benchmark"] == "current_qmom_rd5"]
    matched_rolling = rolling[rolling["benchmark"] == "matched_qmom_rd10"]
    selected_events_train = int((events["period"] == "train").sum())
    selected_events_test = int((events["period"] == "test").sum())
    selected_test_returns = evaluation_returns(selected_costed.returns).loc[
        split_date + pd.Timedelta(days=1) :
    ]
    matched_test_returns = evaluation_returns(matched_costed.returns).reindex(
        selected_test_returns.index
    )
    test_return_diff_days = int(
        (~np.isclose(selected_test_returns, matched_test_returns)).sum()
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
            "detail": "all four auxiliary signals are non-zero in every trial",
        },
        {
            "check": "current_gate_count",
            "value": int(grid["current_gate"].sum()),
            "detail": "candidates passing the current-strategy training gate",
        },
        {
            "check": "current_and_matched_gate_count",
            "value": int((grid["current_gate"] & grid["matched_gate"]).sum()),
            "detail": "candidates also passing the matched-rd attribution gate",
        },
        {
            "check": "selected_target_diff_days_train",
            "value": selected_events_train,
            "detail": "daily desired target differs from QM20 rd10",
        },
        {
            "check": "selected_target_diff_days_test",
            "value": selected_events_test,
            "detail": "one desired-target change did not alter realized returns",
        },
        {
            "check": "test_return_diff_days_vs_matched_rd",
            "value": test_return_diff_days,
            "detail": "zero realized holdout return differences versus QM20 rd10",
        },
        {
            "check": "test_dual_vs_matched_rd",
            "value": bool(
                selected_test["annual_return"] > matched_test["annual_return"]
                and selected_test["sharpe"] > matched_test["sharpe"]
            ),
            "detail": "primary OOS signal-increment gate",
        },
        {
            "check": "rolling_36m_dual_rate_vs_current",
            "value": float(current_rolling["dual_improvement"].mean()),
            "detail": f"{int(current_rolling['dual_improvement'].sum())}/{len(current_rolling)} windows",
        },
        {
            "check": "rolling_36m_dual_rate_vs_matched",
            "value": float(matched_rolling["dual_improvement"].mean()),
            "detail": f"{int(matched_rolling['dual_improvement'].sum())}/{len(matched_rolling)} windows",
        },
        {
            "check": "local_neighbors_dual_train_test_vs_matched_rate",
            "value": float(
                (
                    neighborhood["train_dual_vs_matched"]
                    & neighborhood["test_dual_vs_matched"]
                ).mean()
            ),
            "detail": f"local grid has {len(neighborhood)} variants",
        },
        {
            "check": "leave_one_out_dual_vs_matched_rate",
            "value": float(leave_out["dual_vs_matched"].mean()),
            "detail": f"{int(leave_out['dual_vs_matched'].sum())}/4 pools",
        },
        {
            "check": "selection_status",
            "value": "rejected_no_oos_signal_increment",
            "detail": "double win versus current is attributable to rd=10 in the holdout",
        },
    ])
    save_csv(checks, "overfit_checks")


if __name__ == "__main__":
    main()
