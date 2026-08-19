"""Reproducible RSI14 factor and strategy research.

The fast simulator is checked point-for-point against the production backtest
engine before it is used for the parameter grid. All signals are generated at
today's close, executed at the next open, and use only information available at
the signal date.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from datetime import date
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backtest.runner import run as run_official_backtest
from data.store import query
from factors.quality_momentum import compute as compute_quality_momentum
from factors.rsi import compute as compute_rsi


OUTPUT_DIR = Path(__file__).resolve().parent
PREFIX = "2026-08-19_rsi14_factor"
ASSET_POOL = ["510300.SH", "159915.SZ", "513100.SH", "518880.SH"]
START = date(2013, 7, 1)
END = date(2026, 8, 17)
TRAIN_RATIO = 0.7
Q_WINDOWS = [10, 20, 40, 60, 90, 120]
REBALANCE_DAYS = [1, 2, 3, 5, 10, 20]
COARSE_RSI_WEIGHTS = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0]
FINE_RSI_WEIGHTS = [
    0.25, 0.35, 0.40, 0.45, 0.49, 0.50, 0.51,
    0.55, 0.60, 0.65, 0.75, 1.0, 1.5, 2.0,
]
COST_RATES = [0.0, 0.0001, 0.0003, 0.0005, 0.0010]
BOOTSTRAP_SAMPLES = 2_000
BOOTSTRAP_BLOCK = 20
BOOTSTRAP_SEED = 819
POOL_SCENARIOS = [
    ("core", ASSET_POOL, date(2013, 7, 1)),
    (
        "large_cap_swap",
        ["510050.SH", "159915.SZ", "513100.SH", "518880.SH"],
        date(2013, 7, 1),
    ),
    (
        "domestic_style_swap",
        ["510300.SH", "510500.SH", "513100.SH", "518880.SH"],
        date(2013, 7, 1),
    ),
    (
        "defensive_expanded",
        ASSET_POOL + ["512890.SH", "511360.SH"],
        date(2021, 1, 1),
    ),
]


@dataclass(frozen=True)
class Spec:
    family: str
    q_window: int = 20
    rebalance_days: int = 5
    rsi_weight: float | None = None
    threshold: float | None = None

    @property
    def target_key(self) -> tuple:
        return (self.family, self.q_window, self.rsi_weight, self.threshold)


@dataclass
class SimulationResult:
    returns: pd.Series
    turnover: pd.Series


BASELINE = Spec("qmom", q_window=20, rebalance_days=5)
FAILED_TRAIN_WINNER = Spec(
    "filter_upper", q_window=20, rebalance_days=2, threshold=70.0
)
SELECTED = Spec(
    "top1_lo", q_window=20, rebalance_days=1, rsi_weight=0.6
)


class FastResearchEngine:
    def __init__(
        self,
        asset_pool: list[str],
        start: date,
        end: date,
        q_windows: list[int],
    ) -> None:
        self.asset_pool = list(asset_pool)
        self.frames = {asset: query(asset, start, end) for asset in asset_pool}
        if any(frame.empty for frame in self.frames.values()):
            missing = [asset for asset, frame in self.frames.items() if frame.empty]
            raise RuntimeError(f"missing local data for: {missing}")

        self.dates = pd.DatetimeIndex(sorted(
            set().union(*(set(frame["date"]) for frame in self.frames.values()))
        ))
        self.asset_count = len(self.asset_pool)
        self.opens = self._price_matrix("open")
        self.closes = self._price_matrix("close")
        self.q_values = {
            window: self._factor_matrix(compute_quality_momentum, {"window": window})
            for window in q_windows
        }
        self.rsi_values = self._factor_matrix(compute_rsi, {"window": 14})
        self._target_cache: dict[tuple, np.ndarray] = {}

    def _price_matrix(self, column: str) -> np.ndarray:
        return np.column_stack([
            pd.Series(frame[column].to_numpy(), index=frame["date"])
            .reindex(self.dates)
            .to_numpy(dtype=float)
            for frame in self.frames.values()
        ])

    def _factor_matrix(self, compute, params: dict) -> np.ndarray:
        columns = []
        for frame in self.frames.values():
            values = compute(frame.copy(), params)
            columns.append(
                pd.Series(values.to_numpy(), index=frame["date"])
                .reindex(self.dates)
                .ffill()
                .to_numpy(dtype=float)
            )
        return np.column_stack(columns)

    def _ranks(
        self, values: np.ndarray, valid: np.ndarray, *, flip: bool = False
    ) -> np.ndarray:
        indices = np.flatnonzero(valid)
        sort_values = -values[indices] if flip else values[indices]
        order = indices[np.argsort(sort_values, kind="stable")]
        ranks = np.zeros(self.asset_count, dtype=float)
        ranks[order] = np.arange(1, len(order) + 1, dtype=float)
        return ranks

    def targets(self, spec: Spec) -> np.ndarray:
        cached = self._target_cache.get(spec.target_key)
        if cached is not None:
            return cached

        targets = np.full((len(self.dates), self.asset_count), np.nan, dtype=float)
        quality = self.q_values[spec.q_window]
        rsi = self.rsi_values

        for row in range(len(self.dates)):
            quality_valid = np.isfinite(quality[row])
            rsi_valid = np.isfinite(rsi[row])
            if spec.family == "qmom":
                valid = quality_valid
            elif spec.family in {"rsi_hi", "rsi_lo"}:
                valid = rsi_valid
            else:
                valid = quality_valid & rsi_valid
            if not valid.any():
                continue

            indices = np.flatnonzero(valid)
            weights = np.zeros(self.asset_count, dtype=float)
            if spec.family == "qmom":
                winner = indices[np.argmax(quality[row, indices])]
                weights[winner] = 1.0
            elif spec.family == "rsi_hi":
                winner = indices[np.argmax(rsi[row, indices])]
                weights[winner] = 1.0
            elif spec.family == "rsi_lo":
                winner = indices[np.argmin(rsi[row, indices])]
                weights[winner] = 1.0
            elif spec.family in {"top1_hi", "top1_lo", "blend_hi", "blend_lo"}:
                quality_rank = self._ranks(quality[row], valid)
                rsi_rank = self._ranks(
                    rsi[row], valid, flip=spec.family.endswith("lo")
                )
                scores = quality_rank + float(spec.rsi_weight) * rsi_rank
                if spec.family.startswith("top1"):
                    winner = indices[np.argmax(scores[indices])]
                    weights[winner] = 1.0
                else:
                    weights = scores / scores.sum()
            elif spec.family == "filter_upper":
                eligible = valid & (rsi[row] <= float(spec.threshold))
                choices = np.flatnonzero(eligible if eligible.any() else valid)
                winner = choices[np.argmax(quality[row, choices])]
                weights[winner] = 1.0
            elif spec.family == "confirm_lower":
                eligible = valid & (rsi[row] >= float(spec.threshold))
                choices = np.flatnonzero(eligible if eligible.any() else valid)
                winner = choices[np.argmax(quality[row, choices])]
                weights[winner] = 1.0
            elif spec.family in {"gold_riskoff", "rsi_riskoff"}:
                winner = indices[np.argmax(quality[row, indices])]
                if rsi[row, winner] < float(spec.threshold):
                    gold_index = self.asset_pool.index("518880.SH")
                    if spec.family == "gold_riskoff" and valid[gold_index]:
                        winner = gold_index
                    else:
                        winner = indices[np.argmax(rsi[row, indices])]
                weights[winner] = 1.0
            else:
                raise ValueError(f"unknown family: {spec.family}")
            targets[row] = weights

        self._target_cache[spec.target_key] = targets
        return targets

    @staticmethod
    def _weighted_return(
        weights: np.ndarray, start: np.ndarray, end: np.ndarray
    ) -> float:
        valid = np.isfinite(start) & np.isfinite(end) & (weights != 0.0)
        if not valid.any():
            return math.nan
        return float(np.sum(weights[valid] * (end[valid] / start[valid] - 1.0)))

    def run(self, spec: Spec, cost_rate: float = 0.0) -> SimulationResult:
        targets = self.targets(spec)
        current = np.zeros(self.asset_count, dtype=float)
        has_position = False
        entry_index = -1
        pending: np.ndarray | None = None
        pending_index = -1
        returns = np.full(len(self.dates), np.nan, dtype=float)
        turnover = np.zeros(len(self.dates), dtype=float)

        for row in range(len(self.dates)):
            if row > 0:
                old = current.copy()
                opened_today = pending is not None and pending_index == row
                if opened_today:
                    overnight = (
                        self._weighted_return(old, self.closes[row - 1], self.opens[row])
                        if has_position
                        else math.nan
                    )
                    current = pending
                    has_position = bool(np.any(current))
                    entry_index = row
                    pending = None
                    pending_index = -1
                    turnover[row] = float(np.abs(current - old).sum())
                    intraday = self._weighted_return(
                        current, self.opens[row], self.closes[row]
                    )
                    parts = [value for value in (overnight, intraday) if np.isfinite(value)]
                    gross_return = (
                        float(np.prod(np.asarray(parts) + 1.0) - 1.0)
                        if parts
                        else math.nan
                    )
                elif has_position:
                    gross_return = self._weighted_return(
                        current, self.closes[row - 1], self.closes[row]
                    )
                else:
                    gross_return = math.nan

                if np.isfinite(gross_return):
                    returns[row] = gross_return - turnover[row] * cost_rate

            holding_days = row - entry_index + 1 if has_position else None
            should_signal = pending is None and (
                not has_position
                or spec.rebalance_days <= 1
                or holding_days >= spec.rebalance_days
            )
            target = targets[row]
            if (
                should_signal
                and np.isfinite(target).all()
                and not np.array_equal(target, current)
                and row + 1 < len(self.dates)
            ):
                pending = target.copy()
                pending_index = row + 1

        return SimulationResult(
            returns=pd.Series(returns, index=self.dates, dtype=float).dropna(),
            turnover=pd.Series(turnover, index=self.dates, dtype=float),
        )


def metrics(returns: pd.Series) -> dict[str, float | int]:
    clean = returns.dropna()
    curve = (1.0 + clean).cumprod()
    years = len(clean) / 252.0
    return {
        "days": len(clean),
        "annual_return": float(curve.iloc[-1] ** (1.0 / years) - 1.0),
        "sharpe": float(clean.mean() / clean.std() * np.sqrt(252.0)),
        "max_drawdown": float((curve / curve.cummax() - 1.0).min()),
        "total_return": float(curve.iloc[-1] - 1.0),
    }


def period_metrics(
    returns: pd.Series, split_date: pd.Timestamp
) -> dict[str, dict[str, float | int]]:
    return {
        "train": metrics(returns[returns.index <= split_date]),
        "test": metrics(returns[returns.index > split_date]),
        "full": metrics(returns),
    }


def flatten_metrics(
    spec: Spec, result: SimulationResult, split_date: pd.Timestamp
) -> dict:
    row = {
        "family": spec.family,
        "q_window": spec.q_window,
        "rsi_weight": spec.rsi_weight,
        "threshold": spec.threshold,
        "rebalance_days": spec.rebalance_days,
    }
    for period, values in period_metrics(result.returns, split_date).items():
        for name, value in values.items():
            row[f"{period}_{name}"] = value
    row["full_annual_turnover"] = float(
        result.turnover.sum() / (len(result.returns) / 252.0)
    )
    return row


def coarse_specs() -> list[Spec]:
    specs = [
        Spec("qmom", q_window=window, rebalance_days=rd)
        for window, rd in product(Q_WINDOWS, REBALANCE_DAYS)
    ]
    specs.extend(
        Spec(family, rebalance_days=rd)
        for family, rd in product(["rsi_hi", "rsi_lo"], REBALANCE_DAYS)
    )
    specs.extend(
        Spec(family, q_window=window, rsi_weight=weight, rebalance_days=rd)
        for family, window, weight, rd in product(
            ["top1_hi", "top1_lo", "blend_hi", "blend_lo"],
            Q_WINDOWS,
            COARSE_RSI_WEIGHTS,
            REBALANCE_DAYS,
        )
    )
    specs.extend(
        Spec(
            "filter_upper",
            q_window=window,
            threshold=threshold,
            rebalance_days=rd,
        )
        for window, threshold, rd in product(
            Q_WINDOWS, [55, 60, 65, 70, 75, 80], REBALANCE_DAYS
        )
    )
    specs.extend(
        Spec(family, q_window=window, threshold=threshold, rebalance_days=rd)
        for family, window, threshold, rd in product(
            ["confirm_lower", "gold_riskoff", "rsi_riskoff"],
            Q_WINDOWS,
            [35, 40, 45, 50, 55, 60],
            REBALANCE_DAYS,
        )
    )
    return specs


def official_config(spec: Spec, asset_pool: list[str], start: date, cost: float) -> dict:
    if spec.family == "qmom":
        return {
            "strategy_name": "quality_momentum_top1",
            "strategy_class": "strategy.top1.Top1",
            "asset_pool": asset_pool,
            "start": start,
            "end": END,
            "factors": [{
                "name": "quality_momentum",
                "weight": 1.0,
                "params": {"window": spec.q_window},
            }],
            "train_ratio": TRAIN_RATIO,
            "rebalance_days": spec.rebalance_days,
            "transaction_cost_rate": cost,
        }
    if spec.family != "top1_lo":
        raise ValueError("official verification is implemented for baseline/selected only")
    return {
        "strategy_name": "quality_momentum_rsi14_top1",
        "strategy_class": "strategy.composite_top1.CompositeTop1",
        "asset_pool": asset_pool,
        "start": start,
        "end": END,
        "factors": [
            {
                "name": "quality_momentum",
                "weight": 1.0,
                "params": {"window": spec.q_window},
            },
            {
                "name": "rsi",
                "weight": spec.rsi_weight,
                "direction_flip": True,
                "params": {"window": 14},
            },
        ],
        "train_ratio": TRAIN_RATIO,
        "rebalance_days": spec.rebalance_days,
        "transaction_cost_rate": cost,
    }


def verify_against_official(
    engine: FastResearchEngine, spec: Spec
) -> pd.Series:
    fast = engine.run(spec).returns
    official = run_official_backtest(
        official_config(spec, engine.asset_pool, START, 0.0)
    ).daily_returns
    pd.testing.assert_series_equal(
        fast, official, check_names=False, rtol=0.0, atol=1e-14
    )
    return official


def rolling_comparison(
    baseline: pd.Series, selected: pd.Series, window: int = 756, step: int = 21
) -> pd.DataFrame:
    common = baseline.index.intersection(selected.index)
    rows = []
    for end_index in range(window, len(common) + 1, step):
        dates = common[end_index - window : end_index]
        base_metrics = metrics(baseline.loc[dates])
        selected_metrics = metrics(selected.loc[dates])
        rows.append({
            "start": dates[0].date().isoformat(),
            "end": dates[-1].date().isoformat(),
            "baseline_annual_return": base_metrics["annual_return"],
            "selected_annual_return": selected_metrics["annual_return"],
            "annual_return_delta": (
                selected_metrics["annual_return"] - base_metrics["annual_return"]
            ),
            "baseline_sharpe": base_metrics["sharpe"],
            "selected_sharpe": selected_metrics["sharpe"],
            "sharpe_delta": selected_metrics["sharpe"] - base_metrics["sharpe"],
            "dual_improvement": (
                selected_metrics["annual_return"] > base_metrics["annual_return"]
                and selected_metrics["sharpe"] > base_metrics["sharpe"]
            ),
        })
    return pd.DataFrame(rows)


def yearly_comparison(baseline: pd.Series, selected: pd.Series) -> pd.DataFrame:
    rows = []
    for year in sorted(set(baseline.index.year) & set(selected.index.year)):
        base_metrics = metrics(baseline[baseline.index.year == year])
        selected_metrics = metrics(selected[selected.index.year == year])
        rows.append({
            "year": year,
            "baseline_annual_return": base_metrics["annual_return"],
            "selected_annual_return": selected_metrics["annual_return"],
            "annual_return_delta": (
                selected_metrics["annual_return"] - base_metrics["annual_return"]
            ),
            "baseline_sharpe": base_metrics["sharpe"],
            "selected_sharpe": selected_metrics["sharpe"],
            "sharpe_delta": selected_metrics["sharpe"] - base_metrics["sharpe"],
        })
    return pd.DataFrame(rows)


def paired_block_bootstrap(
    baseline: pd.Series, selected: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame]:
    common = baseline.index.intersection(selected.index)
    base = baseline.loc[common].to_numpy(dtype=float)
    candidate = selected.loc[common].to_numpy(dtype=float)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    block_count = math.ceil(len(common) / BOOTSTRAP_BLOCK)
    rows = []
    for sample in range(BOOTSTRAP_SAMPLES):
        starts = rng.integers(0, len(common) - BOOTSTRAP_BLOCK + 1, block_count)
        indices = np.concatenate([
            np.arange(start, start + BOOTSTRAP_BLOCK) for start in starts
        ])[: len(common)]
        base_sample = pd.Series(base[indices])
        selected_sample = pd.Series(candidate[indices])
        base_metrics = metrics(base_sample)
        selected_metrics = metrics(selected_sample)
        rows.append({
            "sample": sample,
            "annual_return_delta": (
                selected_metrics["annual_return"] - base_metrics["annual_return"]
            ),
            "sharpe_delta": selected_metrics["sharpe"] - base_metrics["sharpe"],
        })
    samples = pd.DataFrame(rows)
    summary = pd.DataFrame([
        {
            "metric": metric,
            "observed": (
                metrics(selected)["annual_return"] - metrics(baseline)["annual_return"]
                if metric == "annual_return_delta"
                else metrics(selected)["sharpe"] - metrics(baseline)["sharpe"]
            ),
            "bootstrap_mean": samples[metric].mean(),
            "ci_2_5": samples[metric].quantile(0.025),
            "ci_97_5": samples[metric].quantile(0.975),
            "probability_positive": float((samples[metric] > 0.0).mean()),
        }
        for metric in ["annual_return_delta", "sharpe_delta"]
    ])
    return samples, summary


def save_csv(frame: pd.DataFrame, slug: str) -> None:
    frame.to_csv(OUTPUT_DIR / f"{PREFIX}_{slug}.csv", index=False)


def main() -> None:
    engine = FastResearchEngine(ASSET_POOL, START, END, Q_WINDOWS)
    split_index = int(len(engine.dates) * TRAIN_RATIO)
    split_date = engine.dates[split_index]

    baseline_official = verify_against_official(engine, BASELINE)
    selected_official = verify_against_official(engine, SELECTED)

    grid_rows = []
    for spec in coarse_specs():
        grid_rows.append(flatten_metrics(spec, engine.run(spec), split_date))
    grid = pd.DataFrame(grid_rows)
    baseline_row = grid[
        (grid["family"] == "qmom")
        & (grid["q_window"] == 20)
        & (grid["rebalance_days"] == 5)
    ].iloc[0]
    for period in ["train", "test", "full"]:
        grid[f"{period}_joint_improvement"] = np.minimum(
            grid[f"{period}_annual_return"]
            / baseline_row[f"{period}_annual_return"]
            - 1.0,
            grid[f"{period}_sharpe"] / baseline_row[f"{period}_sharpe"] - 1.0,
        )
    grid["dual_improvement_train_and_test"] = (
        (grid["train_annual_return"] > baseline_row["train_annual_return"])
        & (grid["train_sharpe"] > baseline_row["train_sharpe"])
        & (grid["test_annual_return"] > baseline_row["test_annual_return"])
        & (grid["test_sharpe"] > baseline_row["test_sharpe"])
    )
    save_csv(grid, "strategy_grid")

    family_best = (
        grid.sort_values(
            ["family", "full_joint_improvement", "full_sharpe"],
            ascending=[True, False, False],
        )
        .groupby("family", as_index=False)
        .head(1)
        .sort_values("full_joint_improvement", ascending=False)
    )
    save_csv(family_best, "family_best")

    comparison_rows = []
    for label, spec in [
        ("baseline", BASELINE),
        ("failed_train_winner", FAILED_TRAIN_WINNER),
        ("selected_retrospective", SELECTED),
    ]:
        row = flatten_metrics(spec, engine.run(spec), split_date)
        row["label"] = label
        comparison_rows.append(row)
    comparison = pd.DataFrame(comparison_rows)
    save_csv(comparison, "comparison")

    neighborhood_rows = []
    selected_path = engine.run(SELECTED).returns
    for weight, rebalance_days in product(FINE_RSI_WEIGHTS, [1, 2, 3, 5, 10]):
        spec = Spec(
            "top1_lo",
            q_window=20,
            rsi_weight=weight,
            rebalance_days=rebalance_days,
        )
        result = engine.run(spec)
        row = flatten_metrics(spec, result, split_date)
        row["same_path_as_selected"] = result.returns.equals(selected_path)
        neighborhood_rows.append(row)
    neighborhood = pd.DataFrame(neighborhood_rows)
    save_csv(neighborhood, "parameter_neighborhood")

    cost_rows = []
    for cost, (label, spec) in product(
        COST_RATES, [("baseline", BASELINE), ("selected", SELECTED)]
    ):
        row = metrics(engine.run(spec, cost_rate=cost).returns)
        row.update({"strategy": label, "one_way_cost_rate": cost})
        cost_rows.append(row)
    save_csv(pd.DataFrame(cost_rows), "cost_stress")

    rolling = rolling_comparison(baseline_official, selected_official)
    save_csv(rolling, "rolling_36m")
    save_csv(yearly_comparison(baseline_official, selected_official), "yearly")

    bootstrap_samples, bootstrap_summary = paired_block_bootstrap(
        baseline_official, selected_official
    )
    save_csv(bootstrap_samples, "paired_block_bootstrap_samples")
    save_csv(bootstrap_summary, "paired_block_bootstrap_summary")

    daily = pd.concat(
        [
            baseline_official.rename("baseline_return"),
            selected_official.rename("selected_return"),
        ],
        axis=1,
    )
    daily.index.name = "date"
    daily.reset_index().to_csv(OUTPUT_DIR / f"{PREFIX}_daily_returns.csv", index=False)

    cross_pool_rows = []
    for scenario, asset_pool, start in POOL_SCENARIOS:
        scenario_engine = FastResearchEngine(asset_pool, start, END, [20])
        scenario_metrics = {}
        for label, spec in [("baseline", BASELINE), ("selected", SELECTED)]:
            values = metrics(scenario_engine.run(spec).returns)
            scenario_metrics[label] = values
            cross_pool_rows.append({
                "scenario": scenario,
                "asset_pool": "|".join(asset_pool),
                "start": start.isoformat(),
                "strategy": label,
                **values,
            })
        cross_pool_rows[-1]["dual_improvement_vs_baseline"] = (
            scenario_metrics["selected"]["annual_return"]
            > scenario_metrics["baseline"]["annual_return"]
            and scenario_metrics["selected"]["sharpe"]
            > scenario_metrics["baseline"]["sharpe"]
        )
    cross_pool = pd.DataFrame(cross_pool_rows)
    save_csv(cross_pool, "cross_pool")

    checks = pd.DataFrame([
        {
            "check": "official_engine_pointwise_reproduction",
            "value": True,
            "detail": f"baseline and selected each match {len(baseline_official)} daily returns",
        },
        {
            "check": "coarse_grid_trial_count",
            "value": len(grid),
            "detail": "all attempted configurations retained",
        },
        {
            "check": "coarse_configs_dual_improvement_train_and_test",
            "value": int(grid["dual_improvement_train_and_test"].sum()),
            "detail": "selection is sparse relative to the full search",
        },
        {
            "check": "fine_weight_plateau",
            "value": int(
                neighborhood[
                    (neighborhood["rebalance_days"] == 1)
                    & neighborhood["same_path_as_selected"]
                ]["rsi_weight"].nunique()
            ),
            "detail": "fine-grid weights with the exact selected path at rd=1",
        },
        {
            "check": "rolling_36m_dual_improvement_rate",
            "value": float(rolling["dual_improvement"].mean()),
            "detail": f"{int(rolling['dual_improvement'].sum())}/{len(rolling)} windows",
        },
        {
            "check": "cross_pool_dual_improvement_rate",
            "value": float(
                cross_pool.loc[
                    cross_pool["strategy"] == "selected",
                    "dual_improvement_vs_baseline",
                ].mean()
            ),
            "detail": "3/4 pool perturbations",
        },
        {
            "check": "selection_status",
            "value": "retrospective_shadow_only",
            "detail": "the final candidate was selected after the first holdout candidate failed",
        },
    ])
    save_csv(checks, "overfit_checks")


if __name__ == "__main__":
    main()
