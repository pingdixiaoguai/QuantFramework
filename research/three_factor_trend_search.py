"""Broad, causal search around the production 20-day momentum score.

The 20-day point-to-point momentum term is fixed in every candidate.  This
research-only harness searches alternative path-quality definitions, their
hyperparameters, and several soft low-volatility integrations.  Candidate
selection is frozen using data through 2022-09-02 before the later sample is
evaluated.

Run from the repository root::

    uv run python research/three_factor_trend_search.py
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backtest.runner import run as run_official  # noqa: E402
from data import store  # noqa: E402
from run_backtest import _load_config_from_yaml  # noqa: E402


CORE = ("510300.SH", "159915.SZ", "513100.SH", "518880.SH")
SIMULATION_START = pd.Timestamp("2013-07-01")
EVALUATION_START = pd.Timestamp("2014-01-02")
DEVELOPMENT_END = pd.Timestamp("2018-12-31")
VALIDATION_START = pd.Timestamp("2019-01-01")
TRAIN_END = pd.Timestamp("2022-09-02")
REBALANCE_DAYS = 5
MAIN_FEE = 0.0001
STRESS_FEE = 0.0005
OUTPUT = ROOT / "experiments/20260821_three_factor_trend_search"
BASELINE_CONFIG = ROOT / "strategy/configs/quality_momentum_top1.yaml"

ER_WINDOWS = (5, 10, 15, 20, 30, 40, 60)
ER_POWERS = (0.50, 0.75, 1.00, 1.25, 1.50, 2.00)
CONSISTENCY_WINDOWS = (10, 20, 40)
CONSISTENCY_POWERS = (0.25, 0.50, 1.00)
VOLATILITY_WINDOWS = (5, 10, 20, 40)
VOLATILITY_QUANTILES = (0.60, 0.80, 0.90)
VOLATILITY_HISTORIES: tuple[str | int, ...] = ("expanding", 252, 756)
LOW_VOL_SHAPES = ("cap", "symmetric", "exponential", "linear")
LOW_VOL_POWERS = (0.25, 0.50, 1.00)
RANK_WEIGHTS = (0.10, 0.25, 0.50, 1.00)


@dataclass
class Simulation:
    gross: np.ndarray
    turnover: np.ndarray
    held: np.ndarray


@dataclass
class RetainedCandidate:
    row: dict[str, Any]
    score: pd.DataFrame
    targets: np.ndarray
    simulation: Simulation


def native_series(code: str, column: str) -> pd.Series:
    frame = store.read_local(code)
    if frame is None or frame.empty:
        raise RuntimeError(f"missing local data for {code}")
    frame = frame.sort_values("date")
    return pd.Series(
        frame[column].to_numpy(dtype=float),
        index=pd.DatetimeIndex(frame["date"]),
        name=code,
    )


def load_prices() -> dict[str, pd.DataFrame]:
    calendar = native_series(CORE[0], "close").loc[SIMULATION_START:].index
    panels = {
        field: pd.DataFrame(index=calendar, columns=CORE, dtype=float)
        for field in ("open", "close")
    }
    for code in CORE:
        for field in panels:
            panels[field][code] = native_series(code, field).reindex(calendar)
    return panels


def rolling_r_squared(log_close: pd.DataFrame, window: int) -> pd.DataFrame:
    x = np.arange(window, dtype=float)
    centered_x = x - x.mean()
    x_sum_squares = float(centered_x @ centered_x)

    def calculate(values: np.ndarray) -> float:
        centered_y = values - values.mean()
        y_sum_squares = float(centered_y @ centered_y)
        if y_sum_squares <= 0.0:
            return 0.0
        covariance = float(centered_x @ centered_y)
        return float(covariance * covariance / (x_sum_squares * y_sum_squares))

    result = pd.DataFrame(index=log_close.index, columns=log_close.columns, dtype=float)
    for code in log_close:
        result[code] = log_close[code].rolling(window, min_periods=window).apply(
            calculate,
            raw=True,
        )
    return result.clip(lower=0.0, upper=1.0)


def build_features(close: pd.DataFrame) -> dict[str, Any]:
    # Each factor is calculated on the asset's own observation sequence, then
    # mapped to the union calendar.  This mirrors backtest.runner exactly when
    # one ETF has a suspended/missing day; rolling directly on the union panel
    # would incorrectly count that NaN as a calendar observation forever after.
    index = close.index
    momentum20 = pd.DataFrame(index=index, columns=CORE, dtype=float)
    efficiency = {
        window: pd.DataFrame(index=index, columns=CORE, dtype=float)
        for window in ER_WINDOWS
    }
    log_efficiency = {
        window: pd.DataFrame(index=index, columns=CORE, dtype=float)
        for window in ER_WINDOWS
    }
    up_ratio = {
        window: pd.DataFrame(index=index, columns=CORE, dtype=float)
        for window in CONSISTENCY_WINDOWS
    }
    r_squared = {
        window: pd.DataFrame(index=index, columns=CORE, dtype=float)
        for window in CONSISTENCY_WINDOWS
    }
    volatility = {
        window: pd.DataFrame(index=index, columns=CORE, dtype=float)
        for window in VOLATILITY_WINDOWS
    }

    for code in CORE:
        native_close = close[code].dropna()
        native_daily_return = native_close.pct_change(fill_method=None)
        native_log_close = np.log(native_close)
        native_log_daily_return = native_log_close.diff()
        momentum20[code] = native_close.pct_change(20, fill_method=None).reindex(index)

        for window in ER_WINDOWS:
            displacement = (native_close - native_close.shift(window)).abs()
            path = native_close.diff().abs().rolling(window, min_periods=window).sum()
            efficiency[window][code] = (
                displacement / path.replace(0.0, np.nan)
            ).reindex(index)

            log_displacement = native_log_close.diff(window).abs()
            log_path = native_log_daily_return.abs().rolling(
                window,
                min_periods=window,
            ).sum()
            log_efficiency[window][code] = (
                log_displacement / log_path.replace(0.0, np.nan)
            ).reindex(index)

        signed_up = native_daily_return.gt(0.0).astype(float).where(
            native_daily_return.notna()
        )
        native_log_frame = native_log_close.to_frame(code)
        for window in CONSISTENCY_WINDOWS:
            up_ratio[window][code] = signed_up.rolling(
                window,
                min_periods=window,
            ).mean().reindex(index)
            r_squared[window][code] = rolling_r_squared(
                native_log_frame,
                window,
            )[code].reindex(index)

        for window in VOLATILITY_WINDOWS:
            volatility[window][code] = native_daily_return.rolling(
                window,
                min_periods=window,
            ).std(ddof=1).reindex(index)
    return {
        "momentum20": momentum20,
        "efficiency": efficiency,
        "log_efficiency": log_efficiency,
        "up_ratio": up_ratio,
        "r_squared": r_squared,
        "volatility": volatility,
    }


def targets_from_score(score: pd.DataFrame) -> np.ndarray:
    # On a union-calendar day where one ETF has no new observation, the
    # official runner truncates that asset at its latest earlier row and uses
    # the last available factor value.  Forward fill reproduces that snapshot.
    values = score.loc[:, CORE].ffill().to_numpy(dtype=float)
    finite = np.isfinite(values)
    valid = finite.any(axis=1)
    safe = np.where(finite, values, -np.inf)
    targets = np.argmax(safe, axis=1).astype(np.int8)
    targets[~valid] = -1
    return targets


def safe_return(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator == 0.0:
        return 0.0
    return float(numerator / denominator - 1.0)


def simulate(targets: np.ndarray, opens: np.ndarray, closes: np.ndarray) -> Simulation:
    rows = len(targets)
    gross = np.zeros(rows, dtype=float)
    turnover = np.zeros(rows, dtype=float)
    held = np.full(rows, -1, dtype=np.int8)
    current = -1
    entry_index = -1
    pending = -1
    pending_index = -1

    for index in range(rows):
        if index > 0:
            old = current
            if pending_index == index and pending >= 0:
                overnight = (
                    0.0
                    if old < 0
                    else safe_return(opens[index, old], closes[index - 1, old])
                )
                current = pending
                entry_index = index
                turnover[index] = 1.0 if old < 0 else (0.0 if old == current else 2.0)
                intraday = safe_return(closes[index, current], opens[index, current])
                gross[index] = (1.0 + overnight) * (1.0 + intraday) - 1.0
                pending = -1
                pending_index = -1
            elif current >= 0:
                gross[index] = safe_return(closes[index, current], closes[index - 1, current])

        held[index] = current
        holding_days = index - entry_index + 1 if current >= 0 else None
        should_signal = pending < 0 and (
            current < 0 or holding_days is None or holding_days >= REBALANCE_DAYS
        )
        if should_signal and index + 1 < rows:
            proposed = int(targets[index])
            if proposed >= 0 and proposed != current:
                if np.isfinite(opens[index + 1, proposed]) and np.isfinite(
                    closes[index + 1, proposed]
                ):
                    pending = proposed
                    pending_index = index + 1

    return Simulation(gross=gross, turnover=turnover, held=held)


def period_mask(
    dates: pd.DatetimeIndex,
    start: pd.Timestamp,
    end: pd.Timestamp | None,
) -> np.ndarray:
    mask = dates >= start
    if end is not None:
        mask &= dates <= end
    return np.asarray(mask, dtype=bool)


def metrics(returns: np.ndarray) -> dict[str, float]:
    values = returns[np.isfinite(returns)]
    if not len(values):
        return {"annual_return": 0.0, "sharpe": 0.0, "max_drawdown": 0.0}
    standard_deviation = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    annual_return = float(np.prod(1.0 + values) ** (252.0 / len(values)) - 1.0)
    sharpe = (
        float(np.mean(values) / standard_deviation * math.sqrt(252.0))
        if standard_deviation > 0.0
        else 0.0
    )
    wealth = np.cumprod(1.0 + values)
    drawdown = wealth / np.maximum.accumulate(wealth) - 1.0
    return {
        "annual_return": annual_return,
        "sharpe": sharpe,
        "max_drawdown": float(np.min(drawdown)),
    }


def score_tie_rate(score: pd.DataFrame, mask: np.ndarray) -> float:
    values = score.loc[:, CORE].ffill().to_numpy(dtype=float)[mask]
    finite = np.isfinite(values)
    valid = finite.any(axis=1)
    if not valid.any():
        return 1.0
    safe = np.where(finite, values, -np.inf)
    maximum = safe.max(axis=1, keepdims=True)
    ties = np.isclose(safe, maximum, rtol=0.0, atol=1e-15).sum(axis=1) > 1
    return float(ties[valid].mean())


def path_hash(targets: np.ndarray, mask: np.ndarray) -> str:
    return hashlib.sha1(targets[mask].tobytes()).hexdigest()[:12]


def cross_sectional_rank(frame: pd.DataFrame, *, low_is_high: bool = False) -> pd.DataFrame:
    return frame.rank(
        axis=1,
        pct=True,
        method="average",
        ascending=not low_is_high,
    )


def relative_volatility(
    volatility: pd.DataFrame,
    quantile: float,
    history: str | int,
) -> pd.DataFrame:
    ratio = pd.DataFrame(index=volatility.index, columns=volatility.columns, dtype=float)
    for code in volatility:
        native_volatility = volatility[code].dropna()
        lagged = native_volatility.shift(1)
        if history == "expanding":
            threshold = lagged.expanding(min_periods=60).quantile(quantile)
        else:
            threshold = lagged.rolling(
                int(history),
                min_periods=60,
            ).quantile(quantile)
        ratio[code] = (
            native_volatility / threshold.replace(0.0, np.nan)
        ).reindex(volatility.index)
    return ratio.replace([np.inf, -np.inf], np.nan)


def low_vol_multiplier(ratio: pd.DataFrame, shape: str) -> pd.DataFrame:
    inverse = 1.0 / ratio.replace(0.0, np.nan)
    if shape == "cap":
        return inverse.clip(lower=0.25, upper=1.0).fillna(1.0)
    if shape == "symmetric":
        return inverse.clip(lower=0.25, upper=2.0).fillna(1.0)
    if shape == "exponential":
        return np.exp(1.0 - ratio).clip(lower=0.25, upper=2.0).fillna(1.0)
    if shape == "linear":
        return (1.5 - 0.5 * ratio).clip(lower=0.25, upper=1.5).fillna(1.0)
    raise ValueError(f"unknown low-vol shape: {shape}")


def params_json(params: dict[str, Any]) -> str:
    return json.dumps(params, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def evaluate(
    name: str,
    family: str,
    params: dict[str, Any],
    score: pd.DataFrame,
    dates: pd.DatetimeIndex,
    opens: np.ndarray,
    closes: np.ndarray,
    masks: dict[str, np.ndarray],
    baseline_metrics: dict[str, dict[str, float]],
) -> tuple[dict[str, Any], np.ndarray, Simulation]:
    targets = targets_from_score(score)
    simulation = simulate(targets, opens, closes)
    net = simulation.gross - simulation.turnover * MAIN_FEE
    row: dict[str, Any] = {
        "name": name,
        "family": family,
        "params": params_json(params),
        "path_hash": path_hash(targets, masks["evaluation"]),
        "tie_rate": score_tie_rate(score, masks["evaluation"]),
    }
    for period in ("development", "validation", "train"):
        measured = metrics(net[masks[period]])
        for metric_name, value in measured.items():
            row[f"{period}_{metric_name}"] = value
            row[f"{period}_{metric_name}_delta"] = (
                value - baseline_metrics[period][metric_name]
            )

    row["selection_score"] = (
        min(
            row["development_sharpe_delta"],
            row["validation_sharpe_delta"],
        )
        + 0.35 * row["train_sharpe_delta"]
        + 0.10
        * min(
            row["development_annual_return_delta"],
            row["validation_annual_return_delta"],
        )
    )
    row["train_feasible"] = bool(
        row["development_sharpe_delta"] >= 0.0
        and row["validation_sharpe_delta"] >= 0.0
        and row["train_sharpe_delta"] >= 0.0
        and row["train_max_drawdown_delta"] >= -0.02
    )
    return row, targets, simulation


def retained_sort_key(candidate: RetainedCandidate) -> tuple[bool, float]:
    return (
        bool(candidate.row["train_feasible"]),
        float(candidate.row["selection_score"]),
    )


def retain_best(
    pool: list[RetainedCandidate],
    candidate: RetainedCandidate,
    limit: int = 240,
) -> None:
    pool.append(candidate)
    if len(pool) > limit:
        pool.sort(key=retained_sort_key, reverse=True)
        del pool[limit:]


def official_anchor(
    dates: pd.DatetimeIndex,
    simulation: Simulation,
    evaluation_mask: np.ndarray,
) -> pd.DataFrame:
    config = _load_config_from_yaml(BASELINE_CONFIG)
    config["end"] = date.today()
    config["transaction_cost_rate"] = MAIN_FEE
    official = run_official(config)
    research = pd.Series(
        simulation.gross - simulation.turnover * MAIN_FEE,
        index=dates,
        dtype=float,
    ).loc[EVALUATION_START:]
    joined = pd.concat(
        [
            official.daily_returns.rename("official"),
            research.rename("research"),
        ],
        axis=1,
        join="inner",
    ).dropna()
    difference = joined["research"] - joined["official"]
    official_metrics = metrics(joined["official"].to_numpy(dtype=float))
    research_metrics = metrics(joined["research"].to_numpy(dtype=float))
    return pd.DataFrame(
        [
            {
                "overlap_start": joined.index.min().date().isoformat(),
                "overlap_end": joined.index.max().date().isoformat(),
                "overlap_days": len(joined),
                "max_abs_daily_return_difference": float(difference.abs().max()),
                "official_annual_return": official_metrics["annual_return"],
                "research_annual_return": research_metrics["annual_return"],
                "official_sharpe": official_metrics["sharpe"],
                "research_sharpe": research_metrics["sharpe"],
                "official_max_drawdown": official_metrics["max_drawdown"],
                "research_max_drawdown": research_metrics["max_drawdown"],
                "evaluation_rows": int(evaluation_mask.sum()),
            }
        ]
    )


def add_revealed_metrics(
    candidate: RetainedCandidate,
    dates: pd.DatetimeIndex,
    masks: dict[str, np.ndarray],
) -> dict[str, Any]:
    row = dict(candidate.row)
    years = float(masks["evaluation"].sum()) / 252.0
    for fee_name, fee in (("1bp", MAIN_FEE), ("5bp", STRESS_FEE)):
        net = candidate.simulation.gross - candidate.simulation.turnover * fee
        for period in ("evaluation", "oos"):
            measured = metrics(net[masks[period]])
            for metric_name, value in measured.items():
                row[f"{period}_{fee_name}_{metric_name}"] = value
        row[f"annual_one_way_turnover_{fee_name}"] = (
            float(candidate.simulation.turnover[masks["evaluation"]].sum())
            / 2.0
            / years
        )
    row["switches"] = int(
        (candidate.simulation.turnover[masks["evaluation"]] >= 2.0).sum()
    )
    held = candidate.simulation.held[masks["evaluation"]]
    for index, code in enumerate(CORE):
        row[f"held_share_{code}"] = float(np.mean(held == index))
    row["evaluation_start"] = dates[masks["evaluation"]].min().date().isoformat()
    row["evaluation_end"] = dates[masks["evaluation"]].max().date().isoformat()
    return row


def rolling_36_months(
    selected: RetainedCandidate,
    baseline: RetainedCandidate,
    dates: pd.DatetimeIndex,
    mask: np.ndarray,
) -> pd.DataFrame:
    selected_net = selected.simulation.gross - selected.simulation.turnover * MAIN_FEE
    baseline_net = baseline.simulation.gross - baseline.simulation.turnover * MAIN_FEE
    selected_net = selected_net[mask]
    baseline_net = baseline_net[mask]
    selected_dates = dates[mask]
    rows = []
    for end in range(756, len(selected_net) + 1, 21):
        section = slice(end - 756, end)
        selected_metrics = metrics(selected_net[section])
        baseline_section = metrics(baseline_net[section])
        rows.append(
            {
                "window_end": selected_dates[end - 1].date().isoformat(),
                "selected_sharpe": selected_metrics["sharpe"],
                "baseline_sharpe": baseline_section["sharpe"],
                "sharpe_delta": selected_metrics["sharpe"]
                - baseline_section["sharpe"],
                "selected_leads": selected_metrics["sharpe"]
                > baseline_section["sharpe"],
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    prices = load_prices()
    close = prices["close"]
    dates = close.index
    opens_array = prices["open"].to_numpy(dtype=float)
    closes_array = close.to_numpy(dtype=float)
    features = build_features(close)
    masks = {
        "evaluation": period_mask(dates, EVALUATION_START, None),
        "development": period_mask(dates, EVALUATION_START, DEVELOPMENT_END),
        "validation": period_mask(dates, VALIDATION_START, TRAIN_END),
        "train": period_mask(dates, EVALUATION_START, TRAIN_END),
        "oos": period_mask(dates, TRAIN_END + pd.Timedelta(days=1), None),
    }

    momentum20 = features["momentum20"]
    efficiency = features["efficiency"]
    baseline_score = momentum20 * efficiency[20]
    baseline_targets = targets_from_score(baseline_score)
    baseline_simulation = simulate(baseline_targets, opens_array, closes_array)
    baseline_net = baseline_simulation.gross - baseline_simulation.turnover * MAIN_FEE
    baseline_metrics = {
        period: metrics(baseline_net[mask])
        for period, mask in masks.items()
        if period in {"development", "validation", "train"}
    }
    baseline_row, _, _ = evaluate(
        "production_mom20_er20_k1",
        "production",
        {"momentum_window": 20, "er_window": 20, "er_power": 1.0},
        baseline_score,
        dates,
        opens_array,
        closes_array,
        masks,
        baseline_metrics,
    )
    baseline_candidate = RetainedCandidate(
        row=baseline_row,
        score=baseline_score,
        targets=baseline_targets,
        simulation=baseline_simulation,
    )

    anchor = official_anchor(dates, baseline_simulation, masks["evaluation"])
    anchor.to_csv(OUTPUT / "anchor.csv", index=False)
    max_anchor_error = float(anchor.at[0, "max_abs_daily_return_difference"])
    if max_anchor_error > 1e-12:
        raise AssertionError(f"research harness anchor failed: {max_anchor_error:.3e}")

    rows: list[dict[str, Any]] = [baseline_row]
    base_scores: dict[str, pd.DataFrame] = {
        "production_mom20_er20_k1": baseline_score,
    }

    def evaluate_base(
        name: str,
        family: str,
        params: dict[str, Any],
        score: pd.DataFrame,
    ) -> None:
        if name in base_scores:
            return
        row, _, _ = evaluate(
            name,
            family,
            params,
            score,
            dates,
            opens_array,
            closes_array,
            masks,
            baseline_metrics,
        )
        rows.append(row)
        base_scores[name] = score

    evaluate_base(
        "mom20_only",
        "momentum_only",
        {"momentum_window": 20},
        momentum20,
    )
    for metric_name, metric_cache in (
        ("price_er", efficiency),
        ("log_er", features["log_efficiency"]),
    ):
        for window in ER_WINDOWS:
            for power in ER_POWERS:
                name = f"mom20_{metric_name}{window}_k{power:g}"
                evaluate_base(
                    name,
                    "efficiency",
                    {
                        "momentum_window": 20,
                        "quality_metric": metric_name,
                        "quality_window": window,
                        "quality_power": power,
                    },
                    momentum20 * metric_cache[window].pow(power),
                )

    for er_window in (10, 15, 20, 30, 40):
        for er_power in (0.75, 1.00, 1.25, 1.50):
            base = momentum20 * efficiency[er_window].pow(er_power)
            for consistency_name, cache in (
                ("up_ratio", features["up_ratio"]),
                ("r_squared", features["r_squared"]),
            ):
                for window in CONSISTENCY_WINDOWS:
                    soft_quality = 0.5 + cache[window]
                    for power in CONSISTENCY_POWERS:
                        name = (
                            f"mom20_er{er_window}_k{er_power:g}_"
                            f"{consistency_name}{window}_p{power:g}"
                        )
                        evaluate_base(
                            name,
                            "path_consistency",
                            {
                                "momentum_window": 20,
                                "er_window": er_window,
                                "er_power": er_power,
                                "consistency_metric": consistency_name,
                                "consistency_window": window,
                                "consistency_power": power,
                                "consistency_offset": 0.5,
                            },
                            base * soft_quality.pow(power),
                        )

    for er_window in (15, 20, 30):
        for er_power in (0.75, 1.00, 1.25, 1.50):
            base = momentum20 * efficiency[er_window].pow(er_power)
            for up_window in (10, 20):
                for up_power in (0.25, 0.50):
                    up = (0.5 + features["up_ratio"][up_window]).pow(up_power)
                    for r2_window in (10, 20):
                        for r2_power in (0.25, 0.50):
                            linearity = (0.5 + features["r_squared"][r2_window]).pow(
                                r2_power
                            )
                            name = (
                                f"mom20_er{er_window}_k{er_power:g}_up{up_window}_"
                                f"p{up_power:g}_r2{r2_window}_p{r2_power:g}"
                            )
                            evaluate_base(
                                name,
                                "combined_path_quality",
                                {
                                    "momentum_window": 20,
                                    "er_window": er_window,
                                    "er_power": er_power,
                                    "up_window": up_window,
                                    "up_power": up_power,
                                    "r2_window": r2_window,
                                    "r2_power": r2_power,
                                    "quality_offset": 0.5,
                                },
                                base * up * linearity,
                            )

    stage_one = pd.DataFrame(rows)
    ordered_stage_one = stage_one.sort_values(
        ["train_feasible", "selection_score"],
        ascending=[False, False],
    )
    backbone_names: list[str] = []
    seen_paths: set[str] = set()
    for record in ordered_stage_one.to_dict("records"):
        if record["path_hash"] in seen_paths:
            continue
        seen_paths.add(record["path_hash"])
        backbone_names.append(str(record["name"]))
        if len(backbone_names) == 12:
            break
    for mandatory in (
        "production_mom20_er20_k1",
        "mom20_price_er20_k1.5",
    ):
        if mandatory in base_scores and mandatory not in backbone_names:
            backbone_names.append(mandatory)

    ratio_cache: dict[tuple[int, float, str | int], pd.DataFrame] = {}
    for window in VOLATILITY_WINDOWS:
        for quantile in VOLATILITY_QUANTILES:
            for history in VOLATILITY_HISTORIES:
                ratio_cache[(window, quantile, history)] = relative_volatility(
                    features["volatility"][window],
                    quantile,
                    history,
                )

    retained: list[RetainedCandidate] = [baseline_candidate]
    stage_one_lookup = stage_one.set_index("name", drop=False)
    for name in backbone_names:
        score = base_scores[name]
        row, targets, simulation = evaluate(
            name,
            str(stage_one_lookup.at[name, "family"]),
            json.loads(str(stage_one_lookup.at[name, "params"])),
            score,
            dates,
            opens_array,
            closes_array,
            masks,
            baseline_metrics,
        )
        retain_best(
            retained,
            RetainedCandidate(row, score, targets, simulation),
        )

    completed = 0
    total_low_vol = (
        len(backbone_names)
        * len(VOLATILITY_WINDOWS)
        * len(VOLATILITY_QUANTILES)
        * len(VOLATILITY_HISTORIES)
        * len(LOW_VOL_SHAPES)
        * len(LOW_VOL_POWERS)
    )
    for base_name in backbone_names:
        base_score = base_scores[base_name]
        for (window, quantile, history), ratio in ratio_cache.items():
            for shape in LOW_VOL_SHAPES:
                multiplier = low_vol_multiplier(ratio, shape)
                for power in LOW_VOL_POWERS:
                    name = (
                        f"{base_name}__lv_v{window}_q{quantile:g}_h{history}_"
                        f"{shape}_p{power:g}"
                    )
                    params = {
                        "base_name": base_name,
                        "momentum_window": 20,
                        "volatility_window": window,
                        "volatility_quantile": quantile,
                        "volatility_history": history,
                        "low_vol_shape": shape,
                        "low_vol_power": power,
                        "quantile_min_history": 60,
                    }
                    score = base_score * multiplier.pow(power)
                    row, targets, simulation = evaluate(
                        name,
                        "soft_low_volatility",
                        params,
                        score,
                        dates,
                        opens_array,
                        closes_array,
                        masks,
                        baseline_metrics,
                    )
                    rows.append(row)
                    retain_best(
                        retained,
                        RetainedCandidate(row, score, targets, simulation),
                    )
                    completed += 1
                    if completed % 500 == 0:
                        print(
                            f"soft-low-vol search {completed}/{total_low_vol}",
                            flush=True,
                        )

    for base_name in backbone_names[:8]:
        base_score = base_scores[base_name]
        base_rank = cross_sectional_rank(base_score)
        for window in VOLATILITY_WINDOWS:
            absolute_low_vol_rank = cross_sectional_rank(
                features["volatility"][window],
                low_is_high=True,
            )
            own_relative_rank = cross_sectional_rank(
                ratio_cache[(window, 0.80, "expanding")],
                low_is_high=True,
            )
            for risk_name, risk_rank in (
                ("absolute", absolute_low_vol_rank),
                ("own_relative", own_relative_rank),
            ):
                for weight in RANK_WEIGHTS:
                    name = f"{base_name}__rank_lv_{risk_name}_v{window}_w{weight:g}"
                    params = {
                        "base_name": base_name,
                        "momentum_window": 20,
                        "combination": "cross_sectional_rank_addition",
                        "low_vol_metric": risk_name,
                        "volatility_window": window,
                        "low_vol_rank_weight": weight,
                    }
                    score = base_rank + weight * risk_rank
                    row, targets, simulation = evaluate(
                        name,
                        "rank_low_volatility",
                        params,
                        score,
                        dates,
                        opens_array,
                        closes_array,
                        masks,
                        baseline_metrics,
                    )
                    rows.append(row)
                    retain_best(
                        retained,
                        RetainedCandidate(row, score, targets, simulation),
                    )

    momentum_rank = cross_sectional_rank(momentum20)
    for er_window in ER_WINDOWS:
        er_rank = cross_sectional_rank(efficiency[er_window])
        for er_weight in (0.25, 0.50, 1.00, 1.50):
            for vol_window in VOLATILITY_WINDOWS:
                low_vol_rank = cross_sectional_rank(
                    features["volatility"][vol_window],
                    low_is_high=True,
                )
                for vol_weight in RANK_WEIGHTS:
                    name = (
                        f"component_rank_mom20_er{er_window}_w{er_weight:g}_"
                        f"vol{vol_window}_w{vol_weight:g}"
                    )
                    params = {
                        "momentum_window": 20,
                        "combination": "component_cross_sectional_ranks",
                        "er_window": er_window,
                        "er_rank_weight": er_weight,
                        "volatility_window": vol_window,
                        "low_vol_rank_weight": vol_weight,
                    }
                    score = momentum_rank + er_weight * er_rank + vol_weight * low_vol_rank
                    row, targets, simulation = evaluate(
                        name,
                        "component_rank",
                        params,
                        score,
                        dates,
                        opens_array,
                        closes_array,
                        masks,
                        baseline_metrics,
                    )
                    rows.append(row)
                    retain_best(
                        retained,
                        RetainedCandidate(row, score, targets, simulation),
                    )

    broad = pd.DataFrame(rows).drop_duplicates("name", keep="first")
    broad = broad.sort_values(
        ["train_feasible", "selection_score"],
        ascending=[False, False],
    ).reset_index(drop=True)
    broad.insert(0, "train_rank", np.arange(1, len(broad) + 1))
    broad.to_csv(OUTPUT / "broad_search_train_only.csv", index=False)

    # A point maximum can be a narrow in-sample accident.  Independently keep
    # a plateau representative: choose the low-vol parameter block with the
    # highest median selection score, require at least 80% feasible neighbors
    # and a non-negative lower quartile, then take the grid-center q=0.8/p=0.5
    # rather than optimizing within that block.
    soft_rows = broad.loc[broad["family"].eq("soft_low_volatility")].copy()
    expanded_params = soft_rows["params"].map(json.loads).apply(pd.Series)
    soft_rows = pd.concat(
        [soft_rows.reset_index(drop=True), expanded_params.reset_index(drop=True)],
        axis=1,
    )
    plateau_columns = [
        "base_name",
        "volatility_window",
        "volatility_history",
        "low_vol_shape",
    ]
    plateau_groups = (
        soft_rows.groupby(plateau_columns, dropna=False)
        .agg(
            neighbors=("name", "size"),
            feasible_rate=("train_feasible", "mean"),
            median_selection_score=("selection_score", "median"),
            p25_selection_score=("selection_score", lambda values: values.quantile(0.25)),
            worst_selection_score=("selection_score", "min"),
            best_selection_score=("selection_score", "max"),
            median_development_sharpe_delta=("development_sharpe_delta", "median"),
            median_validation_sharpe_delta=("validation_sharpe_delta", "median"),
            median_train_sharpe_delta=("train_sharpe_delta", "median"),
        )
        .reset_index()
        .sort_values(
            ["median_selection_score", "p25_selection_score"],
            ascending=False,
        )
    )
    plateau_groups.to_csv(OUTPUT / "plateau_groups_train_only.csv", index=False)
    eligible_plateaus = plateau_groups.loc[
        plateau_groups["neighbors"].eq(9)
        & plateau_groups["feasible_rate"].ge(0.80)
        & plateau_groups["p25_selection_score"].ge(0.0)
    ]
    if eligible_plateaus.empty:
        raise AssertionError("no low-vol parameter plateau passed the train-only gate")
    plateau = eligible_plateaus.iloc[0]
    plateau_match = (
        soft_rows["base_name"].eq(plateau["base_name"])
        & soft_rows["volatility_window"].eq(plateau["volatility_window"])
        & soft_rows["volatility_history"].eq(plateau["volatility_history"])
        & soft_rows["low_vol_shape"].eq(plateau["low_vol_shape"])
        & soft_rows["volatility_quantile"].eq(0.80)
        & soft_rows["low_vol_power"].eq(0.50)
    )
    plateau_name = str(soft_rows.loc[plateau_match, "name"].iloc[0])

    retained.sort(key=retained_sort_key, reverse=True)
    retained_by_name = {candidate.row["name"]: candidate for candidate in retained}
    if plateau_name not in retained_by_name:
        raise AssertionError(f"plateau representative was not retained: {plateau_name}")
    plateau_candidate = retained_by_name[plateau_name]
    distinct: list[RetainedCandidate] = []
    seen_paths.clear()
    for candidate in retained:
        candidate_path = str(candidate.row["path_hash"])
        if candidate_path in seen_paths:
            continue
        seen_paths.add(candidate_path)
        distinct.append(candidate)
        if len(distinct) == 30:
            break
    if baseline_candidate.row["path_hash"] not in seen_paths:
        distinct.append(baseline_candidate)
        seen_paths.add(str(baseline_candidate.row["path_hash"]))
    if plateau_candidate.row["path_hash"] not in seen_paths:
        distinct.append(plateau_candidate)
        seen_paths.add(str(plateau_candidate.row["path_hash"]))

    non_baseline = [
        candidate
        for candidate in distinct
        if candidate.row["name"] != baseline_candidate.row["name"]
    ]
    feasible_non_baseline = [
        candidate for candidate in non_baseline if candidate.row["train_feasible"]
    ]
    selected = (feasible_non_baseline or non_baseline)[0]
    shortlist = pd.DataFrame(
        [add_revealed_metrics(candidate, dates, masks) for candidate in distinct]
    ).sort_values(["train_feasible", "selection_score"], ascending=[False, False])
    shortlist.insert(
        0,
        "selected_as_plateau_representative",
        shortlist["name"].eq(plateau_candidate.row["name"]),
    )
    shortlist.insert(
        0,
        "selected_as_point_maximum",
        shortlist["name"].eq(selected.row["name"]),
    )
    shortlist.to_csv(OUTPUT / "shortlist_revealed.csv", index=False)

    rolling = rolling_36_months(selected, baseline_candidate, dates, masks["evaluation"])
    rolling.to_csv(OUTPUT / "point_maximum_rolling_36m.csv", index=False)
    plateau_rolling = rolling_36_months(
        plateau_candidate,
        baseline_candidate,
        dates,
        masks["evaluation"],
    )
    plateau_rolling.to_csv(OUTPUT / "plateau_representative_rolling_36m.csv", index=False)
    selected_payload = {
        "point_maximum": {
            "name": selected.row["name"],
            "family": selected.row["family"],
            "params": json.loads(selected.row["params"]),
            "path_hash": selected.row["path_hash"],
            "passed_train_feasibility_gate": bool(selected.row["train_feasible"]),
            "rolling_36m_windows": int(len(rolling)),
            "rolling_36m_win_rate": float(rolling["selected_leads"].mean())
            if len(rolling)
            else None,
        },
        "plateau_representative": {
            "name": plateau_candidate.row["name"],
            "family": plateau_candidate.row["family"],
            "params": json.loads(plateau_candidate.row["params"]),
            "path_hash": plateau_candidate.row["path_hash"],
            "passed_train_feasibility_gate": bool(
                plateau_candidate.row["train_feasible"]
            ),
            "plateau_median_selection_score": float(
                plateau["median_selection_score"]
            ),
            "plateau_p25_selection_score": float(plateau["p25_selection_score"]),
            "plateau_feasible_rate": float(plateau["feasible_rate"]),
            "rolling_36m_windows": int(len(plateau_rolling)),
            "rolling_36m_win_rate": float(
                plateau_rolling["selected_leads"].mean()
            )
            if len(plateau_rolling)
            else None,
        },
        "selection_uses_data_through": TRAIN_END.date().isoformat(),
        "oos_starts": (TRAIN_END + pd.Timedelta(days=1)).date().isoformat(),
    }
    (OUTPUT / "selected.json").write_text(
        json.dumps(selected_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(selected_payload, ensure_ascii=False, indent=2), flush=True)
    print(f"wrote {len(broad)} train-only variants to {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
