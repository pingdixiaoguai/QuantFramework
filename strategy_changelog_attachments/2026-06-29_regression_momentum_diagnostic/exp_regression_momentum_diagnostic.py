"""Read-only regression-slope momentum diagnostic.

Outputs research artifacts under this attachment directory only. The script
injects exp_* factors in memory and does not edit production registry/configs.
"""

from __future__ import annotations

import contextlib
import math
import sys
import warnings
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import backtest.runner as runner
from backtest.runner import BacktestResult
from data.store import query
from factors.quality_momentum import compute as compute_quality_momentum
from factors.registry import load_registered_factors

OUTPUT_DIR = Path(__file__).resolve().parent
PREFIX = "2026-06-29_regression_momentum_diagnostic"

ASSET_POOL = ["510300.SH", "159915.SZ", "513100.SH", "518880.SH"]
WARMUP_START = date(2013, 1, 1)
EVAL_START = pd.Timestamp("2014-01-01")
END = date(2026, 6, 4)
TRAIN_RATIO = 0.7
REBALANCE_DAYS = 5
COST_RATES = [0.00005, 0.0001, 0.0003, 0.0005]
ROUND_TRIP_DAYS = 15
ANCHOR_TOLERANCE = 1e-12
FACTOR_NAME = "exp_regression_momentum_score"


@dataclass(frozen=True)
class ArmSpec:
    arm: str
    description: str
    strength: str
    cleanliness: str
    weight_mode: str
    points: int
    min_history: int


ARM_SPECS = [
    ArmSpec(
        "B",
        "production baseline: pct_change(20) x ER(20)",
        "pct_change",
        "er",
        "none",
        21,
        21,
    ),
    ArmSpec(
        "B_prime",
        "window-matched baseline: pct_change(25) x ER(25)",
        "pct_change",
        "er",
        "equal",
        26,
        26,
    ),
    ArmSpec(
        "A3",
        "equal-weight OLS slope x ER(25)",
        "slope",
        "er",
        "equal",
        26,
        27,
    ),
    ArmSpec(
        "A2",
        "equal-weight OLS slope x R2(26)",
        "slope",
        "r2",
        "equal",
        26,
        27,
    ),
    ArmSpec(
        "A1",
        "recency-weighted WLS slope x weighted R2(26)",
        "slope",
        "r2",
        "recency_squared",
        26,
        27,
    ),
]


def _base_config(cost_rate: float, arm: ArmSpec) -> dict:
    return {
        "strategy_name": f"exp_regression_momentum_{arm.arm.lower()}",
        "strategy_class": "strategy.top1.Top1",
        "asset_pool": list(ASSET_POOL),
        "start": WARMUP_START,
        "end": END,
        "factors": [
            {
                "name": FACTOR_NAME,
                "weight": 1.0,
                "params": {
                    "arm": arm.arm,
                    "strength": arm.strength,
                    "cleanliness": arm.cleanliness,
                    "weight_mode": arm.weight_mode,
                    "points": arm.points,
                },
            }
        ],
        "train_ratio": TRAIN_RATIO,
        "rebalance_days": REBALANCE_DAYS,
        "transaction_cost_rate": cost_rate,
    }


def _rolling_weighted_sum(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    out = np.full(len(values), np.nan, dtype=float)
    if len(values) >= len(weights):
        out[len(weights) - 1 :] = np.convolve(values, weights[::-1], mode="valid")
    return out


def _efficiency_ratio(close: pd.Series, window: int) -> pd.Series:
    displacement = (close - close.shift(window)).abs()
    path_length = close.diff().abs().rolling(window).sum()
    return displacement / path_length.replace(0.0, np.nan)


def _slope_score(
    close: pd.Series,
    *,
    points: int,
    cleanliness: str,
    weight_mode: str,
) -> pd.Series:
    log_close = np.log(close.astype(float))
    y = log_close.to_numpy(dtype=float)
    x = np.arange(points, dtype=float)
    if weight_mode == "recency_squared":
        weights = np.linspace(1.0, 2.0, points) ** 2
    elif weight_mode == "equal":
        weights = np.ones(points, dtype=float)
    else:
        raise ValueError(f"unknown slope weight_mode: {weight_mode}")

    sum_w = float(weights.sum())
    sum_wx = float(np.dot(weights, x))
    sum_wx2 = float(np.dot(weights, x * x))
    x_bar = sum_wx / sum_w
    sxx = float(np.dot(weights, (x - x_bar) ** 2))

    sum_wy = _rolling_weighted_sum(y, weights)
    sum_wy2 = _rolling_weighted_sum(y * y, weights)
    sum_wxy = _rolling_weighted_sum(y, weights * x)

    slope = (sum_wxy - x_bar * sum_wy) / sxx
    annualized = np.exp(slope * 250.0) - 1.0

    if cleanliness == "er":
        clean = _efficiency_ratio(close, points - 1).to_numpy(dtype=float)
    elif cleanliness == "r2":
        ss_tot = sum_wy2 - (sum_wy * sum_wy) / sum_w
        ss_res = np.maximum(ss_tot - slope * slope * sxx, 0.0)
        clean = np.where(ss_tot > 0.0, 1.0 - ss_res / ss_tot, np.nan)
    else:
        raise ValueError(f"unknown cleanliness: {cleanliness}")

    return pd.Series(annualized * clean, index=close.index, dtype=float)


def _arm_score(df: pd.DataFrame, arm: ArmSpec) -> pd.Series:
    close = df["close"].astype(float)
    if arm.strength == "pct_change":
        window = arm.points - 1
        score = close.pct_change(window) * _efficiency_ratio(close, window)
    elif arm.strength == "slope":
        score = _slope_score(
            close,
            points=arm.points,
            cleanliness=arm.cleanliness,
            weight_mode=arm.weight_mode,
        )
    else:
        raise ValueError(f"unknown strength: {arm.strength}")
    score = score.astype(float)
    score.index = pd.DatetimeIndex(df["date"])
    return score


def _make_factor(arm: ArmSpec) -> dict[str, object]:
    def compute(df: pd.DataFrame, params: dict | None = None) -> pd.Series:
        p = params or {}
        selected = next(
            spec for spec in ARM_SPECS if spec.arm == str(p.get("arm", arm.arm))
        )
        return _arm_score(df, selected)

    return {
        "METADATA": {
            "name": FACTOR_NAME,
            "author": "quantframework",
            "version": "0.0.0-research",
            "params": {
                "arm": arm.arm,
                "strength": arm.strength,
                "cleanliness": arm.cleanliness,
                "weight_mode": arm.weight_mode,
                "points": arm.points,
            },
            "min_history": arm.min_history,
            "direction": "higher_better",
            "description": "Research-only regression momentum diagnostic factor",
        },
        "compute": compute,
    }


@contextlib.contextmanager
def _patched_factor_registry(arm: ArmSpec) -> Iterator[None]:
    original = runner.load_registered_factors

    def patched() -> dict[str, dict[str, object]]:
        factors = load_registered_factors()
        factors[FACTOR_NAME] = _make_factor(arm)
        return factors

    runner.load_registered_factors = patched
    try:
        yield
    finally:
        runner.load_registered_factors = original


def _run_one(arm: ArmSpec, cost_rate: float) -> BacktestResult:
    with _patched_factor_registry(arm):
        return runner.run(_base_config(cost_rate, arm))


def _eval_calendar() -> pd.DatetimeIndex:
    dates: set[pd.Timestamp] = set()
    for asset in ASSET_POOL:
        df = query(asset, WARMUP_START, END)
        dates.update(pd.Timestamp(dt) for dt in df["date"])
    return pd.DatetimeIndex(
        [dt for dt in sorted(dates) if EVAL_START <= dt <= pd.Timestamp(END)]
    )


def _train_end(index: pd.DatetimeIndex) -> pd.Timestamp:
    split_idx = int(len(index) * TRAIN_RATIO)
    if split_idx >= len(index):
        split_idx = len(index) - 1
    return pd.Timestamp(index[split_idx])


def _common_eval_index(
    full_index: pd.DatetimeIndex,
    results: dict[tuple[str, float], BacktestResult],
) -> pd.DatetimeIndex:
    common = set(full_index)
    for result in results.values():
        available = set(
            result.daily_returns.loc[
                (result.daily_returns.index >= full_index[0])
                & (result.daily_returns.index <= full_index[-1])
            ].index
        )
        common &= available
    return pd.DatetimeIndex([dt for dt in full_index if dt in common])


def _calendar_audit(
    full_index: pd.DatetimeIndex,
    common_index: pd.DatetimeIndex,
    results: dict[tuple[str, float], BacktestResult],
) -> pd.DataFrame:
    common_set = set(common_index)
    rows = [
        {
            "scope": "all_arms_common_index",
            "arm": "",
            "transaction_cost_rate": "",
            "full_calendar_days": len(full_index),
            "common_eval_days": len(common_index),
            "missing_days": len(full_index) - len(common_index),
            "missing_dates": ";".join(
                dt.date().isoformat() for dt in full_index if dt not in common_set
            ),
        }
    ]
    for (arm, cost_rate), result in sorted(results.items()):
        available = set(result.daily_returns.index)
        missing = [dt for dt in full_index if dt not in available]
        rows.append(
            {
                "scope": "arm_cost_vs_full_calendar",
                "arm": arm,
                "transaction_cost_rate": cost_rate,
                "full_calendar_days": len(full_index),
                "common_eval_days": "",
                "missing_days": len(missing),
                "missing_dates": ";".join(dt.date().isoformat() for dt in missing),
            }
        )
    return pd.DataFrame(rows)


def _sharpe(returns: pd.Series) -> float:
    if returns.empty:
        return 0.0
    std = returns.std()
    return float(returns.mean() / std * math.sqrt(252.0)) if std > 0 else 0.0


def _annual_return(returns: pd.Series) -> float:
    if returns.empty:
        return 0.0
    total = float((1.0 + returns).prod() - 1.0)
    return float((1.0 + total) ** (252.0 / len(returns)) - 1.0)


def _max_drawdown(returns: pd.Series) -> float:
    if returns.empty:
        return 0.0
    cumulative = (1.0 + returns).cumprod()
    return float((cumulative / cumulative.cummax() - 1.0).min())


def _event_positions_for_eval(
    positions: pd.DataFrame,
    eval_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    if positions.empty:
        return pd.DataFrame(index=eval_index)

    start = eval_index[0]
    before_or_at = positions[positions.index <= start]
    after = positions[(positions.index > start) & (positions.index <= eval_index[-1])]
    pieces: list[pd.DataFrame] = []
    if not before_or_at.empty:
        first = before_or_at.tail(1).copy()
        first.index = pd.DatetimeIndex([start])
        pieces.append(first)
    pieces.append(after)
    out = pd.concat(pieces).sort_index() if pieces else pd.DataFrame()
    out = out[~out.index.duplicated(keep="first")]
    return out.reindex(columns=ASSET_POOL).fillna(0.0)


def _daily_positions(
    positions: pd.DataFrame,
    eval_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    events = _event_positions_for_eval(positions, eval_index)
    daily = events.reindex(eval_index, method="ffill").fillna(0.0)
    return daily.reindex(columns=ASSET_POOL).fillna(0.0)


def _selected_asset(row: pd.Series) -> str:
    if row.empty or float(row.max()) <= 0.0:
        return ""
    return str(row.idxmax())


def _episodes(
    arm: str,
    positions: pd.DataFrame,
    eval_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    daily = _daily_positions(positions, eval_index)
    selected = daily.apply(_selected_asset, axis=1)
    rows: list[dict[str, object]] = []
    if selected.empty:
        return pd.DataFrame(rows)

    start_loc = 0
    current = selected.iloc[0]
    episode_id = 1
    for loc in range(1, len(selected)):
        if selected.iloc[loc] != current:
            rows.append(
                {
                    "arm": arm,
                    "episode_id": episode_id,
                    "asset": current,
                    "start_date": selected.index[start_loc].date().isoformat(),
                    "end_date": selected.index[loc - 1].date().isoformat(),
                    "start_pos": start_loc,
                    "end_pos": loc - 1,
                    "holding_days": loc - start_loc,
                }
            )
            episode_id += 1
            start_loc = loc
            current = selected.iloc[loc]
    rows.append(
        {
            "arm": arm,
            "episode_id": episode_id,
            "asset": current,
            "start_date": selected.index[start_loc].date().isoformat(),
            "end_date": selected.index[-1].date().isoformat(),
            "start_pos": start_loc,
            "end_pos": len(selected) - 1,
            "holding_days": len(selected) - start_loc,
        }
    )
    return pd.DataFrame(rows)


def _round_trips(episodes: pd.DataFrame) -> int:
    count = 0
    for idx in range(len(episodes) - 2):
        first = episodes.iloc[idx]
        middle = episodes.iloc[idx + 1]
        third = episodes.iloc[idx + 2]
        if not first["asset"] or not middle["asset"]:
            continue
        if first["asset"] == third["asset"] and first["asset"] != middle["asset"]:
            gap = int(third["start_pos"]) - int(middle["start_pos"])
            if gap <= ROUND_TRIP_DAYS:
                count += 1
    return count


def _full_metrics(
    arm: ArmSpec,
    cost_rate: float,
    result: BacktestResult,
    eval_index: pd.DatetimeIndex,
    train_end: pd.Timestamp,
    episodes: pd.DataFrame,
) -> dict[str, object]:
    returns = result.daily_returns.reindex(eval_index)
    if returns.isna().any():
        missing = [dt.date().isoformat() for dt in returns[returns.isna()].index[:10]]
        raise RuntimeError(
            f"{arm.arm} cost {cost_rate} has missing eval returns, examples: {missing}"
        )
    turnover = result.turnover.reindex(eval_index, fill_value=0.0)
    years = len(eval_index) / 252.0
    train = returns[returns.index <= train_end]
    oos = returns[returns.index > train_end]
    return {
        "arm": arm.arm,
        "description": arm.description,
        "transaction_cost_rate": cost_rate,
        "transaction_cost_bps_one_side": cost_rate * 10000.0,
        "eval_start": eval_index[0].date().isoformat(),
        "eval_end": eval_index[-1].date().isoformat(),
        "eval_days": len(eval_index),
        "train_end": train_end.date().isoformat(),
        "annual_return": _annual_return(returns),
        "sharpe": _sharpe(returns),
        "max_drawdown": _max_drawdown(returns),
        "annual_turnover_single_side": float(0.5 * turnover.sum() / years),
        "avg_holding_days": float(episodes["holding_days"].mean())
        if len(episodes)
        else 0.0,
        "is_annual_return": _annual_return(train),
        "is_sharpe": _sharpe(train),
        "oos_annual_return": _annual_return(oos),
        "oos_sharpe": _sharpe(oos),
    }


def _split_rows(
    arm: ArmSpec,
    cost_rate: float,
    result: BacktestResult,
    eval_index: pd.DatetimeIndex,
    train_end: pd.Timestamp,
) -> list[dict[str, object]]:
    returns = result.daily_returns.reindex(eval_index)
    rows = []
    for split_name, split_returns in [
        ("IS", returns[returns.index <= train_end]),
        ("OOS", returns[returns.index > train_end]),
    ]:
        rows.append(
            {
                "arm": arm.arm,
                "transaction_cost_rate": cost_rate,
                "transaction_cost_bps_one_side": cost_rate * 10000.0,
                "split": split_name,
                "start": split_returns.index.min().date().isoformat(),
                "end": split_returns.index.max().date().isoformat(),
                "days": len(split_returns),
                "annual_return": _annual_return(split_returns),
                "sharpe": _sharpe(split_returns),
            }
        )
    return rows


def _whipsaw_row(arm: ArmSpec, episodes: pd.DataFrame) -> dict[str, object]:
    switches = max(len(episodes) - 1, 0)
    first_possible = (
        float((episodes["holding_days"] == REBALANCE_DAYS).mean())
        if len(episodes)
        else 0.0
    )
    return {
        "arm": arm.arm,
        "description": arm.description,
        "switch_count": switches,
        "round_trip_count_15d": _round_trips(episodes),
        "first_possible_switch_episode_share": first_possible,
        "avg_holding_days": float(episodes["holding_days"].mean())
        if len(episodes)
        else 0.0,
        "episode_count": len(episodes),
    }


def _anchor_gate() -> pd.DataFrame:
    rows = []
    arm_b = ARM_SPECS[0]
    for asset in ASSET_POOL:
        df = query(asset, WARMUP_START, END)
        production = compute_quality_momentum(df, {"window": 20})
        candidate = _arm_score(df, arm_b)
        both_valid = production.notna() & candidate.notna()
        diff = (production[both_valid] - candidate[both_valid]).abs()
        rows.append(
            {
                "asset": asset,
                "comparable_points": int(len(diff)),
                "max_abs_diff": float(diff.max()) if len(diff) else float("nan"),
                "passed": bool(len(diff) and diff.max() <= ANCHOR_TOLERANCE),
            }
        )
    return pd.DataFrame(rows)


def _warmup_gate(eval_index: pd.DatetimeIndex) -> pd.DataFrame:
    rows = []
    for asset in ASSET_POOL:
        df = query(asset, WARMUP_START, END)
        rows.append(
            {
                "asset": asset,
                "data_start": df["date"].min().date().isoformat(),
                "data_end": df["date"].max().date().isoformat(),
                "rows_before_2014_01_01": int((df["date"] < EVAL_START).sum()),
                "first_eval_trading_day": eval_index[0].date().isoformat(),
                "w26_min_history_27_eligible_date": df["date"]
                .iloc[26]
                .date()
                .isoformat(),
            }
        )
    return pd.DataFrame(rows)


def _fmt_pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def _fmt_float(value: float) -> str:
    return f"{value:.2f}"


def _markdown_table(df: pd.DataFrame, pct_cols: list[str], float_cols: list[str]) -> str:
    display = df.copy()
    for col in pct_cols:
        display[col] = display[col].map(_fmt_pct)
    for col in float_cols:
        display[col] = display[col].map(_fmt_float)
    return display.to_markdown(index=False)


def _fact_statement(metrics: pd.DataFrame, whipsaw: pd.DataFrame) -> str:
    bp1 = metrics[metrics["transaction_cost_bps_one_side"] == 1.0].set_index("arm")
    whip = whipsaw.set_index("arm")
    b_prime = whip.loc["B_prime"]
    a3 = whip.loc["A3"]
    q1_parts = []
    for col, label in [
        ("annual_turnover_single_side", "single-side annual turnover"),
        ("round_trip_count_15d", "15d round trips"),
        ("avg_holding_days", "avg holding days"),
        ("switch_count", "switches"),
    ]:
        left = (
            bp1.loc["A3", col]
            if col == "annual_turnover_single_side"
            else a3[col]
        )
        right = (
            bp1.loc["B_prime", col]
            if col == "annual_turnover_single_side"
            else b_prime[col]
        )
        direction = "lower" if left < right else "higher" if left > right else "equal"
        q1_parts.append(f"{label}: A3 {direction} than B_prime ({left:.4g} vs {right:.4g})")

    reg_arms = ["A3", "A2", "A1"]
    q2_parts = []
    for cost_bps in [0.5, 1.0, 3.0, 5.0]:
        sub = metrics[
            np.isclose(metrics["transaction_cost_bps_one_side"], cost_bps)
        ].set_index("arm")
        winners = [
            arm
            for arm in reg_arms
            if sub.loc[arm, "annual_return"] > sub.loc["B", "annual_return"]
            and sub.loc[arm, "sharpe"] > sub.loc["B", "sharpe"]
        ]
        q2_parts.append(
            f"{cost_bps:g}bp annual-return+Sharpe winners vs B: "
            + (", ".join(winners) if winners else "none")
        )
    return "\n".join(
        [
            "Q1 mechanism facts (A3 vs B_prime, no deployment conclusion):",
            "- " + "\n- ".join(q1_parts),
            "",
            "Q2 realization facts (regression arms vs B, no deployment conclusion):",
            "- " + "\n- ".join(q2_parts),
        ]
    )


def _write_report(
    metrics: pd.DataFrame,
    split_metrics: pd.DataFrame,
    whipsaw: pd.DataFrame,
    anchor: pd.DataFrame,
    warmup: pd.DataFrame,
    calendar_audit: pd.DataFrame,
    full_index: pd.DatetimeIndex,
    eval_index: pd.DatetimeIndex,
    train_end: pd.Timestamp,
) -> None:
    metric_cols = [
        "arm",
        "transaction_cost_bps_one_side",
        "annual_return",
        "sharpe",
        "max_drawdown",
        "annual_turnover_single_side",
        "avg_holding_days",
        "is_annual_return",
        "is_sharpe",
        "oos_annual_return",
        "oos_sharpe",
    ]
    split_cols = [
        "arm",
        "transaction_cost_bps_one_side",
        "split",
        "start",
        "end",
        "days",
        "annual_return",
        "sharpe",
    ]
    whipsaw_cols = [
        "arm",
        "switch_count",
        "round_trip_count_15d",
        "first_possible_switch_episode_share",
        "avg_holding_days",
        "episode_count",
    ]
    lines = [
        "# Regression-Slope Momentum Diagnostic",
        "",
        "Mode C read-only diagnostic. Artifacts are written only in this attachment directory; production configs, registry, run_daily, changelog, summary, and state files are not modified.",
        "",
        "## Controls",
        "",
        f"- Asset pool: {', '.join(ASSET_POOL)}",
        f"- Strategy: Top1 full allocation; rebalance_days={REBALANCE_DAYS}",
        "- Execution/cost semantics: existing backtest engine T+1 open execution, HFQ data, future-info truncation, transaction_cost_rate x sum(abs(delta weights)).",
        f"- Full asset-union calendar in requested window: {full_index[0].date().isoformat()} to {full_index[-1].date().isoformat()}, {len(full_index)} rows.",
        f"- Paired evaluation index used for all arms: {eval_index[0].date().isoformat()} to {eval_index[-1].date().isoformat()}, {len(eval_index)} rows.",
        f"- IS/OOS split: train_ratio={TRAIN_RATIO}, train_end={train_end.date().isoformat()}.",
        "- Cost grid: one-side 0.5/1/3/5 bp.",
        "",
        "## Calendar Audit",
        "",
        calendar_audit.to_markdown(index=False),
        "",
        "## Anchor And Warmup Gates",
        "",
        anchor.to_markdown(index=False),
        "",
        warmup.to_markdown(index=False),
        "",
        "## Full Metrics",
        "",
        _markdown_table(
            metrics[metric_cols],
            [
                "annual_return",
                "max_drawdown",
                "annual_turnover_single_side",
                "is_annual_return",
                "oos_annual_return",
            ],
            ["sharpe", "avg_holding_days", "is_sharpe", "oos_sharpe"],
        ),
        "",
        "## IS/OOS Metrics",
        "",
        _markdown_table(
            split_metrics[split_cols],
            ["annual_return"],
            ["sharpe"],
        ),
        "",
        "## Whipsaw Panel",
        "",
        _markdown_table(
            whipsaw[whipsaw_cols],
            ["first_possible_switch_episode_share"],
            ["avg_holding_days"],
        ),
        "",
        "## Pre-Registered Questions: Facts Only",
        "",
        _fact_statement(metrics, whipsaw),
        "",
        "Notes: round-trip count is consecutive A->B->A episode triples where the second switch occurs within 15 trading days of the first switch. Episode lengths are computed from forward-filled daily holdings, not sparse execution-only position rows.",
        "",
    ]
    (OUTPUT_DIR / f"{PREFIX}_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    full_index = _eval_calendar()

    anchor = _anchor_gate()
    anchor.to_csv(OUTPUT_DIR / f"{PREFIX}_anchor_gate.csv", index=False, encoding="utf-8-sig")
    if not bool(anchor["passed"].all()):
        raise RuntimeError("B anchor gate failed; see anchor_gate CSV")

    warmup = _warmup_gate(full_index)
    warmup.to_csv(OUTPUT_DIR / f"{PREFIX}_warmup_gate.csv", index=False, encoding="utf-8-sig")

    results: dict[tuple[str, float], BacktestResult] = {}
    for arm in ARM_SPECS:
        for cost_rate in COST_RATES:
            print(f"running {arm.arm} @ {cost_rate * 10000:g}bp", flush=True)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                results[(arm.arm, cost_rate)] = _run_one(arm, cost_rate)

    eval_index = _common_eval_index(full_index, results)
    if eval_index.empty:
        raise RuntimeError("no common eval index across all arms")
    train_end = _train_end(eval_index)
    calendar_audit = _calendar_audit(full_index, eval_index, results)
    calendar_audit.to_csv(
        OUTPUT_DIR / f"{PREFIX}_calendar_audit.csv", index=False, encoding="utf-8-sig"
    )

    daily_return_rows: list[pd.DataFrame] = []
    position_rows: list[pd.DataFrame] = []
    event_position_rows: list[pd.DataFrame] = []
    turnover_rows: list[pd.DataFrame] = []
    episode_rows: list[pd.DataFrame] = []
    metric_rows: list[dict[str, object]] = []
    split_metric_rows: list[dict[str, object]] = []
    whipsaw_rows: list[dict[str, object]] = []

    for arm in ARM_SPECS:
        first_result = results[(arm.arm, COST_RATES[0])]
        arm_episodes = _episodes(arm.arm, first_result.positions, eval_index)
        episode_rows.append(arm_episodes)
        daily_pos = _daily_positions(first_result.positions, eval_index).copy()
        daily_pos.insert(0, "date", eval_index.date.astype(str))
        daily_pos.insert(1, "arm", arm.arm)
        position_rows.append(daily_pos)
        event_pos = first_result.positions.loc[
            (first_result.positions.index >= eval_index[0])
            & (first_result.positions.index <= eval_index[-1])
        ].reindex(columns=ASSET_POOL).fillna(0.0)
        event_pos = event_pos.reset_index().rename(columns={"index": "date"})
        event_pos["date"] = pd.to_datetime(event_pos["date"]).dt.date.astype(str)
        event_pos.insert(1, "arm", arm.arm)
        event_position_rows.append(event_pos)
        whipsaw_rows.append(_whipsaw_row(arm, arm_episodes))

        for cost_rate in COST_RATES:
            result = results[(arm.arm, cost_rate)]
            metric_rows.append(
                _full_metrics(arm, cost_rate, result, eval_index, train_end, arm_episodes)
            )
            split_metric_rows.extend(
                _split_rows(arm, cost_rate, result, eval_index, train_end)
            )

            returns = pd.DataFrame(
                {
                    "date": eval_index.date.astype(str),
                    "arm": arm.arm,
                    "transaction_cost_rate": cost_rate,
                    "transaction_cost_bps_one_side": cost_rate * 10000.0,
                    "gross_return": result.gross_daily_returns.reindex(eval_index).to_numpy(),
                    "net_return": result.daily_returns.reindex(eval_index).to_numpy(),
                    "benchmark_return": result.benchmark_returns.reindex(eval_index).to_numpy(),
                }
            )
            if returns[["gross_return", "net_return", "benchmark_return"]].isna().any().any():
                raise RuntimeError(f"{arm.arm} @ {cost_rate} has missing raw returns")
            daily_return_rows.append(returns)

            turnover = pd.DataFrame(
                {
                    "date": eval_index.date.astype(str),
                    "arm": arm.arm,
                    "transaction_cost_rate": cost_rate,
                    "transaction_cost_bps_one_side": cost_rate * 10000.0,
                    "turnover_two_side_sum_abs_delta": result.turnover.reindex(
                        eval_index, fill_value=0.0
                    ).to_numpy(),
                    "cost": result.costs.reindex(eval_index, fill_value=0.0).to_numpy(),
                }
            )
            turnover_rows.append(turnover)

    metrics = pd.DataFrame(metric_rows).sort_values(
        ["transaction_cost_bps_one_side", "arm"]
    )
    split_metrics = pd.DataFrame(split_metric_rows).sort_values(
        ["transaction_cost_bps_one_side", "arm", "split"]
    )
    whipsaw = pd.DataFrame(whipsaw_rows).sort_values("arm")
    daily_returns = pd.concat(daily_return_rows, ignore_index=True)
    positions = pd.concat(position_rows, ignore_index=True)
    event_positions = pd.concat(event_position_rows, ignore_index=True)
    turnover_costs = pd.concat(turnover_rows, ignore_index=True)
    episodes = pd.concat(episode_rows, ignore_index=True)

    metrics.to_csv(OUTPUT_DIR / f"{PREFIX}_metrics_full.csv", index=False, encoding="utf-8-sig")
    split_metrics.to_csv(
        OUTPUT_DIR / f"{PREFIX}_metrics_is_oos.csv", index=False, encoding="utf-8-sig"
    )
    whipsaw.to_csv(OUTPUT_DIR / f"{PREFIX}_whipsaw_panel.csv", index=False, encoding="utf-8-sig")
    daily_returns.to_csv(
        OUTPUT_DIR / f"{PREFIX}_raw_daily_returns.csv", index=False, encoding="utf-8-sig"
    )
    positions.to_csv(
        OUTPUT_DIR / f"{PREFIX}_raw_positions_daily_ffill.csv", index=False, encoding="utf-8-sig"
    )
    event_positions.to_csv(
        OUTPUT_DIR / f"{PREFIX}_raw_positions_execution_events.csv",
        index=False,
        encoding="utf-8-sig",
    )
    turnover_costs.to_csv(
        OUTPUT_DIR / f"{PREFIX}_raw_turnover_costs.csv", index=False, encoding="utf-8-sig"
    )
    episodes.to_csv(
        OUTPUT_DIR / f"{PREFIX}_raw_episodes.csv", index=False, encoding="utf-8-sig"
    )

    _write_report(
        metrics,
        split_metrics,
        whipsaw,
        anchor,
        warmup,
        calendar_audit,
        full_index,
        eval_index,
        train_end,
    )
    print(f"output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
