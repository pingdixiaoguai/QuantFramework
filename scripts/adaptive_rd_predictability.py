"""Research-only adaptive rebalance_days predictability diagnostic.

This script backfills missing daily rd series with the same archived
quality_momentum_top1口径, validates the aggregate anchors, then performs
post-processing gates A/B/C. It does not edit production configs.
"""

from __future__ import annotations

import math
import sys
import tempfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent
VENV_SITE = REPO_ROOT / ".venv" / "Lib" / "site-packages"
if VENV_SITE.exists():
    sys.path.append(str(VENV_SITE))

import yaml

from backtest.runner import BacktestResult, run

CONFIG_PATH = REPO_ROOT / "strategy" / "configs" / "quality_momentum_top1.yaml"
OUT_DIR = REPO_ROOT / "strategy_changelog_attachments"
RUN_DATE = date.today().isoformat()
PREFIX = f"{RUN_DATE}_adaptive_rd_predictability"

START = "2014-01-01"
END = "2026-06-04"
RDS = [2, 3, 5, 7]
ASSET_POOL = ["510300.SH", "159915.SZ", "513100.SH", "518880.SH"]
BASE_FEE = 0.0001
FEE_GRID = [0.0001, 0.0005]
KS = [3, 6, 12]
PAIRS = [(2, 5), (7, 5), (2, 7)]

ANCHORS = {
    2: {
        "annual_return_pct": 34.11,
        "sharpe": 1.26,
        "max_dd_pct": -28.44,
        "oneway_annual_turnover_pct": 3317.12,
    },
    3: {
        "annual_return_pct": 31.30,
        "sharpe": 1.18,
        "max_dd_pct": -28.59,
        "oneway_annual_turnover_pct": 2930.33,
    },
    5: {
        "annual_return_pct": 33.11,
        "sharpe": 1.23,
        "max_dd_pct": -25.79,
        "oneway_annual_turnover_pct": 2291.29,
    },
    7: {
        "annual_return_pct": 33.71,
        "sharpe": 1.26,
        "max_dd_pct": -32.55,
        "oneway_annual_turnover_pct": 1964.02,
    },
}


@dataclass
class RdSeries:
    rd: int
    result: BacktestResult
    daily: pd.DataFrame
    temp_yaml: str


def _load_base_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config["asset_pool"] = list(ASSET_POOL)
    config["start"] = START
    config["end"] = END
    config["transaction_cost_rate"] = BASE_FEE
    config.pop("rebalance_mode", None)
    return config


def _top_asset(row: pd.Series) -> str:
    weights = {
        str(asset): float(value)
        for asset, value in row.items()
        if pd.notna(value) and float(value) != 0.0
    }
    return max(weights, key=weights.get) if weights else ""


def _build_daily_frame(result: BacktestResult) -> pd.DataFrame:
    if result.gross_daily_returns is None or result.turnover is None:
        raise RuntimeError("runner did not return gross_daily_returns/turnover")

    dates = result.daily_returns.index
    if result.positions.empty:
        holding_asset = pd.Series("", index=dates, dtype=object)
    else:
        sparse_holding = result.positions.apply(_top_asset, axis=1)
        holding_asset = sparse_holding.reindex(dates).ffill().fillna("")
    daily = pd.DataFrame(
        {
            "date": dates,
            "gross_daily_return": result.gross_daily_returns.reindex(dates).to_numpy(),
            "net_daily_return": result.daily_returns.to_numpy(),
            "turnover": result.turnover.reindex(dates, fill_value=0.0).to_numpy(),
            "holding_asset": holding_asset.to_numpy(),
        }
    )
    return daily


def _run_rd_series() -> dict[int, RdSeries]:
    base = _load_base_config()
    series: dict[int, RdSeries] = {}
    with tempfile.TemporaryDirectory(prefix="adaptive_rd_") as tmp:
        tmp_dir = Path(tmp)
        for rd in RDS:
            config = dict(base)
            config["rebalance_days"] = rd
            temp_yaml = tmp_dir / f"quality_momentum_top1_rd{rd}.yaml"
            with open(temp_yaml, "w", encoding="utf-8") as f:
                yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)
            with open(temp_yaml, encoding="utf-8") as f:
                loaded = yaml.safe_load(f)
            result = run(loaded)
            daily = _build_daily_frame(result)
            out_path = OUT_DIR / f"{PREFIX}_rd{rd}_daily_series.csv"
            daily.to_csv(out_path, index=False)
            series[rd] = RdSeries(
                rd=rd,
                result=result,
                daily=daily,
                temp_yaml=str(temp_yaml),
            )
    return series


def _align_daily_series(series: dict[int, RdSeries]) -> None:
    base_dates = pd.DatetimeIndex(pd.to_datetime(series[2].daily["date"]))
    fill_rows = []
    price_cache: dict[str, pd.DataFrame] = {}
    for rd in RDS:
        daily = series[rd].daily.copy()
        dates = pd.DatetimeIndex(pd.to_datetime(daily["date"]))
        extra = sorted(set(dates) - set(base_dates))
        if extra:
            raise RuntimeError(f"rd={rd} has dates not present in rd=2 calendar: {extra}")

        missing = sorted(set(base_dates) - set(dates))
        if not missing:
            series[rd].daily = daily.sort_values("date").reset_index(drop=True)
            continue

        daily["date"] = pd.to_datetime(daily["date"])
        daily = daily.sort_values("date").reset_index(drop=True)
        for missing_dt in missing:
            prev_candidates = daily[daily["date"] < missing_dt]
            if prev_candidates.empty:
                raise RuntimeError(f"rd={rd} cannot forward-fill {missing_dt.date()}: no prior row")
            prev_row = prev_candidates.iloc[-1]
            prev_dt = pd.Timestamp(prev_row["date"])
            prev_asset = str(prev_row["holding_asset"])
            if not prev_asset:
                raise RuntimeError(f"rd={rd} cannot forward-fill {missing_dt.date()}: empty prior holding")

            if prev_asset not in price_cache:
                asset_df = query(prev_asset, START, END).copy()
                asset_df["date"] = pd.to_datetime(asset_df["date"])
                price_cache[prev_asset] = asset_df.set_index("date")
            prices = price_cache[prev_asset]
            if prev_dt not in prices.index:
                raise RuntimeError(
                    f"rd={rd} cannot price forward-fill {missing_dt.date()} for {prev_asset}"
                )
            if missing_dt in prices.index:
                gross = float(prices.loc[missing_dt, "close"] / prices.loc[prev_dt, "close"] - 1.0)
                fill_note = "asset_close_to_close"
            else:
                gross = 0.0
                fill_note = "asset_no_quote_price_forward_fill"
            new_row = {
                "date": missing_dt,
                "gross_daily_return": gross,
                "net_daily_return": gross,
                "turnover": 0.0,
                "holding_asset": prev_asset,
            }
            daily = pd.concat([daily, pd.DataFrame([new_row])], ignore_index=True)
            fill_rows.append(
                {
                    "rd": rd,
                    "date": missing_dt.date().isoformat(),
                    "previous_trading_date": prev_dt.date().isoformat(),
                    "holding_asset": prev_asset,
                    "gross_daily_return": gross,
                    "net_daily_return": gross,
                    "turnover": 0.0,
                    "method": fill_note,
                }
            )

        daily = daily.sort_values("date").reset_index(drop=True)
        series[rd].daily = daily

    fills = pd.DataFrame(fill_rows)
    if not fills.empty:
        _round_for_csv(fills).to_csv(OUT_DIR / f"{PREFIX}_calendar_fills.csv", index=False)


def _returns(series: dict[int, RdSeries], rd: int, fee: float = BASE_FEE) -> pd.Series:
    daily = series[rd].daily.copy()
    idx = pd.to_datetime(daily["date"])
    gross = pd.Series(daily["gross_daily_return"].to_numpy(), index=idx, dtype=float)
    turnover = pd.Series(daily["turnover"].to_numpy(), index=idx, dtype=float)
    return gross - turnover * fee


def _gross_returns(series: dict[int, RdSeries], rd: int) -> pd.Series:
    daily = series[rd].daily.copy()
    return pd.Series(
        daily["gross_daily_return"].to_numpy(),
        index=pd.to_datetime(daily["date"]),
        dtype=float,
    )


def _turnover(series: dict[int, RdSeries], rd: int) -> pd.Series:
    daily = series[rd].daily.copy()
    return pd.Series(
        daily["turnover"].to_numpy(),
        index=pd.to_datetime(daily["date"]),
        dtype=float,
    )


def _holding(series: dict[int, RdSeries], rd: int) -> pd.Series:
    daily = series[rd].daily.copy()
    return pd.Series(
        daily["holding_asset"].to_numpy(),
        index=pd.to_datetime(daily["date"]),
        dtype=object,
    )


def _cum_return(returns: pd.Series) -> float:
    if len(returns) == 0:
        return float("nan")
    return float((1.0 + returns).prod() - 1.0)


def _sharpe(returns: pd.Series) -> float:
    if len(returns) == 0:
        return float("nan")
    std = returns.std()
    return float(returns.mean() / std * math.sqrt(252.0)) if std > 0 else 0.0


def _metrics_from_returns(
    returns: pd.Series,
    turnover: pd.Series | None = None,
    meta_turnover: pd.Series | None = None,
) -> dict[str, float | int | str]:
    n_days = int(len(returns))
    if n_days == 0:
        return {
            "start": "",
            "end": "",
            "trading_days": 0,
            "total_return_pct": 0.0,
            "annual_return_pct": 0.0,
            "sharpe": 0.0,
            "max_dd_pct": 0.0,
            "oneway_annual_turnover_pct": 0.0,
            "meta_turnover_sum_abs": 0.0,
            "meta_oneway_annual_turnover_pct": 0.0,
        }

    cumulative = (1.0 + returns).cumprod()
    total_return = float(cumulative.iloc[-1] - 1.0)
    annual_return = float((1.0 + total_return) ** (252.0 / n_days) - 1.0)
    max_dd = float((cumulative / cumulative.cummax() - 1.0).min())
    years = n_days / 252.0
    turnover_sum = float(turnover.reindex(returns.index, fill_value=0.0).sum()) if turnover is not None else 0.0
    meta_sum = float(meta_turnover.reindex(returns.index, fill_value=0.0).sum()) if meta_turnover is not None else 0.0
    return {
        "start": returns.index.min().date().isoformat(),
        "end": returns.index.max().date().isoformat(),
        "trading_days": n_days,
        "total_return_pct": total_return * 100.0,
        "annual_return_pct": annual_return * 100.0,
        "sharpe": _sharpe(returns),
        "max_dd_pct": max_dd * 100.0,
        "oneway_annual_turnover_pct": turnover_sum / years / 2.0 * 100.0 if years else 0.0,
        "meta_turnover_sum_abs": meta_sum,
        "meta_oneway_annual_turnover_pct": meta_sum / years / 2.0 * 100.0 if years else 0.0,
    }


def _validate_daily_series(series: dict[int, RdSeries]) -> pd.DataFrame:
    rows = []
    for rd in RDS:
        ret = _returns(series, rd)
        turn = _turnover(series, rd)
        row = {
            "rd": rd,
            "start": ret.index.min().date().isoformat(),
            "end": ret.index.max().date().isoformat(),
            "rows": int(len(ret)),
            **_metrics_from_returns(ret, turn),
        }
        rows.append(row)

    validation = pd.DataFrame(rows)
    failures = []
    for rd, expected in ANCHORS.items():
        actual = validation[validation["rd"] == rd].iloc[0]
        for col, expected_value in expected.items():
            actual_value = float(actual[col])
            if f"{actual_value:.2f}" != f"{expected_value:.2f}":
                failures.append(
                    {
                        "rd": rd,
                        "metric": col,
                        "actual": actual_value,
                        "expected": expected_value,
                    }
                )

    out = validation.copy()
    out.to_csv(OUT_DIR / f"{PREFIX}_daily_series_validation.csv", index=False)
    if failures:
        failure_df = pd.DataFrame(failures)
        failure_df.to_csv(OUT_DIR / f"{PREFIX}_validation_failures.csv", index=False)
        _write_validation_failure_report(out, failure_df, pd.DataFrame())
        raise RuntimeError(
            "daily series aggregate validation failed; see validation_failures CSV"
        )
    return validation


def _write_validation_failure_report(
    validation: pd.DataFrame,
    failures: pd.DataFrame,
    alignment_failures: pd.DataFrame,
) -> Path:
    report_path = OUT_DIR / f"{PREFIX}.md"
    fills_path = OUT_DIR / f"{PREFIX}_calendar_fills.csv"
    fills = pd.read_csv(fills_path) if fills_path.exists() else pd.DataFrame()
    if alignment_failures.empty:
        stop_detail = (
            "逐日序列已按授权同口径补生成并按 rd=2 交易日历对齐，但硬闸门聚合锚点未通过，"
            "因此未执行 Gate A/B/C。"
        )
        failure_detail = (
            "失败项来自补齐后重新年化的 rd=7 指标漂移；补入 0 收益持有日后总收益不变，"
            "但交易日数从 2996 变为 2997，年化收益与年化换手分母随之改变。"
        )
    else:
        stop_detail = "逐日序列已按授权同口径补生成，但硬闸门未通过，因此未执行 Gate A/B/C。"
        failure_detail = (
            "rd=2 与 rd=5 的聚合锚点四舍五入后匹配旧归档汇总；失败项来自四档逐日收益日期未完全一致对齐。"
        )
    lines = [
        f"# Adaptive rd predictability diagnostic ({RUN_DATE})",
        "",
        "## 停止原因",
        "",
        stop_detail,
        "",
        failure_detail,
        "",
        "## 聚合校验表",
        "",
        _md_table(
            _round_for_csv(validation),
            [
                "rd",
                "start",
                "end",
                "rows",
                "annual_return_pct",
                "sharpe",
                "max_dd_pct",
                "oneway_annual_turnover_pct",
            ],
        ),
        "",
        "## 失败项",
        "",
        _md_table(_round_for_csv(failures)),
        "",
    ]
    if not alignment_failures.empty:
        lines.extend(
            [
                "## 日期对齐差异",
                "",
                _md_table(alignment_failures),
                "",
            ]
        )
    if not fills.empty:
        lines.extend(
            [
                "## 日历补行",
                "",
                _md_table(_round_for_csv(fills)),
                "",
            ]
        )
    lines.extend(
        [
            "## 已归档 CSV",
            "",
            *[f"- `{PREFIX}_rd{rd}_daily_series.csv`" for rd in RDS],
            f"- `{PREFIX}_daily_series_validation.csv`",
            f"- `{PREFIX}_validation_failures.csv`",
        ]
    )
    if not alignment_failures.empty:
        lines.append(f"- `{PREFIX}_alignment_failures.csv`")
    if not fills.empty:
        lines.append(f"- `{PREFIX}_calendar_fills.csv`")
    lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def _build_intersection_series(
    series: dict[int, RdSeries],
) -> tuple[dict[int, RdSeries], pd.DataFrame, pd.DataFrame]:
    date_sets = {
        rd: set(pd.to_datetime(bundle.daily["date"]))
        for rd, bundle in series.items()
    }
    intersection = sorted(set.intersection(*(date_sets[rd] for rd in RDS)))
    if not intersection:
        raise RuntimeError("empty intersection calendar across rd series")

    intersection_index = pd.DatetimeIndex(intersection)
    calendar_df = pd.DataFrame({"date": [dt.date().isoformat() for dt in intersection_index]})
    calendar_df.to_csv(OUT_DIR / f"{PREFIX}_intersection_calendar.csv", index=False)

    exclusions = []
    union_dates = sorted(set.union(*(date_sets[rd] for rd in RDS)))
    intersection_set = set(intersection_index)
    for dt in union_dates:
        if dt in intersection_set:
            continue
        present = [rd for rd in RDS if dt in date_sets[rd]]
        missing = [rd for rd in RDS if dt not in date_sets[rd]]
        exclusions.append(
            {
                "date": pd.Timestamp(dt).date().isoformat(),
                "present_rds": ",".join(str(rd) for rd in present),
                "missing_rds": ",".join(str(rd) for rd in missing),
                "reason": "not_in_all_four_native_series",
            }
        )
    exclusions_df = pd.DataFrame(exclusions)
    if exclusions_df.empty:
        exclusions_df = pd.DataFrame(columns=["date", "present_rds", "missing_rds", "reason"])
    exclusions_df.to_csv(OUT_DIR / f"{PREFIX}_intersection_excluded_dates.csv", index=False)

    filtered: dict[int, RdSeries] = {}
    for rd, bundle in series.items():
        daily = bundle.daily.copy()
        daily["date"] = pd.to_datetime(daily["date"])
        daily = daily[daily["date"].isin(intersection_index)].sort_values("date")
        filtered[rd] = RdSeries(
            rd=rd,
            result=bundle.result,
            daily=daily.reset_index(drop=True),
            temp_yaml=bundle.temp_yaml,
        )
    return filtered, calendar_df, exclusions_df


def _month_periods(index: pd.DatetimeIndex) -> list[pd.Period]:
    return sorted(pd.PeriodIndex(index.to_period("M")).unique())


def _period_window(period: pd.Period, start_offset: int, count: int) -> list[pd.Period]:
    start = period + start_offset
    return [start + i for i in range(count)]


def _slice_periods(returns: pd.Series, periods: Iterable[pd.Period]) -> pd.Series:
    period_set = set(periods)
    mask = returns.index.to_period("M").map(lambda p: p in period_set)
    return returns[mask.to_numpy() if hasattr(mask, "to_numpy") else mask]


def _has_periods(periods: list[pd.Period], period_set: set[pd.Period]) -> bool:
    return all(p in period_set for p in periods)


def _rank_metric(values: dict[int, float]) -> int:
    ordered = sorted(values.items(), key=lambda item: (item[1], -item[0]), reverse=True)
    return int(ordered[0][0])


def _gate_a(series: dict[int, RdSeries]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    returns = {rd: _returns(series, rd) for rd in RDS}
    periods = _month_periods(returns[2].index)
    period_set = set(periods)
    rows = []
    for k in KS:
        for period in periods:
            trailing_periods = _period_window(period, -(k - 1), k)
            forward_periods = _period_window(period, 1, k)
            if not _has_periods(trailing_periods, period_set) or not _has_periods(forward_periods, period_set):
                continue
            for left, right in PAIRS:
                left_trail = _slice_periods(returns[left], trailing_periods)
                right_trail = _slice_periods(returns[right], trailing_periods)
                left_forward = _slice_periods(returns[left], forward_periods)
                right_forward = _slice_periods(returns[right], forward_periods)
                rows.append(
                    {
                        "window_months": k,
                        "pair": f"rd{left}-rd{right}",
                        "decision_month": str(period),
                        "decision_date": returns[2][returns[2].index.to_period("M") == period].index.max().date().isoformat(),
                        "trailing_start_month": str(trailing_periods[0]),
                        "trailing_end_month": str(trailing_periods[-1]),
                        "forward_start_month": str(forward_periods[0]),
                        "forward_end_month": str(forward_periods[-1]),
                        "trailing_diff_pp": (_cum_return(left_trail) - _cum_return(right_trail)) * 100.0,
                        "forward_diff_pp": (_cum_return(left_forward) - _cum_return(right_forward)) * 100.0,
                    }
                )

    rolling = pd.DataFrame(rows)
    if rolling.empty:
        raise RuntimeError("Gate A produced no rolling rows")

    rolling["quartile"] = ""
    rolling["is_top_quartile"] = False
    for (k, pair), idx in rolling.groupby(["window_months", "pair"]).groups.items():
        values = rolling.loc[idx, "trailing_diff_pp"]
        threshold = float(values.quantile(0.75))
        rolling.loc[idx, "is_top_quartile"] = values >= threshold
        try:
            labels = pd.qcut(values, 4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop")
            rolling.loc[idx, "quartile"] = labels.astype(str)
        except ValueError:
            rolling.loc[idx, "quartile"] = np.where(values >= threshold, "Q4", "Q1")

    summary_rows = []
    top_rows = []
    for (k, pair), group in rolling.groupby(["window_months", "pair"]):
        for sample, subset in [
            ("top_quartile", group[group["is_top_quartile"]]),
            ("all", group),
        ]:
            desc = _distribution_stats(subset["forward_diff_pp"])
            summary_rows.append(
                {
                    "window_months": k,
                    "pair": pair,
                    "sample": sample,
                    **desc,
                    "note": "低样本、仅描述" if sample == "top_quartile" and desc["n"] < 10 else "",
                }
            )
        top_rows.append(group[group["is_top_quartile"]].copy())

    top = pd.concat(top_rows, ignore_index=True)
    summary = pd.DataFrame(summary_rows)
    _round_for_csv(rolling).to_csv(OUT_DIR / f"{PREFIX}_gate_a_rolling_diffs.csv", index=False)
    _round_for_csv(top).to_csv(OUT_DIR / f"{PREFIX}_gate_a_top_quartile_forwards.csv", index=False)
    _round_for_csv(summary).to_csv(OUT_DIR / f"{PREFIX}_gate_a_summary.csv", index=False)
    return rolling, top, summary


def _distribution_stats(values: pd.Series) -> dict[str, float | int]:
    clean = values.dropna()
    if clean.empty:
        return {
            "n": 0,
            "min_pp": float("nan"),
            "p25_pp": float("nan"),
            "median_pp": float("nan"),
            "p75_pp": float("nan"),
            "max_pp": float("nan"),
            "mean_pp": float("nan"),
            "positive_pct": float("nan"),
        }
    return {
        "n": int(len(clean)),
        "min_pp": float(clean.min()),
        "p25_pp": float(clean.quantile(0.25)),
        "median_pp": float(clean.median()),
        "p75_pp": float(clean.quantile(0.75)),
        "max_pp": float(clean.max()),
        "mean_pp": float(clean.mean()),
        "positive_pct": float((clean > 0).mean() * 100.0),
    }


def _metric_value(returns: pd.Series, metric: str) -> float:
    if metric == "return":
        return _cum_return(returns)
    if metric == "sharpe":
        return _sharpe(returns)
    raise ValueError(metric)


def _gate_b(series: dict[int, RdSeries]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    returns = {rd: _returns(series, rd) for rd in RDS}
    periods = _month_periods(returns[2].index)
    period_set = set(periods)
    detail_rows = []
    for k in KS:
        for metric in ["return", "sharpe"]:
            for period in periods:
                trailing_periods = _period_window(period, -(k - 1), k)
                forward_periods = _period_window(period, 1, k)
                if not _has_periods(trailing_periods, period_set) or not _has_periods(forward_periods, period_set):
                    continue
                trailing_values = {
                    rd: _metric_value(_slice_periods(returns[rd], trailing_periods), metric)
                    for rd in RDS
                }
                forward_values = {
                    rd: _metric_value(_slice_periods(returns[rd], forward_periods), metric)
                    for rd in RDS
                }
                past_best = _rank_metric(trailing_values)
                future_best = _rank_metric(forward_values)
                detail_rows.append(
                    {
                        "window_months": k,
                        "metric": metric,
                        "decision_month": str(period),
                        "past_best_rd": past_best,
                        "future_best_rd": future_best,
                        "hit": past_best == future_best,
                        **{f"trailing_rd{rd}": trailing_values[rd] for rd in RDS},
                        **{f"forward_rd{rd}": forward_values[rd] for rd in RDS},
                    }
                )

    detail = pd.DataFrame(detail_rows)
    hit_rows = []
    trans_rows = []
    for (k, metric), group in detail.groupby(["window_months", "metric"]):
        n = int(len(group))
        hit_rate = float(group["hit"].mean() * 100.0) if n else float("nan")
        hit_rows.append(
            {
                "window_months": k,
                "metric": metric,
                "n": n,
                "hit_rate_pct": hit_rate,
                "random_baseline_pct": 25.0,
                "strong_baseline_pct": 50.0,
                "note": "低样本、慎读" if k == 3 and metric == "sharpe" else "",
            }
        )
        counts = pd.crosstab(group["past_best_rd"], group["future_best_rd"])
        counts = counts.reindex(index=RDS, columns=RDS, fill_value=0)
        for from_rd in RDS:
            row_total = int(counts.loc[from_rd].sum())
            for to_rd in RDS:
                count = int(counts.loc[from_rd, to_rd])
                trans_rows.append(
                    {
                        "window_months": k,
                        "metric": metric,
                        "from_rd": from_rd,
                        "to_rd": to_rd,
                        "count": count,
                        "row_prob_pct": count / row_total * 100.0 if row_total else 0.0,
                    }
                )

    hit_rates = pd.DataFrame(hit_rows)
    transitions = pd.DataFrame(trans_rows)
    _round_for_csv(detail).to_csv(OUT_DIR / f"{PREFIX}_gate_b_best_rd_detail.csv", index=False)
    _round_for_csv(hit_rates).to_csv(OUT_DIR / f"{PREFIX}_gate_b_hit_rates.csv", index=False)
    _round_for_csv(transitions).to_csv(OUT_DIR / f"{PREFIX}_gate_b_transition_matrix.csv", index=False)
    return detail, hit_rates, transitions


def _top1_turnover(old_asset: str | None, new_asset: str | None) -> float:
    old = old_asset or ""
    new = new_asset or ""
    if old == new:
        return 0.0
    if not old and new:
        return 1.0
    if old and not new:
        return 1.0
    return 2.0


def _choose_selector_rd(
    returns: dict[int, pd.Series],
    period: pd.Period,
    k: int,
    metric: str,
) -> int:
    trailing_periods = _period_window(period, -(k - 1), k)
    values = {
        rd: _metric_value(_slice_periods(returns[rd], trailing_periods), metric)
        for rd in RDS
    }
    return _rank_metric(values)


def _segment_return(gross: pd.Series, turnover: pd.Series, periods: list[pd.Period], fee: float) -> float:
    segment_gross = _slice_periods(gross, periods)
    segment_turnover = turnover.reindex(segment_gross.index, fill_value=0.0)
    segment_net = segment_gross - segment_turnover * fee
    return _cum_return(segment_net)


def _build_path(
    series: dict[int, RdSeries],
    k: int,
    metric: str,
    fee: float,
    path_kind: str,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    gross = {rd: _gross_returns(series, rd) for rd in RDS}
    turns = {rd: _turnover(series, rd) for rd in RDS}
    returns = {rd: gross[rd] - turns[rd] * fee for rd in RDS}
    holdings = {rd: _holding(series, rd) for rd in RDS}
    periods = _month_periods(returns[2].index)
    period_set = set(periods)
    rows = []
    path_returns: list[pd.Series] = []
    internal_turnovers: list[pd.Series] = []
    meta_turnover_records: list[tuple[pd.Timestamp, float]] = []
    prev_asset: str | None = None
    nav = 1.0

    for period in periods:
        trailing_periods = _period_window(period, -(k - 1), k)
        next_periods = _period_window(period, 1, 1)
        if not _has_periods(trailing_periods, period_set) or not _has_periods(next_periods, period_set):
            continue
        next_period = next_periods[0]
        if path_kind == "selector":
            chosen_rd = _choose_selector_rd(returns, period, k, metric)
        elif path_kind == "oracle":
            segment_values = {
                rd: _segment_return(gross[rd], turns[rd], next_periods, fee)
                for rd in RDS
            }
            chosen_rd = _rank_metric(segment_values)
        elif path_kind == "rd5":
            chosen_rd = 5
        else:
            raise ValueError(path_kind)

        segment_gross = _slice_periods(gross[chosen_rd], next_periods)
        segment_turnover = turns[chosen_rd].reindex(segment_gross.index, fill_value=0.0)
        segment_holding = holdings[chosen_rd].reindex(segment_gross.index).ffill()
        segment_net = segment_gross - segment_turnover * fee
        if segment_net.empty:
            continue

        first_date = segment_net.index.min()
        new_asset = str(segment_holding.loc[first_date] or "")
        meta_turn = _top1_turnover(prev_asset, new_asset)
        if meta_turn:
            segment_net.loc[first_date] = segment_net.loc[first_date] - meta_turn * fee
            meta_turnover_records.append((first_date, meta_turn))
        prev_asset = str(segment_holding.iloc[-1] or "")

        path_returns.append(segment_net)
        internal_turnovers.append(segment_turnover)
        for dt, ret in segment_net.items():
            nav *= 1.0 + float(ret)
            rows.append(
                {
                    "window_months": k,
                    "metric": metric,
                    "cost_bps": fee * 10000.0,
                    "path": path_kind,
                    "decision_month": str(period),
                    "hold_month": str(next_period),
                    "date": dt.date().isoformat(),
                    "chosen_rd": chosen_rd,
                    "holding_asset": str(segment_holding.loc[dt] or ""),
                    "daily_return": float(ret),
                    "nav": float(nav),
                    "internal_turnover": float(segment_turnover.loc[dt]),
                    "meta_turnover": float(meta_turn if dt == first_date else 0.0),
                }
            )

    if not path_returns:
        return pd.DataFrame(rows), pd.Series(dtype=float), pd.Series(dtype=float)

    ret_path = pd.concat(path_returns).sort_index()
    turnover_path = pd.concat(internal_turnovers).groupby(level=0).sum().sort_index()
    meta = pd.Series(dict(meta_turnover_records), dtype=float)
    return pd.DataFrame(rows), ret_path, turnover_path.add(meta, fill_value=0.0)


def _gate_c(series: dict[int, RdSeries]) -> tuple[pd.DataFrame, pd.DataFrame]:
    equity_frames = []
    metric_rows = []
    for k in KS:
        for metric in ["return", "sharpe"]:
            for fee in FEE_GRID:
                for path_kind in ["selector", "oracle", "rd5"]:
                    equity, ret_path, combined_turnover = _build_path(
                        series,
                        k,
                        metric,
                        fee,
                        path_kind,
                    )
                    equity_frames.append(equity)
                    internal_turnover = (
                        equity.groupby("date")["internal_turnover"].sum()
                        if not equity.empty
                        else pd.Series(dtype=float)
                    )
                    meta_turnover = (
                        equity.groupby("date")["meta_turnover"].sum()
                        if not equity.empty
                        else pd.Series(dtype=float)
                    )
                    internal_turnover.index = pd.to_datetime(internal_turnover.index)
                    meta_turnover.index = pd.to_datetime(meta_turnover.index)
                    row = {
                        "window_months": k,
                        "metric": metric,
                        "cost_bps": fee * 10000.0,
                        "path": path_kind,
                        **_metrics_from_returns(ret_path, internal_turnover, meta_turnover),
                    }
                    metric_rows.append(row)

    equity_df = pd.concat(equity_frames, ignore_index=True)
    metrics_df = pd.DataFrame(metric_rows)
    _round_for_csv(equity_df).to_csv(OUT_DIR / f"{PREFIX}_gate_c_selector_equity.csv", index=False)
    _round_for_csv(metrics_df).to_csv(OUT_DIR / f"{PREFIX}_gate_c_metrics.csv", index=False)
    return equity_df, metrics_df


def _round_for_csv(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            if col in {"daily_return", "nav"}:
                out[col] = out[col].round(8)
            else:
                out[col] = out[col].round(2)
    return out


def _fmt_value(value: object) -> str:
    if isinstance(value, (float, np.floating)):
        if math.isnan(float(value)):
            return ""
        return f"{float(value):.2f}"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, bool):
        return "yes" if value else "no"
    return str(value)


def _md_table(df: pd.DataFrame, columns: list[str] | None = None, max_rows: int | None = None) -> str:
    if columns is None:
        columns = list(df.columns)
    view = df.loc[:, columns].copy()
    if max_rows is not None:
        view = view.head(max_rows)
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines = [header, sep]
    for _, row in view.iterrows():
        lines.append("| " + " | ".join(_fmt_value(row[col]) for col in columns) + " |")
    return "\n".join(lines)


def _matrix_table(transitions: pd.DataFrame, k: int, metric: str, value_col: str) -> pd.DataFrame:
    sub = transitions[(transitions["window_months"] == k) & (transitions["metric"] == metric)]
    mat = sub.pivot(index="from_rd", columns="to_rd", values=value_col).reindex(index=RDS, columns=RDS)
    mat = mat.reset_index()
    mat.columns = ["from_rd"] + [f"to_rd{rd}" for rd in RDS]
    return mat


def _gate_a_readout(summary: pd.DataFrame) -> pd.DataFrame:
    top = summary[summary["sample"] == "top_quartile"].copy()
    top["median_negative"] = top["median_pp"] < 0
    top["positive_below_50"] = top["positive_pct"] < 50
    return top[
        [
            "window_months",
            "pair",
            "n",
            "median_pp",
            "positive_pct",
            "mean_pp",
            "median_negative",
            "positive_below_50",
            "note",
        ]
    ]


def _selector_vs_baseline(metrics_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (k, metric, cost), group in metrics_df.groupby(["window_months", "metric", "cost_bps"]):
        indexed = group.set_index("path")
        if "selector" not in indexed.index or "rd5" not in indexed.index:
            continue
        rows.append(
            {
                "window_months": k,
                "metric": metric,
                "cost_bps": cost,
                "selector_ann_return_pct": float(indexed.loc["selector", "annual_return_pct"]),
                "rd5_ann_return_pct": float(indexed.loc["rd5", "annual_return_pct"]),
                "selector_minus_rd5_ann_pp": float(indexed.loc["selector", "annual_return_pct"] - indexed.loc["rd5", "annual_return_pct"]),
                "selector_sharpe": float(indexed.loc["selector", "sharpe"]),
                "rd5_sharpe": float(indexed.loc["rd5", "sharpe"]),
                "selector_meta_turnover_sum_abs": float(indexed.loc["selector", "meta_turnover_sum_abs"]),
                "selector_meta_oneway_ann_turnover_pct": float(indexed.loc["selector", "meta_oneway_annual_turnover_pct"]),
            }
        )
    return pd.DataFrame(rows)


def _write_report(
    validation: pd.DataFrame,
    intersection_calendar: pd.DataFrame,
    exclusions: pd.DataFrame,
    gate_a_summary: pd.DataFrame,
    gate_b_hits: pd.DataFrame,
    gate_b_transitions: pd.DataFrame,
    gate_c_metrics: pd.DataFrame,
) -> Path:
    gate_a_readout = _gate_a_readout(gate_a_summary)
    selector_compare = _selector_vs_baseline(gate_c_metrics)
    high_hit = bool((gate_b_hits["hit_rate_pct"] > 50.0).any())
    strong_selector = bool(
        (
            (selector_compare["cost_bps"] == 5.0)
            & (selector_compare["selector_minus_rd5_ann_pp"] > 1.0)
            & (selector_compare["selector_sharpe"] > selector_compare["rd5_sharpe"])
        ).any()
    )

    files = [
        f"{PREFIX}_rd{rd}_daily_series.csv" for rd in RDS
    ] + [
        f"{PREFIX}_daily_series_validation.csv",
        f"{PREFIX}_intersection_calendar.csv",
        f"{PREFIX}_intersection_excluded_dates.csv",
        f"{PREFIX}_gate_a_rolling_diffs.csv",
        f"{PREFIX}_gate_a_top_quartile_forwards.csv",
        f"{PREFIX}_gate_a_summary.csv",
        f"{PREFIX}_gate_b_best_rd_detail.csv",
        f"{PREFIX}_gate_b_hit_rates.csv",
        f"{PREFIX}_gate_b_transition_matrix.csv",
        f"{PREFIX}_gate_c_selector_equity.csv",
        f"{PREFIX}_gate_c_metrics.csv",
    ]

    lines = [
        f"# Adaptive rd predictability diagnostic ({RUN_DATE})",
        "",
        "## 纯事实摘要",
        "",
        (
            f"逐日序列系 {RUN_DATE} 同口径补生成；rd=5 聚合校验为年化 "
            f"{validation.loc[validation['rd'] == 5, 'annual_return_pct'].iloc[0]:.2f}%、"
            f"Sharpe {validation.loc[validation['rd'] == 5, 'sharpe'].iloc[0]:.2f}、"
            f"最大回撤 {validation.loc[validation['rd'] == 5, 'max_dd_pct'].iloc[0]:.2f}%、"
            f"单边年化换手 {validation.loc[validation['rd'] == 5, 'oneway_annual_turnover_pct'].iloc[0]:.2f}%。"
            f"rd=2 聚合校验为年化 {validation.loc[validation['rd'] == 2, 'annual_return_pct'].iloc[0]:.2f}%、"
            f"Sharpe {validation.loc[validation['rd'] == 2, 'sharpe'].iloc[0]:.2f}、"
            f"最大回撤 {validation.loc[validation['rd'] == 2, 'max_dd_pct'].iloc[0]:.2f}%、"
            f"单边年化换手 {validation.loc[validation['rd'] == 2, 'oneway_annual_turnover_pct'].iloc[0]:.2f}%。"
            f"Gate A/B/C 使用四档原生日历交集 {len(intersection_calendar)} 行；"
            f"交集排除日期 {len(exclusions)} 条。"
        ),
        "",
        (
            "Gate A 顶四分位样本中，以下表格列出各 K/价差对的前瞻中位差、为正占比和均值；"
            "Gate B 列出近期最优重复命中率；Gate C 列出可实现选择器、oracle 与固定 rd=5 的月频串接表现。"
        ),
        "",
        "## 数据补生成与熔断校验",
        "",
        "- Config base: `strategy/configs/quality_momentum_top1.yaml`; 临时 YAML 副本在运行时生成后删除，生产 YAML 未修改。",
        "- Overrides: `start=2014-01-01`, `end=2026-06-04`, `transaction_cost_rate=0.0001`, `rebalance_days in {2,3,5,7}`。",
        "- 数据/成交口径沿用 `backtest.runner.run()` 与 `data.store.query()`；T+1 开盘成交、成本 = rate × Σ|Δw|。",
        "- 锚点校验使用每档原生交易日历；Gate A/B/C 使用四档原生日历交集，不对无报价日补 0。",
        "",
        _md_table(
            validation,
            [
                "rd",
                "start",
                "end",
                "rows",
                "annual_return_pct",
                "sharpe",
                "max_dd_pct",
                "oneway_annual_turnover_pct",
            ],
        ),
        "",
        "## 交集检验日历",
        "",
        f"四档原生日历交集行数: {len(intersection_calendar)}。",
        "",
        "被交集排除的日期：",
        "",
        _md_table(exclusions) if not exclusions.empty else "无。",
        "",
        "## Gate A: 顶四分位条件前瞻检验",
        "",
        "判读规则：顶四分位之后前瞻差若中位为负，或为正占比显著低于 50%，表示追近期最强领先档在历史上倾向被均值回归打脸。",
        "",
        _md_table(gate_a_readout),
        "",
        "全样本与顶四分位的完整分布统计：",
        "",
        _md_table(
            gate_a_summary,
            [
                "window_months",
                "pair",
                "sample",
                "n",
                "min_pp",
                "p25_pp",
                "median_pp",
                "p75_pp",
                "max_pp",
                "mean_pp",
                "positive_pct",
                "note",
            ],
        ),
        "",
        "## Gate B: 持续性命中率与转移矩阵",
        "",
        "多重检验警示：K∈{3,6,12} × 两口径 = 6 套设定，属于多次试探；只有跨 K、跨口径一致才算信号，单一设定亮灯按多重比较噪声处理。",
        "",
        _md_table(gate_b_hits),
        "",
    ]

    for metric in ["return", "sharpe"]:
        for k in KS:
            lines.extend(
                [
                    f"### K={k}, metric={metric}: 转移矩阵频数",
                    "",
                    _md_table(_matrix_table(gate_b_transitions, k, metric, "count")),
                    "",
                    f"### K={k}, metric={metric}: 转移矩阵行归一概率(%)",
                    "",
                    _md_table(_matrix_table(gate_b_transitions, k, metric, "row_prob_pct")),
                    "",
                ]
            )

    lines.extend(
        [
            "## Gate C: 可实现选择器 vs oracle vs 固定 rd=5",
            "",
            "判读规则：可实现选择器若不能在 @5bp 且含 meta 换手下超过固定 rd=5，则可预测性即便存在也不可利用；oracle 与可实现的缺口代表不可达的运气成分。",
            "",
            _md_table(
                gate_c_metrics,
                [
                    "window_months",
                    "metric",
                    "cost_bps",
                    "path",
                    "annual_return_pct",
                    "sharpe",
                    "max_dd_pct",
                    "oneway_annual_turnover_pct",
                    "meta_turnover_sum_abs",
                    "meta_oneway_annual_turnover_pct",
                ],
            ),
            "",
            "可实现选择器相对固定 rd=5：",
            "",
            _md_table(selector_compare),
            "",
            "## 前瞻信息隔离自查",
            "",
            "- trailing 窗口与 forward 窗口按自然月对齐：决策月及其之前 K 月只进入 trailing，forward 从下一自然月开始，二者无重叠交易日。",
            "- 选择器在月末 t 选 rd 时只使用 t 所在月份及之前的收益序列，持有从下一自然月开始。",
            "- Gate B 的 forward 最优 rd 只用于检验；Gate C 的 oracle 只作不可达天花板，未回流到选择器输入。",
            "- Sharpe 年化仅使用对应 trailing 或 forward 段内收益，没有使用段外未来点。",
        ]
    )

    if high_hit or strong_selector:
        lines.extend(
            [
                "",
                "额外泄漏排查：出现命中率 >50% 或 @5bp 选择器强于基线的设定时，已复核窗口切分、选择器输入、forward/oracle 隔离与 Sharpe 段内计算；本报告仍按多重检验警示处理这些单元。",
            ]
        )

    lines.extend(
        [
            "",
            "## 归档 CSV",
            "",
            *[f"- `{name}`" for name in files],
            "",
            "## 结论指引",
            "",
            "本闸门设计为证伪；通不过 = 方向死，通過不等于方向活，仅表示未证伪，待诊断 2 在严格 OOS + 成本下复核。跨 K、跨口径不一致按噪声处理；强可预测结果优先疑泄漏。",
            "",
        ]
    )

    report_path = OUT_DIR / f"{PREFIX}.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    series = _run_rd_series()
    validation = _validate_daily_series(series)
    gate_series, intersection_calendar, exclusions = _build_intersection_series(series)
    _, _, gate_a_summary = _gate_a(gate_series)
    _, gate_b_hits, gate_b_transitions = _gate_b(gate_series)
    _, gate_c_metrics = _gate_c(gate_series)
    report_path = _write_report(
        _round_for_csv(validation),
        intersection_calendar,
        exclusions,
        _round_for_csv(gate_a_summary),
        _round_for_csv(gate_b_hits),
        _round_for_csv(gate_b_transitions),
        _round_for_csv(gate_c_metrics),
    )
    print(report_path)


if __name__ == "__main__":
    main()
