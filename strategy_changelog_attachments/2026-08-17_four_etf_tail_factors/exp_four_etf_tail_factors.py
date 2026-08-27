"""Fixed four-ETF Tushare multi-field tail-factor study."""

from __future__ import annotations

import math
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from backtest.runner import run as run_official  # noqa: E402
from data import store  # noqa: E402
from run_backtest import _load_config_from_yaml  # noqa: E402

HERE = Path(__file__).resolve().parent
CACHE = ROOT / "data/db/four_etf_tushare_fields.parquet"
CONFIG = ROOT / "strategy/configs/quality_momentum_top1.yaml"
PREFIX = "2026-08-17_four_etf_tail_factors"
CORE = ["510300.SH", "159915.SZ", "513100.SH", "518880.SH"]
WARMUP_START = pd.Timestamp("2013-01-01")
EVAL_START = pd.Timestamp("2014-01-02")
END = pd.Timestamp("2026-08-14")
D_END = pd.Timestamp("2018-12-31")
V_START = pd.Timestamp("2019-01-01")
V_END = pd.Timestamp("2022-12-30")
T_START = pd.Timestamp("2023-01-01")
REBALANCE_DAYS = 5
FEE_MAIN = 0.0001
FEE_STRESS = 0.0005


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


def robust_positive_z(series: pd.Series, window: int = 60) -> pd.Series:
    median = series.rolling(window, min_periods=40).median()
    absolute_deviation = (series - median).abs()
    mad = absolute_deviation.rolling(window, min_periods=40).median()
    difference = series - median
    scale = 1.4826 * mad
    z = difference / scale.replace(0.0, np.nan)
    zero_scale = scale.eq(0.0) & difference.notna()
    z = z.mask(zero_scale & difference.le(0.0), 0.0)
    z = z.mask(zero_scale & difference.gt(0.0), 10.0)
    return z.clip(lower=0.0, upper=10.0)


def last_percentile(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if not len(finite) or not np.isfinite(values[-1]):
        return np.nan
    return float((finite <= values[-1]).mean())


def rolling_cvar(series: pd.Series, window: int = 20) -> pd.Series:
    count = max(1, math.ceil(window * 0.20))
    return series.rolling(window, min_periods=window).apply(
        lambda values: -float(np.sort(values)[:count].mean()), raw=True
    )


def load_panels() -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    if not CACHE.exists():
        raise RuntimeError(f"missing {CACHE}; run fetch_four_etf_tushare_fields.py")
    extra = pd.read_parquet(CACHE)
    extra["date"] = pd.to_datetime(extra["date"])
    calendar = native_series(CORE[0], "close").loc[WARMUP_START:END].index
    prices = {
        field: pd.DataFrame(index=calendar, columns=CORE, dtype=float)
        for field in ("open", "high", "low", "close")
    }
    fields = {
        field: pd.DataFrame(index=calendar, columns=CORE, dtype=float)
        for field in ("amount", "fd_share", "unit_nav", "raw_close")
    }
    for code in CORE:
        for field in prices:
            prices[field][code] = native_series(code, field).reindex(calendar)
        one = extra.loc[extra["ts_code"] == code].sort_values("date")
        exact = one.drop_duplicates("date", keep="last").set_index("date")
        fields["amount"][code] = pd.to_numeric(exact["amount"], errors="coerce").reindex(calendar)
        fields["raw_close"][code] = pd.to_numeric(exact["close"], errors="coerce").reindex(calendar)
        # Share and NAV become available only on their Tushare observation or
        # announcement date; forward filling never moves a value backward.
        fields["fd_share"][code] = pd.to_numeric(exact["fd_share"], errors="coerce").dropna().reindex(
            calendar, method="ffill"
        )
        fields["unit_nav"][code] = pd.to_numeric(exact["unit_nav"], errors="coerce").dropna().reindex(
            calendar, method="ffill"
        )
    return prices, fields


def build_factors(
    prices: dict[str, pd.DataFrame], fields: dict[str, pd.DataFrame]
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    index = prices["close"].index
    raw_risk = {
        name: pd.DataFrame(index=index, columns=CORE, dtype=float)
        for name in (
            "downside_lpm20",
            "cvar20",
            "range20",
            "gap_tail20",
            "amihud20",
            "amount_shock20",
            "share_flow20",
            "premium_crowding",
        )
    }
    qm20 = pd.DataFrame(index=index, columns=CORE, dtype=float)
    for code in CORE:
        close = prices["close"][code]
        daily_return = close.pct_change(fill_method=None)
        momentum = close.pct_change(20, fill_method=None)
        efficiency = (close - close.shift(20)).abs() / close.diff().abs().rolling(20).sum().replace(0.0, np.nan)
        qm20[code] = momentum * efficiency
        downside = daily_return.clip(upper=0.0)
        raw_risk["downside_lpm20"][code] = np.sqrt(downside.pow(2).rolling(20).mean())
        raw_risk["cvar20"][code] = rolling_cvar(daily_return)
        previous_close = close.shift(1)
        day_range = (prices["high"][code] - prices["low"][code]) / previous_close
        raw_risk["range20"][code] = day_range.rolling(20).mean()
        gap = (prices["open"][code] / previous_close - 1.0).abs()
        raw_risk["gap_tail20"][code] = gap.rolling(20).quantile(0.90)
        amount = fields["amount"][code].replace(0.0, np.nan)
        amihud = daily_return.abs() / amount
        amihud20 = amihud.rolling(20).mean()
        raw_risk["amihud20"][code] = amihud20.rolling(252, min_periods=60).apply(
            last_percentile, raw=True
        )
        raw_risk["amount_shock20"][code] = amount.rolling(20).mean() / amount.rolling(60).median()
        share_flow = fields["fd_share"][code].pct_change(20, fill_method=None)
        raw_risk["share_flow20"][code] = robust_positive_z(share_flow)
        premium = fields["raw_close"][code] / fields["unit_nav"][code] - 1.0
        raw_risk["premium_crowding"][code] = robust_positive_z(premium)

    risk_ranks = {
        name: frame.rank(axis=1, pct=True, method="average")
        for name, frame in raw_risk.items()
    }
    qm_rank = qm20.rank(axis=1, pct=True, method="average")
    downside_risk = (risk_ranks["downside_lpm20"] + risk_ranks["cvar20"]) / 2.0
    range_risk = (risk_ranks["range20"] + risk_ranks["gap_tail20"]) / 2.0
    liquidity_risk = (risk_ranks["amihud20"] + risk_ranks["amount_shock20"]) / 2.0
    flow_premium_risk = (risk_ranks["share_flow20"] + risk_ranks["premium_crowding"]) / 2.0
    multifield_risk = sum(risk_ranks.values()) / len(risk_ranks)

    scores = {"BASE_QM20": qm20}
    families = {
        "DOWNSIDE": downside_risk,
        "RANGE": range_risk,
        "LIQUIDITY": liquidity_risk,
        "FLOW_PREMIUM": flow_premium_risk,
        "MULTIFIELD": multifield_risk,
    }
    for family, risk in families.items():
        for suffix, penalty in (("25", 0.25), ("50", 0.50)):
            scores[f"{family}_{suffix}"] = qm_rank - penalty * risk

    diagnostics = []
    for name, frame in raw_risk.items():
        for code in CORE:
            evaluation = frame.loc[EVAL_START:END, code]
            diagnostics.append(
                {
                    "field": name,
                    "asset": code,
                    "non_null": int(evaluation.notna().sum()),
                    "coverage": float(evaluation.notna().mean()),
                    "first_valid": evaluation.first_valid_index(),
                    "last_valid": evaluation.last_valid_index(),
                    "median": float(evaluation.median()),
                    "p95": float(evaluation.quantile(0.95)),
                }
            )
    return {
        "scores": scores,
        "qm20": qm20,
        "multifield_risk": multifield_risk,
        "risk_ranks": risk_ranks,
        "raw_risk": raw_risk,
    }, pd.DataFrame(diagnostics)


def targets_from_score(score: pd.DataFrame) -> pd.Series:
    valid = score[CORE].notna().any(axis=1)
    target = score[CORE].fillna(-np.inf).idxmax(axis=1)
    target[~valid] = None
    return target


def veto_targets(qm20: pd.DataFrame, multifield_risk: pd.DataFrame, threshold: float) -> pd.Series:
    adjusted = qm20.copy()
    winner = targets_from_score(qm20)
    risk_available = multifield_risk.notna().any(axis=1)
    worst_risk = multifield_risk.fillna(-np.inf).idxmax(axis=1)
    worst_risk[~risk_available] = None
    own_percentile = pd.DataFrame(index=multifield_risk.index, columns=CORE, dtype=float)
    for code in CORE:
        own_percentile[code] = multifield_risk[code].rolling(252, min_periods=60).apply(
            last_percentile, raw=True
        )
    for timestamp in adjusted.index:
        code = winner.at[timestamp]
        if not isinstance(code, str):
            continue
        if worst_risk.at[timestamp] == code and own_percentile.at[timestamp, code] >= threshold:
            adjusted.at[timestamp, code] = -np.inf
    return targets_from_score(adjusted)


def safe_ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator == 0:
        return 1.0
    return float(numerator / denominator)


def simulate(targets: pd.Series, opens: pd.DataFrame, closes: pd.DataFrame) -> dict[str, pd.Series]:
    dates = closes.index
    gross = pd.Series(0.0, index=dates, dtype=float)
    turnover = pd.Series(0.0, index=dates, dtype=float)
    held = pd.Series(index=dates, dtype="object")
    current = None
    entry_idx = None
    pending = None
    pending_idx = None
    for i, timestamp in enumerate(dates):
        if i > 0:
            previous = dates[i - 1]
            old = current
            if pending_idx == i and pending is not None:
                overnight = 0.0 if old is None else safe_ratio(opens.at[timestamp, old], closes.at[previous, old]) - 1.0
                current = pending
                entry_idx = i
                turnover.at[timestamp] = 1.0 if old is None else (0.0 if old == current else 2.0)
                intraday = safe_ratio(closes.at[timestamp, current], opens.at[timestamp, current]) - 1.0
                gross.at[timestamp] = (1.0 + overnight) * (1.0 + intraday) - 1.0
                pending = None
                pending_idx = None
            elif current is not None:
                gross.at[timestamp] = safe_ratio(closes.at[timestamp, current], closes.at[previous, current]) - 1.0
        held.at[timestamp] = current
        holding_days = i - entry_idx + 1 if current is not None and entry_idx is not None else None
        should_signal = pending is None and (
            current is None or holding_days is None or holding_days >= REBALANCE_DAYS
        )
        if should_signal and i + 1 < len(dates):
            new = targets.at[timestamp]
            if isinstance(new, str) and new != current:
                next_timestamp = dates[i + 1]
                if np.isfinite(opens.at[next_timestamp, new]) and np.isfinite(closes.at[next_timestamp, new]):
                    pending = new
                    pending_idx = i + 1
    return {
        "gross": gross.loc[EVAL_START:END],
        "turnover": turnover.loc[EVAL_START:END],
        "held": held.loc[EVAL_START:END],
    }


def net(simulation: dict[str, pd.Series], fee: float) -> pd.Series:
    return simulation["gross"] - simulation["turnover"] * fee


def sharpe(returns: pd.Series) -> float:
    standard_deviation = float(returns.std())
    return float(returns.mean() / standard_deviation * math.sqrt(252.0)) if standard_deviation > 0 else 0.0


def annual_return(returns: pd.Series) -> float:
    return float((1.0 + returns).prod() ** (252.0 / len(returns)) - 1.0) if len(returns) else 0.0


def max_drawdown(returns: pd.Series) -> float:
    wealth = (1.0 + returns).cumprod()
    return float((wealth / wealth.cummax() - 1.0).min()) if len(wealth) else 0.0


def drawdown_episodes(returns: pd.Series) -> pd.DataFrame:
    if returns.empty:
        return pd.DataFrame(columns=["peak_date", "trough_date", "recovery_date", "depth", "recovery_days"])
    wealth = (1.0 + returns).cumprod()
    peak_value = float(wealth.iloc[0])
    peak_date = wealth.index[0]
    underwater = False
    trough_value = peak_value
    trough_date = peak_date
    episodes = []
    for timestamp, value_raw in wealth.iloc[1:].items():
        value = float(value_raw)
        if value >= peak_value:
            if underwater:
                recovery_days = int(wealth.loc[trough_date:timestamp].shape[0] - 1)
                episodes.append(
                    {
                        "peak_date": peak_date,
                        "trough_date": trough_date,
                        "recovery_date": timestamp,
                        "depth": trough_value / peak_value - 1.0,
                        "recovery_days": recovery_days,
                    }
                )
                underwater = False
            peak_value = value
            peak_date = timestamp
            trough_value = value
            trough_date = timestamp
        else:
            if not underwater:
                underwater = True
                trough_value = value
                trough_date = timestamp
            elif value < trough_value:
                trough_value = value
                trough_date = timestamp
    if underwater:
        episodes.append(
            {
                "peak_date": peak_date,
                "trough_date": trough_date,
                "recovery_date": pd.NaT,
                "depth": trough_value / peak_value - 1.0,
                "recovery_days": np.nan,
            }
        )
    frame = pd.DataFrame(episodes)
    if frame.empty:
        return frame
    return frame.sort_values("depth").reset_index(drop=True)


def top10_summary(returns: pd.Series) -> dict[str, float]:
    episodes = drawdown_episodes(returns).head(10)
    return {
        "top10_count": int(len(episodes)),
        "top10_mean_depth": float(episodes["depth"].mean()) if len(episodes) else 0.0,
        "top10_worst_depth": float(episodes["depth"].min()) if len(episodes) else 0.0,
        "top10_mean_recovery_days": float(episodes["recovery_days"].mean()) if len(episodes) else 0.0,
    }


def segment(returns: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    return returns.loc[start:end]


def metric_row(name: str, period: str, returns: pd.Series, turnover: pd.Series) -> dict[str, object]:
    top10 = top10_summary(returns)
    years = len(returns) / 252.0
    return {
        "candidate": name,
        "period": period,
        "days": len(returns),
        "annual_return": annual_return(returns),
        "sharpe": sharpe(returns),
        "max_drawdown": max_drawdown(returns),
        "annual_turnover_sum_abs": float(turnover.sum() / years) if years else 0.0,
        **top10,
    }


def rolling36(candidate: pd.Series, baseline: pd.Series) -> pd.DataFrame:
    joined = pd.concat([baseline.rename("baseline"), candidate.rename("candidate")], axis=1).dropna()
    rows = []
    for end in range(756, len(joined) + 1, 21):
        window = joined.iloc[end - 756:end]
        baseline_sharpe = sharpe(window["baseline"])
        candidate_sharpe = sharpe(window["candidate"])
        rows.append(
            {
                "window_end": window.index[-1],
                "baseline_sharpe": baseline_sharpe,
                "candidate_sharpe": candidate_sharpe,
                "candidate_leads": candidate_sharpe >= baseline_sharpe,
            }
        )
    return pd.DataFrame(rows)


def same_window_comparison(baseline: pd.Series, candidate: pd.Series) -> pd.DataFrame:
    episodes = drawdown_episodes(baseline).head(10).copy()
    candidate_returns = []
    for row in episodes.itertuples():
        window = candidate.loc[(candidate.index > row.peak_date) & (candidate.index <= row.trough_date)]
        candidate_returns.append(float((1.0 + window).prod() - 1.0))
    episodes["candidate_same_window_return"] = candidate_returns
    episodes["same_window_improvement"] = episodes["candidate_same_window_return"] - episodes["depth"]
    episodes["candidate_improves"] = episodes["same_window_improvement"] > 1e-12
    return episodes


def official_baseline_check(baseline: pd.Series) -> float:
    config = _load_config_from_yaml(CONFIG)
    config["end"] = date(2026, 8, 14)
    config["transaction_cost_rate"] = FEE_MAIN
    official = run_official(config).daily_returns.loc[EVAL_START:END]
    overlap = baseline.index.intersection(official.index)
    difference = float((baseline.loc[overlap] - official.loc[overlap]).abs().max())
    if difference > 1e-12:
        raise RuntimeError(f"baseline does not match official runner: max diff={difference}")
    return difference


def main() -> None:
    prices, fields = load_panels()
    factor_pack, field_coverage = build_factors(prices, fields)
    targets = {
        name: targets_from_score(score)
        for name, score in factor_pack["scores"].items()
    }
    targets["MULTIFIELD_VETO80"] = veto_targets(
        factor_pack["qm20"], factor_pack["multifield_risk"], 0.80
    )
    targets["MULTIFIELD_VETO90"] = veto_targets(
        factor_pack["qm20"], factor_pack["multifield_risk"], 0.90
    )
    simulations = {
        name: simulate(target, prices["open"], prices["close"])
        for name, target in targets.items()
    }
    returns_1bp = {name: net(simulation, FEE_MAIN) for name, simulation in simulations.items()}
    returns_5bp = {name: net(simulation, FEE_STRESS) for name, simulation in simulations.items()}
    baseline = returns_1bp["BASE_QM20"]
    baseline_difference = official_baseline_check(baseline)

    periods = {
        "D": (EVAL_START, D_END),
        "V": (V_START, V_END),
        "T_pseudo_oos": (T_START, END),
        "FULL": (EVAL_START, END),
    }
    metric_rows = []
    for name, returns in returns_1bp.items():
        for period, (start, end) in periods.items():
            metric_rows.append(
                metric_row(
                    name,
                    period,
                    segment(returns, start, end),
                    segment(simulations[name]["turnover"], start, end),
                )
            )
    metrics = pd.DataFrame(metric_rows)

    baseline_metrics = metrics.loc[metrics["candidate"] == "BASE_QM20"].set_index("period")
    screen_rows = []
    dv_start, dv_end = EVAL_START, V_END
    baseline_dv = segment(baseline, dv_start, dv_end)
    baseline_dv_top10 = top10_summary(baseline_dv)
    for name, returns in returns_1bp.items():
        if name == "BASE_QM20":
            continue
        candidate_metrics = metrics.loc[metrics["candidate"] == name].set_index("period")
        d_delta = float(candidate_metrics.at["D", "sharpe"] - baseline_metrics.at["D", "sharpe"])
        v_delta = float(candidate_metrics.at["V", "sharpe"] - baseline_metrics.at["V", "sharpe"])
        candidate_dv_top10 = top10_summary(segment(returns, dv_start, dv_end))
        screen_rows.append(
            {
                "candidate": name,
                "sharpe_delta_D": d_delta,
                "sharpe_delta_V": v_delta,
                "min_sharpe_delta_DV": min(d_delta, v_delta),
                "dv_top10_mean_improvement": candidate_dv_top10["top10_mean_depth"] - baseline_dv_top10["top10_mean_depth"],
                "dv_worst_drawdown_improvement": candidate_dv_top10["top10_worst_depth"] - baseline_dv_top10["top10_worst_depth"],
                "annual_turnover_DV": float(
                    simulations[name]["turnover"].loc[dv_start:dv_end].sum()
                    / (len(segment(returns, dv_start, dv_end)) / 252.0)
                ),
                "eligible_DV": d_delta >= 0.0 and v_delta >= 0.0,
            }
        )
    screen = pd.DataFrame(screen_rows).sort_values(
        ["eligible_DV", "dv_top10_mean_improvement", "min_sharpe_delta_DV", "annual_turnover_DV", "candidate"],
        ascending=[False, False, False, True, True],
    )
    eligible = screen.loc[screen["eligible_DV"]]
    selected = str(eligible.iloc[0]["candidate"]) if len(eligible) else None

    comparison_rows = []
    gates = []
    rolling = pd.DataFrame()
    same_windows = pd.DataFrame()
    selected_top10 = pd.DataFrame()
    yearly = pd.DataFrame()
    if selected is not None:
        for fee, returns_map in ((1.0, returns_1bp), (5.0, returns_5bp)):
            for period, (start, end) in periods.items():
                baseline_segment = segment(returns_map["BASE_QM20"], start, end)
                candidate_segment = segment(returns_map[selected], start, end)
                comparison_rows.append(
                    {
                        "selected_candidate": selected,
                        "fee_bps_one_side": fee,
                        "period": period,
                        "baseline_annual_return": annual_return(baseline_segment),
                        "candidate_annual_return": annual_return(candidate_segment),
                        "baseline_sharpe": sharpe(baseline_segment),
                        "candidate_sharpe": sharpe(candidate_segment),
                        "sharpe_delta": sharpe(candidate_segment) - sharpe(baseline_segment),
                        "baseline_max_drawdown": max_drawdown(baseline_segment),
                        "candidate_max_drawdown": max_drawdown(candidate_segment),
                    }
                )
        comparison = pd.DataFrame(comparison_rows)
        one = comparison.loc[comparison["fee_bps_one_side"] == 1.0].set_index("period")
        five = comparison.loc[comparison["fee_bps_one_side"] == 5.0].set_index("period")
        rolling = rolling36(returns_1bp[selected], baseline)
        same_windows = same_window_comparison(baseline, returns_1bp[selected])
        baseline_top10 = drawdown_episodes(baseline).head(10).assign(strategy="BASE_QM20")
        candidate_top10 = drawdown_episodes(returns_1bp[selected]).head(10).assign(strategy=selected)
        selected_top10 = pd.concat([baseline_top10, candidate_top10], ignore_index=True)
        base_top10_summary = top10_summary(baseline)
        candidate_top10_summary = top10_summary(returns_1bp[selected])
        one_deltas = one.loc[["D", "V", "T_pseudo_oos", "FULL"], "sharpe_delta"]
        top10_mean_improvement = candidate_top10_summary["top10_mean_depth"] - base_top10_summary["top10_mean_depth"]
        worst_improvement = candidate_top10_summary["top10_worst_depth"] - base_top10_summary["top10_worst_depth"]
        same_window_wins = int(same_windows["candidate_improves"].sum())
        rolling_lead = float(rolling["candidate_leads"].mean())
        gates = pd.DataFrame(
            [
                {"gate": "D/V/T/FULL 1bp Sharpe nonnegative", "value": ";".join(f"{key}={value:.4f}" for key, value in one_deltas.items()), "passed": bool((one_deltas >= 0.0).all())},
                {"gate": "FULL 5bp Sharpe nonnegative", "value": float(five.at["FULL", "sharpe_delta"]), "passed": bool(five.at["FULL", "sharpe_delta"] >= 0.0)},
                {"gate": "FULL top10 mean improves >=1pp", "value": top10_mean_improvement, "passed": bool(top10_mean_improvement >= 0.01)},
                {"gate": "FULL worst drawdown improves >=0.5pp", "value": worst_improvement, "passed": bool(worst_improvement >= 0.005)},
                {"gate": "baseline top10 same-window wins >=7", "value": same_window_wins, "passed": bool(same_window_wins >= 7)},
                {"gate": "candidate top10 no worse than baseline worst -2pp", "value": candidate_top10_summary["top10_worst_depth"] - base_top10_summary["top10_worst_depth"], "passed": bool(candidate_top10_summary["top10_worst_depth"] >= base_top10_summary["top10_worst_depth"] - 0.02)},
                {"gate": "rolling36 Sharpe lead >=60%", "value": rolling_lead, "passed": bool(rolling_lead >= 0.60)},
                {"gate": "official baseline max daily diff <=1e-12", "value": baseline_difference, "passed": bool(baseline_difference <= 1e-12)},
            ]
        )
        yearly = pd.DataFrame(
            {
                "baseline": (1.0 + baseline).groupby(baseline.index.year).prod() - 1.0,
                "candidate": (1.0 + returns_1bp[selected]).groupby(baseline.index.year).prod() - 1.0,
            }
        )
        yearly["candidate_minus_baseline"] = yearly["candidate"] - yearly["baseline"]
        yearly.index.name = "year"
    else:
        comparison = pd.DataFrame(comparison_rows)
        gates = pd.DataFrame(
            [{"gate": "at least one D/V Sharpe nonnegative candidate", "value": 0, "passed": False}]
        )

    field_coverage.to_csv(HERE / f"{PREFIX}_factor_coverage.csv", index=False)
    metrics.to_csv(HERE / f"{PREFIX}_all_metrics_1bp.csv", index=False)
    screen.to_csv(HERE / f"{PREFIX}_dv_screen.csv", index=False)
    comparison.to_csv(HERE / f"{PREFIX}_selected_comparison.csv", index=False)
    gates.to_csv(HERE / f"{PREFIX}_gates.csv", index=False)
    rolling.to_csv(HERE / f"{PREFIX}_rolling36m.csv", index=False)
    same_windows.to_csv(HERE / f"{PREFIX}_same_window_top10.csv", index=False)
    selected_top10.to_csv(HERE / f"{PREFIX}_top10_episodes.csv", index=False)
    yearly.to_csv(HERE / f"{PREFIX}_yearly.csv")

    print(f"official baseline max daily difference: {baseline_difference:.3g}")
    print("\nD/V screen")
    print(screen.to_string(index=False))
    print(f"\nSelected without T: {selected}")
    if len(comparison):
        print("\nComparison")
        print(comparison.to_string(index=False))
    print("\nGates")
    print(gates.to_string(index=False))


if __name__ == "__main__":
    main()
