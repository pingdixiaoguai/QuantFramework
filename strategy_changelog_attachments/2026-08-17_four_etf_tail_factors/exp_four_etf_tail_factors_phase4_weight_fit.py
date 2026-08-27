"""Fit eight-field risk weights, then tune the conditional risk diversifier."""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
PHASE2_PATH = HERE / "exp_four_etf_tail_factors_phase2.py"
PHASE3_PATH = HERE / "exp_four_etf_tail_factors_phase3.py"
PREFIX = "2026-08-17_four_etf_tail_factors_phase4"
FEATURES = [
    "downside_lpm20",
    "cvar20",
    "range20",
    "gap_tail20",
    "amihud20",
    "amount_shock20",
    "share_flow20",
    "premium_crowding",
]
LAMBDAS = (0.0, 0.01, 0.1, 1.0)
HORIZON = 20
TAIL_COUNT = 4


def load_phase2():
    spec = importlib.util.spec_from_file_location("four_etf_phase2_for_weight_fit", PHASE2_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {PHASE2_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_phase3():
    spec = importlib.util.spec_from_file_location("four_etf_phase3_for_weight_fit", PHASE3_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {PHASE3_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def forward_tail_label(close: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    calendar = close.index
    future_drawdown = pd.DataFrame(np.nan, index=calendar, columns=close.columns)
    future_cvar = pd.DataFrame(np.nan, index=calendar, columns=close.columns)
    label_end = pd.Series(pd.NaT, index=calendar, dtype="datetime64[ns]")
    returns = close.pct_change(fill_method=None)
    for position in range(len(calendar) - HORIZON):
        future_slice = slice(position + 1, position + HORIZON + 1)
        current = close.iloc[position]
        path = close.iloc[future_slice].div(current, axis=1) - 1.0
        future_drawdown.iloc[position] = (-path.min(axis=0)).clip(lower=0.0)
        future_returns = returns.iloc[future_slice]
        for code in close.columns:
            values = future_returns[code].dropna().to_numpy(dtype=float)
            if len(values) == HORIZON:
                future_cvar.at[calendar[position], code] = -float(np.sort(values)[:TAIL_COUNT].mean())
        label_end.iloc[position] = calendar[position + HORIZON]
    drawdown_rank = future_drawdown.rank(axis=1, pct=True, method="average")
    cvar_rank = future_cvar.rank(axis=1, pct=True, method="average")
    label = (drawdown_rank + cvar_rank) / 2.0
    return label, label_end


def build_sample(
    p1,
    risk_ranks: dict[str, pd.DataFrame],
    label: pd.DataFrame,
    label_end: pd.Series,
) -> pd.DataFrame:
    eligible_calendar = label.index[(label.index >= p1.EVAL_START) & (label.index <= p1.END)]
    sampled_dates = eligible_calendar[:: p1.REBALANCE_DAYS]
    rows: list[dict[str, object]] = []
    for timestamp in sampled_dates:
        for code in p1.CORE:
            row: dict[str, object] = {
                "date": timestamp,
                "asset": code,
                "label_end": label_end.at[timestamp],
                "future_tail_risk": label.at[timestamp, code],
            }
            row.update({name: risk_ranks[name].at[timestamp, code] for name in FEATURES})
            rows.append(row)
    sample = pd.DataFrame(rows)
    numeric = FEATURES + ["future_tail_risk"]
    sample[numeric] = sample[numeric].apply(pd.to_numeric, errors="coerce")
    sample = sample.dropna(subset=numeric + ["label_end"]).reset_index(drop=True)
    return sample


def fit_weights(sample: pd.DataFrame, regularization: float) -> tuple[np.ndarray, dict[str, object]]:
    x = sample[FEATURES].to_numpy(dtype=float)
    y = sample["future_tail_risk"].to_numpy(dtype=float)
    equal = np.repeat(1.0 / len(FEATURES), len(FEATURES))

    def objective(weights: np.ndarray) -> float:
        residual = x @ weights - y
        return float(np.mean(residual**2) + regularization * np.sum((weights - equal) ** 2))

    result = minimize(
        objective,
        equal,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * len(FEATURES),
        constraints=[{"type": "eq", "fun": lambda weights: float(weights.sum() - 1.0)}],
        options={"ftol": 1e-12, "maxiter": 2000},
    )
    if not result.success:
        raise RuntimeError(f"weight fit failed: {result.message}")
    weights = np.clip(result.x, 0.0, 1.0)
    weights /= weights.sum()
    diagnostics = {
        "success": bool(result.success),
        "iterations": int(result.nit),
        "objective": float(result.fun),
        "sample_rows": int(len(sample)),
    }
    return weights, diagnostics


def prediction_metrics(sample: pd.DataFrame, weights: np.ndarray) -> dict[str, float]:
    frame = sample[["date", "asset", "future_tail_risk"]].copy()
    frame["prediction"] = sample[FEATURES].to_numpy(dtype=float) @ weights
    daily = []
    hit = []
    for _, one in frame.groupby("date"):
        if len(one) < 3:
            continue
        if one["prediction"].nunique() > 1 and one["future_tail_risk"].nunique() > 1:
            pearson = one["prediction"].corr(one["future_tail_risk"], method="pearson")
            spearman = one["prediction"].corr(one["future_tail_risk"], method="spearman")
            if np.isfinite(pearson) and np.isfinite(spearman):
                daily.append((float(pearson), float(spearman)))
        predicted_max = set(one.loc[one["prediction"] >= one["prediction"].max() - 1e-12, "asset"])
        realized_max = set(
            one.loc[one["future_tail_risk"] >= one["future_tail_risk"].max() - 1e-12, "asset"]
        )
        hit.append(bool(predicted_max & realized_max))
    prediction = frame["prediction"].to_numpy(dtype=float)
    realized = frame["future_tail_risk"].to_numpy(dtype=float)
    return {
        "rows": int(len(frame)),
        "dates": int(frame["date"].nunique()),
        "mse": float(np.mean((prediction - realized) ** 2)),
        "mean_daily_pearson_ic": float(np.mean([item[0] for item in daily])) if daily else np.nan,
        "mean_daily_spearman_ic": float(np.mean([item[1] for item in daily])) if daily else np.nan,
        "highest_risk_hit_rate": float(np.mean(hit)) if hit else np.nan,
    }


def development_cv(sample: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    folds = (
        ("2016", pd.Timestamp("2015-12-31"), pd.Timestamp("2016-01-01"), pd.Timestamp("2016-12-31")),
        ("2017", pd.Timestamp("2016-12-30"), pd.Timestamp("2017-01-01"), pd.Timestamp("2017-12-31")),
        ("2018", pd.Timestamp("2017-12-29"), pd.Timestamp("2018-01-01"), pd.Timestamp("2018-12-31")),
    )
    rows = []
    for regularization in LAMBDAS:
        for fold, train_end, validation_start, validation_end in folds:
            train = sample.loc[sample["label_end"] <= train_end]
            validation = sample.loc[
                sample["date"].between(validation_start, validation_end)
                & (sample["label_end"] <= validation_end)
            ]
            weights, fit = fit_weights(train, regularization)
            metrics = prediction_metrics(validation, weights)
            row = {
                "lambda": regularization,
                "fold": fold,
                "train_end": train_end,
                "validation_start": validation_start,
                "validation_end": validation_end,
                "train_rows": len(train),
                **metrics,
                **{f"weight_{name}": weight for name, weight in zip(FEATURES, weights)},
                "fit_objective": fit["objective"],
            }
            rows.append(row)
    cv = pd.DataFrame(rows)
    summary = cv.groupby("lambda", as_index=False).agg(
        cv_mse=("mse", "mean"),
        cv_spearman=("mean_daily_spearman_ic", "mean"),
        cv_hit_rate=("highest_risk_hit_rate", "mean"),
    )
    summary = summary.sort_values(["cv_mse", "cv_spearman", "lambda"], ascending=[True, False, False])
    selected_lambda = float(summary.iloc[0]["lambda"])
    cv = cv.merge(summary, on="lambda", how="left")
    cv["selected_lambda"] = cv["lambda"].eq(selected_lambda)
    return cv, selected_lambda


def weighted_risk(risk_ranks: dict[str, pd.DataFrame], weights: np.ndarray) -> pd.DataFrame:
    result = risk_ranks[FEATURES[0]] * weights[0]
    for name, weight in zip(FEATURES[1:], weights[1:]):
        result = result + risk_ranks[name] * weight
    return result


def base_target_weights(p1, qm20: pd.DataFrame) -> pd.DataFrame:
    result = pd.DataFrame(0.0, index=qm20.index, columns=p1.CORE)
    for timestamp, code in p1.targets_from_score(qm20).items():
        if isinstance(code, str):
            result.at[timestamp, code] = 1.0
    return result


def make_predictive_table(p1, sample: pd.DataFrame, fitted: np.ndarray) -> pd.DataFrame:
    equal = np.repeat(1.0 / len(FEATURES), len(FEATURES))
    periods = {
        "D_fit": (p1.EVAL_START, p1.D_END),
        "V_unseen_by_fit": (p1.V_START, p1.V_END),
        "T_unseen_by_fit": (p1.T_START, p1.END),
        "FULL": (p1.EVAL_START, p1.END),
    }
    rows = []
    for model, weights in (("equal_weight", equal), ("fitted_weight", fitted)):
        for period, (start, end) in periods.items():
            subset = sample.loc[sample["date"].between(start, end) & (sample["label_end"] <= end)]
            rows.append({"model": model, "period": period, **prediction_metrics(subset, weights)})
    table = pd.DataFrame(rows)
    equal_rows = table.loc[table["model"] == "equal_weight"].set_index("period")
    for index, row in table.iterrows():
        base = equal_rows.loc[row["period"]]
        table.at[index, "mse_delta_vs_equal"] = row["mse"] - base["mse"]
        table.at[index, "spearman_delta_vs_equal"] = (
            row["mean_daily_spearman_ic"] - base["mean_daily_spearman_ic"]
        )
        table.at[index, "hit_rate_delta_vs_equal"] = row["highest_risk_hit_rate"] - base[
            "highest_risk_hit_rate"
        ]
    return table


def annual_turnover(turnover: pd.Series, returns: pd.Series) -> float:
    years = len(returns) / 252.0
    return float(turnover.sum() / years) if years else 0.0


def main() -> None:
    p2 = load_phase2()
    p3 = load_phase3()
    p1 = p2.load_phase1()
    prices, fields = p1.load_panels()
    factors, _ = p1.build_factors(prices, fields)
    qm20 = factors["qm20"]
    risk_ranks = factors["risk_ranks"]
    label, label_end = forward_tail_label(prices["close"][p1.CORE])
    sample = build_sample(p1, risk_ranks, label, label_end)

    cv, selected_lambda = development_cv(sample)
    development = sample.loc[sample["label_end"] <= p1.D_END]
    fitted_weights, fit_diagnostics = fit_weights(development, selected_lambda)
    equal_weights = np.repeat(1.0 / len(FEATURES), len(FEATURES))
    weights_table = pd.DataFrame(
        {
            "field": FEATURES,
            "equal_weight": equal_weights,
            "fitted_weight": fitted_weights,
            "difference": fitted_weights - equal_weights,
        }
    ).sort_values("fitted_weight", ascending=False)
    weights_table["selected_lambda"] = selected_lambda
    weights_table["fit_rows"] = fit_diagnostics["sample_rows"]
    predictive = make_predictive_table(p1, sample, fitted_weights)

    fitted_risk = weighted_risk(risk_ranks, fitted_weights)
    equal_risk = weighted_risk(risk_ranks, equal_weights)
    target_map = {"BASE_QM20": base_target_weights(p1, qm20)}
    trigger_frames = []
    parameter_rows = []
    for threshold in (0.70, 0.75, 0.80):
        for budget in (0.40, 0.50, 0.60):
            name = f"FIT_T{int(threshold * 100)}_W{int(budget * 100)}"
            target, triggers = p2.build_targets(p1, qm20, fitted_risk, threshold, "SAFE", budget)
            target_map[name] = target
            triggers.insert(0, "candidate", name)
            trigger_frames.append(triggers)
            parameter_rows.append({"candidate": name, "threshold": threshold, "budget": budget})

    # Governance comparators are not eligible for parameter selection.
    equal_prior, equal_prior_triggers = p2.build_targets(p1, qm20, equal_risk, 0.75, "SAFE", 0.50)
    target_map["EQUAL_T75_W50"] = equal_prior
    equal_prior_triggers.insert(0, "candidate", "EQUAL_T75_W50")
    trigger_frames.append(equal_prior_triggers)
    triggers = pd.concat(trigger_frames, ignore_index=True)

    simulations = {
        name: p2.simulate_weighted(p1, target, prices["open"], prices["close"])
        for name, target in target_map.items()
    }
    returns_1bp = {name: p1.net(simulation, p1.FEE_MAIN) for name, simulation in simulations.items()}
    returns_5bp = {name: p1.net(simulation, p1.FEE_STRESS) for name, simulation in simulations.items()}
    baseline = returns_1bp["BASE_QM20"]
    baseline_difference = p1.official_baseline_check(baseline)

    periods = {
        "D": (p1.EVAL_START, p1.D_END),
        "V": (p1.V_START, p1.V_END),
        "T_pseudo_oos": (p1.T_START, p1.END),
        "FULL": (p1.EVAL_START, p1.END),
    }
    metrics_rows = []
    for name, returns in returns_1bp.items():
        for period, (start, end) in periods.items():
            metrics_rows.append(
                p1.metric_row(
                    name,
                    period,
                    p1.segment(returns, start, end),
                    p1.segment(simulations[name]["turnover"], start, end),
                )
            )
    metrics = pd.DataFrame(metrics_rows)
    base_v = p1.segment(baseline, p1.V_START, p1.V_END)
    base_v_metrics = metrics.loc[
        (metrics["candidate"] == "BASE_QM20") & (metrics["period"] == "V")
    ].iloc[0]
    screen_rows = []
    for params in parameter_rows:
        name = params["candidate"]
        candidate_v = p1.segment(returns_1bp[name], p1.V_START, p1.V_END)
        candidate_v_metrics = metrics.loc[
            (metrics["candidate"] == name) & (metrics["period"] == "V")
        ].iloc[0]
        screen_rows.append(
            {
                **params,
                "validation_sharpe_delta": candidate_v_metrics["sharpe"] - base_v_metrics["sharpe"],
                "validation_top10_mean_improvement": candidate_v_metrics["top10_mean_depth"]
                - base_v_metrics["top10_mean_depth"],
                "validation_maxdd_improvement": candidate_v_metrics["max_drawdown"]
                - base_v_metrics["max_drawdown"],
                "validation_annual_turnover": annual_turnover(
                    p1.segment(simulations[name]["turnover"], p1.V_START, p1.V_END), candidate_v
                ),
                "validation_trigger_days": int(
                    triggers.loc[
                        (triggers["candidate"] == name)
                        & triggers["date"].between(p1.V_START, p1.V_END),
                        "date",
                    ].nunique()
                ),
                "eligible": bool(candidate_v_metrics["sharpe"] >= base_v_metrics["sharpe"]),
            }
        )
    screen = pd.DataFrame(screen_rows).sort_values(
        [
            "eligible",
            "validation_top10_mean_improvement",
            "validation_sharpe_delta",
            "validation_annual_turnover",
            "candidate",
        ],
        ascending=[False, False, False, True, True],
    )
    eligible = screen.loc[screen["eligible"]]
    selected = str(eligible.iloc[0]["candidate"]) if len(eligible) else None
    if selected is None:
        raise RuntimeError("no validation-period candidate has Sharpe >= baseline")

    selected_params = screen.loc[screen["candidate"] == selected].iloc[0]
    equal_same_name = "EQUAL_SAME_PARAMETERS"
    equal_same, equal_same_triggers = p2.build_targets(
        p1,
        qm20,
        equal_risk,
        float(selected_params["threshold"]),
        "SAFE",
        float(selected_params["budget"]),
    )
    target_map[equal_same_name] = equal_same
    equal_same_triggers.insert(0, "candidate", equal_same_name)
    triggers = pd.concat([triggers, equal_same_triggers], ignore_index=True)
    simulations[equal_same_name] = p2.simulate_weighted(p1, equal_same, prices["open"], prices["close"])
    returns_1bp[equal_same_name] = p1.net(simulations[equal_same_name], p1.FEE_MAIN)
    returns_5bp[equal_same_name] = p1.net(simulations[equal_same_name], p1.FEE_STRESS)

    comparison_rows = []
    for strategy in ("BASE_QM20", "EQUAL_T75_W50", equal_same_name, selected):
        for fee, returns_map in ((1.0, returns_1bp), (5.0, returns_5bp)):
            for period, (start, end) in periods.items():
                one = p1.segment(returns_map[strategy], start, end)
                comparison_rows.append(
                    {
                        "strategy": strategy,
                        "selected": strategy == selected,
                        "fee_bps_one_side": fee,
                        "period": period,
                        "annual_return": p1.annual_return(one),
                        "sharpe": p1.sharpe(one),
                        "max_drawdown": p1.max_drawdown(one),
                        **p1.top10_summary(one),
                    }
                )
    comparison = pd.DataFrame(comparison_rows)
    one = comparison.loc[comparison["fee_bps_one_side"] == 1.0].set_index(["strategy", "period"])
    five = comparison.loc[comparison["fee_bps_one_side"] == 5.0].set_index(["strategy", "period"])
    sharpe_deltas = pd.Series(
        {
            period: one.at[(selected, period), "sharpe"] - one.at[("BASE_QM20", period), "sharpe"]
            for period in periods
        }
    )
    selected_full = returns_1bp[selected]
    base_summary = p1.top10_summary(baseline)
    selected_summary = p1.top10_summary(selected_full)
    same_windows = p1.same_window_comparison(baseline, selected_full)
    rolling = p1.rolling36(selected_full, baseline)
    mean_improvement = selected_summary["top10_mean_depth"] - base_summary["top10_mean_depth"]
    maxdd_improvement = p1.max_drawdown(selected_full) - p1.max_drawdown(baseline)
    same_window_wins = int(same_windows["candidate_improves"].sum())
    rolling_lead = float(rolling["candidate_leads"].mean())
    target = target_map[selected]
    target_sum = target.sum(axis=1)
    active = target_sum > 0
    invalid_assets = sorted(set(target.columns) - set(p1.CORE))
    target_valid = bool(
        not invalid_assets
        and (target.loc[active] >= -1e-12).all().all()
        and np.allclose(target_sum.loc[active], 1.0, atol=1e-12)
    )

    fitted_pred = predictive.loc[predictive["model"] == "fitted_weight"].set_index("period")
    predictive_gate = bool(
        fitted_pred.at["V_unseen_by_fit", "mse_delta_vs_equal"] <= 0.0
        and fitted_pred.at["T_unseen_by_fit", "mse_delta_vs_equal"] <= 0.0
    )
    gates = pd.DataFrame(
        [
            {
                "gate": "V/T predictive MSE <= equal weight",
                "value": f"V={fitted_pred.at['V_unseen_by_fit', 'mse_delta_vs_equal']:.6f};T={fitted_pred.at['T_unseen_by_fit', 'mse_delta_vs_equal']:.6f}",
                "passed": predictive_gate,
            },
            {
                "gate": "D/V/T/FULL 1bp Sharpe nonnegative",
                "value": ";".join(f"{key}={value:.4f}" for key, value in sharpe_deltas.items()),
                "passed": bool((sharpe_deltas >= 0.0).all()),
            },
            {
                "gate": "FULL 5bp Sharpe nonnegative",
                "value": float(five.at[(selected, "FULL"), "sharpe"] - five.at[("BASE_QM20", "FULL"), "sharpe"]),
                "passed": bool(five.at[(selected, "FULL"), "sharpe"] >= five.at[("BASE_QM20", "FULL"), "sharpe"]),
            },
            {
                "gate": "FULL top10 mean improves >=1pp",
                "value": mean_improvement,
                "passed": bool(mean_improvement >= 0.01),
            },
            {
                "gate": "FULL max drawdown improves >=0.5pp",
                "value": maxdd_improvement,
                "passed": bool(maxdd_improvement >= 0.005),
            },
            {
                "gate": "baseline top10 same-window wins >=7",
                "value": same_window_wins,
                "passed": bool(same_window_wins >= 7),
            },
            {
                "gate": "rolling36 Sharpe lead >=60%",
                "value": rolling_lead,
                "passed": bool(rolling_lead >= 0.60),
            },
            {
                "gate": "official baseline max daily diff <=1e-12",
                "value": baseline_difference,
                "passed": bool(baseline_difference <= 1e-12),
            },
            {
                "gate": "target invariants",
                "value": f"invalid_assets={invalid_assets};active_rows={int(active.sum())}",
                "passed": target_valid,
            },
        ]
    )

    audit_rows = []
    for name, candidate_1 in returns_1bp.items():
        if name == "BASE_QM20":
            continue
        candidate_5 = returns_5bp[name]
        summary = p1.top10_summary(candidate_1)
        candidate_same_windows = p1.same_window_comparison(baseline, candidate_1)
        candidate_rolling = p1.rolling36(candidate_1, baseline)
        row = {
            "candidate": name,
            "selected_by_frozen_validation_rule": name == selected,
            "full_sharpe": p1.sharpe(candidate_1),
            "full_sharpe_delta": p1.sharpe(candidate_1) - p1.sharpe(baseline),
            "full_5bp_sharpe_delta": p1.sharpe(candidate_5) - p1.sharpe(returns_5bp["BASE_QM20"]),
            "full_annual_return": p1.annual_return(candidate_1),
            "full_annual_return_delta": p1.annual_return(candidate_1) - p1.annual_return(baseline),
            "full_max_drawdown": p1.max_drawdown(candidate_1),
            "full_maxdd_improvement": p1.max_drawdown(candidate_1) - p1.max_drawdown(baseline),
            "full_top10_mean_depth": summary["top10_mean_depth"],
            "full_top10_mean_improvement": summary["top10_mean_depth"] - base_summary["top10_mean_depth"],
            "same_window_wins": int(candidate_same_windows["candidate_improves"].sum()),
            "rolling36_sharpe_lead": float(candidate_rolling["candidate_leads"].mean()),
        }
        for period, (start, end) in periods.items():
            row[f"sharpe_delta_{period}"] = p1.sharpe(candidate_1.loc[start:end]) - p1.sharpe(
                baseline.loc[start:end]
            )
        row["passes_original_final_gates"] = bool(
            min(row[f"sharpe_delta_{period}"] for period in periods) >= 0.0
            and row["full_5bp_sharpe_delta"] >= 0.0
            and row["full_top10_mean_improvement"] >= 0.01
            and row["full_maxdd_improvement"] >= 0.005
            and row["same_window_wins"] >= 7
            and row["rolling36_sharpe_lead"] >= 0.60
        )
        audit_rows.append(row)
    candidate_audit = pd.DataFrame(audit_rows).sort_values(
        ["passes_original_final_gates", "full_top10_mean_improvement", "full_sharpe_delta"],
        ascending=[False, False, False],
    )
    bootstrap = p3.paired_block_bootstrap(p1, baseline, selected_full)

    top10 = pd.concat(
        [
            p1.drawdown_episodes(baseline).head(10).assign(strategy="BASE_QM20"),
            p1.drawdown_episodes(selected_full).head(10).assign(strategy=selected),
        ],
        ignore_index=True,
    )
    yearly = pd.DataFrame(
        {
            "baseline": (1.0 + baseline).groupby(baseline.index.year).prod() - 1.0,
            "selected": (1.0 + selected_full).groupby(selected_full.index.year).prod() - 1.0,
        }
    )
    yearly["selected_minus_baseline"] = yearly["selected"] - yearly["baseline"]
    yearly.index.name = "year"
    trigger_summary = triggers.groupby("candidate", as_index=False).agg(
        first_trigger=("date", "min"), last_trigger=("date", "max"), trigger_days=("date", "nunique")
    )
    fit_summary = pd.DataFrame(
        [
            {
                "selected_lambda": selected_lambda,
                "fit_rows": fit_diagnostics["sample_rows"],
                "fit_objective": fit_diagnostics["objective"],
                "selected_strategy": selected,
                "all_gates_passed": bool(gates["passed"].all()),
            }
        ]
    )

    sample.to_parquet(HERE / f"{PREFIX}_model_sample.parquet", index=False)
    cv.to_csv(HERE / f"{PREFIX}_time_series_cv.csv", index=False)
    weights_table.to_csv(HERE / f"{PREFIX}_weights.csv", index=False)
    predictive.to_csv(HERE / f"{PREFIX}_predictive_metrics.csv", index=False)
    screen.to_csv(HERE / f"{PREFIX}_validation_strategy_screen.csv", index=False)
    comparison.to_csv(HERE / f"{PREFIX}_strategy_comparison.csv", index=False)
    metrics.to_csv(HERE / f"{PREFIX}_all_strategy_metrics_1bp.csv", index=False)
    candidate_audit.to_csv(HERE / f"{PREFIX}_candidate_audit.csv", index=False)
    bootstrap.to_csv(HERE / f"{PREFIX}_selected_bootstrap.csv", index=False)
    gates.to_csv(HERE / f"{PREFIX}_gates.csv", index=False)
    rolling.to_csv(HERE / f"{PREFIX}_rolling36m.csv", index=False)
    same_windows.to_csv(HERE / f"{PREFIX}_same_window_top10.csv", index=False)
    top10.to_csv(HERE / f"{PREFIX}_top10_episodes.csv", index=False)
    yearly.to_csv(HERE / f"{PREFIX}_yearly.csv")
    trigger_summary.to_csv(HERE / f"{PREFIX}_trigger_summary.csv", index=False)
    fit_summary.to_csv(HERE / f"{PREFIX}_fit_summary.csv", index=False)

    print("Selected lambda:", selected_lambda)
    print("\nFitted weights")
    print(weights_table.to_string(index=False))
    print("\nPredictive metrics")
    print(predictive.to_string(index=False))
    print("\nValidation-only strategy screen")
    print(screen.to_string(index=False))
    print("\nSelected strategy:", selected)
    print("\nStrategy comparison")
    print(comparison.to_string(index=False))
    print("\nGates")
    print(gates.to_string(index=False))
    print("\nAll-candidate audit (not a new selection step)")
    print(candidate_audit.to_string(index=False))
    print("\nSelected-strategy paired block bootstrap")
    print(bootstrap.to_string(index=False))


if __name__ == "__main__":
    main()
