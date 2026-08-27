"""Causal yearly walk-forward meta-model for conditional risk intervention."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PHASE5_PATH = HERE / "exp_four_etf_tail_factors_phase5_multi_mechanism.py"
PREFIX = "2026-08-17_four_etf_tail_factors_phase9"
ALPHAS = (0.1, 1.0, 10.0, 100.0)
FEATURE_COLUMNS = [
    "winner_risk",
    "risk_spread",
    "confidence",
    "winner_mom20",
    "alt_mom20",
    "momentum_gap",
    "positive_breadth",
    "winner_drawdown60",
    "winner_mom60",
    "winner_vol20",
] + [f"winner_{name}" for name in ("downside_lpm20", "cvar20", "range20", "gap_tail20", "amihud20", "amount_shock20", "share_flow20", "premium_crowding")] + [
    f"spread_{name}" for name in ("downside_lpm20", "cvar20", "range20", "gap_tail20", "amihud20", "amount_shock20", "share_flow20", "premium_crowding")
] + ["winner_510300", "winner_159915", "winner_513100", "winner_518880"]


def load_phase5():
    spec = importlib.util.spec_from_file_location("phase5_for_phase9", PHASE5_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {PHASE5_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_event_frame(p1, prices, factors, fitted_risk, mode: str) -> pd.DataFrame:
    close = prices["close"][p1.CORE]
    open_prices = prices["open"][p1.CORE]
    qm20 = factors["qm20"]
    qm_rank = qm20.rank(axis=1, pct=True, method="average")
    mom20 = close.pct_change(20, fill_method=None)
    mom60 = close.pct_change(60, fill_method=None)
    drawdown60 = close / close.rolling(60).max() - 1.0
    vol20 = close.pct_change(fill_method=None).rolling(20).std()
    calendar = close.index
    sampled_dates = set(calendar[(calendar >= p1.EVAL_START) & (calendar <= p1.END)][:: p1.REBALANCE_DAYS])
    rows = []
    for position, timestamp in enumerate(calendar):
        qm_row = qm20.loc[timestamp, p1.CORE].dropna()
        risk_row = fitted_risk.loc[timestamp, p1.CORE].dropna()
        if len(qm_row) < 2 or len(risk_row) < len(p1.CORE):
            continue
        winner = str(qm_row.idxmax())
        positive = [
            code
            for code in p1.CORE
            if code != winner and np.isfinite(mom20.at[timestamp, code]) and mom20.at[timestamp, code] > 0.0
        ]
        if not positive:
            continue
        if mode == "SAFE":
            alternative = str(risk_row[positive].idxmin())
        elif mode == "RA":
            alternative = str((qm_rank.loc[timestamp, positive] - 0.75 * risk_row[positive]).idxmax())
        else:
            raise ValueError(mode)
        values = qm_row.sort_values(ascending=False)
        scale = float(qm_row.std(ddof=0))
        confidence = float((values.iloc[0] - values.iloc[1]) / scale) if scale > 0.0 else 0.0
        row: dict[str, object] = {
            "date": timestamp,
            "mode": mode,
            "winner": winner,
            "alternative": alternative,
            "sampled": timestamp in sampled_dates,
            "winner_is_riskiest": bool(float(risk_row[winner]) >= float(risk_row.max()) - 1e-12),
            "winner_risk": float(risk_row[winner]),
            "risk_spread": float(risk_row[winner] - risk_row[alternative]),
            "confidence": confidence,
            "winner_mom20": mom20.at[timestamp, winner],
            "alt_mom20": mom20.at[timestamp, alternative],
            "momentum_gap": mom20.at[timestamp, winner] - mom20.at[timestamp, alternative],
            "positive_breadth": float((mom20.loc[timestamp, p1.CORE] > 0.0).mean()),
            "winner_drawdown60": drawdown60.at[timestamp, winner],
            "winner_mom60": mom60.at[timestamp, winner],
            "winner_vol20": vol20.at[timestamp, winner],
        }
        for name, frame in factors["risk_ranks"].items():
            row[f"winner_{name}"] = frame.at[timestamp, winner]
            row[f"spread_{name}"] = frame.at[timestamp, winner] - frame.at[timestamp, alternative]
        for code, suffix in zip(p1.CORE, ("510300", "159915", "513100", "518880")):
            row[f"winner_{suffix}"] = float(winner == code)
        if position + 6 < len(calendar):
            entry_date = calendar[position + 1]
            exit_date = calendar[position + 6]
            winner_entry = open_prices.at[entry_date, winner]
            winner_exit = open_prices.at[exit_date, winner]
            alt_entry = open_prices.at[entry_date, alternative]
            alt_exit = open_prices.at[exit_date, alternative]
            if all(np.isfinite(value) and value > 0.0 for value in (winner_entry, winner_exit, alt_entry, alt_exit)):
                row["label_end"] = exit_date
                row["alternative_excess"] = alt_exit / alt_entry - winner_exit / winner_entry
        rows.append(row)
    frame = pd.DataFrame(rows)
    numeric = FEATURE_COLUMNS + ["alternative_excess"]
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="coerce")
    return frame


def fit_ridge(train: pd.DataFrame, alpha: float):
    x = train[FEATURE_COLUMNS].to_numpy(dtype=float)
    y = train["alternative_excess"].to_numpy(dtype=float)
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-12] = 1.0
    z = (x - mean) / scale
    lower, upper = np.quantile(y, (0.025, 0.975))
    y_fit = np.clip(y, lower, upper)
    design = np.column_stack([np.ones(len(z)), z])
    penalty = np.eye(design.shape[1]) * alpha
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ y_fit)
    return mean, scale, coefficients


def predict(frame: pd.DataFrame, model) -> np.ndarray:
    mean, scale, coefficients = model
    x = frame[FEATURE_COLUMNS].to_numpy(dtype=float)
    z = (x - mean) / scale
    return np.column_stack([np.ones(len(z)), z]) @ coefficients


def walkforward_predictions(p1, events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    complete = events.dropna(subset=FEATURE_COLUMNS).copy()
    sampled = complete.loc[complete["sampled"] & complete["alternative_excess"].notna()].copy()
    predictions = []
    diagnostics = []
    for year in range(2017, p1.END.year + 1):
        validation_year = year - 1
        validation = sampled.loc[
            sampled["date"].dt.year.eq(validation_year)
            & sampled["label_end"].dt.year.eq(validation_year)
        ]
        inner_train = sampled.loc[sampled["label_end"] < pd.Timestamp(f"{validation_year}-01-01")]
        final_train = sampled.loc[sampled["label_end"] < pd.Timestamp(f"{year}-01-01")]
        forecast = complete.loc[complete["date"].dt.year.eq(year)].copy()
        if len(inner_train) < 80 or len(validation) < 20 or len(final_train) < 100 or forecast.empty:
            continue
        alpha_rows = []
        for alpha in ALPHAS:
            model = fit_ridge(inner_train, alpha)
            values = predict(validation, model)
            mse = float(np.mean((values - validation["alternative_excess"].to_numpy(dtype=float)) ** 2))
            alpha_rows.append((alpha, mse))
        selected_alpha, validation_mse = min(alpha_rows, key=lambda item: (item[1], item[0]))
        final_model = fit_ridge(final_train, selected_alpha)
        forecast["predicted_excess"] = predict(forecast, final_model)
        predictions.append(forecast[["date", "mode", "winner", "alternative", "winner_risk", "winner_is_riskiest", "predicted_excess"]])
        realized = forecast.dropna(subset=["alternative_excess"])
        realized_prediction = predict(realized, final_model)
        diagnostics.append(
            {
                "mode": str(events["mode"].iloc[0]),
                "forecast_year": year,
                "selected_alpha": selected_alpha,
                "inner_train_rows": len(inner_train),
                "validation_rows": len(validation),
                "final_train_rows": len(final_train),
                "validation_mse": validation_mse,
                "forecast_label_mse": float(np.mean((realized_prediction - realized["alternative_excess"].to_numpy(dtype=float)) ** 2)) if len(realized) else np.nan,
                "forecast_sign_hit_rate": float((np.sign(realized_prediction) == np.sign(realized["alternative_excess"].to_numpy(dtype=float))).mean()) if len(realized) else np.nan,
                "forecast_prediction_mean": float(realized_prediction.mean()) if len(realized) else np.nan,
                "forecast_realized_mean": float(realized["alternative_excess"].mean()) if len(realized) else np.nan,
            }
        )
    prediction_frame = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
    return prediction_frame, pd.DataFrame(diagnostics)


def policy_target(p1, p5, qm20, predictions, threshold, hurdle, budget):
    target = p5.base_target(p1, qm20)
    triggers = []
    for row in predictions.itertuples():
        if not row.winner_is_riskiest or row.winner_risk < threshold or row.predicted_excess <= hurdle:
            continue
        timestamp = row.date
        target.loc[timestamp] = 0.0
        target.at[timestamp, row.winner] = 1.0 - budget
        target.at[timestamp, row.alternative] = budget
        triggers.append(
            {
                "date": timestamp,
                "winner": row.winner,
                "alternative": row.alternative,
                "winner_risk": row.winner_risk,
                "predicted_excess": row.predicted_excess,
                "budget": budget,
            }
        )
    return target, pd.DataFrame(triggers)


def main() -> None:
    p5 = load_phase5()
    p2 = p5.load_module("phase2_for_phase9", p5.PHASE2_PATH)
    p1 = p2.load_phase1()
    prices, fields = p1.load_panels()
    factors, _ = p1.build_factors(prices, fields)
    qm20 = factors["qm20"]
    risk = p5.weighted_risk(factors["risk_ranks"], p5.load_fitted_weights())
    prediction_map = {}
    diagnostic_frames = []
    for mode in ("SAFE", "RA"):
        events = build_event_frame(p1, prices, factors, risk, mode)
        predictions, diagnostics = walkforward_predictions(p1, events)
        prediction_map[mode] = predictions
        diagnostic_frames.append(diagnostics)
    diagnostics = pd.concat(diagnostic_frames, ignore_index=True)
    base_target = p5.base_target(p1, qm20)
    base_sim = p2.simulate_weighted(p1, base_target, prices["open"], prices["close"])
    baseline_1 = p1.net(base_sim, p1.FEE_MAIN)
    baseline_5 = p1.net(base_sim, p1.FEE_STRESS)
    periods = {"D": (p1.EVAL_START, p1.D_END), "V": (p1.V_START, p1.V_END), "T": (p1.T_START, p1.END), "FULL": (p1.EVAL_START, p1.END)}
    rows = []
    artifacts = {}
    candidate_id = 0
    for mode in ("SAFE", "RA"):
        for threshold in (0.70, 0.80):
            for hurdle in (0.0, 0.001, 0.002):
                for budget in (0.15, 0.25, 0.35):
                    candidate_id += 1
                    name = f"P9_{candidate_id:02d}"
                    target, triggers = policy_target(p1, p5, qm20, prediction_map[mode], threshold, hurdle, budget)
                    simulation = p2.simulate_weighted(p1, target, prices["open"], prices["close"])
                    returns_1 = p1.net(simulation, p1.FEE_MAIN)
                    returns_5 = p1.net(simulation, p1.FEE_STRESS)
                    artifacts[name] = (target, triggers, simulation, returns_1, returns_5)
                    row = {"candidate": name, "mode": mode, "threshold": threshold, "hurdle": hurdle, "budget": budget, "trigger_days": int(triggers["date"].nunique() if len(triggers) else 0)}
                    improvements = []
                    for period in ("D", "V"):
                        start, end = periods[period]
                        candidate_segment = returns_1.loc[start:end]
                        base_segment = baseline_1.loc[start:end]
                        deltas = {
                            "sharpe": p1.sharpe(candidate_segment) - p1.sharpe(base_segment),
                            "annual": p1.annual_return(candidate_segment) - p1.annual_return(base_segment),
                            "top10": p1.top10_summary(candidate_segment)["top10_mean_depth"] - p1.top10_summary(base_segment)["top10_mean_depth"],
                        }
                        for metric, value in deltas.items():
                            row[f"{period}_{metric}_delta"] = value
                        row[f"{period}_trigger_days"] = int(
                            triggers.loc[triggers["date"].between(start, end), "date"].nunique()
                            if len(triggers)
                            else 0
                        )
                        improvements.extend([deltas["sharpe"] / 0.05, deltas["annual"] / 0.01, deltas["top10"] / 0.005])
                    row["robust_score"] = min(improvements)
                    row["eligible_DV_triple"] = bool(
                        min(improvements) > 1e-6
                        and row["D_trigger_days"] >= 3
                        and row["V_trigger_days"] >= 3
                    )
                    rows.append(row)
    screen = pd.DataFrame(rows).sort_values(["eligible_DV_triple", "robust_score", "trigger_days", "budget"], ascending=[False, False, True, True])
    eligible = screen.loc[screen["eligible_DV_triple"]]
    selected = str(eligible.iloc[0]["candidate"]) if len(eligible) else None
    screen.to_csv(HERE / f"{PREFIX}_dv_screen.csv", index=False)
    diagnostics.to_csv(HERE / f"{PREFIX}_model_diagnostics.csv", index=False)
    if selected is None:
        pd.DataFrame([{"selected": None, "eligible_count": 0}]).to_csv(HERE / f"{PREFIX}_summary.csv", index=False)
        print("No D/V triple candidate")
        print(screen.to_string(index=False))
        print("\nModel diagnostics")
        print(diagnostics.to_string(index=False))
        return
    audit_rows = []
    for name in eligible["candidate"]:
        target, triggers, simulation, returns_1, returns_5 = artifacts[name]
        row = {"candidate": name}
        all_triple = True
        for period, (start, end) in periods.items():
            candidate_segment = returns_1.loc[start:end]
            base_segment = baseline_1.loc[start:end]
            deltas = {
                "sharpe": p1.sharpe(candidate_segment) - p1.sharpe(base_segment),
                "annual": p1.annual_return(candidate_segment) - p1.annual_return(base_segment),
                "top10": p1.top10_summary(candidate_segment)["top10_mean_depth"] - p1.top10_summary(base_segment)["top10_mean_depth"],
            }
            for metric, value in deltas.items():
                row[f"{period}_{metric}_delta"] = value
            all_triple &= min(deltas.values()) >= -1e-12
        row["all_segments_triple"] = all_triple
        row["FULL_5bp_sharpe_delta"] = p1.sharpe(returns_5) - p1.sharpe(baseline_5)
        row["FULL_5bp_annual_delta"] = p1.annual_return(returns_5) - p1.annual_return(baseline_5)
        audit_rows.append(row)
    audit = pd.DataFrame(audit_rows).merge(screen, on="candidate").sort_values(["all_segments_triple", "FULL_annual_delta", "FULL_sharpe_delta"], ascending=[False, False, False])
    audit.to_csv(HERE / f"{PREFIX}_eligible_audit.csv", index=False)
    summary = pd.DataFrame([{"selected": selected, "eligible_DV_count": len(eligible), "total_candidates": len(screen), "selected_all_segments_triple": bool(audit.loc[audit["candidate"] == selected, "all_segments_triple"].iloc[0]), "any_all_segments_triple": bool(audit["all_segments_triple"].any())}])
    summary.to_csv(HERE / f"{PREFIX}_summary.csv", index=False)
    print("Selected", selected)
    print(screen.loc[screen["candidate"] == selected].to_string(index=False))
    print("\nEligible audit")
    print(audit.to_string(index=False))
    print("\nModel diagnostics")
    print(diagnostics.to_string(index=False))


if __name__ == "__main__":
    main()
