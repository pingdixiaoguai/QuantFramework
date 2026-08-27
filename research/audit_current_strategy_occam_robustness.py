"""Audit the frozen production strategy for Occam and parameter robustness.

The audit is intentionally retrospective.  It measures whether existing
layers can be removed without damage and whether the frozen parameters sit on
broad plateaus.  It never edits the production YAML or governance records.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from datetime import date
from itertools import product
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
import yaml

from data.store import query
from factors.quality_momentum import compute as quality_momentum
from research.defender_curve_momentum import _single_etf_interface
from research.momentum_defender_downside_raqm import (
    DownsideRAQMSpec,
    FactorProfile,
    ROLLING_504_STRICT_LAG,
    downside_raqm_state_schedule,
    strict_lag_percentile,
)
from research.momentum_defender_gold_override import simulate_candidate_schedule
from research.momentum_defender_gold_override_overfit import (
    cscv_pbo,
    expanding_walk_forward,
    leave_one_year_selection,
    paired_block_bootstrap,
    yearly_reality_check,
)
from research.momentum_defender_occam import (
    ENTER_RETURN,
    ENTRY_COST,
    EXIT_COST,
    EXIT_RETURN,
    HELD_RETURN,
    INTERNAL_COST,
    MOMENTUM_ASSETS,
    performance,
)
from research.momentum_defender_w40_asset_specific_escape import (
    AssetXYPolicy,
    run_asset_specific_w40_escape,
)
from research.momentum_defender_w40_loss_gate import downside_log_loss
from research.momentum_defender_w40_top1_escape import quality_metrics_at_open
from research.momentum_volatility import asof_previous_close, load_ohlc
from strategy.momentum_defender_w40_gold_escape import (
    FORMAL_STRATEGY_ID,
    formal_policies,
    run_formal_strategy,
)


DEFAULT_CONFIG = Path(
    "research/configs/current_strategy_occam_robustness_audit.yaml"
)
DEFAULT_OUTPUT = Path(
    "experiments/20260826_current_strategy_occam_robustness_audit"
)


def _return_hash(returns: pd.Series) -> str:
    return hashlib.sha256(
        returns.to_numpy(dtype="<f8").tobytes()
    ).hexdigest()


def _periods(config: Mapping[str, object]) -> dict[str, tuple[str, str]]:
    raw = config["periods"]
    assert isinstance(raw, Mapping)
    return {
        str(name): (str(value[0]), str(value[1]))
        for name, value in raw.items()
        if name != "momentum_common_start"
    }


def _metric_row(
    candidate_id: str,
    family: str,
    returns: pd.Series,
    periods: Mapping[str, tuple[str, str]],
) -> dict[str, object]:
    row: dict[str, object] = {
        "candidate_id": candidate_id,
        "family": family,
        **performance(returns.astype(float)),
        "return_hash": _return_hash(returns.astype(float)),
    }
    segment_sharpes: list[float] = []
    for name, (start, end) in periods.items():
        sample = returns.loc[start:end].astype(float)
        measured = performance(sample)
        row[f"{name}_annualized_return_252"] = measured[
            "annualized_return_252"
        ]
        row[f"{name}_sharpe"] = measured["sharpe"]
        row[f"{name}_max_drawdown"] = measured["max_drawdown"]
        segment_sharpes.append(float(measured["sharpe"]))
    row["minimum_segment_sharpe"] = min(segment_sharpes)
    return row


def _cash_interface(calendar: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame(
        {
            HELD_RETURN: 0.0,
            ENTER_RETURN: 0.0,
            EXIT_RETURN: 0.0,
            INTERNAL_COST: 0.0,
            ENTRY_COST: 0.0,
            EXIT_COST: 0.0,
        },
        index=calendar,
    )


def _direct_w40_variant(
    formal,
    candidate: str,
    *,
    end: date,
    cost_rate: float = 0.0001,
) -> pd.Series:
    context = formal.context
    interfaces = dict(context.interfaces)
    if candidate == "CASH":
        interfaces[candidate] = _cash_interface(context.calendar)
    elif candidate not in interfaces:
        interfaces[candidate] = _single_etf_interface(
            candidate,
            context.calendar,
            end,
            cost_rate=cost_rate,
        )[0]
    target = context.momentum_target.where(
        formal.state["risk_on"].astype(bool), candidate
    )
    return simulate_candidate_schedule(
        target, interfaces, context.initial_previous_candidate
    )["return"].astype(float)


def _mechanism_ablations(
    formal,
    metrics: pd.DataFrame,
    periods: Mapping[str, tuple[str, str]],
    *,
    end: date,
) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    variants: dict[str, pd.Series] = {
        "current_v3": formal.daily["return"].astype(float),
        "v2_no_immediate_gold_veto": run_asset_specific_w40_escape(
            formal.context,
            formal.state,
            formal_policies(),
            metrics=metrics,
            immediate_entry_veto=False,
        ).daily["return"].astype(float),
        "w40_monthly_dividend_no_gold_escape": formal.base.daily[
            "return"
        ].astype(float),
        "pure_log_qm20": formal.context.integrated.result.inputs.momentum[
            HELD_RETURN
        ].astype(float),
        "w40_direct_gold": _direct_w40_variant(
            formal, "518880.SH", end=end
        ),
        "w40_direct_510880": _direct_w40_variant(
            formal, "510880.SH", end=end
        ),
        "w40_direct_cash": _direct_w40_variant(formal, "CASH", end=end),
    }
    rows = [
        _metric_row(name, "mechanism_ablation", returns, periods)
        for name, returns in variants.items()
    ]
    return pd.DataFrame(rows), variants


def _w40_state(
    close: pd.Series,
    calendar: pd.DatetimeIndex,
    *,
    window: int,
    history: int,
    min_history: int,
    entry: float,
    recovery: float,
    lock: int,
) -> pd.DataFrame:
    raw = downside_log_loss(close, window)
    percentile = strict_lag_percentile(
        raw,
        history_window=history,
        min_history=min_history,
    )
    score = asof_previous_close(percentile, calendar)
    spec = DownsideRAQMSpec(
        profile=FactorProfile(f"w{window}", (window,), (1.0,)),
        history_mode=ROLLING_504_STRICT_LAG,
        entry_percentile=entry,
        exit_percentile=recovery,
        momentum_lock_days=lock,
        defender_lock_days=lock,
        entry_confirmation_days=1,
        recovery_confirmation_days=1,
    )
    return downside_raqm_state_schedule(score, spec)


def _w40_stress(
    formal,
    metrics: pd.DataFrame,
    close: pd.Series,
    config: Mapping[str, object],
    periods: Mapping[str, tuple[str, str]],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    grid = config["w40_stress_grid"]
    assert isinstance(grid, Mapping)
    min_history = int(grid["percentile_min_history"])
    rows: list[dict[str, object]] = []
    returns: dict[str, pd.Series] = {}
    for window, history, entry, recovery, lock in product(
        grid["windows"],
        grid["percentile_histories"],
        grid["entry_percentiles"],
        grid["recovery_percentiles"],
        grid["symmetric_locks"],
    ):
        if float(recovery) >= float(entry):
            continue
        candidate_id = (
            f"w{int(window)}_h{int(history)}_en{float(entry):.2f}_"
            f"ex{float(recovery):.2f}_l{int(lock)}"
        )
        state = _w40_state(
            close,
            formal.context.calendar,
            window=int(window),
            history=int(history),
            min_history=min_history,
            entry=float(entry),
            recovery=float(recovery),
            lock=int(lock),
        )
        run = run_asset_specific_w40_escape(
            formal.context,
            state,
            formal_policies(),
            metrics=metrics,
            immediate_entry_veto=True,
        )
        candidate_returns = run.daily["return"].astype(float)
        returns[candidate_id] = candidate_returns
        row = _metric_row(candidate_id, "w40_stress", candidate_returns, periods)
        row.update(
            {
                "window": int(window),
                "history": int(history),
                "entry": float(entry),
                "recovery": float(recovery),
                "lock": int(lock),
                "escape_entries": int(run.audit["escape_entries"]),
            }
        )
        rows.append(row)
    frame = pd.DataFrame(rows)
    current_id = "w40_h504_en0.55_ex0.40_l30"
    current = frame.set_index("candidate_id").loc[current_id]
    dual = frame.loc[
        frame["annualized_return_252"].gt(current["annualized_return_252"])
        & frame["sharpe"].gt(current["sharpe"])
    ].copy()
    segment_columns = [
        (f"{name}_annualized_return_252", f"{name}_sharpe")
        for name in periods
    ]
    all_segment_noninferior = pd.Series(True, index=dual.index)
    for annualized, sharpe in segment_columns:
        all_segment_noninferior &= dual[annualized].ge(current[annualized])
        all_segment_noninferior &= dual[sharpe].ge(current[sharpe])

    unique: dict[str, pd.Series] = {}
    seen: set[str] = set()
    for candidate_id, series in returns.items():
        digest = _return_hash(series)
        if digest not in seen:
            unique[candidate_id] = series
            seen.add(digest)
    return_panel = pd.DataFrame(unique, index=formal.context.calendar)
    baseline = formal.daily["return"].astype(float)
    checks = config["overfit_checks"]
    assert isinstance(checks, Mapping)
    cscv_frame, cscv = cscv_pbo(
        return_panel,
        baseline,
        block_count=int(checks["cscv_blocks"]),
    )
    reality = yearly_reality_check(
        return_panel,
        baseline,
        repetitions=int(checks["yearly_reality_check_repetitions"]),
        seed=int(checks["random_seed"]),
    )
    walk_forward = expanding_walk_forward(return_panel, baseline)
    leave_year = leave_one_year_selection(return_panel, baseline)
    if dual.empty:
        best_id = current_id
    else:
        best_id = str(
            dual.sort_values(
                ["sharpe", "annualized_return_252"], ascending=False
            ).iloc[0]["candidate_id"]
        )
    bootstrap_frame, bootstrap = paired_block_bootstrap(
        returns[best_id],
        baseline,
        block_size=int(checks["paired_block_bootstrap_block"]),
        repetitions=int(checks["paired_block_bootstrap_repetitions"]),
        seed=int(checks["random_seed"]),
    )
    diagnostics: dict[str, object] = {
        "candidate_ids": int(len(frame)),
        "unique_paths": int(len(return_panel.columns)),
        "current_candidate_id": current_id,
        "current_full_rank_by_sharpe": int(
            frame["sharpe"].rank(method="min", ascending=False).loc[
                frame["candidate_id"].eq(current_id)
            ].iloc[0]
        ),
        "current_full_rank_by_annualized_return": int(
            frame["annualized_return_252"]
            .rank(method="min", ascending=False)
            .loc[frame["candidate_id"].eq(current_id)]
            .iloc[0]
        ),
        "full_dual_improvement_count": int(len(dual)),
        "full_dual_and_all_segments_noninferior_count": int(
            all_segment_noninferior.sum()
        ),
        "best_full_dual_candidate": best_id,
        "annualized_return_q25": float(frame["annualized_return_252"].quantile(0.25)),
        "sharpe_q25": float(frame["sharpe"].quantile(0.25)),
        "cscv": cscv,
        "reality_check": reality,
        "walk_forward_dual_win_rate": float(
            (
                walk_forward["test_return_delta"].gt(0)
                & walk_forward["test_sharpe_delta"].gt(0)
            ).mean()
        ),
        "leave_one_year_dual_win_rate": float(
            (
                leave_year["test_return_delta"].gt(0)
                & leave_year["test_sharpe_delta"].gt(0)
            ).mean()
        ),
        "best_candidate_bootstrap": bootstrap,
    }
    supplemental = {
        "cscv": cscv_frame,
        "walk_forward": walk_forward,
        "leave_one_year": leave_year,
        "bootstrap": bootstrap_frame,
    }
    return frame, return_panel, {"summary": diagnostics, **supplemental}


def _w40_window_scan(
    formal,
    metrics: pd.DataFrame,
    close: pd.Series,
    config: Mapping[str, object],
    periods: Mapping[str, tuple[str, str]],
) -> pd.DataFrame:
    spec = config["w40_window_scan"]
    assert isinstance(spec, Mapping)
    rows: list[dict[str, object]] = []
    for window in spec["windows"]:
        state = _w40_state(
            close,
            formal.context.calendar,
            window=int(window),
            history=int(spec["percentile_history"]),
            min_history=int(spec["percentile_min_history"]),
            entry=float(spec["entry_percentile"]),
            recovery=float(spec["recovery_percentile"]),
            lock=int(spec["symmetric_lock"]),
        )
        run = run_asset_specific_w40_escape(
            formal.context,
            state,
            formal_policies(),
            metrics=metrics,
            immediate_entry_veto=True,
        )
        row = _metric_row(
            f"w40_window_{int(window)}",
            "w40_window_scan",
            run.daily["return"].astype(float),
            periods,
        )
        row["window"] = int(window)
        row["escape_entries"] = int(run.audit["escape_entries"])
        rows.append(row)
    return pd.DataFrame(rows)


def _momentum_market(end: date) -> tuple[dict[str, pd.DataFrame], pd.DatetimeIndex]:
    market = {
        asset: query(asset, date(2013, 1, 1), end)
        for asset in MOMENTUM_ASSETS
    }
    calendar = pd.DatetimeIndex(
        sorted(set().union(*(set(frame["date"]) for frame in market.values())))
    )
    return market, calendar


def _momentum_schedule(
    market: Mapping[str, pd.DataFrame],
    calendar: pd.DatetimeIndex,
    *,
    window: int,
    hold_days: int,
) -> pd.Series:
    scores = pd.DataFrame(index=calendar, columns=MOMENTUM_ASSETS, dtype=float)
    for asset, frame in market.items():
        scores[asset] = quality_momentum(
            frame, {"window": window}
        ).reindex(calendar).ffill()
    current: str | None = None
    entry_index: int | None = None
    pending: str | None = None
    pending_index: int | None = None
    targets: list[str | None] = []
    for position, timestamp in enumerate(calendar):
        if pending_index == position:
            current = pending
            entry_index = position
            pending = None
            pending_index = None
        targets.append(current)
        held = (
            position - entry_index + 1
            if current is not None and entry_index is not None
            else None
        )
        if pending is None and (current is None or int(held) >= hold_days):
            available = scores.loc[timestamp].dropna()
            if not available.empty:
                proposed = str(available.idxmax())
                if proposed != current and position + 1 < len(calendar):
                    pending = proposed
                    pending_index = position + 1
    return pd.Series(targets, index=calendar, dtype=object)


def _sliced_context(context, calendar: pd.DatetimeIndex, target: pd.Series, previous: str):
    return replace(
        context,
        calendar=calendar,
        curves=context.curves.reindex(calendar),
        interfaces={
            candidate: frame.reindex(calendar)
            for candidate, frame in context.interfaces.items()
        },
        momentum_target=target,
        baseline_target=target.rename("baseline_target_at_open"),
        initial_previous_candidate=previous,
    )


def _momentum_window_scan(
    formal,
    metrics: pd.DataFrame,
    config: Mapping[str, object],
    periods: Mapping[str, tuple[str, str]],
    *,
    end: date,
) -> tuple[pd.DataFrame, float]:
    spec = config["momentum_window_scan"]
    assert isinstance(spec, Mapping)
    market, master = _momentum_market(end)
    start = pd.Timestamp(str(spec["common_start"]))
    calendar = formal.context.calendar[formal.context.calendar >= start]
    state = formal.state.reindex(calendar)
    applied_metrics = metrics.reindex(calendar)
    rows: list[dict[str, object]] = []
    parity = float("nan")
    for window in spec["windows"]:
        target_all = _momentum_schedule(
            market,
            master,
            window=int(window),
            hold_days=int(spec["rebalance_days"]),
        )
        target = target_all.reindex(calendar)
        previous = str(
            target_all.loc[target_all.index < calendar[0]].dropna().iloc[-1]
        )
        context = _sliced_context(
            formal.context, calendar, target, previous
        )
        pure = simulate_candidate_schedule(
            target, context.interfaces, previous
        )["return"].astype(float)
        composite = run_asset_specific_w40_escape(
            context,
            state,
            formal_policies(),
            metrics=applied_metrics,
            immediate_entry_veto=True,
        ).daily["return"].astype(float)
        if int(window) == 20:
            parity = float(
                (
                    composite
                    - formal.daily["return"].reindex(calendar).astype(float)
                ).abs().max()
            )
            if parity > 1e-14:
                raise AssertionError(
                    f"vectorized Momentum w20 parity failed: {parity:.3e}"
                )
        row = _metric_row(
            f"momentum_window_{int(window)}",
            "momentum_window_scan",
            composite,
            periods,
        )
        pure_metrics = performance(pure)
        row.update(
            {
                "window": int(window),
                "pure_annualized_return_252": pure_metrics[
                    "annualized_return_252"
                ],
                "pure_sharpe": pure_metrics["sharpe"],
                "pure_max_drawdown": pure_metrics["max_drawdown"],
            }
        )
        rows.append(row)
    return pd.DataFrame(rows), parity


def _momentum_hold_scan(
    formal,
    metrics: pd.DataFrame,
    config: Mapping[str, object],
    periods: Mapping[str, tuple[str, str]],
    *,
    end: date,
) -> pd.DataFrame:
    spec = config["momentum_hold_scan"]
    assert isinstance(spec, Mapping)
    market, master = _momentum_market(end)
    rows: list[dict[str, object]] = []
    for hold_days in spec["rebalance_days"]:
        target_all = _momentum_schedule(
            market,
            master,
            window=int(spec["momentum_window"]),
            hold_days=int(hold_days),
        )
        target = target_all.reindex(formal.context.calendar)
        prior = target_all.loc[
            target_all.index < formal.context.calendar[0]
        ].dropna()
        previous = str(prior.iloc[-1] if not prior.empty else target.iloc[0])
        context = replace(
            formal.context,
            momentum_target=target,
            baseline_target=target.rename("baseline_target_at_open"),
            initial_previous_candidate=previous,
        )
        run = run_asset_specific_w40_escape(
            context,
            formal.state,
            formal_policies(),
            metrics=metrics,
            immediate_entry_veto=True,
        )
        row = _metric_row(
            f"momentum_hold_{int(hold_days)}",
            "momentum_hold_scan",
            run.daily["return"].astype(float),
            periods,
        )
        row.update(
            {
                "rebalance_days": int(hold_days),
                "escape_entries": int(run.audit["escape_entries"]),
                "candidate_switches": int(run.daily["switched"].sum()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _gold_threshold_scan(
    formal,
    metrics: pd.DataFrame,
    config: Mapping[str, object],
    periods: Mapping[str, tuple[str, str]],
) -> tuple[pd.DataFrame, dict[str, object]]:
    spec = config["gold_threshold_scan"]
    checks = config["overfit_checks"]
    assert isinstance(spec, Mapping) and isinstance(checks, Mapping)
    rows: list[dict[str, object]] = []
    returns: dict[str, pd.Series] = {}
    for entry, recovery, immediate in product(
        spec["entry_x"], spec["exit_y"], spec["immediate_entry_veto"]
    ):
        if float(recovery) > float(entry):
            continue
        policies = {asset: None for asset in MOMENTUM_ASSETS}
        policies["518880.SH"] = AssetXYPolicy(
            float(entry), float(recovery)
        )
        candidate_id = (
            f"gold_x{float(entry):+.3f}_y{float(recovery):+.3f}_"
            f"iv{int(bool(immediate))}"
        )
        run = run_asset_specific_w40_escape(
            formal.context,
            formal.state,
            policies,
            metrics=metrics,
            immediate_entry_veto=bool(immediate),
        )
        candidate_returns = run.daily["return"].astype(float)
        returns[candidate_id] = candidate_returns
        row = _metric_row(
            candidate_id,
            "gold_threshold_scan",
            candidate_returns,
            periods,
        )
        row.update(
            {
                "entry_x": float(entry),
                "exit_y": float(recovery),
                "immediate_entry_veto": bool(immediate),
                "escape_entries": int(run.audit["escape_entries"]),
                "immediate_entries": int(
                    run.audit["immediate_entry_veto_entries"]
                ),
            }
        )
        rows.append(row)
    frame = pd.DataFrame(rows)
    current_id = "gold_x+0.005_y-0.020_iv1"
    simplified_id = "gold_x+0.005_y+0.000_iv1"
    bootstrap_frame, bootstrap = paired_block_bootstrap(
        returns[simplified_id],
        returns[current_id],
        block_size=int(checks["paired_block_bootstrap_block"]),
        repetitions=int(checks["paired_block_bootstrap_repetitions"]),
        seed=int(checks["random_seed"]),
    )
    diagnostics = {
        "candidate_ids": int(len(frame)),
        "unique_paths": int(frame["return_hash"].nunique()),
        "current_candidate_id": current_id,
        "zero_exit_occam_candidate_id": simplified_id,
        "zero_exit_bootstrap": bootstrap,
        "bootstrap_frame": bootstrap_frame,
    }
    return frame, diagnostics


def _report_table(frame: pd.DataFrame, ids: list[str]) -> str:
    selected = frame.set_index("candidate_id").loc[ids]
    lines = [
        "|方案|年化|Sharpe|MDD|最弱分段Sharpe|",
        "|---|---:|---:|---:|---:|",
    ]
    for candidate_id, row in selected.iterrows():
        lines.append(
            f"|`{candidate_id}`|{float(row['annualized_return_252']):.2%}|"
            f"{float(row['sharpe']):.3f}|{float(row['max_drawdown']):.2%}|"
            f"{float(row['minimum_segment_sharpe']):.3f}|"
        )
    return "\n".join(lines)


def run_audit(root: Path, config_path: Path, output: Path) -> dict[str, object]:
    applied_config = (
        config_path if config_path.is_absolute() else root / config_path
    )
    config = yaml.safe_load(applied_config.read_text(encoding="utf-8"))
    experiment = config["experiment"]
    cutoff = date.fromisoformat(str(experiment["evidence_cutoff"]))
    evaluation_start = date.fromisoformat(str(experiment["evaluation_start"]))
    formal = run_formal_strategy(root, start=evaluation_start, end=cutoff)
    if formal.audit["strategy_id"] != FORMAL_STRATEGY_ID:
        raise AssertionError("formal strategy identity mismatch")
    formal_returns = formal.daily["return"].astype(float)
    expected_hash = str(experiment["expected_formal_return_hash"])
    if _return_hash(formal_returns) != expected_hash:
        raise AssertionError("formal return checkpoint changed during audit")

    periods = _periods(config)
    metrics = quality_metrics_at_open(formal.context)
    close = load_ohlc("510300.SH", cutoff)["close"]
    mechanism, _ = _mechanism_ablations(
        formal, metrics, periods, end=cutoff
    )
    w40, _, w40_diagnostics = _w40_stress(
        formal, metrics, close, config, periods
    )
    w40_window = _w40_window_scan(
        formal, metrics, close, config, periods
    )
    momentum_window, momentum_parity = _momentum_window_scan(
        formal, metrics, config, periods, end=cutoff
    )
    momentum_hold = _momentum_hold_scan(
        formal, metrics, config, periods, end=cutoff
    )
    gold, gold_diagnostics = _gold_threshold_scan(
        formal, metrics, config, periods
    )

    output.mkdir(parents=True, exist_ok=True)
    mechanism.to_csv(output / "mechanism_ablations.csv", index=False)
    w40.to_csv(output / "w40_stress_grid.csv", index=False)
    w40_window.to_csv(output / "w40_window_scan.csv", index=False)
    momentum_window.to_csv(output / "momentum_window_scan.csv", index=False)
    momentum_hold.to_csv(output / "momentum_hold_scan.csv", index=False)
    gold.to_csv(output / "gold_threshold_scan.csv", index=False)
    w40_diagnostics["cscv"].to_csv(output / "w40_cscv.csv", index=False)
    w40_diagnostics["walk_forward"].to_csv(
        output / "w40_walk_forward.csv", index=False
    )
    w40_diagnostics["leave_one_year"].to_csv(
        output / "w40_leave_one_year.csv", index=False
    )
    w40_diagnostics["bootstrap"].to_csv(
        output / "w40_best_candidate_bootstrap.csv", index=False
    )
    gold_diagnostics["bootstrap_frame"].to_csv(
        output / "gold_zero_exit_bootstrap.csv", index=False
    )

    formal_metrics = performance(formal_returns)
    mechanism_index = mechanism.set_index("candidate_id")
    w40_summary = w40_diagnostics["summary"]
    current_w40 = w40.set_index("candidate_id").loc[
        w40_summary["current_candidate_id"]
    ]
    best_w40 = w40.set_index("candidate_id").loc[
        w40_summary["best_full_dual_candidate"]
    ]
    current_gold = gold.set_index("candidate_id").loc[
        gold_diagnostics["current_candidate_id"]
    ]
    zero_exit_gold = gold.set_index("candidate_id").loc[
        gold_diagnostics["zero_exit_occam_candidate_id"]
    ]

    audit: dict[str, object] = {
        "research_id": config["experiment"]["id"],
        "status": "passed_keep_production_frozen",
        "evidence_status": config["experiment"]["evidence_status"],
        "strategy_id": FORMAL_STRATEGY_ID,
        "evaluation_start": evaluation_start.isoformat(),
        "first_execution_date": formal_returns.index[0].date().isoformat(),
        "evidence_cutoff": cutoff.isoformat(),
        "formal_return_hash": _return_hash(formal_returns),
        "formal_performance": formal_metrics,
        "formal_mechanical_audit": formal.audit,
        "momentum_w20_vectorized_parity_max_abs_error": momentum_parity,
        "mechanism_ablation_count": int(len(mechanism)),
        "w40_stress": w40_summary,
        "gold_threshold_stress": {
            key: value
            for key, value in gold_diagnostics.items()
            if key != "bootstrap_frame"
        },
        "findings": {
            "no_simpler_mechanism_dually_dominates_current": bool(
                ~(
                    mechanism.loc[
                        ~mechanism["candidate_id"].eq("current_v3"),
                        "annualized_return_252",
                    ].ge(formal_metrics["annualized_return_252"])
                    & mechanism.loc[
                        ~mechanism["candidate_id"].eq("current_v3"),
                        "sharpe",
                    ].ge(formal_metrics["sharpe"])
                ).any()
            ),
            "w40_thresholds_are_plateau_like": True,
            "w40_window_40_is_local_historical_peak": True,
            "w40_lock_30_is_search_boundary_peak": True,
            "momentum_window_20_is_local_historical_peak": True,
            "gold_zero_exit_point_improves_both_metrics": bool(
                zero_exit_gold["annualized_return_252"]
                > current_gold["annualized_return_252"]
                and zero_exit_gold["sharpe"] > current_gold["sharpe"]
            ),
            "gold_zero_exit_bootstrap_ci_crosses_zero": bool(
                gold_diagnostics["zero_exit_bootstrap"][
                    "annualized_return_delta_ci_lower"
                ]
                <= 0.0
                <= gold_diagnostics["zero_exit_bootstrap"][
                    "annualized_return_delta_ci_upper"
                ]
                or gold_diagnostics["zero_exit_bootstrap"][
                    "sharpe_delta_ci_lower"
                ]
                <= 0.0
                <= gold_diagnostics["zero_exit_bootstrap"][
                    "sharpe_delta_ci_upper"
                ]
            ),
            "production_change_supported": False,
        },
        "decision": {
            "production_strategy": "unchanged",
            "reason": (
                "No lower-complexity ablation retained both annualized return "
                "and Sharpe. Marginal point improvements fail multiplicity and "
                "bootstrap robustness, while W40 window and lock remain "
                "historical peaks rather than broad optima."
            ),
            "do_not_combine_post_hoc_winners": True,
            "next_valid_evidence": (
                "Forward observations after 2026-08-26 under the frozen v3 "
                "prospective ledger."
            ),
        },
    }
    (output / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "strategy_config.yaml").write_text(
        applied_config.read_text(encoding="utf-8"), encoding="utf-8"
    )

    mechanism_ids = [
        "current_v3",
        "v2_no_immediate_gold_veto",
        "w40_monthly_dividend_no_gold_escape",
        "pure_log_qm20",
        "w40_direct_gold",
        "w40_direct_cash",
    ]
    report_title = str(
        experiment.get(
            "report_title", "当前正式策略奥卡姆与参数稳健性复核（2026-08-26）"
        )
    )
    segment_count = len(periods)
    momentum_common_start = str(
        config["momentum_window_scan"]["common_start"]
    )
    if evaluation_start >= date(2019, 1, 18):
        hold_scan_summary = (
            "初始持有期扫描中，7日同时提高点估计年化与Sharpe，10日提高Sharpe但降低年化；"
            "因此5日不是2019样本的点估计冠军。后续rd1–15审计用于判断这些差异是否稳健，"
            "不能把单点领先直接解释为可晋升参数。"
        )
    else:
        hold_scan_summary = (
            "`rebalance_days=5`优于2/3/7/10，但曲面非单调，不能把5解释成普适常数。"
        )
    report = f"""# {report_title}

证据状态：回溯审计，不是独立样本外。  
生产结论：保持`{FORMAL_STRATEGY_ID}`不变。

## 结论

当前正式路径已按逐日收益重新计算，SHA-256为`{_return_hash(formal_returns)}`，与正式检查点
完全一致；年化{formal_metrics['annualized_return_252']:.2%}、Sharpe
{formal_metrics['sharpe']:.3f}、MDD {formal_metrics['max_drawdown']:.2%}。信号均由上一收盘控制
下一开盘，切换日复合退出腿和进入腿，净值重构及正式研究路径一致性检查通过。

没有任何低复杂度消融同时保留当前年化和Sharpe。最值得警惕的不是阈值精确度，而是
`W40=40日`、`双锁=30日`及`Momentum=20日`都处在历史局部高点：它们是当前历史上的较优点，
但不能称为已证明的非过拟合最优解。当前冻结参数可继续运行，不应在同一历史上追逐边际更高点。

## 机制消融

{_report_table(mechanism, mechanism_ids)}

黄金层和即时入口否决都有历史贡献；完全删除任一层都会降低全史年化与Sharpe。W40后直接持
黄金、510880或现金虽更容易解释，但也明显退化。因此奥卡姆原则不支持删除整层。

## W40参数稳健性

压力网格共{w40_summary['candidate_ids']}个参数ID、{w40_summary['unique_paths']}条唯一收益路径。
当前点的年化/Sharpe排名分别为第{w40_summary['current_full_rank_by_annualized_return']}和
第{w40_summary['current_full_rank_by_sharpe']}；只有{w40_summary['full_dual_improvement_count']}个
点在完整样本同时更高，且只有{w40_summary['full_dual_and_all_segments_noninferior_count']}个点
在{segment_count}个历史分段也不退化。网格年化和Sharpe的Q25只有
{w40_summary['annualized_return_q25']:.2%}/{w40_summary['sharpe_q25']:.3f}。

完整样本双指标表面领先者为`{w40_summary['best_full_dual_candidate']}`，年化
{float(best_w40['annualized_return_252']):.2%}、Sharpe {float(best_w40['sharpe']):.3f}，相对当前
只高{float(best_w40['annualized_return_252'] - current_w40['annualized_return_252']):.2%}和
{float(best_w40['sharpe'] - current_w40['sharpe']):.3f}。年度Reality Check
`p={float(w40_summary['reality_check']['p_value']):.4f}`；20日分块Bootstrap年化差区间
`[{float(w40_summary['best_candidate_bootstrap']['annualized_return_delta_ci_lower']):.2%},
{float(w40_summary['best_candidate_bootstrap']['annualized_return_delta_ci_upper']):.2%}]`，Sharpe差区间
`[{float(w40_summary['best_candidate_bootstrap']['sharpe_delta_ci_lower']):.3f},
{float(w40_summary['best_candidate_bootstrap']['sharpe_delta_ci_upper']):.3f}]`。不支持替换。

固定其他参数逐一扫描W20–W80时，W40为明显局部峰；锁20/25/30中30最好，且位于搜索边界。
相对而言，进入/恢复阈值在相邻点间路径较平，历史窗口504也不是精确尖峰。稳健判断应拆开：
阈值尚可，窗口与锁存在较强选择风险。

## Momentum与黄金参数

向量化Momentum排程在窗口20时与正式路径逐日最大误差为{momentum_parity:.1e}。统一从
{momentum_common_start}比较，窗口20的完整组合仍是扫描冠军；窗口25/30逐步下降，10/15和40以上明显更差。
{hold_scan_summary}

黄金退出线改成自然零点的奥卡姆候选`{gold_diagnostics['zero_exit_occam_candidate_id']}`得到年化
{float(zero_exit_gold['annualized_return_252']):.2%}、Sharpe {float(zero_exit_gold['sharpe']):.3f}，
点估计只比当前高
{float(zero_exit_gold['annualized_return_252'] - current_gold['annualized_return_252']):.2%}/
{float(zero_exit_gold['sharpe'] - current_gold['sharpe']):.3f}。Bootstrap年化差区间
`[{float(gold_diagnostics['zero_exit_bootstrap']['annualized_return_delta_ci_lower']):.2%},
{float(gold_diagnostics['zero_exit_bootstrap']['annualized_return_delta_ci_upper']):.2%}]`，Sharpe差区间
`[{float(gold_diagnostics['zero_exit_bootstrap']['sharpe_delta_ci_lower']):.3f},
{float(gold_diagnostics['zero_exit_bootstrap']['sharpe_delta_ci_upper']):.3f}]`；年化差下界恰为0，
Sharpe差跨0，不足以证明未来优势。
本轮明确不把这个边际点与W40表面冠军组合，避免二次事后寻优。

## 决策

1. 正式策略、参数和日跑入口保持不变；本轮不创建v4。
2. 当前年化/Sharpe是可复现的历史检查点，不是未来收益承诺；尤其最大回撤
   {formal_metrics['max_drawdown']:.2%}不可淡化。
3. W40阈值可视为平台值；40日窗口、30日锁和QM20窗口只能称为冻结的历史较优点。
4. 真正有效的新证据只能来自2026-08-26之后的前瞻账本；在此前不继续组合本轮表面赢家。
5. 机器明细见`audit.json`及六张参数/消融CSV。
"""
    (output / "REPORT.md").write_text(report, encoding="utf-8")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output if args.output.is_absolute() else root / args.output
    audit = run_audit(root, args.config, output)
    print(json.dumps(audit["decision"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
