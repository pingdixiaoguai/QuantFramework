"""Compare and tune four causal volatility alerts for Momentum/Defender switching.

The experiment keeps the frozen slow gate and 30-day state lock from the
accepted Occam candidate.  Only the emergency Momentum-to-Defender alert is
changed.  Parameter selection is reported twice: once using 2019-2022 only,
and once as a clearly labelled full-sample hindsight oracle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from data.store import query
from research.momentum_defender_occam import (
    HELD_RETURN,
    MOMENTUM_ASSETS,
    OccamParams,
    _momentum_target_schedule,
    apply_state_schedule,
    build_inputs,
    indicator_at_effective_open,
    load_defender_bundle,
    performance,
    simulate_switch,
    slow_regime_at_open,
)
from research.run_momentum_defender_occam import _generate_standard_report


DEFAULT_DEFENDER_DIR = Path(
    "/Users/hujiaoyuan/Desktop/Quant/Defender/defender/deliverable"
)
DEFAULT_OUTPUT = Path("experiments/20260821_momentum_volatility_signal_abcd")
DEFAULT_END = date(2026, 8, 17)
SLOW_PARAMS = OccamParams(
    lookback=40,
    risk_on_threshold=0.025,
    min_hold_days=30,
    emergency_daily_loss=None,
)

VOLATILITY_WINDOWS = (10, 20, 40, 60)
EXPANDING_QUANTILES = (0.70, 0.80, 0.90, 0.95)
CAP_TRIGGER_MAXIMUMS = (0.8, 0.6, 0.4)
DOWNSIDE_WINDOWS = (5, 10, 20)
DOWNSIDE_THRESHOLDS = (0.00, -0.02, -0.05)
CAP_STEP = 0.20
CAP_THRESHOLD_MIN_HISTORY = 20

PERIODS = {
    "development_2019_2022": (
        pd.Timestamp("2019-01-18"),
        pd.Timestamp("2022-12-30"),
    ),
    "validation_2023": (
        pd.Timestamp("2023-01-01"),
        pd.Timestamp("2023-12-31"),
    ),
    "evaluation_2024_cutoff": (
        pd.Timestamp("2024-01-01"),
        pd.Timestamp(DEFAULT_END),
    ),
    "full": (
        pd.Timestamp("2019-01-18"),
        pd.Timestamp(DEFAULT_END),
    ),
}

EVENT_WINDOWS = {
    "2023_alert_window": (
        pd.Timestamp("2023-05-16"),
        pd.Timestamp("2023-06-01"),
    ),
    "2023_full_A_defender_episode": (
        pd.Timestamp("2023-05-16"),
        pd.Timestamp("2023-07-25"),
    ),
    "2024_early_entry_through_defense": (
        pd.Timestamp("2024-09-30"),
        pd.Timestamp("2024-11-19"),
    ),
    "2024_original_A_defense_window": (
        pd.Timestamp("2024-10-08"),
        pd.Timestamp("2024-11-19"),
    ),
}


@dataclass(frozen=True)
class AlertSpec:
    scheme: str
    label: str
    volatility_window: int | None = None
    expanding_quantile: float | None = None
    cap_trigger_maximum: float | None = None
    downside_window: int | None = None
    downside_threshold: float | None = None

    def variant_id(self) -> str:
        parts = [self.scheme]
        if self.volatility_window is not None:
            parts.append(f"vw{self.volatility_window}")
        if self.expanding_quantile is not None:
            parts.append(f"q{self.expanding_quantile:.2f}")
        if self.cap_trigger_maximum is not None:
            parts.append(f"cap{self.cap_trigger_maximum:.1f}")
        if self.downside_window is not None:
            parts.append(f"dw{self.downside_window}")
        if self.downside_threshold is not None:
            parts.append(f"dt{self.downside_threshold:+.2f}")
        return "_".join(parts).replace("+", "p").replace("-", "m")


def rogers_satchell_volatility(prices: pd.DataFrame, window: int) -> pd.Series:
    """Annualized Rogers-Satchell realized volatility on asset observations."""
    required = {"open", "high", "low", "close"}
    missing = required - set(prices.columns)
    if missing:
        raise ValueError(f"price history missing OHLC columns: {sorted(missing)}")
    if window < 2:
        raise ValueError("volatility window must be at least 2")
    ohlc = prices[list(required)].astype(float)
    if (ohlc <= 0).any().any():
        raise ValueError("Rogers-Satchell volatility requires positive OHLC")
    variance = (
        np.log(ohlc["high"] / ohlc["close"])
        * np.log(ohlc["high"] / ohlc["open"])
        + np.log(ohlc["low"] / ohlc["close"])
        * np.log(ohlc["low"] / ohlc["open"])
    ).clip(lower=0.0)
    realized = np.sqrt(252.0 * variance.rolling(window, min_periods=window).mean())
    realized.name = f"rs_volatility_{window}"
    return realized


def expanding_volatility_cap(
    realized_volatility: pd.Series,
    quantile: float,
    *,
    step: float = CAP_STEP,
    min_history: int = CAP_THRESHOLD_MIN_HISTORY,
) -> pd.DataFrame:
    """Strict-lag expanding-quantile cap on the close-observation calendar."""
    if not 0.0 < quantile < 1.0:
        raise ValueError("expanding quantile must be strictly between zero and one")
    if not 0.0 < step <= 1.0:
        raise ValueError("cap step must be in (0, 1]")
    volatility = realized_volatility.astype(float)
    threshold = (
        volatility.shift(1)
        .expanding(min_periods=min_history)
        .quantile(quantile)
    )
    raw_cap = (threshold / volatility).clip(upper=1.0)
    cap = np.floor(raw_cap / step + 1e-12) * step
    cap = cap.clip(lower=0.0, upper=1.0).where(raw_cap.notna(), 1.0)
    return pd.DataFrame(
        {
            "realized_volatility": volatility,
            "threshold": threshold,
            "raw_cap": raw_cap,
            "cap": cap,
        }
    )


def asof_previous_close(series: pd.Series, calendar: pd.DatetimeIndex) -> pd.Series:
    """Map each open to the latest strictly earlier close observation."""
    source = series.copy().sort_index()
    if source.index.duplicated().any():
        raise ValueError("source close series contains duplicate dates")
    source_index = pd.DatetimeIndex(source.index)
    positions = source_index.searchsorted(calendar, side="left") - 1
    values = np.full(len(calendar), np.nan, dtype=object)
    valid = positions >= 0
    source_values = source.to_numpy()
    values[valid] = source_values[positions[valid]]
    mapped = pd.Series(values, index=calendar, name=series.name)
    return pd.to_numeric(mapped, errors="coerce")


def _load_ohlc(asset: str, end: date) -> pd.DataFrame:
    frame = query(asset, date(2013, 1, 1), end).sort_values("date")
    frame = frame.drop_duplicates("date").set_index("date")
    required = ["open", "high", "low", "close"]
    if frame.empty or frame[required].isna().any().any():
        raise ValueError(f"invalid OHLC history for {asset}")
    return frame[required].astype(float)


def momentum_asset_at_previous_close(
    momentum_result,
    calendar: pd.DatetimeIndex,
) -> pd.Series:
    """Asset owned by the Momentum sleeve through the close before each open."""
    prior_dates = momentum_result.daily_returns.index[
        momentum_result.daily_returns.index < calendar.min()
    ]
    if len(prior_dates) == 0:
        raise AssertionError("Momentum signal study requires a warm-up holding date")
    replay_calendar = pd.DatetimeIndex([prior_dates.max()]).append(calendar)
    targets = _momentum_target_schedule(momentum_result, replay_calendar)
    assets = targets.idxmax(axis=1)
    previous = assets.shift(1).reindex(calendar)
    if previous.isna().any():
        raise AssertionError("Momentum previous-close asset is missing")
    previous.name = "momentum_asset_at_previous_close"
    return previous


def choose_by_asset(
    values_by_asset: Mapping[str, pd.Series],
    asset_at_open: pd.Series,
) -> pd.Series:
    """Select one causal per-asset signal using the previous-close holding."""
    result = pd.Series(np.nan, index=asset_at_open.index, dtype=float)
    for asset, values in values_by_asset.items():
        mask = asset_at_open.eq(asset)
        result.loc[mask] = values.reindex(result.index).loc[mask].astype(float)
    if result.isna().any():
        missing = sorted(asset_at_open.loc[result.isna()].unique())
        raise AssertionError(f"missing per-asset signal values for: {missing}")
    return result


def _metric_columns(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    prefix: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict[str, object]:
    strategy = strategy_returns.loc[start:end]
    benchmark = benchmark_returns.loc[strategy.index]
    measured = performance(strategy)
    baseline = performance(benchmark)
    return {
        f"{prefix}_observations": int(measured["observations"]),
        f"{prefix}_total_return": float(measured["total_return"]),
        f"{prefix}_annualized_return_252": float(measured["annualized_return_252"]),
        f"{prefix}_annualized_volatility": float(measured["annualized_volatility"]),
        f"{prefix}_sharpe": float(measured["sharpe"]),
        f"{prefix}_max_drawdown": float(measured["max_drawdown"]),
        f"{prefix}_annualized_delta": float(measured["annualized_return_252"])
        - float(baseline["annualized_return_252"]),
        f"{prefix}_sharpe_delta": float(measured["sharpe"])
        - float(baseline["sharpe"]),
        f"{prefix}_max_drawdown_improvement": float(measured["max_drawdown"])
        - float(baseline["max_drawdown"]),
    }


def evaluate_alert(
    spec: AlertSpec,
    alert: pd.Series,
    slow_signal: pd.Series,
    momentum: pd.DataFrame,
    defender: pd.DataFrame,
    benchmark_returns: pd.Series,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    calendar = pd.DatetimeIndex(slow_signal.index)
    emergency = alert.reindex(calendar).fillna(False).astype(bool)
    state = apply_state_schedule(
        slow_signal,
        emergency,
        calendar,
        SLOW_PARAMS.min_hold_days,
        emergency_override=True,
    )
    simulated = simulate_switch(momentum, defender, state["risk_on"])
    emergency_entry = state["state_changed"].astype(bool) & state[
        "state_reason"
    ].eq("emergency_exit")
    row: dict[str, object] = {
        **asdict(spec),
        "variant_id": spec.variant_id(),
        "alert_days": int(emergency.sum()),
        "switches": int(simulated["sleeve_switch"].sum()),
        "defender_days": int((~state["risk_on"]).sum()),
        "defender_share": float((~state["risk_on"]).mean()),
        "emergency_entries": int(emergency_entry.sum()),
    }
    for prefix, (start, end) in PERIODS.items():
        row.update(
            _metric_columns(
                simulated["return"],
                benchmark_returns,
                prefix,
                start,
                end,
            )
        )
        row[f"{prefix}_alert_days"] = int(emergency.loc[start:end].sum())
        row[f"{prefix}_emergency_entries"] = int(
            emergency_entry.loc[start:end].sum()
        )
        row[f"{prefix}_switches"] = int(
            simulated["sleeve_switch"].loc[start:end].sum()
        )
        row[f"{prefix}_defender_days"] = int((~state["risk_on"]).loc[start:end].sum())
    return row, state, simulated


def _eligible(frame: pd.DataFrame, prefix: str) -> pd.Series:
    return (
        frame[f"{prefix}_annualized_delta"].gt(0.0)
        & frame[f"{prefix}_sharpe_delta"].gt(0.0)
        & frame[f"{prefix}_max_drawdown_improvement"].ge(-1e-12)
    )


def select_candidate(
    grid: pd.DataFrame,
    scheme: str,
    prefix: str,
) -> pd.Series:
    candidates = grid.loc[grid["scheme"].eq(scheme)].copy()
    candidates["selection_metric_gate"] = _eligible(candidates, prefix)
    candidates["selection_activity_gate"] = candidates[
        f"{prefix}_emergency_entries"
    ].gt(0)
    pool_name = "metric_and_activity"
    pool = candidates.loc[
        candidates["selection_metric_gate"]
        & candidates["selection_activity_gate"]
    ]
    if pool.empty:
        pool_name = "activity_only_fallback"
        pool = candidates.loc[candidates["selection_activity_gate"]]
    if pool.empty:
        pool_name = "metric_only_fallback"
        pool = candidates.loc[candidates["selection_metric_gate"]]
    if pool.empty:
        pool_name = "all_candidates_fallback"
        pool = candidates
    selected = pool.sort_values(
        [
            f"{prefix}_sharpe_delta",
            f"{prefix}_max_drawdown_improvement",
            f"{prefix}_annualized_delta",
            f"{prefix}_switches",
            f"{prefix}_alert_days",
            "variant_id",
        ],
        ascending=[False, False, False, True, True, True],
    ).iloc[0]
    selected["selection_pool"] = pool_name
    return selected


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _fmt_parameters(row: pd.Series) -> str:
    values = []
    for field, label in (
        ("volatility_window", "vol"),
        ("expanding_quantile", "q"),
        ("cap_trigger_maximum", "cap≤"),
        ("downside_window", "down"),
        ("downside_threshold", "ret≤"),
    ):
        value = row.get(field)
        if pd.notna(value):
            values.append(f"{label}{value:g}")
    return ", ".join(values) if values else "frozen"


def _markdown_table(rows: pd.DataFrame, prefix: str) -> str:
    lines = [
        "|方案|参数|年化|Sharpe|MDD|年化差|Sharpe差|MDD改善|2023收益|2024-截止收益|",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows.itertuples(index=False):
        record = row._asdict()
        series = pd.Series(record)
        lines.append(
            "|{scheme}|{params}|{annual:.2%}|{sharpe:.3f}|{mdd:.2%}|"
            "{annual_delta:+.2%}|{sharpe_delta:+.3f}|{mdd_delta:+.2%}|"
            "{y2023:+.2%}|{later:+.2%}|".format(
                scheme=record["scheme"],
                params=_fmt_parameters(series),
                annual=float(record[f"{prefix}_annualized_return_252"]),
                sharpe=float(record[f"{prefix}_sharpe"]),
                mdd=float(record[f"{prefix}_max_drawdown"]),
                annual_delta=float(record[f"{prefix}_annualized_delta"]),
                sharpe_delta=float(record[f"{prefix}_sharpe_delta"]),
                mdd_delta=float(record[f"{prefix}_max_drawdown_improvement"]),
                y2023=float(record["validation_2023_total_return"]),
                later=float(record["evaluation_2024_cutoff_total_return"]),
            )
        )
    return "\n".join(lines)


def _markdown_incremental_table(rows: pd.DataFrame, prefix: str) -> str:
    lines = [
        "|方案|参数|年化差 vs 无cap|Sharpe差 vs 无cap|MDD改善 vs 无cap|紧急切入次数|Defender占比|",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for record in rows.to_dict("records"):
        series = pd.Series(record)
        lines.append(
            "|{scheme}|{params}|{annual_delta:+.2%}|{sharpe_delta:+.3f}|"
            "{mdd_delta:+.2%}|{entries}|{share:.1%}|".format(
                scheme=record["scheme"],
                params=_fmt_parameters(series),
                annual_delta=float(
                    record[f"{prefix}_annualized_delta_vs_no_cap"]
                ),
                sharpe_delta=float(record[f"{prefix}_sharpe_delta_vs_no_cap"]),
                mdd_delta=float(
                    record[f"{prefix}_max_drawdown_improvement_vs_no_cap"]
                ),
                entries=int(record["emergency_entries"]),
                share=float(record["defender_share"]),
            )
        )
    return "\n".join(lines)


def _neighbor_mask(candidates: pd.DataFrame, selected: pd.Series) -> pd.Series:
    dimensions = {
        "volatility_window": VOLATILITY_WINDOWS,
        "expanding_quantile": EXPANDING_QUANTILES,
        "cap_trigger_maximum": CAP_TRIGGER_MAXIMUMS,
        "downside_window": DOWNSIDE_WINDOWS,
        "downside_threshold": DOWNSIDE_THRESHOLDS,
    }
    mask = pd.Series(True, index=candidates.index)
    for field, values in dimensions.items():
        selected_value = selected.get(field)
        if pd.isna(selected_value):
            continue
        position = next(
            index
            for index, value in enumerate(values)
            if np.isclose(float(value), float(selected_value))
        )
        allowed = values[max(0, position - 1) : position + 2]
        mask &= candidates[field].isin(allowed)
    return mask


def run_experiment(
    root: Path,
    defender_dir: Path,
    final_output: Path,
    end: date,
) -> None:
    final_output.parent.mkdir(parents=True, exist_ok=True)
    git_status_before = _git(root, "status", "--short").splitlines()
    stage = Path(
        tempfile.mkdtemp(
            prefix=f".{final_output.name}.staging-",
            dir=final_output.parent,
        )
    )
    inputs = build_inputs(
        root,
        defender_dir / "relative_defender_rotation_switch_returns.csv",
        end,
    )
    bundle = load_defender_bundle(defender_dir, end)
    calendar = inputs.calendar
    exact_momentum = inputs.momentum[HELD_RETURN].astype(float)
    slow = slow_regime_at_open(
        inputs.risk_close,
        calendar,
        SLOW_PARAMS.lookback,
        SLOW_PARAMS.risk_on_threshold,
    )
    previous_asset = momentum_asset_at_previous_close(
        inputs.momentum_result,
        calendar,
    )

    ohlc = {asset: _load_ohlc(asset, end) for asset in MOMENTUM_ASSETS}
    volatility_cache = {
        (asset, window): rogers_satchell_volatility(prices, window)
        for asset, prices in ohlc.items()
        for window in VOLATILITY_WINDOWS
    }
    cap_open_cache: dict[tuple[str, int, float], pd.Series] = {}
    for asset in MOMENTUM_ASSETS:
        for window in VOLATILITY_WINDOWS:
            volatility = volatility_cache[(asset, window)]
            for quantile in EXPANDING_QUANTILES:
                cap = expanding_volatility_cap(volatility, quantile)["cap"]
                cap_open_cache[(asset, window, quantile)] = asof_previous_close(
                    cap,
                    calendar,
                ).fillna(1.0)
    downside_open_cache = {
        (asset, window): asof_previous_close(
            prices["close"] / prices["close"].shift(window) - 1.0,
            calendar,
        )
        for asset, prices in ohlc.items()
        for window in DOWNSIDE_WINDOWS
    }

    alerts: dict[str, pd.Series] = {}
    specs: list[AlertSpec] = []

    no_cap = AlertSpec("N", "No cap")
    specs.append(no_cap)
    alerts[no_cap.variant_id()] = pd.Series(False, index=calendar)

    frozen_cap = pd.to_numeric(
        indicator_at_effective_open(
            bundle.indicators,
            "signal_volatility_cap",
            calendar,
        ),
        errors="coerce",
    ).fillna(1.0)
    current = AlertSpec(
        "A",
        "Frozen Defender 512890 cap",
        volatility_window=20,
        expanding_quantile=0.80,
        cap_trigger_maximum=0.80,
    )
    specs.append(current)
    alerts[current.variant_id()] = frozen_cap.le(0.80)

    for window in VOLATILITY_WINDOWS:
        for quantile in EXPANDING_QUANTILES:
            csi300_cap = cap_open_cache[("510300.SH", window, quantile)]
            held_cap = choose_by_asset(
                {
                    asset: cap_open_cache[(asset, window, quantile)]
                    for asset in MOMENTUM_ASSETS
                },
                previous_asset,
            )
            for cap_max in CAP_TRIGGER_MAXIMUMS:
                spec_b = AlertSpec(
                    "B",
                    "CSI300 volatility cap",
                    window,
                    quantile,
                    cap_max,
                )
                specs.append(spec_b)
                alerts[spec_b.variant_id()] = csi300_cap.le(cap_max)

                spec_c = AlertSpec(
                    "C",
                    "Previous-close Momentum asset cap",
                    window,
                    quantile,
                    cap_max,
                )
                specs.append(spec_c)
                alerts[spec_c.variant_id()] = held_cap.le(cap_max)

                for downside_window in DOWNSIDE_WINDOWS:
                    held_return = choose_by_asset(
                        {
                            asset: downside_open_cache[(asset, downside_window)]
                            for asset in MOMENTUM_ASSETS
                        },
                        previous_asset,
                    )
                    for downside_threshold in DOWNSIDE_THRESHOLDS:
                        spec_d = AlertSpec(
                            "D",
                            "CSI300 cap plus held-asset downside",
                            window,
                            quantile,
                            cap_max,
                            downside_window,
                            downside_threshold,
                        )
                        specs.append(spec_d)
                        alerts[spec_d.variant_id()] = csi300_cap.le(cap_max) & held_return.le(
                            downside_threshold
                        )

    rows: list[dict[str, object]] = []
    for spec in specs:
        row, _, _ = evaluate_alert(
            spec,
            alerts[spec.variant_id()],
            slow,
            inputs.momentum,
            inputs.defender,
            exact_momentum,
        )
        rows.append(row)
    grid = pd.DataFrame(rows)
    no_cap_row = grid.loc[grid["scheme"].eq("N")].iloc[0]
    for prefix in PERIODS:
        grid[f"{prefix}_total_return_delta_vs_no_cap"] = (
            grid[f"{prefix}_total_return"]
            - float(no_cap_row[f"{prefix}_total_return"])
        )
        grid[f"{prefix}_annualized_delta_vs_no_cap"] = (
            grid[f"{prefix}_annualized_return_252"]
            - float(no_cap_row[f"{prefix}_annualized_return_252"])
        )
        grid[f"{prefix}_sharpe_delta_vs_no_cap"] = (
            grid[f"{prefix}_sharpe"] - float(no_cap_row[f"{prefix}_sharpe"])
        )
        grid[f"{prefix}_max_drawdown_improvement_vs_no_cap"] = (
            grid[f"{prefix}_max_drawdown"]
            - float(no_cap_row[f"{prefix}_max_drawdown"])
        )
    grid.to_csv(stage / "abcd_parameter_grid.csv", index=False)

    selected_records: list[dict[str, object]] = []
    for scheme in ("A", "B", "C", "D"):
        development = select_candidate(grid, scheme, "development_2019_2022")
        selected_records.append(
            {**development.to_dict(), "selection": "development_selected"}
        )
        oracle = select_candidate(grid, scheme, "full")
        selected_records.append({**oracle.to_dict(), "selection": "full_oracle"})
    no_cap_row = grid.loc[grid["scheme"].eq("N")].iloc[0]
    selected_records.append({**no_cap_row.to_dict(), "selection": "baseline"})
    selected = pd.DataFrame(selected_records)
    selected.to_csv(stage / "selected_candidates.csv", index=False)

    report_config = {
        "strategy_name": "momentum_volatility_signal_abcd",
        **asdict(SLOW_PARAMS),
        "selection_period": "2019-01-18 through 2022-12-30",
        "research_cutoff": end.isoformat(),
    }
    selected_daily_rows: list[pd.DataFrame] = []
    development_selected = selected.loc[
        selected["selection"].eq("development_selected")
    ].copy()
    report_names = {
        "A": "A_frozen_512890_cap_vs_momentum.html",
        "B": "B_csi300_cap_dev_selected_vs_momentum.html",
        "C": "C_held_asset_cap_dev_selected_vs_momentum.html",
        "D": "D_confirmed_cap_dev_selected_vs_momentum.html",
        "N": "N_no_cap_vs_momentum.html",
    }
    report_selected = pd.concat(
        [
            development_selected,
            selected.loc[selected["selection"].eq("baseline")],
        ],
        ignore_index=True,
    )
    event_records: list[dict[str, object]] = []
    for _, chosen in report_selected.iterrows():
        variant_id = str(chosen["variant_id"])
        scheme = str(chosen["scheme"])
        spec = next(item for item in specs if item.variant_id() == variant_id)
        _, state, simulated = evaluate_alert(
            spec,
            alerts[variant_id],
            slow,
            inputs.momentum,
            inputs.defender,
            exact_momentum,
        )
        daily = state.join(simulated.drop(columns=["risk_on"]))
        daily["emergency_alert"] = alerts[variant_id]
        daily["momentum_asset_at_previous_close"] = previous_asset
        daily["momentum_exact_return"] = exact_momentum
        daily["scheme"] = scheme
        daily["variant_id"] = variant_id
        daily.index.name = "date"
        daily.to_csv(stage / f"selected_{scheme}_daily.csv")
        selected_daily_rows.append(daily)
        _generate_standard_report(
            simulated["return"],
            exact_momentum,
            "Original Momentum Strategy",
            stage / report_names[scheme],
            {**report_config, **asdict(spec)},
        )
        emergency_entry = state["state_changed"].astype(bool) & state[
            "state_reason"
        ].eq("emergency_exit")
        for event_name, (start, finish) in EVENT_WINDOWS.items():
            event_return = simulated["return"].loc[start:finish]
            event_metrics = performance(event_return)
            event_records.append(
                {
                    "event": event_name,
                    "start": start.date().isoformat(),
                    "end": finish.date().isoformat(),
                    "scheme": scheme,
                    "variant_id": variant_id,
                    "total_return": event_metrics["total_return"],
                    "max_drawdown": event_metrics["max_drawdown"],
                    "alert_days": int(alerts[variant_id].loc[start:finish].sum()),
                    "emergency_entries": int(
                        emergency_entry.loc[start:finish].sum()
                    ),
                    "defender_days": int((~state["risk_on"]).loc[start:finish].sum()),
                }
            )

    for event_name, (start, finish) in EVENT_WINDOWS.items():
        event_metrics = performance(exact_momentum.loc[start:finish])
        event_records.append(
            {
                "event": event_name,
                "start": start.date().isoformat(),
                "end": finish.date().isoformat(),
                "scheme": "M",
                "variant_id": "Original Momentum Strategy",
                "total_return": event_metrics["total_return"],
                "max_drawdown": event_metrics["max_drawdown"],
                "alert_days": 0,
                "emergency_entries": 0,
                "defender_days": 0,
            }
        )
    event_comparison = pd.DataFrame(event_records).sort_values(["event", "scheme"])
    event_comparison.to_csv(stage / "event_window_comparison.csv", index=False)

    robustness_records: list[dict[str, object]] = []
    for scheme in ("A", "B", "C", "D"):
        candidates = grid.loc[grid["scheme"].eq(scheme)].copy()
        chosen = development_selected.loc[
            development_selected["scheme"].eq(scheme)
        ].iloc[0]
        neighbors = candidates.loc[_neighbor_mask(candidates, chosen)]
        full_vs_momentum = _eligible(candidates, "full")
        full_vs_no_cap = (
            candidates["full_annualized_delta_vs_no_cap"].gt(0.0)
            & candidates["full_sharpe_delta_vs_no_cap"].gt(0.0)
            & candidates["full_max_drawdown_improvement_vs_no_cap"].ge(-1e-12)
        )
        neighbor_vs_momentum = _eligible(neighbors, "full")
        neighbor_vs_no_cap = (
            neighbors["full_annualized_delta_vs_no_cap"].gt(0.0)
            & neighbors["full_sharpe_delta_vs_no_cap"].gt(0.0)
            & neighbors["full_max_drawdown_improvement_vs_no_cap"].ge(-1e-12)
        )
        robustness_records.append(
            {
                "scheme": scheme,
                "candidate_count": len(candidates),
                "development_active_count": int(
                    candidates["development_2019_2022_emergency_entries"].gt(0).sum()
                ),
                "development_metric_and_activity_count": int(
                    (
                        _eligible(candidates, "development_2019_2022")
                        & candidates[
                            "development_2019_2022_emergency_entries"
                        ].gt(0)
                    ).sum()
                ),
                "validation_2023_no_alert_count": int(
                    candidates["validation_2023_alert_days"].eq(0).sum()
                ),
                "full_triple_vs_momentum_count": int(full_vs_momentum.sum()),
                "full_triple_vs_no_cap_count": int(full_vs_no_cap.sum()),
                "neighbor_count": len(neighbors),
                "neighbor_full_triple_vs_momentum_count": int(
                    neighbor_vs_momentum.sum()
                ),
                "neighbor_full_triple_vs_no_cap_count": int(
                    neighbor_vs_no_cap.sum()
                ),
                "neighbor_full_annualized_return_min": float(
                    neighbors["full_annualized_return_252"].min()
                ),
                "neighbor_full_annualized_return_median": float(
                    neighbors["full_annualized_return_252"].median()
                ),
                "neighbor_full_annualized_return_max": float(
                    neighbors["full_annualized_return_252"].max()
                ),
                "neighbor_full_sharpe_min": float(neighbors["full_sharpe"].min()),
                "neighbor_full_sharpe_median": float(
                    neighbors["full_sharpe"].median()
                ),
                "neighbor_full_sharpe_max": float(neighbors["full_sharpe"].max()),
                "neighbor_full_max_drawdown_min": float(
                    neighbors["full_max_drawdown"].min()
                ),
                "neighbor_full_max_drawdown_median": float(
                    neighbors["full_max_drawdown"].median()
                ),
                "neighbor_full_max_drawdown_max": float(
                    neighbors["full_max_drawdown"].max()
                ),
                "development_selected_variant": chosen["variant_id"],
            }
        )
    robustness = pd.DataFrame(robustness_records)
    robustness.to_csv(stage / "scheme_robustness_summary.csv", index=False)

    baseline_metrics = performance(exact_momentum)
    baseline_summary = pd.DataFrame(
        [{"strategy": "Original Momentum Strategy", **baseline_metrics}]
    )
    baseline_summary.to_csv(stage / "momentum_baseline_metrics.csv", index=False)

    dev_table = development_selected.sort_values("scheme")
    oracle_table = selected.loc[selected["selection"].eq("full_oracle")].sort_values(
        "scheme"
    )
    report = f"""# Momentum × Defender 波动信号 A/B/C/D 寻参研究

## 研究设计

- 所有方案保留相同的40日沪深300慢门控、2.5%阈值、30日最短持有期、下一开盘执行和相同交易费用。
- A：冻结的512890 Defender cap；B：510300自身RS波动cap；C：Momentum上一收盘实际持仓ETF的自身RS波动cap；D：B且上一收盘持仓ETF的短期收益低于阈值。
- B/C/D搜索：波动窗口 {VOLATILITY_WINDOWS}、严格滞后扩展分位 {EXPANDING_QUANTILES}、触发档位 cap≤{CAP_TRIGGER_MAXIMUMS}；D另搜索持仓收益窗口 {DOWNSIDE_WINDOWS} 和收益阈值 {DOWNSIDE_THRESHOLDS}。
- 主选择只使用2019-01-18至2022-12-30，且要求该紧急信号在开发期至少实际触发过一次Momentum→Defender切换；随后在“年化、Sharpe均高于Momentum且MDD不差于Momentum”的候选中优先最大化Sharpe。若交集为空，依次退回有实际触发、只满足指标、全体候选，并在selection_pool字段留痕。2023单列验证；2024以后因研究者已知2024事件，不能称为真正未观察样本。
- 全样本oracle只说明事后优化上限，不能作为可部署参数。

## 2019-2022选择出的参数：全样本表现

{_markdown_table(dev_table, 'full')}

## cap相对“保留慢门控、取消紧急cap”的增量贡献

{_markdown_incremental_table(dev_table, 'full')}

## 全样本事后最优：仅作过拟合参照

{_markdown_table(oracle_table, 'full')}

## 解释边界

- B仍可能用中国宽基风险错误外推海外或黄金持仓；C减少跨资产错配，但会把不同资产各自的波动分位直接视为可比；D增加方向确认但引入更多参数和多重检验自由度。
- 所有结果都来自同一历史数据库；开发期选择不能消除本研究方向由2023/2024已知事件启发的后见偏差。
- 报告必须同时关注2023误防守、2024防守、全样本MDD和参数邻域，而不能只看最高Sharpe。
- `scheme_robustness_summary.csv` 的“邻域”定义为：每个已寻参维度最多移动一个离散档位；它用来识别孤立最优点，但不能替代真正的未来样本。
"""
    (stage / "research_report.md").write_text(report, encoding="utf-8")

    code_files = [
        root / "research/run_momentum_volatility_signal_abcd.py",
        root / "research/momentum_defender_occam.py",
        root / "research/run_momentum_defender_occam.py",
        root / "research/tests/test_momentum_volatility_signal_abcd.py",
        root / "backtest/report.py",
        root / "backtest/runner.py",
    ]
    input_files = [
        defender_dir / "relative_defender_rotation_switch_returns.csv",
        defender_dir / "relative_defender_rotation_daily_indicators.csv",
        root / "strategy/configs/quality_momentum_top1.yaml",
        *[root / "data/db" / f"{asset}.parquet" for asset in MOMENTUM_ASSETS],
    ]
    manifest = {
        "experiment": "momentum_volatility_signal_abcd",
        "generated_on": date.today().isoformat(),
        "research_cutoff": end.isoformat(),
        "calendar_rows": len(calendar),
        "slow_parameters": asdict(SLOW_PARAMS),
        "parameter_grid": {
            "volatility_windows": VOLATILITY_WINDOWS,
            "expanding_quantiles": EXPANDING_QUANTILES,
            "cap_trigger_maximums": CAP_TRIGGER_MAXIMUMS,
            "downside_windows": DOWNSIDE_WINDOWS,
            "downside_thresholds": DOWNSIDE_THRESHOLDS,
            "candidate_count": len(grid),
        },
        "periods": {
            key: [start.date().isoformat(), finish.date().isoformat()]
            for key, (start, finish) in PERIODS.items()
        },
        "git_commit": _git(root, "rev-parse", "HEAD"),
        "git_status_short": git_status_before,
        "inputs": [
            {"path": str(path), "sha256": _sha256(path)} for path in input_files
        ],
        "code_sources": [
            {"path": str(path), "sha256": _sha256(path)} for path in code_files
        ],
    }
    (stage / "experiment_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    final_output.mkdir(parents=True, exist_ok=True)
    for path in stage.iterdir():
        path.replace(final_output / path.name)
    stage.rmdir()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--defender-dir", type=Path, default=DEFAULT_DEFENDER_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--end", type=date.fromisoformat, default=DEFAULT_END)
    args = parser.parse_args()
    run_experiment(
        args.root.resolve(),
        args.defender_dir.resolve(),
        args.output.resolve(),
        args.end,
    )


if __name__ == "__main__":
    main()
