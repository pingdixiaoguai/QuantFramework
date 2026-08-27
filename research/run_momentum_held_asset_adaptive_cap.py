"""Tune asset-aware emergency volatility signals for Momentum scheme C.

The study compares the legacy one-threshold held-asset cap with two remedies:

* C1: one cap trigger severity per Momentum ETF;
* C2: one lagged expanding volatility quantile per Momentum ETF.

All parameter selection uses 2019-2022 only.  A candidate must have caused at
least one real emergency entry and must beat the no-cap fused strategy on
annualized return and Sharpe without worsening maximum drawdown.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import subprocess
import tempfile
from dataclasses import asdict, dataclass, replace
from datetime import date
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from research.momentum_defender_occam import (
    HELD_RETURN,
    MOMENTUM_ASSETS,
    OccamParams,
    build_inputs,
    load_defender_bundle,
    performance,
    simulate_switch,
    slow_regime_at_open,
)
from research.run_momentum_defender_occam import _generate_standard_report
from research.run_momentum_volatility_signal_abcd import (
    CAP_TRIGGER_MAXIMUMS,
    DEFAULT_DEFENDER_DIR,
    DEFAULT_END,
    EVENT_WINDOWS,
    EXPANDING_QUANTILES,
    PERIODS,
    VOLATILITY_WINDOWS,
    _load_ohlc,
    asof_previous_close,
    evaluate_alert,
    expanding_volatility_cap,
    momentum_asset_at_previous_close,
    rogers_satchell_volatility,
)


DEFAULT_OUTPUT = Path("experiments/20260821_momentum_held_asset_adaptive_cap")
SLOW_PARAMS = OccamParams(
    lookback=40,
    risk_on_threshold=0.025,
    min_hold_days=30,
    emergency_daily_loss=None,
)
ASSET_NAMES = {
    "510300.SH": "csi300",
    "159915.SZ": "chinext",
    "513100.SH": "nasdaq100",
    "518880.SH": "gold",
}


@dataclass(frozen=True)
class AdaptiveCSpec:
    scheme: str
    label: str
    volatility_window: int | None = None
    expanding_quantile: float | None = None
    cap_trigger_maximum: float | None = None
    cap_510300: float | None = None
    cap_159915: float | None = None
    cap_513100: float | None = None
    cap_518880: float | None = None
    q_510300: float | None = None
    q_159915: float | None = None
    q_513100: float | None = None
    q_518880: float | None = None

    def variant_id(self) -> str:
        parts = [self.scheme]
        if self.volatility_window is not None:
            parts.append(f"vw{self.volatility_window}")
        if self.expanding_quantile is not None:
            parts.append(f"q{self.expanding_quantile:.2f}")
        if self.cap_trigger_maximum is not None:
            parts.append(f"cap{self.cap_trigger_maximum:.1f}")
        for short, value in (
            ("c300", self.cap_510300),
            ("cyb", self.cap_159915),
            ("ndx", self.cap_513100),
            ("au", self.cap_518880),
            ("qc300", self.q_510300),
            ("qcyb", self.q_159915),
            ("qndx", self.q_513100),
            ("qau", self.q_518880),
        ):
            if value is not None:
                parts.append(f"{short}{value:.2f}")
        return "_".join(parts)

    def cap_thresholds(self) -> dict[str, float]:
        return {
            "510300.SH": float(self.cap_510300),
            "159915.SZ": float(self.cap_159915),
            "513100.SH": float(self.cap_513100),
            "518880.SH": float(self.cap_518880),
        }

    def asset_quantiles(self) -> dict[str, float]:
        return {
            "510300.SH": float(self.q_510300),
            "159915.SZ": float(self.q_159915),
            "513100.SH": float(self.q_513100),
            "518880.SH": float(self.q_518880),
        }


def held_asset_cap_alert(
    caps_by_asset: Mapping[str, pd.Series],
    previous_asset: pd.Series,
    thresholds: Mapping[str, float],
) -> pd.Series:
    """Compare each previous-close holding with its own cap severity."""
    alert = pd.Series(False, index=previous_asset.index)
    for asset in MOMENTUM_ASSETS:
        held = previous_asset.eq(asset)
        alert.loc[held] = (
            caps_by_asset[asset].reindex(alert.index).loc[held]
            <= float(thresholds[asset])
        )
    return alert.astype(bool)


def _no_cap_gate(frame: pd.DataFrame, prefix: str) -> pd.Series:
    return (
        frame[f"{prefix}_annualized_delta_vs_no_cap"].gt(0.0)
        & frame[f"{prefix}_sharpe_delta_vs_no_cap"].gt(0.0)
        & frame[f"{prefix}_max_drawdown_improvement_vs_no_cap"].ge(-1e-12)
    )


def select_candidate(frame: pd.DataFrame, scheme: str, prefix: str) -> pd.Series:
    candidates = frame.loc[frame["scheme"].eq(scheme)].copy()
    candidates["selection_metric_gate"] = _no_cap_gate(candidates, prefix)
    candidates["selection_activity_gate"] = candidates[
        f"{prefix}_emergency_entries"
    ].gt(0)
    pool_name = "beats_no_cap_and_active"
    pool = candidates.loc[
        candidates["selection_metric_gate"]
        & candidates["selection_activity_gate"]
    ]
    if pool.empty:
        pool_name = "active_only_fallback"
        pool = candidates.loc[candidates["selection_activity_gate"]]
    if pool.empty:
        pool_name = "all_candidates_fallback"
        pool = candidates
    selected = pool.sort_values(
        [
            f"{prefix}_sharpe_delta_vs_no_cap",
            f"{prefix}_max_drawdown_improvement_vs_no_cap",
            f"{prefix}_annualized_delta_vs_no_cap",
            f"{prefix}_emergency_entries",
            f"{prefix}_switches",
            "variant_id",
        ],
        ascending=[False, False, False, True, True, True],
    ).iloc[0]
    selected["selection_pool"] = pool_name
    return selected


def _add_no_cap_deltas(grid: pd.DataFrame) -> pd.DataFrame:
    result = grid.copy()
    baseline = result.loc[result["scheme"].eq("N")].iloc[0]
    for prefix in PERIODS:
        for metric in ("total_return", "annualized_return_252", "sharpe"):
            suffix = {
                "total_return": "total_return_delta_vs_no_cap",
                "annualized_return_252": "annualized_delta_vs_no_cap",
                "sharpe": "sharpe_delta_vs_no_cap",
            }[metric]
            result[f"{prefix}_{suffix}"] = (
                result[f"{prefix}_{metric}"] - float(baseline[f"{prefix}_{metric}"])
            )
        result[f"{prefix}_max_drawdown_improvement_vs_no_cap"] = (
            result[f"{prefix}_max_drawdown"]
            - float(baseline[f"{prefix}_max_drawdown"])
        )
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()


def _format_parameters(row: pd.Series) -> str:
    if row["scheme"] == "C1":
        return (
            f"vw{int(row.volatility_window)}, q{row.expanding_quantile:.2f}; "
            f"300≤{row.cap_510300:.1f}, 创业板≤{row.cap_159915:.1f}, "
            f"纳指≤{row.cap_513100:.1f}, 黄金≤{row.cap_518880:.1f}"
        )
    if row["scheme"] == "C2":
        return (
            f"vw{int(row.volatility_window)}, cap<1; "
            f"300 q{row.q_510300:.2f}, 创业板 q{row.q_159915:.2f}, "
            f"纳指 q{row.q_513100:.2f}, 黄金 q{row.q_518880:.2f}"
        )
    if row["scheme"] == "C0":
        return (
            f"vw{int(row.volatility_window)}, q{row.expanding_quantile:.2f}, "
            f"cap≤{row.cap_trigger_maximum:.1f}"
        )
    return "无紧急cap"


def _neighbor_mask(candidates: pd.DataFrame, selected: pd.Series) -> pd.Series:
    dimensions = {
        "volatility_window": VOLATILITY_WINDOWS,
        "expanding_quantile": EXPANDING_QUANTILES,
        "cap_trigger_maximum": CAP_TRIGGER_MAXIMUMS,
        "cap_510300": CAP_TRIGGER_MAXIMUMS,
        "cap_159915": CAP_TRIGGER_MAXIMUMS,
        "cap_513100": CAP_TRIGGER_MAXIMUMS,
        "cap_518880": CAP_TRIGGER_MAXIMUMS,
        "q_510300": EXPANDING_QUANTILES,
        "q_159915": EXPANDING_QUANTILES,
        "q_513100": EXPANDING_QUANTILES,
        "q_518880": EXPANDING_QUANTILES,
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


def _markdown_table(rows: pd.DataFrame) -> str:
    lines = [
        "|方案|参数|年化|Sharpe|MDD|年化差vs无cap|Sharpe差vs无cap|MDD改善vs无cap|2023收益|",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in rows.iterrows():
        scheme_label = (
            "C-legacy" if row.get("selection") == "legacy_C_reference" else row.scheme
        )
        lines.append(
            f"|{scheme_label}|{_format_parameters(row)}|"
            f"{row.full_annualized_return_252:.2%}|{row.full_sharpe:.3f}|"
            f"{row.full_max_drawdown:.2%}|"
            f"{row.full_annualized_delta_vs_no_cap:+.2%}|"
            f"{row.full_sharpe_delta_vs_no_cap:+.3f}|"
            f"{row.full_max_drawdown_improvement_vs_no_cap:+.2%}|"
            f"{row.validation_2023_total_return:+.2%}|"
        )
    return "\n".join(lines)


def _markdown_robustness_table(rows: pd.DataFrame) -> str:
    lines = [
        "|方案|候选数|开发期有效且胜无cap|邻域数|邻域全样本胜无cap|邻域年化范围|邻域Sharpe范围|邻域MDD范围|",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in rows.iterrows():
        lines.append(
            f"|{row.scheme}|{int(row.candidate_count)}|"
            f"{int(row.development_beats_no_cap_and_active_count)}|"
            f"{int(row.neighbor_count)}|{int(row.neighbor_full_beats_no_cap_count)}|"
            f"{row.neighbor_full_annualized_return_min:.2%}–"
            f"{row.neighbor_full_annualized_return_max:.2%}|"
            f"{row.neighbor_full_sharpe_min:.3f}–{row.neighbor_full_sharpe_max:.3f}|"
            f"{row.neighbor_full_max_drawdown_min:.2%}–"
            f"{row.neighbor_full_max_drawdown_max:.2%}|"
        )
    return "\n".join(lines)


def run_experiment(
    root: Path,
    defender_dir: Path,
    final_output: Path,
    end: date,
) -> None:
    final_output.parent.mkdir(parents=True, exist_ok=True)
    git_status_before = _git(root, "status", "--short").splitlines()
    stage = Path(
        tempfile.mkdtemp(prefix=f".{final_output.name}.staging-", dir=final_output.parent)
    )

    inputs = build_inputs(
        root,
        defender_dir / "relative_defender_rotation_switch_returns.csv",
        end,
    )
    bundle = load_defender_bundle(defender_dir, end)
    del bundle  # The adaptive study needs Defender returns, not its 512890 signal.
    calendar = inputs.calendar
    exact_momentum = inputs.momentum[HELD_RETURN].astype(float)
    slow = slow_regime_at_open(
        inputs.risk_close,
        calendar,
        SLOW_PARAMS.lookback,
        SLOW_PARAMS.risk_on_threshold,
    )
    previous_asset = momentum_asset_at_previous_close(inputs.momentum_result, calendar)

    ohlc = {asset: _load_ohlc(asset, end) for asset in MOMENTUM_ASSETS}
    cap_cache: dict[tuple[str, int, float], pd.Series] = {}
    for asset, prices in ohlc.items():
        for window in VOLATILITY_WINDOWS:
            volatility = rogers_satchell_volatility(prices, window)
            for quantile in EXPANDING_QUANTILES:
                close_cap = expanding_volatility_cap(volatility, quantile)["cap"]
                cap_cache[(asset, window, quantile)] = asof_previous_close(
                    close_cap, calendar
                ).fillna(1.0)

    no_cap = AdaptiveCSpec("N", "No emergency cap")
    specs: list[AdaptiveCSpec] = [no_cap]

    for window, quantile, cap_maximum in itertools.product(
        VOLATILITY_WINDOWS,
        EXPANDING_QUANTILES,
        CAP_TRIGGER_MAXIMUMS,
    ):
        specs.append(
            AdaptiveCSpec(
                "C0",
                "Global held-asset cap",
                volatility_window=window,
                expanding_quantile=quantile,
                cap_trigger_maximum=cap_maximum,
            )
        )

    for window, quantile in itertools.product(
        VOLATILITY_WINDOWS, EXPANDING_QUANTILES
    ):
        for thresholds in itertools.product(CAP_TRIGGER_MAXIMUMS, repeat=4):
            specs.append(
                AdaptiveCSpec(
                    "C1",
                    "Asset-specific cap thresholds",
                    volatility_window=window,
                    expanding_quantile=quantile,
                    cap_510300=thresholds[0],
                    cap_159915=thresholds[1],
                    cap_513100=thresholds[2],
                    cap_518880=thresholds[3],
                )
            )

    for window in VOLATILITY_WINDOWS:
        for quantiles in itertools.product(EXPANDING_QUANTILES, repeat=4):
            specs.append(
                AdaptiveCSpec(
                    "C2",
                    "Asset-specific volatility quantiles",
                    volatility_window=window,
                    cap_trigger_maximum=0.8,
                    q_510300=quantiles[0],
                    q_159915=quantiles[1],
                    q_513100=quantiles[2],
                    q_518880=quantiles[3],
                )
            )

    def build_alert(spec: AdaptiveCSpec) -> pd.Series:
        if spec.scheme == "N":
            return pd.Series(False, index=calendar)
        if spec.scheme == "C0":
            caps = {
                asset: cap_cache[
                    (asset, int(spec.volatility_window), float(spec.expanding_quantile))
                ]
                for asset in MOMENTUM_ASSETS
            }
            thresholds = {
                asset: float(spec.cap_trigger_maximum) for asset in MOMENTUM_ASSETS
            }
            return held_asset_cap_alert(caps, previous_asset, thresholds)
        if spec.scheme == "C1":
            caps = {
                asset: cap_cache[
                    (asset, int(spec.volatility_window), float(spec.expanding_quantile))
                ]
                for asset in MOMENTUM_ASSETS
            }
            return held_asset_cap_alert(caps, previous_asset, spec.cap_thresholds())
        if spec.scheme == "C2":
            caps = {
                asset: cap_cache[
                    (
                        asset,
                        int(spec.volatility_window),
                        spec.asset_quantiles()[asset],
                    )
                ]
                for asset in MOMENTUM_ASSETS
            }
            return held_asset_cap_alert(
                caps,
                previous_asset,
                {asset: 0.8 for asset in MOMENTUM_ASSETS},
            )
        raise ValueError(f"unsupported scheme: {spec.scheme}")

    rows: list[dict[str, object]] = []
    for spec in specs:
        row, _, _ = evaluate_alert(
            spec,
            build_alert(spec),
            slow,
            inputs.momentum,
            inputs.defender,
            exact_momentum,
        )
        rows.append(row)
    grid = _add_no_cap_deltas(pd.DataFrame(rows))
    grid.to_csv(stage / "adaptive_c_parameter_grid.csv", index=False)

    selected_records: list[dict[str, object]] = []
    for scheme in ("C0", "C1", "C2"):
        development = select_candidate(grid, scheme, "development_2019_2022")
        selected_records.append(
            {**development.to_dict(), "selection": "development_selected"}
        )
        oracle = select_candidate(grid, scheme, "full")
        selected_records.append({**oracle.to_dict(), "selection": "full_oracle"})

    legacy_id = "C0_vw20_q0.70_cap0.8"
    legacy = grid.loc[grid["variant_id"].eq(legacy_id)].iloc[0]
    selected_records.append({**legacy.to_dict(), "selection": "legacy_C_reference"})
    baseline = grid.loc[grid["scheme"].eq("N")].iloc[0]
    selected_records.append({**baseline.to_dict(), "selection": "baseline"})
    selected = pd.DataFrame(selected_records)
    selected.to_csv(stage / "selected_candidates.csv", index=False)

    development_selected = selected.loc[
        selected["selection"].eq("development_selected")
    ].copy()

    robustness_records: list[dict[str, object]] = []
    for scheme in ("C0", "C1", "C2"):
        candidates = grid.loc[grid["scheme"].eq(scheme)].copy()
        chosen = development_selected.loc[
            development_selected["scheme"].eq(scheme)
        ].iloc[0]
        neighbors = candidates.loc[_neighbor_mask(candidates, chosen)]
        robustness_records.append(
            {
                "scheme": scheme,
                "candidate_count": len(candidates),
                "development_active_count": int(
                    candidates["development_2019_2022_emergency_entries"].gt(0).sum()
                ),
                "development_beats_no_cap_and_active_count": int(
                    (
                        _no_cap_gate(candidates, "development_2019_2022")
                        & candidates[
                            "development_2019_2022_emergency_entries"
                        ].gt(0)
                    ).sum()
                ),
                "full_beats_no_cap_count": int(_no_cap_gate(candidates, "full").sum()),
                "neighbor_count": len(neighbors),
                "neighbor_full_beats_no_cap_count": int(
                    _no_cap_gate(neighbors, "full").sum()
                ),
                "neighbor_2023_no_alert_count": int(
                    neighbors["validation_2023_alert_days"].eq(0).sum()
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
    robustness.to_csv(stage / "robustness_summary.csv", index=False)

    selected_c2_row = development_selected.loc[
        development_selected["scheme"].eq("C2")
    ].iloc[0]
    selected_c2_spec = next(
        item for item in specs if item.variant_id() == selected_c2_row["variant_id"]
    )
    sensitivity_records: list[dict[str, object]] = []
    sensitivity_dimensions = {
        "volatility_window": VOLATILITY_WINDOWS,
        "q_510300": EXPANDING_QUANTILES,
        "q_159915": EXPANDING_QUANTILES,
        "q_513100": EXPANDING_QUANTILES,
        "q_518880": EXPANDING_QUANTILES,
    }
    sensitivity_specs = [("selected", selected_c2_spec)]
    for field, values in sensitivity_dimensions.items():
        current = float(getattr(selected_c2_spec, field))
        position = next(
            index
            for index, value in enumerate(values)
            if np.isclose(float(value), current)
        )
        for neighbor_position in (position - 1, position + 1):
            if 0 <= neighbor_position < len(values):
                neighbor_value = values[neighbor_position]
                sensitivity_specs.append(
                    (
                        f"{field}:{current:g}->{neighbor_value:g}",
                        replace(selected_c2_spec, **{field: neighbor_value}),
                    )
                )
    for perturbation, spec in sensitivity_specs:
        row = grid.loc[grid["variant_id"].eq(spec.variant_id())].iloc[0]
        sensitivity_records.append(
            {
                "perturbation": perturbation,
                "variant_id": spec.variant_id(),
                "development_annualized_return_252": row[
                    "development_2019_2022_annualized_return_252"
                ],
                "development_sharpe": row["development_2019_2022_sharpe"],
                "development_max_drawdown": row[
                    "development_2019_2022_max_drawdown"
                ],
                "full_annualized_return_252": row["full_annualized_return_252"],
                "full_sharpe": row["full_sharpe"],
                "full_max_drawdown": row["full_max_drawdown"],
                "full_annualized_delta_vs_no_cap": row[
                    "full_annualized_delta_vs_no_cap"
                ],
                "full_sharpe_delta_vs_no_cap": row["full_sharpe_delta_vs_no_cap"],
                "full_max_drawdown_improvement_vs_no_cap": row[
                    "full_max_drawdown_improvement_vs_no_cap"
                ],
                "validation_2023_total_return": row["validation_2023_total_return"],
                "evaluation_2024_cutoff_total_return": row[
                    "evaluation_2024_cutoff_total_return"
                ],
                "emergency_entries": row["emergency_entries"],
            }
        )
    pd.DataFrame(sensitivity_records).to_csv(
        stage / "C2_one_at_a_time_sensitivity.csv", index=False
    )

    candidates_to_render = pd.concat(
        [
            selected.loc[selected["selection"].eq("legacy_C_reference")],
            development_selected,
            selected.loc[selected["selection"].eq("baseline")],
        ],
        ignore_index=True,
    )
    report_names = {
        legacy_id: "C_legacy_q70_vs_momentum.html",
        str(
            development_selected.loc[
                development_selected["scheme"].eq("C0"), "variant_id"
            ].iloc[0]
        ): "C0_global_balanced_vs_momentum.html",
        str(
            development_selected.loc[
                development_selected["scheme"].eq("C1"), "variant_id"
            ].iloc[0]
        ): "C1_asset_cap_vs_momentum.html",
        str(
            development_selected.loc[
                development_selected["scheme"].eq("C2"), "variant_id"
            ].iloc[0]
        ): "C2_asset_quantile_vs_momentum.html",
        "N": "N_no_cap_vs_momentum.html",
    }
    config = {
        "strategy_name": "momentum_held_asset_adaptive_cap",
        **asdict(SLOW_PARAMS),
        "selection_period": "2019-01-18 through 2022-12-30",
        "research_cutoff": end.isoformat(),
    }
    rendered: dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.Series]] = {}
    event_records: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []
    for _, chosen in candidates_to_render.iterrows():
        variant_id = str(chosen["variant_id"])
        spec = next(item for item in specs if item.variant_id() == variant_id)
        alert = build_alert(spec)
        _, state, simulated = evaluate_alert(
            spec,
            alert,
            slow,
            inputs.momentum,
            inputs.defender,
            exact_momentum,
        )
        rendered[variant_id] = (state, simulated, alert)
        daily = state.join(simulated.drop(columns=["risk_on"]))
        daily["emergency_alert"] = alert
        daily["momentum_asset_at_previous_close"] = previous_asset
        daily["momentum_exact_return"] = exact_momentum
        daily["scheme"] = chosen["scheme"]
        daily["variant_id"] = variant_id
        daily.index.name = "date"
        daily.to_csv(stage / f"{report_names[variant_id][:-5]}_daily.csv")
        _generate_standard_report(
            simulated["return"],
            exact_momentum,
            "Original Momentum Strategy",
            stage / report_names[variant_id],
            {**config, **asdict(spec)},
        )

        emergency_entry = state["state_changed"].astype(bool) & state[
            "state_reason"
        ].eq("emergency_exit")
        for period, (start, finish) in PERIODS.items():
            period_mask = previous_asset.index.to_series().between(start, finish).to_numpy()
            for asset in MOMENTUM_ASSETS:
                held = previous_asset.eq(asset) & period_mask
                held_days = int(held.sum())
                diagnostics.append(
                    {
                        "variant_id": variant_id,
                        "scheme": chosen["scheme"],
                        "period": period,
                        "asset": asset,
                        "asset_name": ASSET_NAMES[asset],
                        "held_days": held_days,
                        "alert_days_while_held": int((alert & held).sum()),
                        "alert_rate_while_held": (
                            float((alert & held).sum() / held_days) if held_days else np.nan
                        ),
                        "emergency_entries": int((emergency_entry & held).sum()),
                    }
                )
        for event, (start, finish) in EVENT_WINDOWS.items():
            measured = performance(simulated["return"].loc[start:finish])
            event_records.append(
                {
                    "event": event,
                    "start": start.date().isoformat(),
                    "end": finish.date().isoformat(),
                    "scheme": chosen["scheme"],
                    "variant_id": variant_id,
                    "total_return": measured["total_return"],
                    "max_drawdown": measured["max_drawdown"],
                    "alert_days": int(alert.loc[start:finish].sum()),
                    "emergency_entries": int(emergency_entry.loc[start:finish].sum()),
                    "defender_days": int((~state["risk_on"]).loc[start:finish].sum()),
                }
            )

    no_cap_returns = rendered["N"][1]["return"]
    adaptive_choices = development_selected.loc[
        development_selected["scheme"].isin(["C1", "C2"])
    ].copy()
    best_adaptive = adaptive_choices.sort_values(
        [
            "development_2019_2022_sharpe_delta_vs_no_cap",
            "development_2019_2022_max_drawdown_improvement_vs_no_cap",
            "development_2019_2022_annualized_delta_vs_no_cap",
        ],
        ascending=False,
    ).iloc[0]
    best_variant = str(best_adaptive["variant_id"])
    best_spec = next(item for item in specs if item.variant_id() == best_variant)
    _generate_standard_report(
        rendered[best_variant][1]["return"],
        no_cap_returns,
        "No-cap Slow-gate Fusion",
        stage / "best_adaptive_C_vs_no_cap.html",
        {**config, **asdict(best_spec)},
    )

    for event, (start, finish) in EVENT_WINDOWS.items():
        measured = performance(exact_momentum.loc[start:finish])
        event_records.append(
            {
                "event": event,
                "start": start.date().isoformat(),
                "end": finish.date().isoformat(),
                "scheme": "M",
                "variant_id": "Original Momentum Strategy",
                "total_return": measured["total_return"],
                "max_drawdown": measured["max_drawdown"],
                "alert_days": 0,
                "emergency_entries": 0,
                "defender_days": 0,
            }
        )
    pd.DataFrame(event_records).sort_values(["event", "scheme", "variant_id"]).to_csv(
        stage / "event_window_comparison.csv", index=False
    )
    pd.DataFrame(diagnostics).to_csv(stage / "asset_signal_diagnostics.csv", index=False)

    summary_rows = pd.concat(
        [
            selected.loc[selected["selection"].eq("legacy_C_reference")],
            development_selected,
        ],
        ignore_index=True,
    )
    legacy_summary = summary_rows.loc[
        summary_rows["selection"].eq("legacy_C_reference")
    ].iloc[0]
    c2_summary = development_selected.loc[
        development_selected["scheme"].eq("C2")
    ].iloc[0]
    c2_robustness = robustness.loc[robustness["scheme"].eq("C2")].iloc[0]
    c2_oracle = selected.loc[
        selected["scheme"].eq("C2") & selected["selection"].eq("full_oracle")
    ].iloc[0]
    report = f"""# 方案C：按资产校准波动紧急信号

## 设计

- C legacy：此前Sharpe优先选中的统一20日、q70、cap≤0.8。
- C0：仍使用统一参数，但改为只在开发期同时优于无cap年化、Sharpe和MDD的候选中选择，作为更公平的统一口径基准。
- C1：共同波动窗口和分位数，但510300、创业板、纳指和黄金分别搜索cap阈值0.8/0.6/0.4，共1296组。
- C2：共同波动窗口、统一触发cap<1，但四种ETF分别搜索q70/q80/q90/q95，共1024组。
- 全部信号严格使用上一收盘及更早数据，下一开盘执行；主选择期为2019-01-18至2022-12-30，且必须实际触发过紧急切入。

## 开发期选择结果的全样本表现

{_markdown_table(summary_rows)}

## 核心结论

- C2相对旧C的全样本年化提高 {(c2_summary.full_annualized_return_252 - legacy_summary.full_annualized_return_252) * 100:+.2f}个百分点，Sharpe提高 {c2_summary.full_sharpe - legacy_summary.full_sharpe:+.3f}，MDD变化 {(c2_summary.full_max_drawdown - legacy_summary.full_max_drawdown) * 100:+.2f}个百分点。
- 开发期选中的C2与全样本oracle{'完全相同' if c2_summary.variant_id == c2_oracle.variant_id else '不同'}，但相邻参数中只有 {int(c2_robustness.neighbor_full_beats_no_cap_count)}/{int(c2_robustness.neighbor_count)} 在全样本同时胜过无cap三项指标；结果改善是真实的历史回测结果，但参数峰值偏尖，不宜直接视为可部署结论。
- 开发期选中分位为沪深300q{c2_summary.q_510300:.2f}、创业板q{c2_summary.q_159915:.2f}、纳指q{c2_summary.q_513100:.2f}、黄金q{c2_summary.q_518880:.2f}。其中创业板q{c2_summary.q_159915:.2f}与q{c2_oracle.q_159915:.2f}在开发期产生完全相同收益，开发期无法识别二者优劣；不能把全样本更好的q{c2_oracle.q_159915:.2f}倒推为开发期已发现的参数。

## 参数邻域稳定性

{_markdown_robustness_table(robustness)}

## 解释边界

- C1/C2分别增加4个资产级参数，多重检验和稀疏持仓会放大过拟合风险；全样本oracle不能用于部署。
- 2023单列验证；研究方向已受2024事件启发，因此2024以后不能称为真正未观察样本。
- `asset_signal_diagnostics.csv` 应用于检查改善是否来自所有资产，还是由少数持仓期和单一事件贡献。
- `C2_one_at_a_time_sensitivity.csv` 固定其余参数，只把一个资产分位或波动窗口移动一个档位，用于检查所选点是否尖锐。
"""
    (stage / "research_report.md").write_text(report, encoding="utf-8")

    input_files = [
        defender_dir / "relative_defender_rotation_switch_returns.csv",
        root / "strategy/configs/quality_momentum_top1.yaml",
        *[root / "data/db" / f"{asset}.parquet" for asset in MOMENTUM_ASSETS],
    ]
    code_files = [
        root / "research/run_momentum_held_asset_adaptive_cap.py",
        root / "research/tests/test_momentum_held_asset_adaptive_cap.py",
        root / "research/run_momentum_volatility_signal_abcd.py",
        root / "research/momentum_defender_occam.py",
        root / "research/run_momentum_defender_occam.py",
        root / "backtest/report.py",
    ]
    manifest = {
        "experiment": "momentum_held_asset_adaptive_cap",
        "generated_on": date.today().isoformat(),
        "research_cutoff": end.isoformat(),
        "calendar_rows": len(calendar),
        "slow_parameters": asdict(SLOW_PARAMS),
        "candidate_counts": {
            "N": 1,
            "C0": len(grid.loc[grid["scheme"].eq("C0")]),
            "C1": len(grid.loc[grid["scheme"].eq("C1")]),
            "C2": len(grid.loc[grid["scheme"].eq("C2")]),
        },
        "selection_rule": "development active and triple-positive versus no-cap; maximize Sharpe, then MDD improvement, then annualized return",
        "git_commit": _git(root, "rev-parse", "HEAD"),
        "git_status_short": git_status_before,
        "inputs": [{"path": str(path), "sha256": _sha256(path)} for path in input_files],
        "code_sources": [
            {"path": str(path), "sha256": _sha256(path)} for path in code_files
        ],
    }
    (stage / "experiment_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
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
