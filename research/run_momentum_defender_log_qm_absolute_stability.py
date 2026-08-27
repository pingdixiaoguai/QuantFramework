"""Reproduce the selected absolute-stability log-QM research candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import yaml

from research.defender_curve_momentum import DEFENDER_CANDIDATE
from research.gold_min5_risk_adjusted_momentum import risk_adjusted_momentum_at_open
from research.gold_min5_risk_adjusted_momentum_w5 import (
    GoldRAQMW5Params,
    run_gold_raqm_w5,
)
from research.momentum_defender_gold_override import build_gold_override_context
from research.momentum_defender_gold_override_overfit import paired_block_bootstrap
from research.momentum_defender_log_qm_robust import (
    EmergencySpec,
    GateSpec,
    RobustSpec,
    StatePolicy,
    build_feature_bundle,
    run_robust_spec,
)
from research.momentum_defender_log_qm_switch import (
    build_fast_switch_data,
    fast_candidate_schedule,
)
from research.momentum_defender_occam import HELD_RETURN, performance
from research.run_momentum_defender_log_qm_robust import (
    _event_stress,
    _friction,
    _selected_cost_schedule,
)
from research.standard_report import generate_standard_report


DEFAULT_CONFIG = Path(
    "research/configs/momentum_defender_log_qm_absolute_stability_selected.yaml"
)
DEFAULT_OUTPUT = Path(
    "experiments/20260824_momentum_defender_log_qm_absolute_stability_candidate"
)


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_candidate(root: Path, config_path: Path, output: Path) -> dict[str, object]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    end = pd.Timestamp(config["checkpoint"]["end"]).date()
    context = build_gold_override_context(root, end=end)
    gold_metrics = risk_adjusted_momentum_at_open(context.curves, window=5)
    data = build_fast_switch_data(context, gold_metrics)
    baseline = run_gold_raqm_w5(
        context, GoldRAQMW5Params(2.20, 0.60), metrics=gold_metrics
    )
    calendar = context.calendar
    momentum_curve = (
        1.0 + context.integrated.result.inputs.momentum[HELD_RETURN].astype(float)
    ).cumprod()
    emergency_config = config["emergency"]
    features = build_feature_bundle(
        calendar,
        context.integrated.result.previous_asset,
        end=end,
        return_lookbacks=[1, int(emergency_config["negative_trend_confirmation_window"]), int(config["regime_gate"]["lookback"])],
        drawdown_windows=[int(emergency_config["volatility_window"])],
        volatility_windows=[int(emergency_config["volatility_window"])],
        quantiles=[float(emergency_config["quantile"])],
        histories=["expanding_strict_lag"],
        minimum_history=20,
        rolling_history=504,
        momentum_curve=momentum_curve,
        defender_curve=context.curves[DEFENDER_CANDIDATE],
    )
    policy_config = config["state_policy"]
    policy = StatePolicy(
        "momentum_20_defender_40",
        int(policy_config["min_momentum_days"]),
        int(policy_config["min_defender_days"]),
        int(policy_config["risk_off_confirmation"]),
        int(policy_config["risk_on_confirmation"]),
    )
    gate_config = config["regime_gate"]
    gate = GateSpec(
        str(gate_config["rule"]),
        int(gate_config["lookback"]),
        float(gate_config["threshold"]),
        2,
        policy,
    )
    emergency = EmergencySpec(
        "downside_vol",
        int(emergency_config["volatility_window"]),
        quantile=float(emergency_config["quantile"]),
        history="expanding_strict_lag",
    )
    spec = RobustSpec(gate, emergency)
    result = run_robust_spec(
        data,
        features,
        spec,
        negative_trend_window=int(
            emergency_config["negative_trend_confirmation_window"]
        ),
    )
    returns = pd.Series(result.returns, index=calendar, name="return")
    baseline_returns = baseline.daily["return"].astype(float)
    target = pd.Series(
        [data.candidates[index] for index in result.target_candidate],
        index=calendar,
        name="target_candidate",
    )
    baseline_target = baseline.state["target_candidate"].astype(str)
    metrics = performance(returns)
    expected = config["checkpoint"]
    for field in (
        "observations",
        "annualized_return_252",
        "annualized_volatility",
        "sharpe",
        "max_drawdown",
    ):
        actual = metrics[field]
        target_value = expected[field]
        if abs(float(actual) - float(target_value)) > 1e-12:
            raise AssertionError(f"candidate checkpoint mismatch: {field}")
    leave_year = pd.DataFrame(
        [
            {
                "removed_year": int(year),
                **performance(returns.loc[calendar.year != year]),
            }
            for year in sorted(calendar.year.unique())
        ]
    )
    events, leave_events, deletions, event_summary = _event_stress(
        returns,
        baseline_returns,
        target,
        baseline_target,
        [1, 2, 3],
    )
    bootstrap, bootstrap_summary = paired_block_bootstrap(
        returns,
        baseline_returns,
        block_size=20,
        repetitions=5000,
        seed=20260824,
    )
    friction = _friction(
        returns,
        _selected_cost_schedule(context, data, result.target_candidate),
        [1.0, 2.0, 3.0],
    )
    no_gold_target = context.momentum_target.map(data.candidate_index).where(
        pd.Series(result.risk_on, index=calendar),
        data.candidate_index[DEFENDER_CANDIDATE],
    ).to_numpy(int)
    no_gold, _, _ = fast_candidate_schedule(data, no_gold_target)
    audit = {
        "status": "passed",
        "strategy_id": config["strategy_id"],
        "spec": {"gate": asdict(gate), "emergency": asdict(emergency)},
        "metrics": metrics,
        "baseline_metrics": performance(baseline_returns),
        "leave_one_year_min_annualized_return_252": float(
            leave_year["annualized_return_252"].min()
        ),
        "leave_one_year_min_sharpe": float(leave_year["sharpe"].min()),
        "leave_one_year_worst_mdd": float(leave_year["max_drawdown"].min()),
        "event_stress": event_summary,
        "paired_block_bootstrap": bootstrap_summary,
        "no_gold_metrics": performance(pd.Series(no_gold, index=calendar)),
    }
    output.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(
        {
            "risk_on": result.risk_on,
            "target_candidate": target,
            "return": returns,
            "nav": (1.0 + returns).cumprod(),
        },
        index=calendar,
    ).to_csv(output / "daily.csv")
    leave_year.to_csv(output / "leave_one_year.csv", index=False)
    events.to_csv(output / "events.csv", index=False)
    leave_events.to_csv(output / "leave_one_event.csv", index=False)
    deletions.to_csv(output / "top_event_deletion.csv", index=False)
    bootstrap.to_csv(output / "paired_bootstrap.csv", index=False)
    friction.to_csv(output / "friction.csv", index=False)
    (output / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "selected_research_config.yaml").write_text(
        config_path.read_text(encoding="utf-8"), encoding="utf-8"
    )
    generate_standard_report(
        returns,
        baseline_returns,
        "Current formal baseline",
        output / "absolute_stability_vs_formal.html",
        config,
    )
    sources = [
        config_path,
        root / "factors/quality_momentum.py",
        root / "research/momentum_defender_log_qm_robust.py",
        root / "research/run_momentum_defender_log_qm_absolute_stability.py",
        root / "research/DEVELOPMENT_VALIDATION.md",
    ]
    manifest = {
        "strategy_id": config["strategy_id"],
        "sources": [
            {"path": str(path.relative_to(root)), "sha256": _hash(path)}
            for path in sources
        ],
    }
    (output / "experiment_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    root = args.root.resolve()
    config_path = args.config if args.config.is_absolute() else root / args.config
    output = args.output if args.output.is_absolute() else root / args.output
    print(json.dumps(run_candidate(root, config_path, output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
