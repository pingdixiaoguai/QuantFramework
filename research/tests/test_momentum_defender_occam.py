from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from research.momentum_defender_occam import (
    ENTER_RETURN,
    ENTRY_COST,
    EXIT_COST,
    EXIT_RETURN,
    HELD_RETURN,
    INTERNAL_COST,
    OccamParams,
    apply_state_schedule,
    build_inputs,
    load_defender_bundle,
    performance,
    scale_interface_costs,
    simulate_switch,
    state_schedule,
    volatility_cap_at_open,
)
from research.momentum_defender_occam_report import generate_html_report
from research.run_momentum_defender_occam import (
    _episode_attribution,
    _finite_sample_upper_tail_p_value,
    _generate_standard_report,
)


DEFENDER_DELIVERABLE = Path(
    "/Users/hujiaoyuan/Desktop/Quant/Defender/defender/deliverable"
)
EXPERIMENT_OUTPUT = Path(__file__).resolve().parents[2] / "experiments" / (
    "20260821_momentum_defender_occam"
)


def _interface(
    index: pd.DatetimeIndex,
    held: list[float],
    enter: list[float],
    exit_: list[float],
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            HELD_RETURN: held,
            ENTER_RETURN: enter,
            EXIT_RETURN: exit_,
            INTERNAL_COST: 0.0,
            ENTRY_COST: 0.001,
            EXIT_COST: [np.nan, *([0.001] * (len(index) - 1))],
        },
        index=index,
    )


def test_switch_day_chains_old_exit_and_new_entry_segments() -> None:
    dates = pd.date_range("2026-01-01", periods=3, freq="D")
    momentum = _interface(dates, [0.01, 0.02, 0.03], [0.10, 0.20, 0.30], [np.nan, 0.04, 0.05])
    defender = _interface(dates, [0.001, 0.002, 0.003], [0.01, 0.02, 0.03], [np.nan, 0.004, 0.005])

    result = simulate_switch(
        momentum,
        defender,
        pd.Series([True, False, False], index=dates),
        initial_previous_state=None,
    )

    assert result.iloc[0]["return"] == pytest.approx(0.10)
    assert result.iloc[1]["return"] == pytest.approx((1.04 * 1.02) - 1.0)
    assert result.iloc[2]["return"] == pytest.approx(0.003)
    assert result["transition"].tolist() == [
        "cash_to_momentum",
        "momentum_to_defender",
        "defender_hold",
    ]
    assert np.isnan(result.iloc[1]["held_return_leg_used"])
    assert result.iloc[1]["exit_return_leg_used"] == pytest.approx(0.04)
    assert result.iloc[1]["enter_return_leg_used"] == pytest.approx(0.02)


def test_switch_day_ignores_extreme_held_returns_in_both_directions() -> None:
    dates = pd.date_range("2026-01-01", periods=3, freq="D")
    momentum = _interface(dates, [99.0, 99.0, 99.0], [0.10, 0.20, 0.30], [np.nan, 0.04, 0.05])
    defender = _interface(dates, [-0.99, -0.99, -0.99], [0.01, 0.02, 0.03], [np.nan, 0.004, 0.005])

    result = simulate_switch(
        momentum,
        defender,
        pd.Series([True, False, True], index=dates),
        initial_previous_state="momentum",
    )

    assert result.iloc[1]["return"] == pytest.approx((1.04 * 1.02) - 1.0)
    assert result.iloc[2]["return"] == pytest.approx((1.005 * 1.30) - 1.0)
    assert result["held_return_leg_used"].iloc[1:].isna().all()


def test_close_signal_and_emergency_only_change_next_open() -> None:
    dates = pd.date_range("2026-01-01", periods=5, freq="D")
    close = pd.Series([100.0, 106.0, 112.0, 118.0, 124.0], index=dates)
    momentum_return = pd.Series([0.0, 0.0, -0.20, 0.0, 0.0], index=dates)

    schedule = state_schedule(
        close,
        momentum_return,
        dates,
        OccamParams(
            lookback=1,
            risk_on_threshold=0.05,
            min_hold_days=1,
            emergency_daily_loss=-0.10,
        ),
    )

    # Day-3 close has a positive slow signal and a -20% Momentum shock.  The
    # portfolio remains risk-on at day-3 open, then the emergency wins at
    # day-4 open.  Slow recovery may only act at a later open.
    assert schedule["risk_on"].tolist() == [True, True, True, False, True]
    assert schedule["state_reason"].tolist()[3] == "emergency_exit"


def test_emergency_threshold_must_be_negative() -> None:
    dates = pd.date_range("2026-01-01", periods=2, freq="D")
    with pytest.raises(ValueError, match="must be negative"):
        state_schedule(
            pd.Series([1.0, 1.0], index=dates),
            pd.Series([0.0, 0.0], index=dates),
            dates,
            OccamParams(1, 0.0, 1, 0.0),
        )


def test_external_emergency_overrides_minimum_hold_but_not_reentry_lock() -> None:
    dates = pd.date_range("2026-01-01", periods=6, freq="D")
    slow = pd.Series([True] * 6, index=dates)
    emergency = pd.Series([False, True, False, False, False, False], index=dates)

    schedule = apply_state_schedule(slow, emergency, dates, min_hold_days=3)

    assert schedule["risk_on"].tolist() == [True, False, False, False, True, True]
    assert schedule.iloc[1]["state_reason"] == "emergency_exit"
    assert schedule.iloc[4]["state_reason"] == "slow_regime_switch"


def test_proportional_cost_stress_rebuilds_all_three_net_segments() -> None:
    dates = pd.date_range("2026-01-01", periods=2, freq="D")
    frame = _interface(dates, [0.0, 0.0], [0.0, 0.0], [np.nan, 0.0])
    frame["overnight_gross_return"] = [0.0, 0.02]
    frame["intraday_gross_return_if_held"] = [0.03, -0.01]
    frame["intraday_gross_return_if_entered"] = [0.03, -0.01]
    frame[INTERNAL_COST] = [0.001, 0.002]

    stressed = scale_interface_costs(frame, 2.0)

    assert stressed.iloc[1][HELD_RETURN] == pytest.approx(1.02 * (1 - 0.004) * 0.99 - 1)
    assert stressed.iloc[1][ENTER_RETURN] == pytest.approx((1 - 0.002) * 0.99 - 1)
    assert stressed.iloc[1][EXIT_RETURN] == pytest.approx(1.02 * (1 - 0.002) - 1)


def test_performance_rejects_missing_returns_and_counts_first_interval() -> None:
    dates = pd.date_range("2026-01-01", periods=2, freq="D")
    with pytest.raises(ValueError, match="missing returns"):
        performance(pd.Series([0.01, np.nan], index=dates))

    measured = performance(pd.Series([0.10, 0.0], index=dates))
    expected = 1.10 ** (365.2425 / 2.0) - 1.0
    assert measured["cagr_calendar"] == pytest.approx(expected)


def test_finite_sample_p_value_never_reports_zero() -> None:
    assert _finite_sample_upper_tail_p_value([0.0, 0.1, 0.2], 1.0) == pytest.approx(0.25)


def test_standard_report_uses_project_generator_and_full_aligned_sample(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dates = pd.date_range("2026-01-05", periods=3, freq="B")
    captured: dict[str, object] = {}

    def fake_generate(result, output_path, benchmark_title=None):
        captured["result"] = result
        captured["benchmark_title"] = benchmark_title
        output_path.write_text("standard report", encoding="utf-8")
        return output_path

    monkeypatch.setattr(
        "research.run_momentum_defender_occam.generate_backtest_report",
        fake_generate,
    )
    output = _generate_standard_report(
        pd.Series([0.01, 0.02, 0.03], index=dates),
        pd.Series([0.0, -0.01, 0.01], index=dates),
        "Original Momentum Strategy",
        tmp_path / "report.html",
        {"strategy_name": "momentum_defender_occam"},
    )

    result = captured["result"]
    assert output.read_text(encoding="utf-8") == "standard report"
    assert captured["benchmark_title"] == "Original Momentum Strategy"
    assert len(result.daily_returns) == 3
    assert result.baseline_strategy_name == "Original Momentum Strategy"


def test_episode_metrics_include_entry_and_exit_open_days() -> None:
    dates = pd.date_range("2026-01-05", periods=4, freq="B")
    selected = pd.DataFrame(
        {
            "return": [0.10, 0.02, 0.03, 0.0],
            "sleeve": ["defender", "defender", "momentum", "momentum"],
            "transition": [
                "momentum_to_defender",
                "defender_hold",
                "defender_to_momentum",
                "momentum_hold",
            ],
            "state_reason": [
                "emergency_exit",
                "emergency_hold",
                "slow_regime_switch",
                "hold",
            ],
        },
        index=dates,
    )
    baseline = pd.Series([0.05, 0.0, 0.01, 0.0], index=dates)

    row = _episode_attribution(selected, baseline).iloc[0]

    assert row["window_observations"] == 3
    assert row["defender_days"] == 2
    assert bool(row["cap_triggered_entry"])
    assert row["candidate_return"] == pytest.approx(1.10 * 1.02 * 1.03 - 1.0)
    assert row["momentum_return"] == pytest.approx(1.05 * 1.01 - 1.0)
    assert row["arithmetic_excess_return"] == pytest.approx(
        row["candidate_return"] - row["momentum_return"]
    )
    assert row["candidate_max_drawdown"] <= 0.0


@pytest.mark.skipif(
    not (EXPERIMENT_OUTPUT / "defender_episode_metrics.csv").exists(),
    reason="final Occam experiment outputs are not available",
)
def test_html_report_contains_all_episode_rows_and_2024_forensics(
    tmp_path: Path,
) -> None:
    output = generate_html_report(
        EXPERIMENT_OUTPUT,
        tmp_path / "backtest_report.html",
    )
    report = output.read_text(encoding="utf-8")

    assert '<html lang="zh-CN">' in report
    assert 'id="episode-table"' in report
    assert report.count('data-positive="') == 22
    assert "2024-09-30" in report
    assert "signal_volatility_cap" in report
    assert "p=0.210" in report


@pytest.mark.skipif(
    not (EXPERIMENT_OUTPUT / "momentum_defender_occam_vs_original_base.html").exists(),
    reason="final standard QuantStats reports are not available",
)
def test_standard_quantstats_reports_use_both_expected_benchmarks() -> None:
    expected = {
        "momentum_defender_occam_vs_original_base.html": (
            "Original 4ETF Equal-Weight Base"
        ),
        "momentum_defender_occam_vs_original_momentum.html": (
            "Original Momentum Strategy"
        ),
        "momentum_defender_occam_no_cap_vs_original_momentum.html": (
            "Original Momentum Strategy"
        ),
    }
    for filename, benchmark_name in expected.items():
        report = (EXPERIMENT_OUTPUT / filename).read_text(encoding="utf-8")
        assert "Strategy Tearsheet (Compounded)" in report
        assert "18 Jan, 2019 - 17 Aug, 2026 (matched dates)" in report
        assert benchmark_name in report


@pytest.mark.skipif(
    not (DEFENDER_DELIVERABLE / "relative_defender_rotation_switch_returns.csv").exists(),
    reason="accepted Defender handoff is not available on this machine",
)
def test_accepted_handoff_and_corrected_calendar_integrate_with_momentum() -> None:
    bundle = load_defender_bundle(DEFENDER_DELIVERABLE, date(2026, 8, 17))
    assert len(bundle.switch_returns) == 1837
    assert bundle.audit["passed"].all()
    assert {
        "asset_transaction_cost_rates",
        "target_cash_weight_is_zero",
    }.issubset(set(bundle.audit["check"]))
    suspension = bundle.switch_returns.loc[pd.Timestamp("2021-10-22")]
    assert suspension[HELD_RETURN] == pytest.approx(0.0004379319103466)
    assert suspension["target_weight_511260"] == pytest.approx(1.0)

    root = Path(__file__).resolve().parents[2]
    inputs = build_inputs(
        root,
        DEFENDER_DELIVERABLE / "relative_defender_rotation_switch_returns.csv",
        date(2026, 8, 17),
    )
    official = inputs.momentum_result.daily_returns.reindex(inputs.calendar)
    daily_error = (inputs.momentum[HELD_RETURN] - official).abs().max()
    nav_ratio = (
        (1.0 + inputs.momentum[HELD_RETURN]).prod()
        / (1.0 + official).prod()
        - 1.0
    )
    assert daily_error <= 1.3e-5
    assert abs(nav_ratio) <= 8e-5

    cap = volatility_cap_at_open(bundle.indicators, inputs.calendar)
    assert cap.index.equals(inputs.calendar)
    # 2021-10-21 close signal is explicitly effective on the suspension-day open.
    assert bool(cap.loc[pd.Timestamp("2021-10-22")])
