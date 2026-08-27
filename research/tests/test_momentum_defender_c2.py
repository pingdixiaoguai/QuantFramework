from pathlib import Path

import pandas as pd
import pytest

from research.momentum_defender_c2 import (
    DEFAULT_CONFIG_PATH,
    held_asset_cap_alert,
    load_frozen_c2_config,
    run_frozen_c2,
    validate_frozen_checkpoint,
)


ROOT = Path(__file__).resolve().parents[2]
V1_CONFIG_PATH = (
    ROOT / "research/configs/momentum_defender_c2_frozen_v1.yaml"
)
HISTORICAL_SWITCH_PATH = (
    ROOT / "defender/deliverable/relative_defender_rotation_switch_returns.csv"
)


def test_current_frozen_config_records_q95_chinext_and_expanding_history() -> None:
    config = load_frozen_c2_config()

    assert config.asset_quantiles["510300.SH"] == 0.70
    assert config.asset_quantiles["159915.SZ"] == 0.95
    assert config.asset_quantiles["513100.SH"] == 0.95
    assert config.asset_quantiles["518880.SH"] == 0.90
    assert config.quantile_history == "expanding_all_available_strict_lag"
    assert config.variant_id() == (
        "C2_vw10_cap0.8_qc3000.70_qcyb0.95_qndx0.95_qau0.90"
    )
    assert config.strategy_id == "momentum_defender_c2_frozen_v2"
    assert DEFAULT_CONFIG_PATH.exists()


def test_q90_v1_remains_reproducible_as_historical_checkpoint() -> None:
    config = load_frozen_c2_config(V1_CONFIG_PATH)

    assert config.strategy_id == "momentum_defender_c2_frozen_v1"
    assert config.asset_quantiles["159915.SZ"] == 0.90


def test_held_asset_alert_uses_only_previous_close_holding() -> None:
    dates = pd.date_range("2026-01-05", periods=4, freq="B")
    caps = {
        "510300.SH": pd.Series([0.8, 1.0, 1.0, 1.0], index=dates),
        "159915.SZ": pd.Series([0.2, 0.8, 0.2, 0.2], index=dates),
        "513100.SH": pd.Series([0.2, 0.2, 1.0, 0.2], index=dates),
        "518880.SH": pd.Series([0.2, 0.2, 0.2, 1.0], index=dates),
    }
    previous = pd.Series(
        ["510300.SH", "159915.SZ", "513100.SH", "518880.SH"],
        index=dates,
    )

    assert held_asset_cap_alert(caps, previous, 0.8).tolist() == [
        True,
        True,
        False,
        False,
    ]


@pytest.mark.skipif(
    not HISTORICAL_SWITCH_PATH.exists(),
    reason="historical frozen-v2 external Defender handoff is not vendored",
)
def test_frozen_backtest_matches_versioned_checkpoint() -> None:
    config = load_frozen_c2_config()
    result = run_frozen_c2(ROOT, config)

    audit = validate_frozen_checkpoint(result)

    assert audit["status"] == "passed"
