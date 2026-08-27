from __future__ import annotations

from pathlib import Path

from research.momentum_defender_downside_raqm import FactorProfile
from research.momentum_defender_selected_asset_draqm import (
    STICKY_ENTRY_ASSET,
    AssetDRAQMPolicy,
    SelectedAssetDRAQMSpec,
)
from research.run_momentum_defender_dual_regime_search import (
    _load_config,
    _policy_grid,
    _profiles,
)


CONFIG = Path("research/configs/momentum_defender_dual_regime_research.yaml")


def test_config_searches_only_nonnegative_five_day_multiple_locks() -> None:
    config = _load_config(CONFIG)
    for field in ("momentum_lock_days", "defender_lock_days"):
        values = config["joint_stage"][field]
        assert values == [0, 5, 10, 15, 20, 25, 30]
        assert all(value >= 0 and value % 5 == 0 for value in values)


def test_asset_score_profiles_may_differ_and_zero_lock_is_valid() -> None:
    csi = AssetDRAQMPolicy(
        "510300.SH", FactorProfile("csi", (30, 40), (0.25, 0.75)), 0.5, 0.2, 1, 1
    )
    gold = AssetDRAQMPolicy(
        "518880.SH", FactorProfile("gold", (20, 40), (0.25, 0.75)), 0.7, 0.1, 3, 1
    )
    spec = SelectedAssetDRAQMSpec(
        {"510300.SH": csi, "518880.SH": gold}, 0, 0, STICKY_ENTRY_ASSET
    )
    assert spec.momentum_lock_days == 0
    assert csi.profile.horizons != gold.profile.horizons


def test_single_asset_grid_covers_every_profile_for_both_assets() -> None:
    config = _load_config(CONFIG)
    profiles = _profiles(config)
    grid = _policy_grid(config, profiles)
    assert set(grid["510300.SH"]) == set(profiles)
    assert set(grid["518880.SH"]) == set(profiles)
    assert all(grid[asset][profile] for asset in grid for profile in profiles)
