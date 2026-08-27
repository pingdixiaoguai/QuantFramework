from __future__ import annotations

import pandas as pd
import pytest

from research.momentum_defender_downside_raqm import (
    DownsideRAQMFeatures,
    FactorProfile,
)
from research.momentum_defender_selected_asset_draqm import (
    SHADOW_TOP1_RECOVER_OTHER,
    STICKY_ENTRY_ASSET,
    AssetDRAQMPolicy,
    SelectedAssetDRAQMSpec,
    selected_asset_state_schedule,
)


def _policy(
    asset: str,
    *,
    entry: float = 0.7,
    recovery: float = 0.2,
    entry_confirmation: int = 1,
    recovery_confirmation: int = 1,
) -> AssetDRAQMPolicy:
    return AssetDRAQMPolicy(
        asset,
        FactorProfile(f"{asset}_w20", (20,), (1.0,)),
        entry,
        recovery,
        entry_confirmation,
        recovery_confirmation,
    )


def _features(
    index: pd.DatetimeIndex,
    csi: list[float],
    gold: list[float],
) -> dict[str, DownsideRAQMFeatures]:
    result = {}
    for asset, values in (("510300.SH", csi), ("518880.SH", gold)):
        profile_id = f"{asset}_w20"
        result[asset] = DownsideRAQMFeatures(
            calendar=index,
            raw_at_open={},
            percentile_at_open={},
            composite_at_open={
                (profile_id, "rolling_504_strict_lag"): pd.Series(
                    values, index=index
                )
            },
        )
    return result


def _spec(mode: str = STICKY_ENTRY_ASSET, **policy_overrides) -> SelectedAssetDRAQMSpec:
    return SelectedAssetDRAQMSpec(
        {
            "510300.SH": _policy("510300.SH", **policy_overrides),
            "518880.SH": _policy("518880.SH", **policy_overrides),
        },
        momentum_lock_days=20,
        defender_lock_days=20,
        recovery_mode=mode,
    )


def test_only_two_requested_assets_can_have_policies() -> None:
    with pytest.raises(ValueError, match="only 510300"):
        _policy("513100.SH")


def test_other_momentum_assets_are_never_gated() -> None:
    index = pd.date_range("2026-01-01", periods=4, freq="B")
    target = pd.Series("513100.SH", index=index)
    state = selected_asset_state_schedule(
        index,
        target,
        _features(index, [1.0] * 4, [1.0] * 4),
        _spec(),
    )
    assert state["risk_on"].all()
    assert state["state_reason"].eq("other_momentum_asset_not_gated").all()


def test_entry_uses_only_the_current_top1_assets_own_score() -> None:
    index = pd.date_range("2026-01-01", periods=4, freq="B")
    target = pd.Series(
        ["510300.SH", "518880.SH", "518880.SH", "518880.SH"], index=index
    )
    features = _features(index, [0.9, 0.9, 0.9, 0.9], [0.1, 0.1, 0.9, 0.9])
    state = selected_asset_state_schedule(index, target, features, _spec())
    assert not bool(state.iloc[0]["risk_on"])
    assert state.iloc[0]["evidence_asset"] == "510300.SH"


def test_entry_confirmation_resets_when_top1_changes() -> None:
    index = pd.date_range("2026-01-01", periods=5, freq="B")
    target = pd.Series(
        ["510300.SH", "518880.SH", "510300.SH", "510300.SH", "510300.SH"],
        index=index,
    )
    state = selected_asset_state_schedule(
        index,
        target,
        _features(index, [0.9] * 5, [0.9] * 5),
        _spec(entry_confirmation=3),
    )
    assert state.index[state["state_changed"]][0] == index[-1]


def test_sticky_recovery_keeps_monitoring_entry_asset() -> None:
    index = pd.date_range("2026-01-01", periods=22, freq="B")
    target = pd.Series(["510300.SH"] + ["513100.SH"] * 21, index=index)
    features = _features(index, [0.9] + [0.1] * 21, [0.9] * 22)
    state = selected_asset_state_schedule(index, target, features, _spec())
    assert not bool(state.iloc[19]["risk_on"])
    assert bool(state.iloc[20]["risk_on"])
    assert state.iloc[20]["evidence_asset"] == "510300.SH"


def test_shadow_mode_recovers_when_top1_is_an_ungated_asset() -> None:
    index = pd.date_range("2026-01-01", periods=22, freq="B")
    target = pd.Series(["510300.SH"] + ["513100.SH"] * 21, index=index)
    features = _features(index, [0.9] * 22, [0.9] * 22)
    state = selected_asset_state_schedule(
        index,
        target,
        features,
        _spec(mode=SHADOW_TOP1_RECOVER_OTHER),
    )
    assert not bool(state.iloc[19]["risk_on"])
    assert bool(state.iloc[20]["risk_on"])
    assert state.iloc[20]["state_reason"] == "shadow_other_asset_to_momentum"


def test_both_sleeve_locks_are_nonnegative_multiples_of_five() -> None:
    with pytest.raises(ValueError, match="Momentum lock"):
        SelectedAssetDRAQMSpec(
            {"510300.SH": None, "518880.SH": None}, 3, 20, STICKY_ENTRY_ASSET
        )
    with pytest.raises(ValueError, match="Defender lock"):
        SelectedAssetDRAQMSpec(
            {"510300.SH": None, "518880.SH": None}, 20, 6, STICKY_ENTRY_ASSET
        )
    SelectedAssetDRAQMSpec(
        {"510300.SH": None, "518880.SH": None}, 0, 0, STICKY_ENTRY_ASSET
    )
