"""Reconstruct executable open targets for the formal W40 strategy."""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

from defender.relative_defender_rotation import DEFENSIVE_ASSET
from defender.w40_reversal_full_equity import FORMAL_DIVIDEND_ASSETS
from research.defender_curve_momentum import DEFENDER_CANDIDATE
from research.momentum_defender_occam import MOMENTUM_ASSETS


DEFENDER_ASSETS = (*FORMAL_DIVIDEND_ASSETS, DEFENSIVE_ASSET)
ALL_ASSETS = (*MOMENTUM_ASSETS, *DEFENDER_ASSETS)


def build_formal_target_schedule(formal_run) -> pd.DataFrame:
    """Return the actual post-open target for the replayed formal candidate.

    ``IntegratedC2Backtest.targets`` follows the old C2 state schedule and must
    not be used to describe holdings on dates where the formal W40 schedule
    disagrees with C2.  The candidate-level formal ledger supplies the active
    sleeve; the in-memory Defender interface supplies its executable weights.
    """

    daily = formal_run.daily
    defender = formal_run.context.integrated.result.inputs.defender
    defender_interface = formal_run.context.interfaces[DEFENDER_CANDIDATE]
    targets = pd.DataFrame(
        0.0,
        index=daily.index,
        columns=[*ALL_ASSETS, "target_cash_weight"],
    )
    for timestamp, candidate in daily["candidate"].astype(str).items():
        if candidate == DEFENDER_CANDIDATE:
            for asset in DEFENDER_ASSETS:
                code = asset.split(".", maxsplit=1)[0]
                promoted_column = f"target_{asset}"
                legacy_column = f"target_weight_{code}"
                if promoted_column in defender_interface:
                    value = defender_interface.at[timestamp, promoted_column]
                elif legacy_column in defender:
                    value = defender.at[timestamp, legacy_column]
                else:
                    value = 0.0
                targets.at[timestamp, asset] = float(value)
            targets.at[timestamp, "target_cash_weight"] = float(
                defender_interface.at[timestamp, "target_cash_weight"]
                if "target_cash_weight" in defender_interface
                else defender.at[timestamp, "target_cash_weight"]
            )
        else:
            if candidate not in MOMENTUM_ASSETS:
                raise AssertionError(f"unknown formal candidate: {candidate}")
            targets.at[timestamp, candidate] = 1.0

    total = targets.sum(axis=1)
    if not np.allclose(total, 1.0, atol=1e-12):
        bad = total.loc[~np.isclose(total, 1.0, atol=1e-12)]
        raise AssertionError(
            "formal target plus cash does not sum to one: "
            f"{bad.head().to_dict()}"
        )
    targets.index.name = "date"
    return targets


def portfolio_key(
    row: pd.Series,
    assets: tuple[str, ...] = ALL_ASSETS,
) -> tuple[float, ...]:
    """Return a stable grouping key for an executable target row."""

    return tuple(
        round(float(row.get(asset, 0.0)), 12)
        for asset in (*assets, "target_cash_weight")
    )


def format_portfolio(
    row: pd.Series,
    asset_names: Mapping[str, str],
    assets: tuple[str, ...] = ALL_ASSETS,
) -> str:
    """Format non-zero target weights for a human-readable ledger."""

    parts = []
    for asset in assets:
        weight = float(row.get(asset, 0.0))
        if weight > 1e-12:
            parts.append(f"{asset}（{asset_names.get(asset, asset)}）{weight:.0%}")
    cash = float(row.get("target_cash_weight", 0.0))
    if cash > 1e-12:
        parts.append(f"现金{cash:.0%}")
    return " + ".join(parts) if parts else "无可执行目标"
