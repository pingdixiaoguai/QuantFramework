"""Frozen entrypoint for the current formal Defender signal.

The formal strategy is the listing-aware history extension promoted on
2026-08-22. It uses 510880.SH as the causal signal bridge before 512890.SH
exists, switches to the 512890.SH anchor from the next open after its first
close, and rotates only among listed, ranking-eligible equity ETFs.
"""

from __future__ import annotations

from typing import Mapping

import pandas as pd

from .defender_opt_v2 import CostRateSpec
from .relative_defender_rotation import (
    BASE_ANCHOR_SIGNAL,
    RelativeDefenderRotationParams,
    rotation_params,
)
from .relative_defender_rotation_2013_report import (
    PROMOTION_DATE,
    STRATEGY_ID,
    run_backtest as run_listing_aware_backtest,
)


FORMAL_STRATEGY_ID = STRATEGY_ID
FORMAL_PROMOTION_DATE = PROMOTION_DATE
CurrentStrategyParams = RelativeDefenderRotationParams


def current_strategy_params() -> CurrentStrategyParams:
    """Return a fresh copy of the frozen formal rotation parameters."""
    params = rotation_params()
    if params.primary_signal_source != BASE_ANCHOR_SIGNAL:
        raise RuntimeError("formal strategy must remain anchored to 512890")
    if params.selected_asset_weight != 1.0:
        raise RuntimeError("formal strategy must rotate 100% of the primary sleeve")
    return params


def run_backtest(
    market: Mapping[str, pd.DataFrame] | None = None,
    params: CurrentStrategyParams | None = None,
    cost_rate: CostRateSpec | None = None,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict[str, object],
    pd.DataFrame,
]:
    """Run the formal listing-aware rotation and expose monthly events."""
    selected = current_strategy_params() if params is None else params
    if selected.primary_signal_source != BASE_ANCHOR_SIGNAL:
        raise ValueError("formal strategy only accepts the base_anchor signal source")
    if selected.selected_asset_weight != 1.0:
        raise ValueError("formal strategy only accepts 100% primary-sleeve rotation")
    daily, trades, metrics, _, events = run_listing_aware_backtest(
        market=market,
        params=selected,
        cost_rate=cost_rate,
    )
    formal_metrics: dict[str, object] = {
        **metrics,
        "strategy": FORMAL_STRATEGY_ID,
        "formal_status": "production_signal_frozen",
        "formal_promotion_date": FORMAL_PROMOTION_DATE,
        "evidence_status": "retrospective_history_extension_not_oos",
    }
    return daily, trades, formal_metrics, events


def main() -> None:
    daily, _, metrics, _ = run_backtest()
    latest = daily.iloc[-1]
    print(f"strategy: {metrics['strategy']}")
    print(f"signal_date: {daily.index[-1].date()}")
    print(f"selected_asset: {latest['selected_asset']}")
    print(f"primary_target: {float(latest['primary_target']):.0%}")
    print(f"defensive_target: {float(latest['defensive_target']):.0%}")


if __name__ == "__main__":
    main()
