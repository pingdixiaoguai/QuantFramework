"""Execution layer — see docs/DESIGN.md §2.5."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class Order:
    """A single rebalance instruction."""

    asset: str
    action: Literal["buy", "sell", "hold"]
    weight_delta: float


def diff(target: dict[str, float], current: dict[str, float]) -> list[Order]:
    """Compare target vs current weights and return rebalance orders.

    Assets in target but not current → buy.
    Assets in current but not target → sell (full exit).
    Assets in both → buy/sell/hold based on weight change.
    """
    orders: list[Order] = []
    all_assets = sorted(set(target) | set(current))

    for asset in all_assets:
        t = target.get(asset, 0.0)
        c = current.get(asset, 0.0)
        delta = t - c

        if delta > 0:
            action: Literal["buy", "sell", "hold"] = "buy"
        elif delta < 0:
            action = "sell"
        else:
            action = "hold"

        orders.append(Order(asset=asset, action=action, weight_delta=delta))

    return orders
