"""Execution layer interface — see docs/DESIGN.md §2.5."""

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

    See docs/DESIGN.md §2.5 for the full interface contract.
    """
    raise NotImplementedError
