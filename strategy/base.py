"""Strategy base class — see docs/DESIGN.md §2.4."""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseStrategy(ABC):
    def __init__(self, config: dict):
        self.config = config

    @abstractmethod
    def generate_weights(
        self, factor_values: dict[str, dict[str, float]]
    ) -> dict[str, float]:
        """Generate target portfolio weights from cross-sectional factor values.

        Args:
            factor_values: asset_code -> factor_name -> value (today's snapshot)

        Returns:
            asset_code -> weight (must sum to 1.0), or empty dict if no input.
        """
        ...
