"""Strategy layer interface — see docs/DESIGN.md §2.4."""

import pandas as pd


def generate_weights(
    standardized: dict[str, pd.Series],
    config: dict,
) -> dict[str, float]:
    """Consume standardized factor values and produce target portfolio weights.

    Output weights must sum to 1.0.
    See docs/DESIGN.md §2.4 for the full interface contract.
    """
    raise NotImplementedError
