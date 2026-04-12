"""Standardization layer interface — see docs/DESIGN.md §2.3."""

import pandas as pd


def standardize(
    raw: dict[str, pd.Series],
    method: str = "cross_sectional_rank",
) -> dict[str, pd.Series]:
    """Map raw factor values to a comparable space.

    Methods: cross_sectional_rank, z_score, percentile.
    See docs/DESIGN.md §2.3 for the full interface contract.
    """
    raise NotImplementedError
