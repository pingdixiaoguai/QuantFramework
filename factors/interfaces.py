"""Factor layer interface — see docs/DESIGN.md §2.2."""

import pandas as pd


def compute(df: pd.DataFrame, params: dict | None = None) -> pd.Series:
    """Compute a single factor from standardized OHLCV data.

    Input:  df with columns [date, open, high, low, close, volume]
    Output: pd.Series with date index and float values.
    See docs/DESIGN.md §2.2 for the full interface contract.
    """
    raise NotImplementedError
