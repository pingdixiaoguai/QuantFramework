"""Data layer interface — see docs/DESIGN.md §2.1."""

from datetime import date

import pandas as pd


def query(asset_code: str, start: date, end: date) -> pd.DataFrame:
    """Return OHLCV DataFrame for a single asset over [start, end].

    Columns: [date, open, high, low, close, volume].
    See docs/DESIGN.md §2.1 for the full interface contract.
    """
    raise NotImplementedError
