"""Standardization methods — see docs/DESIGN.md §2.3."""

import pandas as pd


def cross_sectional_rank(raw: dict[str, pd.Series], **kwargs) -> dict[str, pd.Series]:
    """Cross-sectional rank standardization.

    This method requires multi-asset input (ranking assets against each other
    at each time point). Phase 2 operates on single-asset factor dicts, so
    this is a planned gap. Will be wired in Phase 4 when multi-asset input
    is available. See specs/phase-2-factor-and-standardization.md Req 9.
    """
    raise NotImplementedError(
        "cross-sectional rank needs multi-asset input; will be wired in Phase 4"
    )


def z_score(raw: dict[str, pd.Series], *, window: int = 60) -> dict[str, pd.Series]:
    """Rolling z-score: (x - rolling_mean) / rolling_std."""
    result = {}
    for name, series in raw.items():
        rolling_mean = series.rolling(window).mean()
        rolling_std = series.rolling(window).std()
        result[name] = (series - rolling_mean) / rolling_std
    return result


def percentile(raw: dict[str, pd.Series], *, window: int = 60) -> dict[str, pd.Series]:
    """Rolling percentile rank within a window."""
    result = {}
    for name, series in raw.items():
        result[name] = series.rolling(window).rank(pct=True)
    return result


def standardize(
    raw: dict[str, pd.Series], method: str = "cross_sectional_rank", **kwargs
) -> dict[str, pd.Series]:
    """Dispatch to a specific standardization method."""
    methods = {
        "cross_sectional_rank": cross_sectional_rank,
        "z_score": z_score,
        "percentile": percentile,
    }
    if method not in methods:
        raise ValueError(f"unknown standardization method: '{method}', available: {list(methods.keys())}")
    return methods[method](raw, **kwargs)
