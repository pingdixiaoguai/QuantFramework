"""Factor output validator — see docs/DESIGN.md §2.2."""

import pandas as pd


def validate(series: pd.Series, df: pd.DataFrame, metadata: dict) -> None:
    """Validate factor output against the contract.

    Raises ValueError with all violations if validation fails.
    """
    errors: list[str] = []
    name = metadata.get("name", "<unknown>")
    min_history = metadata["min_history"]

    # Length check
    if len(series) != len(df):
        errors.append(
            f"length mismatch: series has {len(series)} rows, df has {len(df)}"
        )

    # Index check
    expected_index = pd.Index(df["date"])
    if not series.index.equals(expected_index):
        errors.append("index does not match df['date']")

    # Dtype check
    if not pd.api.types.is_float_dtype(series):
        errors.append(f"dtype is {series.dtype}, expected float")

    # NaN check: tail (from min_history-1 onward) must not contain NaN
    if len(series) >= min_history:
        tail = series.iloc[min_history - 1 :]
        tail_nans = tail.isna().sum()
        if tail_nans > 0:
            errors.append(
                f"{tail_nans} NaN(s) found after position {min_history - 1} "
                f"(min_history={min_history})"
            )

    if errors:
        raise ValueError(
            f"factor '{name}' validation failed: " + "; ".join(errors)
        )
