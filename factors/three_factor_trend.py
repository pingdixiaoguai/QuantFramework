"""Research score with fixed 20-day momentum and soft quality adjustments.

Default formula (the train-only plateau representative from the broad search)::

    M20 = close_t / close_{t-20} - 1
    ER20 = |close_t - close_{t-20}| / sum(|close_i - close_{i-1}|)
    R2_10 = R-squared of a linear fit to the latest 10 log closes
    sigma20 = standard deviation of the latest 20 simple daily returns
    Q80 = lagged 80th percentile of sigma20 over the latest 252 observations
    V = clip(exp(1 - sigma20 / Q80), 0.25, 2.0)
    score = M20 * ER20^0.75 * (0.5 + R2_10) * V^0.50

The momentum horizon is deliberately fixed at 20 days.  The volatility
percentile is strictly lagged by one observation and needs 60 prior finite
volatility estimates before it activates.  Until then its multiplier is
neutral, so the factor remains usable after the 20-day base has warmed up.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


METADATA = {
    "name": "three_factor_trend",
    "author": "quantframework",
    "version": "2.0.0",
    "params": {
        "momentum_window": 20,
        "er_window": 20,
        "er_power": 0.75,
        "linearity_window": 10,
        "linearity_offset": 0.5,
        "linearity_power": 1.0,
        "volatility_window": 20,
        "volatility_quantile": 0.80,
        "volatility_history": 252,
        "quantile_min_history": 60,
        "low_vol_shape": "exponential",
        "low_vol_power": 0.50,
        "low_vol_floor": 0.25,
        "low_vol_cap": 2.0,
    },
    "min_history": 21,
    "direction": "higher_better",
    "description": (
        "固定20日动量 × ER路径质量 × 对数价格线性度 × "
        "严格滞后的相对低波软乘子（研究候选）"
    ),
}


def _validate_params(params: dict) -> None:
    if int(params["momentum_window"]) != 20:
        raise ValueError("momentum_window is fixed at 20 for three_factor_trend")
    for name in (
        "er_window",
        "linearity_window",
        "volatility_window",
        "volatility_history",
        "quantile_min_history",
    ):
        if int(params[name]) < 2:
            raise ValueError(f"{name} must be >= 2, got {params[name]}")
    for name in ("er_power", "linearity_power", "low_vol_power"):
        if float(params[name]) < 0.0:
            raise ValueError(f"{name} must be >= 0, got {params[name]}")
    quantile = float(params["volatility_quantile"])
    if not 0.0 < quantile < 1.0:
        raise ValueError(f"volatility_quantile must be in (0, 1), got {quantile}")
    floor = float(params["low_vol_floor"])
    cap = float(params["low_vol_cap"])
    if not 0.0 < floor <= 1.0 <= cap:
        raise ValueError(
            "low_vol_floor and low_vol_cap must satisfy 0 < floor <= 1 <= cap"
        )
    if params["low_vol_shape"] not in {
        "cap",
        "symmetric",
        "exponential",
        "linear",
    }:
        raise ValueError(f"unknown low_vol_shape: {params['low_vol_shape']}")


def _rolling_r_squared(log_close: pd.Series, window: int) -> pd.Series:
    x = np.arange(window, dtype=float)
    centered_x = x - x.mean()
    x_sum_squares = float(centered_x @ centered_x)

    def calculate(values: np.ndarray) -> float:
        centered_y = values - values.mean()
        y_sum_squares = float(centered_y @ centered_y)
        if y_sum_squares <= 0.0:
            return 0.0
        covariance = float(centered_x @ centered_y)
        return float(covariance * covariance / (x_sum_squares * y_sum_squares))

    return log_close.rolling(window, min_periods=window).apply(calculate, raw=True)


def _low_vol_multiplier(
    relative_volatility: pd.Series,
    shape: str,
    floor: float,
    cap: float,
) -> pd.Series:
    inverse = 1.0 / relative_volatility.replace(0.0, np.nan)
    if shape == "cap":
        multiplier = inverse.clip(lower=floor, upper=1.0)
    elif shape == "symmetric":
        multiplier = inverse.clip(lower=floor, upper=cap)
    elif shape == "exponential":
        multiplier = np.exp(1.0 - relative_volatility).clip(lower=floor, upper=cap)
    elif shape == "linear":
        multiplier = (1.5 - 0.5 * relative_volatility).clip(
            lower=floor,
            upper=cap,
        )
    else:  # Protected by _validate_params; keeps this helper total in isolation.
        raise ValueError(f"unknown low_vol_shape: {shape}")
    return multiplier.fillna(1.0)


def compute(df: pd.DataFrame, params: dict | None = None) -> pd.Series:
    """Compute the selected research score without using future observations."""
    p = {**METADATA["params"], **(params or {})}
    _validate_params(p)

    close = pd.to_numeric(df["close"], errors="coerce")
    momentum_window = int(p["momentum_window"])
    er_window = int(p["er_window"])
    linearity_window = int(p["linearity_window"])
    volatility_window = int(p["volatility_window"])

    momentum = close.pct_change(momentum_window, fill_method=None)
    displacement = (close - close.shift(er_window)).abs()
    path = close.diff().abs().rolling(er_window, min_periods=er_window).sum()
    efficiency = displacement / path.replace(0.0, np.nan)
    efficiency = efficiency.mask(path.eq(0.0) & displacement.eq(0.0), 0.0)

    linearity = _rolling_r_squared(np.log(close), linearity_window)
    path_quality = (float(p["linearity_offset"]) + linearity).pow(
        float(p["linearity_power"])
    )

    volatility = close.pct_change(fill_method=None).rolling(
        volatility_window,
        min_periods=volatility_window,
    ).std(ddof=1)
    lagged_volatility = volatility.shift(1)
    threshold = lagged_volatility.rolling(
        int(p["volatility_history"]),
        min_periods=int(p["quantile_min_history"]),
    ).quantile(float(p["volatility_quantile"]))
    relative_volatility = volatility / threshold.replace(0.0, np.nan)
    low_volatility = _low_vol_multiplier(
        relative_volatility,
        str(p["low_vol_shape"]),
        float(p["low_vol_floor"]),
        float(p["low_vol_cap"]),
    )

    score = (
        momentum
        * efficiency.pow(float(p["er_power"]))
        * path_quality
        * low_volatility.pow(float(p["low_vol_power"]))
    )
    score.index = df["date"]
    return score.astype(float)
