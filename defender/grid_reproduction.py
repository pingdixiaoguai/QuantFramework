"""Reproduction candidate for the 512890 exposure chart.

The rule is deliberately kept explicit and causal:

* signals are calculated from the close available on day ``t``;
* a changed target is executed at the next trading day's open;
* the portfolio is marked at the next close;
* exposure is restricted to a configurable grid between 0% and 100%;
* a causal realized-volatility cap reduces exposure in high-volatility regimes.

This is a fitted reconstruction of the supplied chart, not a claim that the
original author's exact implementation has been recovered.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from data.store import read_local


ASSET = "512890.SH"
START = date(2019, 1, 18)
INITIAL_CAPITAL = 1.0
TRADING_DAYS = 252


@dataclass(frozen=True)
class GridParams:
    range_window: int = 55
    lower_percentile: float = 0.10
    upper_percentile: float = 0.90
    breakout_window: int = 5
    breakout_threshold: float = 0.02
    step: float = 0.50
    min_exposure: float = 0.0
    max_exposure: float = 1.00
    volatility_method: str = "rogers_satchell"
    volatility_window: int = 5
    target_volatility: float = 0.16
    # 512890's configured one-way commission is 0.01%.
    cost_rate: float = 0.0001


def load_prices(
    start: date = START,
    end: date | None = None,
) -> pd.DataFrame:
    """Load HFQ OHLC through the latest locally available date by default."""
    frame = read_local(ASSET)
    if frame is None or frame.empty:
        raise RuntimeError(f"missing local data for {ASSET}")
    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.loc[frame["date"] >= pd.Timestamp(start)]
    if end is not None:
        frame = frame.loc[frame["date"] <= pd.Timestamp(end)]
    frame = frame.sort_values("date").drop_duplicates("date")
    required = ["date", "open", "high", "low", "close"]
    if frame.empty or frame[required].isna().any().any():
        raise ValueError("price data is empty or contains missing OHLC values")
    if (frame[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("price data contains non-positive OHLC values")
    return frame.reset_index(drop=True)


def _target_from_close(
    closes: np.ndarray,
    index: int,
    target: float,
    params: GridParams,
) -> tuple[float, str, float]:
    """Advance the state machine using close[index] and prior closes only."""
    if index < params.range_window - 1:
        return target, "warmup", np.nan

    window = closes[index - params.range_window + 1:index + 1]
    low = float(window.min())
    high = float(window.max())
    location = (closes[index] - low) / (high - low) if high > low else 0.5

    if index >= params.breakout_window:
        prior = closes[index - params.breakout_window:index]
        if closes[index] > float(prior.max()) * (1.0 + params.breakout_threshold):
            return params.max_exposure, "breakout_reentry", location

    if location <= params.lower_percentile:
        return min(params.max_exposure, target + params.step), "low_add", location
    if location >= params.upper_percentile:
        return max(params.min_exposure, target - params.step), "high_reduce", location
    return target, "hold", location


def _volatility_cap(realized_volatility: float, params: GridParams) -> float:
    """Return a causal, step-quantized exposure cap for one realized volatility."""
    if not np.isfinite(realized_volatility) or realized_volatility <= 0:
        return params.max_exposure
    raw_cap = min(params.max_exposure, params.target_volatility / realized_volatility)
    stepped_cap = np.floor(raw_cap / params.step + 1e-12) * params.step
    return float(np.clip(stepped_cap, params.min_exposure, params.max_exposure))


def _realized_volatility(frame: pd.DataFrame, params: GridParams) -> np.ndarray:
    """Calculate a causal annualized volatility estimate from 512890 OHLC data."""
    method = params.volatility_method
    window = params.volatility_window
    if window < 2:
        raise ValueError("volatility_window must be at least 2")

    open_ = frame["open"].astype(float)
    high = frame["high"].astype(float)
    low = frame["low"].astype(float)
    close = frame["close"].astype(float)
    close_log_return = np.log(close / close.shift(1))

    if method == "close_to_close":
        annualized = close_log_return.rolling(window).std(ddof=1) * np.sqrt(TRADING_DAYS)
    elif method == "ewma_close_to_close":
        annualized = close_log_return.ewm(
            halflife=window,
            adjust=False,
            min_periods=window,
        ).std(bias=False) * np.sqrt(TRADING_DAYS)
    else:
        log_high_low = np.log(high / low)
        if method == "parkinson":
            variance = log_high_low.pow(2) / (4.0 * np.log(2.0))
        elif method == "garman_klass":
            variance = (
                0.5 * log_high_low.pow(2)
                - (2.0 * np.log(2.0) - 1.0) * np.log(close / open_).pow(2)
            ).clip(lower=0.0)
        elif method == "rogers_satchell":
            variance = (
                np.log(high / close) * np.log(high / open_)
                + np.log(low / close) * np.log(low / open_)
            ).clip(lower=0.0)
        elif method == "atr_rms_percent":
            previous_close = close.shift(1)
            true_range = pd.concat(
                [high - low, (high - previous_close).abs(), (low - previous_close).abs()],
                axis=1,
            ).max(axis=1)
            variance = (true_range / previous_close).pow(2)
        elif method == "downside":
            variance = np.minimum(close_log_return, 0.0).pow(2)
        else:
            supported = ", ".join([
                "close_to_close",
                "ewma_close_to_close",
                "parkinson",
                "garman_klass",
                "rogers_satchell",
                "atr_rms_percent",
                "downside",
            ])
            raise ValueError(f"unsupported volatility_method {method!r}; use one of: {supported}")
        annualized = np.sqrt(variance.rolling(window).mean() * TRADING_DAYS)

    return annualized.to_numpy(dtype=float)


def _execute_target(
    cash: float,
    shares: float,
    target: float,
    open_price: float,
    cost_rate: float,
) -> tuple[float, float, float, float]:
    """Rebalance to target at open; return cash, shares, turnover, cost."""
    nav_open = cash + shares * open_price
    current_value = shares * open_price
    desired_value = nav_open * target
    delta = desired_value - current_value
    if abs(delta) <= 1e-14:
        return cash, shares, 0.0, 0.0

    if delta > 0:
        # Buy value is quoted before commission.
        buy_value = min(delta, cash / (1.0 + cost_rate))
        shares += buy_value / open_price
        cash -= buy_value * (1.0 + cost_rate)
        return cash, shares, buy_value / nav_open, buy_value * cost_rate

    sell_value = min(-delta, current_value)
    shares -= sell_value / open_price
    cash += sell_value * (1.0 - cost_rate)
    return cash, shares, sell_value / nav_open, sell_value * cost_rate


def run_backtest(
    prices: pd.DataFrame | None = None,
    params: GridParams = GridParams(),
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float | int | str]]:
    """Run the causal candidate and return daily curve, trades, and metrics."""
    frame = load_prices() if prices is None else prices.copy()
    frame = frame.sort_values("date").reset_index(drop=True)
    closes = frame["close"].to_numpy(dtype=float)
    realized_volatility = _realized_volatility(frame, params)
    grid_target = params.max_exposure
    target = params.max_exposure
    # The first trading day's open is the initial full-position execution.
    pending_target: float | None = target
    pending_reason = "initial_buy"
    cash = INITIAL_CAPITAL
    shares = 0.0
    last_nav = 0.0
    previous_close: float | None = None
    rows: list[dict[str, object]] = []
    trades: list[dict[str, object]] = []

    for index, row in frame.iterrows():
        timestamp = pd.Timestamp(row["date"])
        open_price = float(row["open"])
        close_price = float(row["close"])
        day_gross_pnl = (
            shares * (open_price - previous_close) if previous_close is not None else 0.0
        )
        day_cost = 0.0

        if pending_target is not None:
            old_target = target
            cash, shares, turnover, cost = _execute_target(
                cash, shares, pending_target, open_price, params.cost_rate
            )
            if turnover > 0:
                day_cost = cost
                side = (
                    "buy"
                    if pending_reason == "initial_buy" or pending_target > old_target
                    else "sell"
                )
                trades.append({
                    "date": timestamp,
                    "asset": ASSET,
                    "side": side,
                    "reason": pending_reason,
                    "old_target": old_target,
                    "new_target": pending_target,
                    "execution_price": open_price,
                    "turnover": turnover,
                    "cost": cost,
                })
            target = pending_target
            pending_target = None

        day_gross_pnl += shares * (close_price - open_price)
        nav = cash + shares * close_price
        deposit = INITIAL_CAPITAL if index == 0 else 0.0
        daily_return = (nav - deposit) / last_nav - 1.0 if last_nav > 0 else np.nan
        etf_weight = shares * close_price / nav if nav > 0 else 0.0
        rows.append({
            "date": timestamp,
            "nav": nav,
            "return": daily_return,
            "cash": cash,
            "etf_weight": etf_weight,
            "target_weight": target,
            "grid_target_weight": grid_target,
            "realized_volatility": realized_volatility[index],
            "volatility_cap": _volatility_cap(realized_volatility[index], params),
            f"gross_pnl_{ASSET}": day_gross_pnl,
            f"transaction_cost_{ASSET}": day_cost,
            f"net_pnl_{ASSET}": day_gross_pnl - day_cost,
        })
        last_nav = nav
        previous_close = close_price

        if index < len(frame) - 1:
            next_grid_target, grid_reason, location = _target_from_close(
                closes, index, grid_target, params
            )
            cap = _volatility_cap(realized_volatility[index], params)
            next_target = min(next_grid_target, cap)
            if next_target != target:
                pending_target = next_target
                if next_grid_target != grid_target:
                    pending_reason = grid_reason
                elif next_target < target:
                    pending_reason = "volatility_cap"
                else:
                    pending_reason = "volatility_release"
                trades.append({
                    "date": timestamp,
                    "asset": ASSET,
                    "side": "signal",
                    "reason": pending_reason,
                    "old_target": target,
                    "new_target": next_target,
                    "old_grid_target": grid_target,
                    "new_grid_target": next_grid_target,
                    "signal_close": close_price,
                    "range_location": location,
                    "realized_volatility": realized_volatility[index],
                    "volatility_cap": cap,
                })
            grid_target = next_grid_target

    daily = pd.DataFrame(rows).set_index("date")
    trade_frame = pd.DataFrame(trades)
    returns = daily["return"].dropna().astype(float)
    curve = (1.0 + returns).cumprod()
    drawdown = curve / curve.cummax() - 1.0
    stdev = float(returns.std(ddof=1)) if len(returns) > 1 else 0.0
    years = len(returns) / TRADING_DAYS
    metrics: dict[str, float | int | str] = {
        "asset": ASSET,
        "start": str(daily.index.min().date()),
        "end": str(daily.index.max().date()),
        "observations": int(len(returns)),
        "final_nav": float(daily["nav"].iloc[-1]),
        "total_return": float(daily["nav"].iloc[-1] / INITIAL_CAPITAL - 1.0),
        "annualized_return": float((daily["nav"].iloc[-1] / INITIAL_CAPITAL) ** (1.0 / years) - 1.0),
        "max_drawdown": float(drawdown.min()),
        "sharpe": float(returns.mean() / stdev * np.sqrt(TRADING_DAYS)) if stdev > 0 else 0.0,
        "average_exposure": float(daily["etf_weight"].mean()),
        "signal_count": int((trade_frame["side"] == "signal").sum()) if not trade_frame.empty else 0,
        "execution_count": int((trade_frame["side"] != "signal").sum()) if not trade_frame.empty else 0,
        "total_cost": float(trade_frame.loc[trade_frame["side"] != "signal", "cost"].sum()) if not trade_frame.empty else 0.0,
    }
    return daily, trade_frame, metrics


def main() -> None:
    daily, trades, metrics = run_backtest()
    output = Path(__file__).parent / "deliverable"
    output.mkdir(parents=True, exist_ok=True)
    daily.to_csv(output / "grid_reproduction_daily.csv")
    trades.to_csv(output / "grid_reproduction_trades.csv", index=False)
    pd.Series(metrics).to_json(output / "grid_reproduction_metrics.json", force_ascii=False, indent=2)
    print("params", asdict(GridParams()))
    for key, value in metrics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
