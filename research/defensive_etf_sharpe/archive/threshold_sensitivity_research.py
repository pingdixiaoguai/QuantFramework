"""Fine-grid search around the 10% monthly-capped portfolio-drift threshold."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .factor_research import _extended_metrics
from .rebalance_timing import daily_reversal_targets
from .reduced_pool_research import REDUCED_TARGET
from .strategy import CASH_ASSET, load_confirmed_market
from .threshold_rebalance import TriggerSpec, simulate_threshold_rebalance


OUTPUT = Path(__file__).parent / "threshold_rebalance_experiments"
THRESHOLDS = tuple(np.arange(0.075, 0.125 + 0.0001, 0.005).round(3))


def _name(threshold: float) -> str:
    return f"portfolio_drift_{threshold * 100:g}pct_monthly_cap1".replace(".", "p")


def run() -> pd.DataFrame:
    _, market = load_confirmed_market()
    daily_targets = daily_reversal_targets(market, REDUCED_TARGET)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    results = {}

    for threshold in THRESHOLDS:
        spec = TriggerSpec(
            name=_name(threshold),
            kind="portfolio_drift",
            threshold=float(threshold),
            description=f"组合单边偏离达到{threshold:.1%}，每月最多再平衡一次",
            max_rebalances_per_month=1,
        )
        result = simulate_threshold_rebalance(
            market,
            daily_targets,
            spec,
            initial_target=REDUCED_TARGET,
            cash_asset=CASH_ASSET,
            min_rebalance_notional=10_000.0,
        )
        results[float(threshold)] = result
        metrics = _extended_metrics(result)
        rebalance_trades = result.trades.loc[result.trades["reason"] == "threshold_rebalance"]
        rebalance_dates = (
            pd.to_datetime(rebalance_trades["date"]).nunique()
            if not rebalance_trades.empty
            else 0
        )
        rows.append({
            "threshold": float(threshold),
            **metrics,
            "trigger_signals": len(result.signals),
            "rebalance_dates": rebalance_dates,
            "deposit_trade_count": int((result.trades["reason"] == "deposit_invest").sum()),
            "rebalance_trade_count": int(
                (result.trades["reason"] == "threshold_rebalance").sum()
            ),
            "average_cash_weight": float(result.daily["cash_weight"].mean()),
        })

    metrics = pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)
    eligible = metrics.loc[metrics["meets_5pct_return_floor"]]
    best_row = eligible.sort_values(
        ["sharpe", "annualized_return"], ascending=[False, False]
    ).iloc[0]
    metrics["is_best"] = np.isclose(metrics["threshold"], best_row["threshold"])
    metrics.to_csv(OUTPUT / "portfolio_drift_threshold_sensitivity.csv", index=False)

    best_threshold = float(best_row["threshold"])
    best = results[best_threshold]
    best.daily.to_csv(OUTPUT / "threshold_sensitivity_best_daily.csv")
    best.trades.to_csv(OUTPUT / "threshold_sensitivity_best_trades.csv", index=False)
    best.signals.to_csv(
        OUTPUT / "threshold_sensitivity_best_trigger_signals.csv", index=False
    )
    return metrics


if __name__ == "__main__":
    print(run().to_string(index=False))
