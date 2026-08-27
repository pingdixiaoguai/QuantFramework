"""Run the confirmed eight-ETF momentum-times-ER research procedure."""

from __future__ import annotations

from pathlib import Path

from .strategy import load_confirmed_market, run_score_rotation


ROOT = Path(__file__).parent
OUTPUT_DIR = ROOT / "outputs"


def run():
    universe, market = load_confirmed_market()
    strategy = run_score_rotation(universe, market)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    strategy.daily.to_csv(OUTPUT_DIR / "strategy_daily.csv")
    strategy.trades.to_csv(OUTPUT_DIR / "strategy_trades.csv", index=False)
    strategy.signals.to_csv(OUTPUT_DIR / "strategy_signals.csv", index=False)
    for stale_name in ("walk_forward.csv", "profile_metrics.csv"):
        (OUTPUT_DIR / stale_name).unlink(missing_ok=True)
    return strategy


if __name__ == "__main__":
    result = run()
    print(result.metrics())
