"""Offline tests for exporting rolling OHLC ER research checkpoints."""

from datetime import date

from scripts.fit_rolling_ohlc_er import apply_checkpoint
from strategy.rolling_ohlc_er import RollingWeights


def test_checkpoint_exports_weights_and_rebalance_semantics():
    output = {
        "factors": [{"name": "ohlc_quality_momentum", "params": {}}],
        "rebalance_days": 1,
    }
    research = {"history_days": 1008, "rebalance_days": 5}
    state = RollingWeights(
        effective_date=date(2026, 7, 1),
        training_start=date(2022, 5, 5),
        training_end=date(2026, 6, 30),
        values=(0.8, 0.2, 0.03, 0.19),
    )

    result = apply_checkpoint(output, research, state)

    assert result["rebalance_days"] == 5
    assert result["factors"][0]["params"]["weights"]["close"] == 0.8
    assert result["parameter_checkpoint"]["rebalance_days"] == 5
    assert result["parameter_checkpoint"]["training_end"] == "2026-06-30"
