"""Focused binary high-state position search after the broad W40 stage."""

from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path

from research.momentum_defender_occam_position import (
    CONTRARIAN_TREND,
    FIXED_WEIGHT,
    FROZEN_CHAMPION,
    RANGE_HIGH_CUT,
    VOLATILITY_HIGH_CUT,
    PositionSpec,
)
from research.run_momentum_defender_w40_occam_position_search import (
    _load,
    run_experiment,
)


DEFAULT_CONFIG = Path(
    "research/configs/momentum_defender_w40_occam_position_focused.yaml"
)
DEFAULT_OUTPUT = Path(
    "experiments/20260825_momentum_defender_w40_occam_position_focused"
)


def _focused_specs(config: dict) -> list[PositionSpec]:
    values = config["focused_position_families"]
    result = [PositionSpec(FROZEN_CHAMPION), PositionSpec(FIXED_WEIGHT, level=1.0)]
    family = values[RANGE_HIGH_CUT]
    result.extend(
        PositionSpec(
            RANGE_HIGH_CUT,
            str(source),
            int(window),
            float(threshold),
            float(fallback),
        )
        for source, window, threshold, fallback in product(
            family["signal_sources"],
            family["windows"],
            family["high_thresholds"],
            family["high_state_equity_weights"],
        )
    )
    family = values[VOLATILITY_HIGH_CUT]
    result.extend(
        PositionSpec(
            VOLATILITY_HIGH_CUT,
            str(source),
            int(window),
            float(quantile),
            float(fallback),
        )
        for source, window, quantile, fallback in product(
            family["signal_sources"],
            family["windows"],
            family["strict_lag_rolling_504_quantiles"],
            family["high_state_equity_weights"],
        )
    )
    family = values[CONTRARIAN_TREND]
    result.extend(
        PositionSpec(
            CONTRARIAN_TREND,
            str(source),
            int(window),
            None,
            float(fallback),
        )
        for source, window, fallback in product(
            family["signal_sources"],
            family["windows"],
            family["positive_trend_equity_weights"],
        )
    )
    return list({spec.candidate_id: spec for spec in result}.values())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config_path = args.config.resolve()
    specs = _focused_specs(_load(config_path))
    audit = run_experiment(
        root,
        config_path,
        args.output.resolve(),
        specs_override=specs,
    )
    print(json.dumps(audit["search"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
