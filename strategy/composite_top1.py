"""Top-1 strategy selected by a weighted blend of factor ranks."""

from __future__ import annotations

from strategy.base import BaseStrategy


class CompositeTop1(BaseStrategy):
    """Allocate fully to the asset with the highest composite rank score."""

    def generate_weights(
        self, factor_values: dict[str, dict[str, float]]
    ) -> dict[str, float]:
        if not factor_values:
            return {}

        factor_configs = self.config.get("factors", [])
        if not factor_configs:
            return {}

        assets = list(factor_values)
        total_scores = {asset: 0.0 for asset in assets}

        for factor_config in factor_configs:
            name = factor_config["name"]
            weight = float(factor_config["weight"])
            direction_flip = bool(factor_config.get("direction_flip", False))
            values = sorted(
                ((asset, factor_values[asset][name]) for asset in assets),
                key=lambda item: item[1],
            )
            asset_count = len(values)

            for rank_index, (asset, _) in enumerate(values, start=1):
                rank = asset_count - rank_index + 1 if direction_flip else rank_index
                total_scores[asset] += weight * rank

        best = max(total_scores, key=total_scores.get)
        return {best: 1.0}
