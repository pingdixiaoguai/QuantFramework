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
            asset_weights = factor_config.get("asset_weights", {})
            direction_flip = bool(factor_config.get("direction_flip", False))
            center_rank = bool(factor_config.get("center_rank", False))
            values = sorted(
                ((asset, factor_values[asset][name]) for asset in assets),
                key=lambda item: item[1],
            )
            asset_count = len(values)
            rank_center = (asset_count + 1.0) / 2.0 if center_rank else 0.0

            for rank_index, (asset, _) in enumerate(values, start=1):
                rank = asset_count - rank_index + 1 if direction_flip else rank_index
                asset_weight = float(asset_weights.get(asset, weight))
                total_scores[asset] += asset_weight * (rank - rank_center)

        best = max(total_scores, key=total_scores.get)
        return {best: 1.0}
