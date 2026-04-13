"""Momentum rotation strategy — rank assets by weighted factor scores."""

from __future__ import annotations

from strategy.base import BaseStrategy


class MomentumRotation(BaseStrategy):
    def generate_weights(
        self, factor_values: dict[str, dict[str, float]]
    ) -> dict[str, float]:
        if not factor_values:
            return {}

        assets = list(factor_values.keys())
        if len(assets) == 1:
            return {assets[0]: 1.0}

        factor_configs = self.config.get("factors", [])
        total_scores: dict[str, float] = {a: 0.0 for a in assets}

        for fc in factor_configs:
            fname = fc["name"]
            weight = fc["weight"]
            flip = fc.get("direction_flip", False)

            values = [(a, factor_values[a][fname]) for a in assets]
            values.sort(key=lambda x: x[1])
            n = len(values)

            for rank_idx, (asset, _) in enumerate(values):
                rank = rank_idx + 1  # 1-based
                if flip:
                    rank = n - rank + 1
                total_scores[asset] += weight * rank

        total = sum(total_scores.values())
        if total == 0:
            return {a: 1.0 / len(assets) for a in assets}

        return {a: score / total for a, score in total_scores.items()}
