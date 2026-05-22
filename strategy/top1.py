"""Top-1 策略 — 全仓持有单因子得分最高的资产。"""

from __future__ import annotations

from strategy.base import BaseStrategy


class Top1(BaseStrategy):
    def generate_weights(
        self,
        factor_values: dict[str, dict[str, float]],
        current_weights: dict[str, float] | None = None,
    ) -> dict[str, float]:
        if not factor_values:
            return {}

        factor_configs = self.config.get("factors", [])
        if not factor_configs:
            return {}

        # 只取第一个因子的值做排序
        fname = factor_configs[0]["name"]
        flip = factor_configs[0].get("direction_flip", False)

        scored = {
            asset: vals[fname]
            for asset, vals in factor_values.items()
            if fname in vals
        }
        if not scored:
            return {}

        threshold = float(self.config.get("hysteresis_threshold", 0.0))
        incumbent = self._incumbent_asset(current_weights)
        if threshold > 0 and incumbent in scored:
            challengers = {
                asset: score
                for asset, score in scored.items()
                if asset != incumbent
            }
            if not challengers:
                return {incumbent: 1.0}

            challenger = (
                min(challengers, key=challengers.get)
                if flip
                else max(challengers, key=challengers.get)
            )
            challenger_score = challengers[challenger]
            incumbent_score = scored[incumbent]
            clears_threshold = (
                challenger_score < incumbent_score - threshold
                if flip
                else challenger_score > incumbent_score + threshold
            )
            return {challenger if clears_threshold else incumbent: 1.0}

        if flip:
            best = min(scored, key=scored.get)
        else:
            best = max(scored, key=scored.get)

        return {best: 1.0}

    @staticmethod
    def _incumbent_asset(
        current_weights: dict[str, float] | None,
    ) -> str | None:
        if not current_weights or len(current_weights) != 1:
            return None
        incumbent, weight = next(iter(current_weights.items()))
        return incumbent if weight > 0 else None
