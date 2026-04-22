"""Top-N 策略 — 等权持有单因子得分前 N 名的资产。"""

from __future__ import annotations

from strategy.base import BaseStrategy


class TopN(BaseStrategy):
    def generate_weights(
        self, factor_values: dict[str, dict[str, float]]
    ) -> dict[str, float]:
        if not factor_values:
            return {}

        factor_configs = self.config.get("factors", [])
        if not factor_configs:
            return {}

        fname = factor_configs[0]["name"]
        flip = factor_configs[0].get("direction_flip", False)

        scored = {
            asset: vals[fname]
            for asset, vals in factor_values.items()
            if fname in vals
        }
        if not scored:
            return {}

        top_n = self.config.get("top_n", 5)
        k = min(top_n, len(scored))

        ranked = sorted(scored.items(), key=lambda kv: kv[1], reverse=not flip)
        picks = [asset for asset, _ in ranked[:k]]

        weight = 1.0 / k
        return {asset: weight for asset in picks}
