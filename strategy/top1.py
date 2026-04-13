"""Top-1 策略 — 全仓持有单因子得分最高的资产。"""

from __future__ import annotations

from strategy.base import BaseStrategy


class Top1(BaseStrategy):
    def generate_weights(
        self, factor_values: dict[str, dict[str, float]]
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

        if flip:
            best = min(scored, key=scored.get)
        else:
            best = max(scored, key=scored.get)

        return {best: 1.0}
