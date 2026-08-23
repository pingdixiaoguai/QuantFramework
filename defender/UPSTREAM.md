# Upstream provenance

- Repository: https://github.com/Castle47/Defender
- Branch requested: `main`
- Fetched commit: `b5e34191a7d445521de330e998bfe0804d6ebd43`
- Commit subject: `Codex/promote 2013 listing aware strategy (#2)`
- Fetched on: 2026-08-22

The production dependency closure was copied without logic changes:

- `current_strategy.py`
- `defender_opt_v2.py`
- `engine.py`
- `grid_reproduction.py`
- `market_risk_overlay.py`
- `relative_defender.py`
- `relative_defender_champion.py`
- `relative_defender_rotation.py`
- `relative_defender_rotation_2013_report.py`
- `relative_defender_rotation_2013_export.py`
- `configs/relative_defender_rotation.yaml`

QuantFramework deliberately reuses its existing `data.store` implementation;
the upstream data layer is schema-identical and was not duplicated.
