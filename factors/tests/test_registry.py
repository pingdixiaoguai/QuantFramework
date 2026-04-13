"""Tests for factors.registry."""

import pytest

from factors.registry import load_registered_factors


class TestLoadBothFactors:
    def test_returns_momentum_and_volatility(self):
        facs = load_registered_factors()
        assert "momentum" in facs
        assert "volatility" in facs
        assert "quality_momentum" in facs
        assert len(facs) == 3
        for name, fac in facs.items():
            assert "METADATA" in fac
            assert "compute" in fac
            assert callable(fac["compute"])


class TestMissingMetadataFieldRaises:
    def test_missing_direction_raises(self, monkeypatch):
        import factors.momentum as mod

        original = mod.METADATA.copy()
        patched = {k: v for k, v in original.items() if k != "direction"}
        monkeypatch.setattr(mod, "METADATA", patched)

        with pytest.raises(RuntimeError, match="registry load failed"):
            load_registered_factors()
