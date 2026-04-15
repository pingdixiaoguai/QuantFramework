"""Factor registry: load registered factors from registry.yaml."""

import importlib
from pathlib import Path
from types import ModuleType

import yaml

_REQUIRED_METADATA_FIELDS = {
    "name", "author", "version", "params", "min_history", "direction", "description",
}
_VALID_DIRECTIONS = {"higher_better", "lower_better"}

_REGISTRY_PATH = Path(__file__).parent / "registry.yaml"


def load_registered_factors() -> dict[str, dict]:
    """Load all factors listed in registry.yaml.

    Returns dict mapping factor name to {"METADATA": dict, "compute": callable}.
    """
    with open(_REGISTRY_PATH, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    factors: dict[str, dict] = {}

    for entry in config.get("factors", []):
        module_path = entry["module"]
        try:
            mod: ModuleType = importlib.import_module(module_path)
        except ImportError as exc:
            raise RuntimeError(
                f"registry load failed: cannot import '{module_path}': {exc}"
            ) from exc

        if not hasattr(mod, "METADATA") or not hasattr(mod, "compute"):
            raise RuntimeError(
                f"registry load failed: '{module_path}' missing METADATA or compute"
            )

        metadata = mod.METADATA
        missing = _REQUIRED_METADATA_FIELDS - set(metadata.keys())
        if missing:
            raise RuntimeError(
                f"registry load failed: '{module_path}' METADATA missing fields: {missing}"
            )

        if metadata["direction"] not in _VALID_DIRECTIONS:
            raise RuntimeError(
                f"registry load failed: '{module_path}' direction "
                f"'{metadata['direction']}' not in {_VALID_DIRECTIONS}"
            )

        factors[metadata["name"]] = {
            "METADATA": metadata,
            "compute": mod.compute,
        }

    return factors
