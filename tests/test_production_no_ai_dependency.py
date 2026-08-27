from __future__ import annotations

import ast
from pathlib import Path
import tomllib


ROOT = Path(__file__).resolve().parents[1]
RUNTIME_DIRS = (
    "backtest",
    "data",
    "defender",
    "execution",
    "factors",
    "notification",
    "research",
    "standardization",
    "strategy",
)
RUNTIME_ENTRY_POINTS = (
    "run_daily.py",
    "run_daily_momentum_defender.py",
)
FORBIDDEN_MODEL_MODULES = (
    "openai",
    "anthropic",
    "cohere",
    "mistralai",
    "langchain",
    "langchain_openai",
    "google.generativeai",
    "transformers",
)


def _matches_forbidden(module: str) -> bool:
    return any(
        module == forbidden or module.startswith(f"{forbidden}.")
        for forbidden in FORBIDDEN_MODEL_MODULES
    )


def test_production_python_path_has_no_model_service_imports() -> None:
    paths = [ROOT / entry for entry in RUNTIME_ENTRY_POINTS]
    for directory in RUNTIME_DIRS:
        paths.extend((ROOT / directory).rglob("*.py"))
    violations = []
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            modules = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules = [node.module]
            for module in modules:
                if _matches_forbidden(module):
                    violations.append(f"{path.relative_to(ROOT)}: {module}")
    assert violations == []


def test_production_dependencies_have_no_model_sdk() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = project["project"]["dependencies"]
    normalized = [dependency.lower().split("[", 1)[0] for dependency in dependencies]
    assert not any(
        dependency.startswith(
            ("openai", "anthropic", "cohere", "mistralai", "langchain", "transformers")
        )
        for dependency in normalized
    )
