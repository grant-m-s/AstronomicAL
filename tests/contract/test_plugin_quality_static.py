from __future__ import annotations

import ast
import importlib
import os
import time
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PLUGIN_ROOT = PROJECT_ROOT / "astronomicAL" / "plugins"


def plugin_python_files() -> list[Path]:
    if not PLUGIN_ROOT.exists():
        return []
    return sorted(
        path
        for path in PLUGIN_ROOT.rglob("*.py")
        if "__pycache__" not in path.parts
    )


def plugin_entrypoint_files() -> list[Path]:
    return [
        path
        for path in plugin_python_files()
        if path.name == "plugin.py"
    ]


def module_name_from_path(path: Path) -> str:
    rel = path.relative_to(PROJECT_ROOT).with_suffix("")
    return ".".join(rel.parts)


def parse_file(path: Path) -> ast.AST:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def dotted_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = dotted_name(node.value)
        if base:
            return f"{base}.{node.attr}"
        return node.attr
    return None


FORBIDDEN_IMPORT_PREFIXES = (
    "astronomicAL.config",
    "astronomicAL.dashboard",
)

FORBIDDEN_CONFIG_WRITE_PREFIXES = (
    "config.",
    "context.config.",
    "self.context.config.",
)

FORBIDDEN_DIRECT_LAYOUT_PATTERNS = (
    ".main.append(",
    ".main.extend(",
    "template.main.append(",
    "template.main.extend(",
    "react.main.append(",
    "react.main.extend(",
)


@pytest.mark.plugin_quality
@pytest.mark.contract
@pytest.mark.parametrize("path", plugin_python_files(), ids=lambda p: str(p.relative_to(PROJECT_ROOT)))
def test_plugin_files_do_not_import_legacy_global_modules(path: Path) -> None:
    """
    Opt-in quality test.

    New plugin code should use context/platform services instead of importing
    old global config or old dashboard modules.
    """

    tree = parse_file(path)
    violations: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                for prefix in FORBIDDEN_IMPORT_PREFIXES:
                    if alias.name == prefix or alias.name.startswith(prefix + "."):
                        violations.append(f"import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for prefix in FORBIDDEN_IMPORT_PREFIXES:
                if module == prefix or module.startswith(prefix + "."):
                    violations.append(f"from {module} import ...")

    assert not violations, (
        f"{path.relative_to(PROJECT_ROOT)} imports legacy global modules:\n"
        + "\n".join(f"  - {item}" for item in violations)
        + "\nUse context.datasets/context.selection/context.artifacts/"
        "context.events/context.jobs/context.workspace/context.services instead."
    )


@pytest.mark.plugin_quality
@pytest.mark.contract
@pytest.mark.parametrize("path", plugin_python_files(), ids=lambda p: str(p.relative_to(PROJECT_ROOT)))
def test_plugin_files_do_not_write_runtime_state_to_config(path: Path) -> None:
    """
    Opt-in quality test.

    context.config is a migration bridge, not the home for new runtime state.
    This catches obvious assignments such as context.config.current_row = ...
    """

    tree = parse_file(path)
    violations: list[str] = []

    assignment_nodes = (
        ast.Assign,
        ast.AnnAssign,
        ast.AugAssign,
    )

    for node in ast.walk(tree):
        if not isinstance(node, assignment_nodes):
            continue

        targets = []
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
        else:
            targets = [node.target]

        for target in targets:
            name = dotted_name(target)
            if not name:
                continue

            for prefix in FORBIDDEN_CONFIG_WRITE_PREFIXES:
                if name.startswith(prefix):
                    violations.append(f"{name} = ...")

    assert not violations, (
        f"{path.relative_to(PROJECT_ROOT)} writes runtime state to config:\n"
        + "\n".join(f"  - {item}" for item in violations)
        + "\nMove focus/selection to context.selection, derived outputs to "
        "context.artifacts, source data to context.datasets, and live clients "
        "to context.services."
    )


@pytest.mark.plugin_quality
@pytest.mark.contract
@pytest.mark.parametrize("path", plugin_python_files(), ids=lambda p: str(p.relative_to(PROJECT_ROOT)))
def test_plugin_files_do_not_directly_mutate_template_main_layout(path: Path) -> None:
    """
    Opt-in quality test.

    Plugins should add/remove visible panels via context.workspace, not by
    directly appending to a Panel template/grid from arbitrary code.
    """

    text = path.read_text(encoding="utf-8")
    violations = [
        pattern
        for pattern in FORBIDDEN_DIRECT_LAYOUT_PATTERNS
        if pattern in text
    ]

    assert not violations, (
        f"{path.relative_to(PROJECT_ROOT)} appears to mutate layout directly:\n"
        + "\n".join(f"  - contains {pattern!r}" for pattern in violations)
        + "\nUse context.workspace.add_panel(...) / remove_panel(...) instead."
    )


@pytest.mark.plugin_quality
@pytest.mark.contract
@pytest.mark.parametrize("path", plugin_entrypoint_files(), ids=lambda p: str(p.relative_to(PROJECT_ROOT)))
def test_plugin_entrypoints_import_under_budget(path: Path) -> None:
    """
    Opt-in quality test for plugin developers.

    Plugin import should be cheap. Heavy imports, remote clients, and expensive
    setup should usually move into panel factories, services, or jobs.
    """

    module_name = module_name_from_path(path)
    budget_seconds = float(
        os.environ.get("ASTRONOMICAL_PLUGIN_IMPORT_BUDGET_SECONDS", "0.75")
    )

    start = time.perf_counter()
    importlib.import_module(module_name)
    elapsed = time.perf_counter() - start

    assert elapsed <= budget_seconds, (
        f"{module_name} took {elapsed:.3f}s to import; "
        f"budget is {budget_seconds:.3f}s. "
        "Move heavy work out of import time."
    )


@pytest.mark.plugin_quality
@pytest.mark.contract
@pytest.mark.parametrize("path", plugin_entrypoint_files(), ids=lambda p: str(p.relative_to(PROJECT_ROOT)))
def test_plugin_entrypoints_expose_manifest_and_register(path: Path) -> None:
    """
    Opt-in quality test.

    Every plugin.py entrypoint should expose a manifest and register(api).
    """

    module_name = module_name_from_path(path)
    module = importlib.import_module(module_name)

    assert hasattr(module, "manifest"), f"{module_name} is missing manifest"
    assert hasattr(module, "register"), f"{module_name} is missing register(api)"
    assert callable(module.register), f"{module_name}.register must be callable"

    from astronomicAL.platform.plugins.manifest import PluginManifest, coerce_manifest

    manifest = coerce_manifest(module.manifest)

    assert isinstance(manifest, PluginManifest)

    assert manifest.id
    assert manifest.name
    assert manifest.version
    assert " " not in manifest.id, "plugin manifest id should be stable and space-free"
