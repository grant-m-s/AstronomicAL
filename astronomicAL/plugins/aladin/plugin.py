from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

from astronomicAL.platform.plugins import PluginManifest


PLUGIN_ID = "astro.aladin"
_SPLIT_PACKAGE = "astronomicAL.plugins.aladin"


def _ensure_split_package() -> None:
    """Allow this plugin to work when PluginManager imports plugin.py directly.

    Local plugin discovery can import this file under a synthetic module name.
    Relative imports then fail unless we expose the real package path.
    """

    package_path = Path(__file__).resolve().parent
    existing = sys.modules.get(_SPLIT_PACKAGE)

    if existing is not None:
        existing_path = getattr(existing, "__path__", None)
        if existing_path is None:
            existing.__path__ = [str(package_path)]
        elif str(package_path) not in list(existing_path):
            try:
                existing.__path__.append(str(package_path))
            except Exception:
                existing.__path__ = [str(package_path)]
        return

    package = types.ModuleType(_SPLIT_PACKAGE)
    package.__file__ = str(package_path / "__init__.py")
    package.__path__ = [str(package_path)]
    package.__package__ = _SPLIT_PACKAGE
    sys.modules[_SPLIT_PACKAGE] = package


def _impl(module_name: str):
    _ensure_split_package()
    return importlib.import_module(f"{_SPLIT_PACKAGE}.{module_name}")


manifest = PluginManifest(
    id=PLUGIN_ID,
    name="Aladin Lite",
    version="0.1.0",
    description=(
        "Aladin Lite sky viewer for the active AstronomicAL dataset. The panel "
        "centres on the focused row using semantic `coords.ra` and `coords.dec` "
        "mappings and refreshes on platform selection changes."
    ),
    requires=["panel"],
    capabilities=["panel", "datasets", "selection"],
    tags=["astronomy", "aladin", "hips", "sky-viewer", "coordinates"],
)


def register(api) -> None:
    panel_mod = _impl("panel")

    api.register_panel(
        id="panel",
        title="Aladin Lite",
        factory=panel_mod.create_aladin_panel,
        description=(
            "Interactive Aladin Lite sky viewer centred on the focused source. "
            "Requires mapped `record_id`, `coords.ra` and `coords.dec` columns."
        ),
        category="Sky Viewers",
        icon="globe",
        tags=["astronomy", "aladin", "hips", "sky-viewer", "selection"],
        required_mappings=["record_id", "coords.ra", "coords.dec"],
        produces=["astro.aladin.updated"],
        default_layout={"x": 0, "y": 0, "w": 6, "h": 5},
        state_version=1,
        persist_layout=True,
        persist_state=True,
        restore_policy="best_effort",
    )