from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

from astronomicAL.platform.plugins import PluginManifest

PLUGIN_ID = "astro.euclid_cutout"
_SPLIT_PACKAGE = "astronomicAL.plugins.euclid_cutout"


def _ensure_split_package() -> None:
    """Allow this plugin to work when loaded as a local plugin file.

    The PluginManager can import ``plugin.py`` under a synthetic module name.
    In that mode normal relative imports fail, so mirror the pattern used by the
    bundled visualisation plugin and ensure the real package path exists in
    ``sys.modules`` before importing implementation modules.
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
    name="Euclid Cutout",
    version="0.1.0",
    description=(
        "Euclid archive cutout viewer for the active AstronomicAL dataset. "
        "The plugin resolves RA/Dec through semantic dataset mappings, reacts "
        "to platform focus changes, runs archive retrieval through JobManager, "
        "and publishes cutout artifacts for other panels."
    ),
    requires=["astroquery", "astropy", "reproject", "mocpy"],
    capabilities=["panel", "service", "artifact_viewer", "datasets", "selection", "jobs"],
    tags=["astronomy", "euclid", "cutout", "image", "wcs"],
)


def register(api) -> None:
    panel_mod = _impl("panel")
    service_mod = _impl("service")

    api.register_service(
        key="runtime",
        factory=service_mod.create_euclid_runtime,
        lazy=True,
        replace=True,
        description=(
            "Runtime service wrapping the legacy Euclid cutout archive/WCS/reprojection code."
        ),
        optional_requires=["astroquery", "astropy", "reproject", "mocpy"],
    )

    api.register_panel(
        id="panel",
        title="Euclid Cutout",
        factory=panel_mod.create_euclid_cutout_panel,
        description=(
            "Fetch, render and inspect Euclid VIS/NIR cutouts for the focused row. "
            "Requires mapped `record_id`, `coords.ra` and `coords.dec` columns."
        ),
        category="Images / Cutouts",
        icon="image",
        tags=["astronomy", "euclid", "cutout", "image", "selection"],
        required_mappings=["record_id", "coords.ra", "coords.dec"],
        uses_services=[f"{PLUGIN_ID}.runtime"],
        produces=["astro.cutout.euclid", "astro.cutout.updated", "astro.cutout.running"],
        default_layout={"x": 0, "y": 0, "w": 5, "h": 5},
        state_version=1,
        persist_layout=True,
        persist_state=True,
        restore_policy="best_effort",
        optional_requires=["astroquery", "astropy", "reproject", "mocpy"],
    )

    api.register_artifact_viewer(
        artifact_type="astro.cutout.euclid",
        viewer_factory=panel_mod.create_euclid_cutout_artifact_viewer,
        id="viewer",
        title="Euclid Cutout Viewer",
        description="Display Euclid cutout artifacts produced by the Euclid Cutout panel.",
        priority=50,
        default=True,
        optional_requires=["holoviews", "panel"],
    )