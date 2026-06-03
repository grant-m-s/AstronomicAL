from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

from astronomicAL.platform.plugins import PluginManifest


PLUGIN_ID = "astro.spectra"
_SPLIT_PACKAGE = "astronomicAL.plugins.spectra"


def _ensure_split_package() -> None:
    """Allow this plugin to work when PluginManager imports plugin.py directly.

    Local plugin discovery can import this file under a synthetic module name.
    Relative imports then fail unless we also expose the real package path.
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
    name="Astronomy Spectra",
    version="0.1.0",
    description=(
        "DESI, SDSS/BOSS and Euclid spectrum panels for AstronomicAL's plugin "
        "platform. The plugin resolves the active row from DatasetManager, "
        "reacts to SelectionManager focus changes, runs archive retrieval "
        "through JobManager, and publishes spectrum and coordinate artifacts."
    ),
    requires=[
        "numpy",
        "pandas",
        "panel",
        "holoviews",
        "matplotlib",
        "astropy",
        "astroquery",
        "sparclclient",
        "mocpy",
        "requests",
    ],
    capabilities=["panel", "service", "artifact_viewer", "datasets", "selection", "jobs"],
    tags=["astronomy", "spectra", "desi", "sdss", "boss", "euclid", "sparcl"],
)


def register(api) -> None:
    panel_mod = _impl("panel")
    service_mod = _impl("service")

    api.register_service(
        key="runtime",
        factory=service_mod.create_spectra_runtime,
        lazy=True,
        replace=True,
        description=(
            "Runtime service wrapping legacy DESI/SDSS SPARCL and Euclid "
            "spectrum retrieval classes."
        ),
        optional_requires=[
            "numpy",
            "pandas",
            "astropy",
            "astroquery",
            "sparclclient",
            "mocpy",
            "requests",
        ],
    )

    common_produces = [
        "astro.spectra",
        "astro.spectra.updated",
        "astro.spectra.running",
        "astro.coords",
        "astro.coords.updated",
    ]

    api.register_panel(
        id="desi",
        title="DESI Spectra",
        factory=panel_mod.make_spectra_panel_factory("DESI"),
        description="Retrieve and inspect DESI spectra for the focused source.",
        category="Spectra",
        icon="chart-line",
        tags=["astronomy", "spectra", "desi", "sparcl"],
        required_mappings=["record_id", "coords.ra", "coords.dec"],
        optional_mappings=["spectra.desi_target_id", "target_label"],
        uses_services=[f"{PLUGIN_ID}.runtime"],
        produces=common_produces,
        default_layout={"x": 0, "y": 0, "w": 6, "h": 5},
        state_version=1,
        persist_layout=True,
        persist_state=True,
        restore_policy="best_effort",
        optional_requires=["sparclclient", "astropy"],
    )

    api.register_panel(
        id="sdss",
        title="SDSS Spectra",
        factory=panel_mod.make_spectra_panel_factory("SDSS"),
        description="Retrieve and inspect SDSS/BOSS spectra for the focused source.",
        category="Spectra",
        icon="chart-line",
        tags=["astronomy", "spectra", "sdss", "boss", "sparcl"],
        required_mappings=["record_id", "coords.ra", "coords.dec"],
        optional_mappings=["spectra.sdss_target_id", "target_label"],
        uses_services=[f"{PLUGIN_ID}.runtime"],
        produces=common_produces,
        default_layout={"x": 0, "y": 0, "w": 6, "h": 5},
        state_version=1,
        persist_layout=True,
        persist_state=True,
        restore_policy="best_effort",
        optional_requires=["sparclclient", "astropy"],
    )

    api.register_panel(
        id="euclid",
        title="Euclid Spectra",
        factory=panel_mod.make_spectra_panel_factory("EuclidSpec"),
        description="Retrieve and inspect Euclid Q1 spectra for the focused source.",
        category="Spectra",
        icon="chart-line",
        tags=["astronomy", "spectra", "euclid"],
        required_mappings=["record_id", "coords.ra", "coords.dec"],
        optional_mappings=["spectra.euclid_source_id", "target_label"],
        uses_services=[f"{PLUGIN_ID}.runtime", "euclid.client"],
        produces=common_produces,
        default_layout={"x": 0, "y": 0, "w": 6, "h": 5},
        state_version=1,
        persist_layout=True,
        persist_state=True,
        restore_policy="best_effort",
        optional_requires=["astroquery", "astropy", "mocpy", "requests"],
    )

    api.register_artifact_viewer(
        artifact_type="astro.spectra",
        viewer_factory=panel_mod.create_spectra_artifact_viewer,
        id="viewer",
        title="Spectrum Viewer",
        description="Display spectrum artifacts produced by the astronomy spectra plugin.",
        priority=50,
        default=True,
        optional_requires=["holoviews", "panel"],
    )