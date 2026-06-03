from __future__ import annotations

from importlib import import_module

from astronomicAL.platform.plugins import PluginManifest

PLUGIN_ID = "astro.sed"

# Do not use __package__ here.
# The plugin manager may load plugin.py directly from a file path, in which case
# __package__ can be empty and import_module(f"{__package__}.panel") becomes
# import_module(".panel"), which fails.
#
# Because this is a bundled plugin under astronomicAL/plugins/sed, use the
# absolute package path instead.
PLUGIN_PACKAGE = "astronomicAL.plugins.sed"


def _impl(module: str):
    return import_module(f"{PLUGIN_PACKAGE}.{module}")


manifest = PluginManifest(
    id=PLUGIN_ID,
    name="Broadband SED",
    version="0.1.0",
    description=(
        "Broadband spectral-energy-distribution panel for the active AstronomicAL "
        "dataset. The plugin uses semantic record-id mappings, reacts to platform "
        "focus changes, reads the same JSON photometry-band files used by the "
        "legacy custom plot, and publishes reusable SED artifacts."
    ),
    requires=[
        "numpy",
        "pandas",
        "panel",
        "holoviews",
    ],
    capabilities=[
        "panel",
        "service",
        "artifact_viewer",
        "datasets",
        "selection",
        "artifacts",
        "jobs",
    ],
    tags=[
        "astronomy",
        "sed",
        "photometry",
        "broadband",
        "spectral-energy-distribution",
    ],
)


def register(api) -> None:
    panel_mod = _impl("panel")
    service_mod = _impl("service")

    api.register_service(
        key="runtime",
        factory=service_mod.create_sed_runtime,
        lazy=True,
        replace=True,
        description=(
            "Runtime service for broadband SED photometry-band files, focused-row "
            "photometry extraction and SED table construction."
        ),
        optional_requires=["numpy", "pandas"],
    )

    api.register_panel(
        id="panel",
        title="Broadband SED",
        factory=panel_mod.create_broadband_sed_panel,
        description=(
            "Plot broadband photometry for the focused row using the legacy "
            "`data/sed_data/*.json` band-definition format. Requires mapped "
            "`record_id`; unresolved JSON column references can be mapped inside "
            "the panel."
        ),
        category="SED / Photometry",
        icon="chart-line",
        tags=[
            "astronomy",
            "sed",
            "photometry",
            "broadband",
            "selection",
        ],
        required_mappings=["record_id"],
        optional_mappings=["target_label", "redshift"],
        uses_services=[f"{PLUGIN_ID}.runtime"],
        produces=[
            "astro.sed.broadband",
            "astro.sed.updated",
            "astro.sed.running",
        ],
        default_layout={"x": 0, "y": 0, "w": 6, "h": 5},
        state_version=1,
        persist_layout=True,
        persist_state=True,
        restore_policy="best_effort",
        optional_requires=["holoviews", "panel", "numpy", "pandas"],
    )

    api.register_artifact_viewer(
        artifact_type="astro.sed.broadband",
        viewer_factory=panel_mod.create_broadband_sed_artifact_viewer,
        id="viewer",
        title="Broadband SED Viewer",
        description="Display broadband SED artifacts produced by the Broadband SED panel.",
        priority=50,
        default=True,
        optional_requires=["holoviews", "panel", "pandas"],
    )