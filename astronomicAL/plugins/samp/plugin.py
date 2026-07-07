from __future__ import annotations

from importlib import import_module

from astronomicAL.platform.plugins import PluginManifest

PLUGIN_ID = "astro.samp"
PLUGIN_PACKAGE = "astronomicAL.plugins.samp"


def _impl(module: str):
    return import_module(f"{PLUGIN_PACKAGE}.{module}")


manifest = PluginManifest(
    id=PLUGIN_ID,
    name="SAMP",
    version="0.2.5",
    description=(
        "Send and receive table datasets through SAMP/TOPCAT using the platform "
        "DatasetSource, jobs, artifacts, and Parquet-backed dataset registration."
    ),
    requires=[
        "panel",
        "pandas",
        "numpy",
        "astropy",
        "pyarrow",
        "duckdb",
    ],
    capabilities=[
        "panel",
        "service",
        "datasets",
        "jobs",
        "artifacts",
        "events",
    ],
    tags=[
        "astronomy",
        "interop",
        "tables",
        "topcat",
        "samp",
        "large-data",
    ],
)


def register(api) -> None:
    panel_mod = _impl("panel")
    service_mod = _impl("service")

    api.register_service(
        key="bridge",
        factory=service_mod.create_samp_bridge,
        lazy=True,
        replace=True,
        description=(
            "Long-lived SAMP bridge. It keeps AstronomicAL visible on the SAMP hub, "
            "sends FITS/VOTable tables, mirrors incoming table URLs locally, and notifies "
            "panel listeners without parsing large tables in the SAMP callback."
        ),
    )

    api.register_panel(
        id="receive",
        title="SAMP Receive",
        factory=panel_mod.create_samp_receive_panel,
        description=(
            "Receive VOTable messages from TOPCAT or other SAMP clients and register "
            "them as lazy Parquet-backed AstronomicAL datasets."
        ),
        category="Interop",
        tags=["astronomy", "interop", "tables", "topcat", "samp", "large-data"],
        required_mappings=[],
        optional_mappings=[],
        uses_services=[f"{PLUGIN_ID}.bridge"],
        produces=[
            "artifact.created",
            "interop.samp.table.received",
            "interop.samp.table.previewed",
            "dataset.loaded",
            "dataset.mapping.updated",
            "dataset.active.changed",
            "dataset.updated",
            "interop.samp.table.imported",
        ],
        default_layout={"x": 0, "y": 0, "w": 6, "h": 8},
        state_version=2,
    )

    api.register_panel(
        id="send",
        title="SAMP Send",
        factory=panel_mod.create_samp_send_panel,
        description=(
            "Send active or selected AstronomicAL table data to TOPCAT or another SAMP client. "
            "Large-table exports use source-side column projection, row limits, and FITS-by-default transfer before materialising pandas."
        ),
        category="Interop",
        tags=["astronomy", "interop", "tables", "topcat", "samp", "large-data"],
        required_mappings=[],
        optional_mappings=[],
        uses_services=[f"{PLUGIN_ID}.bridge"],
        produces=["artifact.created", "interop.samp.table.sent"],
        default_layout={"x": 0, "y": 0, "w": 6, "h": 8},
        state_version=2,
    )
