# BUG: If lots of labels button ends up above bottom label item (phz_classification_phz) 

from __future__ import annotations

from astronomicAL.platform.plugins import PluginManifest

manifest = PluginManifest(
    id="core.record_browser",
    name="Record Browser",
    version="0.1.0",
    description=(
        "Generic active-dataset record browser. It recreates the default "
        "Exploration panel browsing UI without owning dataset loading or "
        "column-mapping setup."
    ),
    capabilities=["panel", "datasets", "selection", "labels"],
    tags=["core", "browser", "records", "datasets", "selection"],
)


def register(api) -> None:
    api.register_panel(
        id="panel",
        title="Record Browser",
        factory=create_record_browser_panel,
        description=(
            "Browse the active dataset, inspect one record at a time, and "
            "publish the focused record through the platform selection service."
        ),
        category="Core",
        icon="list",
        tags=["core", "dataset", "selection", "records"],
        required_mappings=["record_id"],
        produces=["selection.focus.changed", "labels.settings.updated"],
        default_layout={"x": 0, "y": 0, "w": 4, "h": 7},
    )


def create_record_browser_panel(context, **kwargs):
    from astronomicAL.plugins.record_browser.panel import RecordBrowserPanel

    controller = RecordBrowserPanel(context=context)
    return controller.panel(), controller