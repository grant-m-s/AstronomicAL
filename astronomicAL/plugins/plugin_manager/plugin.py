from __future__ import annotations

from astronomicAL.platform.plugins import PluginManifest

from .panel import PluginManagerPanel


manifest = PluginManifest(
    id="core.plugin_manager",
    name="Plugin Manager",
    version="1.0.0",
    description=(
        "Manage which AstronomicAL plugins are available, understand the features "
        "they add, and review plugin or discovery problems."
    ),
    capabilities=["panel", "diagnostics", "plugins", "management"],
    tags=["core", "plugins", "management"],
)


def register(api) -> None:
    api.register_panel(
        id="panel",
        title="Plugin Manager",
        factory=create_plugin_manager_panel,
        description="Manage plugin availability and review plugin health.",
        category="Platform",
        icon="plug",
        tags=["plugins", "management"],
        default_layout={"x": 0, "y": 0, "w": 6, "h": 8},
    )


def create_plugin_manager_panel(context, **_kwargs):
    controller = PluginManagerPanel(context=context)
    return controller.view, controller
