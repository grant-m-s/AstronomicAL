from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from astronomicAL.platform.plugins import PluginManifest


manifest = PluginManifest(
    id="core.resources",
    name="Resource Monitor",
    version="0.1.0",
    description=(
        "Top-like CPU, memory, process, disk, network, and NVIDIA GPU monitor "
        "for local and server-side AstronomicAL sessions."
    ),
    requires=[],
    optional_requires=["psutil>=5.9"],
    capabilities=["panel", "monitoring", "diagnostics"],
    tags=[
        "core",
        "resources",
        "monitoring",
        "diagnostics",
        "gpu",
        "nvidia",
        "server",
        "top",
    ],
)


def register(api) -> None:
    api.register_service(
        key="sampler",
        factory=create_resource_sampler,
        lazy=True,
        replace=True,
        description=(
            "Resource sampler for CPU, memory, disk, network, process, and NVIDIA GPU metrics."
        ),
        optional_requires=["psutil>=5.9"],
    )

    api.register_panel(
        id="monitor",
        title="Resource Monitor",
        factory=create_resource_monitor_panel,
        description=(
            "Live server/resource monitor with CPU, RAM, disk, network, process, "
            "and NVIDIA GPU usage."
        ),
        category="System",
        icon="monitoring",
        tags=["resources", "system", "gpu", "nvidia", "top", "diagnostics"],
        uses_services=["core.resources.sampler"],
        produces=["resources.snapshot"],
        default_layout={"x": 0, "y": 0, "w": 7, "h": 7},
        persist_layout=True,
        persist_state=True,
    )


def create_resource_sampler():
    from . import panel as panel_module

    return panel_module.ResourceSampler()


def create_resource_monitor_panel(context, **kwargs):
    from . import panel as panel_module

    sampler = context.services.get("core.resources.sampler")

    controller = panel_module.ResourceMonitorPanel(
        context=context,
        sampler=sampler,
        restore_state=kwargs.get("restore_state"),
    )

    return controller.panel(), controller