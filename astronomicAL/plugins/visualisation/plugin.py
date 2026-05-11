from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path
from typing import Any, Dict, Optional

from astronomicAL.platform.plugins import PluginManifest

PLUGIN_ID = "core.visualisation"
STATE_SERVICE_KEY = f"{PLUGIN_ID}.state"

_SPLIT_PACKAGE = "astronomicAL.plugins.visualisation"


def _ensure_split_package() -> None:
    """Ensure split visualisation modules can be imported as a package.

    Some plugin-manager paths import this file as a synthetic local module, e.g.

        astronomical_local_plugin_visualisation_<hash>

    In that situation, relative imports such as ``from .state import ...`` fail
    because this module has no package context.

    The split implementation files are still located next to this file, so we
    create/repair the real package entry in ``sys.modules`` and import the
    implementation modules through ``astronomicAL.plugins.visualisation``.
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
    name="Visualisation",
    version="0.3.2",
    description=(
        "Generic HoloViews visualisation panels for the active dataset. "
        "Panels support large datasets through rasterized Datashader rendering, "
        "sampled interactive rendering, platform focus/selection publication, "
        "and semantic mappings."
    ),
    capabilities=["panel", "datasets", "selection", "visualisation"],
    tags=["core", "visualisation", "plots", "holoviews", "datashader"],
)


def register(api) -> None:
    api.register_service(
        key="state",
        factory=create_visualisation_state,
        lazy=True,
        replace=True,
        description=(
            "Shared visualisation state for plugins that explicitly want a "
            "shared X/Y/label/rendering state."
        ),
    )

    api.register_panel(
        id="scatter",
        title="Scatter Plot",
        factory=create_scatter_panel,
        description=(
            "Large-data XY scatter plot with sampled interactive mode, "
            "rasterized Datashader fallback, focus publication, and selection-set publication."
        ),
        category="Visualisation",
        icon="scatter_plot",
        tags=["core", "visualisation", "scatter", "selection", "plot"],
        optional_mappings=["record_id", "target_label"],
        produces=["selection.focus.changed", "selection.set.changed"],
        default_layout={"x": 0, "y": 0, "w": 3, "h": 4},
    )

    api.register_panel(
        id="histogram",
        title="Histogram Plot",
        factory=create_histogram_panel,
        description="Label-aware histogram using the selected X variable and current focus overlay.",
        category="Visualisation",
        icon="bar_chart",
        tags=["core", "visualisation", "histogram", "labels", "plot"],
        optional_mappings=["record_id", "target_label"],
        default_layout={"x": 0, "y": 0, "w": 3, "h": 4},
    )

    api.register_panel(
        id="density",
        title="Density Plot",
        factory=create_density_panel,
        description=(
            "Responsive 2D density plot using shared X/Y variables and "
            "rasterized Datashader rendering for large datasets."
        ),
        category="Visualisation",
        icon="grid_on",
        tags=["core", "visualisation", "density", "datashader", "rasterize", "plot"],
        optional_mappings=["record_id", "target_label"],
        default_layout={"x": 0, "y": 0, "w": 3, "h": 4},
    )

    api.register_panel(
        id="explorer",
        title="Linked Plot Explorer",
        factory=create_explorer_panel,
        description=(
            "Combined scatter, histogram, and density explorer with one shared "
            "settings header and linked visualisation state."
        ),
        category="Visualisation",
        icon="dashboard",
        tags=["core", "visualisation", "explorer", "linked", "dashboard", "plot"],
        optional_mappings=["record_id", "target_label"],
        produces=["selection.focus.changed", "selection.set.changed"],
        default_layout={"x": 0, "y": 0, "w": 4, "h": 5},
    )


def create_visualisation_state(context, **kwargs):
    state_mod = _impl("state")
    state = state_mod.VisualisationState(context=context)

    restore_state = kwargs.get("restore_state")
    if isinstance(restore_state, dict):
        state.restore_state(restore_state)

    return state


def _new_state(context, restore_state: Optional[Dict[str, Any]] = None):
    state_mod = _impl("state")
    state = state_mod.VisualisationState(context=context)

    if isinstance(restore_state, dict):
        state.restore_state(restore_state)

    return state


def _get_shared_state(context):
    """Return the plugin-scoped shared state service.

    Standalone panels currently use independent state instances for better
    workspace behaviour. This shared service remains available for workflows or
    external plugins that explicitly want shared visualisation state.
    """
    state_mod = _impl("state")

    services = getattr(context, "services", None)
    if services is None:
        return state_mod.VisualisationState(context=context)

    try:
        state = services.get(STATE_SERVICE_KEY)
    except Exception:
        state = None

    if state is None:
        state = state_mod.VisualisationState(context=context)
        try:
            services.set(STATE_SERVICE_KEY, state, replace=True, owner=PLUGIN_ID)
        except TypeError:
            services.set(STATE_SERVICE_KEY, state)
        except Exception:
            pass

    return state


def create_scatter_panel(context, **kwargs):
    scatter_mod = _impl("scatter")

    state = _new_state(context, kwargs.get("restore_state"))
    controller = scatter_mod.ScatterPanel(
        context=context,
        state=state,
        show_controls=True,
    )
    return controller.panel(), controller


def create_histogram_panel(context, **kwargs):
    histogram_mod = _impl("histogram")

    state = _new_state(context, kwargs.get("restore_state"))
    controller = histogram_mod.HistogramPanel(
        context=context,
        state=state,
        show_controls=True,
    )
    return controller.panel(), controller


def create_density_panel(context, **kwargs):
    density_mod = _impl("density")

    state = _new_state(context, kwargs.get("restore_state"))
    controller = density_mod.DensityPanel(
        context=context,
        state=state,
        show_controls=True,
    )
    return controller.panel(), controller


def create_explorer_panel(context, **kwargs):
    explorer_mod = _impl("explorer")

    state = _new_state(context, kwargs.get("restore_state"))
    controller = explorer_mod.LinkedExplorerPanel(
        context=context,
        state=state,
    )
    return controller.panel(), controller