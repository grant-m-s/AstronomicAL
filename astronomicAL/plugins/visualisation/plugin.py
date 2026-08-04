from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path
from typing import Any, Dict, Optional

from astronomicAL.platform.plugins import PluginManifest

PLUGIN_ID = "core.visualisation"
STATE_SERVICE_KEY = f"{PLUGIN_ID}.state"
PREPARED_CACHE_SERVICE_KEY = f"{PLUGIN_ID}.prepared_cache"
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

    api.register_service(
        key="prepared_cache",
        factory=create_prepared_frame_cache,
        lazy=True,
        replace=True,
        description="Shared prepared-frame cache for visualisation panels.",
    )

    api.register_service(
        key="data_cache",
        factory=create_visualisation_data_cache,
        lazy=True,
        replace=True,
        description="Shared column, label, and row-id cache for visualisation panels.",
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
        required_mappings=["record_id"],
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
        required_mappings=["record_id"],
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
        required_mappings=["record_id"],
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
        required_mappings=["record_id"],
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

class PreparedFrameCache:
    def __init__(self, max_items: int = 4):
        self.max_items = int(max_items)
        self._cache = {}

    def get(self, key):
        return self._cache.get(key)

    def set(self, key, value):
        if key in self._cache:
            self._cache.pop(key, None)

        self._cache[key] = value

        while len(self._cache) > self.max_items:
            oldest_key = next(iter(self._cache))
            self._cache.pop(oldest_key, None)

    def clear(self):
        self._cache.clear()

    def invalidate_dataset(self, dataset_id):
        for key in list(self._cache):
            key_dataset_id = key[0] if isinstance(key, tuple) and key else None
            if key_dataset_id == dataset_id:
                self._cache.pop(key, None)


def create_prepared_frame_cache(context, **kwargs):
    return PreparedFrameCache(max_items=12)


class VisualisationRuntimeState:
    def __init__(self):
        self.prepared_frame_cache = {}
        self.max_prepared_frames = 4

    def get_prepared_frame(self, key):
        return self.prepared_frame_cache.get(key)

    def set_prepared_frame(self, key, value):
        if key in self.prepared_frame_cache:
            self.prepared_frame_cache.pop(key, None)

        self.prepared_frame_cache[key] = value

        while len(self.prepared_frame_cache) > self.max_prepared_frames:
            oldest_key = next(iter(self.prepared_frame_cache))
            self.prepared_frame_cache.pop(oldest_key, None)

    def clear_prepared_frame_cache(self, *, dataset_id=None):
        if dataset_id is None:
            self.prepared_frame_cache.clear()
            return

        for key in list(self.prepared_frame_cache):
            key_dataset_id = key[0] if isinstance(key, tuple) and key else None
            if key_dataset_id == dataset_id:
                self.prepared_frame_cache.pop(key, None)


class VisualisationDataCache:
    def __init__(
        self,
        max_columns: int = 32,
        max_labels: int = 8,
        max_row_id_arrays: int = 4,
    ):
        self.max_columns = int(max_columns)
        self.max_labels = int(max_labels)
        self.max_row_id_arrays = int(max_row_id_arrays)

        self.columns = {}
        self.labels = {}
        self.row_ids = {}

    def _evict_oldest(self, store, max_items: int):
        while len(store) > max_items:
            store.pop(next(iter(store)), None)

    def get_column(self, key):
        return self.columns.get(key)

    def set_column(self, key, value):
        if key in self.columns:
            self.columns.pop(key, None)

        self.columns[key] = value
        self._evict_oldest(self.columns, self.max_columns)

    def get_label(self, key):
        return self.labels.get(key)

    def set_label(self, key, value):
        if key in self.labels:
            self.labels.pop(key, None)

        self.labels[key] = value
        self._evict_oldest(self.labels, self.max_labels)

    def get_row_ids(self, key):
        return self.row_ids.get(key)

    def set_row_ids(self, key, value):
        if key in self.row_ids:
            self.row_ids.pop(key, None)

        self.row_ids[key] = value
        self._evict_oldest(self.row_ids, self.max_row_id_arrays)

    def clear_dataset(self, dataset_id):
        for store in (self.columns, self.labels, self.row_ids):
            for key in list(store):
                key_dataset_id = None

                if isinstance(key, tuple):
                    if len(key) >= 2:
                        # Current visualisation data-cache keys look like:
                        # ("column", dataset_id, fingerprint, column_name)
                        key_dataset_id = key[1]
                    elif len(key) == 1:
                        key_dataset_id = key[0]

                if key_dataset_id == dataset_id:
                    store.pop(key, None)

    def clear(self):
        self.columns.clear()
        self.labels.clear()
        self.row_ids.clear()

def create_visualisation_data_cache(context, **kwargs):
    return VisualisationDataCache()