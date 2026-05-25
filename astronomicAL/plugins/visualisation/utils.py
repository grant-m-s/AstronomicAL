# BUG: Prepare labels very slow (nearly a minute)
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import time

import numpy as np
import pandas as pd
from bokeh.palettes import Category10, Category20

from bokeh.models import (
    BoxSelectTool,
    BoxZoomTool,
    HoverTool,
    LassoSelectTool,
    PanTool,
    ResetTool,
    SaveTool,
    TapTool,
    WheelZoomTool,
)

from .constants import (
    INTERNAL_LABEL_COLOUR,
    INTERNAL_LABEL_DISPLAY,
    INTERNAL_LABEL_RAW,
    INTERNAL_ROW_ID,
    INTERNAL_X,
    INTERNAL_Y,
)


_HV_EXTENSION_LOADED = False
MAX_HOVER_ROWS = 4

SCATTER_RENDERER = "astronomical_visualisation_scatter_points"
HIST_RENDERER = "astronomical_visualisation_histogram"
DENSITY_RENDERER = "astronomical_visualisation_density"

HOVER_ROW_ID = "hover_record_id"
HOVER_LABEL = "hover_label"

HOVER_CSS = """
.bk-tooltip {
    max-height: 150px !important;
    overflow-y: auto !important;
    overflow-x: hidden !important;
}
.bk-tooltip > div {
    max-height: 150px !important;
    overflow-y: auto !important;
    overflow-x: hidden !important;
}
"""

VISIBLE_DENSITY_CMAP = [
    "#08306b",
    "#08519c",
    "#2171b5",
    "#4292c6",
    "#6baed6",
    "#9ecae1",
    "#c6dbef",
    "#fdd49e",
    "#fdbb84",
    "#fc8d59",
    "#e34a33",
    "#b30000",
]


def ensure_hv_extension() -> None:
    global _HV_EXTENSION_LOADED
    if _HV_EXTENSION_LOADED:
        return

    try:
        import holoviews as hv

        hv.extension("bokeh")
    except Exception:
        pass

    try:
        import panel as pn

        raw_css = list(getattr(pn.config, "raw_css", []) or [])
        if HOVER_CSS not in raw_css:
            try:
                pn.config.raw_css.append(HOVER_CSS)
            except Exception:
                pn.extension(raw_css=[HOVER_CSS])
    except Exception:
        pass

    _HV_EXTENSION_LOADED = True

def keep_pan_tool_active_hook(plot, element):
    try:
        from bokeh.models import PanTool, WheelZoomTool
    except Exception:
        return

    try:
        state = getattr(plot, "state", None)
        toolbar = getattr(state, "toolbar", None)

        if toolbar is None:
            return

        tools = list(getattr(state, "tools", []) or [])

        pan_tool = next((tool for tool in tools if isinstance(tool, PanTool)), None)
        wheel_tool = next((tool for tool in tools if isinstance(tool, WheelZoomTool)), None)

        if pan_tool is not None and getattr(toolbar, "active_drag", None) is None:
            toolbar.active_drag = pan_tool

        if wheel_tool is not None and getattr(toolbar, "active_scroll", None) is None:
            toolbar.active_scroll = wheel_tool
    except Exception:
        return

def force_wheel_zoom_hook(plot, element) -> None:
    """Make wheel zoom active and limit hover rows where supported."""
    try:
        for tool in plot.state.tools:
            if isinstance(tool, WheelZoomTool):
                plot.state.toolbar.active_scroll = tool
            if isinstance(tool, HoverTool):
                _apply_hover_limit(tool)
    except Exception:
        pass


def deduplicate_toolbar_tools_hook(plot, element) -> None:
    try:
        dedupe_classes = (
            PanTool,
            WheelZoomTool,
            BoxZoomTool,
            ResetTool,
            SaveTool,
            TapTool,
            BoxSelectTool,
            LassoSelectTool,
            HoverTool,
        )

        seen = set()
        kept = []

        for tool in list(plot.state.tools):
            key = type(tool)

            if isinstance(tool, dedupe_classes):
                if key in seen:
                    continue
                seen.add(key)

            kept.append(tool)

        plot.state.tools = kept

    except Exception:
        pass

    force_wheel_zoom_hook(plot, element)

def renderer_name_hook(name: str):
    """Name and target the main glyph renderer.

    Do not use HoverTool.names because older Bokeh versions do not support it.
    Bind hover tools to the main renderer through HoverTool.renderers instead.
    """

    def _hook(plot, element) -> None:
        renderer = None

        try:
            renderer = plot.handles.get("glyph_renderer")
            if renderer is not None:
                renderer.name = name
        except Exception:
            renderer = None

        try:
            for tool in plot.state.tools:
                if isinstance(tool, HoverTool):
                    _apply_hover_limit(tool)

                    if renderer is not None and hasattr(tool, "renderers"):
                        try:
                            tool.renderers = [renderer]
                        except Exception:
                            pass
        except Exception:
            pass

        force_wheel_zoom_hook(plot, element)

    return _hook


def _apply_hover_limit(tool: HoverTool) -> HoverTool:
    try:
        tool.limit = MAX_HOVER_ROWS
    except Exception:
        pass
    return tool


def _hover_tool(*, tooltips) -> HoverTool:
    kwargs = {
        "tooltips": tooltips,
        "mode": "mouse",
        "point_policy": "snap_to_data",
    }

    try:
        tool = HoverTool(limit=MAX_HOVER_ROWS, **kwargs)
    except Exception:
        tool = HoverTool(**kwargs)
        _apply_hover_limit(tool)

    return tool


def limited_point_hover_tool() -> HoverTool:
    return _hover_tool(
        tooltips=[
            ("x", "$x"),
            ("y", "$y"),
            ("id", f"@{HOVER_ROW_ID}"),
            ("label", f"@{HOVER_LABEL}"),
        ],
    )


def limited_histogram_hover_tool() -> HoverTool:
    return _hover_tool(
        tooltips=[
            ("x", "$x"),
            ("count", "$y"),
        ],
    )


@dataclass
class PreparedFrame:
    dataset_id: Optional[str]
    frame: pd.DataFrame
    x_name: Any
    y_name: Optional[Any]
    require_y: bool
    row_count_before_filter: int
    row_count_after_filter: int
    sampled_from: Optional[int] = None

    @property
    def empty(self) -> bool:
        return self.frame.empty

    @property
    def row_ids(self) -> np.ndarray:
        if INTERNAL_ROW_ID not in self.frame.columns:
            return np.asarray([], dtype=str)
        return self.frame[INTERNAL_ROW_ID].to_numpy(copy=False)

    @property
    def x_values(self) -> np.ndarray:
        return self.frame[INTERNAL_X].to_numpy(copy=False)

    @property
    def y_values(self) -> np.ndarray:
        if INTERNAL_Y not in self.frame.columns:
            return np.asarray([], dtype=float)
        return self.frame[INTERNAL_Y].to_numpy(copy=False)


def _active_dataset_id(context) -> Optional[str]:
    try:
        return context.datasets.active_id()
    except Exception:
        return None

def _active_df(context) -> Optional[pd.DataFrame]:

    try:
        datasets = getattr(context, "datasets", None)
        if datasets is None:
            return None

        dataset_id = _active_dataset_id(context)
        if dataset_id is None:
            return None

        if hasattr(datasets, "head"):
            return datasets.head(dataset_id, n=0)

        try:
            return datasets.get_df(dataset_id, limit=0)
        except TypeError:
            return datasets.get_df(dataset_id)

    except Exception:
        return None

def _dataset_columns(context, dataset_id: Optional[str]) -> List[str]:
    datasets = getattr(context, "datasets", None)

    if datasets is None or dataset_id is None:
        return []

    if hasattr(datasets, "list_columns"):
        try:
            return [str(col) for col in datasets.list_columns(dataset_id)]
        except Exception:
            pass

    # Legacy fallback. This may materialise, so only use if needed.
    df = _active_df(context)
    if df is None:
        return []

    return [str(col) for col in df.columns]


def _dataset_row_count(context, dataset_id: Optional[str]) -> int:
    datasets = getattr(context, "datasets", None)

    if datasets is None or dataset_id is None:
        return 0

    if hasattr(datasets, "row_count"):
        try:
            count = datasets.row_count(dataset_id)
            return int(count or 0)
        except Exception:
            pass

    # Legacy fallback.
    try:
        df = datasets.get_df(dataset_id)
        return int(len(df))
    except Exception:
        return 0


def _dataset_fingerprint(context, dataset_id: Optional[str]) -> Tuple[Any, ...]:
    """
    Lightweight cache identity for the active dataset.

    Do not use id(df), because that forces a materialised DataFrame and changes
    every time a Parquet-backed view is read.
    """
    datasets = getattr(context, "datasets", None)

    if datasets is None or dataset_id is None:
        return (dataset_id, None)

    try:
        meta = datasets.get_meta(dataset_id)
    except Exception:
        meta = {}

    backend = meta.get("backend")
    source_path = meta.get("source_path") or meta.get("cache_path") or meta.get("parquet_path")
    row_count = meta.get("row_count") or meta.get("rows") or _dataset_row_count(context, dataset_id)

    # Include a lightweight version-ish marker if present. Later you can update
    # this when derived columns, labels, or table mutations occur.
    version = (
        meta.get("version")
        or meta.get("updated_at")
        or meta.get("cache_mtime")
        or meta.get("dataset_version")
    )

    return (
        dataset_id,
        backend,
        str(source_path) if source_path is not None else None,
        int(row_count or 0),
        version,
    )


def _unique_existing_columns(
    columns: Sequence[Any],
    available_columns: Sequence[str],
) -> List[Any]:
    available = set(str(col) for col in available_columns)
    wanted: List[Any] = []
    seen = set()

    for col in columns:
        if col is None:
            continue

        if col in {"Use Index", "No Labels"}:
            continue

        col_str = str(col)
        if col_str not in available:
            continue

        if col_str in seen:
            continue

        wanted.append(col)
        seen.add(col_str)

    return wanted


def _plot_required_columns(context, dataset_id: Optional[str], state, *, require_y: bool) -> List[Any]:
    """
    Determine the minimum columns required to prepare a scatter/density/histogram
    frame.

    This is the key replacement for loading the entire dataset and then taking
    x/y/label/id columns from it.
    """
    available_columns = _dataset_columns(context, dataset_id)

    candidates: List[Any] = [
        state.x,
        state.y if require_y else None,
        getattr(state, "record_id_col", None),
        getattr(state, "label_col", None),
    ]

    # If future state objects add hover columns, this will include them without
    # breaking older state objects.
    hover_cols = getattr(state, "hover_cols", None) or getattr(state, "hover_columns", None) or []
    candidates.extend(list(hover_cols))

    return _unique_existing_columns(candidates, available_columns)

def _get_dataset_view_for_columns(
    context,
    dataset_id: Optional[str],
    columns: Sequence[Any],
    *,
    limit: Optional[int] = None,
) -> pd.DataFrame:

    datasets = getattr(context, "datasets", None)

    if datasets is None or dataset_id is None:
        return pd.DataFrame(columns=[str(col) for col in columns if col is not None])

    wanted: list[Any] = []
    seen: set[str] = set()

    for col in columns:
        if col is None:
            continue

        col_str = str(col)

        if col_str in {"Use Index", "No Labels"}:
            continue

        if col_str in seen:
            continue

        wanted.append(col)
        seen.add(col_str)

    if not wanted:
        return pd.DataFrame()

    get_df = getattr(datasets, "get_df", None)

    if callable(get_df):
        try:
            return get_df(dataset_id, columns=wanted, limit=limit)
        except TypeError:
            # Compatibility with the old DatasetManager.get_df(dataset_id)
            # signature. This may materialise the full dataset, but only on
            # old managers that do not support column-limited access.
            try:
                df = get_df(dataset_id)
                existing = [col for col in wanted if col in df.columns]
                return df.loc[:, existing].copy()
            except Exception:
                pass
        except Exception:
            pass

    try:
        source = datasets.get_source(dataset_id)
        return source.to_pandas(columns=wanted, limit=limit)
    except Exception:
        return pd.DataFrame(columns=wanted)

def _mapped_column(
    context,
    dataset_id: Optional[str],
    semantic: str,
    *,
    allow_index: bool,
) -> Optional[Any]:
    columns = _dataset_columns(context, dataset_id)
    column_set = set(columns)

    datasets = getattr(context, "datasets", None)

    if datasets is not None and dataset_id is not None:
        try:
            value = datasets.get_mapping(dataset_id, semantic)

            if value == "Use Index" and allow_index:
                return value

            if value in column_set:
                return value
        except Exception:
            pass

    cfg = getattr(context, "config", None)
    settings = getattr(cfg, "settings", {}) if cfg is not None else {}

    if semantic == "record_id":
        value = settings.get("id_col")

        if value == "Use Index" and allow_index:
            return value

        if value in column_set:
            return value

    if semantic == "target_label":
        value = settings.get("label_col")

        if value in column_set:
            return value

    return None



def _default_colour_map(raw_labels: Iterable[Any]) -> Dict[Any, str]:
    labels = list(raw_labels)
    if not labels:
        return {}

    palette = Category10[10] if len(labels) <= 10 else Category20[20]
    return {
        label: palette[index % len(palette)]
        for index, label in enumerate(labels)
    }


def _safe_series(df: pd.DataFrame, column: Any) -> pd.Series:
    values = df[column]
    if isinstance(values, pd.DataFrame):
        return values.iloc[:, 0]
    return values

def _numeric_array_from_array(values: Any) -> np.ndarray:
    """Return a numeric NumPy array suitable for plotting.

    This is the array-based equivalent of _numeric_array(df, column).
    """
    series = pd.Series(values)

    if pd.api.types.is_bool_dtype(series.dtype):
        return series.to_numpy(dtype=np.float32, copy=False, na_value=np.nan)

    if pd.api.types.is_numeric_dtype(series.dtype):
        try:
            arr = series.to_numpy(copy=False)
        except Exception:
            arr = series.to_numpy(dtype=np.float64, copy=False, na_value=np.nan)
    else:
        arr = pd.to_numeric(series, errors="coerce").to_numpy(copy=False)

    arr = np.asarray(arr)

    if arr.dtype == np.float32:
        return arr

    if not np.issubdtype(arr.dtype, np.number):
        values64 = arr.astype(np.float64, copy=False)
    else:
        values64 = arr.astype(np.float64, copy=False)

    finite = np.isfinite(values64)

    if not finite.any():
        return values64

    max_abs = np.nanmax(np.abs(values64[finite]))

    if max_abs <= 1.0e20:
        return values64.astype(np.float32, copy=False)

    return values64

def _numeric_array(df: pd.DataFrame, column: Any) -> np.ndarray:
    """
    Return a numeric NumPy array suitable for plotting.

    Uses float32 only when safe. Very large scientific values, such as
    luminosities, must remain float64 or they overflow to inf and disappear
    during np.isfinite masking.
    """
    series = df[column]

    if pd.api.types.is_bool_dtype(series.dtype):
        return series.to_numpy(dtype=np.float32, copy=False, na_value=np.nan)

    if pd.api.types.is_numeric_dtype(series.dtype):
        try:
            values = series.to_numpy(copy=False)
        except Exception:
            values = series.to_numpy(dtype=np.float64, copy=False, na_value=np.nan)
    else:
        values = pd.to_numeric(series, errors="coerce").to_numpy(copy=False)

    values = np.asarray(values)

    if values.dtype == np.float32:
        return values

    # Convert nullable/object/numeric arrays to float64 first. This avoids the
    # overflow warning that happens when pandas casts huge values directly to
    # float32.
    if not np.issubdtype(values.dtype, np.number):
        values64 = values.astype(np.float64, copy=False)
    else:
        values64 = values.astype(np.float64, copy=False)

    finite = np.isfinite(values64)

    if not finite.any():
        return values64

    max_abs = np.nanmax(np.abs(values64[finite]))

    # Only downcast when it cannot overflow.
    if max_abs <= 1.0e20:
        return values64.astype(np.float32, copy=False)

    return values64

def _row_id_array(
    df: pd.DataFrame,
    record_id_col: Optional[Any],
    mask: np.ndarray,
) -> np.ndarray:
    """
    Return row IDs for the prepared frame.

    Do not stringify every ID here. Keep the original dtype and only stringify
    small selected/forced ID sets later when needed.
    """
    if record_id_col == "Use Index" or record_id_col is None:
        values = df.index.to_numpy(copy=False)
    elif record_id_col in df.columns:
        values = df[record_id_col].to_numpy(copy=False)
    else:
        values = df.index.to_numpy(copy=False)

    return values[mask]


def _dataset_dtypes(context, dataset_id: Optional[str]) -> Dict[str, str]:
    datasets = getattr(context, "datasets", None)

    if datasets is None or dataset_id is None:
        return {}

    if hasattr(datasets, "dtypes"):
        try:
            return {
                str(col): str(dtype)
                for col, dtype in datasets.dtypes(dataset_id).items()
            }
        except Exception:
            pass

    df = _active_df(context)
    if df is None:
        return {}

    return {str(col): str(dtype) for col, dtype in df.dtypes.items()}


def _is_numeric_dtype_name(dtype_name: str) -> bool:
    dtype_name = str(dtype_name).lower()

    numeric_markers = [
        "int",
        "integer",
        "bigint",
        "smallint",
        "tinyint",
        "hugeint",
        "uint",
        "float",
        "double",
        "real",
        "decimal",
        "numeric",
        "bool",
        "boolean",
    ]

    return any(marker in dtype_name for marker in numeric_markers)


def _numeric_columns_from_dataset(context, dataset_id: Optional[str]) -> List[str]:
    columns = _dataset_columns(context, dataset_id)
    dtypes = _dataset_dtypes(context, dataset_id)

    if not dtypes:
        return columns

    numeric_columns = [
        column for column in columns
        if _is_numeric_dtype_name(dtypes.get(column, ""))
    ]

    return numeric_columns

def _label_arrays(
    df: pd.DataFrame,
    label_col: Optional[Any],
    mask: np.ndarray,
    state,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    if not label_col or label_col not in df.columns:
        return None, None, None

    label_series = _safe_series(df, label_col)
    raw = label_series.to_numpy(copy=False)

    if state.label_filter and "All" not in state.label_filter:
        selected = state.selected_raw_labels()
        label_mask = pd.Series(raw).isin(selected).to_numpy()
        mask &= label_mask

    raw_filtered = raw[mask]

    display = np.asarray([state.label_display(value) for value in raw_filtered], dtype=object)
    colours = np.asarray([state.label_colour(value) for value in raw_filtered], dtype=object)

    return raw_filtered, display, colours

def _visualisation_data_cache(context):
    services = getattr(context, "services", None)
    if services is None:
        return None

    for key in (
        "core.visualisation.data_cache",
        "visualisation.data_cache",
        "data_cache",
    ):
        try:
            cache = services.get(key)
            if cache is not None:
                return cache
        except Exception:
            pass

    return None

def _dataset_cache_prefix(context, state):
    datasets = getattr(context, "datasets", None)

    dataset_id = getattr(state, "dataset_id", None)
    if dataset_id is None and datasets is not None:
        try:
            dataset_id = datasets.active_id()
        except Exception:
            dataset_id = None

    fingerprint = None
    if datasets is not None and dataset_id is not None:
        for method_name in ("fingerprint", "dataset_fingerprint", "source_fingerprint"):
            method = getattr(datasets, method_name, None)
            if callable(method):
                try:
                    fingerprint = method(dataset_id)
                    break
                except Exception:
                    pass

    return dataset_id, fingerprint
def _load_column_arrays(context, state, columns: Sequence[Any]) -> Dict[Any, np.ndarray]:
    """
    Load multiple source columns in one backend call, while still using the
    visualisation data cache per column.

    This avoids repeated Parquet/DuckDB scans during first render.
    """
    cache = _visualisation_data_cache(context)
    dataset_id, fingerprint = _dataset_cache_prefix(context, state)

    wanted: List[Any] = []
    seen: set[str] = set()

    for column in columns:
        if column is None:
            continue
        if column in {"Use Index", "No Labels"}:
            continue

        column_key = str(column)
        if column_key in seen:
            continue

        wanted.append(column)
        seen.add(column_key)

    if not wanted:
        return {}

    result: Dict[Any, np.ndarray] = {}
    missing: List[Any] = []

    for column in wanted:
        key = ("column", dataset_id, fingerprint, str(column))
        cached = cache.get_column(key) if cache is not None else None

        if cached is not None:
            result[column] = cached
        else:
            missing.append(column)

    if missing:
        source = context.datasets.get_source(dataset_id)
        df = source.to_pandas(columns=missing)

        for column in missing:
            if column not in df.columns:
                continue

            arr = df[column].to_numpy(copy=False)
            key = ("column", dataset_id, fingerprint, str(column))

            if cache is not None:
                cache.set_column(key, arr)

            result[column] = arr

    return result


def _load_column_array(context, state, column):
    """
    Compatibility wrapper for callers that still ask for a single column.
    Prefer _load_column_arrays(...) inside new visualisation code.
    """
    arrays = _load_column_arrays(context, state, [column])
    return arrays[column]

def _load_row_ids_array(context, state):
    cache = _visualisation_data_cache(context)
    dataset_id, fingerprint = _dataset_cache_prefix(context, state)
    record_id_col = getattr(state, "record_id_col", None)

    key = ("row_ids", dataset_id, fingerprint, str(record_id_col))

    if cache is not None:
        cached = cache.get_row_ids(key)
        if cached is not None:
            print("[AstronomicAL visualisation row-id-cache] HIT", flush=True)
            return cached

    print("[AstronomicAL visualisation row-id-cache] MISS", flush=True)

    if not record_id_col or record_id_col == "Use Index":
        row_count = context.datasets.row_count(dataset_id)
        # Keep native integer IDs. Do not astype(str) for millions of rows.
        row_ids = np.arange(row_count)
    else:
        source = context.datasets.get_source(dataset_id)
        df = source.to_pandas(columns=[record_id_col])
        # Keep native dtype. String conversion should happen only for tiny
        # selected/forced ID sets.
        row_ids = df[record_id_col].to_numpy(copy=False)

    if cache is not None:
        cache.set_row_ids(key, row_ids)

    return row_ids

def _load_label_raw_array(context, state):
    """
    Load only the raw label column.

    Display labels and colours should be derived only after filtering, and only
    when the current render path actually needs them.
    """
    label_col = getattr(state, "label_col", None)

    if not label_col:
        return None

    cache = _visualisation_data_cache(context)
    dataset_id, fingerprint = _dataset_cache_prefix(context, state)

    key = (
        "label_raw",
        dataset_id,
        fingerprint,
        str(label_col),
    )

    if cache is not None:
        cached = cache.get_label(key)
        if cached is not None:
            return cached

    raw = _load_column_array(context, state, label_col)

    if cache is not None:
        cache.set_label(key, raw)

    return raw


def _label_display_colour_arrays(values: np.ndarray, state):
    """
    Convert already-filtered label values to display strings and colours.

    This intentionally runs after masks/filters so we do not map millions of
    labels when only a subset is needed.
    """
    if values is None:
        return None, None

    values = np.asarray(values)

    unique_values = pd.unique(pd.Series(values))
    display_map = {value: state.label_display(value) for value in unique_values}
    colour_map = {value: state.label_colour(value) for value in unique_values}

    display = pd.Series(values).map(display_map).to_numpy(dtype=object)
    colours = pd.Series(values).map(colour_map).to_numpy(dtype=object)

    return display, colours


def _labels_needed_for_frame(state) -> bool:
    """
    Decide whether prepare_plot_frame needs the label column at all.

    Loading raw labels is needed for label filtering and label colouring.
    Display/colour arrays are only needed when rendering by labels or when a
    label-filtered frame carries labels forward.
    """
    label_col = getattr(state, "label_col", None)

    if not label_col or label_col == "No Labels":
        return False

    label_filter = getattr(state, "label_filter", None) or []
    if label_filter and "All" not in label_filter:
        return True

    color_by = str(getattr(state, "color_by", "") or "").strip().lower()
    if color_by == "labels":
        return True

    return False


def _load_label_arrays(context, state):
    """
    Backward-compatible wrapper.

    Existing callers expecting (raw, display, colours) still work, but new
    prepare_plot_frame code should prefer _load_label_raw_array and only call
    _label_display_colour_arrays after filtering.
    """
    raw = _load_label_raw_array(context, state)

    if raw is None:
        return None, None, None

    display, colours = _label_display_colour_arrays(raw, state)
    return raw, display, colours


def prepare_plot_frame(context, state, *, require_y: bool) -> PreparedFrame:
    t_total = time.perf_counter()

    dataset_id = _active_dataset_id(context)
    total_rows = _dataset_row_count(context, dataset_id)

    def _empty_frame() -> PreparedFrame:
        return PreparedFrame(
            dataset_id=dataset_id,
            frame=pd.DataFrame(),
            x_name=state.x,
            y_name=state.y if require_y else None,
            require_y=require_y,
            row_count_before_filter=total_rows,
            row_count_after_filter=0,
        )

    if dataset_id is None or not state.x:
        return _empty_frame()

    available_columns = _dataset_columns(context, dataset_id)
    available_set = set(available_columns)

    if state.x not in available_set:
        return _empty_frame()

    if require_y and (not state.y or state.y not in available_set):
        return _empty_frame()

    # ------------------------------------------------------------------
    # 1. Load raw columns in one backend call where possible.
    # ------------------------------------------------------------------
    t_load = time.perf_counter()

    columns_to_load: List[Any] = [state.x]

    if require_y:
        columns_to_load.append(state.y)

    labels_needed = (
        _labels_needed_for_frame(state)
        and getattr(state, "label_col", None) in available_set
    )

    if labels_needed:
        columns_to_load.append(state.label_col)

    try:
        arrays = _load_column_arrays(context, state, columns_to_load)
        x_raw = arrays[state.x]
        y_raw = arrays[state.y] if require_y else None
        label_raw_all = arrays.get(state.label_col) if labels_needed else None
    except Exception as exc:
        print(
            "[AstronomicAL visualisation] failed to load required plot columns "
            f"{columns_to_load!r}: {type(exc).__name__}: {exc}",
            flush=True,
        )
        return _empty_frame()

    load_seconds = time.perf_counter() - t_load

    n_rows = len(x_raw)
    if total_rows <= 0:
        total_rows = n_rows

    if require_y and y_raw is not None and len(y_raw) != n_rows:
        print(
            "[AstronomicAL visualisation] x/y length mismatch "
            f"x={n_rows:,} y={len(y_raw):,}",
            flush=True,
        )
        return _empty_frame()

    if label_raw_all is not None and len(label_raw_all) != n_rows:
        print(
            "[AstronomicAL visualisation] label length mismatch "
            f"labels={len(label_raw_all):,} rows={n_rows:,}; ignoring labels",
            flush=True,
        )
        label_raw_all = None

    if n_rows == 0:
        return _empty_frame()

    # ------------------------------------------------------------------
    # 2. Numeric arrays + finite/log mask.
    # ------------------------------------------------------------------
    t_arrays = time.perf_counter()

    x = _numeric_array_from_array(x_raw)
    mask = np.isfinite(x)

    y = None
    if require_y:
        y = _numeric_array_from_array(y_raw)
        mask &= np.isfinite(y)

    if state.log_x:
        mask &= x > 0

    if require_y and state.log_y and y is not None:
        mask &= y > 0

    arrays_seconds = time.perf_counter() - t_arrays

    # ------------------------------------------------------------------
    # 3. Labels: use raw labels only if filtering/colouring requires them.
    # ------------------------------------------------------------------
    t_labels = time.perf_counter()

    label_raw = None
    label_display = None
    label_colours = None

    if label_raw_all is not None:
        if state.label_filter and "All" not in state.label_filter:
            selected = state.selected_raw_labels()
            label_mask = pd.Series(label_raw_all).isin(selected).to_numpy()
            mask &= label_mask

        label_raw = label_raw_all[mask]

        color_by = str(getattr(state, "color_by", "") or "").strip().lower()
        should_build_label_display = (
            color_by == "labels"
            or bool(state.label_filter and "All" not in state.label_filter)
        )

        if should_build_label_display:
            label_display, label_colours = _label_display_colour_arrays(
                label_raw,
                state,
            )

    labels_seconds = time.perf_counter() - t_labels

    # ------------------------------------------------------------------
    # 4. Row IDs.
    # ------------------------------------------------------------------
    t_row_ids = time.perf_counter()

    try:
        row_ids_all = _load_row_ids_array(context, state)
    except Exception as exc:
        print(
            "[AstronomicAL visualisation] failed to load row ids: "
            f"{type(exc).__name__}: {exc}",
            flush=True,
        )
        row_ids_all = np.arange(len(mask))

    if len(row_ids_all) != len(mask):
        print(
            "[AstronomicAL visualisation] row-id length mismatch "
            f"row_ids={len(row_ids_all):,} rows={len(mask):,}; "
            "falling back to positional row ids",
            flush=True,
        )
        row_ids_all = np.arange(len(mask))

    row_ids = row_ids_all[mask]

    row_ids_seconds = time.perf_counter() - t_row_ids

    # ------------------------------------------------------------------
    # 5. Build compact frame.
    # ------------------------------------------------------------------
    t_data = time.perf_counter()

    data: Dict[str, Any] = {
        INTERNAL_X: x[mask],
        INTERNAL_ROW_ID: row_ids,
    }

    if require_y and y is not None:
        data[INTERNAL_Y] = y[mask]

    if label_raw is not None:
        data[INTERNAL_LABEL_RAW] = label_raw

        if label_display is not None:
            data[INTERNAL_LABEL_DISPLAY] = label_display
        else:
            data[INTERNAL_LABEL_DISPLAY] = label_raw.astype(object, copy=False)

        if label_colours is not None:
            data[INTERNAL_LABEL_COLOUR] = label_colours

    data_seconds = time.perf_counter() - t_data

    t_frame = time.perf_counter()
    frame = pd.DataFrame(data, copy=False)
    frame_seconds = time.perf_counter() - t_frame

    total_seconds = time.perf_counter() - t_total

    print(
        "[AstronomicAL visualisation] prepare_plot_frame total "
        f"{total_seconds:.3f}s "
        f"(load={load_seconds:.3f}s, "
        f"arrays={arrays_seconds:.3f}s, "
        f"labels={labels_seconds:.3f}s, "
        f"row_ids={row_ids_seconds:.3f}s, "
        f"data={data_seconds:.3f}s, "
        f"frame={frame_seconds:.3f}s, "
        f"rows={len(frame):,})",
        flush=True,
    )

    return PreparedFrame(
        dataset_id=dataset_id,
        frame=frame,
        x_name=state.x,
        y_name=state.y if require_y else None,
        require_y=require_y,
        row_count_before_filter=total_rows,
        row_count_after_filter=len(frame),
    )



def prepared_cache_key(context, state, *, require_y: bool) -> Tuple[Any, ...]:
    dataset_id = _active_dataset_id(context)
    fingerprint = _dataset_fingerprint(context, dataset_id)

    return (
        dataset_id,  # keep top-level for reliable invalidation
        fingerprint,
        state.x,
        state.y if require_y else None,
        require_y,
        tuple(state.label_filter or []),
        state.color_by,
        state.log_x,
        state.log_y if require_y else None,
        state.record_id_col,
        state.label_col,
    )

def _stable_unique_int_positions(values: Any) -> np.ndarray:
    """
    Deduplicate integer row positions while preserving input order.

    Do not use np.unique(...) here. np.unique sorts, and sorting by original
    row position can create visually sharp artificial cutoffs when the source
    table order is correlated with a plotted quantity.
    """

    arr = np.asarray(values, dtype=np.int64).ravel()

    if arr.size == 0:
        return arr

    seen: set[int] = set()
    out: list[int] = []

    for value in arr:
        item = int(value)
        if item in seen:
            continue
        seen.add(item)
        out.append(item)

    return np.asarray(out, dtype=np.int64)

def _append_unique_positions(
    *,
    selected: list[int],
    selected_set: set[int],
    candidates: Any,
    max_new: int,
) -> int:
    """
    Append at most max_new unique positions, preserving candidate order.

    Returns the number of positions appended.
    """

    if max_new <= 0:
        return 0

    added = 0

    for value in np.asarray(candidates, dtype=np.int64).ravel():
        item = int(value)

        if item in selected_set:
            continue

        selected.append(item)
        selected_set.add(item)
        added += 1

        if added >= max_new:
            break

    return added

def sample_prepared_frame(
    data: PreparedFrame,
    limit: int,
    *,
    seed: int = 0,
    force_row_ids: Optional[Sequence[Any]] = None,
) -> PreparedFrame:
    frame = data.frame
    n_rows = len(frame)

    if limit <= 0 or n_rows <= limit:
        return data

    limit = int(limit)
    force_row_ids = list(force_row_ids or [])

    # ------------------------------------------------------------------
    # Fast deterministic display sample.
    # ------------------------------------------------------------------
    sample_count = min(limit, n_rows)

    indices = np.linspace(
        0,
        n_rows - 1,
        num=sample_count,
        dtype=np.int64,
    )

    # ------------------------------------------------------------------
    # Add forced IDs without stringifying/scanning everything unless needed.
    # ------------------------------------------------------------------
    if force_row_ids and INTERNAL_ROW_ID in frame.columns:
        id_series = frame[INTERNAL_ROW_ID]
        forced_indices: list[int] = []

        # Try direct dtype-compatible matching first. This is much faster than
        # converting the full ID column to strings.
        for forced_id in force_row_ids:
            found = None

            try:
                matches = np.flatnonzero(id_series.to_numpy(copy=False) == forced_id)
                if len(matches):
                    found = int(matches[0])
            except Exception:
                found = None

            # If direct matching failed, try numeric coercion for numeric IDs.
            if found is None:
                try:
                    numeric_id = pd.to_numeric(forced_id)
                    matches = np.flatnonzero(id_series.to_numpy(copy=False) == numeric_id)
                    if len(matches):
                        found = int(matches[0])
                except Exception:
                    found = None

            # Last-resort fallback: only now do string comparison.
            # This is expensive, but should only happen for genuinely mismatched
            # ID types.
            if found is None:
                try:
                    values_as_str = id_series.astype(str).to_numpy(copy=False)
                    matches = np.flatnonzero(values_as_str == str(forced_id))
                    if len(matches):
                        found = int(matches[0])
                except Exception:
                    found = None

            if found is not None:
                forced_indices.append(found)

        if forced_indices:
            indices = np.concatenate(
                [
                    indices,
                    np.asarray(forced_indices, dtype=np.int64),
                ]
            )

    indices = np.unique(indices)
    indices.sort()

    # If forced rows pushed us above the limit, keep all forced rows and trim
    # regular sample rows from the end. Usually forced_indices is tiny, so this
    # rarely matters.
    if len(indices) > limit and force_row_ids:
        indices = indices[:limit]

    sampled = frame.take(indices).copy()

    return PreparedFrame(
        dataset_id=data.dataset_id,
        frame=sampled,
        x_name=data.x_name,
        y_name=data.y_name,
        require_y=data.require_y,
        row_count_before_filter=data.row_count_before_filter,
        row_count_after_filter=len(sampled),
        sampled_from=n_rows,
    )


def coverage_sample_prepared_frame(
    data: PreparedFrame,
    limit: int,
    *,
    seed: int = 0,
    force_row_ids: Optional[Sequence[Any]] = None,
    x_bins: int = 96,
    y_bins: int = 96,
    coverage_fraction: float = 0.65,
) -> PreparedFrame:
    """
    Spatially coverage-aware display sample.

    This is designed for the interactive reduced-point scatter view. It tries
    to make the visible sample more faithful in plot space without sending the
    full dataset to Bokeh.

    Important implementation detail:
    never use np.unique(...), sort(), then [:limit] for final capping. That
    creates row-order bias and can appear as clean artificial cutoffs when
    catalogue order is correlated with x/y.
    """

    frame = data.frame
    n_rows = len(frame)

    if limit <= 0 or n_rows <= limit:
        return data

    if INTERNAL_X not in frame.columns or INTERNAL_Y not in frame.columns:
        return sample_prepared_frame(
            data,
            limit,
            seed=seed,
            force_row_ids=force_row_ids,
        )

    limit = int(limit)
    force_row_ids = list(force_row_ids or [])

    try:
        x = frame[INTERNAL_X].to_numpy(copy=False)
        y = frame[INTERNAL_Y].to_numpy(copy=False)
        finite = np.isfinite(x) & np.isfinite(y)
    except Exception:
        return sample_prepared_frame(
            data,
            limit,
            seed=seed,
            force_row_ids=force_row_ids,
        )

    valid_positions = np.flatnonzero(finite)

    if len(valid_positions) <= limit:
        sampled = frame.take(valid_positions).copy()
        return PreparedFrame(
            dataset_id=data.dataset_id,
            frame=sampled,
            x_name=data.x_name,
            y_name=data.y_name,
            require_y=data.require_y,
            row_count_before_filter=data.row_count_before_filter,
            row_count_after_filter=len(sampled),
            sampled_from=n_rows,
        )

    try:
        xv = x[valid_positions]
        yv = y[valid_positions]

        x_min = float(np.nanmin(xv))
        x_max = float(np.nanmax(xv))
        y_min = float(np.nanmin(yv))
        y_max = float(np.nanmax(yv))
    except Exception:
        return sample_prepared_frame(
            data,
            limit,
            seed=seed,
            force_row_ids=force_row_ids,
        )

    if not all(np.isfinite(value) for value in (x_min, x_max, y_min, y_max)):
        return sample_prepared_frame(
            data,
            limit,
            seed=seed,
            force_row_ids=force_row_ids,
        )

    if x_max <= x_min or y_max <= y_min:
        return sample_prepared_frame(
            data,
            limit,
            seed=seed,
            force_row_ids=force_row_ids,
        )

    rng = np.random.default_rng(seed)

    selected: list[int] = []
    selected_set: set[int] = set()

    # ------------------------------------------------------------------
    # 1. Forced rows first: focus/selected points should never disappear
    #    because of display sampling.
    # ------------------------------------------------------------------
    forced_indices: list[int] = []

    if force_row_ids and INTERNAL_ROW_ID in frame.columns:
        id_series = frame[INTERNAL_ROW_ID]
        id_values = id_series.to_numpy(copy=False)

        # Build string fallback lazily only if needed.
        values_as_str = None

        for forced_id in force_row_ids:
            found = None

            try:
                matches = np.flatnonzero(id_values == forced_id)
                if len(matches):
                    found = int(matches[0])
            except Exception:
                found = None

            if found is None:
                try:
                    numeric_id = pd.to_numeric(forced_id)
                    matches = np.flatnonzero(id_values == numeric_id)
                    if len(matches):
                        found = int(matches[0])
                except Exception:
                    found = None

            if found is None:
                try:
                    if values_as_str is None:
                        values_as_str = id_series.astype(str).to_numpy(copy=False)
                    matches = np.flatnonzero(values_as_str == str(forced_id))
                    if len(matches):
                        found = int(matches[0])
                except Exception:
                    found = None

            if found is not None:
                forced_indices.append(found)

    _append_unique_positions(
        selected=selected,
        selected_set=selected_set,
        candidates=forced_indices,
        max_new=limit,
    )

    # ------------------------------------------------------------------
    # 2. Coverage pass: one representative per occupied plot-space cell.
    # ------------------------------------------------------------------
    remaining_after_forced = max(0, limit - len(selected))
    coverage_budget = int(
        max(
            1,
            min(
                remaining_after_forced,
                round(limit * float(coverage_fraction)),
            ),
        )
    )

    if coverage_budget > 0:
        try:
            x_bins = max(8, int(x_bins))
            y_bins = max(8, int(y_bins))

            xi = np.floor((xv - x_min) / (x_max - x_min) * x_bins).astype(np.int32)
            yi = np.floor((yv - y_min) / (y_max - y_min) * y_bins).astype(np.int32)

            np.clip(xi, 0, x_bins - 1, out=xi)
            np.clip(yi, 0, y_bins - 1, out=yi)

            cell_ids = (yi * x_bins + xi).astype(np.int32, copy=False)

            # Sort by cell only to recover occupied cells cheaply.
            order = np.argsort(cell_ids, kind="mergesort")
            sorted_cells = cell_ids[order]

            first = np.empty(len(sorted_cells), dtype=bool)
            first[0] = True
            first[1:] = sorted_cells[1:] != sorted_cells[:-1]

            coverage_positions = valid_positions[order[first]]

            # If there are more occupied cells than the coverage budget, choose
            # cells randomly. Do not sort/cap by row position afterward.
            if len(coverage_positions) > coverage_budget:
                keep = rng.choice(
                    len(coverage_positions),
                    size=coverage_budget,
                    replace=False,
                )
                coverage_positions = coverage_positions[keep]

            _append_unique_positions(
                selected=selected,
                selected_set=selected_set,
                candidates=coverage_positions,
                max_new=coverage_budget,
            )
        except Exception:
            # Fall through to random fill. The panel should not fail because
            # the coverage sampler hit an unexpected dtype/range edge case.
            pass

    # ------------------------------------------------------------------
    # 3. Random fill: fill the remaining display budget without reintroducing
    #    row-order bias.
    # ------------------------------------------------------------------
    remaining = max(0, limit - len(selected))

    if remaining > 0:
        attempts = 0

        while remaining > 0 and attempts < 16:
            attempts += 1

            draw_size = min(
                len(valid_positions),
                max(remaining * 4, remaining + 256),
            )

            draw_idx = rng.integers(0, len(valid_positions), size=draw_size)
            draw_positions = valid_positions[draw_idx]

            added = _append_unique_positions(
                selected=selected,
                selected_set=selected_set,
                candidates=draw_positions,
                max_new=remaining,
            )

            remaining = max(0, limit - len(selected))

            if added == 0 and draw_size >= len(valid_positions):
                break

    # ------------------------------------------------------------------
    # 4. Final defensive cap. Preserve the intentional order:
    #    forced rows, coverage rows, random fill.
    # ------------------------------------------------------------------
    if len(selected) > limit:
        selected = selected[:limit]

    if not selected:
        return sample_prepared_frame(
            data,
            limit,
            seed=seed,
            force_row_ids=force_row_ids,
        )

    selected_positions = np.asarray(selected, dtype=np.int64)

    sampled = frame.take(selected_positions).copy()

    return PreparedFrame(
        dataset_id=data.dataset_id,
        frame=sampled,
        x_name=data.x_name,
        y_name=data.y_name,
        require_y=data.require_y,
        row_count_before_filter=data.row_count_before_filter,
        row_count_after_filter=len(sampled),
        sampled_from=n_rows,
    )


def frame_in_ranges(data, x_range=None, y_range=None):
    frame = data.frame if hasattr(data, "frame") else data

    if frame is None or frame.empty:
        return data

    mask = np.ones(len(frame), dtype=bool)

    if x_range is not None:
        try:
            x0, x1 = x_range
        except Exception:
            x0, x1 = None, None

        if x0 is not None and x1 is not None and INTERNAL_X in frame.columns:
            lo = min(float(x0), float(x1))
            hi = max(float(x0), float(x1))
            x = frame[INTERNAL_X].to_numpy(copy=False)
            mask &= np.isfinite(x)
            mask &= x >= lo
            mask &= x <= hi

    before_y = int(mask.sum())

    if y_range is not None:
        try:
            y0, y1 = y_range
        except Exception:
            y0, y1 = None, None

        if y0 is not None and y1 is not None and INTERNAL_Y in frame.columns:
            lo = min(float(y0), float(y1))
            hi = max(float(y0), float(y1))
            y = frame[INTERNAL_Y].to_numpy(copy=False)

            y_mask = np.isfinite(y) & (y >= lo) & (y <= hi)
            mask &= y_mask

    filtered = frame.loc[mask]

    if hasattr(data, "frame"):
        return replace(
            data,
            frame=filtered,
            row_count_after_filter=len(filtered),
        )

    return filtered


def row_ids_in_bounds(
    data: PreparedFrame,
    bounds: Optional[Sequence[float]],
    *,
    max_ids: int,
) -> Tuple[List[str], int, bool]:
    if bounds is None or len(bounds) != 4 or data.empty:
        return [], 0, False

    left, bottom, right, top = bounds

    x_min = min(left, right)
    x_max = max(left, right)
    y_min = min(bottom, top)
    y_max = max(bottom, top)

    frame = data.frame
    if INTERNAL_Y not in frame.columns:
        return [], 0, False

    mask = (
        (frame[INTERNAL_X].to_numpy(copy=False) >= x_min)
        & (frame[INTERNAL_X].to_numpy(copy=False) <= x_max)
        & (frame[INTERNAL_Y].to_numpy(copy=False) >= y_min)
        & (frame[INTERNAL_Y].to_numpy(copy=False) <= y_max)
    )

    indices = np.flatnonzero(mask)
    total = int(len(indices))
    if total == 0:
        return [], 0, False

    truncated = total > int(max_ids)
    if truncated:
        indices = indices[: int(max_ids)]

    row_ids = frame.iloc[indices][INTERNAL_ROW_ID].astype(str).tolist()
    return row_ids, total, truncated