from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import time

import numpy as np
import pandas as pd
from bokeh.models import HoverTool, WheelZoomTool
from bokeh.palettes import Category10, Category20

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

def prepare_plot_frame(context, state, *, require_y: bool) -> PreparedFrame:
    t_total = time.perf_counter()

    print(
        "[AstronomicAL visualisation] prepare_plot_frame called: "
        f"x={state.x!r}, y={state.y!r}, require_y={require_y}",
        flush=True,
    )

    dataset_id = _active_dataset_id(context)
    total_rows = _dataset_row_count(context, dataset_id)

    if not state.x:
        return PreparedFrame(
            dataset_id=dataset_id,
            frame=pd.DataFrame(),
            x_name=state.x,
            y_name=state.y if require_y else None,
            require_y=require_y,
            row_count_before_filter=total_rows,
            row_count_after_filter=0,
        )

    available_columns = _dataset_columns(context, dataset_id)
    available_set = set(available_columns)

    if state.x not in available_set:
        return PreparedFrame(
            dataset_id=dataset_id,
            frame=pd.DataFrame(),
            x_name=state.x,
            y_name=state.y if require_y else None,
            require_y=require_y,
            row_count_before_filter=total_rows,
            row_count_after_filter=0,
        )

    if require_y and (not state.y or state.y not in available_set):
        return PreparedFrame(
            dataset_id=dataset_id,
            frame=pd.DataFrame(),
            x_name=state.x,
            y_name=state.y,
            require_y=require_y,
            row_count_before_filter=total_rows,
            row_count_after_filter=0,
        )

    required_columns = _plot_required_columns(
        context,
        dataset_id,
        state,
        require_y=require_y,
    )

    # ------------------------------------------------------------------
    # 1. Data load timing
    # ------------------------------------------------------------------
    t_load = time.perf_counter()

    df = _get_dataset_view_for_columns(
        context,
        dataset_id,
        required_columns,
    )

    load_seconds = time.perf_counter() - t_load

    if df is None:
        print(
            "[AstronomicAL visualisation] Loaded plot view: df=None "
            f"in {load_seconds:.3f}s",
            flush=True,
        )

        return PreparedFrame(
            dataset_id=dataset_id,
            frame=pd.DataFrame(),
            x_name=state.x,
            y_name=state.y if require_y else None,
            require_y=require_y,
            row_count_before_filter=total_rows,
            row_count_after_filter=0,
        )

    print(
        "[AstronomicAL visualisation] Loaded plot view: "
        f"{len(df):,} rows × {len(df.columns):,} columns "
        f"in {load_seconds:.3f}s. "
        f"Columns: {list(df.columns)}",
        flush=True,
    )

    if df.empty:
        return PreparedFrame(
            dataset_id=dataset_id,
            frame=pd.DataFrame(),
            x_name=state.x,
            y_name=state.y if require_y else None,
            require_y=require_y,
            row_count_before_filter=total_rows,
            row_count_after_filter=0,
        )

    if state.x not in df.columns:
        return PreparedFrame(
            dataset_id=dataset_id,
            frame=pd.DataFrame(),
            x_name=state.x,
            y_name=state.y if require_y else None,
            require_y=require_y,
            row_count_before_filter=total_rows,
            row_count_after_filter=0,
        )

    if require_y and (not state.y or state.y not in df.columns):
        return PreparedFrame(
            dataset_id=dataset_id,
            frame=pd.DataFrame(),
            x_name=state.x,
            y_name=state.y,
            require_y=require_y,
            row_count_before_filter=total_rows,
            row_count_after_filter=0,
        )

    # ------------------------------------------------------------------
    # 2. Numeric arrays + mask timing
    # ------------------------------------------------------------------
    t_arrays = time.perf_counter()

    x = _numeric_array(df, state.x)
    mask = np.isfinite(x)

    y = None
    if require_y:
        y = _numeric_array(df, state.y)
        mask &= np.isfinite(y)

    if state.log_x:
        mask &= x > 0

    if require_y and state.log_y and y is not None:
        mask &= y > 0

    arrays_seconds = time.perf_counter() - t_arrays

    print(
        "[AstronomicAL visualisation] prepare arrays/mask "
        f"{arrays_seconds:.3f}s "
        f"finite_rows={int(mask.sum()):,}/{len(mask):,}",
        flush=True,
    )

    # ------------------------------------------------------------------
    # 3. Label arrays timing
    # ------------------------------------------------------------------
    t_labels = time.perf_counter()

    label_raw = None
    label_display = None
    label_colours = None

    if state.label_col and state.label_col in df.columns:
        label_raw, label_display, label_colours = _label_arrays(
            df,
            state.label_col,
            mask,
            state,
        )

    labels_seconds = time.perf_counter() - t_labels

    print(
        "[AstronomicAL visualisation] prepare labels "
        f"{labels_seconds:.3f}s "
        f"label_col={state.label_col!r}",
        flush=True,
    )

    # ------------------------------------------------------------------
    # 4. Row ID timing
    # ------------------------------------------------------------------
    t_row_ids = time.perf_counter()

    row_ids = _row_id_array(df, state.record_id_col, mask)

    row_ids_seconds = time.perf_counter() - t_row_ids

    print(
        "[AstronomicAL visualisation] prepare row_ids "
        f"{row_ids_seconds:.3f}s "
        f"record_id_col={state.record_id_col!r}",
        flush=True,
    )

    # ------------------------------------------------------------------
    # 6. Masked array extraction + dict construction timing
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
        data[INTERNAL_LABEL_DISPLAY] = label_display
        data[INTERNAL_LABEL_COLOUR] = label_colours

    data_seconds = time.perf_counter() - t_data

    print(
        "[AstronomicAL visualisation] prepare data dict "
        f"{data_seconds:.3f}s",
        flush=True,
    )

    # ------------------------------------------------------------------
    # 7. PreparedFrame DataFrame construction timing
    # ------------------------------------------------------------------
    t_frame = time.perf_counter()

    frame = pd.DataFrame(data, copy=False)

    frame_seconds = time.perf_counter() - t_frame

    print(
        "[AstronomicAL visualisation] prepare dataframe "
        f"{frame_seconds:.3f}s "
        f"frame_rows={len(frame):,} frame_cols={len(frame.columns):,}",
        flush=True,
    )

    total_seconds = time.perf_counter() - t_total

    print(
        "[AstronomicAL visualisation] prepare_plot_frame total "
        f"{total_seconds:.3f}s "
        f"(load={load_seconds:.3f}s, "
        f"arrays={arrays_seconds:.3f}s, "
        f"labels={labels_seconds:.3f}s, "
        f"row_ids={row_ids_seconds:.3f}s, "
        f"data={data_seconds:.3f}s, "
        f"frame={frame_seconds:.3f}s)",
        flush=True,
    )

    return PreparedFrame(
        dataset_id=dataset_id,
        frame=frame,
        x_name=state.x,
        y_name=state.y if require_y else None,
        require_y=require_y,
        row_count_before_filter=total_rows or len(df),
        row_count_after_filter=len(frame),
    )

def prepared_cache_key(context, state, *, require_y: bool) -> Tuple[Any, ...]:
    dataset_id = _active_dataset_id(context)

    return (
        _dataset_fingerprint(context, dataset_id),
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

def frame_in_ranges(
    data: PreparedFrame,
    x_range: Optional[Sequence[float]],
    y_range: Optional[Sequence[float]],
) -> PreparedFrame:
    frame = data.frame
    if frame.empty:
        return data

    mask = np.ones(len(frame), dtype=bool)

    if x_range and len(x_range) == 2 and all(v is not None for v in x_range):
        x0, x1 = float(x_range[0]), float(x_range[1])
        lo, hi = min(x0, x1), max(x0, x1)
        x = frame[INTERNAL_X].to_numpy(copy=False)
        mask &= (x >= lo) & (x <= hi)

    if (
        y_range
        and len(y_range) == 2
        and all(v is not None for v in y_range)
        and INTERNAL_Y in frame.columns
    ):
        y0, y1 = float(y_range[0]), float(y_range[1])
        lo, hi = min(y0, y1), max(y0, y1)
        y = frame[INTERNAL_Y].to_numpy(copy=False)
        mask &= (y >= lo) & (y <= hi)

    if mask.all():
        return data

    restricted = frame.loc[mask].copy()

    return PreparedFrame(
        dataset_id=data.dataset_id,
        frame=restricted,
        x_name=data.x_name,
        y_name=data.y_name,
        require_y=data.require_y,
        row_count_before_filter=data.row_count_before_filter,
        row_count_after_filter=len(restricted),
        sampled_from=None,
    )

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