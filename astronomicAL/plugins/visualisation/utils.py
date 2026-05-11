from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

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


# Deliberately starts dark/visible for low-count pixels on a white background.
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
    """Return a HoloViews hook that names and targets the main glyph renderer.

    Bokeh versions differ here:
    - some support HoverTool.names;
    - some only support HoverTool.renderers;
    - some support HoverTool.limit;
    - some do not.

    We avoid HoverTool.names entirely and bind hover tools to the main renderer
    after HoloViews has created it.
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

                    # Restrict hover to the main renderer where supported.
                    # This avoids overlays/focus/selection glyphs contributing
                    # extra tooltip rows.
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


def _hover_tool(*, tooltips, names: Optional[List[str]] = None) -> HoverTool:
    """Create a compact hover tool.

    Do not pass ``names`` to HoverTool. Some Bokeh versions do not support that
    attribute and raise:

        unexpected attribute 'names' to HoverTool

    Renderer targeting is instead applied later in ``renderer_name_hook``.
    """
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
        names=[SCATTER_RENDERER],
        tooltips=[
            ("x", f"@{INTERNAL_X}"),
            ("y", f"@{INTERNAL_Y}"),
            ("id", f"@{INTERNAL_ROW_ID}"),
            ("label", f"@{INTERNAL_LABEL_DISPLAY}"),
        ],
    )


def limited_histogram_hover_tool() -> HoverTool:
    return _hover_tool(
        names=[HIST_RENDERER],
        tooltips=[
            ("x", "$x"),
            ("count", "$y"),
        ],
    )


def limited_density_hover_tool() -> HoverTool:
    return _hover_tool(
        names=[DENSITY_RENDERER],
        tooltips=[
            ("x", "$x"),
            ("y", "$y"),
            ("count", "$image"),
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
        return context.datasets.get_df()
    except Exception:
        return None


def _mapped_column(
    context,
    dataset_id: Optional[str],
    semantic: str,
    *,
    allow_index: bool,
) -> Optional[Any]:
    datasets = getattr(context, "datasets", None)

    if datasets is not None and dataset_id is not None:
        try:
            value = datasets.get_mapping(dataset_id, semantic)
            if value == "Use Index" and allow_index:
                return value

            df = datasets.get_df(dataset_id)
            if value in df.columns:
                return value
        except Exception:
            pass

    cfg = getattr(context, "config", None)
    settings = getattr(cfg, "settings", {}) if cfg is not None else {}
    df = _active_df(context)

    if semantic == "record_id":
        value = settings.get("id_col")
        if value == "Use Index" and allow_index:
            return value
        if df is not None and value in df.columns:
            return value

    if semantic == "target_label":
        value = settings.get("label_col")
        if df is not None and value in df.columns:
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
    series = _safe_series(df, column)
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype="float64", copy=False)


def _row_id_array(df: pd.DataFrame, record_id_col: Optional[Any], mask: np.ndarray) -> np.ndarray:
    if record_id_col == "Use Index" or not record_id_col or record_id_col not in df.columns:
        try:
            return df.index[mask].astype(str).to_numpy()
        except AttributeError:
            return np.asarray(df.index[mask], dtype=str)

    values = _safe_series(df, record_id_col)
    return values.loc[mask].astype(str).to_numpy()


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
    dataset_id = _active_dataset_id(context)
    df = _active_df(context)

    if df is None or df.empty or not state.x:
        return PreparedFrame(
            dataset_id=dataset_id,
            frame=pd.DataFrame(),
            x_name=state.x,
            y_name=state.y if require_y else None,
            require_y=require_y,
            row_count_before_filter=0,
            row_count_after_filter=0,
        )

    if state.x not in df.columns:
        return PreparedFrame(
            dataset_id=dataset_id,
            frame=pd.DataFrame(),
            x_name=state.x,
            y_name=state.y if require_y else None,
            require_y=require_y,
            row_count_before_filter=len(df),
            row_count_after_filter=0,
        )

    if require_y and (not state.y or state.y not in df.columns):
        return PreparedFrame(
            dataset_id=dataset_id,
            frame=pd.DataFrame(),
            x_name=state.x,
            y_name=state.y,
            require_y=require_y,
            row_count_before_filter=len(df),
            row_count_after_filter=0,
        )

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

    row_ids = _row_id_array(df, state.record_id_col, mask)

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

    frame = pd.DataFrame(data)

    return PreparedFrame(
        dataset_id=dataset_id,
        frame=frame,
        x_name=state.x,
        y_name=state.y if require_y else None,
        require_y=require_y,
        row_count_before_filter=len(df),
        row_count_after_filter=len(frame),
    )


def prepared_cache_key(context, state, *, require_y: bool) -> Tuple[Any, ...]:
    df = _active_df(context)
    dataset_id = _active_dataset_id(context)

    return (
        dataset_id,
        id(df),
        len(df) if df is not None else 0,
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


def sample_prepared_frame(data: PreparedFrame, limit: int, *, seed: int = 0) -> PreparedFrame:
    frame = data.frame
    n_rows = len(frame)
    limit = int(limit)

    if n_rows <= limit or limit <= 0:
        return data

    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(n_rows, size=limit, replace=False))
    sampled = frame.iloc[indices].copy()

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


def row_ids_to_mask(row_ids: Sequence[str], candidate_ids: np.ndarray) -> np.ndarray:
    if not row_ids or len(candidate_ids) == 0:
        return np.zeros(len(candidate_ids), dtype=bool)

    return np.isin(candidate_ids.astype(str), np.asarray([str(value) for value in row_ids], dtype=str))