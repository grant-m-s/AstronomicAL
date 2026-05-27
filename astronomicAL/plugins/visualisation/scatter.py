# BUG: Assign Label col slow
# BUG: Individual legend "on off" colour turns all colours off - works correctly in hist
# BUG: If label set, changing label name or colour (re-apply label settings) no update happens.

from __future__ import annotations

import time
from typing import List, Optional, Sequence

import datashader as ds
import holoviews as hv
import numpy as np
import pandas as pd
import panel as pn
from holoviews import streams
from holoviews.operation.datashader import rasterize

from .base import BaseVisualisationPanel
from .constants import (
    INTERNAL_LABEL_DISPLAY,
    INTERNAL_ROW_ID,
    INTERNAL_X,
    INTERNAL_Y,
    PLOT_MIN_HEIGHT,
    SELECTION_OVERLAY_DEBUG_LIMIT,
    COVERAGE_SAMPLE_MIN_ROWS,
    COVERAGE_SAMPLE_X_BINS,
    COVERAGE_SAMPLE_Y_BINS,
    COVERAGE_SAMPLE_FRACTION,
    INTERACTIVE_DENSITY_UNDERLAY_ENABLED,
    INTERACTIVE_DENSITY_UNDERLAY_MIN_ROWS,
    INTERACTIVE_DENSITY_UNDERLAY_ALPHA,
    INTERACTIVE_DENSITY_CANVAS_WIDTH,
    INTERACTIVE_DENSITY_CANVAS_HEIGHT,
    INTERACTIVE_DENSITY_GREY_RGB,
    INTERACTIVE_DENSITY_ALPHA_GAMMA,
    FULL_RANGE_REL_TOL,
)
from .utils import (
    DENSITY_RENDERER,
    HOVER_LABEL,
    HOVER_ROW_ID,
    PreparedFrame,
    SCATTER_RENDERER,
    VISIBLE_DENSITY_CMAP,
    force_wheel_zoom_hook,
    frame_in_ranges,
    limited_point_hover_tool,
    renderer_name_hook,
    row_ids_in_bounds,
    sample_prepared_frame,
    deduplicate_toolbar_tools_hook,
    coverage_sample_prepared_frame,
    keep_pan_tool_active_hook,
)

SELECTION_OVERLAY_METADATA_KEY = "visualisation.scatter.overlay_points"
SELECTION_OVERLAY_LIMIT_DEFAULT = 5000


class ScatterPanel(BaseVisualisationPanel):
    title = "Scatter Plot"


    def _frame_extent_cache_key(self, data):
        try:
            return id(self._frame_for_cache(data))
        except Exception:
            return id(data)


    def _frame_extent(self, data):
        """
        Cached finite x/y extent for the prepared frame.

        Used to collapse ranges that are effectively the full data extent back to
        None, improving cache reuse and avoiding repeated full-frame density/sample
        work.
        """

        key = self._frame_extent_cache_key(data)
        cache = getattr(self, "_extent_cache", None)

        if cache is None:
            self._extent_cache = {}
            cache = self._extent_cache

        if key in cache:
            return cache[key]

        frame = self._frame_for_cache(data)

        try:
            x = pd.to_numeric(frame[INTERNAL_X], errors="coerce").to_numpy(copy=False)
            y = pd.to_numeric(frame[INTERNAL_Y], errors="coerce").to_numpy(copy=False)

            finite_x = x[np.isfinite(x)]
            finite_y = y[np.isfinite(y)]

            if len(finite_x) == 0 or len(finite_y) == 0:
                extent = None
            else:
                extent = (
                    float(np.nanmin(finite_x)),
                    float(np.nanmax(finite_x)),
                    float(np.nanmin(finite_y)),
                    float(np.nanmax(finite_y)),
                )
        except Exception:
            extent = None

        cache[key] = extent

        # Keep tiny.
        if len(cache) > 12:
            try:
                first_key = next(iter(cache))
                cache.pop(first_key, None)
            except Exception:
                cache.clear()

        return extent


    def _range_covers_axis_extent(self, range_value, min_value: float, max_value: float) -> bool:
        if range_value is None:
            return True

        try:
            lo, hi = range_value
            lo = float(lo)
            hi = float(hi)
        except Exception:
            return False

        if not np.isfinite(lo) or not np.isfinite(hi):
            return False

        span = max(abs(max_value - min_value), 1.0)
        tol = span * float(FULL_RANGE_REL_TOL)

        return lo <= min_value + tol and hi >= max_value - tol


    def _normalise_interactive_ranges(self, data, x_range=None, y_range=None):
        """
        Collapse stream ranges that cover the full prepared-frame extent to None.

        This avoids treating tiny auto-range differences as unique zoom states.
        It improves point-sample cache hits and density cache hits.
        """

        extent = self._frame_extent(data)

        if extent is None:
            return x_range, y_range

        x_min, x_max, y_min, y_max = extent

        if self._range_covers_axis_extent(x_range, x_min, x_max):
            x_range = None

        if self._range_covers_axis_extent(y_range, y_min, y_max):
            y_range = None

        return x_range, y_range

    def _current_focus_forced_ids(self):
        selection = getattr(self.context, "selection", None)

        if selection is None:
            return ()

        try:
            focus = selection.get_focus()
        except Exception:
            return ()

        if focus is None:
            return ()

        dataset_id = self._dataset_id()

        if getattr(focus, "dataset_id", None) != dataset_id:
            return ()

        row_id = str(getattr(focus, "row_id", "") or "")

        if not row_id:
            return ()

        record_id_col = getattr(self.state, "record_id_col", None)

        if not record_id_col or record_id_col == "Use Index":
            return ()

        # Only force if the cached row can be converted into a point for
        # the current x/y/log settings.
        point = self._focus_point_from_cached_focus_row(focus)

        if point is None:
            return ()

        return (row_id,)

    def _range_coverage_fraction(self, range_value, data_min, data_max) -> float:
        if not range_value:
            return 1.0

        try:
            lo, hi = range_value
            lo = float(lo)
            hi = float(hi)
            data_min = float(data_min)
            data_max = float(data_max)
        except Exception:
            return 0.0

        if hi < lo:
            lo, hi = hi, lo

        data_span = max(abs(data_max - data_min), 1.0e-12)

        overlap_lo = max(lo, data_min)
        overlap_hi = min(hi, data_max)

        overlap = max(0.0, overlap_hi - overlap_lo)

        return overlap / data_span

    def _use_coverage_sample(self, visible_count: int, effective_limit: int) -> bool:
        return (
            int(visible_count) > int(effective_limit)
            and int(visible_count) >= int(COVERAGE_SAMPLE_MIN_ROWS)
        )


    def _coverage_sample_note(self, visible_count: int, effective_limit: int) -> str:
        if self._use_coverage_sample(visible_count, effective_limit):
            return "coverage-aware sample"
        return "display sample"


    def _use_interactive_density_underlay(self, data) -> bool:

        if not bool(INTERACTIVE_DENSITY_UNDERLAY_ENABLED):
            return False

        if bool(getattr(self.state, "log_x", False)) or bool(getattr(self.state, "log_y", False)):
            return False

        try:
            n_rows = self._frame_len(data)
        except Exception:
            return False

        if n_rows < int(INTERACTIVE_DENSITY_UNDERLAY_MIN_ROWS):
            return False

        try:
            sample_limit = int(getattr(self.state, "interactive_sample_limit", 0))
        except Exception:
            sample_limit = 0

        return sample_limit > 0 and n_rows > sample_limit


    def _compose_interactive_density_layers(self, *, density, points):

        if density is None:
            return points

        try:
            return density * points
        except Exception as exc:
            print(
                "[AstronomicAL scatter] density underlay composition failed; "
                "falling back to sampled points only "
                f"panel_id={self.panel_id} "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            return points

    def _interactive_density_underlay(self, data):
        """
        Optional passive density underlay for large interactive sampled views.

        This must never prevent the scatter panel from opening. If Datashader or
        HoloViews cannot build the density layer for any reason, return None and
        let the sampled scatter render normally.
        """

        if not self._use_interactive_density_underlay(data):
            return None

        try:
            frame = data.frame[[INTERNAL_X, INTERNAL_Y]].copy(deep=False)
        except Exception:
            return None

        if frame is None or frame.empty:
            return None

        try:
            x = pd.to_numeric(frame[INTERNAL_X], errors="coerce")
            y = pd.to_numeric(frame[INTERNAL_Y], errors="coerce")

            finite = np.isfinite(x.to_numpy(copy=False)) & np.isfinite(
                y.to_numpy(copy=False)
            )

            if bool(getattr(self.state, "log_x", False)):
                finite &= x.to_numpy(copy=False) > 0

            if bool(getattr(self.state, "log_y", False)):
                finite &= y.to_numpy(copy=False) > 0

            if not finite.any():
                return None

            frame = frame.loc[finite].copy()

            # Datashader/HoloViews can behave badly on degenerate ranges.
            if len(frame) < 2:
                return None

            x_min = float(frame[INTERNAL_X].min())
            x_max = float(frame[INTERNAL_X].max())
            y_min = float(frame[INTERNAL_Y].min())
            y_max = float(frame[INTERNAL_Y].max())

            if not all(np.isfinite(value) for value in (x_min, x_max, y_min, y_max)):
                return None

            if x_min == x_max or y_min == y_max:
                return None
        except Exception:
            return None

        try:
            points = hv.Points(
                frame,
                kdims=[INTERNAL_X, INTERNAL_Y],
            )

            range_opts = self._current_range_opts(include_y=True)

            return rasterize(
                points,
                aggregator=ds.count(),
                pixel_ratio=1,
            ).opts(
                cmap=VISIBLE_DENSITY_CMAP,
                cnorm="eq_hist",
                colorbar=False,
                alpha=float(INTERACTIVE_DENSITY_UNDERLAY_ALPHA),
                bgcolor="white",
                responsive=True,
                min_height=PLOT_MIN_HEIGHT,
                xlabel=str(self.state.x),
                ylabel=str(self.state.y),
                logx=self.state.log_x,
                logy=self.state.log_y,
                tools=[],
                active_tools=[],
                toolbar=None,
                hooks=[renderer_name_hook(DENSITY_RENDERER)],
                shared_axes=False,
                axiswise=True,
                framewise=True,
                **range_opts,
            )
        except Exception as exc:
            print(
                "[AstronomicAL scatter] density underlay disabled after build failure "
                f"panel_id={self.panel_id} "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            return None


    def _base_sample_cache_key(self, effective_limit: int):
        return (
            getattr(self, "_last_prepared_cache_key", None),
            int(effective_limit),
            "coverage-v2-no-row-order-cap",
            int(COVERAGE_SAMPLE_X_BINS),
            int(COVERAGE_SAMPLE_Y_BINS),
            float(COVERAGE_SAMPLE_FRACTION),
        )


    def _base_sample_cache_get(self, key):
        value = self._base_sample_cache.get(key)

        return value


    def _base_sample_cache_set(self, key, plot_data):
        size_before = len(self._base_sample_cache)

        if key in self._base_sample_cache:
            self._base_sample_cache.pop(key, None)

        self._base_sample_cache[key] = plot_data

        while len(self._base_sample_cache) > self._base_sample_cache_max:
            try:
                first_key = next(iter(self._base_sample_cache))
                self._base_sample_cache.pop(first_key, None)
            except Exception:
                self._base_sample_cache.clear()
                break

    def _current_focus_row_id(self):
        selection = getattr(self.context, "selection", None)
        if selection is None:
            return None

        try:
            focus = selection.get_focus()
        except Exception:
            return None

        if focus is None:
            return None

        if getattr(focus, "dataset_id", None) != self._dataset_id():
            return None

        row_id = getattr(focus, "row_id", None)
        return None if row_id is None else str(row_id)


    def _forced_row_ids_for_sampling(self):
        """Rows that must be injected into the sampled cloud.

        The current focus is excluded because it is drawn by the focus overlay.
        """
        forced = list(self._forced_row_ids() or [])
        focus_row_id = self._current_focus_row_id()

        if focus_row_id is None:
            return tuple(str(value) for value in forced)

        return tuple(
            str(value)
            for value in forced
            if str(value) != focus_row_id
        )

    def _near_full_range_key(self, x_range, y_range):
        return (
            getattr(self, "_last_prepared_cache_key", None),
            self._range_cache_key(x_range),
            self._range_cache_key(y_range),
        )

    def _range_contains_extent(self, data, x_range, y_range) -> bool:
        extent = self._prepared_data_extent(data)

        if extent is None:
            return False

        x_min, x_max, y_min, y_max = extent

        try:
            x0, x1 = x_range
            y0, y1 = y_range

            x0 = float(x0)
            x1 = float(x1)
            y0 = float(y0)
            y1 = float(y1)

            return (
                min(x0, x1) <= float(x_min)
                and max(x0, x1) >= float(x_max)
                and min(y0, y1) <= float(y_min)
                and max(y0, y1) >= float(y_max)
            )
        except Exception:
            return False

    def _is_known_near_full_range(self, x_range, y_range) -> bool:
        return self._near_full_range_key(x_range, y_range) in self._near_full_range_keys


    def _remember_near_full_range(self, x_range, y_range) -> None:
        key = self._near_full_range_key(x_range, y_range)

        self._near_full_range_keys.add(key)

        while len(self._near_full_range_keys) > self._near_full_range_keys_max:
            try:
                self._near_full_range_keys.pop()
            except Exception:
                self._near_full_range_keys.clear()
                break

    def _prepared_data_extent(self, data):
        """Return cached x/y min/max for the current prepared frame."""
        prepared_key = getattr(self, "_last_prepared_cache_key", None)

        if (
            self._prepared_extent_cache_key == prepared_key
            and self._prepared_extent_cache is not None
        ):
            return self._prepared_extent_cache

        frame = self._frame_for_cache(data)

        try:
            x = frame[INTERNAL_X].to_numpy(copy=False)
            y = frame[INTERNAL_Y].to_numpy(copy=False)

            extent = (
                float(np.nanmin(x)),
                float(np.nanmax(x)),
                float(np.nanmin(y)),
                float(np.nanmax(y)),
            )
        except Exception:
            extent = None

        self._prepared_extent_cache_key = prepared_key
        self._prepared_extent_cache = extent

        return extent


    def _range_contains_extent(self, data, x_range, y_range) -> bool:
        extent = self._prepared_data_extent(data)

        if extent is None:
            return False

        x_min, x_max, y_min, y_max = extent

        try:
            x0, x1 = x_range
            y0, y1 = y_range

            return (
                min(float(x0), float(x1)) <= float(x_min)
                and max(float(x0), float(x1)) >= float(x_max)
                and min(float(y0), float(y1)) <= float(y_min)
                and max(float(y0), float(y1)) >= float(y_max)
            )
        except Exception:
            return False

    def _range_is_near_full_extent(self, data, x_range, y_range) -> bool:
        extent = self._prepared_data_extent(data)

        if extent is None:
            return False

        x_min, x_max, y_min, y_max = extent

        x_fraction = self._range_coverage_fraction(x_range, x_min, x_max)
        y_fraction = self._range_coverage_fraction(y_range, y_min, y_max)

        # Bokeh often emits padded ranges after an axis reset. Treat almost-full
        # ranges as full to avoid building a 13M-row boolean mask.
        if x_fraction >= 0.80 and y_fraction >= 0.80:
            return True

        # Categorical/integer-like axes often reset to exact padded bounds such as
        # (0, 1), (0, 7), etc. If the emitted range fully contains the prepared
        # extent, it is full-range even if the coverage calculation is imperfect.
        try:
            x0, x1 = x_range
            y0, y1 = y_range

            contains_x = float(x0) <= float(x_min) and float(x1) >= float(x_max)
            contains_y = float(y0) <= float(y_min) and float(y1) >= float(y_max)

            if contains_x and contains_y:
                return True
        except Exception:
            pass

        return False

    def _range_is_full_extent(self, data, x_range, y_range) -> bool:
        extent = self._prepared_data_extent(data)

        if extent is None:
            return False

        x_min, x_max, y_min, y_max = extent

        x_full = self._range_contains_extent(x_range, x_min, x_max)
        y_full = self._range_contains_extent(y_range, y_min, y_max)

        return x_full and y_full

    def _range_cache_value(self, value):
        if value is None:
            return None
        try:
            return round(float(value), 8)
        except Exception:
            return str(value)


    def _frame_for_cache(self, data_or_frame):
        if hasattr(data_or_frame, "frame"):
            return data_or_frame.frame
        return data_or_frame


    def _frame_len(self, data_or_frame):
        frame = self._frame_for_cache(data_or_frame)
        try:
            return len(frame)
        except Exception:
            return 0

    def _interactive_sample_cache_key(
        self,
        data,
        x_range,
        y_range,
        *,
        forced_ids=(),
        limit=None,
    ):
        return (
            id(self._frame_for_cache(data)),
            self._range_cache_key(x_range),
            self._range_cache_key(y_range),
            int(limit or 0),
            tuple(str(row_id) for row_id in forced_ids or ()),
            str(getattr(self.state, "color_by", None)),
            str(getattr(self.state, "label_col", None)),
            tuple(getattr(self.state, "label_filter", None) or ()),
            "coverage-v2-no-row-order-cap",
            int(COVERAGE_SAMPLE_X_BINS),
            int(COVERAGE_SAMPLE_Y_BINS),
            float(COVERAGE_SAMPLE_FRACTION),
            bool(self._use_interactive_density_underlay(data)),
        )


    def _interactive_sample_cache_get(self, key):
        value = self._interactive_sample_cache.get(key)

        return value


    def _interactive_sample_cache_set(self, key, element, status_text):
        size_before = len(self._interactive_sample_cache)

        if key in self._interactive_sample_cache:
            self._interactive_sample_cache.pop(key, None)

        self._interactive_sample_cache[key] = (element, status_text)

        while len(self._interactive_sample_cache) > self._interactive_sample_cache_max:
            try:
                first_key = next(iter(self._interactive_sample_cache))
                self._interactive_sample_cache.pop(first_key, None)
            except Exception:
                self._interactive_sample_cache.clear()
                break

    def _forced_positions_in_frame(self, frame: pd.DataFrame, forced_ids) -> list[int]:
        """
        Find forced row-id positions without stringifying the full row-id column.

        This avoids the expensive path:
            frame[INTERNAL_ROW_ID].astype(str)

        for every range update.
        """
        if not forced_ids or frame is None or frame.empty:
            return []

        if INTERNAL_ROW_ID not in frame.columns:
            return []

        values = frame[INTERNAL_ROW_ID].to_numpy(copy=False)
        positions: list[int] = []

        for forced_id in forced_ids:
            found = None

            # Fast path for numeric row IDs.
            try:
                if np.issubdtype(values.dtype, np.integer):
                    lookup_value = np.int64(forced_id)
                elif np.issubdtype(values.dtype, np.floating):
                    lookup_value = float(forced_id)
                else:
                    lookup_value = forced_id

                matches = np.flatnonzero(values == lookup_value)
                if len(matches):
                    found = int(matches[0])
            except Exception:
                found = None

            # Fallback for object/string ID columns only.
            if found is None:
                try:
                    matches = np.flatnonzero(values == forced_id)
                    if len(matches):
                        found = int(matches[0])
                except Exception:
                    found = None

            # Last-resort fallback. This should rarely run.
            if found is None and values.dtype == object:
                try:
                    matches = np.flatnonzero(values.astype(str) == str(forced_id))
                    if len(matches):
                        found = int(matches[0])
                except Exception:
                    found = None

            if found is not None:
                positions.append(found)

        return positions

    def _focus_point_from_dataset_row(self) -> Optional[tuple[float, Optional[float]]]:
        """
        Resolve the focused point from the DatasetSource instead of scanning the
        full prepared plotting frame.

        This is important after axis changes: existing focus metadata may not
        match the new x/y variables, but the focused row can be fetched directly
        from the Parquet-backed dataset.
        """
        selection = getattr(self.context, "selection", None)
        datasets = getattr(self.context, "datasets", None)

        if selection is None or datasets is None:
            return None

        try:
            focus = selection.get_focus()
        except Exception:
            return None

        dataset_id = self._dataset_id()

        if focus is None or getattr(focus, "dataset_id", None) != dataset_id:
            return None

        row_id = getattr(focus, "row_id", None)
        if row_id is None:
            return None

        x_col = getattr(self.state, "x", None)
        y_col = getattr(self.state, "y", None)
        record_id_col = getattr(self.state, "record_id_col", None)

        if not x_col or not y_col or not record_id_col or record_id_col == "Use Index":
            return None

        try:
            source = datasets.get_source(dataset_id)
        except Exception:
            return None

        try:
            row_df = source.to_pandas(
                columns=[record_id_col, x_col, y_col],
                where_sql=f'CAST("{record_id_col}" AS VARCHAR) = ?',
                params=[str(row_id)],
                limit=1,
            )
        except TypeError:
            # Older DatasetSource implementations may not support where_sql/params.
            return None
        except Exception:
            return None

        if row_df is None or row_df.empty:
            return None

        try:
            x = float(row_df.iloc[0][x_col])
            y = float(row_df.iloc[0][y_col])
        except Exception:
            return None

        if not np.isfinite(x) or not np.isfinite(y):
            return None

        if getattr(self.state, "log_x", False) and x <= 0:
            return None

        if getattr(self.state, "log_y", False) and y <= 0:
            return None

        return x, y

    def _row_id_lookup_for_frame(self, data):
        """
        Build a raw row-id -> integer position lookup for the current prepared frame.

        This is only used when forced IDs need to be included in an interactive
        sample.
        """
        if not hasattr(self, "_row_id_lookup_cache"):
            self._row_id_lookup_cache = {}

        cache_key = (id(data.frame), self._frame_len(data))

        cached = self._row_id_lookup_cache.get(cache_key)
        if cached is not None:
            return cached

        frame = data.frame

        if INTERNAL_ROW_ID not in frame.columns:
            lookup = {}
        else:
            values = frame[INTERNAL_ROW_ID].to_numpy(copy=False)

            lookup = {}
            for idx, value in enumerate(values):
                lookup[value] = idx
                lookup[str(value)] = idx

        self._row_id_lookup_cache[cache_key] = lookup

        if len(self._row_id_lookup_cache) > 4:
            try:
                first_key = next(iter(self._row_id_lookup_cache))
                self._row_id_lookup_cache.pop(first_key, None)
            except Exception:
                self._row_id_lookup_cache.clear()

        return lookup

    def _bokeh_safe_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        """
        Return a display-only frame safe to send to Bokeh/JavaScript.

        The full server-side PreparedFrame should stay lean. Hover-only fields are
        added here after sampling, not during 1M-row preparation.
        """
        if frame is None or frame.empty:
            return frame

        out = frame.copy(deep=False)

        if HOVER_ROW_ID not in out.columns and INTERNAL_ROW_ID in out.columns:
            out[HOVER_ROW_ID] = out[INTERNAL_ROW_ID].astype(str)

        if HOVER_LABEL not in out.columns:
            if INTERNAL_LABEL_DISPLAY in out.columns:
                out[HOVER_LABEL] = out[INTERNAL_LABEL_DISPLAY].astype(str)
            else:
                out[HOVER_LABEL] = "—"

        # Avoid Bokeh/JavaScript integer precision warnings.
        id_columns = [
            INTERNAL_ROW_ID,
            HOVER_ROW_ID,
        ]

        for col in id_columns:
            if col in out.columns:
                out[col] = out[col].astype(str)

        js_safe_max = 2**53 - 1
        js_safe_min = -(2**53 - 1)

        for col in out.columns:
            if col in id_columns:
                continue

            try:
                if pd.api.types.is_integer_dtype(out[col].dtype):
                    col_min = out[col].min()
                    col_max = out[col].max()

                    if col_min < js_safe_min or col_max > js_safe_max:
                        out[col] = out[col].astype(str)
            except Exception:
                pass

        return out

    def _sample_visible_frame(self, data, visible, effective_limit: int, forced_ids):
        forced_ids = list(forced_ids or [])

        visible_count = self._frame_len(visible)
        use_coverage = self._use_coverage_sample(visible_count, effective_limit)

        if use_coverage:
            print(
                "[AstronomicAL scatter] using coverage-aware sample "
                f"panel_id={self.panel_id} "
                f"visible={visible_count:,} "
                f"limit={int(effective_limit):,} "
                f"x_bins={COVERAGE_SAMPLE_X_BINS} "
                f"y_bins={COVERAGE_SAMPLE_Y_BINS}",
                flush=True,
            )

        base_sample_key = None

        # Only cache the global full-frame no-forced sample. Range-filtered samples
        # are already cached by _interactive_sample_cache.
        if self._same_underlying_frame(visible, data) and not forced_ids:
            base_sample_key = self._base_sample_cache_key(effective_limit)
            cached = self._base_sample_cache_get(base_sample_key)
            if cached is not None:
                return cached

        if use_coverage:
            plot_data = coverage_sample_prepared_frame(
                visible,
                int(effective_limit),
                seed=0,
                force_row_ids=[],
                x_bins=int(COVERAGE_SAMPLE_X_BINS),
                y_bins=int(COVERAGE_SAMPLE_Y_BINS),
                coverage_fraction=float(COVERAGE_SAMPLE_FRACTION),
            )
        else:
            plot_data = sample_prepared_frame(
                visible,
                int(effective_limit),
                seed=0,
                force_row_ids=[],
            )

        if base_sample_key is not None:
            self._base_sample_cache_set(base_sample_key, plot_data)

        # Add forced rows afterward, without forcing the sampling function into its
        # slow ID-scanning branch. Forced rows are usually just focus/important rows.
        if forced_ids:
            visible_frame = self._frame_for_cache(visible)

            forced_positions = self._forced_positions_in_frame(
                visible_frame,
                forced_ids,
            )

            if forced_positions:
                extra = visible_frame.take(
                    np.asarray(forced_positions, dtype=np.int64)
                )

                combined = pd.concat(
                    [plot_data.frame, extra],
                    ignore_index=True,
                )

                if INTERNAL_ROW_ID in combined.columns:
                    combined = combined.drop_duplicates(
                        subset=[INTERNAL_ROW_ID],
                        keep="first",
                    )

                plot_data = PreparedFrame(
                    dataset_id=plot_data.dataset_id,
                    frame=combined,
                    x_name=plot_data.x_name,
                    y_name=plot_data.y_name,
                    require_y=plot_data.require_y,
                    row_count_before_filter=plot_data.row_count_before_filter,
                    row_count_after_filter=len(combined),
                    sampled_from=plot_data.sampled_from,
                )

        return plot_data


    def _selection_overlay_limit(self) -> int:
        try:
            return max(
                1,
                min(
                    int(getattr(self.state, "max_selection_ids", SELECTION_OVERLAY_LIMIT_DEFAULT)),
                    SELECTION_OVERLAY_LIMIT_DEFAULT,
                ),
            )
        except Exception:
            return SELECTION_OVERLAY_LIMIT_DEFAULT


    def _active_selection_state(self):
        selection = getattr(self.context, "selection", None)
        if selection is None:
            return None

        try:
            active = selection.get_active_set()
        except Exception:
            return None

        if active is None:
            return None

        if getattr(active, "dataset_id", None) != self._dataset_id():
            return None

        return active


    def _active_selection_metadata(self) -> dict:
        active = self._active_selection_state()
        if active is None:
            return {}

        metadata = getattr(active, "metadata", None)
        if isinstance(metadata, dict):
            return metadata

        return {}


    def _selection_overlay_payload_from_metadata(self) -> dict:
        metadata = self._active_selection_metadata()
        payload = metadata.get(SELECTION_OVERLAY_METADATA_KEY)

        if not isinstance(payload, dict):
            return {}

        # Only draw cached coordinates when they correspond to this panel's axes.
        # If the user changes X/Y, these coordinates are no longer meaningful.
        if str(payload.get("x_variable")) != str(self.state.x):
            return {}

        if str(payload.get("y_variable")) != str(self.state.y):
            return {}

        return payload


    def _selection_overlay_records_from_metadata(self) -> list[dict]:
        payload = self._selection_overlay_payload_from_metadata()
        records = payload.get("points")

        if not isinstance(records, list):
            return []

        cleaned: list[dict] = []

        for record in records:
            if not isinstance(record, dict):
                continue

            try:
                x = float(record[INTERNAL_X])
                y = float(record[INTERNAL_Y])
            except Exception:
                continue

            if not np.isfinite(x) or not np.isfinite(y):
                continue

            row_id = record.get(INTERNAL_ROW_ID, record.get("row_id", ""))

            cleaned.append(
                {
                    INTERNAL_X: x,
                    INTERNAL_Y: y,
                    INTERNAL_ROW_ID: "" if row_id is None else str(row_id),
                }
            )

        return cleaned


    def _selection_overlay_from_records(self, records: list[dict]):
        if not records:
            return None

        frame = pd.DataFrame(records)

        if frame.empty or INTERNAL_X not in frame.columns or INTERNAL_Y not in frame.columns:
            return None

        return hv.Points(
            frame,
            kdims=[INTERNAL_X, INTERNAL_Y],
            vdims=[INTERNAL_ROW_ID] if INTERNAL_ROW_ID in frame.columns else [],
        ).opts(
            marker="circle",
            size=max(float(self.state.point_size) + 5, 9),
            fill_alpha=0.0,
            line_color="#ffd400",
            line_width=2.5,
            tools=[],
            active_tools=[],
            toolbar=None,
            logx=self.state.log_x,
            logy=self.state.log_y,
            shared_axes=False,
            axiswise=True,
            framewise=True,
            **self._current_range_opts(include_y=True),
        )


    def _selection_overlay_records_from_indices(
        self,
        frame: pd.DataFrame,
        indices,
        *,
        limit: Optional[int] = None,
    ) -> list[dict]:
        if frame is None or frame.empty:
            return []

        if INTERNAL_X not in frame.columns or INTERNAL_Y not in frame.columns:
            return []

        limit = int(limit or self._selection_overlay_limit())

        seen: set[str] = set()
        records: list[dict] = []

        for index in list(indices or []):
            if len(records) >= limit:
                break

            try:
                idx = int(index)
            except Exception:
                continue

            if idx < 0 or idx >= len(frame):
                continue

            row = frame.iloc[idx]

            try:
                x = float(row[INTERNAL_X])
                y = float(row[INTERNAL_Y])
            except Exception:
                continue

            if not np.isfinite(x) or not np.isfinite(y):
                continue

            row_id = row.get(INTERNAL_ROW_ID, "")
            row_id = "" if row_id is None else str(row_id)

            if row_id in seen:
                continue

            seen.add(row_id)

            records.append(
                {
                    INTERNAL_ROW_ID: row_id,
                    INTERNAL_X: x,
                    INTERNAL_Y: y,
                }
            )

        return records


    def _selection_overlay_records_from_bounds(
        self,
        data: PreparedFrame,
        bounds,
        *,
        limit: Optional[int] = None,
    ) -> list[dict]:
        if data is None or data.empty or bounds is None or len(bounds) != 4:
            return []

        frame = self._frame_for_cache(data)

        if frame is None or frame.empty:
            return []

        if INTERNAL_X not in frame.columns or INTERNAL_Y not in frame.columns:
            return []

        limit = int(limit or self._selection_overlay_limit())

        try:
            left, bottom, right, top = bounds
            x_min, x_max = min(float(left), float(right)), max(float(left), float(right))
            y_min, y_max = min(float(bottom), float(top)), max(float(bottom), float(top))
        except Exception:
            return []

        try:
            x = frame[INTERNAL_X].to_numpy(copy=False)
            y = frame[INTERNAL_Y].to_numpy(copy=False)

            mask = (
                (x >= x_min)
                & (x <= x_max)
                & (y >= y_min)
                & (y <= y_max)
            )

            sub = frame.loc[mask, [column for column in [INTERNAL_ROW_ID, INTERNAL_X, INTERNAL_Y] if column in frame.columns]]
        except Exception:
            return []

        if sub.empty:
            return []

        if len(sub) > limit:
            try:
                sub = sub.sample(n=limit, random_state=0)
            except Exception:
                sub = sub.head(limit)

        records: list[dict] = []

        for row in sub.itertuples(index=False):
            values = row._asdict()

            try:
                x_value = float(values[INTERNAL_X])
                y_value = float(values[INTERNAL_Y])
            except Exception:
                continue

            if not np.isfinite(x_value) or not np.isfinite(y_value):
                continue

            row_id = values.get(INTERNAL_ROW_ID, "")

            records.append(
                {
                    INTERNAL_ROW_ID: "" if row_id is None else str(row_id),
                    INTERNAL_X: x_value,
                    INTERNAL_Y: y_value,
                }
            )

        return records


    def _selection_points(self, data: PreparedFrame):
        """
        Draw active selection rings.

        For large Parquet/DatasetSource-backed datasets, BaseVisualisationPanel's
        row-id lookup intentionally refuses to scan the full prepared frame. That
        protects memory and time, but it also means selected IDs cannot always be
        resolved back into rows for visual rings.

        Scatter selections therefore store a small coordinate payload in selection
        metadata when the selection is created. Use that first.
        """

        records = self._selection_overlay_records_from_metadata()
        overlay = self._selection_overlay_from_records(records)

        if overlay is not None:
            return overlay

        # Small-dataset or external-selection fallback.
        return super()._selection_points(data)

    def _interactive_dynamic_overlays(self, data: PreparedFrame):

        selection_overlay = None
        focus_overlay = None

        try:
            selection_overlay = self._selection_points(data)
        except Exception as exc:
            print(
                "[AstronomicAL scatter] interactive selection overlay failed "
                f"panel_id={getattr(self, 'panel_id', None)} "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )

        try:
            focus_overlay = self._focus_overlay(
                data,
                size=max(float(self.state.point_size) + 8, 12),
            )
        except Exception as exc:
            print(
                "[AstronomicAL scatter] interactive focus overlay failed "
                f"panel_id={getattr(self, 'panel_id', None)} "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )

        return selection_overlay, focus_overlay

    def _compose_element_layers(self, layers):
        """
        Compose concrete HoloViews Elements.

        This helper is only for already-materialised Elements, not for mixing
        a DynamicMap with static overlays. That mixed case must be handled inside
        the DynamicMap callback.
        """

        layers = [layer for layer in layers if layer is not None]

        if not layers:
            return self._empty("No scatter layers to display")

        composed = layers[0]

        for layer in layers[1:]:
            try:
                composed = composed * layer
            except Exception as exc:
                print(
                    "[AstronomicAL scatter] element layer composition failed; "
                    "dropping optional layer "
                    f"panel_id={getattr(self, 'panel_id', None)} "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )

        return composed

    def _compose_render_layers(self, layers):
        """
        Safely compose scatter base, selection overlay, and focus overlay.

        Avoid hv.Overlay(...).collate(); with DynamicMap/static overlay mixtures
        HoloViews can raise:

            reduce() of empty iterable with no initial value

        The scatter should never fail panel creation because an optional overlay
        composition path failed. Fall back progressively.
        """

        layers = [layer for layer in layers if layer is not None]

        if not layers:
            return self._empty("No scatter layers to display")

        if len(layers) == 1:
            return layers[0]

        try:
            composed = layers[0]
            for layer in layers[1:]:
                composed = composed * layer
            return composed
        except Exception as exc:
            print(
                "[AstronomicAL scatter] layer composition failed; "
                "falling back to base layer only "
                f"panel_id={getattr(self, 'panel_id', None)} "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            return layers[0]


    def _focus_overlay(self, data: PreparedFrame, *, size: float = 12):
        """
        Draw the black focus ring.

        Order of attempts:

        1. Use focus_x/focus_y from the focus metadata when the focus came from
        this same scatter axes.
        2. Use source-backed row lookup for other scatter panels.
        3. Fall back to the base implementation, if available.

        This keeps the black ring independent of whether the focused source is
        present in the reduced interactive sample.
        """

        overlay = self._focus_overlay_from_focus_metadata(size=size)
        if overlay is not None:
            return overlay

        overlay = self._focus_overlay_from_dataset_source(size=size)
        if overlay is not None:
            return overlay

        try:
            return super()._focus_overlay(data, size=size)
        except Exception:
            return None

    def _scatter_render_identity(self, data: PreparedFrame, *, use_raster: bool):
        selection = getattr(self.context, "selection", None)

        focus_id = None
        selection_signature = None

        if selection is not None:
            try:
                focus = selection.get_focus()
            except Exception:
                focus = None

            if focus is not None and getattr(focus, "dataset_id", None) == self._dataset_id():
                focus_id = str(getattr(focus, "row_id", "") or "")

            try:
                active_set = selection.get_active_set()
            except Exception:
                active_set = None

            if active_set is not None and getattr(active_set, "dataset_id", None) == self._dataset_id():
                set_id = (
                    getattr(active_set, "selection_id", None)
                    or getattr(active_set, "id", None)
                    or getattr(active_set, "artifact_id", None)
                    or ""
                )
                row_ids = list(getattr(active_set, "row_ids", []) or [])
                selection_signature = (str(set_id), len(row_ids))

        return (
            getattr(self, "_last_prepared_cache_key", None),
            bool(use_raster),
            str(getattr(self.state, "x", "") or ""),
            str(getattr(self.state, "y", "") or ""),
            str(getattr(self.state, "color_by", "") or ""),
            tuple(getattr(self.state, "label_filter", []) or []),
            bool(getattr(self.state, "log_x", False)),
            bool(getattr(self.state, "log_y", False)),
            int(getattr(self.state, "interactive_sample_limit", 0)),
            float(getattr(self.state, "point_size", 0.0)),
            float(getattr(self.state, "point_alpha", 0.0)),
            focus_id,
            selection_signature,
            self._last_x_range,
            self._last_y_range,
        )


    def _render(self) -> None:
        t0 = time.perf_counter()

        self._clear_stream_watchers()

        data = self._plot_data(require_y=True)

        prepared_key = getattr(self, "_last_prepared_cache_key", None)
        last_key = getattr(self, "_last_interactive_prepared_key", None)

        if last_key != prepared_key:
            self._interactive_sample_cache.clear()
            self._last_interactive_prepared_key = prepared_key
            self._prepared_extent_cache_key = None
            self._prepared_extent_cache = None
            self._near_full_range_keys.clear()

        t1 = time.perf_counter()

        if data.empty:
            empty = self._empty("No finite X/Y data")
            if getattr(self, "_last_scatter_assigned_object", None) is not empty:
                self.plot_pane.object = empty
                self._last_scatter_assigned_object = empty

            self.status_pane.object = "0 plotted rows"

            print(
                "[AstronomicAL scatter] empty render "
                f"prepare={t1 - t0:.2f}s",
                flush=True,
            )
            return

        use_raster = self._should_rasterize(data)
        t2 = time.perf_counter()

        render_identity = self._scatter_render_identity(
            data,
            use_raster=use_raster,
        )

        if (
            getattr(self, "_last_scatter_render_identity", None) == render_identity
            and getattr(self, "_last_scatter_assigned_object", None) is not None
            and self.plot_pane.object is getattr(self, "_last_scatter_assigned_object", None)
        ):
            plotted_count = (
                self._frame_len(data)
                if use_raster
                else min(self._frame_len(data), int(self.state.interactive_sample_limit))
            )

            render_label = "rasterized" if use_raster else "interactive"
            sampled_note = ""
            if not use_raster:
                sampled_note = (
                    f" · coverage-aware sample from {self._frame_len(data):,}"
                    if self._frame_len(data) > int(self.state.interactive_sample_limit)
                    else ""
                )
                if self._use_interactive_density_underlay(data):
                    sampled_note += " · density underlay"

            selection_note = getattr(self, "_selection_overlay_status_note", "") or ""
            status = (
                f"{self._frame_len(data):,} eligible rows · "
                f"{plotted_count:,} shown · {render_label}{sampled_note}"
            )
            if selection_note:
                status += f" · {selection_note}"

            self.status_pane.object = status

            print(
                "[AstronomicAL scatter] skipped duplicate pane assignment "
                f"mode={render_label} "
                f"rows={self._frame_len(data):,} "
                f"prepare={t1 - t0:.2f}s "
                f"total={time.perf_counter() - t0:.2f}s",
                flush=True,
            )
            return

        if use_raster:
            base = self._scatter_rasterized(data)
            render_label = "rasterized"
            plotted_count = self._frame_len(data)
            sampled_note = ""
        else:
            base = self._scatter_interactive_dynamic(data)
            render_label = "interactive"
            plotted_count = min(
                self._frame_len(data),
                int(self.state.interactive_sample_limit),
            )
            sampled_note = (
                f" · coverage-aware sample from {self._frame_len(data):,}"
                if self._frame_len(data) > int(self.state.interactive_sample_limit)
                else ""
            )
            if self._use_interactive_density_underlay(data):
                sampled_note += " · density underlay"

        t3 = time.perf_counter()

        t_sel0 = time.perf_counter()

        selection_overlay = self._selection_points(data)
        t_sel1 = time.perf_counter()

        focus_overlay = self._focus_overlay(
            data,
            size=max(float(self.state.point_size) + 8, 12),
        )
        t_focus1 = time.perf_counter()

        overlay = self._compose_element_layers(
            [
                base,
                selection_overlay,
                focus_overlay,
            ]
        )

        overlay = overlay.opts(
            responsive=True,
            min_height=PLOT_MIN_HEIGHT,
            xlabel=str(self.state.x),
            ylabel=str(self.state.y),
            legend_position="right",
            show_grid=True,
            toolbar="right",
            hooks=[deduplicate_toolbar_tools_hook, keep_pan_tool_active_hook],
            shared_axes=False,
            axiswise=True,
            framewise=True,
        )

        t4 = time.perf_counter()

        print(
            "[AstronomicAL scatter] overlay timing "
            f"selection={t_sel1 - t_sel0:.3f}s "
            f"focus={t_focus1 - t_sel1:.3f}s "
            f"compose={t4 - t_focus1:.3f}s",
            flush=True,
        )

        self.plot_pane.object = overlay
        self._last_scatter_assigned_object = overlay
        self._last_scatter_render_identity = render_identity

        t5 = time.perf_counter()

        selection_note = getattr(self, "_selection_overlay_status_note", "") or ""

        status = (
            f"{self._frame_len(data):,} eligible rows · "
            f"{plotted_count:,} shown · {render_label}{sampled_note}"
        )

        if selection_note:
            status += f" · {selection_note}"

        self.status_pane.object = status

        print(
            "[AstronomicAL scatter] render timing "
            f"mode={render_label} "
            f"rows={self._frame_len(data):,} "
            f"prepare={t1 - t0:.2f}s "
            f"mode_check={t2 - t1:.2f}s "
            f"build_base={t3 - t2:.2f}s "
            f"build_overlay={t4 - t3:.2f}s "
            f"assign_pane={t5 - t4:.2f}s "
            f"total={t5 - t0:.2f}s",
            flush=True,
        )


    def _range_cache_key(self, range_value):
        if not range_value:
            return None

        try:
            lo, hi = range_value
        except Exception:
            return None

        if lo is None or hi is None:
            return None

        try:
            lo = float(lo)
            hi = float(hi)
        except Exception:
            return (str(lo), str(hi))

        if not np.isfinite(lo) or not np.isfinite(hi):
            return None

        span = abs(hi - lo)

        if span <= 0:
            return (round(lo, 8), round(hi, 8))

        # Quantize to about 1e-4 of the current span.
        # This makes near-identical Bokeh range emissions reuse the same cache entry.
        step = span * 1.0e-4

        return (
            round(lo / step) * step,
            round(hi / step) * step,
        )

    def _should_rasterize(self, data: PreparedFrame) -> bool:
        if self.state.render_mode == "datashader":
            return True
        if self.state.render_mode == "interactive":
            return False
        return self._frame_len(data) > int(self.state.datashade_threshold)

    def _forced_row_ids(self) -> List[str]:
        """Rows that should always be included in interactive samples.

        Only force the current focus into the main sampled point cloud. Active
        selections are drawn as a separate overlay using direct row-id lookup.
        """
        selection = getattr(self.context, "selection", None)
        if selection is None:
            return []

        try:
            focus = selection.get_focus()
        except Exception:
            focus = None

        if focus is None or getattr(focus, "dataset_id", None) != self._dataset_id():
            return []

        row_id = getattr(focus, "row_id", None)
        if row_id is None:
            return []

        return [str(row_id)]

    def _effective_interactive_sample_limit(self, visible_count: int) -> int:
        """
        Return the actual number of points to send to Bokeh in interactive mode.

        The full prepared frame is still retained server-side. This only controls
        the frontend Bokeh/HoloViews display size.
        """
        user_limit = int(getattr(self.state, "interactive_sample_limit", 50_000))

        if visible_count >= 500_000:
            return min(user_limit, 10_000)

        if visible_count >= 100_000:
            return min(user_limit, 15_000)

        if visible_count >= 25_000:
            return min(user_limit, 20_000)

        return min(user_limit, visible_count)

    def _empty_density_underlay(self, x_range=None, y_range=None):

        width = int(INTERACTIVE_DENSITY_CANVAS_WIDTH)
        height = int(INTERACTIVE_DENSITY_CANVAS_HEIGHT)

        rgba = np.zeros((height, width, 4), dtype=np.float32)

        try:
            if x_range is not None:
                x0, x1 = x_range
            else:
                x0, x1 = 0.0, 1.0

            if y_range is not None:
                y0, y1 = y_range
            else:
                y0, y1 = 0.0, 1.0

            x0 = float(x0)
            x1 = float(x1)
            y0 = float(y0)
            y1 = float(y1)

            if not np.isfinite(x0) or not np.isfinite(x1) or x1 <= x0:
                x0, x1 = 0.0, 1.0

            if not np.isfinite(y0) or not np.isfinite(y1) or y1 <= y0:
                y0, y1 = 0.0, 1.0
        except Exception:
            x0, x1 = 0.0, 1.0
            y0, y1 = 0.0, 1.0

        return hv.RGB(
            rgba,
            bounds=(x0, y0, x1, y1),
            vdims=["R", "G", "B", "A"],
        ).opts(
            tools=[],
            active_tools=[],
            toolbar=None,
            shared_axes=False,
            axiswise=True,
            framewise=True,
            hooks=[renderer_name_hook(DENSITY_RENDERER)],
        )

    def _density_ranges_for_canvas(self, frame: pd.DataFrame, x_range=None, y_range=None):
        """
        Resolve finite x/y ranges for Datashader Canvas.
        """

        try:
            x = pd.to_numeric(frame[INTERNAL_X], errors="coerce").to_numpy(copy=False)
            y = pd.to_numeric(frame[INTERNAL_Y], errors="coerce").to_numpy(copy=False)
            finite = np.isfinite(x) & np.isfinite(y)
        except Exception:
            return None, None

        if not finite.any():
            return None, None

        def _range_from_stream(range_value, values):
            if range_value is not None:
                try:
                    lo, hi = range_value
                    lo = float(lo)
                    hi = float(hi)
                    if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                        return (lo, hi)
                except Exception:
                    pass

            finite_values = values[np.isfinite(values)]
            if len(finite_values) < 2:
                return None

            lo = float(np.nanmin(finite_values))
            hi = float(np.nanmax(finite_values))

            if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                return None

            return (lo, hi)

        resolved_x = _range_from_stream(x_range, x)
        resolved_y = _range_from_stream(y_range, y)

        return resolved_x, resolved_y


    def _density_frame_for_ranges(self, data: PreparedFrame, x_range=None, y_range=None):

        try:
            if x_range is None and y_range is None:
                return data.frame

            try:
                visible = frame_in_ranges(
                    data,
                    x_range,
                    y_range,
                    log_x=bool(getattr(self.state, "log_x", False)),
                    log_y=bool(getattr(self.state, "log_y", False)),
                )
            except TypeError:
                # Compatibility with older/newer helper signatures.
                visible = frame_in_ranges(
                    data,
                    x_range,
                    y_range,
                )

            return visible.frame
        except Exception as exc:
            print(
                "[AstronomicAL scatter] density range filtering failed "
                f"panel_id={self.panel_id} "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            return data.frame

    def _aggregate_axis_values(self, agg, axis_name: str):
        try:
            coords = getattr(agg, "coords", None)
            if coords is not None and axis_name in coords:
                values = np.asarray(coords[axis_name].values, dtype=float)
                if values.size:
                    return values
        except Exception:
            pass

        return None

    def _density_rgba_from_aggregate(self, agg):

        try:
            counts = np.asarray(agg.values, dtype=np.float64)
        except Exception:
            return None

        if counts.size == 0:
            return None

        counts = np.nan_to_num(counts, nan=0.0, posinf=0.0, neginf=0.0)
        counts[counts < 0] = 0.0

        max_count = float(np.nanmax(counts))

        if not np.isfinite(max_count) or max_count <= 0:
            return None

        # Log compression keeps high-density cores from saturating everything.
        norm = np.log1p(counts) / np.log1p(max_count)
        norm = np.clip(norm, 0.0, 1.0)

        gamma = float(INTERACTIVE_DENSITY_ALPHA_GAMMA)
        if gamma > 0:
            norm = np.power(norm, gamma)

        alpha = norm * float(INTERACTIVE_DENSITY_UNDERLAY_ALPHA)

        r, g, b = INTERACTIVE_DENSITY_GREY_RGB

        rgba = np.zeros((*counts.shape, 4), dtype=np.float32)
        rgba[..., 0] = float(r)
        rgba[..., 1] = float(g)
        rgba[..., 2] = float(b)
        rgba[..., 3] = alpha.astype(np.float32)

        return rgba

    def _density_rgba_element_from_aggregate(
        self,
        agg,
        *,
        x_range,
        y_range,
    ):
        rgba = self._density_rgba_from_aggregate(agg)

        if rgba is None:
            return self._empty_density_underlay(x_range=x_range, y_range=y_range)

        try:
            x0, x1 = x_range
            y0, y1 = y_range

            x_values = self._aggregate_axis_values(agg, INTERNAL_X)
            y_values = self._aggregate_axis_values(agg, INTERNAL_Y)

            print(
                "[AstronomicAL scatter] density aggregate orientation "
                f"panel_id={self.panel_id} "
                f"rgba_shape={getattr(rgba, 'shape', None)} "
                f"x_len={None if x_values is None else len(x_values)} "
                f"y_len={None if y_values is None else len(y_values)} "
                f"x_range={x_range} "
                f"y_range={y_range}",
                flush=True,
            )

            # Datashader aggregates are normally shaped as (y, x). HoloViews RGB
            # is most reliable when given explicit x/y coordinates alongside the
            # RGBA cube.
            if x_values is not None and y_values is not None:
                if rgba.shape[0] == len(y_values) and rgba.shape[1] == len(x_values):
                    rgb = hv.RGB(
                        (
                            x_values,
                            y_values,
                            rgba,
                        ),
                        kdims=[INTERNAL_X, INTERNAL_Y],
                        vdims=["R", "G", "B", "A"],
                    )
                elif rgba.shape[0] == len(x_values) and rgba.shape[1] == len(y_values):
                    # Defensive fallback for backends that have already transposed
                    # the aggregate. This branch should usually not be used, but it
                    # prevents silent x/y reversal.
                    rgb = hv.RGB(
                        (
                            x_values,
                            y_values,
                            np.swapaxes(rgba, 0, 1),
                        ),
                        kdims=[INTERNAL_X, INTERNAL_Y],
                        vdims=["R", "G", "B", "A"],
                    )
                else:
                    rgb = hv.RGB(
                        rgba,
                        bounds=(float(x0), float(y0), float(x1), float(y1)),
                        vdims=["R", "G", "B", "A"],
                    )
            else:
                rgb = hv.RGB(
                    rgba,
                    bounds=(float(x0), float(y0), float(x1), float(y1)),
                    vdims=["R", "G", "B", "A"],
                )

            return rgb.opts(
                tools=[],
                active_tools=[],
                toolbar=None,
                shared_axes=False,
                axiswise=True,
                framewise=True,
                hooks=[renderer_name_hook(DENSITY_RENDERER)],
            )
        except Exception as exc:
            print(
                "[AstronomicAL scatter] failed to build grey-alpha density RGB "
                f"panel_id={self.panel_id} "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            return self._empty_density_underlay(x_range=x_range, y_range=y_range)


    def _density_underlay_element(self, data: PreparedFrame, x_range=None, y_range=None):
        """
        Build a passive density underlay as a concrete hv.Image.

        This intentionally avoids HoloViews rasterize(...), because nested
        DynamicMap/rasterize/Overlay composition was the source of earlier panel
        creation failures.
        """

        if not self._use_interactive_density_underlay(data):
            return self._empty_density_underlay(x_range=x_range, y_range=y_range)

        try:
            frame = self._density_frame_for_ranges(data, x_range, y_range)

            if frame is None or frame.empty:
                return self._empty_density_underlay(x_range=x_range, y_range=y_range)

            if INTERNAL_X not in frame.columns or INTERNAL_Y not in frame.columns:
                return self._empty_density_underlay(x_range=x_range, y_range=y_range)

            frame = frame[[INTERNAL_X, INTERNAL_Y]].copy(deep=False)

            x_range_resolved, y_range_resolved = self._density_ranges_for_canvas(
                frame,
                x_range=x_range,
                y_range=y_range,
            )

            if x_range_resolved is None or y_range_resolved is None:
                return self._empty_density_underlay(x_range=x_range, y_range=y_range)

            canvas = ds.Canvas(
                plot_width=int(INTERACTIVE_DENSITY_CANVAS_WIDTH),
                plot_height=int(INTERACTIVE_DENSITY_CANVAS_HEIGHT),
                x_range=x_range_resolved,
                y_range=y_range_resolved,
            )

            agg = canvas.points(
                frame,
                INTERNAL_X,
                INTERNAL_Y,
                agg=ds.count(),
            )

            density = self._density_rgba_element_from_aggregate(
                agg,
                x_range=x_range_resolved,
                y_range=y_range_resolved,
            )

            return density.opts(
                **self._current_range_opts(include_y=True),
            )

        except Exception as exc:
            print(
                "[AstronomicAL scatter] density underlay failed; using empty underlay "
                f"panel_id={self.panel_id} "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            return self._empty_density_underlay(x_range=x_range, y_range=y_range)


    def _compose_density_and_points_dynamic(self, *, density_dmap, points_dmap):
        """
        Compose passive density underlay with the selectable points DynamicMap.

        Selection streams must stay attached to points_dmap, not to this combined
        overlay.
        """

        try:
            return density_dmap * points_dmap
        except Exception as exc:
            print(
                "[AstronomicAL scatter] density/points composition failed; "
                "falling back to points only "
                f"panel_id={self.panel_id} "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            return points_dmap



    def _scatter_interactive_dynamic(self, data: PreparedFrame):
        range_stream = streams.RangeXY(
            x_range=self._last_x_range,
            y_range=self._last_y_range,
        )

        self._selection_event_seq = getattr(self, "_selection_event_seq", 0)
        self._latest_selection_payload = None

        def _ranges_are_unbounded(x_range, y_range):
            def unbounded(r):
                if not r:
                    return True
                lo, hi = r
                return lo is None or hi is None

            return unbounded(x_range) and unbounded(y_range)

        def make_points(x_range=None, y_range=None):
            x_range, y_range = self._normalise_interactive_ranges(
                data,
                x_range=x_range,
                y_range=y_range,
            )

            t0 = time.perf_counter()

            effective_x_range = x_range or self._last_x_range
            effective_y_range = y_range or self._last_y_range

            # Do not force the focused row into the sampled base point layer.
            # The focus marker is already drawn separately by _focus_overlay().
            #
            # Keeping focus IDs out of this cache key prevents every focus change from
            # invalidating the expensive interactive sample cache.
            forced_ids: tuple[str, ...] = ()

            limit = int(self.state.interactive_sample_limit)

            cache_key = self._interactive_sample_cache_key(
                data,
                effective_x_range,
                effective_y_range,
                forced_ids=forced_ids,
                limit=limit,
            )

            cached = self._interactive_sample_cache_get(cache_key)

            if cached is not None:
                element, status_text = cached
                self.status_pane.object = status_text

                print(
                    "[AstronomicAL scatter] interactive make_points cache hit "
                    f"eligible={len(data.frame):,} "
                    f"x_range={self._range_cache_key(effective_x_range)} "
                    f"y_range={self._range_cache_key(effective_y_range)}",
                    flush=True,
                )

                return element
            else:
                print(
                    "[AstronomicAL scatter] interactive make_points cache miss "
                    f"x_range={self._range_cache_key(effective_x_range)} "
                    f"y_range={self._range_cache_key(effective_y_range)} "
                    f"forced_ids={forced_ids}",
                    flush=True,
                )

            self._remember_ranges(effective_x_range, effective_y_range)

            t1 = time.perf_counter()
            
            if self._is_known_near_full_range(effective_x_range, effective_y_range):
                print(
                    "[AstronomicAL scatter] known near-full range; skipping filter "
                    f"panel_id={self.panel_id}",
                    flush=True,
                )

            # range_is_known_full = (
            #     effective_x_range is None
            #     or effective_y_range is None
            #     or self._range_contains_extent(data, effective_x_range, effective_y_range)
            #     or self._range_is_near_full_extent(data, effective_x_range, effective_y_range)
            #     or self._is_known_near_full_range(effective_x_range, effective_y_range)
            # )

            if effective_x_range is None or effective_y_range is None:
                visible = data
            elif self._allow_one_full_range_skip and self._range_contains_extent(
                data,
                effective_x_range,
                effective_y_range,
            ):
                visible = data
                self._allow_one_full_range_skip = False
            else:
                self._allow_one_full_range_skip = False
                visible = frame_in_ranges(
                    data,
                    effective_x_range,
                    effective_y_range,
                )

                try:
                    if len(visible.frame) == len(data.frame):
                        self._remember_near_full_range(effective_x_range, effective_y_range)
                except Exception:
                    pass

            data_len = self._frame_len(data)
            visible_len = self._frame_len(visible)

            if visible is not data:
                visible_len = self._frame_len(visible)
                data_len = self._frame_len(data)

                if visible_len == data_len and not self._range_is_full_extent(
                    data,
                    effective_x_range,
                    effective_y_range,
                ):
                    print(
                        "[AstronomicAL scatter] WARNING range filter returned all rows "
                        "despite non-full extent "
                        f"panel_id={self.panel_id} "
                        f"x_range={effective_x_range!r} "
                        f"y_range={effective_y_range!r}",
                        flush=True,
                    )

            if visible is not data and data_len > 0:
                visible_fraction = visible_len / data_len

                if visible_fraction >= 0.85:
                    self._remember_near_full_range(
                        effective_x_range,
                        effective_y_range,
                    )
                    visible = data
                    visible_len = data_len

            t2 = time.perf_counter()

            effective_limit = self._effective_interactive_sample_limit(visible_len)

            plot_data = self._sample_visible_frame(
                data,
                visible,
                effective_limit,
                forced_ids,
            )

            plot_len = self._frame_len(plot_data)

            t3 = time.perf_counter()

            self._interactive_current_frame = self._frame_for_cache(plot_data)

            sample_strategy = self._coverage_sample_note(visible_len, effective_limit)

            sampled_note = (
                f" · {sample_strategy} from {plot_data.sampled_from:,} visible"
                if hasattr(plot_data, "sampled_from") and plot_data.sampled_from
                else ""
            )

            limit_note = (
                f" · display cap {effective_limit:,}"
                if visible_len > effective_limit
                else ""
            )

            density_note = (
                " · density underlay"
                if self._use_interactive_density_underlay(data)
                else ""
            )

            status_text = (
                f"{data_len:,} eligible rows · "
                f"{visible_len:,} visible · "
                f"{plot_len:,} shown · interactive"
                f"{sampled_note}{limit_note}{density_note}"
            )

            self.status_pane.object = status_text

            # points = self._scatter_points_element(plot_data)

            # selection_overlay, focus_overlay = self._interactive_dynamic_overlays(data)

            # element = self._compose_element_layers(
            #     [
            #         points,
            #         selection_overlay,
            #         focus_overlay,
            #     ]
            # )

            element = self._scatter_points_element(plot_data)

            t4 = time.perf_counter()

            self._interactive_sample_cache_set(
                cache_key,
                element,
                status_text,
            )

            print(
                "[AstronomicAL scatter] interactive make_points "
                f"eligible={self._frame_len(data):,} "
                f"visible={self._frame_len(visible):,} "
                f"shown={self._frame_len(plot_data):,} "
                f"ranges={t1 - t0:.3f}s "
                f"filter={t2 - t1:.3f}s "
                f"sample={t3 - t2:.3f}s "
                f"element={t4 - t3:.3f}s "
                f"total={t4 - t0:.3f}s",
                flush=True,
            )

            return element

        points_dmap = hv.DynamicMap(make_points, streams=[range_stream])

        def make_density(x_range=None, y_range=None):
            x_range, y_range = self._normalise_interactive_ranges(
                data,
                x_range=x_range,
                y_range=y_range,
            )
            return self._density_underlay_element(
                data,
                x_range=x_range,
                y_range=y_range,
            )

        density_dmap = hv.DynamicMap(make_density, streams=[range_stream])

        selection_stream = streams.Selection1D(source=points_dmap)
        bounds_stream = streams.BoundsXY(source=points_dmap)

        last_bounds = {
            "value": None,
            "time": 0.0,
        }

        def on_bounds(event):
            last_bounds["value"] = event.new
            last_bounds["time"] = time.monotonic()

        def rendered_count_in_bounds(frame: pd.DataFrame, bounds) -> int:
            if bounds is None or len(bounds) != 4 or frame.empty:
                return 0

            left, bottom, right, top = bounds
            x_min, x_max = min(left, right), max(left, right)
            y_min, y_max = min(bottom, top), max(bottom, top)

            x = frame[INTERNAL_X].to_numpy(copy=False)
            y = frame[INTERNAL_Y].to_numpy(copy=False)

            return int(
                (
                    (x >= x_min)
                    & (x <= x_max)
                    & (y >= y_min)
                    & (y <= y_max)
                ).sum()
            )

        def publish_latest_selection(seq: int):
            if seq != getattr(self, "_selection_event_seq", None):
                return

            payload = getattr(self, "_latest_selection_payload", None)
            if not payload:
                return

            now = time.monotonic()

            # Wait until lasso/box events have gone quiet. Lasso emits many
            # partial Selection1D updates while drawing; publishing any partial
            # update causes the tiny-subset bug.
            last_event_time = float(payload.get("time", 0.0))
            quiet_for = now - last_event_time
            quiet_required = .9

            if quiet_for < quiet_required:
                self._schedule_selection_publish(
                    seq,
                    int((quiet_required - quiet_for) * 1000) + 50,
                )
                return

            if now < getattr(self, "_ignore_selection_events_until", 0.0):
                return

            indices = list(payload.get("indices") or [])
            if not indices:
                return

            frame = getattr(self, "_interactive_current_frame", pd.DataFrame())
            if frame.empty or INTERNAL_ROW_ID not in frame.columns:
                return

            bounds = payload.get("bounds")

            # Box-select path. If the selected indices are effectively the same
            # as the rendered points inside the rectangle, treat it as a box and
            # compute exact row IDs from the full prepared data.
            if bounds is not None:
                box_rendered_count = rendered_count_in_bounds(frame, bounds)
                selected_count = len(indices)

                tolerance = max(3, int(0.02 * max(box_rendered_count, selected_count, 1)))
                is_box_like = abs(box_rendered_count - selected_count) <= tolerance

                if is_box_like:
                    row_ids, total, truncated = row_ids_in_bounds(
                        data,
                        bounds,
                        max_ids=int(self.state.max_selection_ids),
                    )

                    if row_ids:
                        overlay_points = self._selection_overlay_records_from_bounds(
                            data,
                            bounds,
                        )

                        self._publish_selection(
                            row_ids,
                            bounds=bounds,
                            total_matches=total,
                            truncated=truncated,
                            overlay_points=overlay_points,
                        )
                        return

            # Lasso path. This uses the final stable selected indices from the
            # current rendered frame. When zoomed in under the sample limit, all
            # visible points are rendered, so this is accurate.
            ids = frame[INTERNAL_ROW_ID].astype(str).to_numpy(copy=False)

            selected_ids = [
                str(ids[int(index)])
                for index in indices
                if 0 <= int(index) < len(ids)
            ]

            if not selected_ids:
                return

            seen = set()
            deduped = []
            for row_id in selected_ids:
                if row_id in seen:
                    continue
                seen.add(row_id)
                deduped.append(row_id)

            if self._is_single_focus_selection(
                deduped,
                bounds,
                total_matches=len(deduped),
                truncated=False,
            ):
                metadata = {}

                if indices:
                    try:
                        metadata = self._focus_metadata_from_rendered_row(
                            frame,
                            int(indices[0]),
                        )
                    except Exception:
                        metadata = {}

                self._publish_focus(
                    deduped[0],
                    metadata=metadata,
                )
                return

            overlay_points = self._selection_overlay_records_from_indices(
                frame,
                indices,
            )

            self._publish_selection(
                deduped,
                bounds=None,
                total_matches=len(deduped),
                truncated=False,
                overlay_points=overlay_points,
            )

        self._publish_latest_selection_callback = publish_latest_selection

        def on_select(event):
            if time.monotonic() < getattr(self, "_ignore_selection_events_until", 0.0):
                return

            indices = list(event.new or [])
            if not indices:
                return

            bounds = None
            if last_bounds.get("value") is not None:
                bounds_age = time.monotonic() - float(last_bounds.get("time", 0.0))
                if bounds_age < 0.25:
                    bounds = last_bounds.get("value")

            self._selection_event_seq = getattr(self, "_selection_event_seq", 0) + 1
            seq = self._selection_event_seq

            self._latest_selection_payload = {
                "indices": indices,
                "bounds": bounds,
                "time": time.monotonic(),
            }

            self._schedule_selection_publish(seq, 450)

        self._watch_param(bounds_stream, on_bounds, "bounds", render_scoped=True)
        self._watch_param(selection_stream, on_select, "index", render_scoped=True)

        if self._use_interactive_density_underlay(data):
            print(
                "[AstronomicAL scatter] enabling interactive density underlay "
                f"panel_id={self.panel_id} "
                f"rows={self._frame_len(data):,}",
                flush=True,
            )

            return self._compose_density_and_points_dynamic(
                density_dmap=density_dmap,
                points_dmap=points_dmap,
            )

        return points_dmap




    def _scatter_points_element(self, data: PreparedFrame):
        frame = self._bokeh_safe_frame(data.frame)

        vdims = [HOVER_ROW_ID, HOVER_LABEL, INTERNAL_ROW_ID]

        if INTERNAL_LABEL_DISPLAY in frame.columns:
            vdims.append(INTERNAL_LABEL_DISPLAY)

        if frame.empty:
            empty_frame = pd.DataFrame(
                {
                    INTERNAL_X: pd.Series(dtype="float64"),
                    INTERNAL_Y: pd.Series(dtype="float64"),
                    HOVER_ROW_ID: pd.Series(dtype="object"),
                    HOVER_LABEL: pd.Series(dtype="object"),
                    INTERNAL_ROW_ID: pd.Series(dtype="object"),
                }
            )

            if INTERNAL_LABEL_DISPLAY in vdims:
                empty_frame[INTERNAL_LABEL_DISPLAY] = pd.Series(dtype="object")

            points = hv.Points(
                empty_frame,
                kdims=[INTERNAL_X, INTERNAL_Y],
                vdims=vdims,
            )

            opts = dict(
                **self._base_opts(
                    xlabel=self.state.x,
                    ylabel=self.state.y,
                    tools=[],
                    active_tools=[],
                ),
                color="#1f77b4",
                size=self.state.point_size,
                alpha=0.0,
                line_alpha=0.0,
            )

            opts["hooks"] = list(opts.get("hooks", [])) + [
                renderer_name_hook(SCATTER_RENDERER),
            ]

            return points.opts(**opts)

        points = hv.Points(
            frame,
            kdims=[INTERNAL_X, INTERNAL_Y],
            vdims=vdims,
        )

        hooks = [
            renderer_name_hook(SCATTER_RENDERER),
        ]

        opts = dict(
            size=self.state.point_size,
            alpha=self.state.point_alpha,
            line_alpha=0,
            selection_alpha=1.0,
            selection_color="orange",
            selection_line_color="black",
            nonselection_alpha=0.18,
            muted_alpha=0.03,
            **self._base_opts(
                xlabel=self.state.x,
                ylabel=self.state.y,
                tools=[
                    "tap",
                    "box_select",
                    "lasso_select",
                    limited_point_hover_tool(),
                    "pan",
                    "wheel_zoom",
                    "box_zoom",
                    "reset",
                ],
                active_tools=["wheel_zoom"],
            ),
        )

        opts["hooks"] = list(opts.get("hooks", [])) + hooks

        if (
            self.state.color_by == "Labels"
            and INTERNAL_LABEL_DISPLAY in frame.columns
            and frame[INTERNAL_LABEL_DISPLAY].nunique(dropna=True) <= 40
        ):
            colour_key = _colour_key_from_frame(frame)
            if colour_key:
                opts["color"] = INTERNAL_LABEL_DISPLAY
                opts["cmap"] = colour_key
                opts["legend_position"] = "right"
            else:
                opts["color"] = "#1f77b4"
        else:
            opts["color"] = "#1f77b4"

        return points.opts(**opts)

    def _scatter_rasterized(self, data: PreparedFrame):
        frame = self._bokeh_safe_frame(data.frame[[INTERNAL_X, INTERNAL_Y]])

        points = hv.Points(
            frame,
            kdims=[INTERNAL_X, INTERNAL_Y],
        )

        range_opts = self._current_range_opts(include_y=True)

        raster = rasterize(
            points,
            aggregator=ds.count(),
            pixel_ratio=2,
        ).opts(
            cmap=VISIBLE_DENSITY_CMAP,
            colorbar=True,
            cnorm="eq_hist",
            clipping_colors={"NaN": "white"},
            bgcolor="white",
            responsive=True,
            min_height=PLOT_MIN_HEIGHT,
            xlabel=str(self.state.x),
            ylabel=str(self.state.y),
            logx=self.state.log_x,
            logy=self.state.log_y,

            # Raster is passive. Bounds source owns tools.
            tools=[],
            active_tools=[],
            toolbar=None,

            hooks=[renderer_name_hook(DENSITY_RENDERER)],
            show_grid=True,
            shared_axes=False,
            axiswise=True,
            framewise=True,
            **range_opts,
        )

        bounds_source = self._raster_bounds_source(data)
        bounds_stream = streams.BoundsXY(source=bounds_source)

        def on_bounds(event):
            bounds = event.new
            if not bounds:
                return

            row_ids, total, truncated = row_ids_in_bounds(
                data,
                bounds,
                max_ids=int(self.state.max_selection_ids),
            )

            overlay_points = self._selection_overlay_records_from_bounds(
                data,
                bounds,
            )

            self._publish_selection(
                row_ids,
                bounds=bounds,
                total_matches=total,
                truncated=truncated,
                overlay_points=overlay_points,
            )

        self._watch_param(bounds_stream, on_bounds, "bounds", render_scoped=True)

        return raster * bounds_source

    def _raster_bounds_source(self, data: PreparedFrame):

        frame = self._bokeh_safe_frame(data.frame)

        try:
            x_min = float(np.nanmin(frame[INTERNAL_X].to_numpy(copy=False)))
            x_max = float(np.nanmax(frame[INTERNAL_X].to_numpy(copy=False)))
            y_min = float(np.nanmin(frame[INTERNAL_Y].to_numpy(copy=False)))
            y_max = float(np.nanmax(frame[INTERNAL_Y].to_numpy(copy=False)))
        except Exception:
            x_min, x_max, y_min, y_max = 0.0, 1.0, 0.0, 1.0

        source = hv.Points(
            pd.DataFrame(
                {
                    INTERNAL_X: [x_min, x_max],
                    INTERNAL_Y: [y_min, y_max],
                }
            ),
            kdims=[INTERNAL_X, INTERNAL_Y],
        ).opts(
            size=0,
            alpha=0.0,
            line_alpha=0.0,
            tools=["box_select", "pan", "wheel_zoom", "box_zoom", "reset"],
            active_tools=["wheel_zoom"],
            hooks=[force_wheel_zoom_hook],
            shared_axes=False,
            axiswise=True,
            framewise=True,
        )

        return source

    def _schedule_selection_publish(self, seq: int, delay_ms: int) -> None:
        """Publish a debounced Selection1D event.

        Lasso sends several partial updates. Each update increments
        _selection_event_seq. Only the latest sequence is allowed to publish.
        """

        def _run():
            if getattr(self, "_disposed", False):
                return

            if seq != getattr(self, "_selection_event_seq", None):
                return

            callback = getattr(self, "_publish_latest_selection_callback", None)
            if callback is None:
                return

            callback(seq)

        try:
            doc = pn.state.curdoc
            if doc is not None:
                doc.add_timeout_callback(_run, int(delay_ms))
            else:
                _run()
        except Exception:
            _run()

    def _schedule_post_selection_refresh(self, delay_ms: int = 300) -> None:

        self._ignore_selection_events_until = time.monotonic() + 0.90

        def _run():
            if getattr(self, "_disposed", False):
                return

            # Suppress Selection1D replay only during this redraw window.
            self._ignore_selection_events_until = time.monotonic() + 0.75

            try:
                self.refresh()
            finally:
                # Do not keep a long-lived suppression timer. This avoids
                # selection state feeling like it expires.
                self._ignore_selection_events_until = 0.0

        try:
            doc = pn.state.curdoc
            if doc is not None:
                doc.add_timeout_callback(_run, int(delay_ms))
            else:
                _run()
        except Exception:
            _run()

    def _active_focus_state(self):
        selection = getattr(self.context, "selection", None)
        if selection is None:
            return None

        try:
            focus = selection.get_focus()
        except Exception:
            return None

        if focus is None:
            return None

        try:
            if getattr(focus, "dataset_id", None) != self._dataset_id():
                return None
        except Exception:
            return None

        return focus


    def _focus_overlay_from_xy(
        self,
        *,
        row_id,
        x_value,
        y_value,
        size: float,
    ):
        try:
            x = float(x_value)
            y = float(y_value)
        except Exception:
            return None

        if not np.isfinite(x) or not np.isfinite(y):
            return None

        if bool(getattr(self.state, "log_x", False)) and x <= 0:
            return None

        if bool(getattr(self.state, "log_y", False)) and y <= 0:
            return None

        frame = pd.DataFrame(
            [
                {
                    INTERNAL_ROW_ID: "" if row_id is None else str(row_id),
                    INTERNAL_X: x,
                    INTERNAL_Y: y,
                }
            ]
        )

        return hv.Points(
            frame,
            kdims=[INTERNAL_X, INTERNAL_Y],
            vdims=[INTERNAL_ROW_ID],
        ).opts(
            marker="circle",
            size=size,
            fill_alpha=0.0,
            line_color="black",
            line_width=3.0,
            tools=[],
            active_tools=[],
            toolbar=None,
            logx=self.state.log_x,
            logy=self.state.log_y,
            shared_axes=False,
            axiswise=True,
            framewise=True,
            **self._current_range_opts(include_y=True),
        )


    def _focus_overlay_from_focus_metadata(self, *, size: float):
        """
        Fast path for the scatter panel that published the focus.

        _publish_focus(...) already stores focus_x/focus_y in the focus metadata.
        If those coordinates belong to this panel's current x/y variables, draw the
        black focus ring directly from that metadata instead of waiting for an
        async/source lookup.
        """

        focus = self._active_focus_state()
        if focus is None:
            return None

        metadata = getattr(focus, "metadata", None)
        if not isinstance(metadata, dict):
            return None

        if str(metadata.get("x_variable")) != str(self.state.x):
            return None

        if str(metadata.get("y_variable")) != str(self.state.y):
            return None

        if "focus_x" not in metadata or "focus_y" not in metadata:
            return None

        return self._focus_overlay_from_xy(
            row_id=getattr(focus, "row_id", None),
            x_value=metadata.get("focus_x"),
            y_value=metadata.get("focus_y"),
            size=size,
        )


    def _focus_overlay_from_dataset_source(self, *, size: float):
        """
        Source-backed fallback for other scatter panels.

        The tapped panel can use focus metadata. Other scatter panels have different
        x/y axes, so they need to resolve the focused row ID into their own current
        axes. Reuse the source-backed selected-row lookup added for cross-scatter
        selection rings.
        """

        focus = self._active_focus_state()
        if focus is None:
            return None

        row_id = getattr(focus, "row_id", None)
        if row_id is None:
            return None

        lookup = getattr(self, "_selection_rows_from_dataset_source", None)
        if not callable(lookup):
            return None

        try:
            frame = lookup([str(row_id)])
        except Exception:
            return None

        if frame is None or frame.empty:
            return None

        if INTERNAL_X not in frame.columns or INTERNAL_Y not in frame.columns:
            return None

        try:
            row = frame.iloc[0]
        except Exception:
            return None

        return self._focus_overlay_from_xy(
            row_id=row.get(INTERNAL_ROW_ID, row_id),
            x_value=row.get(INTERNAL_X),
            y_value=row.get(INTERNAL_Y),
            size=size,
        )

    def _publish_focus(
        self,
        row_id: str,
        *,
        origin: str = "core.visualisation.scatter.tap",
        metadata: Optional[dict] = None,
    ) -> None:
        dataset_id = self._dataset_id()
        selection = getattr(self.context, "selection", None)

        if not row_id or not dataset_id or selection is None:
            return
        
        print("[Scatter] publishing focus", row_id, metadata, flush=True)

        selection.set_focus(
            dataset_id=dataset_id,
            row_id=str(row_id),
            origin=origin,
            panel_id=self.panel_id,
            metadata=metadata or {},
        )
        # Do not schedule a full refresh on the origin panel for a focus event.
        # The focus event payload already contains the clicked coordinates, and other
        # panels will receive the normal selection.focus.changed event.
        #
        # Scheduling the origin panel here causes a full HoloViews reassignment even
        # though only the focus marker changed.
        print(
            "[AstronomicAL scatter] skipping origin focus full refresh "
            f"panel_id={self.panel_id}",
            flush=True,
        )
        self._schedule_post_selection_refresh(delay_ms=120)

    @staticmethod
    def _is_single_focus_selection(
        row_ids: List[str],
        bounds: Optional[Sequence[float]],
        *,
        total_matches: Optional[int],
        truncated: bool,
    ) -> bool:
        """
        A single point selected without geometry should be treated as focus,
        not as a one-row multi-selection set.

        Tap events reach this path through Selection1D with:
        - one selected row id
        - no BoundsXY geometry
        - total_matches often equal to 1
        """
        if len(row_ids) != 1:
            return False

        if bounds is not None:
            return False

        if truncated:
            return False

        if total_matches is None:
            return True

        try:
            return int(total_matches) == 1
        except Exception:
            return False

    def _focus_metadata_from_rendered_row(
        self,
        frame: pd.DataFrame,
        rendered_index: int,
    ) -> dict:
        metadata = {
            "panel_type": "scatter",
            "x_variable": str(self.state.x),
            "y_variable": str(self.state.y),
            "id_column": self.state.record_id_col or "Use Index",
        }

        try:
            idx = int(rendered_index)
        except Exception:
            return metadata

        if idx < 0 or idx >= len(frame):
            return metadata

        row = frame.iloc[idx]

        try:
            metadata["focus_x"] = float(row[INTERNAL_X])
        except Exception:
            pass

        try:
            metadata["focus_y"] = float(row[INTERNAL_Y])
        except Exception:
            pass

        return metadata

    def _publish_selection(
        self,
        row_ids: List[str],
        bounds: Optional[Sequence[float]] = None,
        *,
        total_matches: Optional[int] = None,
        truncated: bool = False,
        overlay_points: Optional[list[dict]] = None,
    ) -> None:
        dataset_id = self._dataset_id()
        selection = getattr(self.context, "selection", None)

        if not row_ids or not dataset_id or selection is None:
            return

        row_ids = [str(row_id) for row_id in row_ids if row_id is not None]

        if not row_ids:
            return

        if self._is_single_focus_selection(
            row_ids,
            bounds,
            total_matches=total_matches,
            truncated=truncated,
        ):
            self._publish_focus(row_ids[0])
            return

        metadata = {
            "panel_type": "scatter",
            "x_variable": str(self.state.x),
            "y_variable": str(self.state.y),
        }

        if bounds and len(bounds) == 4:
            left, bottom, right, top = bounds
            metadata["geometry"] = {
                "kind": "box",
                "x_variable": str(self.state.x),
                "y_variable": str(self.state.y),
                "bounds": [left, right, bottom, top],
            }

        if total_matches is not None:
            metadata["total_matches"] = int(total_matches)

        metadata["published_ids"] = int(len(row_ids))
        metadata["truncated"] = bool(truncated)

        if truncated:
            metadata["truncation_reason"] = "max_selection_ids"

        overlay_points = list(overlay_points or [])

        if overlay_points:
            metadata[SELECTION_OVERLAY_METADATA_KEY] = {
                "panel_type": "scatter",
                "x_variable": str(self.state.x),
                "y_variable": str(self.state.y),
                "points": overlay_points[: self._selection_overlay_limit()],
                "points_count": min(len(overlay_points), self._selection_overlay_limit()),
                "total_matches": int(total_matches) if total_matches is not None else len(overlay_points),
                "truncated": bool(truncated) or len(overlay_points) > self._selection_overlay_limit(),
            }

        selection.set_selection_set(
            dataset_id=dataset_id,
            row_ids=row_ids,
            origin="core.visualisation.scatter.selection",
            panel_id=self.panel_id,
            mode="replace",
            metadata=metadata,
            create_artifact=True,
            update_focus_policy="preserve_or_first",
        )

        self._schedule_post_selection_refresh()

def _colour_key_from_frame(frame: pd.DataFrame) -> dict:
    if INTERNAL_LABEL_DISPLAY not in frame.columns:
        return {}

    if "__label_colour__" not in frame.columns:
        return {}

    pairs = (
        frame[[INTERNAL_LABEL_DISPLAY, "__label_colour__"]]
        .dropna()
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )

    return {str(label): str(colour) for label, colour in pairs}