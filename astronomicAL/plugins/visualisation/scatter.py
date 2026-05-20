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
)


class ScatterPanel(BaseVisualisationPanel):
    title = "Scatter Plot"

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

    def _row_id_lookup_for_frame(self, data):
        """
        Build a raw row-id -> integer position lookup for the current prepared frame.

        This is only used when forced IDs need to be included in an interactive
        sample.
        """
        if not hasattr(self, "_row_id_lookup_cache"):
            self._row_id_lookup_cache = {}

        cache_key = (id(data.frame), len(data.frame))

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

        if not hasattr(self, "_interactive_sample_cache"):
            self._interactive_sample_cache = {}

        forced_ids = list(forced_ids or [])
        forced_key = tuple(str(value) for value in forced_ids)

        is_full = visible is data

        cache_key = None
        if is_full:
            cache_key = (
                id(data.frame),
                len(data.frame),
                int(effective_limit),
                forced_key,
            )

            cached = self._interactive_sample_cache.get(cache_key)
            if cached is not None:
                return cached

        # Always use the fast no-forced-ID sampling path first.
        plot_data = sample_prepared_frame(
            visible,
            int(effective_limit),
            seed=0,
            force_row_ids=[],
        )

        # Add forced rows afterward, without forcing sample_prepared_frame into its
        # slow ID-scanning branch.
        if forced_ids:
            forced_positions = self._forced_positions_in_frame(
                visible.frame,
                forced_ids,
            )

            if forced_positions:
                extra = visible.frame.take(
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

        if cache_key is not None:
            self._interactive_sample_cache[cache_key] = plot_data

            if len(self._interactive_sample_cache) > 12:
                try:
                    first_key = next(iter(self._interactive_sample_cache))
                    self._interactive_sample_cache.pop(first_key, None)
                except Exception:
                    self._interactive_sample_cache.clear()

        return plot_data


    def _render(self) -> None:
        t0 = time.perf_counter()

        self._clear_stream_watchers()

        data = self._plot_data(require_y=True)

        t1 = time.perf_counter()

        if data.empty:
            self.plot_pane.object = self._empty("No finite X/Y data")
            self.status_pane.object = "0 plotted rows"
            print(
                "[AstronomicAL scatter] empty render "
                f"prepare={t1 - t0:.2f}s",
                flush=True,
            )
            return

        use_raster = self._should_rasterize(data)

        t2 = time.perf_counter()

        if use_raster:
            base = self._scatter_rasterized(data)
            render_label = "rasterized"
            plotted_count = len(data.frame)
            sampled_note = ""
        else:
            base = self._scatter_interactive_dynamic(data)
            render_label = "interactive"
            plotted_count = min(len(data.frame), int(self.state.interactive_sample_limit))
            sampled_note = (
                f" · range-aware sample from {len(data.frame):,}"
                if len(data.frame) > int(self.state.interactive_sample_limit)
                else ""
            )

        t3 = time.perf_counter()

        overlays = [
            base,
            self._selection_points(data),
            self._focus_overlay(data, size=max(float(self.state.point_size) + 8, 12)),
        ]

        overlay = hv.Overlay([item for item in overlays if item is not None]).collate().opts(
            responsive=True,
            min_height=PLOT_MIN_HEIGHT,
            xlabel=str(self.state.x),
            ylabel=str(self.state.y),
            legend_position="right",
            show_grid=True,
            toolbar="right",
            tools=["box_select", "pan", "wheel_zoom", "box_zoom", "reset"],
            active_tools=["wheel_zoom"],
            hooks=[force_wheel_zoom_hook],
            shared_axes=False,
            axiswise=True,
            framewise=True,
        )

        t4 = time.perf_counter()

        self.plot_pane.object = overlay

        t5 = time.perf_counter()

        self.status_pane.object = (
            f"{len(data.frame):,} eligible rows · "
            f"{plotted_count:,} shown · {render_label}{sampled_note}"
        )

        print(
            "[AstronomicAL scatter] render timing "
            f"mode={render_label} "
            f"rows={len(data.frame):,} "
            f"prepare={t1 - t0:.2f}s "
            f"mode_check={t2 - t1:.2f}s "
            f"build_base={t3 - t2:.2f}s "
            f"build_overlay={t4 - t3:.2f}s "
            f"assign_pane={t5 - t4:.2f}s "
            f"total={t5 - t0:.2f}s",
            flush=True,
        )

    def _range_cache_key(self, value):
        if value is None:
            return None

        try:
            if len(value) != 2:
                return None

            result = []
            for item in value:
                if item is None:
                    result.append(None)
                else:
                    # Round to avoid tiny floating-point range changes causing
                    # unnecessary cache misses.
                    result.append(round(float(item), 8))

            return tuple(result)
        except Exception:
            return None

    def _should_rasterize(self, data: PreparedFrame) -> bool:
        if self.state.render_mode == "datashader":
            return True
        if self.state.render_mode == "interactive":
            return False
        return len(data.frame) > int(self.state.datashade_threshold)

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

    def _range_is_full_extent(self, data, x_range, y_range) -> bool:
        if x_range is None and y_range is None:
            return True

        frame = data.frame

        try:
            x_min = float(frame[INTERNAL_X].min())
            x_max = float(frame[INTERNAL_X].max())
        except Exception:
            x_min = x_max = None

        try:
            y_min = float(frame[INTERNAL_Y].min())
            y_max = float(frame[INTERNAL_Y].max())
        except Exception:
            y_min = y_max = None

        def covers(bounds, data_min, data_max):
            if bounds is None:
                return True

            if data_min is None or data_max is None:
                return False

            try:
                low, high = bounds
                low = float(low)
                high = float(high)
            except Exception:
                return False

            span = max(abs(data_max - data_min), 1e-12)
            tolerance = span * 0.01

            return low <= data_min + tolerance and high >= data_max - tolerance

        return covers(x_range, x_min, x_max) and covers(y_range, y_min, y_max)

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

    def _scatter_interactive_dynamic(self, data: PreparedFrame):
        range_stream = streams.RangeXY(
            x_range=self._last_x_range,
            y_range=self._last_y_range,
        )

        self._selection_event_seq = getattr(self, "_selection_event_seq", 0)
        self._latest_selection_payload = None

        point_cache = {}

        def make_points(x_range=None, y_range=None):
            t0 = time.perf_counter()

            effective_x_range = x_range or self._last_x_range
            effective_y_range = y_range or self._last_y_range

            x_key = self._range_cache_key(effective_x_range)
            y_key = self._range_cache_key(effective_y_range)

            forced_ids = tuple(self._forced_row_ids())
            limit = int(self.state.interactive_sample_limit)

            cache_key = (
                id(data.frame),
                len(data.frame),
                x_key,
                y_key,
                limit,
                forced_ids,
            )

            cached = point_cache.get(cache_key)
            if cached is not None:
                element, status_text = cached
                self.status_pane.object = status_text

                print(
                    "[AstronomicAL scatter] interactive make_points cache hit "
                    f"eligible={len(data.frame):,} "
                    f"x_range={x_key} y_range={y_key}",
                    flush=True,
                )

                return element

            self._remember_ranges(effective_x_range, effective_y_range)

            t1 = time.perf_counter()

            if self._range_is_full_extent(data, effective_x_range, effective_y_range):
                visible = data
            else:
                visible = frame_in_ranges(
                    data,
                    effective_x_range,
                    effective_y_range,
                )

            # If the Bokeh-reported range excludes only a tiny edge fraction, treat it as
            # full extent. This avoids repeated almost-full filtering/sampling after pane
            # assignment.
            if visible is not data and len(data.frame) > 0:
                visible_fraction = len(visible.frame) / len(data.frame)

                if visible_fraction >= 0.995:
                    visible = data

            t2 = time.perf_counter()

            effective_limit = self._effective_interactive_sample_limit(len(visible.frame))

            forced_ids = self._forced_row_ids()

            plot_data = self._sample_visible_frame(
                data,
                visible,
                effective_limit,
                forced_ids,
            )

            t3 = time.perf_counter()

            self._interactive_current_frame = plot_data.frame

            sampled_note = (
                f" · sampled from {plot_data.sampled_from:,} visible"
                if plot_data.sampled_from
                else ""
            )

            limit_note = (
                f" · display cap {effective_limit:,}"
                if len(visible.frame) > effective_limit
                else ""
            )

            status_text = (
                f"{len(data.frame):,} eligible rows · "
                f"{len(visible.frame):,} visible · "
                f"{len(plot_data.frame):,} shown · interactive"
                f"{sampled_note}{limit_note}"
            )

            self.status_pane.object = status_text

            element = self._scatter_points_element(plot_data)

            t4 = time.perf_counter()

            point_cache[cache_key] = (element, status_text)

            # Keep this small. Range interactions can generate many slightly different
            # ranges.
            if len(point_cache) > 20:
                try:
                    first_key = next(iter(point_cache))
                    point_cache.pop(first_key, None)
                except Exception:
                    point_cache.clear()

            print(
                "[AstronomicAL scatter] interactive make_points "
                f"eligible={len(data.frame):,} "
                f"visible={len(visible.frame):,} "
                f"shown={len(plot_data.frame):,} "
                f"ranges={t1 - t0:.3f}s "
                f"filter={t2 - t1:.3f}s "
                f"sample={t3 - t2:.3f}s "
                f"element={t4 - t3:.3f}s "
                f"total={t4 - t0:.3f}s",
                flush=True,
            )

            return element

        dmap = hv.DynamicMap(make_points, streams=[range_stream])

        selection_stream = streams.Selection1D(source=dmap)
        bounds_stream = streams.BoundsXY(source=dmap)

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
                        self._publish_selection(
                            row_ids,
                            bounds=bounds,
                            total_matches=total,
                            truncated=truncated,
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

            self._publish_selection(
                deduped,
                bounds=None,
                total_matches=len(deduped),
                truncated=False,
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

        return dmap

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
            tools=["box_select", "pan", "wheel_zoom", "box_zoom", "reset"],
            active_tools=["wheel_zoom"],
            hooks=[force_wheel_zoom_hook, renderer_name_hook(DENSITY_RENDERER)],
            show_grid=True,
            toolbar="right",
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
            self._publish_selection(
                row_ids,
                bounds=bounds,
                total_matches=total,
                truncated=truncated,
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

    def _publish_focus(
        self,
        row_id: str,
        *,
        origin: str = "core.visualisation.scatter.tap",
    ) -> None:
        dataset_id = self._dataset_id()
        selection = getattr(self.context, "selection", None)

        if not row_id or not dataset_id or selection is None:
            return

        selection.set_focus(
            dataset_id=dataset_id,
            row_id=str(row_id),
            origin=origin,
            panel_id=self.panel_id,
        )

        # This panel ignores its own selection events to avoid feedback loops,
        # so schedule a local refresh to show the focus overlay.
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

    def _publish_selection(
        self,
        row_ids: List[str],
        bounds: Optional[Sequence[float]] = None,
        *,
        total_matches: Optional[int] = None,
        truncated: bool = False,
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