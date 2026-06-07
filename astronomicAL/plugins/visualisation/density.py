from __future__ import annotations

from dataclasses import replace
from typing import Any, Optional, Tuple

import datashader as ds
import holoviews as hv
import panel as pn
import numpy as np
import pandas as pd
from holoviews import streams
from holoviews.operation.datashader import rasterize

from bokeh.models import ColumnDataSource

import time

from .base import BaseVisualisationPanel
from .constants import INTERNAL_ROW_ID, INTERNAL_X, INTERNAL_Y, PLOT_MIN_HEIGHT
from .utils import (
    DENSITY_RENDERER,
    PreparedFrame,
    VISIBLE_DENSITY_CMAP,
    force_wheel_zoom_hook,
    renderer_name_hook,
    sample_prepared_frame,
)
from .widgets import (
    settings_box,
    settings_checkbox,
    settings_float_input,
    settings_int_input,
    settings_int_slider,
    settings_multichoice,
    settings_select,
)

DensityExtent = Tuple[float, float, float, float]


class DensityPanel(BaseVisualisationPanel):
    title = "2D Density"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._density_range_hook_keys = set()
        self._density_range_sync_pending = False
        self._density_pending_bokeh_ranges = None
        self._density_range_syncing = False

        self._density_active_range_ids = None
        self._density_current_plot_extent = None

        self._density_focus_stream = None
        self._density_focus_dmap = None
        self._density_focus_signature = None
        self._density_focus_size = None
        self._density_current_raw_extent = None

        self._density_focus_source = None
        self._density_focus_hook_keys = set()
        self._density_focus_pending_frame = None

        self._watch_state(
            [
                "density_bins",
                "density_interactive_sample_limit",
                "x_min",
                "x_max",
                "y_min",
                "y_max",
            ]
        )

    def _settings_controls(self):
        return settings_box(
            self.status_pane,
            settings_multichoice(self.state.param.label_filter, name="Labels", width=210),
            settings_int_slider(self.state.param.density_bins, name="Density bins", width=175),
            settings_select(self.state.param.render_mode, name="Render", width=130),
            settings_int_input(self.state.param.datashade_threshold, name="Shade threshold", width=145),
            settings_int_input(self.state.param.density_interactive_sample_limit, name="Sample limit", width=150),
            settings_float_input(self.state.param.x_min, name="xmin", width=105),
            settings_float_input(self.state.param.x_max, name="xmax", width=105),
            settings_float_input(self.state.param.y_min, name="ymin", width=105),
            settings_float_input(self.state.param.y_max, name="ymax", width=105),
            settings_checkbox(self.state.param.log_x, name="Log X"),
            settings_checkbox(self.state.param.log_y, name="Log Y"),
            settings_checkbox(self.state.param.log_density, name="Log density"),
        )

    def _density_view_range_hook(self, plot: Any, element: Any) -> None:
        """Attach Bokeh range listeners so plot zoom/pan updates xmin/xmax/ymin/ymax.

        The density plot manually transforms log axes with np.log10 before
        rendering. Therefore Bokeh's visible range is in display-space for log
        axes, and must be converted back to raw data-space before writing the
        limit widgets.
        """
        try:
            figure = plot.state
            x_range = figure.x_range
            y_range = figure.y_range
            range_ids = (id(x_range), id(y_range))
            self._density_active_range_ids = range_ids
        except Exception:
            return

        key = (id(figure), id(x_range), id(y_range))
        if key in self._density_range_hook_keys:
            return

        # Avoid unbounded growth across repeated HoloViews re-renders.
        if len(self._density_range_hook_keys) > 64:
            self._density_range_hook_keys.clear()

        self._density_range_hook_keys.add(key)

        def _changed(attr: str, old: Any, new: Any) -> None:
            self._schedule_density_view_range_sync(
                x_range,
                y_range,
                range_ids=range_ids,
            )

        for range_obj in (x_range, y_range):
            for attr in ("start", "end"):
                try:
                    range_obj.on_change(attr, _changed)
                except Exception:
                    pass

    def _schedule_density_view_range_sync(
        self,
        x_range: Any,
        y_range: Any,
        *,
        range_ids=None,
    ) -> None:
        if getattr(self, "_disposed", False):
            return

        if getattr(self, "_density_range_syncing", False):
            return

        if range_ids is not None:
            active = getattr(self, "_density_active_range_ids", None)
            if active is not None and range_ids != active:
                return

        self._density_pending_bokeh_ranges = (x_range, y_range)

        if getattr(self, "_density_range_sync_pending", False):
            return

        self._density_range_sync_pending = True

        def _run() -> None:
            self._density_range_sync_pending = False
            ranges = getattr(self, "_density_pending_bokeh_ranges", None)
            self._density_pending_bokeh_ranges = None

            if not ranges:
                return

            pending_x_range, pending_y_range = ranges

            pending_ids = (id(pending_x_range), id(pending_y_range))
            active = getattr(self, "_density_active_range_ids", None)
            if active is not None and pending_ids != active:
                return

            self._sync_density_limits_from_view(pending_x_range, pending_y_range)

        try:
            doc = pn.state.curdoc
            if doc is not None:
                doc.add_timeout_callback(_run, 150)
            else:
                _run()
        except Exception:
            _run()

    def _sync_density_limits_from_view(self, x_range: Any, y_range: Any) -> None:
        try:
            display_x = self._range_pair_from_bokeh(x_range)
            display_y = self._range_pair_from_bokeh(y_range)

            if display_x is None or display_y is None:
                return

            raw_x = self._display_range_to_raw(display_x, log_axis=bool(self.state.log_x))
            raw_y = self._display_range_to_raw(display_y, log_axis=bool(self.state.log_y))

            if raw_x is None or raw_y is None:
                return

            xmin, xmax = raw_x
            ymin, ymax = raw_y

            changed = False

            self._density_range_syncing = True
            old_suppress = bool(getattr(self, "_suppress_state_refresh", False))
            self._suppress_state_refresh = True

            try:
                changed |= self._set_density_limit_if_changed("x_min", xmin)
                changed |= self._set_density_limit_if_changed("x_max", xmax)
                changed |= self._set_density_limit_if_changed("y_min", ymin)
                changed |= self._set_density_limit_if_changed("y_max", ymax)
            finally:
                self._suppress_state_refresh = old_suppress
                self._density_range_syncing = False

            # Keep BaseVisualisationPanel's remembered ranges in display-space.
            # This is useful for any overlay code that checks the current plot view.
            try:
                self._remember_ranges(x_range=display_x, y_range=display_y)
            except Exception:
                pass

            if changed:
                self._schedule_refresh(reason="density.view.range_changed", delay_ms=250)

        except Exception:
            return

    def _range_pair_from_bokeh(self, range_obj: Any) -> Optional[Tuple[float, float]]:
        try:
            start = float(getattr(range_obj, "start", None))
            end = float(getattr(range_obj, "end", None))
        except Exception:
            return None

        if not np.isfinite(start) or not np.isfinite(end) or start == end:
            return None

        return min(start, end), max(start, end)

    def _display_range_to_raw(
        self,
        value: Tuple[float, float],
        *,
        log_axis: bool,
    ) -> Optional[Tuple[float, float]]:
        lo, hi = value

        if log_axis:
            try:
                lo = float(np.power(10.0, lo))
                hi = float(np.power(10.0, hi))
            except Exception:
                return None

        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            return None

        if log_axis and (lo <= 0 or hi <= 0):
            return None

        return min(lo, hi), max(lo, hi)

    def _set_density_limit_if_changed(self, name: str, value: float) -> bool:
        try:
            value = float(value)
        except Exception:
            return False

        if not np.isfinite(value):
            return False

        current = getattr(self.state, name, None)

        try:
            if current is not None and np.isclose(
                float(current),
                value,
                rtol=1e-8,
                atol=1e-12,
            ):
                return False
        except Exception:
            pass

        try:
            setattr(self.state, name, value)
            return True
        except Exception:
            return False

    def _supports_incremental_focus_marker(self) -> bool:
        return True

    def _empty_density_focus_marker_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                INTERNAL_ROW_ID: pd.Series([], dtype="object"),
                INTERNAL_X: pd.Series([], dtype="float64"),
                INTERNAL_Y: pd.Series([], dtype="float64"),
            }
        )

    def _density_focus_source_data(self, frame: Optional[pd.DataFrame]) -> dict:
        """Convert a 0/1-row focus marker frame into Bokeh CDS data."""
        if frame is None or getattr(frame, "empty", True):
            return {
                INTERNAL_ROW_ID: [],
                INTERNAL_X: [],
                INTERNAL_Y: [],
            }

        try:
            return {
                INTERNAL_ROW_ID: [str(v) for v in frame[INTERNAL_ROW_ID].tolist()],
                INTERNAL_X: [float(v) for v in frame[INTERNAL_X].tolist()],
                INTERNAL_Y: [float(v) for v in frame[INTERNAL_Y].tolist()],
            }
        except Exception:
            return {
                INTERNAL_ROW_ID: [],
                INTERNAL_X: [],
                INTERNAL_Y: [],
            }

    def _set_density_focus_source_frame(self, frame: Optional[pd.DataFrame]) -> bool:
        """Update the live Bokeh focus marker source without touching HoloViews."""
        source = getattr(self, "_density_focus_source", None)
        if source is None:
            self._density_focus_pending_frame = frame
            return False

        try:
            source.data = self._density_focus_source_data(frame)
            return True
        except Exception as exc:
            print(
                "[AstronomicAL density] direct focus source update failed "
                f"panel_id={getattr(self, 'panel_id', None)} "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            return False

    def _density_focus_bokeh_hook(self, plot: Any, element: Any) -> None:
        """Attach a lightweight Bokeh glyph used for focus marker updates.

        This replaces the HoloViews DynamicMap/Pipe focus marker because Pipe
        updates were taking ~1 second with 13M rows.
        """
        try:
            figure = plot.state
        except Exception:
            return

        key = id(figure)

        # Prevent unbounded growth across repeated re-renders.
        if len(self._density_focus_hook_keys) > 64:
            self._density_focus_hook_keys.clear()

        if key in self._density_focus_hook_keys:
            return

        self._density_focus_hook_keys.add(key)

        try:
            source = ColumnDataSource(
                data=self._density_focus_source_data(
                    getattr(self, "_density_focus_pending_frame", None)
                )
            )

            figure.scatter(
                x=INTERNAL_X,
                y=INTERNAL_Y,
                source=source,
                marker="circle",
                size=14,
                fill_alpha=0.0,
                line_color="black",
                line_width=3.0,
            )

            self._density_focus_source = source

            print(
                "[AstronomicAL density] attached direct Bokeh focus marker "
                f"panel_id={getattr(self, 'panel_id', None)} "
                f"figure_key={key}",
                flush=True,
            )

        except Exception as exc:
            print(
                "[AstronomicAL density] failed to attach direct Bokeh focus marker "
                f"panel_id={getattr(self, 'panel_id', None)} "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )

    def _density_extent_signature(self, extent):
        if extent is None:
            return None
        try:
            return tuple(round(float(v), 12) for v in extent)
        except Exception:
            return None


    def _density_focus_stream_signature(self, plot_extent=None):
        return (
            str(self._dataset_id()),
            str(getattr(self.state, "x", "") or ""),
            str(getattr(self.state, "y", "") or ""),
            str(getattr(self.state, "record_id_col", "") or ""),
            bool(getattr(self.state, "log_x", False)),
            bool(getattr(self.state, "log_y", False)),
            self._density_extent_signature(plot_extent),
        )

    def _density_focus_marker_frame(
        self,
        *,
        row_id: str,
        point,
        raw_extent: Optional[DensityExtent] = None,
    ) -> pd.DataFrame:
        if point is None:
            return self._empty_density_focus_marker_frame()

        raw_extent = raw_extent or getattr(self, "_density_current_raw_extent", None)
        if raw_extent is None:
            return self._empty_density_focus_marker_frame()

        try:
            x, y = point
            x = float(x)
            y = float(y)
        except Exception:
            return self._empty_density_focus_marker_frame()

        if not np.isfinite(x) or not np.isfinite(y):
            return self._empty_density_focus_marker_frame()

        xmin, xmax, ymin, ymax = raw_extent
        if x < xmin or x > xmax or y < ymin or y > ymax:
            return self._empty_density_focus_marker_frame()

        if bool(getattr(self.state, "log_x", False)):
            if x <= 0:
                return self._empty_density_focus_marker_frame()
            x = float(np.log10(x))

        if bool(getattr(self.state, "log_y", False)):
            if y <= 0:
                return self._empty_density_focus_marker_frame()
            y = float(np.log10(y))

        return pd.DataFrame(
            [
                {
                    INTERNAL_ROW_ID: str(row_id or ""),
                    INTERNAL_X: x,
                    INTERNAL_Y: y,
                }
            ]
        )

    def _density_focus_marker_element(
        self,
        frame: pd.DataFrame,
        *,
        size: float,
        plot_extent: Optional[DensityExtent] = None,
    ):
        if frame is None or frame.empty:
            frame = self._empty_density_focus_marker_frame()

        opts = dict(
            marker="circle",
            size=float(size),
            fill_alpha=0.0,
            line_color="black",
            line_width=3.0,
            tools=[],
            active_tools=[],
            toolbar=None,
            logx=False,
            logy=False,
            shared_axes=False,
            axiswise=True,
            framewise=True,
        )

        if plot_extent is not None:
            xmin, xmax, ymin, ymax = plot_extent
            opts["xlim"] = (xmin, xmax)
            opts["ylim"] = (ymin, ymax)

        return hv.Points(
            frame,
            kdims=[INTERNAL_X, INTERNAL_Y],
            vdims=[INTERNAL_ROW_ID],
        ).opts(**opts)

    def _current_density_focus_marker_frame(
        self,
        raw_data: Optional[PreparedFrame] = None,
        raw_extent: Optional[DensityExtent] = None,
    ) -> pd.DataFrame:
        focus = self._active_focus_state()
        if focus is None:
            return self._empty_density_focus_marker_frame()

        row_id = str(getattr(focus, "row_id", "") or "")
        if not row_id:
            return self._empty_density_focus_marker_frame()

        point = None

        try:
            point = self._focus_point_from_metadata(focus)
        except Exception:
            point = None

        if point is None:
            row_df = self._cached_focus_row_for_focus(focus)
            point = self._focus_point_from_row_df_for_current_axes(
                row_df,
                row_id=row_id,
            )

        if point is None and raw_data is not None:
            # This follows the existing density behaviour. For large datasets,
            # BaseVisualisationPanel should avoid expensive synchronous scans
            # and schedule async resolution instead.
            try:
                point = self._focus_point(raw_data)
            except Exception:
                point = None

        return self._density_focus_marker_frame(
            row_id=row_id,
            point=point,
            raw_extent=raw_extent,
        )

    def _density_focus_dynamic_overlay(
        self,
        raw_data: PreparedFrame,
        raw_extent: DensityExtent,
        *,
        plot_extent: DensityExtent,
        size: float = 14,
    ):
        signature = self._density_focus_stream_signature(plot_extent)
        size = float(size)

        self._density_current_raw_extent = raw_extent
        self._density_current_plot_extent = plot_extent

        if (
            self._density_focus_stream is None
            or self._density_focus_dmap is None
            or self._density_focus_signature != signature
            or self._density_focus_size != size
        ):
            initial = self._current_density_focus_marker_frame(
                raw_data,
                raw_extent,
            )

            self._density_focus_stream = streams.Pipe(data=initial)
            self._density_focus_signature = signature
            self._density_focus_size = size

            def _make_focus_marker(data):
                return self._density_focus_marker_element(
                    data,
                    size=size,
                    plot_extent=plot_extent,
                )

            self._density_focus_dmap = hv.DynamicMap(
                _make_focus_marker,
                streams=[self._density_focus_stream],
            )

        return self._density_focus_dmap

    def _apply_density_focus_marker_from_current_state_without_rebuild(
        self,
        *,
        raw_data: Optional[PreparedFrame] = None,
        raw_extent: Optional[DensityExtent] = None,
    ) -> bool:
        frame = self._current_density_focus_marker_frame(
            raw_data,
            raw_extent,
        )
        self._density_focus_pending_frame = frame
        return self._set_density_focus_source_frame(frame)

    def _apply_focus_marker_point(self, *, row_id: str, point, clear: bool = False) -> bool:
        """BaseVisualisationPanel hook: update only the direct Bokeh focus marker."""
        if clear:
            frame = self._empty_density_focus_marker_frame()
        else:
            frame = self._density_focus_marker_frame(
                row_id=str(row_id or ""),
                point=point,
                raw_extent=getattr(self, "_density_current_raw_extent", None),
            )

        self._density_focus_pending_frame = frame

        ok = self._set_density_focus_source_frame(frame)

        print(
            "[AstronomicAL density] direct focus marker updated "
            f"panel_id={getattr(self, 'panel_id', None)} "
            f"row_id={row_id!r} "
            f"rows={len(frame)} "
            f"ok={ok}",
            flush=True,
        )

        return ok

    def _render(self) -> None:
        total_t0 = time.perf_counter()
        stage = "start"

        def mark(label: str, start: float, **fields) -> None:
            self._timing(
                f"density.render.{label}",
                time.perf_counter() - start,
                **fields,
            )

        try:
            stage = "plot_data"
            t0 = time.perf_counter()
            raw_data = self._plot_data(require_y=True)
            mark(
                "plot_data",
                t0,
                rows=len(raw_data.frame),
                empty=raw_data.empty,
            )

            if raw_data.empty:
                stage = "empty.raw_data"
                t0 = time.perf_counter()
                self.plot_pane.object = self._empty("No finite X/Y data")
                mark("plot_pane_empty_assign", t0, reason="raw_data_empty")

                t0 = time.perf_counter()
                self.status_pane.object = "0 density rows"
                mark("status_assign", t0, reason="raw_data_empty")

                return

            try:
                stage = "density_extent"
                t0 = time.perf_counter()
                raw_extent = self._density_extent(raw_data)
                mark("density_extent", t0, extent=raw_extent)
            except ValueError as exc:
                stage = "density_extent.error"
                t0 = time.perf_counter()
                self.plot_pane.object = self._empty(str(exc))
                mark("plot_pane_empty_assign", t0, reason="invalid_density_limits")

                t0 = time.perf_counter()
                self.status_pane.object = "Invalid density limits"
                mark("status_assign", t0, reason="invalid_density_limits")
                return

            stage = "clip_to_density_extent"
            t0 = time.perf_counter()
            clipped_raw_data = self._clip_to_density_extent(raw_data, raw_extent)
            mark(
                "clip_to_density_extent",
                t0,
                input_rows=len(raw_data.frame),
                output_rows=len(clipped_raw_data.frame),
            )

            if clipped_raw_data.empty:
                stage = "empty.clipped"
                t0 = time.perf_counter()
                self.plot_pane.object = self._empty("No finite X/Y data inside limits")
                mark("plot_pane_empty_assign", t0, reason="clipped_empty")

                t0 = time.perf_counter()
                self.status_pane.object = "0 density rows inside limits"
                mark("status_assign", t0, reason="clipped_empty")
                return

            try:
                stage = "transform_for_plotting"
                t0 = time.perf_counter()
                plot_data, plot_extent = self._transform_for_plotting(
                    clipped_raw_data,
                    raw_extent,
                )
                mark(
                    "transform_for_plotting",
                    t0,
                    input_rows=len(clipped_raw_data.frame),
                    output_rows=len(plot_data.frame),
                    plot_extent=plot_extent,
                )
            except ValueError as exc:
                stage = "transform_for_plotting.error"
                t0 = time.perf_counter()
                self.plot_pane.object = self._empty(str(exc))
                mark("plot_pane_empty_assign", t0, reason="invalid_log_density_data")

                t0 = time.perf_counter()
                self.status_pane.object = "Invalid log-density data"
                mark("status_assign", t0, reason="invalid_log_density_data")
                return

            if plot_data.empty:
                stage = "empty.plot_data"
                t0 = time.perf_counter()
                self.plot_pane.object = self._empty("No finite X/Y data inside limits")
                mark("plot_pane_empty_assign", t0, reason="plot_data_empty")

                t0 = time.perf_counter()
                self.status_pane.object = "0 density rows inside limits"
                mark("status_assign", t0, reason="plot_data_empty")
                return

            stage = "should_rasterize"
            t0 = time.perf_counter()
            use_raster = self._should_rasterize(clipped_raw_data)
            mark(
                "should_rasterize",
                t0,
                use_raster=use_raster,
                rows=len(clipped_raw_data.frame),
                render_mode=getattr(self.state, "render_mode", None),
            )

            if use_raster:
                stage = "density_rasterized"
                t0 = time.perf_counter()
                base = self._density_rasterized(plot_data, extent=plot_extent)
                mark(
                    "density_rasterized",
                    t0,
                    rows=len(plot_data.frame),
                    bins=getattr(self.state, "density_bins", None),
                )

                render_label = "rasterized grid"
                plotted_count = len(plot_data.frame)
                sampled_note = ""
            else:
                stage = "sample_prepared_frame"
                t0 = time.perf_counter()
                sampled_plot_data = sample_prepared_frame(
                    plot_data,
                    int(self.state.density_interactive_sample_limit),
                    seed=1,
                )
                mark(
                    "sample_prepared_frame",
                    t0,
                    input_rows=len(plot_data.frame),
                    output_rows=len(sampled_plot_data.frame),
                    sampled_from=sampled_plot_data.sampled_from,
                )

                stage = "density_hextiles"
                t0 = time.perf_counter()
                base = self._density_hextiles(sampled_plot_data, extent=plot_extent)
                mark(
                    "density_hextiles",
                    t0,
                    rows=len(sampled_plot_data.frame),
                    bins=getattr(self.state, "density_bins", None),
                )

                render_label = "hexbin grid"
                plotted_count = len(sampled_plot_data.frame)
                sampled_note = (
                    f" · sampled from {sampled_plot_data.sampled_from:,} inside limits"
                    if sampled_plot_data.sampled_from
                    else ""
                )

            stage = "density_focus_dynamic_overlay"
            t0 = time.perf_counter()
            self._density_current_raw_extent = raw_extent
            self._density_current_plot_extent = plot_extent
            self._density_focus_pending_frame = self._current_density_focus_marker_frame(
                clipped_raw_data,
                raw_extent,
            )

            # Do not add the focus marker as a HoloViews DynamicMap/Pipe overlay.
            # That path re-renders too much of the density plot on every focus
            # change. The marker is attached as a direct Bokeh glyph via hook below.
            items = [base]
            x_label, y_label = self._axis_labels()
            overlay = hv.Overlay(items).collate().opts(
                responsive=True,
                min_height=PLOT_MIN_HEIGHT,
                xlabel=x_label,
                ylabel=y_label,
                xlim=(plot_extent[0], plot_extent[1]),
                ylim=(plot_extent[2], plot_extent[3]),
                show_grid=True,
                toolbar="right",
                tools=["pan", "wheel_zoom", "box_zoom", "reset"],
                active_tools=["wheel_zoom"],
                hooks=[
                    force_wheel_zoom_hook,
                    self._density_view_range_hook,
                    self._density_focus_bokeh_hook,
                ],
                shared_axes=False,
                axiswise=True,
                framewise=True,
            )
            mark("overlay_build", t0, item_count=len(items))

            stage = "plot_pane_assign"
            t0 = time.perf_counter()
            self.plot_pane.object = overlay
            mark(
                "plot_pane_assign",
                t0,
                item_count=len(items),
                plotted_count=plotted_count,
                render_label=render_label,
            )

            full_count = len(raw_data.frame)
            clipped_count = len(clipped_raw_data.frame)

            stage = "status_assign"
            t0 = time.perf_counter()
            self.status_pane.object = (
                f"{full_count:,} finite rows · "
                f"{clipped_count:,} inside limits · "
                f"{plotted_count:,} shown · {render_label}{sampled_note}"
            )
            mark(
                "status_assign",
                t0,
                full_count=full_count,
                clipped_count=clipped_count,
                plotted_count=plotted_count,
            )

        finally:
            self._timing(
                "density.render.total",
                time.perf_counter() - total_t0,
                stage=stage,
            )

    def _should_rasterize(self, data: PreparedFrame) -> bool:
        if self.state.render_mode == "datashader":
            return True
        if self.state.render_mode == "interactive":
            return False
        return len(data.frame) > int(self.state.datashade_threshold)

    def _state_limit(self, name: str) -> Optional[float]:
        value = getattr(self.state, name, None)
        if value is None:
            return None
        try:
            value = float(value)
        except Exception:
            return None
        if not np.isfinite(value):
            return None
        return value

    def _axis_values_for_extent(
        self,
        data: PreparedFrame,
        column: str,
        *,
        log_axis: bool,
        axis_name: str,
    ) -> np.ndarray:
        values = np.asarray(data.frame[column].to_numpy(copy=False), dtype=float)
        values = values[np.isfinite(values)]

        if log_axis:
            values = values[values > 0]
            if len(values) == 0:
                raise ValueError(f"No positive finite {axis_name} data for log-density plot")

        if len(values) == 0:
            raise ValueError(f"No finite {axis_name} data for density plot")

        return values

    def _density_extent(self, data: PreparedFrame) -> DensityExtent:
        x = self._axis_values_for_extent(
            data,
            INTERNAL_X,
            log_axis=bool(self.state.log_x),
            axis_name="X",
        )
        y = self._axis_values_for_extent(
            data,
            INTERNAL_Y,
            log_axis=bool(self.state.log_y),
            axis_name="Y",
        )

        xmin = self._state_limit("x_min")
        xmax = self._state_limit("x_max")
        ymin = self._state_limit("y_min")
        ymax = self._state_limit("y_max")

        if xmin is None:
            xmin = float(np.nanmin(x))
        if xmax is None:
            xmax = float(np.nanmax(x))
        if ymin is None:
            ymin = float(np.nanmin(y))
        if ymax is None:
            ymax = float(np.nanmax(y))

        if not all(np.isfinite([xmin, xmax, ymin, ymax])):
            raise ValueError("Density limits must be finite numbers or empty")

        if xmax <= xmin:
            raise ValueError("xmax must be greater than xmin")
        if ymax <= ymin:
            raise ValueError("ymax must be greater than ymin")

        if self.state.log_x and (xmin <= 0 or xmax <= 0):
            raise ValueError("xmin and xmax must be positive when Log X is enabled")
        if self.state.log_y and (ymin <= 0 or ymax <= 0):
            raise ValueError("ymin and ymax must be positive when Log Y is enabled")

        return float(xmin), float(xmax), float(ymin), float(ymax)

    def _clip_to_density_extent(
        self,
        data: PreparedFrame,
        extent: DensityExtent,
    ) -> PreparedFrame:
        xmin, xmax, ymin, ymax = extent
        frame = data.frame
        mask = (
            (frame[INTERNAL_X] >= xmin)
            & (frame[INTERNAL_X] <= xmax)
            & (frame[INTERNAL_Y] >= ymin)
            & (frame[INTERNAL_Y] <= ymax)
        )

        if self.state.log_x:
            mask &= frame[INTERNAL_X] > 0
        if self.state.log_y:
            mask &= frame[INTERNAL_Y] > 0

        clipped = frame.loc[mask]

        return replace(
            data,
            frame=clipped,
            row_count_after_filter=len(clipped),
            sampled_from=None,
        )

    def _transform_for_plotting(
        self,
        data: PreparedFrame,
        raw_extent: DensityExtent,
    ) -> Tuple[PreparedFrame, DensityExtent]:
        xmin, xmax, ymin, ymax = raw_extent

        frame = data.frame.copy()

        if self.state.log_x:
            x = np.asarray(frame[INTERNAL_X].to_numpy(copy=False), dtype=float)
            if np.any(~np.isfinite(x)) or np.any(x <= 0):
                frame = frame[np.isfinite(frame[INTERNAL_X]) & (frame[INTERNAL_X] > 0)].copy()
            frame[INTERNAL_X] = np.log10(frame[INTERNAL_X].astype(float))
            xmin = float(np.log10(xmin))
            xmax = float(np.log10(xmax))

        if self.state.log_y:
            y = np.asarray(frame[INTERNAL_Y].to_numpy(copy=False), dtype=float)
            if np.any(~np.isfinite(y)) or np.any(y <= 0):
                frame = frame[np.isfinite(frame[INTERNAL_Y]) & (frame[INTERNAL_Y] > 0)].copy()
            frame[INTERNAL_Y] = np.log10(frame[INTERNAL_Y].astype(float))
            ymin = float(np.log10(ymin))
            ymax = float(np.log10(ymax))

        if frame.empty:
            raise ValueError("No positive finite data remain after log transform")

        return (
            replace(
                data,
                frame=frame,
                row_count_after_filter=len(frame),
                sampled_from=None,
            ),
            (float(xmin), float(xmax), float(ymin), float(ymax)),
        )

    def _axis_labels(self) -> Tuple[str, str]:
        x_label = str(self.state.x)
        y_label = str(self.state.y)

        if self.state.log_x:
            x_label = f"log({x_label})"
        if self.state.log_y:
            y_label = f"log({y_label})"

        return x_label, y_label

    def _density_hextiles(self, data: PreparedFrame, *, extent: DensityExtent):
        xmin, xmax, ymin, ymax = extent
        gridsize = max(5, min(500, int(self.state.density_bins)))
        cnorm = "log" if self.state.log_density else "linear"
        x_label, y_label = self._axis_labels()

        return hv.HexTiles(
            data.frame,
            kdims=[INTERNAL_X, INTERNAL_Y],
        ).opts(
            gridsize=gridsize,
            cmap=VISIBLE_DENSITY_CMAP,
            colorbar=True,
            cnorm=cnorm,
            responsive=True,
            min_height=PLOT_MIN_HEIGHT,
            xlabel=x_label,
            ylabel=y_label,
            xlim=(xmin, xmax),
            ylim=(ymin, ymax),
            tools=["pan", "wheel_zoom", "box_zoom", "reset"],
            active_tools=["wheel_zoom"],
            hooks=[
                force_wheel_zoom_hook,
                self._density_view_range_hook,
                renderer_name_hook(DENSITY_RENDERER),
            ],
            show_grid=True,
            toolbar="right",
            line_alpha=0.15,
            shared_axes=False,
            axiswise=True,
            framewise=True,
        )

    def _density_rasterized(self, data: PreparedFrame, *, extent: DensityExtent):
        xmin, xmax, ymin, ymax = extent
        bins = max(5, min(500, int(self.state.density_bins)))
        cnorm = "log" if self.state.log_density else "linear"
        x_label, y_label = self._axis_labels()

        points = hv.Points(
            data.frame[[INTERNAL_X, INTERNAL_Y]],
            kdims=[INTERNAL_X, INTERNAL_Y],
        )

        return rasterize(
            points,
            aggregator=ds.count(),
            width=bins,
            height=bins,
            x_range=(xmin, xmax),
            y_range=(ymin, ymax),
            dynamic=False,
        ).opts(
            cmap=VISIBLE_DENSITY_CMAP,
            colorbar=True,
            cnorm=cnorm,
            clipping_colors={"NaN": "white"},
            bgcolor="white",
            responsive=True,
            min_height=PLOT_MIN_HEIGHT,
            xlabel=x_label,
            ylabel=y_label,
            xlim=(xmin, xmax),
            ylim=(ymin, ymax),
            tools=["pan", "wheel_zoom", "box_zoom", "reset"],
            active_tools=["wheel_zoom"],
            hooks=[
                force_wheel_zoom_hook,
                self._density_view_range_hook,
                renderer_name_hook(DENSITY_RENDERER),
            ],
            show_grid=True,
            toolbar="right",
            shared_axes=False,
            axiswise=True,
            framewise=True,
        )


    def _density_focus_overlay(
        self,
        raw_data: PreparedFrame,
        raw_extent: DensityExtent,
        *,
        size: float = 14,
    ):
        """
        Focus overlay for density plots with manually transformed log10 axes.
    
        BaseVisualisationPanel._focus_overlay() assumes raw coordinates plus
        HoloViews/Bokeh log axes. The density panel does not use HoloViews log axes:
        it transforms x/y with np.log10 before plotting. Therefore the focused point
        must be transformed here as well.
        """
        point = self._focus_point(raw_data)
    
        if point is None:
            return hv.Overlay([])
    
        x, y = point
    
        if y is None:
            return hv.Overlay([])
    
        try:
            x = float(x)
            y = float(y)
        except Exception:
            return hv.Overlay([])
    
        if not np.isfinite(x) or not np.isfinite(y):
            return hv.Overlay([])
    
        xmin, xmax, ymin, ymax = raw_extent
    
        if x < xmin or x > xmax or y < ymin or y > ymax:
            return hv.Overlay([])
    
        if self.state.log_x:
            if x <= 0:
                return hv.Overlay([])
            x = float(np.log10(x))
    
        if self.state.log_y:
            if y <= 0:
                return hv.Overlay([])
            y = float(np.log10(y))
    
        return hv.Points(
            [(x, y)],
            kdims=[INTERNAL_X, INTERNAL_Y],
        ).opts(
            marker="circle",
            size=size,
            fill_alpha=0.0,
            line_color="black",
            line_width=3,
            tools=[],
            active_tools=[],
            toolbar=None,
            logx=False,
            logy=False,
            shared_axes=False,
            axiswise=True,
            framewise=True,
        )