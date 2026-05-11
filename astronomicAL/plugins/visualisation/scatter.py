from __future__ import annotations

from typing import List, Optional, Sequence

import datashader as ds
import holoviews as hv
import numpy as np
import pandas as pd
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

    def _render(self) -> None:
        self._clear_stream_watchers()

        data = self._plot_data(require_y=True)
        self._full_interactive_data = data
        self._interactive_current_frame = data.frame

        if data.empty:
            self.plot_pane.object = self._empty("No finite X/Y data")
            self.status_pane.object = "0 plotted rows"
            return

        use_raster = self._should_rasterize(data)

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

        overlays = [
            base,
            self._selection_points(data),
            self._focus_overlay(data, size=max(float(self.state.point_size) + 8, 12)),
        ]

        self.plot_pane.object = hv.Overlay([item for item in overlays if item is not None]).collate().opts(
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

        self.status_pane.object = (
            f"{len(data.frame):,} eligible rows · "
            f"{plotted_count:,} shown · {render_label}{sampled_note}"
        )

    def _should_rasterize(self, data: PreparedFrame) -> bool:
        if self.state.render_mode == "datashader":
            return True
        if self.state.render_mode == "interactive":
            return False
        return len(data.frame) > int(self.state.datashade_threshold)

    def _scatter_interactive_dynamic(self, data: PreparedFrame):
        range_stream = streams.RangeXY(x_range=None, y_range=None)

        def make_points(x_range=None, y_range=None):
            visible = frame_in_ranges(data, x_range, y_range)
            plot_data = sample_prepared_frame(
                visible,
                int(self.state.interactive_sample_limit),
                seed=0,
            )

            self._interactive_current_frame = plot_data.frame

            sampled_note = (
                f" · sampled from {plot_data.sampled_from:,} visible"
                if plot_data.sampled_from
                else ""
            )
            self.status_pane.object = (
                f"{len(data.frame):,} eligible rows · "
                f"{len(visible.frame):,} visible · "
                f"{len(plot_data.frame):,} shown · interactive{sampled_note}"
            )

            return self._scatter_points_element(plot_data)

        dmap = hv.DynamicMap(make_points, streams=[range_stream])

        selection_stream = streams.Selection1D(source=dmap)
        bounds_stream = streams.BoundsXY(source=dmap)

        last_bounds = {"value": None}

        def on_bounds(event):
            last_bounds["value"] = event.new

        def on_select(event):
            indices = list(event.new or [])
            if not indices:
                return

            bounds = last_bounds.get("value")
            if bounds:
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

            frame = getattr(self, "_interactive_current_frame", pd.DataFrame())
            if frame.empty or INTERNAL_ROW_ID not in frame.columns:
                return

            row_ids = frame[INTERNAL_ROW_ID].astype(str).to_numpy(copy=False)
            selected_ids = [
                str(row_ids[index])
                for index in indices
                if 0 <= index < len(row_ids)
            ]
            self._publish_selection(selected_ids, bounds=bounds)

        self._watch_param(bounds_stream, on_bounds, "bounds", render_scoped=True)
        self._watch_param(selection_stream, on_select, "index", render_scoped=True)

        return dmap

    def _scatter_points_element(self, data: PreparedFrame):
        frame = data.frame
        if frame.empty:
            return self._empty("No visible rows")

        vdims = [INTERNAL_ROW_ID]
        if INTERNAL_LABEL_DISPLAY in frame.columns:
            vdims.append(INTERNAL_LABEL_DISPLAY)

        points = hv.Points(
            frame,
            kdims=[INTERNAL_X, INTERNAL_Y],
            vdims=vdims,
        )

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

        opts["hooks"] = list(opts.get("hooks", [])) + [renderer_name_hook(SCATTER_RENDERER)]

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
        frame = data.frame[[INTERNAL_X, INTERNAL_Y]]

        points = hv.Points(
            frame,
            kdims=[INTERNAL_X, INTERNAL_Y],
        )

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
        frame = data.frame

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

        if len(row_ids) == 1 and not total_matches:
            selection.set_focus(
                dataset_id=dataset_id,
                row_id=row_ids[0],
                origin="core.visualisation.scatter.tap",
                panel_id=self.panel_id,
            )
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