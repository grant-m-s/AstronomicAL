from __future__ import annotations

from dataclasses import replace
from typing import Optional, Tuple

import datashader as ds
import holoviews as hv
import numpy as np
from holoviews.operation.datashader import rasterize

from .base import BaseVisualisationPanel
from .constants import INTERNAL_X, INTERNAL_Y, PLOT_MIN_HEIGHT
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

        # BaseVisualisationPanel currently does not watch density_bins or the
        # optional density-domain limits. Keep this local to the density panel
        # so we do not need to replace the large shared base.py file.
        self._watch_state(
            [
                "density_bins",
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
            settings_int_input(self.state.param.interactive_sample_limit, name="Sample limit", width=130),
            settings_float_input(self.state.param.x_min, name="xmin", width=105),
            settings_float_input(self.state.param.x_max, name="xmax", width=105),
            settings_float_input(self.state.param.y_min, name="ymin", width=105),
            settings_float_input(self.state.param.y_max, name="ymax", width=105),
            settings_checkbox(self.state.param.log_x, name="Log X"),
            settings_checkbox(self.state.param.log_y, name="Log Y"),
            settings_checkbox(self.state.param.log_density, name="Log density"),
        )

    def _render(self) -> None:
        raw_data = self._plot_data(require_y=True)

        if raw_data.empty:
            self.plot_pane.object = self._empty("No finite X/Y data")
            self.status_pane.object = "0 density rows"
            return

        try:
            raw_extent = self._density_extent(raw_data)
        except ValueError as exc:
            self.plot_pane.object = self._empty(str(exc))
            self.status_pane.object = "Invalid density limits"
            return

        clipped_raw_data = self._clip_to_density_extent(raw_data, raw_extent)

        if clipped_raw_data.empty:
            self.plot_pane.object = self._empty("No finite X/Y data inside limits")
            self.status_pane.object = "0 density rows inside limits"
            return

        try:
            plot_data, plot_extent = self._transform_for_plotting(
                clipped_raw_data,
                raw_extent,
            )
        except ValueError as exc:
            self.plot_pane.object = self._empty(str(exc))
            self.status_pane.object = "Invalid log-density data"
            return

        if plot_data.empty:
            self.plot_pane.object = self._empty("No finite X/Y data inside limits")
            self.status_pane.object = "0 density rows inside limits"
            return

        use_raster = self._should_rasterize(clipped_raw_data)

        if use_raster:
            base = self._density_rasterized(plot_data, extent=plot_extent)
            render_label = "rasterized grid"
            plotted_count = len(plot_data.frame)
            sampled_note = ""
        else:
            sampled_plot_data = sample_prepared_frame(
                plot_data,
                int(self.state.interactive_sample_limit),
                seed=1,
            )
            base = self._density_hextiles(sampled_plot_data, extent=plot_extent)
            render_label = "hexbin grid"
            plotted_count = len(sampled_plot_data.frame)
            sampled_note = (
                f" · sampled from {sampled_plot_data.sampled_from:,} inside limits"
                if sampled_plot_data.sampled_from
                else ""
            )
        
        focus = self._density_focus_overlay(clipped_raw_data,raw_extent,size=14,)
        items = [item for item in [base, focus] if item is not None]

        x_label, y_label = self._axis_labels()

        self.plot_pane.object = hv.Overlay(items).collate().opts(
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
            hooks=[force_wheel_zoom_hook],
            shared_axes=False,
            axiswise=True,
            framewise=True,
        )

        full_count = len(raw_data.frame)
        clipped_count = len(clipped_raw_data.frame)
        self.status_pane.object = (
            f"{full_count:,} finite rows · "
            f"{clipped_count:,} inside limits · "
            f"{plotted_count:,} shown · {render_label}{sampled_note}"
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
            hooks=[force_wheel_zoom_hook, renderer_name_hook(DENSITY_RENDERER)],
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
            hooks=[force_wheel_zoom_hook, renderer_name_hook(DENSITY_RENDERER)],
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