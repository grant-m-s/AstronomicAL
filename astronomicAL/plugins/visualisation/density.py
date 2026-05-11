from __future__ import annotations

import datashader as ds
import holoviews as hv
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
    settings_int_input,
    settings_int_slider,
    settings_multichoice,
    settings_select,
)


class DensityPanel(BaseVisualisationPanel):
    title = "2D Density"

    def _settings_controls(self):
        return settings_box(
            self.status_pane,
            settings_multichoice(self.state.param.label_filter, name="Labels", width=210),
            settings_int_slider(self.state.param.density_bins, name="Density bins", width=175),
            settings_select(self.state.param.render_mode, name="Render", width=130),
            settings_int_input(self.state.param.datashade_threshold, name="Shade threshold", width=145),
            settings_int_input(self.state.param.interactive_sample_limit, name="Sample limit", width=130),
            settings_checkbox(self.state.param.log_x, name="Log X"),
            settings_checkbox(self.state.param.log_y, name="Log Y"),
            settings_checkbox(self.state.param.log_density, name="Log density"),
        )

    def _render(self) -> None:
        data = self._plot_data(require_y=True)

        if data.empty:
            self.plot_pane.object = self._empty("No finite X/Y data")
            self.status_pane.object = "0 density rows"
            return

        use_raster = self._should_rasterize(data)

        if use_raster:
            base = self._density_rasterized(data)
            render_label = "rasterized"
            plotted_count = len(data.frame)
            sampled_note = ""
        else:
            plot_data = sample_prepared_frame(
                data,
                int(self.state.interactive_sample_limit),
                seed=1,
            )
            base = self._density_hextiles(plot_data)
            render_label = "hexbin"
            plotted_count = len(plot_data.frame)
            sampled_note = (
                f" · sampled from {plot_data.sampled_from:,}"
                if plot_data.sampled_from
                else ""
            )

        focus = self._focus_overlay(data, size=14)

        self.plot_pane.object = hv.Overlay([item for item in [base, focus] if item is not None]).collate().opts(
            responsive=True,
            min_height=PLOT_MIN_HEIGHT,
            xlabel=str(self.state.x),
            ylabel=str(self.state.y),
            show_grid=True,
            toolbar="right",
            tools=["pan", "wheel_zoom", "box_zoom", "reset"],
            active_tools=["wheel_zoom"],
            hooks=[force_wheel_zoom_hook],
            shared_axes=False,
            axiswise=True,
            framewise=True,
        )

        self.status_pane.object = (
            f"{len(data.frame):,} density rows · "
            f"{plotted_count:,} shown · {render_label}{sampled_note}"
        )

    def _should_rasterize(self, data: PreparedFrame) -> bool:
        if self.state.render_mode == "datashader":
            return True
        if self.state.render_mode == "interactive":
            return False
        return len(data.frame) > int(self.state.datashade_threshold)

    def _density_hextiles(self, data: PreparedFrame):
        gridsize = max(5, min(150, int(self.state.density_bins)))

        return hv.HexTiles(
            data.frame,
            kdims=[INTERNAL_X, INTERNAL_Y],
        ).opts(
            gridsize=gridsize,
            cmap=VISIBLE_DENSITY_CMAP,
            colorbar=True,
            responsive=True,
            min_height=PLOT_MIN_HEIGHT,
            xlabel=str(self.state.x),
            ylabel=str(self.state.y),
            logx=self.state.log_x,
            logy=self.state.log_y,
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

    def _density_rasterized(self, data: PreparedFrame):
        points = hv.Points(
            data.frame[[INTERNAL_X, INTERNAL_Y]],
            kdims=[INTERNAL_X, INTERNAL_Y],
        )

        cnorm = "log" if self.state.log_density else "eq_hist"

        return rasterize(
            points,
            aggregator=ds.count(),
            pixel_ratio=2,
        ).opts(
            cmap=VISIBLE_DENSITY_CMAP,
            colorbar=True,
            cnorm=cnorm,
            clipping_colors={"NaN": "white"},
            bgcolor="white",
            responsive=True,
            min_height=PLOT_MIN_HEIGHT,
            xlabel=str(self.state.x),
            ylabel=str(self.state.y),
            logx=self.state.log_x,
            logy=self.state.log_y,
            tools=["pan", "wheel_zoom", "box_zoom", "reset"],
            active_tools=["wheel_zoom"],
            hooks=[force_wheel_zoom_hook, renderer_name_hook(DENSITY_RENDERER)],
            show_grid=True,
            toolbar="right",
            shared_axes=False,
            axiswise=True,
            framewise=True,
        )