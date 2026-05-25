# BUG: Hide Y Var dropdown
# BUG: Change X var slow
# BUG: Assign Label col slow

from __future__ import annotations

import holoviews as hv
import numpy as np

from .base import BaseVisualisationPanel
from .constants import (
    INTERNAL_LABEL_COLOUR,
    INTERNAL_LABEL_DISPLAY,
    INTERNAL_X,
    PLOT_MIN_HEIGHT,
)
from .utils import (
    HIST_RENDERER,
    force_wheel_zoom_hook,
    limited_histogram_hover_tool,
    renderer_name_hook,
)
from .widgets import (
    settings_box,
    settings_checkbox,
    settings_int_slider,
    settings_multichoice,
    settings_select,
)


class HistogramPanel(BaseVisualisationPanel):
    title = "Histogram"

    def _settings_controls(self):
        return settings_box(
            self.status_pane,
            settings_select(self.state.param.color_by, name="Colour", width=130),
            settings_multichoice(self.state.param.label_filter, name="Labels", width=210),
            settings_int_slider(self.state.param.bins, name="Bins", width=180),
            settings_checkbox(self.state.param.density, name="Percent"),
            settings_checkbox(self.state.param.cumulative, name="Cumulative"),
            settings_checkbox(self.state.param.log_x, name="Log X"),
            settings_checkbox(self.state.param.log_y, name="Log Y"),
        )

    def _render(self) -> None:
        data = self._plot_data(require_y=False)

        if data.empty:
            self.plot_pane.object = self._empty("No finite X data")
            self.status_pane.object = "0 histogram rows"
            return

        frame = data.frame
        layers = []

        use_labels = (
            self.state.color_by == "Labels"
            and INTERNAL_LABEL_DISPLAY in frame.columns
            and frame[INTERNAL_LABEL_DISPLAY].nunique(dropna=True) <= 40
        )

        if use_labels:
            for label, sub in frame.groupby(INTERNAL_LABEL_DISPLAY, dropna=False, sort=False):
                colour = _group_colour(sub)
                hist = self._hist(sub[INTERNAL_X].to_numpy(copy=False), str(label), colour)
                if hist is not None:
                    layers.append(hist)
        else:
            hist = self._hist(frame[INTERNAL_X].to_numpy(copy=False), "All", "#1f77b4")
            if hist is not None:
                layers.append(hist)

        focus = self._focus_point(data)
        if focus is not None:
            x, _ = focus
            try:
                layers.append(
                    hv.VLine(float(x)).opts(
                        color="black",
                        line_dash="dashed",
                        line_width=1,
                    )
                )
            except Exception:
                pass

        if not layers:
            self.plot_pane.object = self._empty("No histogram bins")
            self.status_pane.object = "0 histogram bins"
            return

        ylabel = "% of rows" if self.state.density else "# rows"

        self.plot_pane.object = hv.Overlay(layers).opts(
            responsive=True,
            min_height=PLOT_MIN_HEIGHT,
            xlabel=str(self.state.x),
            ylabel=ylabel,
            legend_position="right",
            show_grid=True,
            toolbar="right",
            shared_axes=False,
            axiswise=True,
            framewise=True,
        )

        self.status_pane.object = f"{len(frame):,} histogram rows"

    def _hist(self, values: np.ndarray, label: str, colour: str):
        x = np.asarray(values, dtype=float)
        x = x[np.isfinite(x)]

        if self.state.log_x:
            x = x[x > 0]

        if len(x) == 0:
            return None

        bins = self._bins_for(x)

        if self.state.density:
            weights = np.ones_like(x, dtype=float) * (100.0 / float(len(x)))
        else:
            weights = None

        stats, edges = np.histogram(x, bins=bins, weights=weights)

        if self.state.cumulative:
            stats = np.cumsum(stats)

        return hv.Histogram(
            (edges, stats),
            kdims=[str(self.state.x)],
            vdims=["count"],
            label=label,
        ).opts(
            fill_color=colour,
            line_color=colour,
            fill_alpha=0.45,
            line_width=1.5,
            logx=self.state.log_x,
            logy=self.state.log_y,
            tools=[limited_histogram_hover_tool(), "pan", "wheel_zoom", "box_zoom", "reset"],
            active_tools=["wheel_zoom"],
            hooks=[force_wheel_zoom_hook, renderer_name_hook(HIST_RENDERER)],
            responsive=True,
            min_height=PLOT_MIN_HEIGHT,
            show_grid=True,
            toolbar="right",
            shared_axes=False,
            axiswise=True,
            framewise=True,
        )

    def _bins_for(self, x: np.ndarray):
        n_bins = int(self.state.bins)

        if self.state.log_x and np.min(x) > 0 and np.max(x) > np.min(x):
            return np.geomspace(np.min(x), np.max(x), n_bins + 1)

        return n_bins


def _group_colour(frame) -> str:
    if INTERNAL_LABEL_COLOUR not in frame.columns or frame.empty:
        return "#1f77b4"

    try:
        return str(frame[INTERNAL_LABEL_COLOUR].iloc[0])
    except Exception:
        return "#1f77b4"