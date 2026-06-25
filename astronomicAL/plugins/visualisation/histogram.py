from __future__ import annotations

from dataclasses import replace
from typing import Optional, Tuple

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
    PreparedFrame,
    force_wheel_zoom_hook,
    limited_histogram_hover_tool,
    renderer_name_hook,
    AXIS_KIND_CATEGORICAL,
    _runtime_axis_kind,
    axis_tick_label_hook,
)
from .widgets import (
    settings_box,
    settings_checkbox,
    settings_float_input,
    settings_int_slider,
    settings_multichoice,
    settings_select,
)

HistogramExtent = Tuple[float, float]


class HistogramPanel(BaseVisualisationPanel):
    title = "Histogram"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # BaseVisualisationPanel in the plugin_system branch does not know
        # about the optional x-domain limits introduced here, so the histogram
        # panel watches them locally.
        self._watch_state(["x_min", "x_max"])

    def _settings_controls(self):
        return settings_box(
            self.status_pane,
            settings_select(self.state.param.color_by, name="Colour", width=130),
            settings_multichoice(self.state.param.label_filter, name="Labels", width=210),
            settings_int_slider(self.state.param.bins, name="Bins", width=180),
            settings_float_input(self.state.param.x_min, name="xmin", width=105),
            settings_float_input(self.state.param.x_max, name="xmax", width=105),
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

        try:
            extent = self._histogram_extent(data)
        except ValueError as exc:
            self.plot_pane.object = self._empty(str(exc))
            self.status_pane.object = "Invalid histogram limits"
            return

        data_for_hist = self._clip_to_histogram_extent(data, extent)

        if data_for_hist.empty:
            self.plot_pane.object = self._empty("No finite X data inside limits")
            self.status_pane.object = "0 histogram rows inside limits"
            return

        frame = data_for_hist.frame
        layers = []

        global_x = frame[INTERNAL_X].to_numpy(copy=False)
        global_x = self._valid_histogram_values(global_x)

        if len(global_x) == 0:
            self.plot_pane.object = self._empty("No histogram values inside limits")
            self.status_pane.object = "0 histogram rows inside limits"
            return

        bins = self._bins_for(global_x, extent)

        use_labels = (
            self.state.color_by == "Labels"
            and INTERNAL_LABEL_DISPLAY in frame.columns
            and frame[INTERNAL_LABEL_DISPLAY].nunique(dropna=True) <= 40
        )

        if use_labels:
            for label, sub in frame.groupby(INTERNAL_LABEL_DISPLAY, dropna=False, sort=False):
                colour = _group_colour(sub)
                hist = self._hist(
                    sub[INTERNAL_X].to_numpy(copy=False),
                    str(label),
                    colour,
                    bins=bins,
                )
                if hist is not None:
                    layers.append(hist)
        else:
            hist = self._hist(
                frame[INTERNAL_X].to_numpy(copy=False),
                "All",
                "#1f77b4",
                bins=bins,
            )
            if hist is not None:
                layers.append(hist)

        focus = self._focus_point(data_for_hist)
        if focus is not None:
            x, _ = focus
            try:
                x = float(x)
                if extent[0] <= x <= extent[1] and (not self.state.log_x or x > 0):
                    layers.append(
                        hv.VLine(x).opts(
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
            xlim=extent,
            logx=self.state.log_x,
            logy=self.state.log_y,
            legend_position="right",
            show_grid=True,
            toolbar="right",
            tools=[limited_histogram_hover_tool(), "pan", "wheel_zoom", "box_zoom", "reset"],
            active_tools=["wheel_zoom"],
            hooks=[
                force_wheel_zoom_hook,
                axis_tick_label_hook(self.state),
                   ],
            shared_axes=False,
            axiswise=True,
            framewise=True,
        )

        full_count = len(data.frame)
        clipped_count = len(frame)
        self.status_pane.object = (
            f"{full_count:,} finite rows · {clipped_count:,} inside limits · "
            f"{len(global_x):,} histogram values"
        )

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

    def _valid_histogram_values(self, values: np.ndarray) -> np.ndarray:
        x = np.asarray(values, dtype=float)
        x = x[np.isfinite(x)]
        if self.state.log_x:
            x = x[x > 0]
        return x

    def _histogram_extent(self, data: PreparedFrame) -> HistogramExtent:
        x = self._valid_histogram_values(
            data.frame[INTERNAL_X].to_numpy(copy=False)
        )

        if len(x) == 0:
            if self.state.log_x:
                raise ValueError("No positive finite X data for Log X histogram")

            raise ValueError("No finite X data for histogram")

        xmin = self._state_limit("x_min")
        xmax = self._state_limit("x_max")

        if xmin is None:
            xmin = float(np.nanmin(x))

        if xmax is None:
            xmax = float(np.nanmax(x))

        if not all(np.isfinite([xmin, xmax])):
            raise ValueError("Histogram limits must be finite numbers or empty")

        if _runtime_axis_kind(self.state, self.state.x) == AXIS_KIND_CATEGORICAL:
            return float(np.floor(xmin) - 0.5), float(np.ceil(xmax) + 0.5)

        if xmax <= xmin:
            raise ValueError("xmax must be greater than xmin")

        if self.state.log_x and (xmin <= 0 or xmax <= 0):
            raise ValueError("xmin and xmax must be positive when Log X is enabled")

        return float(xmin), float(xmax)

    def _clip_to_histogram_extent(
        self,
        data: PreparedFrame,
        extent: HistogramExtent,
    ) -> PreparedFrame:
        xmin, xmax = extent
        frame = data.frame
        mask = (frame[INTERNAL_X] >= xmin) & (frame[INTERNAL_X] <= xmax)
        if self.state.log_x:
            mask &= frame[INTERNAL_X] > 0
        clipped = frame.loc[mask]

        return replace(
            data,
            frame=clipped,
            row_count_after_filter=len(clipped),
            sampled_from=None,
        )

    def _hist(self, values: np.ndarray, label: str, colour: str, *, bins):
        x = self._valid_histogram_values(values)
        if len(x) == 0:
            return None

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
            hooks=[
                force_wheel_zoom_hook,
                renderer_name_hook(HIST_RENDERER),
                axis_tick_label_hook(self.state)
            ],
            responsive=True,
            min_height=PLOT_MIN_HEIGHT,
            show_grid=True,
            toolbar="right",
            shared_axes=False,
            axiswise=True,
            framewise=True,
        )

    def _bins_for(self, x: np.ndarray, extent: HistogramExtent):
        xmin, xmax = extent

        if _runtime_axis_kind(self.state, self.state.x) == AXIS_KIND_CATEGORICAL:
            finite = np.asarray(x, dtype=float)
            finite = finite[np.isfinite(finite)]

            if len(finite) == 0:
                return np.asarray([-0.5, 0.5], dtype=float)

            first = int(np.floor(np.nanmin(finite)))
            last = int(np.ceil(np.nanmax(finite)))
            return np.arange(first - 0.5, last + 1.5, 1.0)

        n_bins = int(self.state.bins)

        if self.state.log_x:
            return np.geomspace(xmin, xmax, n_bins + 1)

        return np.linspace(xmin, xmax, n_bins + 1)


def _group_colour(frame) -> str:
    if INTERNAL_LABEL_COLOUR not in frame.columns or frame.empty:
        return "#1f77b4"
    try:
        return str(frame[INTERNAL_LABEL_COLOUR].iloc[0])
    except Exception:
        return "#1f77b4"
