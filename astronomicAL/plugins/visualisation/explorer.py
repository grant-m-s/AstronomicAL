from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import panel as pn
import param

from .constants import SETTINGS_HEIGHT
from .density import DensityPanel
from .histogram import HistogramPanel
from .scatter import ScatterPanel
from .utils import _active_dataset_id, _dataset_row_count, ensure_hv_extension
from .widgets import (
    header_select,
    settings_box,
    settings_checkbox,
    settings_float_slider,
    settings_int_input,
    settings_int_slider,
    settings_multichoice,
    settings_select,
)


class LinkedExplorerPanel(param.Parameterized):
    """One compact settings header and multiple linked plots sharing one state."""

    def __init__(self, context, state, **params):
        ensure_hv_extension()
        super().__init__(**params)

        self.context = context
        self.state = state
        self.settings_visible = False
        self._layout: Optional[pn.Column] = None
        self._settings_built = False
        self._disposed = False


        self.scatter = ScatterPanel(
            context=context,
            state=state,
            show_controls=False,
            show_header=False,
        )
        self.histogram = HistogramPanel(
            context=context,
            state=state,
            show_controls=False,
            show_header=False,
        )
        self.density = DensityPanel(
            context=context,
            state=state,
            show_controls=False,
            show_header=False,
        )

        self.status_pane = pn.pane.HTML(
            "",
            width=260,
            height=30,
            margin=(2, 8, 0, 0),
        )

        self.settings_pane = pn.Column(
            sizing_mode="stretch_width",
            height=SETTINGS_HEIGHT,
            min_height=SETTINGS_HEIGHT,
            max_height=SETTINGS_HEIGHT,
            height_policy="fixed",
            visible=False,
            margin=(0, 0, 0, 0),
            styles={
                "height": f"{SETTINGS_HEIGHT}px",
                "min-height": f"{SETTINGS_HEIGHT}px",
                "max-height": f"{SETTINGS_HEIGHT}px",
                "overflow-y": "auto",
                "overflow-x": "hidden",
                "box-sizing": "border-box",
            },
        )

        self.settings_button = pn.widgets.Button(
            name="⚙",
            width=32,
            height=32,
            button_type="light",
            margin=(14, 0, 0, 0),
            sizing_mode="fixed",
        )
        self.settings_button.on_click(self._toggle_settings)

        self.tabs: Optional[pn.Tabs] = None
        self._watchers: List[Tuple[Any, Any]] = []

        for name in [
            "x",
            "y",
            "label_filter",
            "color_by",
            "render_mode",
            "datashade_threshold",
            "interactive_sample_limit",
            "max_selection_ids",
            "log_x",
            "log_y",
            "bins",
            "density_bins",
        ]:
            try:
                watcher = self.state.param.watch(self._update_status, name)
                self._watchers.append((self.state, watcher))
            except Exception:
                pass

    def _ensure_settings_built(self) -> None:
        if self._settings_built:
            return

        self.settings_pane[:] = [self._settings_controls()]
        self._settings_built = True

    def _apply_settings_visibility(self) -> None:
        self._ensure_settings_built()
        self.settings_pane.visible = self.settings_visible
        self.settings_button.button_type = "primary" if self.settings_visible else "light"

    def _toggle_settings(self, _event=None) -> None:
        self.settings_visible = not self.settings_visible
        self._apply_settings_visibility()

    def _update_status(self, _event=None) -> None:
        dataset_id = _active_dataset_id(self.context)
        rows = _dataset_row_count(self.context, dataset_id)

        self.status_pane.object = (
            f"{rows:,} rows · "
            f"X: {self.state.x or '-'} · "
            f"Y: {self.state.y or '-'}"
        )

    def _settings_controls(self):
        return settings_box(
            self.status_pane,
            settings_select(self.state.param.color_by, name="Colour", width=130),
            settings_multichoice(self.state.param.label_filter, name="Labels", width=210),
            settings_select(self.state.param.render_mode, name="Scatter render", width=140),
            settings_int_input(self.state.param.datashade_threshold, name="Shade threshold", width=145),
            settings_int_input(self.state.param.interactive_sample_limit, name="Sample limit", width=130),
            settings_int_input(self.state.param.max_selection_ids, name="Max selected IDs", width=145),
            settings_float_slider(self.state.param.point_size, name="Size", width=175),
            settings_float_slider(self.state.param.point_alpha, name="Alpha", width=175),
            settings_int_slider(self.state.param.bins, name="Hist bins", width=165),
            settings_int_slider(self.state.param.density_bins, name="Density bins", width=175),
            settings_checkbox(self.state.param.log_x, name="Log X"),
            settings_checkbox(self.state.param.log_y, name="Log Y"),
            settings_checkbox(self.state.param.log_density, name="Log density"),
        )

    def _header(self):
        return pn.GridBox(
            header_select(self.state.param.x, name="X"),
            header_select(self.state.param.y, name="Y"),
            self.settings_button,
            ncols=3,
            sizing_mode="stretch_width",
            height=48,
            margin=(0, 0, 0, 0),
            styles={
                "display": "grid",
                "grid-template-columns": "minmax(70px, 1fr) minmax(70px, 1fr) 34px",
                "gap": "4px",
                "align-items": "start",
            },
        )

    def _build_tabs(self):
        if self.tabs is None:
            self.tabs = pn.Tabs(
                ("Scatter", self.scatter.panel()),
                ("Histogram", self.histogram.panel()),
                ("2D Density", self.density.panel()),
                dynamic=True,
                sizing_mode="stretch_both",
                height_policy="max",
                margin=(0, 0, 0, 0),
                styles={
                    "min-height": "0",
                    "overflow": "hidden",
                },
            )

        return self.tabs

    def panel(self):
        self._update_status()
        self._ensure_settings_built()
        self._apply_settings_visibility()

        self._layout = pn.Column(
            self._header(),
            self.settings_pane,
            self._build_tabs(),
            sizing_mode="stretch_both",
            height_policy="max",
            min_height=0,
            margin=(0, 0, 0, 0),
            styles={
                "min-height": "0",
                "overflow": "hidden",
            },
        )

        return self._layout

    def get_state(self) -> Dict[str, Any]:
        state = self.state.get_state()
        state["settings_visible"] = self.settings_visible
        return state

    def restore_state(self, state: Dict[str, Any]) -> None:
        if isinstance(state, dict):
            self.settings_visible = bool(state.get("settings_visible", False))
            self.state.restore_state(state)

            for child in (self.scatter, self.histogram, self.density):
                try:
                    child.refresh()
                except Exception:
                    pass

            self._update_status()
            self._apply_settings_visibility()

    def dispose(self) -> None:
        if getattr(self, "_disposed", False):
            return
        self._disposed = True

        for owner, watcher in list(getattr(self, "_watchers", [])):
            try:
                owner.param.unwatch(watcher)
            except Exception:
                pass
        self._watchers.clear()

        for child in (
            getattr(self, "scatter", None),
            getattr(self, "histogram", None),
            getattr(self, "density", None),
        ):
            if child is None:
                continue
            try:
                child.dispose()
            except Exception:
                pass