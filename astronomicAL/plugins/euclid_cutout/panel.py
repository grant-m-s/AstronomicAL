from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple
import traceback
import uuid

import numpy as np
import panel as pn
import holoviews as hv

try:
    hv.extension("bokeh")
except Exception:
    pass

from .image_visualization import ImageVisualizationClass
from .service import (
    DEFAULT_EUCLID_FILTERS,
    DEFAULT_SAVE_DIR,
    EuclidCutoutRuntime,
    euclid_user_error_message,
)

from concurrent.futures import CancelledError

PLUGIN_ID = "astro.euclid_cutout"
RUNTIME_SERVICE_KEY = f"{PLUGIN_ID}.runtime"
CUTOUT_ARTIFACT_TYPE = "astro.cutout.euclid"

SETTINGS_HEIGHT = 210

SETTING_INPUT_HEIGHT = 56
SETTING_CHECKBOX_HEIGHT = 24
SETTING_BUTTON_HEIGHT = 32
SETTING_SLIDER_LABEL_HEIGHT = 16
SETTING_SLIDER_HEIGHT = 30
SETTING_SLIDER_BLOCK_HEIGHT = 50

REQUEST_GROUP_HEIGHT = 202
REQUEST_GROUP_HEIGHT_WITH_LOGIN = 380
DISPLAY_GROUP_HEIGHT = 170
CLIP_GROUP_HEIGHT = 310
SETTINGS_CONTENT_HEIGHT = (
    REQUEST_GROUP_HEIGHT
    + DISPLAY_GROUP_HEIGHT
    + CLIP_GROUP_HEIGHT
    + 40
)

HEADER_HEIGHT = 52

AUTO_LOAD_DELAY_MS = 750
FILTER_LOAD_DELAY_MS = 125
SURFACE_MAX_SAMPLES = 56
# Keep the higher-resolution analysis mesh, but draw a lighter display grid.
# This gives the same peak-preserving data in both face modes while avoiding
# the visually dense 56 x 56 Matplotlib wireframe.
SURFACE_DISPLAY_SAMPLES = 24
SURFACE_AZIMUTH_DEG = -48.0
SURFACE_ELEVATION_DEG = 16.0
SURFACE_SMOOTH_PASSES = 2
SURFACE_NOISE_FLOOR_SIGMA = 0.25
SURFACE_HIGH_PERCENTILE = 99.7
SURFACE_HEIGHT_FRACTION = 0.72
SURFACE_CAMERA_ZOOM = 1.38
SURFACE_EDGE_WIDTH = 0.62
SURFACE_AXIS_TICKS = 5
SURFACE_RENDER_WIDTH_PX = 760
SURFACE_RENDER_HEIGHT_PX = 520
# Render the static Matplotlib framebuffer above CSS resolution, then let the
# browser downsample it into the responsive HoloViews pane.  Text and fine grid
# edges are therefore substantially sharper without changing their layout.
SURFACE_RENDER_SCALE = 2.0
SURFACE_MIN_DISPLAY_SAMPLES = 8
# Screen-space drag sensitivity for the static Matplotlib camera.  Dragging
# follows the pointer direction: left rotates left and up lowers the camera.
# During a gesture a 1x framebuffer is rendered into the existing Bokeh image
# source; release produces the normal 2x high-DPI frame.
SURFACE_DRAG_AZIMUTH_DEG_PER_PX = 0.25
SURFACE_DRAG_ELEVATION_DEG_PER_PX = 0.18
SURFACE_DRAG_PREVIEW_SCALE = 1.0
SURFACE_DRAG_PREVIEW_DELAY_MS = 20

ANALYSIS_TABS_STYLESHEET = """
:host {
  position: relative;
  overflow: hidden;
}
:host .bk-tabs-header,
.bk-tabs-header {
  position: relative;
  z-index: 20;
  pointer-events: auto !important;
}
:host .bk-tabs-header .bk-headers-wrapper,
:host .bk-tabs-header .bk-headers,
.bk-tabs-header .bk-headers-wrapper,
.bk-tabs-header .bk-headers {
  pointer-events: auto !important;
}
:host .bk-tabs-header .bk-tab,
.bk-tabs-header .bk-tab {
  min-height: 30px;
  padding: 6px 10px;
  display: flex;
  align-items: center;
  cursor: pointer !important;
  pointer-events: auto !important;
  touch-action: manipulation;
  position: relative;
  z-index: 21;
}
:host .bk-tabs-header .bk-tab > *,
.bk-tabs-header .bk-tab > * {
  pointer-events: none !important;
}
"""

def _first_not_none(*values: Any) -> Any:
    return next((value for value in values if value is not None), None)

def _settings_overlay_styles(open_settings: bool) -> Dict[str, str]:
    styles = {
        "position": "absolute",
        "top": "0px",
        "left": "6px",
        "right": "6px",
        "height": f"{SETTINGS_HEIGHT}px",
        "max-height": f"{SETTINGS_HEIGHT}px",
        "overflow-y": "auto",
        "overflow-x": "hidden",
        "box-sizing": "border-box",
        "padding": "0",
        "margin": "0",
        "background": "#FAFBFC",
        "border": "1px solid #DFE1E6",
        "border-radius": "6px",
        "box-shadow": "0 6px 18px rgba(9, 30, 66, 0.18)",
        "z-index": "30",
        "transition": "none",
    }

    if open_settings:
        styles.update(
            {
                "display": "block",
                "opacity": "1",
                "visibility": "visible",
                "pointer-events": "auto",
                "transform": "translateY(0)",
            }
        )
    else:
        styles.update(
            {
                "display": "none",
                "opacity": "0",
                "visibility": "hidden",
                "pointer-events": "none",
                "transform": "translateY(-4px)",
            }
        )

    return styles

def _set_fixed_height(widget: Any, height: int) -> None:
    try:
        widget.height = int(height)
        widget.min_height = int(height)
        widget.max_height = int(height)
        widget.height_policy = "fixed"
    except Exception:
        pass

def _compact_input(
    widget: Any,
    *,
    width: int = 120,
    margin: Tuple[int, int, int, int] = (0, 6, 0, 0),
) -> Any:
    try:
        widget.width = width
        widget.sizing_mode = "fixed"
        widget.margin = margin
        _set_fixed_height(widget, SETTING_INPUT_HEIGHT)
    except Exception:
        pass
    return widget

def _wide_input(
    widget: Any,
    *,
    margin: Tuple[int, int, int, int] = (0, 0, 0, 0),
) -> Any:
    try:
        widget.sizing_mode = "stretch_width"
        widget.margin = margin
        _set_fixed_height(widget, SETTING_INPUT_HEIGHT)
    except Exception:
        pass
    return widget

def _compact_checkbox(
    widget: Any,
    *,
    width: int = 140,
    margin: Tuple[int, int, int, int] = (0, 8, 0, 0),
) -> Any:
    try:
        widget.width = width
        widget.sizing_mode = "fixed"
        widget.margin = margin
        _set_fixed_height(widget, SETTING_CHECKBOX_HEIGHT)
    except Exception:
        pass
    return widget

def _compact_button(
    widget: Any,
    *,
    width: int = 118,
    margin: Tuple[int, int, int, int] = (0, 6, 0, 0),
) -> Any:
    try:
        widget.width = width
        widget.sizing_mode = "fixed"
        widget.margin = margin
        _set_fixed_height(widget, SETTING_BUTTON_HEIGHT)
    except Exception:
        pass
    return widget

def _section_title(text: str) -> pn.pane.HTML:
    return pn.pane.HTML(
        f"""
        <div style="
            font-size:11px;
            font-weight:700;
            text-transform:uppercase;
            letter-spacing:0.04em;
            color:#42526E;
            line-height:14px;
            white-space:nowrap;
        ">{text}</div>
        """,
        sizing_mode="stretch_width",
        height=18,
        min_height=18,
        max_height=18,
        margin=(0, 0, 4, 0),
    )

def _settings_row(
    *children: Any,
    height: int,
    margin: Tuple[int, int, int, int] = (0, 0, 6, 0),
) -> pn.Row:
    return pn.Row(
        *children,
        sizing_mode="stretch_width",
        height=height,
        min_height=height,
        max_height=height,
        height_policy="fixed",
        margin=margin,
        styles={
            "box-sizing": "border-box",
            "overflow": "hidden",
        },
    )

def _settings_group(title: str, *children: Any, height: int) -> pn.Column:
    return pn.Column(
        _section_title(title),
        *children,
        sizing_mode="stretch_width",
        height=height,
        min_height=height,
        max_height=height,
        height_policy="fixed",
        margin=(0, 0, 8, 0),
        styles={
            "border": "1px solid #DFE1E6",
            "border-radius": "6px",
            "background": "#FFFFFF",
            "padding": "8px",
            "box-sizing": "border-box",
            "overflow": "hidden",
        },
    )

def _slider_block(label: str, widget: Any) -> pn.Column:
    # Use our own compact label. Bokeh RangeSlider's built-in label consumes
    # too much vertical space in small plugin panels.
    try:
        widget.name = ""
        widget.sizing_mode = "stretch_width"
        widget.margin = (0, 0, 0, 0)
        _set_fixed_height(widget, SETTING_SLIDER_HEIGHT)
    except Exception:
        pass

    label_pane = pn.pane.HTML(
        f"""
        <div style="
            font-size:12px;
            line-height:14px;
            color:#172B4D;
            white-space:nowrap;
        ">{label}: <b>0 .. 1</b></div>
        """,
        sizing_mode="stretch_width",
        height=SETTING_SLIDER_LABEL_HEIGHT,
        min_height=SETTING_SLIDER_LABEL_HEIGHT,
        max_height=SETTING_SLIDER_LABEL_HEIGHT,
        margin=(0, 0, 0, 0),
    )

    return pn.Column(
        label_pane,
        widget,
        sizing_mode="stretch_width",
        height=SETTING_SLIDER_BLOCK_HEIGHT,
        min_height=SETTING_SLIDER_BLOCK_HEIGHT,
        max_height=SETTING_SLIDER_BLOCK_HEIGHT,
        height_policy="fixed",
        margin=(0, 0, 4, 0),
        styles={
            "box-sizing": "border-box",
            "overflow": "hidden",
        },
    )

def _fallback_spectrum_colour(index: int) -> str:
    colours = [
        "#e41a1c",
        "#377eb8",
        "#4daf4a",
        "#984ea3",
        "#ff7f00",
        "#ffff33",
        "#a65628",
        "#f781bf",
        "#999999",
    ]
    return colours[int(index) % len(colours)]

@dataclass
class _ResolvedTarget:
    dataset_id: str
    row_id: Optional[str]
    row: Dict[str, Any]
    ra: float
    dec: float
    ra_column: str
    dec_column: str
    id_column: Optional[str]

class EuclidCutoutPanel:
    """Plugin-native Euclid cutout panel with separated image processing.

    Runtime archive access is handled by ``EuclidCutoutRuntime`` and returns raw
    image arrays + WCS. This controller creates an ``ImageVisualizationClass``
    from those raw products and uses it for display preparation.
    """

    def __init__(
        self,
        *,
        context: Any,
        data: Any = None,
        state: Optional[Dict[str, Any]] = None,
        **_: Any,
    ) -> None:
        self.context = context
        self.data = data
        self.panel_id = f"{PLUGIN_ID}.{uuid.uuid4().hex}"

        self._subscriptions: List[Any] = []
        self._job_handle: Any = None
        self._job_generation: Optional[int] = None
        self._request_generation = 0
        self._disposed = False
        self._initial_load_started = False
        self._settings_built = False
        self.settings_visible = False
        self._restored_active_tab = 0
        self._profile_pixel: Optional[Tuple[int, int]] = None
        self._restore_profile_pixel_pending = False
        self._profile_initialised = False
        self._surface_dirty = True
        self._surface_generation = 0
        self._cutout_tap_stream: Any = None
        self._cutout_tap_watcher: Any = None
        self._analysis_tab_watcher: Any = None

        self._runtime_service: Optional[EuclidCutoutRuntime] = None
        self._owns_runtime_service = False
        self._panel_storage: Any = None
        self._suppress_filter_reload = False

        # The platform may restore controller state immediately after the panel
        # factory has returned its view. Keep request-setting identity explicit so
        # an initial request started with constructor defaults can never be accepted
        # after restored widget values have become visible.
        self._active_request_signature: Optional[Tuple[Any, ...]] = None

        self._current_target: Optional[_ResolvedTarget] = None
        self._cutout_result: Any = None
        self.euclid_object: Any = None
        self.image_container: Optional[ImageVisualizationClass] = None

        self.overplotted_coordinates: List[Any] = []
        self.stored_spectrum_coordinates: Dict[
            str,
            Dict[str, Iterable[float]],
        ] = {}

        self.image_width = 1
        self.image_height = 1
        self.bar_length_pixels = 1
        self.euclid_fig: List[Any] = []
        self._base_cutout_elements: List[Any] = []
        self._analysis_cache_key: Optional[Tuple[Any, ...]] = None
        self._analysis_cache_data: Optional[np.ndarray] = None
        self._surface_cache_key: Optional[Tuple[Any, ...]] = None
        self._surface_cache_value: Optional[Tuple[np.ndarray, Tuple[int, int]]] = None
        self._surface_render_cache: Dict[Tuple[Any, ...], Any] = {}
        self._surface_control_sync = False
        self._surface_drag_active = False
        self._surface_drag_start_sx: Optional[float] = None
        self._surface_drag_start_sy: Optional[float] = None
        self._surface_drag_start_elevation = float(SURFACE_ELEVATION_DEG)
        self._surface_drag_start_azimuth = float(SURFACE_AZIMUTH_DEG)
        self._surface_drag_elevation = float(SURFACE_ELEVATION_DEG)
        self._surface_drag_azimuth = float(SURFACE_AZIMUTH_DEG)
        self._surface_bound_plot_ids: set[int] = set()
        self._surface_live_source: Any = None
        self._surface_live_image_key: Optional[str] = None
        self._surface_live_flip_y = False
        self._surface_live_preview_scheduled = False
        self._surface_last_rgba: Optional[np.ndarray] = None
        self._tap_update_generation = 0

        self._auto_load_generation = 0
        self._target_status_scheduled = False

        runtime = self._runtime()
        self._panel_storage = runtime.acquire_panel_storage(self.panel_id)

        self._build_widgets()
        if state:
            self.restore_state(state)
        self._build_layout()
        self._bind_events()

    # ------------------------------------------------------------------
    # Public plugin-controller API
    # ------------------------------------------------------------------
    def view(self) -> pn.viewable.Viewable:
        self._schedule_initial_load()
        return self.layout

    def panel(self) -> pn.viewable.Viewable:
        return self.view()

    def dispose(self) -> None:
        if self._disposed:
            return

        self._disposed = True
        self._request_generation += 1
        self._auto_load_generation += 1
        self._surface_generation += 1
        self._surface_drag_active = False
        self._surface_bound_plot_ids.clear()
        self._surface_live_source = None
        self._surface_live_image_key = None
        self._surface_live_preview_scheduled = False
        self._surface_last_rgba = None
        self._tap_update_generation += 1
        self._cancel_job(reason="panel.disposed", publish=False)
        self._clear_cutout_tap_stream()
        if self._analysis_tab_watcher is not None and hasattr(self, "analysis_tabs"):
            try:
                self.analysis_tabs.param.unwatch(self._analysis_tab_watcher)
            except Exception:
                pass
            self._analysis_tab_watcher = None

        result = self._cutout_result
        self._cutout_result = None
        self.euclid_object = None
        self.image_container = None
        self._cleanup_result(result)

        runtime = self._runtime_service
        lease = self._panel_storage
        self._panel_storage = None

        if runtime is not None:
            runtime.release_panel_storage(lease)
        elif lease is not None:
            lease.release()

        if self._owns_runtime_service and runtime is not None:
            runtime.dispose()

        events = getattr(self.context, "events", None)
        if events is not None:
            for subscription in list(self._subscriptions):
                try:
                    events.unsubscribe(subscription)
                except Exception:
                    pass

        self._subscriptions.clear()

    def snapshot_state(self) -> Dict[str, Any]:
        return {
            "settings_visible": self.settings_visible,
            "environment": self.environment.value,
            "radius_arcsec": self.radius_input.value,
            "filter_name": self.filter_input.value,
            "stretch": self.stretch_input.value,
            "stretch_scale": self.stretch_scale_input.value,
            "show_source_coords": self.show_source_coords.value,
            "show_scale": self.show_scale.value,
            "show_spectrum_coords": self.show_spectrum_coords.value,
            "contour_levels": self.contour_levels.value,
            "contour_base": self.contour_base.value,
            "contour_exponent": self.contour_exponent.value,
            "auto_reload": self.auto_reload.value,
            "save_dir": self.save_dir_input.value,
            "credentials_filepath": self.credentials_file_input.value,
            "show_profile_crosshair": bool(self.show_profile_crosshair.value),
            "surface_black_cells": bool(self.surface_black_cells.value),
            "surface_elevation_deg": int(self.surface_elevation_input.value),
            "surface_azimuth_deg": int(self.surface_azimuth_input.value),
            "surface_grid_size": int(self.surface_grid_size_input.value),
            "active_view_tab": int(
                getattr(getattr(self, "analysis_tabs", None), "active", self._restored_active_tab)
                or 0
            ),
            "profile_pixel": (
                None
                if self._profile_pixel is None
                else {"row": int(self._profile_pixel[0]), "col": int(self._profile_pixel[1])}
            ),
        }

    def _request_settings_signature(self) -> Tuple[Any, ...]:
        """Return the UI-thread identity of settings that affect archive retrieval.

        Display-only controls are intentionally excluded. The tuple is kept
        private to this controller and is never placed in persisted state or
        published through the event bus.
        """
        try:
            radius_arcsec: Any = float(self.radius_input.value)
        except Exception:
            radius_arcsec = self.radius_input.value

        return (
            str(self.environment.value or "PDR"),
            radius_arcsec,
            str(self.filter_input.value or "Color"),
            str(self.save_dir_input.value or DEFAULT_SAVE_DIR),
            str(self.credentials_file_input.value or "").strip(),
            str(self.user_input.value or "").strip(),
            str(self.password_input.value or ""),
        )

    def _request_settings_changed(
        self,
        request_signature: Optional[Tuple[Any, ...]],
    ) -> bool:
        if request_signature is None:
            return False
        try:
            return tuple(request_signature) != self._request_settings_signature()
        except Exception:
            return False

    def _reload_after_request_settings_change(
        self,
        *,
        reason: str,
    ) -> None:
        """Reload or leave a clear prompt after a request-setting change."""
        if self._disposed:
            return

        if self.auto_reload.value:
            self.status.object = (
                "Euclid request settings changed; loading the current values…"
            )
            self._schedule_auto_load(
                reason=reason,
                delay_ms=0,
            )
        else:
            self.status.object = (
                "Euclid request settings changed. Press **Load** to use the "
                "current values."
            )

    def restore_state(self, state: Dict[str, Any]) -> None:
        if not isinstance(state, dict):
            return

        request_signature_before = self._request_settings_signature()

        self.settings_visible = bool(state.get("settings_visible", False))
        try:
            self._restored_active_tab = max(
                0,
                min(2, int(state.get("active_view_tab", 0))),
            )
        except Exception:
            self._restored_active_tab = 0

        profile_pixel = state.get("profile_pixel")
        if isinstance(profile_pixel, dict):
            try:
                self._profile_pixel = (
                    int(profile_pixel["row"]),
                    int(profile_pixel["col"]),
                )
                self._restore_profile_pixel_pending = True
            except Exception:
                self._profile_pixel = None
                self._restore_profile_pixel_pending = False

        mapping = {
            "environment": self.environment,
            "radius_arcsec": self.radius_input,
            "filter_name": self.filter_input,
            "stretch": self.stretch_input,
            "stretch_scale": self.stretch_scale_input,
            "show_source_coords": self.show_source_coords,
            "show_scale": self.show_scale,
            "show_spectrum_coords": self.show_spectrum_coords,
            "contour_levels": self.contour_levels,
            "contour_base": self.contour_base,
            "contour_exponent": self.contour_exponent,
            "auto_reload": self.auto_reload,
            "save_dir": self.save_dir_input,
            "credentials_filepath": self.credentials_file_input,
            "show_profile_crosshair": self.show_profile_crosshair,
            "surface_black_cells": self.surface_black_cells,
            "surface_elevation_deg": self.surface_elevation_input,
            "surface_azimuth_deg": self.surface_azimuth_input,
            "surface_grid_size": self.surface_grid_size_input,
        }

        self._suppress_filter_reload = True
        try:
            for key, widget in mapping.items():
                if key not in state:
                    continue
                try:
                    widget.value = state[key]
                except Exception:
                    pass
        finally:
            self._suppress_filter_reload = False

        try:
            self._apply_settings_visibility()
        except Exception:
            pass

        request_settings_changed = (
            request_signature_before != self._request_settings_signature()
        )

        # A normal constructor-time restore happens before view(), so no load has
        # been scheduled and no further work is required. The platform can also
        # restore state after the factory has returned the view; in that case an
        # initial request may already have captured the default radius. Invalidate
        # it and schedule one request from the restored widget values.
        if request_settings_changed and self._initial_load_started:
            self._invalidate_request(
                reason="state.restored",
                target=self._current_target,
                publish=True,
            )
            self._active_request_signature = None
            self._reload_after_request_settings_change(
                reason="state.restored",
            )

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_widgets(self) -> None:
        self.status = pn.pane.Markdown(
            "",
            sizing_mode="stretch_width",
            height_policy="fit",
            margin=(0, 8, 0, 8),
            styles={
                "font-size": "12px",
                "line-height": "1.25",
                "max-height": "32px",
                "overflow": "auto",
            },
        )
        self.target_status = pn.pane.HTML(
            "",
            sizing_mode="stretch_width",
            height=22,
            margin=(0, 8, 0, 8),
            styles={
                "font-size": "12px",
                "line-height": "1.2",
                "overflow": "hidden",
                "white-space": "nowrap",
                "text-overflow": "ellipsis",
                "color": "#333",
            },
        )
        self.figure = pn.pane.HoloViews(
            self._empty_image(),
            sizing_mode="stretch_both",
            min_height=320,
            margin=(0, 6, 6, 6),
        )
        self.profile_note = pn.pane.Markdown(
            "Select a scalar filter or use VIS for the colour composite, then click the cutout to inspect a pixel.",
            sizing_mode="stretch_width",
            height=32,
            margin=(4, 8, 0, 8),
            styles={"font-size": "12px", "line-height": "1.25"},
        )
        self.profile_figure = pn.pane.HoloViews(
            self._empty_profile(),
            sizing_mode="stretch_both",
            min_height=300,
            margin=(0, 6, 6, 6),
        )
        self.show_profile_crosshair = pn.widgets.Checkbox(
            name="Show cutout crosshair",
            value=False,
            width=160,
            height=28,
            sizing_mode="fixed",
            margin=(0, 0, 0, 0),
        )
        self.surface_note = pn.pane.Markdown(
            "VIS · bounded background-subtracted raw-intensity mesh",
            sizing_mode="stretch_width",
            height=24,
            min_height=24,
            max_height=24,
            margin=(2, 6, 0, 8),
            styles={
                "font-size": "12px",
                "line-height": "1.2",
                "white-space": "nowrap",
                "overflow": "hidden",
                "text-overflow": "ellipsis",
            },
        )
        self.surface_black_cells = pn.widgets.Checkbox(
            name="Black cells",
            value=True,
            width=94,
            height=28,
            sizing_mode="fixed",
            margin=(0, 10, 0, 0),
        )
        self.surface_elevation_label = pn.pane.HTML(
            "<span>Elev°</span>",
            width=36,
            height=26,
            sizing_mode="fixed",
            margin=(4, 2, 0, 0),
            styles={"font-size": "11px", "line-height": "1.1"},
        )
        self.surface_elevation_input = pn.widgets.IntInput(
            name="",
            value=int(SURFACE_ELEVATION_DEG),
            start=1,
            end=89,
            step=1,
            width=62,
            height=30,
            sizing_mode="fixed",
            margin=(0, 10, 0, 0),
        )
        self.surface_azimuth_label = pn.pane.HTML(
            "<span>Azim°</span>",
            width=40,
            height=26,
            sizing_mode="fixed",
            margin=(4, 2, 0, 0),
            styles={"font-size": "11px", "line-height": "1.1"},
        )
        self.surface_azimuth_input = pn.widgets.IntInput(
            name="",
            value=int(SURFACE_AZIMUTH_DEG),
            start=-180,
            end=180,
            step=5,
            width=66,
            height=30,
            sizing_mode="fixed",
            margin=(0, 10, 0, 0),
        )
        self.surface_grid_label = pn.pane.HTML(
            "<span>Grid</span>",
            width=30,
            height=26,
            sizing_mode="fixed",
            margin=(4, 2, 0, 0),
            styles={"font-size": "11px", "line-height": "1.1"},
        )
        self.surface_grid_size_input = pn.widgets.IntInput(
            name="",
            value=int(SURFACE_DISPLAY_SAMPLES),
            start=SURFACE_MIN_DISPLAY_SAMPLES,
            end=SURFACE_MAX_SAMPLES,
            step=2,
            width=60,
            height=30,
            sizing_mode="fixed",
            margin=(0, 0, 0, 0),
        )
        self.surface_figure = pn.pane.HoloViews(
            self._empty_surface(),
            sizing_mode="stretch_both",
            min_height=300,
            margin=(0, 6, 4, 6),
        )

        self.filter_input = pn.widgets.Select(
            name="Filter",
            options=["Color", *DEFAULT_EUCLID_FILTERS],
            value="Color",
            width=116,
            height=42,
            sizing_mode="fixed",
            margin=(0, 2, 0, 0),
        )

        self.radius_input = pn.widgets.FloatInput(
            name="Radius [arcsec]",
            value=5.0,
            start=0.1,
            step=0.5,
            width=100,
            height=42,
            sizing_mode="fixed",
            margin=(0, 2, 0, 0),
        )

        self.stretch_input = pn.widgets.Select(
            name="Stretch",
            options=["Linear", "Sqrt", "Log", "Asinh", "PowerLaw"],
            value="Linear",
            width=116,
            height=42,
            sizing_mode="fixed",
            margin=(0, 2, 0, 0),
        )

        self.load_button = pn.widgets.Button(
            name="Load",
            button_type="primary",
            width=66,
            height=32,
            sizing_mode="fixed",
            margin=(14, 0, 0, 0),
        )

        self.settings_button = pn.widgets.Button(
            name="⚙",
            width=32,
            height=32,
            button_type="default",
            sizing_mode="fixed",
            margin=(14, 0, 0, 0),
        )

        self.environment = _compact_input(
            pn.widgets.Select(
                name="Environment",
                options=["PDR", "IDR", "OTF", "REG"],
                value="PDR",
            ),
            width=104,
        )

        self.user_input = _wide_input(
            pn.widgets.TextInput(name="Euclid username"),
        )

        self.password_input = _wide_input(
            pn.widgets.PasswordInput(name="Euclid password"),
        )

        self.credentials_file_input = _wide_input(
            pn.widgets.TextInput(
                name="Credentials file",
                value="euclid_credentials.login",
            ),
        )

        self.login_column = pn.Column(
            self.user_input,
            self.password_input,
            self.credentials_file_input,
            sizing_mode="stretch_width",
            visible=False,
            margin=(2, 0, 0, 0),
            styles={
                "gap": "2px",
                "box-sizing": "border-box",
            },
        )

        self.stretch_scale_input = _compact_input(
            pn.widgets.FloatInput(
                name="Stretch scale",
                value=None,
                step=0.1,
                placeholder="default",
            ),
            width=118,
        )

        self.save_dir_input = _wide_input(
            pn.widgets.TextInput(
                name="FITS storage directory",
                value=DEFAULT_SAVE_DIR,
            )
        )

        self.show_source_coords = _compact_checkbox(
            pn.widgets.Checkbox(name="Source marker", value=True),
            width=126,
        )

        self.show_scale = _compact_checkbox(
            pn.widgets.Checkbox(name="Scale bar", value=True),
            width=96,
        )

        self.show_spectrum_coords = _compact_checkbox(
            pn.widgets.Checkbox(name="Spectrum markers", value=True),
            width=142,
        )

        self.auto_reload = _compact_checkbox(
            pn.widgets.Checkbox(name="Auto reload", value=True),
            width=112,
        )

        self.contour_levels = _compact_input(
            pn.widgets.IntInput(
                name="Contour levels",
                value=0,
                start=0,
                end=20,
            ),
            width=112,
        )

        self.contour_base = _compact_input(
            pn.widgets.FloatInput(
                name="Contour base",
                value=2.0,
                start=1.01,
                step=0.5,
            ),
            width=112,
        )

        self.contour_exponent = _compact_input(
            pn.widgets.FloatInput(
                name="Contour exponent",
                value=1.0,
                start=0.1,
                step=0.1,
            ),
            width=128,
        )

        self.clip_slider = pn.widgets.RangeSlider(
            name="",
            start=0,
            end=1,
            step=0.01,
            value=(0, 1),
        )

        self.rgb_clip_r = pn.widgets.RangeSlider(
            name="",
            start=0,
            end=1,
            step=0.004,
            value=(0, 1),
            bar_color="red",
        )

        self.rgb_clip_g = pn.widgets.RangeSlider(
            name="",
            start=0,
            end=1,
            step=0.004,
            value=(0, 1),
            bar_color="green",
        )

        self.rgb_clip_b = pn.widgets.RangeSlider(
            name="",
            start=0,
            end=1,
            step=0.004,
            value=(0, 1),
            bar_color="blue",
        )

        self.gamma_r = _compact_input(
            pn.widgets.FloatInput(name="Gamma R", value=1.0, start=0.05, step=0.1),
            width=86,
        )

        self.gamma_g = _compact_input(
            pn.widgets.FloatInput(name="Gamma G", value=1.0, start=0.05, step=0.1),
            width=86,
        )

        self.gamma_b = _compact_input(
            pn.widgets.FloatInput(name="Gamma B", value=1.0, start=0.05, step=0.1),
            width=86,
        )

        self.refresh_button = _compact_button(
            pn.widgets.Button(
                name="Refresh display",
                button_type="default",
            ),
            width=118,
        )

        self.clean_jobs_button = _compact_button(
            pn.widgets.Button(
                name="Clean async jobs",
                button_type="default",
            ),
            width=124,
        )

        self.load_button.on_click(lambda _event: self.load_cutout(reason="button.load"))
        self.settings_button.on_click(self._toggle_settings)
        self.refresh_button.on_click(lambda _event: self._refresh_display())
        self.clean_jobs_button.on_click(lambda _event: self._clean_async_jobs())
        self.environment.param.watch(self._environment_changed, "value")
        self.filter_input.param.watch(self._filter_changed, "value")

        for widget in [
            self.clip_slider,
            self.stretch_input,
            self.stretch_scale_input,
        ]:
            widget.param.watch(self._analysis_setting_changed, "value")

        for widget in [
            self.rgb_clip_r,
            self.rgb_clip_g,
            self.rgb_clip_b,
            self.gamma_r,
            self.gamma_g,
            self.gamma_b,
        ]:
            widget.param.watch(self._colour_setting_changed, "value")

        for widget in [
            self.show_source_coords,
            self.show_scale,
            self.show_spectrum_coords,
            self.contour_levels,
            self.contour_base,
            self.contour_exponent,
            self.stretch_scale_input,
        ]:
            widget.param.watch(self._overlay_setting_changed, "value")

        self.show_profile_crosshair.param.watch(
            self._profile_crosshair_visibility_changed,
            "value",
        )

        for widget in [
            self.surface_black_cells,
            self.surface_elevation_input,
            self.surface_azimuth_input,
            self.surface_grid_size_input,
        ]:
            widget.param.watch(
                self._surface_style_changed,
                "value",
            )

        self.stretch_input.param.watch(self._update_stretch_scale, "value")

    def _header(self) -> pn.Row:
        return pn.Row(
            self.filter_input,
            self.radius_input,
            self.stretch_input,
            self.load_button,
            self.settings_button,
            sizing_mode="stretch_width",
            height=HEADER_HEIGHT,
            min_height=HEADER_HEIGHT,
            max_height=HEADER_HEIGHT,
            height_policy="fixed",
            margin=(0, 6, 0, 6),
            styles={
                "height": f"{HEADER_HEIGHT}px",
                "min-height": f"{HEADER_HEIGHT}px",
                "max-height": f"{HEADER_HEIGHT}px",
                "overflow": "hidden",
                "box-sizing": "border-box",
                "padding-top": "2px",
                "background": "#ffffff",
                "z-index": "2",
            },
        )

    def _settings_controls(self) -> pn.Column:
        self.request_group = _settings_group(
            "Request",
            _settings_row(
                self.environment,
                self.stretch_scale_input,
                height=SETTING_INPUT_HEIGHT,
            ),
            _settings_row(
                self.save_dir_input,
                height=SETTING_INPUT_HEIGHT,
            ),
            _settings_row(
                self.refresh_button,
                self.clean_jobs_button,
                height=SETTING_BUTTON_HEIGHT,
                margin=(0, 0, 0, 0),
            ),
            self.login_column,
            height=(
                REQUEST_GROUP_HEIGHT_WITH_LOGIN
                if bool(getattr(self.login_column, "visible", False))
                else REQUEST_GROUP_HEIGHT
            ),
        )

        self.display_group = _settings_group(
            "Display",
            _settings_row(
                self.show_source_coords,
                self.show_scale,
                height=SETTING_CHECKBOX_HEIGHT,
                margin=(0, 0, 2, 0),
            ),
            _settings_row(
                self.show_spectrum_coords,
                self.auto_reload,
                height=SETTING_CHECKBOX_HEIGHT,
                margin=(0, 0, 8, 0),
            ),
            _settings_row(
                self.contour_levels,
                self.contour_base,
                self.contour_exponent,
                height=SETTING_INPUT_HEIGHT,
                margin=(0, 0, 0, 0),
            ),
            height=DISPLAY_GROUP_HEIGHT,
        )

        self.clip_group = _settings_group(
            "Colour / clip",
            _slider_block("Clip", self.clip_slider),
            _slider_block("Red clip", self.rgb_clip_r),
            _slider_block("Green clip", self.rgb_clip_g),
            _slider_block("Blue clip", self.rgb_clip_b),
            _settings_row(
                self.gamma_r,
                self.gamma_g,
                self.gamma_b,
                height=SETTING_INPUT_HEIGHT,
                margin=(0, 0, 0, 0),
            ),
            height=CLIP_GROUP_HEIGHT,
        )

        return pn.Tabs(
            ("Request", self.request_group),
            ("Display", self.display_group),
            ("Colour", self.clip_group),
            sizing_mode="stretch_width",
            height=SETTINGS_CONTENT_HEIGHT,
            dynamic=False,
            margin=(0, 0, 0, 0),
        )

    def _ensure_settings_built(self) -> None:
        if self._settings_built:
            return

        self._settings_view = self._settings_controls()
        self.settings_pane.objects = [self._settings_view]
        self._settings_built = True

    def _apply_settings_visibility(self) -> None:
        self._ensure_settings_built()

        open_settings = bool(self.settings_visible)
        self.settings_pane.styles = _settings_overlay_styles(open_settings)
        # A transparent absolute pane can still intercept pointer events in some
        # Panel/Bokeh combinations. Remove it from the rendered layout entirely
        # while closed instead of relying only on opacity/pointer-events CSS.
        self.settings_pane.visible = open_settings
        self._settings_view.visible = open_settings

        try:
            self.settings_button.button_type = (
                "primary" if self.settings_visible else "default"
            )
        except Exception:
            pass

    def _toggle_settings(self, _event: Any = None) -> None:
        self.settings_visible = not self.settings_visible
        self._apply_settings_visibility()

    def _build_layout(self) -> None:
        self.header = self._header()

        self.settings_pane = pn.Column(
            sizing_mode="stretch_width",
            height=SETTINGS_HEIGHT,
            min_height=SETTINGS_HEIGHT,
            max_height=SETTINGS_HEIGHT,
            height_policy="fixed",
            visible=False,
            margin=(0, 0, 0, 0),
            styles=_settings_overlay_styles(False),
        )

        self._settings_view = None
        self._ensure_settings_built()
        self._apply_settings_visibility()

        self.settings_overlay_host = pn.Column(
            self.settings_pane,
            sizing_mode="stretch_width",
            height=0,
            min_height=0,
            max_height=0,
            height_policy="fixed",
            visible=True,
            margin=(0, 0, 0, 0),
            styles={
                "position": "relative",
                "height": "0px",
                "min-height": "0px",
                "max-height": "0px",
                "overflow": "visible",
                "z-index": "30",
                "box-sizing": "border-box",
                "padding": "0",
                "margin": "0",
            },
        )

        self.profile_controls = pn.Row(
            self.show_profile_crosshair,
            sizing_mode="stretch_width",
            height=28,
            min_height=28,
            max_height=28,
            margin=(2, 8, 0, 8),
            styles={"align-items": "center"},
        )
        self.profile_view = pn.Column(
            self.profile_controls,
            self.profile_note,
            self.profile_figure,
            sizing_mode="stretch_both",
            min_height=0,
            margin=(0, 0, 0, 0),
        )
        self.surface_controls = pn.Row(
            self.surface_black_cells,
            self.surface_elevation_label,
            self.surface_elevation_input,
            self.surface_azimuth_label,
            self.surface_azimuth_input,
            self.surface_grid_label,
            self.surface_grid_size_input,
            sizing_mode="stretch_width",
            height=32,
            min_height=32,
            max_height=32,
            margin=(0, 8, 2, 8),
            styles={
                "align-items": "center",
                "overflow": "hidden",
                "white-space": "nowrap",
            },
        )
        self.surface_header = pn.Column(
            self.surface_note,
            self.surface_controls,
            sizing_mode="stretch_width",
            height=58,
            min_height=58,
            max_height=58,
            margin=(0, 0, 0, 0),
            styles={"overflow": "hidden"},
        )
        self.surface_view = pn.Column(
            self.surface_header,
            self.surface_figure,
            sizing_mode="stretch_both",
            min_height=0,
            margin=(0, 0, 0, 0),
        )
        self.analysis_tabs = pn.Tabs(
            ("Cutout", self.figure),
            ("Light profile", self.profile_view),
            ("Surface", self.surface_view),
            active=self._restored_active_tab,
            dynamic=False,
            sizing_mode="stretch_both",
            min_height=320,
            margin=(0, 0, 0, 0),
            styles={
                "position": "relative",
                "overflow": "hidden",
            },
            stylesheets=[ANALYSIS_TABS_STYLESHEET],
        )
        self._analysis_tab_watcher = self.analysis_tabs.param.watch(
            self._analysis_tab_changed,
            "active",
        )

        self.body = pn.Column(
            self.settings_overlay_host,
            self.target_status,
            self.status,
            self.analysis_tabs,
            sizing_mode="stretch_both",
            height_policy="max",
            min_height=0,
            margin=(0, 0, 0, 0),
            styles={
                "position": "relative",
                "min-height": "0",
                "overflow": "hidden",
                "box-sizing": "border-box",
            },
        )

        self.layout = pn.Column(
            self.header,
            self.body,
            sizing_mode="stretch_both",
            height_policy="max",
            min_height=0,
            margin=(0, 0, 0, 0),
            styles={
                "min-height": "0",
                "overflow": "hidden",
                "box-sizing": "border-box",
            },
        )

    def _analysis_tab_changed(self, event: Any) -> None:
        if self._disposed:
            return
        try:
            active = int(event.new)
        except Exception:
            active = 0
        self._restored_active_tab = max(0, min(2, active))
        if active == 1:
            self._refresh_profile()
        elif active == 2:
            self._schedule_surface_refresh()

    def _analysis_setting_changed(self, _event: Any) -> None:
        self._refresh_display()

    def _colour_setting_changed(self, _event: Any) -> None:
        if self.image_container is None or self.filter_input.value != "Color":
            return
        self._refresh_cutout_view()

    def _overlay_setting_changed(self, _event: Any) -> None:
        if self.image_container is None:
            return
        self._refresh_cutout_view()

    def _profile_crosshair_visibility_changed(self, _event: Any) -> None:
        """Refresh only the optional crosshair overlay.

        The HoloViews Tap stream remains attached to the base cutout image, so
        hiding the crosshair does not disable pixel selection or light-profile
        updates.
        """
        if self.image_container is None:
            return
        self._refresh_cutout_crosshair()

    # ------------------------------------------------------------------
    # Event wiring
    # ------------------------------------------------------------------
    def _bind_events(self) -> None:
        if getattr(self.context, "events", None) is None:
            return
        self._subscribe("selection.focus.changed", self._selection_changed)
        self._subscribe("selection.focus.cleared", self._selection_cleared)
        self._subscribe("dataset.active.changed", self._dataset_changed)
        self._subscribe("dataset.mapping.updated", self._dataset_changed)
        self._subscribe("astro.coords.updated", self._coords_updated)

    def _subscribe(self, topic: str, callback: Any) -> None:
        events = getattr(self.context, "events", None)
        if events is None:
            return
        try:
            sub = events.subscribe(
                topic,
                callback,
                owner_id=self.panel_id,
                owner_label="Euclid Cutout",
                owner_kind="panel",
            )
        except TypeError:
            sub = events.subscribe(topic, callback)
        self._subscriptions.append(sub)

    def _publish(self, topic: str, payload: Optional[Dict[str, Any]] = None) -> None:
        events = getattr(self.context, "events", None)
        if events is None:
            return
        try:
            events.publish(topic, payload or {})
        except Exception:
            traceback.print_exc()

    def _schedule_panel_callback(self, callback, *, delay_ms: int = 0) -> None:
        """Schedule UI work outside the EventBus subscriber call stack."""
        if getattr(self, "_disposed", False):
            return

        def _run() -> None:
            if getattr(self, "_disposed", False):
                return
            callback()

        try:
            doc = pn.state.curdoc
            if doc is not None:
                if delay_ms and delay_ms > 0:
                    doc.add_timeout_callback(_run, int(delay_ms))
                else:
                    doc.add_next_tick_callback(_run)
            else:
                _run()
        except Exception:
            _run()

    def _schedule_target_status_refresh(self, *, delay_ms: int = 75) -> None:
        """Resolve target status later; do not block selection.focus.changed."""
        if getattr(self, "_target_status_scheduled", False):
            return

        self._target_status_scheduled = True

        def _run() -> None:
            self._target_status_scheduled = False
            if getattr(self, "_disposed", False):
                return
            try:
                self._update_target_status()
            except Exception:
                traceback.print_exc()

        self._schedule_panel_callback(_run, delay_ms=delay_ms)

    def _schedule_auto_load(
        self,
        *,
        reason: str,
        delay_ms: int = AUTO_LOAD_DELAY_MS,
    ) -> None:
        """Run one trailing load after focus changes have gone quiet."""

        self._auto_load_generation += 1
        generation = self._auto_load_generation

        def _run() -> None:
            if self._disposed:
                return
            if generation != self._auto_load_generation:
                return

            try:
                self.load_cutout(reason=str(reason or "auto"))
            except Exception:
                traceback.print_exc()

        self._schedule_panel_callback(_run, delay_ms=delay_ms)

    def _invalidate_request(
        self,
        *,
        reason: str,
        target: Optional[_ResolvedTarget] = None,
        publish: bool = False,
    ) -> None:
        self._request_generation += 1
        self._auto_load_generation += 1
        self._cancel_job(
            target=target,
            reason=reason,
            publish=publish,
        )

    def _filter_changed(self, _event: Any) -> None:
        if self._disposed or self._suppress_filter_reload:
            return

        selected = str(self.filter_input.value)
        available = set(
            getattr(self.image_container, "available_bands", []) or []
        )

        if self.image_container is not None and selected in available:
            self._schedule_panel_callback(self._refresh_display)
            return

        old_target = self._current_target
        self._invalidate_request(
            reason="filter.changed",
            target=old_target,
            publish=True,
        )

        self.status.object = f"Loading Euclid {selected} cutout…"
        self._schedule_auto_load(
            reason="filter.changed",
            delay_ms=FILTER_LOAD_DELAY_MS,
        )

    def _event_identity(
        self,
        *,
        target: Optional[_ResolvedTarget] = None,
        dataset_id: Optional[str] = None,
        row_id: Optional[Any] = None,
        artifact_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        resolved_dataset_id = dataset_id or getattr(target, "dataset_id", None)

        if row_id is None and target is not None:
            row_id = target.row_id

        payload: Dict[str, Any] = {
            "source": "Euclid",
            "plugin_id": PLUGIN_ID,
            "origin": self.panel_id,
            "panel_id": self.panel_id,
        }

        if resolved_dataset_id is not None:
            payload["dataset_id"] = str(resolved_dataset_id)

        if row_id is not None:
            row_id_str = str(row_id)
            payload["row_id"] = row_id_str
            payload["row_ids"] = [row_id_str]

            # Transitional compatibility for older Euclid/Spectra code paths.
            payload["selected_id"] = row_id_str

        if artifact_id is not None:
            payload["artifact_id"] = str(artifact_id)

        return payload

    def _publish_cutout_running(
        self,
        running: bool,
        *,
        target: Optional[_ResolvedTarget] = None,
        dataset_id: Optional[str] = None,
        row_id: Optional[Any] = None,
        artifact_id: Optional[str] = None,
        reason: Optional[str] = None,
        error: Optional[Any] = None,
    ) -> None:
        payload = self._event_identity(
            target=target,
            dataset_id=dataset_id,
            row_id=row_id,
            artifact_id=artifact_id,
        )
        payload["running"] = bool(running)

        if reason:
            payload["reason"] = str(reason)

        if target is not None:
            payload["ra"] = target.ra
            payload["dec"] = target.dec

        if error is not None:
            payload["error"] = str(error)
            payload["error_message"] = euclid_user_error_message(error)

        self._publish("astro.cutout.running", payload)

    def _publish_cutout_artifact_created(
        self,
        *,
        artifact_id: Optional[str],
        target: _ResolvedTarget,
    ) -> None:
        if not artifact_id:
            return

        payload = self._event_identity(target=target, artifact_id=artifact_id)
        payload["type"] = CUTOUT_ARTIFACT_TYPE
        self._publish("artifact.created", payload)

    def _publish_plugin_error(
        self,
        *,
        stage: str,
        error: Any,
        target: Optional[_ResolvedTarget] = None,
    ) -> None:
        payload = self._event_identity(target=target)
        payload.update(
            {
                "stage": str(stage),
                "error": str(error),
                "error_message": euclid_user_error_message(error),
                "error_type": type(error).__name__,
            }
        )
        self._publish("plugin.error", payload)

    def _reset_loaded_cutout(self, *, clear_figure: bool = True) -> None:
        result = self._cutout_result
        self._cutout_result = None
        self.euclid_object = None
        self.image_container = None
        self._cleanup_result(result)
        self._clear_cutout_tap_stream()
        self._profile_pixel = None
        self._restore_profile_pixel_pending = False
        self._profile_initialised = False
        self._surface_dirty = True
        self._surface_generation += 1
        self._tap_update_generation += 1
        self._base_cutout_elements = []
        self._invalidate_analysis_cache()

        if clear_figure:
            self.figure.object = self._empty_image()
            self.profile_figure.object = self._empty_profile()
            self.surface_figure.object = self._empty_surface()

    def _selection_changed(self, topic: str, payload: Any) -> None:
        del payload

        reason = str(topic or "selection.focus.changed")
        old_target = self._current_target

        self._invalidate_request(
            reason=reason,
            target=old_target,
            publish=False,
        )

        self._current_target = None
        self.stored_spectrum_coordinates.clear()
        self.overplotted_coordinates = []

        def _update() -> None:
            self._publish_cutout_running(
                False,
                target=old_target,
                reason=reason,
            )
            self.target_status.object = (
                "New focused row queued; showing the previous cutout until "
                "the replacement is ready…"
            )

            if self.auto_reload.value:
                self._schedule_auto_load(reason=reason)
            else:
                self._schedule_target_status_refresh(delay_ms=75)

        self._schedule_panel_callback(_update)

    def _selection_cleared(self, topic: str, payload: Any) -> None:
        del payload

        reason = str(topic or "selection.focus.cleared")
        old_target = self._current_target

        self._invalidate_request(
            reason=reason,
            target=old_target,
            publish=False,
        )

        self._current_target = None
        self.stored_spectrum_coordinates.clear()
        self.overplotted_coordinates = []

        def _clear() -> None:
            self._publish_cutout_running(
                False,
                target=old_target,
                reason=reason,
            )
            self.status.object = "No focused row selected."
            self.target_status.object = ""
            self._reset_loaded_cutout()

        self._schedule_panel_callback(_clear)

    def _dataset_changed(self, topic: str, payload: Any) -> None:
        del payload

        reason = str(topic or "dataset.changed")
        old_target = self._current_target

        self._invalidate_request(
            reason=reason,
            target=old_target,
            publish=False,
        )

        self._current_target = None
        self.stored_spectrum_coordinates.clear()
        self.overplotted_coordinates = []

        def _update() -> None:
            self._publish_cutout_running(
                False,
                target=old_target,
                reason=reason,
            )
            self.target_status.object = (
                "Dataset changed; resolving the latest focused row…"
            )

            if self.auto_reload.value:
                self._schedule_auto_load(reason=reason)
            else:
                self._schedule_target_status_refresh(delay_ms=100)

        self._schedule_panel_callback(_update)

    def _coords_updated(self, topic: str, payload: Any) -> None:
        if not isinstance(payload, dict):
            return
        artifact_id = payload.get("artifact_id")
        source = payload.get("source") or payload.get("dataset") or "Spectra"
        selected_id = payload.get("selected_id")
        current_id = self._current_row_id()
        if selected_id is not None and current_id is not None and str(selected_id) != str(current_id):
            return

        coords = None
        if artifact_id and getattr(self.context, "artifacts", None) is not None:
            try:
                coords = self.context.artifacts.get(artifact_id)
            except Exception:
                coords = None
        if coords is None:
            coords = {
                "ra": payload.get("ra", []),
                "dec": payload.get("dec", []),
                "colors": payload.get("colors") or payload.get("colours") or [],
                "labels": payload.get("labels", []),
                "points": payload.get("points", []),
            }

        normalised = self._normalise_spectrum_coordinates(
            coords,
            fallback_source=str(source),
        )
        if not normalised["ra"] or not normalised["dec"]:
            return

        storage_key = str(source)
        if artifact_id:
            storage_key = f"{source}:{artifact_id}"
        for key in list(self.stored_spectrum_coordinates.keys()):
            if key == str(source) or key.startswith(f"{source}:"):
                del self.stored_spectrum_coordinates[key]
        self.stored_spectrum_coordinates[storage_key] = normalised
        self._schedule_panel_callback(self._refresh_cutout_view)

    # ------------------------------------------------------------------
    # Data/mapping helpers
    # ------------------------------------------------------------------
    def _active_dataset_id(self) -> str:
        datasets = getattr(self.context, "datasets", None)
        if datasets is not None:
            for name in ("active_id", "get_active_id"):
                method = getattr(datasets, name, None)
                if callable(method):
                    try:
                        active = method()
                        if active:
                            return str(active)
                    except Exception:
                        pass
        return "default"

    def _mapping(self, dataset_id: str, *roles: str) -> Optional[str]:
        datasets = getattr(self.context, "datasets", None)
        if datasets is not None:
            for role in roles:
                for method_name in ("get_mapping", "mapping", "get_column_mapping"):
                    method = getattr(datasets, method_name, None)
                    if not callable(method):
                        continue
                    for args in ((dataset_id, role), (role,), (dataset_id, role, None)):
                        try:
                            value = method(*args)
                        except TypeError:
                            continue
                        except Exception:
                            value = None
                        if value:
                            return str(value)

        config = getattr(self.context, "config", None)
        settings = getattr(config, "settings", {}) if config is not None else {}
        aliases = {
            "record_id": ["id_col", "id", "ID", "source_id", "object_id"],
            "coords.ra": ["ra_dec", "ra", "RA", "Right Ascension"],
            "coords.dec": ["ra_dec", "dec", "DEC", "Declination"],
        }
        for role in roles:
            for key in aliases.get(role, [role]):
                value = settings.get(key) if isinstance(settings, dict) else None
                if isinstance(value, str):
                    if role == "coords.ra" and "," in value:
                        return value.split(",", 1)[0].strip()
                    if role == "coords.dec" and "," in value:
                        return value.split(",", 1)[1].strip()
                    return value
        return None

    def _columns(self, dataset_id: str) -> List[str]:
        datasets = getattr(self.context, "datasets", None)
        if datasets is None:
            return []
        for method_name in ("list_columns", "columns"):
            method = getattr(datasets, method_name, None)
            if callable(method):
                try:
                    return list(method(dataset_id))
                except TypeError:
                    try:
                        return list(method())
                    except Exception:
                        pass
                except Exception:
                    pass
        try:
            df = datasets.get_df(dataset_id)
            return list(df.columns)
        except Exception:
            return []

    def _guess_column(self, dataset_id: str, candidates: Iterable[str]) -> Optional[str]:
        columns = self._columns(dataset_id)
        lower_map = {str(col).lower(): str(col) for col in columns}
        for candidate in candidates:
            if candidate in columns:
                return candidate
            found = lower_map.get(candidate.lower())
            if found:
                return found
        return None

    def _focus_state(self) -> Any:
        selection = getattr(self.context, "selection", None)
        if selection is None:
            return None
        method = getattr(selection, "get_focus", None)
        if callable(method):
            try:
                return method()
            except Exception:
                return None
        return None

    @staticmethod
    def _get_from_obj(obj: Any, *names: str) -> Any:
        if obj is None:
            return None
        if isinstance(obj, dict):
            for name in names:
                if name in obj:
                    return obj[name]
            return None
        for name in names:
            if hasattr(obj, name):
                return getattr(obj, name)
        return None

    def _current_row_id(self) -> Optional[str]:
        if self._current_target is not None:
            return self._current_target.row_id
        focus = self._focus_state()
        value = self._get_from_obj(focus, "row_id", "record_id", "id", "source_id")
        return None if value is None else str(value)

    def _target_matches_current_focus(self, target: _ResolvedTarget) -> bool:
        focus = self._focus_state()
        if focus is None:
            return True
        focus_dataset = self._get_from_obj(focus, "dataset_id", "dataset")
        focus_row_id = self._get_from_obj(focus, "row_id", "record_id", "id", "source_id")
        if focus_dataset is not None and str(focus_dataset) != str(target.dataset_id):
            return False
        if focus_row_id is not None:
            if target.row_id is None:
                return False
            if str(focus_row_id) != str(target.row_id):
                return False
        return True

    def _resolve_target(self) -> _ResolvedTarget:
        dataset_id = self._active_dataset_id()
        focus = self._focus_state()
        focus_dataset = self._get_from_obj(focus, "dataset_id", "dataset")
        if focus_dataset:
            dataset_id = str(focus_dataset)

        metadata = self._get_from_obj(focus, "metadata")
        if not isinstance(metadata, dict):
            metadata = {}

        id_column = self._mapping(dataset_id, "record_id", "id", "row_id")
        ra_column = self._mapping(dataset_id, "coords.ra", "ra") or self._guess_column(
            dataset_id,
            ["ra", "RA", "right_ascension", "alpha", "source_ra"],
        )
        dec_column = self._mapping(dataset_id, "coords.dec", "dec") or self._guess_column(
            dataset_id,
            ["dec", "DEC", "declination", "delta", "source_dec"],
        )
        if id_column is None:
            id_column = self._guess_column(dataset_id, ["id", "ID", "source_id", "object_id", "row_id"])

        if not ra_column or not dec_column:
            raise RuntimeError("Euclid Cutout requires mapped coordinate columns `coords.ra` and `coords.dec`.")

        row_id = self._get_from_obj(focus, "row_id", "record_id", "id", "source_id")
        row_pos = self._get_from_obj(
            focus,
            "row_position",
            "row_pos",
            "row_index",
            "position",
            "index",
        )

        if row_pos is None:
            row_pos = _first_not_none(
                metadata.get("row_position"),
                metadata.get("row_pos"),
                metadata.get("row_index"),
                metadata.get("position"),
                metadata.get("index"),
            )

        row = None
        for key in ("row", "record", "row_data", "data"):
            candidate = metadata.get(key)
            if isinstance(candidate, dict):
                if row_id is None or id_column is None:
                    row = dict(candidate)
                    break
                candidate_id = candidate.get(id_column)
                if candidate_id is not None and str(candidate_id) == str(row_id):
                    row = dict(candidate)
                    break

        if row is None:
            row = self._fetch_row(
                dataset_id,
                id_column=id_column,
                row_id=row_id,
                row_pos=row_pos,
                required_columns=[ra_column, dec_column],
            )

        if row is None:
            row = self._row_from_legacy_data(
                id_column=id_column,
                row_id=row_id,
                allow_first_row=(row_id is None),
            )

        if row is None:
            raise RuntimeError("No focused row is available for the Euclid cutout panel.")

        if row_id is None and id_column and id_column != "Use Index" and id_column in row:
            row_id = row.get(id_column)
        row_id_str = None if row_id is None else str(row_id)

        try:
            ra = float(row[ra_column])
            dec = float(row[dec_column])
        except Exception as exc:
            raise RuntimeError(f"Could not read numeric RA/Dec from columns {ra_column!r}/{dec_column!r}.") from exc

        return _ResolvedTarget(
            dataset_id=dataset_id,
            row_id=row_id_str,
            row=dict(row),
            ra=ra,
            dec=dec,
            ra_column=ra_column,
            dec_column=dec_column,
            id_column=id_column,
        )

    def _row_from_legacy_data(
        self,
        *,
        id_column: Optional[str],
        row_id: Any,
        allow_first_row: bool = False,
    ) -> Optional[Dict[str, Any]]:
        data = self.data
        if data is None:
            return None
        try:
            if hasattr(data, "iloc") and len(data) > 0:
                if row_id is not None and id_column:
                    if id_column == "Use Index":
                        matches = data.loc[data.index.astype(str) == str(row_id)]
                    elif id_column in data.columns:
                        matches = data[data[id_column].astype(str) == str(row_id)]
                    else:
                        return None
                    if len(matches) > 0:
                        return matches.iloc[0].to_dict()
                    return None
                if allow_first_row:
                    return data.iloc[0].to_dict()
            if isinstance(data, dict):
                if row_id is not None and id_column and id_column in data:
                    if str(data[id_column]) != str(row_id):
                        return None
                return dict(data)
        except Exception:
            return None
        return None

    def _fetch_row(
        self,
        dataset_id: str,
        *,
        id_column: Optional[str],
        row_id: Any,
        row_pos: Any,
        required_columns: Optional[List[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        datasets = getattr(self.context, "datasets", None)
        if datasets is None:
            return None

        columns: List[str] = []
        for col in [id_column, *(required_columns or [])]:
            if col and col != "Use Index" and col not in columns:
                columns.append(col)

        source = None
        for method_name in ("get_source", "source"):
            method = getattr(datasets, method_name, None)
            if callable(method):
                try:
                    source = method(dataset_id)
                    break
                except Exception:
                    pass
        if source is not None:
            row = self._fetch_row_from_source(
                source,
                id_column=id_column,
                row_id=row_id,
                row_pos=row_pos,
                columns=columns,
            )
            if row is not None:
                return row

        try:
            df = datasets.get_df(dataset_id)
            if row_id is not None and id_column:
                if id_column == "Use Index":
                    matches = df.loc[df.index.astype(str) == str(row_id)]
                    if len(matches) > 0:
                        return matches.iloc[0].to_dict()
                elif id_column in df.columns:
                    matches = df[df[id_column].astype(str) == str(row_id)]
                    if len(matches) > 0:
                        return matches.iloc[0].to_dict()
            if row_pos is not None:
                return df.iloc[int(row_pos)].to_dict()
            return None
        except Exception:
            traceback.print_exc()
            return None

    @staticmethod
    def _normalise_rows(result: Any) -> Optional[Dict[str, Any]]:
        if result is None:
            return None
        try:
            if hasattr(result, "to_pandas"):
                result = result.to_pandas()
        except Exception:
            pass
        try:
            if hasattr(result, "iloc") and len(result) > 0:
                return result.iloc[0].to_dict()
        except Exception:
            pass
        if isinstance(result, list) and result:
            first = result[0]
            if isinstance(first, dict):
                return first
        if isinstance(result, dict):
            return result
        return None

    def _fetch_row_from_source(
        self,
        source: Any,
        *,
        id_column: Optional[str],
        row_id: Any,
        row_pos: Any,
        columns: List[str],
    ) -> Optional[Dict[str, Any]]:
        if row_id is not None and id_column:
            for name, kwargs in [
                ("get_row_by_id", {"row_id": row_id, "id_column": id_column, "columns": columns or None}),
                ("get_rows_by_ids", {"row_ids": [row_id], "id_column": id_column, "columns": columns or None}),
                ("get_rows_by_id", {"row_ids": [row_id], "id_column": id_column, "columns": columns or None}),
                ("read_rows_by_id", {"row_ids": [row_id], "id_column": id_column, "columns": columns or None}),
                ("rows_by_id", {"row_ids": [row_id], "id_column": id_column, "columns": columns or None}),
                ("get_rows", {"row_ids": [row_id], "id_column": id_column, "columns": columns or None}),
            ]:
                method = getattr(source, name, None)
                if not callable(method):
                    continue
                for call_kwargs in (kwargs, {k: v for k, v in kwargs.items() if k != "columns"}):
                    try:
                        row = self._normalise_rows(method(**call_kwargs))
                        if row is not None:
                            return row
                    except TypeError:
                        continue
                    except Exception:
                        continue

        if row_pos is not None:
            for name, kwargs in [
                ("get_row_by_position", {"row_pos": int(row_pos), "columns": columns or None}),
                ("get_rows", {"row_positions": [row_pos], "columns": columns or None}),
                ("read_rows", {"row_positions": [row_pos], "columns": columns or None}),
                ("take", {"indices": [row_pos], "columns": columns or None}),
            ]:
                method = getattr(source, name, None)
                if not callable(method):
                    continue
                for call_kwargs in (kwargs, {k: v for k, v in kwargs.items() if k != "columns"}):
                    try:
                        row = self._normalise_rows(method(**call_kwargs))
                        if row is not None:
                            return row
                    except TypeError:
                        continue
                    except Exception:
                        continue
        return None

    # ------------------------------------------------------------------
    # Loading / rendering
    # ------------------------------------------------------------------
    def _runtime(self) -> EuclidCutoutRuntime:
        if self._runtime_service is not None:
            return self._runtime_service

        services = getattr(self.context, "services", None)
        if services is not None:
            try:
                if services.has(RUNTIME_SERVICE_KEY):
                    self._runtime_service = services.get(RUNTIME_SERVICE_KEY)
                    return self._runtime_service
            except Exception:
                pass

        self._runtime_service = EuclidCutoutRuntime(context=self.context)
        self._owns_runtime_service = True
        return self._runtime_service

    def _cleanup_result(self, result: Any) -> None:
        if result is None:
            return

        runtime = self._runtime_service
        if runtime is not None:
            runtime.cleanup_result(result)
            return

        cleanup = getattr(result, "cleanup", None)
        if callable(cleanup):
            cleanup()

    def _schedule_initial_load(self) -> None:
        if self._initial_load_started:
            return
        self._initial_load_started = True

        # Share the same generation guard as focus/filter auto-load callbacks.
        # A late platform restore can therefore invalidate this callback before it
        # reads constructor defaults from the widgets.
        self._auto_load_generation += 1
        generation = self._auto_load_generation

        def _run() -> None:
            if self._disposed:
                return
            if generation != self._auto_load_generation:
                return

            self._update_target_status()
            if self.auto_reload.value:
                self.load_cutout(reason="initial")

        self._schedule_panel_callback(_run, delay_ms=0)

    def _target_html(self, target: _ResolvedTarget) -> str:
        return (
            "<div>"
            f"Dataset: <code>{target.dataset_id}</code> &nbsp; "
            f"Record: <code>{target.row_id or 'unmapped'}</code> &nbsp; "
            f"RA/Dec: <code>{target.ra:.6f}, {target.dec:.6f}</code>"
            "</div>"
        )

    def _update_target_status(self) -> None:
        try:
            target = self._resolve_target()
            self._current_target = target
            self.target_status.object = self._target_html(target)
        except Exception as exc:
            self.target_status.object = f"<div style='color:#b55'>⚠️ {exc}</div>"

    def load_cutout(self, *, reason: str = "manual") -> None:
        if self._disposed:
            return

        try:
            target = self._resolve_target()
        except Exception as exc:
            self.status.object = f"**Euclid cutout unavailable:** {exc}"
            return

        previous_target = self._current_target

        # Invalidate both pending debounce callbacks and older result callbacks.
        self._auto_load_generation += 1
        self._request_generation += 1
        generation = self._request_generation

        self._cancel_job(
            target=previous_target,
            reason="superseded",
        )

        self.status.object = "Loading Euclid cutout…"
        self.target_status.object = self._target_html(target)
        self._publish_cutout_running(
            True,
            target=target,
            reason=reason,
        )

        runtime = self._runtime()
        lease = self._panel_storage
        if lease is None or lease.released:
            lease = runtime.acquire_panel_storage(self.panel_id)
            self._panel_storage = lease

        # Read all widget state on the UI thread. The worker receives plain values.
        credentials = self.credentials_file_input.value.strip() or None
        user = self.user_input.value.strip() or None
        password = self.password_input.value or None
        environment = self.environment.value or "PDR"
        radius_arcsec = float(self.radius_input.value)
        filter_name = str(self.filter_input.value)
        stretch = str(self.stretch_input.value)
        stretch_scale = self._stretch_scale_value()
        save_dir = self.save_dir_input.value or DEFAULT_SAVE_DIR
        request_signature = self._request_settings_signature()
        self._active_request_signature = request_signature

        def _worker(cancel_token: Any = None) -> Any:
            return runtime.fetch_cutout(
                ra=target.ra,
                dec=target.dec,
                radius_arcsec=radius_arcsec,
                filter_name=filter_name,
                stretch=stretch,
                stretch_scale=stretch_scale,
                environment=environment,
                user=user,
                password=password,
                credentials_filepath=credentials,
                save_dir=save_dir,
                panel_storage=lease,
                request_id=generation,
                cancel_token=cancel_token,
                verbose=True,
            )

        def _done(result: Any) -> None:
            self._on_cutout_loaded(
                result,
                target=target,
                reason=reason,
                generation=generation,
                request_signature=request_signature,
            )

        def _error(exc: BaseException) -> None:
            self._on_cutout_error(
                exc,
                target=target,
                reason=reason,
                generation=generation,
                request_signature=request_signature,
            )

        jobs = getattr(self.context, "jobs", None)
        if jobs is None:
            try:
                _done(_worker(cancel_token=None))
            except BaseException as exc:
                _error(exc)
            return

        try:
            self._job_handle = jobs.submit(
                _worker,
                title="Fetch Euclid cutout",
                key=f"{self.panel_id}:request:{generation}",
                on_done=_done,
                on_error=_error,
            )
        except TypeError:
            self._job_handle = jobs.submit(
                _worker,
                title="Fetch Euclid cutout",
                on_done=_done,
                on_error=_error,
            )

        self._job_generation = generation

    def _cancel_job(
        self,
        *,
        target: Optional[_ResolvedTarget] = None,
        reason: str = "cancelled",
        publish: bool = True,
    ) -> bool:
        handle = self._job_handle
        self._job_handle = None
        self._job_generation = None
        self._active_request_signature = None

        if handle is None:
            return False

        try:
            handle.cancel()
        except Exception:
            pass

        if publish:
            self._publish_cutout_running(
                False,
                target=target or self._current_target,
                reason=reason,
            )

        return True

    def _on_cutout_loaded(
        self,
        result: Any,
        *,
        target: _ResolvedTarget,
        reason: str,
        generation: int,
        request_signature: Tuple[Any, ...],
    ) -> None:
        if self._job_generation == generation:
            self._job_handle = None
            self._job_generation = None

        generation_is_current = generation == self._request_generation
        focus_is_current = self._target_matches_current_focus(target)
        settings_changed = self._request_settings_changed(request_signature)
        stale = (
            self._disposed
            or not generation_is_current
            or not focus_is_current
            or settings_changed
        )

        if stale:
            self._cleanup_result(result)
            if self._active_request_signature == request_signature:
                self._active_request_signature = None

            stale_reason = (
                "request_settings_changed"
                if settings_changed and generation_is_current and focus_is_current
                else "stale_result"
            )
            self._publish_cutout_running(
                False,
                target=target,
                reason=stale_reason,
            )

            # This also protects against request-setting changes made while an
            # archive job is already running, even when no widget watcher fires.
            if (
                settings_changed
                and generation_is_current
                and focus_is_current
                and not self._disposed
            ):
                self._reload_after_request_settings_change(
                    reason="request.settings.changed",
                )
            return

        self._active_request_signature = None

        previous_result = self._cutout_result
        previous_object = self.euclid_object
        previous_container = self.image_container
        previous_target = self._current_target

        self._cutout_result = result
        self.euclid_object = result.cutout

        try:
            self._create_image_container(result)
            self._current_target = target
            self.status.object = ""
            self._initialise_profile_pixel()
            self._refresh_display()
        except Exception as exc:
            self._cutout_result = previous_result
            self.euclid_object = previous_object
            self.image_container = previous_container
            self._current_target = previous_target
            self._cleanup_result(result)

            self.status.object = (
                f"**Could not initialise Euclid image visualisation:** {exc}"
            )
            self._publish_cutout_running(
                False,
                target=target,
                reason="visualisation_error",
                error=exc,
            )
            self._publish_plugin_error(
                stage="initialise_cutout_visualisation",
                error=exc,
                target=target,
            )
            return

        self._cleanup_result(previous_result)

        artifact_id: Optional[str] = None
        retained_paths = False

        # Navbar/selection-driven loads are transient. Only explicit Load actions
        # create retained FITS files and platform artifacts.
        retain_result = reason in {"button.load", "manual"}

        if retain_result:
            try:
                self._runtime().promote_result(result)
                retained_paths = bool(result.fits_paths)
                artifact_id = self._put_cutout_artifact(
                    result,
                    target=target,
                )
            except Exception as exc:
                self.status.object = (
                    "Cutout loaded, but its retained FITS artifact could not "
                    f"be written: {exc}"
                )
                self._publish_plugin_error(
                    stage="retain_cutout",
                    error=exc,
                    target=target,
                )

        # The visualisation owns detached NumPy arrays, so scratch files can now
        # be removed even though the displayed image remains available.
        self._cleanup_result(result)

        if not retained_paths:
            result.fits_paths = {}
            try:
                result.cutout.cutouts_paths = {}
            except Exception:
                pass

        self._publish_cutout_artifact_created(
            artifact_id=artifact_id,
            target=target,
        )

        updated_payload = self._event_identity(
            target=target,
            artifact_id=artifact_id,
        )
        updated_payload.update(
            {
                "ra": target.ra,
                "dec": target.dec,
                "reason": reason,
            }
        )
        self._publish("astro.cutout.updated", updated_payload)

        self._publish_cutout_running(
            False,
            target=target,
            artifact_id=artifact_id,
            reason="completed",
        )

    def _create_image_container(self, result: Any) -> None:
        bands = list(result.filters)
        if not bands:
            raise RuntimeError("Euclid result contains no image bands.")

        images = [result.images[band] for band in bands]
        wcs_list = [result.wcs.get(band) for band in bands]

        reference_band = "VIS" if "VIS" in result.wcs else bands[0]
        target_wcs = result.wcs.get(reference_band)

        preferred_color_band_sets = [
            ["NIR_H", "NIR_Y", "VIS"],
            ["NIR_H", "NIR_J", "VIS"],
            ["NIR_J", "NIR_Y", "VIS"],
        ]
        color_bands = next(
            (
                candidate
                for candidate in preferred_color_band_sets
                if all(band in bands for band in candidate)
            ),
            None,
        )

        color_bands = None

        for candidate in preferred_color_band_sets:
            if all(band in bands for band in candidate):
                color_bands = candidate
                break

        has_color = color_bands is not None

        self._invalidate_analysis_cache()
        self._tap_update_generation += 1
        self._base_cutout_elements = []
        self.image_container = ImageVisualizationClass(
            images=images,
            wcs=wcs_list,
            band_names=bands,
            color_image=color_bands is not None,
            color_bands=color_bands,
            color_name="Color",
            target_wcs=target_wcs,
        )

        available = list(self.image_container.available_bands)
        if not available:
            raise RuntimeError("Euclid visualisation contains no available bands.")

        # Keep every archive filter selectable. Choosing a filter not present in
        # the current result starts a new selected-band request.
        self.filter_input.options = ["Color", *DEFAULT_EUCLID_FILTERS]

        selected = str(self.filter_input.value)
        if selected not in available:
            fallback = "Color" if "Color" in available else available[0]
            self._suppress_filter_reload = True
            try:
                self.filter_input.value = fallback
            finally:
                self._suppress_filter_reload = False

    def _on_cutout_error(
        self,
        exc: BaseException,
        *,
        target: _ResolvedTarget,
        reason: str,
        generation: int,
        request_signature: Tuple[Any, ...],
    ) -> None:
        if self._job_generation == generation:
            self._job_handle = None
            self._job_generation = None

        generation_is_current = generation == self._request_generation
        focus_is_current = self._target_matches_current_focus(target)
        settings_changed = self._request_settings_changed(request_signature)
        stale = (
            self._disposed
            or not generation_is_current
            or not focus_is_current
            or settings_changed
        )

        if stale or isinstance(exc, CancelledError):
            if self._active_request_signature == request_signature:
                self._active_request_signature = None

            stale_reason = (
                "cancelled"
                if isinstance(exc, CancelledError)
                else (
                    "request_settings_changed"
                    if settings_changed and generation_is_current and focus_is_current
                    else "stale_result"
                )
            )
            self._publish_cutout_running(
                False,
                target=target,
                reason=stale_reason,
            )

            if (
                settings_changed
                and generation_is_current
                and focus_is_current
                and not self._disposed
            ):
                self._reload_after_request_settings_change(
                    reason="request.settings.changed",
                )
            return

        self._active_request_signature = None

        # Keep the previous successful image visible. The raw exception remains
        # in the runtime events; the status uses a concise, HTML-safe summary.
        self.status.object = (
            "**Euclid cutout unavailable:** "
            f"{euclid_user_error_message(exc)}"
        )

        self._publish_cutout_running(
            False,
            target=target,
            reason=reason,
            error=exc,
        )
        self._publish_plugin_error(
            stage="fetch_cutout",
            error=exc,
            target=target,
        )

    def _put_cutout_artifact(self, result: Any, *, target: _ResolvedTarget) -> Optional[str]:
        artifacts = getattr(self.context, "artifacts", None)
        if artifacts is None:
            return None

        payload = result.artifact_payload(include_pixels=False)

        if self.image_container is not None:
            payload["visualization"] = self.image_container.get_current_plot_config()
            payload["display_band"] = self.filter_input.value

        params = {
            "source": "Euclid",
            "row_id": target.row_id,
            "selected_id": target.row_id,
            "ra": target.ra,
            "dec": target.dec,
            "radius_arcsec": result.radius_arcsec,
            "filter_name": self.filter_input.value,
            "stretch": self.stretch_input.value,
            "stretch_scale": self._stretch_scale_value(),
            "environment": result.environment,
            "origin": self.panel_id,
            "plugin_id": PLUGIN_ID,
        }

        row_ids = [target.row_id] if target.row_id is not None else None

        try:
            return artifacts.put(
                CUTOUT_ARTIFACT_TYPE,
                payload,
                dataset_id=target.dataset_id,
                row_ids=row_ids,
                params=params,
                persist=False,
            )
        except TypeError:
            return artifacts.put(
                type=CUTOUT_ARTIFACT_TYPE,
                payload=payload,
                dataset_id=target.dataset_id,
                row_ids=row_ids,
                params=params,
                persist=False,
            )
        except Exception:
            traceback.print_exc()
            return None

    def _stretch_scale_value(self) -> Optional[float]:
        value = self.stretch_scale_input.value
        if value is None:
            return None
        try:
            value = float(value)
        except Exception:
             return None
        return value if np.isfinite(value) else None

    def _combine_global_and_channel_clip(self, channel_clip):
        """Allows the global clip bar to control all three channels"""
        global_low, global_high = self.clip_slider.value
        channel_low, channel_high = channel_clip

        global_low = float(global_low)
        global_high = float(global_high)
        channel_low = float(channel_low)
        channel_high = float(channel_high)

        span = max(global_high - global_low, 0.0)

        low = global_low + channel_low * span
        high = global_low + channel_high * span

        return low, high

    def _update_stretch_scale(self, event) -> None:
        """Fore sure there's a more elegant way to do this"""
        if self._stretch_scale_value() is None:
            self._refresh_display()
        else:
            self.stretch_scale_input.value = None

    def _analysis_band(self) -> Optional[str]:
        if self.image_container is None:
            return None
        selected = str(self.filter_input.value)
        resolved = self.image_container.resolve_wcs_band(selected)
        if resolved in self.image_container.band_names:
            return resolved
        for candidate in ["VIS", *self.image_container.band_names]:
            if candidate in self.image_container.band_names:
                return candidate
        return None

    def _invalidate_analysis_cache(self) -> None:
        self._analysis_cache_key = None
        self._analysis_cache_data = None
        self._surface_cache_key = None
        self._surface_cache_value = None
        self._surface_render_cache.clear()

    def _analysis_data(self) -> Tuple[str, np.ndarray]:
        if self.image_container is None:
            raise RuntimeError("No Euclid cutout is loaded.")

        band = self._analysis_band()
        if band is None:
            raise RuntimeError("No scalar Euclid band is available for analysis.")

        low_clip, high_clip = self.clip_slider.value
        cache_key = (
            id(self.image_container),
            band,
            str(self.stretch_input.value),
            self._stretch_scale_value(),
            float(low_clip),
            float(high_clip),
        )
        if cache_key == self._analysis_cache_key and self._analysis_cache_data is not None:
            return band, self._analysis_cache_data

        data = self.image_container.get_analysis_data(
            band,
            processed=True,
            stretch=self.stretch_input.value,
            stretch_scale=self._stretch_scale_value(),
            stretch_interval="Asymmetric",
            low_clip=low_clip,
            high_clip=high_clip,
            scale_method="MinMax",
        )
        self._analysis_cache_key = cache_key
        self._analysis_cache_data = data
        return band, data

    def _initialise_profile_pixel(self) -> None:
        if self.image_container is None:
            self._profile_pixel = None
            return
        try:
            _band, data = self._analysis_data()
        except Exception:
            self._profile_pixel = None
            return

        height, width = data.shape
        if self._restore_profile_pixel_pending and self._profile_pixel is not None:
            row, col = self._profile_pixel
            if 0 <= row < height and 0 <= col < width:
                self._restore_profile_pixel_pending = False
                return
        self._restore_profile_pixel_pending = False

        row = height // 2
        col = width // 2
        target = self._current_target
        if target is not None:
            try:
                x, y = self.image_container.world2pixel(
                    ra=target.ra,
                    dec=target.dec,
                    band=self._analysis_band(),
                )
                candidate_col = int(round(float(np.asarray(x).ravel()[0])))
                candidate_row = int(round(float(np.asarray(y).ravel()[0])))
                if 0 <= candidate_row < height and 0 <= candidate_col < width:
                    row, col = candidate_row, candidate_col
            except Exception:
                pass
        self._profile_pixel = (row, col)

    def _ensure_profile_pixel(self, data: np.ndarray) -> Tuple[int, int]:
        height, width = data.shape
        if self._profile_pixel is None:
            self._profile_pixel = (height // 2, width // 2)
        row, col = self._profile_pixel
        row = max(0, min(int(row), height - 1))
        col = max(0, min(int(col), width - 1))
        self._profile_pixel = (row, col)
        return row, col

    def _clear_cutout_tap_stream(self) -> None:
        stream = self._cutout_tap_stream
        watcher = self._cutout_tap_watcher
        self._cutout_tap_stream = None
        self._cutout_tap_watcher = None
        if stream is None or watcher is None:
            return
        try:
            stream.param.unwatch(watcher)
        except Exception:
            pass

    def _set_cutout_tap_stream(self, image: Any) -> None:
        self._clear_cutout_tap_stream()
        try:
            stream = hv.streams.Tap(source=image, x=np.nan, y=np.nan)
            watcher = stream.param.watch_values(
                self._cutout_tapped,
                ["x", "y"],
            )
        except Exception:
            return
        self._cutout_tap_stream = stream
        self._cutout_tap_watcher = watcher

    def _plot_x_to_array_col(self, x: float) -> int:
        return int(np.floor(float(x)))

    def _plot_y_to_array_row(self, y: float) -> int:
        return int(np.floor(float(y)))

    def _array_col_to_plot_x(self, col: float) -> float:
        return float(col) + 0.5

    def _array_row_to_plot_y(self, row: float) -> float:
        return float(row) + 0.5

    def _array_pixel_to_plot(self, x: float, y: float) -> Tuple[float, float]:
        return self._array_col_to_plot_x(x), self._array_row_to_plot_y(y)

    def _cutout_tapped(self, **_values: Any) -> None:
        if self._disposed or self.image_container is None:
            return

        stream = self._cutout_tap_stream
        if stream is None:
            return

        try:
            x = float(stream.x)
            y = float(stream.y)
        except (TypeError, ValueError):
            return

        if not np.isfinite(x) or not np.isfinite(y):
            return

        col = self._plot_x_to_array_col(x)
        row = self._plot_y_to_array_row(y)
        if not (0 <= row < self.image_height and 0 <= col < self.image_width):
            return

        self._profile_pixel = (row, col)

        # Do not replace the HoloViews object while its Tap callback is still
        # processing. Doing so destroys the active plot before HoloViews exits
        # ``process_on_event`` and can leave the callback with ``plot=None``.
        self._tap_update_generation += 1
        generation = self._tap_update_generation

        def _apply_tap_update() -> None:
            if self._disposed or generation != self._tap_update_generation:
                return
            self._refresh_cutout_crosshair()
            self._refresh_profile()

        self._schedule_panel_callback(_apply_tap_update)

    def _refresh_display(self) -> None:
        if self.image_container is None:
            return
        self._invalidate_analysis_cache()
        try:
            self._refresh_cutout_view()
            self._surface_dirty = True
            self._surface_generation += 1
            active = int(getattr(getattr(self, "analysis_tabs", None), "active", 0) or 0)
            if self._profile_initialised or active == 1:
                self._refresh_profile()
            if active == 2:
                self._schedule_surface_refresh()
        except Exception as exc:
            self.status.object = f"**Could not display Euclid cutout:** {exc}"
            self.figure.object = self._empty_image()

    def _display_data(self) -> np.ndarray:
        if self.image_container is None:
            raise RuntimeError("No Euclid cutout is loaded.")
        filter_name = self.filter_input.value
        if filter_name == "Color":
            r_low, r_high = self._combine_global_and_channel_clip(self.rgb_clip_r.value)
            g_low, g_high = self._combine_global_and_channel_clip(self.rgb_clip_g.value)
            b_low, b_high = self._combine_global_and_channel_clip(self.rgb_clip_b.value)
            low_clip = [r_low, g_low, b_low]
            high_clip = [r_high, g_high, b_high]
            gamma_color = [self.gamma_r.value, self.gamma_g.value, self.gamma_b.value]
        else:
            low_clip, high_clip = self.clip_slider.value
            gamma_color = 1
        return self.image_container.get_plot_data(
            band=filter_name,
            stretch=self.stretch_input.value,
            stretch_scale=self._stretch_scale_value(),
            stretch_interval="Asymmetric",
            low_clip=low_clip,
            high_clip=high_clip,
            gamma_color=gamma_color,
            scale_method="MinMax",
        )

    def _refresh_cutout_view(self) -> None:
        data = self._display_data()
        self._build_hv_figure(data)
        self._update_figure_object()

    def _build_hv_figure(self, data: Any) -> None:
        self.image_height, self.image_width = data.shape[:2]
        bounds = (0, 0, self.image_width, self.image_height)

        if len(data.shape) == 3:
            image = hv.RGB(data[::-1, ...], bounds=bounds).opts(
                active_tools=[],
                toolbar=None,
                padding=0,
                border=0,
                framewise=True,
                shared_axes=False,
                xaxis=None,
                yaxis=None,
            )
        else:
            image = hv.Image(data[::-1, ...], bounds=bounds).opts(
                active_tools=[],
                toolbar=None,
                padding=0,
                border=0,
                framewise=True,
                shared_axes=False,
                xaxis=None,
                yaxis=None,
                cmap="grey",
            )
        self._set_cutout_tap_stream(image)

        elements: List[Any] = [image]

        if int(self.contour_levels.value or 0) > 0:
            contour = self._contour_element(bounds=bounds)
            if contour is not None:
                elements.append(contour)

        if self.show_scale.value:
            elements.extend(self._scale_bar_elements())

        if self.show_source_coords.value and self._current_target is not None:
            point = self._source_coordinate_element(self._current_target.ra, self._current_target.dec)
            if point is not None:
                elements.append(point)

        self.overplotted_coordinates = []
        if self.show_spectrum_coords.value:
            self.overplotted_coordinates = self._spectrum_coordinate_elements()
            elements.extend(self.overplotted_coordinates)

        self._base_cutout_elements = elements
        self._apply_cutout_crosshair()

    def _apply_cutout_crosshair(self) -> None:
        elements = list(self._base_cutout_elements)
        if self.show_profile_crosshair.value and self._profile_pixel is not None:
            row, col = self._profile_pixel
            if 0 <= row < self.image_height and 0 <= col < self.image_width:
                elements.extend(
                    [
                        hv.VLine(self._array_col_to_plot_x(col)).opts(
                            color="#00E5FF",
                            line_width=1,
                            line_dash="dashed",
                        ),
                        hv.HLine(self._array_row_to_plot_y(row)).opts(
                            color="#00E5FF",
                            line_width=1,
                            line_dash="dashed",
                        ),
                    ]
                )
        self.euclid_fig = elements

    def _refresh_cutout_crosshair(self) -> None:
        if not self._base_cutout_elements:
            self._refresh_cutout_view()
            return
        self._apply_cutout_crosshair()
        self._update_figure_object()

    def _refresh_profile(self) -> None:
        if self.image_container is None:
            self.profile_figure.object = self._empty_profile()
            return

        try:
            band, data = self._analysis_data()
            row, col = self._ensure_profile_pixel(data)
            scale_x, scale_y = self.image_container.get_arcsec_per_pixel(
                band,
                scalar=False,
            )

            height, width = data.shape

            x_offset = (np.arange(width) - col) * float(scale_x)
            horizontal_values = np.asarray(data[row, :], dtype=np.float64)

            # The displayed cutout uses data[::-1], so construct the vertical profile
            # in the same bottom-to-top orientation.
            display_row = height - 1 - row
            y_offset = (np.arange(height) - display_row) * float(scale_y)
            vertical_values = np.asarray(data[::-1, col], dtype=np.float64)

            horizontal = self._profile_curve(
                x_offset,
                horizontal_values,
                label="Horizontal profile",
                color="#1f1f1f",
            )
            vertical = self._profile_curve(
                y_offset,
                vertical_values,
                label="Vertical profile",
                color="#1976D2",
            )

            finite_x = np.concatenate(
                [
                    np.asarray(x_offset, dtype=float)[np.isfinite(x_offset)],
                    np.asarray(y_offset, dtype=float)[np.isfinite(y_offset)],
                ]
            )
            finite_y = np.concatenate(
                [
                    horizontal_values[np.isfinite(horizontal_values)],
                    vertical_values[np.isfinite(vertical_values)],
                ]
            )
            if finite_x.size == 0 or finite_y.size == 0:
                raise ValueError("The selected profile contains no finite samples.")

            x_min = float(np.min(finite_x))
            x_max = float(np.max(finite_x))
            if x_max <= x_min:
                x_min -= 0.5
                x_max += 0.5

            y_min = float(np.min(finite_y))
            y_max = float(np.max(finite_y))
            # Display-analysis data is normally in [0, 1]. Explicit limits avoid
            # the collapsed/empty ranges produced by some HoloViews Layout
            # combinations while still accommodating small numerical excursions.
            y_low = min(-0.02, y_min - 0.04 * max(y_max - y_min, 1.0))
            y_high = max(1.02, y_max + 0.04 * max(y_max - y_min, 1.0))

            elements: List[Any] = [
                horizontal,
                vertical,
                hv.VLine(0).opts(
                    color="#D32F2F",
                    line_width=1,
                    line_dash="dotted",
                ),
            ]

            psf_note = ""
            if self.stretch_input.value == "Linear":
                psf_fwhm = self._profile_psf_fwhm(band)
                sigma = float(psf_fwhm) / 2.35482004503
                peak = float(np.max(finite_y))
                if np.isfinite(sigma) and sigma > 0 and np.isfinite(peak) and peak > 0:
                    reference_offsets = np.linspace(x_min, x_max, 512)
                    reference = peak * np.exp(
                        -0.5 * (reference_offsets / sigma) ** 2
                    )
                    elements.append(
                        hv.Curve(
                            (reference_offsets, reference),
                            kdims="Offset [arcsec]",
                            vdims="Normalised intensity",
                            label="Nominal PSF",
                        ).opts(
                            color="#D32F2F",
                            line_width=1,
                            line_dash="dashed",
                        )
                    )
                    psf_note = " Red dashed curve: nominal PSF reference, not a fit."

            profile = hv.Overlay(elements).opts(
                toolbar="above",
                tools=["pan", "wheel_zoom", "box_zoom", "reset", "save"],
                active_tools=["wheel_zoom"],
                padding=0,
                framewise=False,
                shared_axes=True,
                show_legend=True,
                legend_position="top_right",
                xlabel="Offset [arcsec]",
                ylabel="Normalised intensity",
                xlim=(x_min, x_max),
                ylim=(y_low, y_high),
            )
            self.profile_figure.object = profile

            composite_note = (
                " (VIS used for Color)"
                if self.filter_input.value == "Color"
                else ""
            )
            intensity_note = (
                "Normalised VIS intensity; the RGB composite is not sampled."
                if self.filter_input.value == "Color"
                else "Normalised display intensity."
            )

            self.profile_note.object = (
                f"**Band:** `{band}`{composite_note} &nbsp; "
                f"**Pixel:** row `{row}`, column `{col}`. "
                "Black: horizontal profile; blue: vertical profile. "
                f"{intensity_note}{psf_note} "
                "Click the cutout to move the sampled pixel. "
                "Enable `Show cutout crosshair` to display its position."
            )
            self._profile_initialised = True
        except Exception as exc:
            self.profile_note.object = f"Light profile unavailable: {exc}"
            self.profile_figure.object = self._empty_profile()

    @staticmethod
    def _profile_curve(
        offsets: np.ndarray,
        values: np.ndarray,
        *,
        label: str,
        color: str,
    ) -> Any:
        x = np.asarray(offsets, dtype=np.float64)
        y = np.asarray(values, dtype=np.float64)
        if x.shape != y.shape:
            raise ValueError("Profile offsets and values must have matching shapes.")

        finite = np.isfinite(x) & np.isfinite(y)
        if not finite.any():
            raise ValueError(f"{label} contains no finite samples.")

        # Retain gaps as NaNs rather than allowing infinities to poison Bokeh's
        # automatic range calculation.
        clean_y = np.where(np.isfinite(y), y, np.nan)
        return hv.Curve(
            (x, clean_y),
            kdims="Offset [arcsec]",
            vdims="Normalised intensity",
            label=label,
        ).opts(
            color=color,
            line_width=2,
            muted_alpha=0.15,
        )

    @staticmethod
    def _profile_psf_fwhm(band: str) -> float:
        return 0.16 if str(band) == "VIS" else 0.3

    def _surface_view_settings(self) -> Tuple[float, float, int]:
        """Return validated camera and display-grid values from the UI."""

        try:
            elevation = float(self.surface_elevation_input.value)
        except (TypeError, ValueError):
            elevation = float(SURFACE_ELEVATION_DEG)
        try:
            azimuth = float(self.surface_azimuth_input.value)
        except (TypeError, ValueError):
            azimuth = float(SURFACE_AZIMUTH_DEG)
        try:
            grid_size = int(self.surface_grid_size_input.value)
        except (TypeError, ValueError):
            grid_size = int(SURFACE_DISPLAY_SAMPLES)

        elevation = float(np.clip(elevation, 1.0, 89.0))
        azimuth = float(np.clip(azimuth, -180.0, 180.0))
        grid_size = int(
            np.clip(
                grid_size,
                SURFACE_MIN_DISPLAY_SAMPLES,
                SURFACE_MAX_SAMPLES,
            )
        )
        return elevation, azimuth, grid_size

    @staticmethod
    def _normalise_surface_azimuth(value: float) -> float:
        """Wrap a camera azimuth into the IntInput's inclusive range."""

        raw = float(value)
        wrapped = (raw + 180.0) % 360.0 - 180.0
        if np.isclose(wrapped, -180.0) and raw > 0:
            return 180.0
        return float(wrapped)

    @staticmethod
    def _surface_drag_angles(
        *,
        start_elevation: float,
        start_azimuth: float,
        start_sx: float,
        start_sy: float,
        sx: float,
        sy: float,
    ) -> Tuple[float, float]:
        """Translate one screen-space drag into bounded camera angles."""

        delta_x = float(sx) - float(start_sx)
        delta_y = float(sy) - float(start_sy)
        # Follow the pointer rather than moving the scene in the opposite
        # direction.  Bokeh screen y grows downwards, so an upward drag has a
        # negative delta and lowers the camera elevation.
        elevation = float(
            np.clip(
                float(start_elevation)
                + delta_y * SURFACE_DRAG_ELEVATION_DEG_PER_PX,
                1.0,
                89.0,
            )
        )
        azimuth = EuclidCutoutPanel._normalise_surface_azimuth(
            float(start_azimuth)
            - delta_x * SURFACE_DRAG_AZIMUTH_DEG_PER_PX
        )
        return elevation, azimuth

    def _set_surface_camera_controls(
        self,
        *,
        elevation: float,
        azimuth: float,
    ) -> None:
        """Synchronise camera inputs without recursively scheduling renders."""

        elevation_value = int(round(float(np.clip(elevation, 1.0, 89.0))))
        azimuth_value = int(round(self._normalise_surface_azimuth(azimuth)))
        self._surface_control_sync = True
        try:
            if int(self.surface_elevation_input.value) != elevation_value:
                self.surface_elevation_input.value = elevation_value
            if int(self.surface_azimuth_input.value) != azimuth_value:
                self.surface_azimuth_input.value = azimuth_value
        finally:
            self._surface_control_sync = False

    def _surface_pan_started(self, event: Any) -> None:
        if self._disposed or self.image_container is None:
            return
        if int(getattr(getattr(self, "analysis_tabs", None), "active", 0) or 0) != 2:
            return
        try:
            sx = float(event.sx)
            sy = float(event.sy)
        except (AttributeError, TypeError, ValueError):
            return
        if not np.isfinite(sx) or not np.isfinite(sy):
            return

        elevation, azimuth, _grid_size = self._surface_view_settings()
        self._surface_drag_active = True
        self._surface_drag_start_sx = sx
        self._surface_drag_start_sy = sy
        self._surface_drag_start_elevation = elevation
        self._surface_drag_start_azimuth = azimuth
        self._surface_drag_elevation = elevation
        self._surface_drag_azimuth = azimuth
        self.surface_note.object = (
            f"Rotating · elev `{int(round(elevation))}°` · "
            f"azim `{int(round(azimuth))}°`"
        )

    def _surface_panned(self, event: Any) -> None:
        if not self._surface_drag_active:
            return
        start_sx = self._surface_drag_start_sx
        start_sy = self._surface_drag_start_sy
        if start_sx is None or start_sy is None:
            return
        try:
            sx = float(event.sx)
            sy = float(event.sy)
        except (AttributeError, TypeError, ValueError):
            return
        if not np.isfinite(sx) or not np.isfinite(sy):
            return

        elevation, azimuth = self._surface_drag_angles(
            start_elevation=self._surface_drag_start_elevation,
            start_azimuth=self._surface_drag_start_azimuth,
            start_sx=start_sx,
            start_sy=start_sy,
            sx=sx,
            sy=sy,
        )
        rounded_elevation = float(int(round(elevation)))
        rounded_azimuth = float(int(round(azimuth)))
        if (
            rounded_elevation == self._surface_drag_elevation
            and rounded_azimuth == self._surface_drag_azimuth
        ):
            return

        self._surface_drag_elevation = rounded_elevation
        self._surface_drag_azimuth = rounded_azimuth
        self._set_surface_camera_controls(
            elevation=rounded_elevation,
            azimuth=rounded_azimuth,
        )
        self.surface_note.object = (
            f"Rotating · elev `{int(rounded_elevation)}°` · "
            f"azim `{int(rounded_azimuth)}°`"
        )
        self._schedule_surface_drag_preview()

    @staticmethod
    def _rgba_to_bokeh_image(rgba: np.ndarray) -> np.ndarray:
        """Pack an RGBA framebuffer for Bokeh's ``image_rgba`` glyph."""

        array = np.asarray(rgba, dtype=np.uint8)
        if array.ndim != 3 or array.shape[2] != 4:
            raise ValueError("Surface framebuffer must have shape (height, width, 4).")
        array = np.ascontiguousarray(array)
        packed = np.empty(array.shape[:2], dtype=np.uint32)
        packed.view(np.uint8).reshape(array.shape)[...] = array
        return packed

    def _surface_live_frame(self, *, render_scale: float) -> np.ndarray:
        """Render the current camera into an RGBA framebuffer only."""

        band, mesh, original_shape = self._surface_data()
        scale_x, scale_y = self.image_container.get_arcsec_per_pixel(
            band,
            scalar=False,
        )
        elevation_deg, azimuth_deg, grid_size = self._surface_view_settings()
        mesh_scale_x = float(scale_x) * original_shape[1] / mesh.shape[1]
        mesh_scale_y = float(scale_y) * original_shape[0] / mesh.shape[0]
        return self._render_surface_rgba(
            mesh,
            scale_x=mesh_scale_x,
            scale_y=mesh_scale_y,
            opaque_cells=bool(self.surface_black_cells.value),
            elevation_deg=elevation_deg,
            azimuth_deg=azimuth_deg,
            grid_size=grid_size,
            render_scale=render_scale,
        )

    def _update_surface_live_source(self, rgba: np.ndarray) -> bool:
        """Replace only the image glyph data, preserving the active drag plot."""

        source = self._surface_live_source
        image_key = self._surface_live_image_key
        if source is None or not image_key:
            return False
        try:
            packed = self._rgba_to_bokeh_image(rgba)
            if self._surface_live_flip_y:
                packed = packed[::-1, :]
            data = dict(source.data)
            data[image_key] = [packed]
            source.data = data
            return True
        except Exception:
            return False

    def _schedule_surface_drag_preview(self) -> None:
        """Coalesce pan events into live, in-place low-DPI frame updates."""

        if (
            self._disposed
            or not self._surface_drag_active
            or self.image_container is None
            or self._surface_live_preview_scheduled
        ):
            return
        self._surface_live_preview_scheduled = True

        def _run() -> None:
            self._surface_live_preview_scheduled = False
            if (
                self._disposed
                or not self._surface_drag_active
                or self.image_container is None
            ):
                return
            try:
                rgba = self._surface_live_frame(
                    render_scale=SURFACE_DRAG_PREVIEW_SCALE,
                )
                if not self._update_surface_live_source(rgba):
                    # The current HoloViews plot may have been replaced between
                    # scheduling and execution.  The release render will bind a
                    # fresh source; do not replace the active plot mid-gesture.
                    return
            except Exception as exc:
                self.surface_note.object = f"Live surface rotation unavailable: {exc}"

        self._schedule_panel_callback(
            _run,
            delay_ms=SURFACE_DRAG_PREVIEW_DELAY_MS,
        )

    def _surface_pan_finished(self, event: Any) -> None:
        if not self._surface_drag_active:
            return
        # Capture the final pointer position when PanEnd carries one.
        self._surface_panned(event)
        self._surface_drag_active = False
        self._surface_live_preview_scheduled = False
        self._surface_drag_start_sx = None
        self._surface_drag_start_sy = None
        self._surface_dirty = True
        self._surface_generation += 1
        self.surface_note.object = (
            f"Rendering · elev `{int(self._surface_drag_elevation)}°` · "
            f"azim `{int(self._surface_drag_azimuth)}°`"
        )
        # Defer replacement of the HoloViews object until the Bokeh PanEnd
        # callback has completely unwound.  Replacing it inside the callback can
        # detach the event plot and reproduce the plot.document race fixed for
        # the cutout tap stream.
        self._schedule_surface_refresh()

    def _bind_surface_drag_events(self, plot: Any, _element: Any) -> None:
        """Attach server-side camera dragging to one rendered Bokeh figure."""

        state = getattr(plot, "state", None)
        if state is None:
            return
        plot_id = id(state)
        if plot_id in self._surface_bound_plot_ids:
            return
        self._surface_bound_plot_ids.add(plot_id)

        handles = getattr(plot, "handles", {}) or {}
        source = handles.get("source")
        if source is None:
            renderer = handles.get("glyph_renderer")
            source = getattr(renderer, "data_source", None)
        if source is not None:
            try:
                data = source.data
                image_key = "image" if "image" in data else None
                if image_key is None:
                    for candidate, values in data.items():
                        if (
                            isinstance(values, (list, tuple))
                            and values
                            and np.asarray(values[0]).ndim == 2
                        ):
                            image_key = str(candidate)
                            break
                if image_key is not None:
                    self._surface_live_source = source
                    self._surface_live_image_key = image_key
                    self._surface_live_flip_y = False
                    if self._surface_last_rgba is not None:
                        current = np.asarray(data[image_key][0])
                        packed = self._rgba_to_bokeh_image(self._surface_last_rgba)
                        if current.shape == packed.shape:
                            if np.array_equal(current, packed[::-1, :]):
                                self._surface_live_flip_y = True
            except Exception:
                self._surface_live_source = None
                self._surface_live_image_key = None

        try:
            from bokeh.events import Pan, PanEnd, PanStart
        except Exception:
            return

        # The PanTool is used only as a gesture recogniser. Lock both image
        # ranges to their complete extents so dragging cannot translate the 2-D
        # framebuffer while it controls the Matplotlib camera.
        for range_name in ("x_range", "y_range"):
            range_object = getattr(state, range_name, None)
            if range_object is None:
                continue
            try:
                start = float(range_object.start)
                end = float(range_object.end)
                span = abs(end - start)
                range_object.bounds = (min(start, end), max(start, end))
                if span > 0:
                    range_object.min_interval = span
                    range_object.max_interval = span
            except Exception:
                pass

        state.on_event(PanStart, self._surface_pan_started)
        state.on_event(Pan, self._surface_panned)
        state.on_event(PanEnd, self._surface_pan_finished)

    def _surface_style_changed(self, _event: Any) -> None:
        if self._disposed or self._surface_control_sync:
            return
        self._surface_dirty = True
        self._surface_generation += 1
        active = int(getattr(getattr(self, "analysis_tabs", None), "active", 0) or 0)
        if active == 2:
            self._schedule_surface_refresh()

    def _schedule_surface_refresh(self) -> None:
        if self._disposed or self.image_container is None or not self._surface_dirty:
            return
        self._surface_generation += 1
        generation = self._surface_generation
        self.surface_note.object = "Rendering bounded surface mesh…"

        def _run() -> None:
            if self._disposed or generation != self._surface_generation:
                return
            if int(getattr(self.analysis_tabs, "active", 0) or 0) != 2:
                return
            self._refresh_surface()

        self._schedule_panel_callback(_run)

    def _surface_data(
        self,
    ) -> Tuple[str, np.ndarray, Tuple[int, int]]:
        if self.image_container is None:
            raise RuntimeError("No Euclid cutout is loaded.")

        band = self._analysis_band()
        if band is None:
            raise RuntimeError("No scalar Euclid band is available for the surface view.")

        cache_key = (id(self.image_container), band)
        if cache_key == self._surface_cache_key and self._surface_cache_value is not None:
            mesh, original_shape = self._surface_cache_value
            return band, mesh, original_shape

        # The surface should reveal astronomical structure rather than amplify
        # every display-stretched background pixel. Start from the aligned raw
        # scalar band, then perform bounded robust preprocessing below.
        raw = self.image_container.get_analysis_data(
            band,
            aligned=True,
            processed=False,
        )
        mesh, original_shape = self._bounded_surface_data(raw)
        self._surface_cache_key = cache_key
        self._surface_cache_value = (mesh, original_shape)
        return band, mesh, original_shape

    def _refresh_surface(self) -> None:
        if self.image_container is None:
            self.surface_figure.object = self._empty_surface()
            return
        try:
            band, mesh, original_shape = self._surface_data()
            scale_x, scale_y = self.image_container.get_arcsec_per_pixel(
                band,
                scalar=False,
            )
            opaque_cells = bool(self.surface_black_cells.value)
            elevation_deg, azimuth_deg, grid_size = self._surface_view_settings()
            mesh_scale_x = float(scale_x) * original_shape[1] / mesh.shape[1]
            mesh_scale_y = float(scale_y) * original_shape[0] / mesh.shape[0]
            render_key = (
                self._surface_cache_key,
                mesh.shape,
                mesh_scale_x,
                mesh_scale_y,
                opaque_cells,
                elevation_deg,
                azimuth_deg,
                grid_size,
                SURFACE_RENDER_SCALE,
            )
            # Dicts retain insertion order.  Move cache hits to the end and cap
            # the cache because high-DPI framebuffers are several megabytes.
            surface = self._surface_render_cache.pop(render_key, None)
            if surface is not None:
                self._surface_render_cache[render_key] = surface
            else:
                surface = self._surface_wireframe(
                    mesh,
                    scale_x=mesh_scale_x,
                    scale_y=mesh_scale_y,
                    opaque_cells=opaque_cells,
                    elevation_deg=elevation_deg,
                    azimuth_deg=azimuth_deg,
                    grid_size=grid_size,
                )
                self._surface_render_cache[render_key] = surface
                while len(self._surface_render_cache) > 6:
                    oldest_key = next(iter(self._surface_render_cache))
                    self._surface_render_cache.pop(oldest_key, None)
            self.surface_figure.object = surface
            composite_note = " · Color→VIS" if self.filter_input.value == "Color" else ""
            self.surface_note.object = (
                f"**{band}**{composite_note} · "
                f"mesh `{mesh.shape[1]}×{mesh.shape[0]}` from "
                f"`{original_shape[1]}×{original_shape[0]}` px · "
                "background-subtracted raw intensity · drag to rotate"
            )
            self._surface_dirty = False
        except Exception as exc:
            self.surface_note.object = f"Surface view unavailable: {exc}"
            self.surface_figure.object = self._empty_surface()

    @staticmethod
    def _smooth_surface_array(data: np.ndarray, *, passes: int) -> np.ndarray:
        """Apply a small separable Gaussian-like kernel without a new dependency."""

        smoothed = np.asarray(data, dtype=np.float32)
        for _ in range(max(0, int(passes))):
            padded = np.pad(smoothed, 1, mode="edge")
            smoothed = (
                padded[:-2, :-2]
                + 2.0 * padded[:-2, 1:-1]
                + padded[:-2, 2:]
                + 2.0 * padded[1:-1, :-2]
                + 4.0 * padded[1:-1, 1:-1]
                + 2.0 * padded[1:-1, 2:]
                + padded[2:, :-2]
                + 2.0 * padded[2:, 1:-1]
                + padded[2:, 2:]
            ) / 16.0
        return np.asarray(smoothed, dtype=np.float32)

    @staticmethod
    def _block_peak_surface(
        data: np.ndarray,
        *,
        target_rows: int,
        target_cols: int,
    ) -> np.ndarray:
        """Reduce a surface while retaining compact peaks and broad wings.

        A top-quartile mean is less spiky than max/top-three pooling and avoids
        making unresolved sources appear artificially truncated.
        """

        rows, cols = data.shape
        row_edges = np.linspace(0, rows, target_rows + 1, dtype=int)
        col_edges = np.linspace(0, cols, target_cols + 1, dtype=int)
        reduced = np.zeros((target_rows, target_cols), dtype=np.float32)

        for row_index in range(target_rows):
            row_start = int(row_edges[row_index])
            row_stop = max(row_start + 1, int(row_edges[row_index + 1]))
            for col_index in range(target_cols):
                col_start = int(col_edges[col_index])
                col_stop = max(col_start + 1, int(col_edges[col_index + 1]))
                block = np.asarray(
                    data[row_start:row_stop, col_start:col_stop],
                    dtype=np.float32,
                )
                finite = block[np.isfinite(block)]
                if finite.size == 0:
                    continue
                count = max(1, int(np.ceil(finite.size * 0.25)))
                strongest = np.partition(finite, -count)[-count:]
                reduced[row_index, col_index] = float(np.mean(strongest))

        return reduced

    @staticmethod
    def _bounded_surface_data(data: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
        array = np.asarray(data, dtype=np.float32)
        if array.ndim != 2 or array.size == 0:
            raise ValueError("Surface data must be a non-empty two-dimensional array.")

        original_shape = array.shape
        target_shape = (
            min(array.shape[0], SURFACE_MAX_SAMPLES),
            min(array.shape[1], SURFACE_MAX_SAMPLES),
        )
        finite_mask = np.isfinite(array)
        finite = array[finite_mask]
        if finite.size == 0:
            return np.zeros(target_shape, dtype=np.float32), original_shape

        background = float(np.median(finite))
        working = np.where(finite_mask, array, background).astype(
            np.float32,
            copy=False,
        )
        working = EuclidCutoutPanel._smooth_surface_array(
            working,
            passes=SURFACE_SMOOTH_PASSES,
        )

        smooth_finite = working[np.isfinite(working)]
        background = float(np.median(smooth_finite))
        mad = float(np.median(np.abs(smooth_finite - background)))
        robust_sigma = 1.4826 * mad
        if not np.isfinite(robust_sigma) or robust_sigma <= 0:
            robust_sigma = float(np.std(smooth_finite))
        if not np.isfinite(robust_sigma) or robust_sigma <= 0:
            robust_sigma = 0.0

        # Keep low-level wings and a small amount of the smoothed background.
        # The former 1.75-sigma hard floor made extended sources terminate too
        # early and produced the visibly cut-off surface base.
        floor = background + SURFACE_NOISE_FLOOR_SIGMA * robust_sigma
        signal = np.clip(working - floor, 0.0, None)
        positive = signal[signal > 0]
        if positive.size == 0:
            return np.zeros(target_shape, dtype=np.float32), original_shape

        high = float(np.percentile(positive, SURFACE_HIGH_PERCENTILE))
        peak = float(np.max(positive))
        if not np.isfinite(high) or high <= 0:
            high = peak
        if not np.isfinite(peak) or peak <= 0:
            return np.zeros(target_shape, dtype=np.float32), original_shape

        # Soft asinh scaling: use the high percentile as the softening scale but
        # normalise by the actual peak. Unlike clipping to the percentile, this
        # preserves the full summit and prevents flat/cut-off peaks.
        softening = max(high / 5.0, robust_sigma, np.finfo(np.float32).eps)
        denominator = float(np.arcsinh(peak / softening))
        if not np.isfinite(denominator) or denominator <= 0:
            return np.zeros(target_shape, dtype=np.float32), original_shape
        signal = np.arcsinh(signal / softening) / denominator
        signal = np.clip(signal, 0.0, None)

        reduced = EuclidCutoutPanel._block_peak_surface(
            signal,
            target_rows=target_shape[0],
            target_cols=target_shape[1],
        )
        reduced[~np.isfinite(reduced)] = 0.0
        peak_reduced = float(np.max(reduced)) if reduced.size else 0.0
        if peak_reduced > 0:
            reduced /= peak_reduced
        return reduced, original_shape

    @staticmethod
    def _surface_mesh_coordinates(
        data: np.ndarray,
        *,
        scale_x: float,
        scale_y: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return the single mesh used by both surface display modes."""

        rows, cols = data.shape
        x = (np.arange(cols) - (cols - 1) / 2.0) * float(scale_x)
        y = (np.arange(rows) - (rows - 1) / 2.0) * float(scale_y)
        xx, yy = np.meshgrid(x, y)
        zz = np.asarray(data, dtype=np.float32)
        return xx, yy, zz

    @staticmethod
    def _surface_display_mesh(
        data: np.ndarray,
        *,
        scale_x: float,
        scale_y: float,
        max_samples: int = SURFACE_DISPLAY_SAMPLES,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return a lighter, peak-preserving mesh for the static 3-D renderer.

        The analysis path remains bounded at ``SURFACE_MAX_SAMPLES``.  This
        second reduction exists only to make the wire grid readable.  Both
        transparent and black-face modes receive these exact same vertices.
        """

        source = np.asarray(data, dtype=np.float32)
        if source.ndim != 2 or source.size == 0:
            raise ValueError("Surface data must be a non-empty two-dimensional array.")

        rows, cols = source.shape
        target_rows = min(rows, max(2, int(max_samples)))
        target_cols = min(cols, max(2, int(max_samples)))

        if (target_rows, target_cols) == source.shape:
            reduced = np.array(source, dtype=np.float32, copy=True)
        else:
            reduced = EuclidCutoutPanel._block_peak_surface(
                source,
                target_rows=target_rows,
                target_cols=target_cols,
            )
            source_peak = float(np.nanmax(source)) if source.size else 0.0
            reduced_peak = float(np.nanmax(reduced)) if reduced.size else 0.0
            if source_peak > 0 and reduced_peak > 0:
                reduced *= source_peak / reduced_peak

        x_half_span = max(0.0, (cols - 1) * float(scale_x) / 2.0)
        y_half_span = max(0.0, (rows - 1) * float(scale_y) / 2.0)
        x = np.linspace(-x_half_span, x_half_span, target_cols)
        y = np.linspace(-y_half_span, y_half_span, target_rows)
        xx, yy = np.meshgrid(x, y)
        return xx, yy, reduced

    @staticmethod
    def _render_surface_rgba(
        data: np.ndarray,
        *,
        scale_x: float,
        scale_y: float,
        opaque_cells: bool,
        elevation_deg: float = SURFACE_ELEVATION_DEG,
        azimuth_deg: float = SURFACE_AZIMUTH_DEG,
        grid_size: int = SURFACE_DISPLAY_SAMPLES,
        render_scale: float = SURFACE_RENDER_SCALE,
        width_px: int = SURFACE_RENDER_WIDTH_PX,
        height_px: int = SURFACE_RENDER_HEIGHT_PX,
    ) -> np.ndarray:
        """Render one depth-correct surface with identical geometry in both modes.

        Both checkbox states use the same ``plot_surface`` call, mesh vertices,
        camera, axis limits and white cell edges.  The only variable is the
        black face alpha.  Matplotlib's 3-D renderer performs the facet depth
        ordering, avoiding the reversed painter ordering seen when thousands of
        independent HoloViews polygons were used.
        """

        try:
            from matplotlib.backends.backend_agg import FigureCanvasAgg
            from matplotlib.figure import Figure
        except Exception as exc:  # pragma: no cover - depends on deployment extras
            raise RuntimeError(
                "The Euclid surface view requires matplotlib."
            ) from exc

        grid_size = int(
            np.clip(
                int(grid_size),
                SURFACE_MIN_DISPLAY_SAMPLES,
                SURFACE_MAX_SAMPLES,
            )
        )
        xx, yy, zz = EuclidCutoutPanel._surface_display_mesh(
            data,
            scale_x=scale_x,
            scale_y=scale_y,
            max_samples=grid_size,
        )
        if zz.ndim != 2 or zz.size == 0:
            raise ValueError("Surface data must be a non-empty two-dimensional array.")

        width_px = max(320, int(width_px))
        height_px = max(240, int(height_px))
        render_scale = float(np.clip(float(render_scale), 1.0, 3.0))
        base_dpi = 100.0
        render_dpi = base_dpi * render_scale
        # Keep the figure's physical geometry unchanged while increasing its
        # raster resolution.  The responsive pane downsamples this framebuffer,
        # giving much sharper labels and edges on high-DPI displays.
        figure = Figure(
            figsize=(width_px / base_dpi, height_px / base_dpi),
            dpi=render_dpi,
            facecolor="black",
        )
        canvas = FigureCanvasAgg(figure)
        axis = figure.add_subplot(111, projection="3d", facecolor="black")

        # The same full-resolution quadrilateral mesh is used in both modes.
        # Transparent mode exposes rear lines; opaque mode lets the black faces
        # hide them.  No alternative sampling or line geometry is introduced.
        face_alpha = 1.0 if opaque_cells else 0.0
        axis.plot_surface(
            xx,
            yy,
            zz,
            rstride=1,
            cstride=1,
            color=(0.0, 0.0, 0.0, face_alpha),
            edgecolor="white",
            linewidth=SURFACE_EDGE_WIDTH,
            antialiased=True,
            shade=False,
            zsort="average",
        )

        axis.view_init(
            elev=float(np.clip(elevation_deg, 1.0, 89.0)),
            azim=float(np.clip(azimuth_deg, -180.0, 180.0)),
        )
        try:
            axis.set_proj_type("ortho")
        except Exception:
            pass

        x_min = float(np.nanmin(xx))
        x_max = float(np.nanmax(xx))
        y_min = float(np.nanmin(yy))
        y_max = float(np.nanmax(yy))
        z_max = max(float(np.nanmax(zz)), 1.0)
        axis.set_xlim(x_min, x_max)
        axis.set_ylim(y_min, y_max)
        axis.set_zlim(0.0, z_max)

        x_span = max(x_max - x_min, 1.0)
        y_span = max(y_max - y_min, 1.0)
        footprint = max(x_span, y_span)
        try:
            axis.set_box_aspect(
                (x_span, y_span, footprint * SURFACE_HEIGHT_FRACTION),
                zoom=SURFACE_CAMERA_ZOOM,
            )
        except TypeError:
            # Matplotlib < 3.6 does not expose the zoom keyword.
            axis.set_box_aspect(
                (x_span, y_span, footprint * SURFACE_HEIGHT_FRACTION)
            )
        except Exception:
            pass

        x_ticks = np.linspace(x_min, x_max, SURFACE_AXIS_TICKS)
        y_ticks = np.linspace(y_min, y_max, SURFACE_AXIS_TICKS)
        z_ticks = np.linspace(0.0, z_max, SURFACE_AXIS_TICKS)
        axis.set_xticks(x_ticks)
        axis.set_yticks(y_ticks)
        axis.set_zticks(z_ticks)
        axis.set_xticklabels([f"{value:.3g}" for value in x_ticks])
        axis.set_yticklabels([f"{value:.3g}" for value in y_ticks])
        axis.set_zticklabels([f"{value:.3g}" for value in z_ticks])
        axis.set_xlabel(
            "X [arcsec]",
            color="white",
            labelpad=3,
            fontsize=11,
        )
        axis.set_ylabel(
            "Y [arcsec]",
            color="white",
            labelpad=3,
            fontsize=11,
        )
        axis.set_zlabel(
            "intensity",
            color="white",
            labelpad=5,
            fontsize=11,
        )
        axis.tick_params(colors="white", labelsize=10, pad=0, length=2)

        # Keep only the useful axes.  Pane fills and the default rectangular
        # wall grid obscure the astronomical surface and are not part of the
        # IRAF-style presentation.
        for axis_component in (axis.xaxis, axis.yaxis, axis.zaxis):
            try:
                axis_component.pane.set_facecolor((0.0, 0.0, 0.0, 0.0))
                axis_component.pane.set_edgecolor((0.0, 0.0, 0.0, 0.0))
                axis_component.line.set_color((1.0, 1.0, 1.0, 0.0))
            except Exception:
                pass
            try:
                axis_component._axinfo["grid"]["color"] = (
                    1.0,
                    1.0,
                    1.0,
                    0.0,
                )
            except Exception:
                pass

        # Fill the available pane instead of centring a small surface inside a
        # large empty 3-D cube.  The labels remain inside the framebuffer.
        axis.set_position([0.02, -0.02, 0.96, 1.04])
        figure.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
        canvas.draw()
        rgba = np.asarray(canvas.buffer_rgba(), dtype=np.uint8).copy()
        figure.clear()
        return rgba

    def _surface_wireframe(
        self,
        data: np.ndarray,
        *,
        scale_x: float = 1.0,
        scale_y: float = 1.0,
        opaque_cells: bool = False,
        elevation_deg: float = SURFACE_ELEVATION_DEG,
        azimuth_deg: float = SURFACE_AZIMUTH_DEG,
        grid_size: int = SURFACE_DISPLAY_SAMPLES,
    ) -> Any:
        """Return the Matplotlib-rendered surface as one static HoloViews RGB."""

        rgba = self._render_surface_rgba(
            data,
            scale_x=scale_x,
            scale_y=scale_y,
            opaque_cells=opaque_cells,
            elevation_deg=elevation_deg,
            azimuth_deg=azimuth_deg,
            grid_size=grid_size,
        )
        self._surface_last_rgba = np.asarray(rgba, dtype=np.uint8).copy()
        height, width = rgba.shape[:2]
        # ``hv.RGB`` already displays array rows in the orientation expected for
        # an image element.  The Agg framebuffer is therefore passed through
        # unchanged; flipping it here turns the complete Matplotlib rendering
        # upside down, including its labels and vertical intensity axis.
        return hv.RGB(
            rgba,
            bounds=(0.0, 0.0, float(width), float(height)),
        ).opts(
            tools=["pan"],
            active_tools=["pan"],
            hooks=[self._bind_surface_drag_events],
            toolbar=None,
            padding=0,
            framewise=False,
            shared_axes=False,
            xaxis=None,
            yaxis=None,
        )

    def _empty_profile(self) -> Any:
        return hv.Curve(([], [])).opts(
            active_tools=[],
            toolbar=None,
            xaxis=None,
            yaxis=None,
        )

    def _empty_surface(self) -> Any:
        return hv.Curve(([], [])).opts(
            active_tools=[],
            toolbar=None,
            xaxis=None,
            yaxis=None,
            bgcolor="black",
        )

    def _contour_band(self) -> Optional[str]:
        if self.image_container is None:
            return None
        filter_name = self.filter_input.value
        if filter_name != "Color":
            return filter_name
        for candidate in ["VIS", *self.image_container.band_names]:
            if candidate in self.image_container.band_names:
                return candidate
        return None

    def _contour_element(self, *, bounds: Tuple[float, float, float, float]) -> Optional[Any]:
        if self.image_container is None:
            return None
        try:
            contour_band = self._contour_band()
            if contour_band is None:
                return None
            temp_data = self.image_container.get_plot_data(
                contour_band,
                stretch=self.stretch_input.value,
                stretch_scale=self._stretch_scale_value(),
                stretch_interval="Asymmetric",
                low_clip=0,
                high_clip=1,
                gamma_color=1,
                scale_method="MinMax",
            )
            if temp_data.ndim == 3:
                return None
            finite_max = np.nanmax(temp_data)
            if not np.isfinite(finite_max) or finite_max <= 0:
                return None
            base = max(float(self.contour_base.value), 1.01)
            exponent = max(float(self.contour_exponent.value), 0.01)
            levels = finite_max / (base ** (np.arange(1, int(self.contour_levels.value) + 1) * exponent))
            temp_img = hv.Image(temp_data[::-1, ...], bounds=bounds)
            return hv.operation.contours(temp_img, levels=levels).opts(
                cmap=["red"],
                colorbar=False,
                active_tools=[],
                show_legend=False,
            )
        except Exception:
            return None

    def _scale_bar_elements(self) -> List[Any]:
        if self.image_container is None:
            return []
        try:
            arcsec_per_pix, _ = self.image_container.get_arcsec_per_pixel(
                self.filter_input.value,
                scalar=False,
            )
            arcsec_per_pix = float(arcsec_per_pix)
        except Exception:
            return []
        if not np.isfinite(arcsec_per_pix) or arcsec_per_pix <= 0:
            return []

        target_arcsec = self.image_width * 0.2 * arcsec_per_pix
        candidates = np.asarray(
            [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100],
            dtype=float,
        )
        eligible = candidates[candidates <= target_arcsec]
        scale_arcsec = float(eligible[-1] if eligible.size else candidates[0])
        self.bar_length_pixels = scale_arcsec / arcsec_per_pix

        x0, y0 = 0.1 * self.image_width, 0.1 * self.image_height
        x1 = x0 + self.bar_length_pixels
        label = f'{scale_arcsec:g}"'
        return [
            hv.Curve(([x0, x1], [y0, y0])).opts(color="red", line_width=3),
            hv.Text((x0 + x1) / 2, y0 + y0 / 2, label).opts(
                text_color="red",
                text_align="center",
                text_baseline="bottom",
                fontsize=14,
            ),
        ]

    def _source_coordinate_element(self, ra: float, dec: float) -> Optional[Any]:
        if self.image_container is None:
            return None
        try:
            x, y = self.image_container.world2pixel(ra=ra, dec=dec, band=self.filter_input.value)
            x_value = float(np.asarray(x).ravel()[0])
            y_value = float(np.asarray(y).ravel()[0])
            if 0 <= x_value < self.image_width and 0 <= y_value < self.image_height:
                plot_x, plot_y = self._array_pixel_to_plot(x_value, y_value)
                label = f"{ra:.3f}, {dec:.3f}"
                return hv.Points([(plot_x, plot_y)], label=label).opts(
                    color="blue",
                    marker="+",
                    size=30,
                )
        except Exception:
            return None
        return None

    def _normalise_spectrum_coordinates(
        self,
        coords: Any,
        *,
        fallback_source: str,
    ) -> Dict[str, List[Any]]:
        if coords is None:
            coords = {}
        if not isinstance(coords, dict):
            coords = getattr(coords, "payload", coords)
        if not isinstance(coords, dict):
            coords = {}

        ra_values = coords.get("ra")
        if ra_values is None:
            ra_values = coords.get("RA")
        if ra_values is None:
            ra_values = []

        dec_values = coords.get("dec")
        if dec_values is None:
            dec_values = coords.get("DEC")
        if dec_values is None:
            dec_values = []

        colors = coords.get("colors")
        if colors is None:
            colors = coords.get("colours")
        if colors is None:
            colors = []

        labels = coords.get("labels")
        if labels is None:
            labels = []

        points = coords.get("points")
        if points is None:
            points = []

        if np.isscalar(ra_values):
            ra_values = [ra_values]
        if np.isscalar(dec_values):
            dec_values = [dec_values]
        if np.isscalar(colors):
            colors = [colors]
        if np.isscalar(labels):
            labels = [labels]

        if len(ra_values) == 0 or len(dec_values) == 0:
            point_ra: List[Any] = []
            point_dec: List[Any] = []
            point_colors: List[Any] = []
            point_labels: List[Any] = []
            for point in points:
                if not isinstance(point, dict):
                    continue
                point_ra.append(_first_not_none(point.get("ra"), point.get("RA")))
                point_dec.append(_first_not_none(point.get("dec"), point.get("DEC")))
                point_colors.append(
                    _first_not_none(point.get("color"), point.get("colour"))
                )
                point_labels.append(point.get("label"))
            if point_ra and point_dec:
                ra_values = point_ra
                dec_values = point_dec
                if len(colors) == 0:
                    colors = point_colors
                if len(labels) == 0:
                    labels = point_labels

        out_ra: List[float] = []
        out_dec: List[float] = []
        out_colors: List[str] = []
        out_labels: List[str] = []

        for idx, (ra, dec) in enumerate(zip(list(ra_values), list(dec_values))):
            try:
                ra_f = float(ra)
                dec_f = float(dec)
                if not np.isfinite(ra_f) or not np.isfinite(dec_f):
                    continue
            except Exception:
                continue
            out_ra.append(ra_f)
            out_dec.append(dec_f)
            color = None
            if idx < len(colors):
                color = colors[idx]
            if not color:
                color = _fallback_spectrum_colour(idx)
            out_colors.append(str(color))
            label = None
            if idx < len(labels):
                label = labels[idx]
            if not label:
                label = f"{fallback_source} {idx + 1}"
            out_labels.append(str(label))

        return {"ra": out_ra, "dec": out_dec, "colors": out_colors, "labels": out_labels}

    def _spectrum_coordinate_elements(self) -> List[Any]:
        if self.image_container is None:
            return []
        elements: List[Any] = []
        for _source, coords in self.stored_spectrum_coordinates.items():
            ra_values = coords.get("ra", [])
            dec_values = coords.get("dec", [])
            colors = coords.get("colors", [])
            labels = coords.get("labels", [])
            if not ra_values or not dec_values:
                continue
            try:
                x, y = self.image_container.world2pixel(ra=ra_values, dec=dec_values, band=self.filter_input.value)
            except Exception:
                continue
            for idx, (x_value, y_value) in enumerate(zip(np.asarray(x).ravel(), np.asarray(y).ravel())):
                try:
                    xv = float(x_value)
                    yv = float(y_value)
                except Exception:
                    continue
                if not (0 <= xv < self.image_width and 0 <= yv < self.image_height):
                    continue
                color = colors[idx] if idx < len(colors) else _fallback_spectrum_colour(idx)
                label = labels[idx] if idx < len(labels) else f"Spectrum {idx + 1}"
                plot_x, plot_y = self._array_pixel_to_plot(xv, yv)
                elements.append(
                    hv.Points([(plot_x, plot_y)], label=str(label)).opts(
                        color=str(color),
                        marker="x",
                        size=18,
                        line_width=2,
                    )
                )
        return elements

    def _update_figure_object(self) -> None:
        if not self.euclid_fig:
            self.figure.object = self._empty_image()
            return
        overlay = hv.Overlay(self.euclid_fig).opts(
            active_tools=[],
            toolbar=None,
            padding=0,
            framewise=True,
            shared_axes=False,
            xaxis=None,
            yaxis=None,
        )
        self.figure.object = overlay

    def _empty_image(self) -> Any:
        data = np.zeros((2, 2), dtype=float)
        return hv.Image(data, bounds=(0, 0, 2, 2)).opts(
            active_tools=[],
            toolbar=None,
            padding=0,
            framewise=True,
            shared_axes=False,
            xaxis=None,
            yaxis=None,
            cmap="grey",
        )

    def _environment_changed(self, event: Any) -> None:
        try:
            visible = self.environment.value != "PDR"
            self.login_column.visible = visible

            if hasattr(self, "request_group"):
                height = (
                    REQUEST_GROUP_HEIGHT_WITH_LOGIN
                    if visible
                    else REQUEST_GROUP_HEIGHT
                )
                self.request_group.height = height
                self.request_group.min_height = height
                self.request_group.max_height = height

        except Exception:
            pass

    def _clean_async_jobs(self) -> None:
        try:
            removed = self._runtime().clean_async_jobs()
            self.status.object = (
                f"Removed **{removed}** Euclid archive async "
                f"job{'s' if removed != 1 else ''}."
                if removed
                else "No Euclid archive async jobs needed cleaning."
            )
        except Exception as exc:
            self.status.object = (
                "**Could not clean Euclid archive async jobs:** "
                f"{euclid_user_error_message(exc)}"
            )
            self._publish_plugin_error(
                stage="clean_async_jobs",
                error=exc,
                target=self._current_target,
            )

class EuclidCutoutArtifactViewer:
    """Small viewer for in-memory Euclid cutout artifacts."""

    def __init__(self, *, context: Any = None, artifact: Any = None, payload: Any = None, **_: Any) -> None:
        self.context = context
        self.artifact = artifact
        self.payload = self._payload_from_artifact(artifact, payload)
        self.figure = pn.pane.HoloViews(self._build_view(), sizing_mode="stretch_both", min_height=320)
        self.layout = pn.Column(self.figure, sizing_mode="stretch_both")

    def view(self) -> pn.viewable.Viewable:
        return self.layout

    @staticmethod
    def _payload_from_artifact(artifact: Any, payload: Any) -> Dict[str, Any]:
        if isinstance(payload, dict):
            return payload
        if isinstance(artifact, dict):
            if isinstance(artifact.get("payload"), dict):
                return artifact["payload"]
            return artifact
        candidate = getattr(artifact, "payload", None)
        if isinstance(candidate, dict):
            return candidate
        return {}

    @staticmethod
    def _load_images_from_fits(
        fits_paths: Dict[str, str],
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        images: Dict[str, np.ndarray] = {}
        wcs: Dict[str, Any] = {}

        if not fits_paths:
            return images, wcs

        try:
            from astropy.io import fits
            from astropy.wcs import WCS
        except Exception:
            return images, wcs

        for band, path in dict(fits_paths).items():
            if not path:
                continue

            try:
                with fits.open(str(path), memmap=False) as hdul:
                    selected_hdu = next(
                        (
                            hdu
                            for hdu in hdul
                            if getattr(hdu, "data", None) is not None
                        ),
                        None,
                    )
                    if selected_hdu is None:
                        continue

                    array = np.array(selected_hdu.data, copy=True)
                    header = selected_hdu.header.copy()

                while array.ndim > 2:
                    array = array[0]

                if array.ndim != 2 or array.size == 0:
                    continue

                images[str(band)] = array
                try:
                    wcs[str(band)] = WCS(header)
                except Exception:
                    pass
            except Exception:
                continue

        return images, wcs

    def _build_view(self) -> Any:
        try:
            images = self.payload.get("images") or {}
            wcs = self.payload.get("wcs") or {}
            filters = self.payload.get("filters") or list(images.keys())

            if not images:
                loaded_images, loaded_wcs = self._load_images_from_fits(
                    self.payload.get("fits_paths") or {}
                )
                images = loaded_images
                wcs = loaded_wcs
                filters = self.payload.get("filters") or list(images.keys())

            if not images or not filters:
                return hv.Image(np.zeros((2, 2)), bounds=(0, 0, 2, 2)).opts(cmap="grey")

            bands = [str(band) for band in filters if str(band) in images]
            if not bands:
                bands = list(images.keys())

            reference = "VIS" if "VIS" in bands else bands[0]
            color_bands = [band for band in ["NIR_H", "NIR_Y", "VIS"] if band in bands]

            container = ImageVisualizationClass(
                images=[images[band] for band in bands],
                wcs=[wcs.get(band) for band in bands],
                band_names=bands,
                color_image=len(color_bands) == 3,
                color_bands=color_bands if len(color_bands) == 3 else None,
                color_name="Color",
                target_wcs=wcs.get(reference),
            )

            band = self.payload.get("display_band") or ("Color" if container.has_color else bands[0])
            if band not in container.available_bands:
                band = "Color" if container.has_color else bands[0]

            data = container.get_plot_data(band)
            height, width = data.shape[:2]
            bounds = (0, 0, width, height)

            if data.ndim == 3:
                return hv.RGB(data[::-1, ...], bounds=bounds).opts(
                    xaxis=None,
                    yaxis=None,
                    toolbar=None,
                )

            return hv.Image(data[::-1, ...], bounds=bounds).opts(
                cmap="grey",
                xaxis=None,
                yaxis=None,
                toolbar=None,
            )
        except Exception:
            traceback.print_exc()
            return hv.Image(np.zeros((2, 2)), bounds=(0, 0, 2, 2)).opts(cmap="grey")

def create_euclid_cutout_panel(
    *,
    context: Any,
    data: Any = None,
    state: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Tuple[pn.viewable.Viewable, EuclidCutoutPanel]:
    controller = EuclidCutoutPanel(context=context, data=data, state=state, **kwargs)
    return controller.view(), controller

def create_euclid_cutout_artifact_viewer(
    *,
    context: Any = None,
    artifact: Any = None,
    payload: Any = None,
    **kwargs: Any,
) -> Tuple[pn.viewable.Viewable, EuclidCutoutArtifactViewer]:
    viewer = EuclidCutoutArtifactViewer(context=context, artifact=artifact, payload=payload, **kwargs)
    return viewer.view(), viewer