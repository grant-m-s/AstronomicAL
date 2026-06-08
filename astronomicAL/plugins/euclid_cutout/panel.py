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
from .service import DEFAULT_EUCLID_FILTERS, DEFAULT_SAVE_DIR, EuclidCutoutRuntime


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
                "opacity": "1",
                "visibility": "visible",
                "pointer-events": "auto",
                "transform": "translateY(0)",
            }
        )
    else:
        styles.update(
            {
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
        self._disposed = False
        self._initial_load_started = False
        self._settings_built = False
        self.settings_visible = False
        self._current_target: Optional[_ResolvedTarget] = None
        self._cutout_result: Any = None
        self.euclid_object: Any = None  # retrieval object only, not visualisation
        self.image_container: Optional[ImageVisualizationClass] = None
        self.overplotted_coordinates: List[Any] = []
        self.stored_spectrum_coordinates: Dict[str, Dict[str, Iterable[float]]] = {}
        self.image_width = 1
        self.image_height = 1
        self.bar_length_pixels = 1
        self.euclid_fig: List[Any] = []

        self._auto_load_generation = 0
        self._auto_load_scheduled = False
        self._pending_auto_load_reason: Optional[str] = None
        self._target_status_scheduled = False

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
        self._disposed = True
        self._cancel_job()
        events = getattr(self.context, "events", None)
        if events is not None:
            for sub in list(self._subscriptions):
                try:
                    events.unsubscribe(sub)
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
        }

    def restore_state(self, state: Dict[str, Any]) -> None:
        if not isinstance(state, dict):
            return
        self.settings_visible = bool(state.get("settings_visible", False))
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
        }
        for key, widget in mapping.items():
            if key in state:
                try:
                    widget.value = state[key]
                except Exception:
                    pass
        try:
            self._apply_settings_visibility()
        except Exception:
            pass

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
                name="FITS cache directory",
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

        for widget in [
            self.filter_input,
            self.clip_slider,
            self.rgb_clip_r,
            self.rgb_clip_g,
            self.rgb_clip_b,
            self.gamma_r,
            self.gamma_g,
            self.gamma_b,
            self.show_source_coords,
            self.show_scale,
            self.show_spectrum_coords,
            self.contour_levels,
            self.contour_base,
            self.contour_exponent,
            self.stretch_input,
            self.stretch_scale_input,    
        ]:
            widget.param.watch(lambda _event: self._refresh_display(), "value")

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

        self.settings_pane.visible = True
        self._settings_view.visible = True
        self.settings_pane.styles = _settings_overlay_styles(bool(self.settings_visible))

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
            visible=True,
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

        self.body = pn.Column(
            self.settings_overlay_host,
            self.target_status,
            self.status,
            self.figure,
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

    def _schedule_auto_load(self, *, reason: str, delay_ms: int = 175) -> None:
        """Debounce auto-loads so rapid focus changes only load the latest row."""
        self._auto_load_generation += 1
        generation = int(self._auto_load_generation)
        self._pending_auto_load_reason = str(reason or "auto")

        if getattr(self, "_auto_load_scheduled", False):
            return

        self._auto_load_scheduled = True

        def _run() -> None:
            self._auto_load_scheduled = False
            if getattr(self, "_disposed", False):
                return
            if generation != int(getattr(self, "_auto_load_generation", 0)):
                # A newer focus/dataset event superseded this one.
                if self._pending_auto_load_reason:
                    self._schedule_auto_load(
                        reason=self._pending_auto_load_reason,
                        delay_ms=delay_ms,
                    )
                return

            reason_to_use = self._pending_auto_load_reason or reason
            self._pending_auto_load_reason = None
            try:
                self.load_cutout(reason=reason_to_use)
            except Exception:
                traceback.print_exc()

        self._schedule_panel_callback(_run, delay_ms=delay_ms)

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
                "error_type": type(error).__name__,
            }
        )
        self._publish("plugin.error", payload)

    def _reset_loaded_cutout(self) -> None:
        self._cutout_result = None
        self.euclid_object = None
        self.image_container = None
        self.figure.object = self._empty_image()

    def _selection_changed(self, topic: str, payload: Any) -> None:
        # Keep EventBus subscriber work cheap.
        self._auto_load_generation += 1

        self._cancel_job(reason=str(topic or "selection.focus.changed"))
        self._current_target = None
        self.stored_spectrum_coordinates.clear()
        self.overplotted_coordinates = []

        self._reset_loaded_cutout()
        self.target_status.object = "New focused row queued…"

        if self.auto_reload.value:
            self._schedule_auto_load(
                reason=str(topic or "selection.focus.changed"),
                delay_ms=175,
            )
        else:
            self._schedule_target_status_refresh(delay_ms=75)

    def _selection_cleared(self, topic: str, payload: Any) -> None:
        self._auto_load_generation += 1

        self._cancel_job(reason=str(topic or "selection.focus.cleared"))
        self._current_target = None
        self.stored_spectrum_coordinates.clear()
        self.overplotted_coordinates = []

        self.status.object = "No focused row selected."
        self.target_status.object = ""
        self._reset_loaded_cutout()

    def _dataset_changed(self, topic: str, payload: Any) -> None:
        self._auto_load_generation += 1

        self._cancel_job(reason=str(topic or "dataset.changed"))
        self._current_target = None
        self.stored_spectrum_coordinates.clear()
        self.overplotted_coordinates = []

        self._reset_loaded_cutout()
        self.target_status.object = "Dataset changed; resolving target…"

        if self.auto_reload.value:
            self._schedule_auto_load(
                reason=str(topic or "dataset.changed"),
                delay_ms=225,
            )
        else:
            self._schedule_target_status_refresh(delay_ms=100)

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
        self._refresh_display()

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
            row_pos = (
                metadata.get("row_position")
                or metadata.get("row_pos")
                or metadata.get("row_index")
                or metadata.get("position")
                or metadata.get("index")
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
        services = getattr(self.context, "services", None)
        if services is not None:
            try:
                if services.has(RUNTIME_SERVICE_KEY):
                    return services.get(RUNTIME_SERVICE_KEY)
            except Exception:
                pass
        return EuclidCutoutRuntime(context=self.context)

    def _schedule_initial_load(self) -> None:
        if self._initial_load_started:
            return
        self._initial_load_started = True

        def _run() -> None:
            self._update_target_status()
            if self.auto_reload.value:
                self.load_cutout(reason="initial")

        try:
            doc = pn.state.curdoc
            if doc is not None:
                doc.add_next_tick_callback(_run)
            else:
                _run()
        except Exception:
            _run()

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
            self._current_target = target
        except Exception as exc:
            self.status.object = f"**Euclid cutout unavailable:** {exc}"
            self.figure.object = self._empty_image()
            return

        self._cancel_job(reason="superseded")
        self._reset_loaded_cutout()
        self.status.object = "Loading Euclid cutout…"
        self.target_status.object = self._target_html(target)

        self._publish_cutout_running(True, target=target, reason=reason)

        runtime = self._runtime()
        credentials = self.credentials_file_input.value.strip() or None
        user = self.user_input.value.strip() or None
        password = self.password_input.value or None
        environment = self.environment.value or "PDR"

        def _worker(cancel_token: Any = None) -> Any:
            return runtime.fetch_cutout(
                ra=target.ra,
                dec=target.dec,
                radius_arcsec=float(self.radius_input.value),
                filter_name=self.filter_input.value,
                stretch=self.stretch_input.value,
                stretch_scale=self._stretch_scale_value(),
                environment=environment,
                user=user,
                password=password,
                credentials_filepath=credentials,
                save_dir=self.save_dir_input.value or DEFAULT_SAVE_DIR,
                cancel_token=cancel_token,
                verbose=True,
            )

        def _done(result: Any) -> None:
            self._on_cutout_loaded(result, target=target, reason=reason)

        def _error(exc: BaseException) -> None:
            self._on_cutout_error(exc, target=target, reason=reason)

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
                key=f"{self.panel_id}:{target.dataset_id}:{target.row_id}:{target.ra}:{target.dec}",
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

    def _cancel_job(
        self,
        *,
        target: Optional[_ResolvedTarget] = None,
        reason: str = "cancelled",
        publish: bool = True,
    ) -> None:
        handle = self._job_handle
        self._job_handle = None

        if handle is None:
            return

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

    def _on_cutout_loaded(self, result: Any, *, target: _ResolvedTarget, reason: str) -> None:
        if self._disposed:
            self._publish_cutout_running(False, target=target, reason="panel.disposed")
            return

        if not self._target_matches_current_focus(target):
            self._publish_cutout_running(False, target=target, reason="stale_result")
            return

        self._job_handle = None
        self._cutout_result = result
        self.euclid_object = result.cutout

        try:
            self._create_image_container(result)
        except Exception as exc:
            self.status.object = f"**Could not initialise Euclid image visualisation:** {exc}"
            self.figure.object = self._empty_image()
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

        self.status.object = ""
        self._refresh_display()

        artifact_id = self._put_cutout_artifact(result, target=target)
        self._publish_cutout_artifact_created(artifact_id=artifact_id, target=target)

        updated_payload = self._event_identity(target=target, artifact_id=artifact_id)
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

        color_bands = None

        for candidate in preferred_color_band_sets:
            if all(band in bands for band in candidate):
                color_bands = candidate
                break

        has_color = color_bands is not None

        self.image_container = ImageVisualizationClass(
            images=images,
            wcs=wcs_list,
            band_names=bands,
            color_image=has_color,
            color_bands=color_bands if has_color else None,
            color_name="Color",
            target_wcs=target_wcs,
        )

        options = self.image_container.available_bands
        previous = self.filter_input.value
        self.filter_input.options = options
        if previous in options:
            self.filter_input.value = previous
        elif "Color" in options:
            self.filter_input.value = "Color"
        else:
            self.filter_input.value = options[0]

    def _on_cutout_error(self, exc: BaseException, *, target: _ResolvedTarget, reason: str) -> None:
        if self._disposed:
            self._publish_cutout_running(
                False,
                target=target,
                reason="panel.disposed",
                error=exc,
            )
            return

        if not self._target_matches_current_focus(target):
            self._publish_cutout_running(
                False,
                target=target,
                reason="stale_result",
                error=exc,
            )
            return

        self._job_handle = None
        self.status.object = f"**Euclid cutout unavailable:** {exc}"
        self.figure.object = self._empty_image()

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


    def _refresh_display(self) -> None:
        if self.image_container is None:
            return
        try:
            filter_name = self.filter_input.value
            if filter_name == "Color":
                r_low, r_high = self._combine_global_and_channel_clip(self.rgb_clip_r.value)
                g_low, g_high = self._combine_global_and_channel_clip(self.rgb_clip_g.value)
                b_low, b_high = self._combine_global_and_channel_clip(self.rgb_clip_b.value)
                low_clip = [r_low, g_low, b_low]
                high_clip = [r_high, g_high, b_high]
                gamma_color = [self.gamma_r.value,
                               self.gamma_g.value,
                               self.gamma_b.value,]
            
            else:
                low_clip, high_clip = self.clip_slider.value
                gamma_color = 1
            data = self.image_container.get_plot_data(
                                band=filter_name,
                                stretch=self.stretch_input.value,
                                stretch_scale= self._stretch_scale_value(),
                                stretch_interval="Asymmetric",
                                low_clip=low_clip,
                                high_clip=high_clip,
                                gamma_color=gamma_color,
                                scale_method="MinMax")

            self._build_hv_figure(data)
            self._update_figure_object()
        except Exception as exc:
            self.status.object = f"**Could not display Euclid cutout:** {exc}"
            self.figure.object = self._empty_image()


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

        self.euclid_fig = elements

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
            arcsec_per_pix = self.image_container.get_arcsec_per_pixel(self.filter_input.value, scalar=True)
        except Exception:
            return []
        self.bar_length_pixels = self.image_width * 0.2
        x0, y0 = 0.1 * self.image_width, 0.1 * self.image_height
        x1 = x0 + self.bar_length_pixels
        scale_arcsec = self.bar_length_pixels * arcsec_per_pix
        return [
            hv.Curve(([x0, x1], [y0, y0])).opts(color="red", line_width=3),
            hv.Text((x0 + x1) / 2, y0 + y0 / 2, f'{scale_arcsec:.1f}"').opts(
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
                label = f"{ra:.3f}, {dec:.3f}"
                return hv.Points([(x_value, y_value)], label=label).opts(color="blue", marker="+", size=30)
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

        ra_values = coords.get("ra") or coords.get("RA") or []
        dec_values = coords.get("dec") or coords.get("DEC") or []
        colors = coords.get("colors") or coords.get("colours") or []
        labels = coords.get("labels") or []
        points = coords.get("points") or []

        if points and (not ra_values or not dec_values):
            ra_values = []
            dec_values = []
            colors = [] if not colors else colors
            labels = [] if not labels else labels
            for point in points:
                if not isinstance(point, dict):
                    continue
                ra_values.append(point.get("ra") or point.get("RA"))
                dec_values.append(point.get("dec") or point.get("DEC"))
                if not colors:
                    colors.append(point.get("color") or point.get("colour"))
                if not labels:
                    labels.append(point.get("label"))

        if np.isscalar(ra_values):
            ra_values = [ra_values]
        if np.isscalar(dec_values):
            dec_values = [dec_values]

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
                elements.append(
                    hv.Points([(xv, yv)], label=str(label)).opts(
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
            self._runtime().clean_async_jobs()
            self.status.object = "Cleaned Euclid archive async jobs where possible."
        except Exception as exc:
            self.status.object = f"**Could not clean Euclid archive async jobs:** {exc}"


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
    def _load_images_from_fits(fits_paths: Dict[str, str]) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
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
                with fits.open(str(path)) as hdul:
                    selected_hdu = None

                    for hdu in hdul:
                        data = getattr(hdu, "data", None)
                        if data is None:
                            continue

                        array = np.asarray(data)
                        if array.size and array.ndim >= 2:
                            selected_hdu = hdu
                            break

                    if selected_hdu is None:
                        continue

                    array = np.asarray(selected_hdu.data)
                    while array.ndim > 2:
                        array = array[0]

                    images[str(band)] = array

                    try:
                        wcs[str(band)] = WCS(selected_hdu.header)
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
