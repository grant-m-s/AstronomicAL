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

from .service import DEFAULT_EUCLID_FILTERS, DEFAULT_SAVE_DIR, EuclidCutoutRuntime


PLUGIN_ID = "astro.euclid_cutout"
RUNTIME_SERVICE_KEY = f"{PLUGIN_ID}.runtime"

SETTINGS_HEIGHT = 118


def _style_widget(
    widget: Any,
    *,
    width: int = 145,
    height: int = 40,
    margin: Tuple[int, int, int, int] = (0, 6, 2, 6),
) -> Any:
    """Apply the compact settings-row styling used by plugin panels."""

    try:
        widget.width = width
        widget.height = height
        widget.sizing_mode = "fixed"
        widget.margin = margin
    except Exception:
        pass
    return widget


def _settings_box(*controls: Any) -> pn.FlexBox:
    """Compact scrollable settings row matching the visualisation panels."""

    return pn.FlexBox(
        *controls,
        sizing_mode="stretch_width",
        height_policy="fit",
        margin=(0, 0, 0, 0),
        styles={
            "overflow": "visible",
            "align-content": "flex-start",
            "align-items": "flex-start",
            "gap": "2px 6px",
            "padding": "4px 6px 4px 6px",
            "border-top": "1px solid #ddd",
            "border-bottom": "1px solid #eee",
            "background": "#fafafa",
            "box-sizing": "border-box",
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

def _small_label(text: str, *, width: int = 76) -> pn.pane.HTML:
    return pn.pane.HTML(
        f"<div style='font-size:11px;font-weight:600;color:#555;"
        f"padding-top:11px;white-space:nowrap'>{text}</div>",
        width=width,
        height=34,
        sizing_mode="fixed",
        margin=(0, 2, 0, 6),
    )


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
    """Plugin-native Euclid cutout panel.

    The panel owns UI state only. Runtime archive access is handled by the
    EuclidCutoutRuntime service, focused-row identity comes from SelectionManager,
    source data comes from DatasetManager, slow retrieval goes through JobManager,
    and successful cutouts are published as astro.cutout.euclid artifacts.
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
        self.euclid_object: Any = None
        self.overplotted_coordinates: List[Any] = []
        self.stored_spectrum_coordinates: Dict[str, Dict[str, Iterable[float]]] = {}

        self.image_width = 1
        self.image_height = 1
        self.bar_length_pixels = 1

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
            margin=(0, 8, 0, 8),
            styles={
                "font-size": "12px",
                "line-height": "1.25",
                "max-height": "42px",
                "overflow": "auto",
            },
        )

        self.target_status = pn.pane.HTML(
            "",
            sizing_mode="stretch_width",
            height=34,
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

        # Header controls.
        self.filter_input = pn.widgets.Select(
            name="Filter",
            options=["Color", *DEFAULT_EUCLID_FILTERS],
            value="Color",
            sizing_mode="stretch_width",
            height=44,
            margin=(0, 2, 0, 0),
        )

        self.radius_input = pn.widgets.FloatInput(
            name="Radius [arcsec]",
            value=5.0,
            start=0.1,
            step=0.5,
            width=118,
            height=44,
            sizing_mode="fixed",
            margin=(0, 2, 0, 0),
        )

        self.stretch_input = pn.widgets.Select(
            name="Stretch",
            options=["Linear", "Sqrt", "Log", "Asinh", "PowerLaw"],
            value="Linear",
            sizing_mode="stretch_width",
            height=44,
            margin=(0, 2, 0, 0),
        )

        self.load_button = pn.widgets.Button(
            name="Load",
            button_type="primary",
            width=74,
            height=34,
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

        # Request / auth controls.
        self.environment = _style_widget(
            pn.widgets.Select(
                name="Environment",
                options=["PDR", "IDR", "OTF", "REG"],
                value="PDR",
            ),
            width=120,
        )

        self.user_input = _style_widget(
            pn.widgets.TextInput(name="Euclid username"),
            width=170,
        )

        self.password_input = _style_widget(
            pn.widgets.PasswordInput(name="Euclid password"),
            width=170,
        )

        self.credentials_file_input = _style_widget(
            pn.widgets.TextInput(
                name="Credentials file",
                value="euclid_credentials.login",
            ),
            width=210,
        )

        self.login_column = pn.FlexBox(
            self.user_input,
            self.password_input,
            self.credentials_file_input,
            sizing_mode="stretch_width",
            height_policy="fit",
            visible=False,
            margin=(0, 0, 0, 0),
            styles={
                "gap": "2px 6px",
                "align-items": "flex-start",
            },
        )

        self.stretch_scale_input = _style_widget(
            pn.widgets.FloatInput(
                name="Stretch scale",
                value=1.0,
                step=0.1,
            ),
            width=120,
        )

        self.save_dir_input = _style_widget(
            pn.widgets.TextInput(
                name="FITS cache directory",
                value=DEFAULT_SAVE_DIR,
            ),
            width=210,
        )

        # Display toggles.
        self.show_source_coords = _style_widget(
            pn.widgets.Checkbox(name="Source marker", value=True),
            width=120,
            height=28,
            margin=(10, 8, 0, 6),
        )

        self.show_scale = _style_widget(
            pn.widgets.Checkbox(name="Scale bar", value=True),
            width=95,
            height=28,
            margin=(10, 8, 0, 6),
        )

        self.show_spectrum_coords = _style_widget(
            pn.widgets.Checkbox(name="Spectrum markers", value=True),
            width=145,
            height=28,
            margin=(10, 8, 0, 6),
        )

        self.auto_reload = _style_widget(
            pn.widgets.Checkbox(name="Auto reload", value=True),
            width=105,
            height=28,
            margin=(10, 8, 0, 6),
        )

        # Contour controls.
        self.contour_levels = _style_widget(
            pn.widgets.IntInput(
                name="Contour levels",
                value=0,
                start=0,
                end=20,
            ),
            width=125,
        )

        self.contour_base = _style_widget(
            pn.widgets.FloatInput(
                name="Contour base",
                value=2.0,
                start=1.01,
                step=0.5,
            ),
            width=125,
        )

        self.contour_exponent = _style_widget(
            pn.widgets.FloatInput(
                name="Contour exponent",
                value=1.0,
                start=0.1,
                step=0.1,
            ),
            width=145,
        )

        # Clip/gamma controls.
        self.clip_slider = _style_widget(
            pn.widgets.RangeSlider(
                name="Clip",
                start=0,
                end=1,
                step=0.004,
                value=(0, 1),
            ),
            width=185,
        )

        self.rgb_clip_r = _style_widget(
            pn.widgets.RangeSlider(
                name="Red clip",
                start=0,
                end=1,
                step=0.004,
                value=(0, 1),
                bar_color="red",
            ),
            width=185,
        )

        self.rgb_clip_g = _style_widget(
            pn.widgets.RangeSlider(
                name="Green clip",
                start=0,
                end=1,
                step=0.004,
                value=(0, 1),
                bar_color="green",
            ),
            width=185,
        )

        self.rgb_clip_b = _style_widget(
            pn.widgets.RangeSlider(
                name="Blue clip",
                start=0,
                end=1,
                step=0.004,
                value=(0, 1),
                bar_color="blue",
            ),
            width=185,
        )

        self.gamma_r = _style_widget(
            pn.widgets.FloatInput(name="Gamma R", value=1.0, start=0.05, step=0.1),
            width=95,
        )

        self.gamma_g = _style_widget(
            pn.widgets.FloatInput(name="Gamma G", value=1.0, start=0.05, step=0.1),
            width=95,
        )

        self.gamma_b = _style_widget(
            pn.widgets.FloatInput(name="Gamma B", value=1.0, start=0.05, step=0.1),
            width=95,
        )

        self.refresh_button = pn.widgets.Button(
            name="Refresh display",
            width=125,
            height=34,
            sizing_mode="fixed",
            margin=(6, 6, 0, 6),
        )

        self.clean_jobs_button = pn.widgets.Button(
            name="Clean async jobs",
            width=130,
            height=34,
            sizing_mode="fixed",
            margin=(6, 6, 0, 6),
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
        ]:
            widget.param.watch(lambda _event: self._refresh_display(), "value")

        for widget in [self.stretch_input, self.stretch_scale_input]:
            widget.param.watch(lambda _event: self._recompute_stretch(), "value")

    def _header(self) -> pn.GridBox:
        return pn.GridBox(
            self.filter_input,
            self.radius_input,
            self.stretch_input,
            self.load_button,
            self.settings_button,
            ncols=5,
            sizing_mode="stretch_width",
            height=48,
            margin=(0, 6, 0, 6),
            styles={
                "display": "grid",
                "grid-template-columns": "minmax(100px, 1fr) 118px minmax(100px, 1fr) 78px 34px",
                "gap": "4px",
                "align-items": "start",
                "box-sizing": "border-box",
            },
        )

    def _settings_controls(self) -> pn.FlexBox:
        return _settings_box(
            _small_label("Request", width=58),
            self.environment,
            self.stretch_scale_input,
            self.save_dir_input,
            self.login_column,
            self.refresh_button,
            self.clean_jobs_button,
            _small_label("Display", width=58),
            self.show_source_coords,
            self.show_scale,
            self.show_spectrum_coords,
            self.auto_reload,
            self.contour_levels,
            self.contour_base,
            self.contour_exponent,
            _small_label("Clip", width=38),
            self.clip_slider,
            self.rgb_clip_r,
            self.rgb_clip_g,
            self.rgb_clip_b,
            self.gamma_r,
            self.gamma_g,
            self.gamma_b,
        )

    def _ensure_settings_built(self) -> None:
        if self._settings_built:
            return
        self.settings_pane[:] = [self._settings_controls()]
        self._settings_built = True

    def _apply_settings_visibility(self) -> None:
        self._ensure_settings_built()
        self.settings_pane.visible = self.settings_visible
        try:
            self.settings_button.button_type = "primary" if self.settings_visible else "default"
        except Exception:
            pass

    def _toggle_settings(self, _event: Any = None) -> None:
        self.settings_visible = not self.settings_visible
        self._apply_settings_visibility()

    def _build_layout(self) -> None:
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

        self._ensure_settings_built()
        self._apply_settings_visibility()

        self.layout = pn.Column(
            self._header(),
            self.settings_pane,
            self.target_status,
            self.status,
            self.figure,
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
        events = getattr(self.context, "events", None)
        if events is None:
            return

        self._subscribe("selection.focus.changed", self._selection_changed)
        self._subscribe("selection.focus.cleared", self._selection_cleared)
        self._subscribe("dataset.active.changed", self._dataset_changed)
        self._subscribe("dataset.mapping_updated", self._dataset_changed)
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

    def _selection_changed(self, topic: str, payload: Any) -> None:
        self._cancel_job()
        self._current_target = None
        self._cutout_result = None
        self.euclid_object = None
        self.stored_spectrum_coordinates.clear()
        self.overplotted_coordinates = []

        self.figure.object = self._empty_image()

        self._update_target_status()

        if self.auto_reload.value:
            self.load_cutout(reason=str(topic or "selection.focus.changed"))

    def _selection_cleared(self, topic: str, payload: Any) -> None:
        self._current_target = None
        self.stored_spectrum_coordinates.clear()
        self.overplotted_coordinates = []
        self.status.object = "No focused row selected."
        self.target_status.object = ""
        self.figure.object = self._empty_image()

    def _dataset_changed(self, topic: str, payload: Any) -> None:
        self._cancel_job()
        self._current_target = None
        self._cutout_result = None
        self.euclid_object = None
        self.stored_spectrum_coordinates.clear()
        self.overplotted_coordinates = []

        self.figure.object = self._empty_image()
        self._update_target_status()

        if self.auto_reload.value:
            self.load_cutout(reason=str(topic or "dataset.changed"))

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

        # Event-only fallback. The spectra plugin publishes the colour/point
        # metadata in both the artifact and the event so the cutout can still
        # update if an artifact viewer/store implementation changes.
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
            event_payload=payload,
        )

        if not normalised["ra"] or not normalised["dec"]:
            return

        storage_key = str(source)
        if artifact_id:
            storage_key = f"{source}:{artifact_id}"

        # Remove older coordinate sets for the same source so switching from one
        # DESI/Euclid spectrum result to another does not leave stale markers.
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
        if datasets is None:
            return None

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
                value = settings.get(key)
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
        """Return False when a completed job belongs to an older selection.

        Job cancellation is best-effort. A previously submitted archive request
        can still complete after the user has selected a different source. This
        guard prevents stale Euclid cutouts from overwriting the current panel.
        """

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
            id_column = self._guess_column(
                dataset_id,
                ["id", "ID", "source_id", "object_id", "row_id"],
            )

        if not ra_column or not dec_column:
            raise RuntimeError(
                "Euclid Cutout requires mapped coordinate columns `coords.ra` and `coords.dec`."
            )

        row_id = self._get_from_obj(focus, "row_id", "record_id", "id", "source_id")
        row_pos = self._get_from_obj(focus, "row_pos", "row_index", "position", "index")

        if row_pos is None:
            row_pos = (
                metadata.get("row_pos")
                or metadata.get("row_index")
                or metadata.get("position")
                or metadata.get("index")
            )

        row = None

        # Prefer an explicit row payload when another platform panel provides it.
        # This is only used when it matches the focused row id.
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

        # Main path: resolve every focus change through the active platform
        # dataset/source. This avoids reusing the dataframe originally passed
        # to the panel factory.
        if row is None:
            row = self._fetch_row(
                dataset_id,
                id_column=id_column,
                row_id=row_id,
                row_pos=row_pos,
                required_columns=[ra_column, dec_column],
            )

        # Legacy fallback only after the platform dataset lookup fails. It must
        # never silently return data.iloc[0] for a different focused row.
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
            raise RuntimeError(
                f"Could not read numeric RA/Dec from columns {ra_column!r}/{dec_column!r}."
            ) from exc

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
        """Fallback for old factory-supplied dataframe data.

        This must not return the original first row when the platform focus has
        moved to a different row. That was the cause of stale RA/Dec and stale
        Euclid cutouts after source selection changed.
        """

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

        # Compatibility fallback for old in-memory dataframe paths.
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
            # Current DatasetSource API.
            method = getattr(source, "get_row_by_id", None)
            if callable(method):
                try:
                    row = self._normalise_rows(
                        method(
                            row_id,
                            id_column=id_column,
                            columns=columns or None,
                        )
                    )
                    if row is not None:
                        return row
                except Exception:
                    pass

            method = getattr(source, "get_rows_by_ids", None)
            if callable(method):
                try:
                    row = self._normalise_rows(
                        method(
                            [row_id],
                            id_column=id_column,
                            columns=columns or None,
                        )
                    )
                    if row is not None:
                        return row
                except Exception:
                    pass

            # Older/experimental spellings kept as compatibility fallbacks.
            attempts = [
                ("get_rows_by_id", {"row_ids": [row_id], "id_column": id_column, "columns": columns}),
                ("read_rows_by_id", {"row_ids": [row_id], "id_column": id_column, "columns": columns}),
                ("rows_by_id", {"row_ids": [row_id], "id_column": id_column, "columns": columns}),
                ("get_rows", {"row_ids": [row_id], "id_column": id_column, "columns": columns}),
            ]

            for name, kwargs in attempts:
                method = getattr(source, name, None)
                if callable(method):
                    for call_kwargs in (
                        kwargs,
                        {k: v for k, v in kwargs.items() if k != "columns"},
                    ):
                        try:
                            row = self._normalise_rows(method(**call_kwargs))
                            if row is not None:
                                return row
                        except TypeError:
                            continue
                        except Exception:
                            continue

        if row_pos is not None:
            method = getattr(source, "get_row_by_position", None)
            if callable(method):
                try:
                    row = self._normalise_rows(
                        method(
                            int(row_pos),
                            columns=columns or None,
                        )
                    )
                    if row is not None:
                        return row
                except Exception:
                    pass

            attempts = [
                ("get_rows", {"row_positions": [row_pos], "columns": columns}),
                ("read_rows", {"row_positions": [row_pos], "columns": columns}),
                ("take", {"indices": [row_pos], "columns": columns}),
            ]

            for name, kwargs in attempts:
                method = getattr(source, name, None)
                if callable(method):
                    for call_kwargs in (
                        kwargs,
                        {k: v for k, v in kwargs.items() if k != "columns"},
                    ):
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
            f"<b>Dataset:</b> <code>{target.dataset_id}</code> &nbsp; "
            f"<b>Record:</b> <code>{target.row_id or 'unmapped'}</code> &nbsp; "
            f"<b>RA/Dec:</b> <code>{target.ra:.6f}, {target.dec:.6f}</code>"
            "</div>"
        )

    def _update_target_status(self) -> None:
        try:
            target = self._resolve_target()
            self._current_target = target
            self.target_status.object = self._target_html(target)
        except Exception as exc:
            self.target_status.object = (
                f"<div style='color:#8a5a00'>⚠️ {exc}</div>"
            )

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

        self._cancel_job()

        self._cutout_result = None
        self.euclid_object = None
        self.figure.object = self._empty_image()

        self.status.object = "Loading Euclid cutout…"
        self.target_status.object = self._target_html(target)

        self._publish(
            "astro.cutout.running",
            {
                "source": "Euclid",
                "running": True,
                "panel_id": self.panel_id,
                "dataset_id": target.dataset_id,
                "selected_id": target.row_id,
                "reason": reason,
            },
        )

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

    def _cancel_job(self) -> None:
        handle = self._job_handle
        self._job_handle = None

        if handle is not None:
            try:
                handle.cancel()
            except Exception:
                pass

    def _on_cutout_loaded(self, result: Any, *, target: _ResolvedTarget, reason: str) -> None:
        if self._disposed:
            return

        if not self._target_matches_current_focus(target):
            return

        self._job_handle = None
        self._cutout_result = result
        self.euclid_object = result.cutout
        self.status.object = ""

        self._refresh_display()
        artifact_id = self._put_cutout_artifact(result, target=target)

        self._publish(
            "astro.cutout.updated",
            {
                "source": "Euclid",
                "artifact_id": artifact_id,
                "panel_id": self.panel_id,
                "dataset_id": target.dataset_id,
                "selected_id": target.row_id,
                "ra": target.ra,
                "dec": target.dec,
                "reason": reason,
            },
        )

        self._publish(
            "astro.cutout.running",
            {
                "source": "Euclid",
                "running": False,
                "panel_id": self.panel_id,
                "dataset_id": target.dataset_id,
                "selected_id": target.row_id,
                "reason": reason,
            },
        )

    def _on_cutout_error(self, exc: BaseException, *, target: _ResolvedTarget, reason: str) -> None:
        if self._disposed:
            return

        if not self._target_matches_current_focus(target):
            return

        self._job_handle = None
        self.status.object = f"**Euclid cutout unavailable:** {exc}"
        self.figure.object = self._empty_image()

        self._publish(
            "astro.cutout.running",
            {
                "source": "Euclid",
                "running": False,
                "panel_id": self.panel_id,
                "dataset_id": target.dataset_id,
                "selected_id": target.row_id,
                "reason": reason,
                "error": str(exc),
            },
        )

    def _put_cutout_artifact(self, result: Any, *, target: _ResolvedTarget) -> Optional[str]:
        artifacts = getattr(self.context, "artifacts", None)
        if artifacts is None:
            return None

        payload = result.artifact_payload()
        params = {
            "source": "Euclid",
            "selected_id": target.row_id,
            "ra": target.ra,
            "dec": target.dec,
            "radius_arcsec": result.radius_arcsec,
            "filter_name": result.filter_name,
            "stretch": result.stretch,
            "stretch_scale": result.stretch_scale,
            "environment": result.environment,
        }

        try:
            return artifacts.put(
                "astro.cutout.euclid",
                payload,
                dataset_id=target.dataset_id,
                row_ids=[target.row_id] if target.row_id is not None else None,
                params=params,
                persist=False,
            )
        except TypeError:
            return artifacts.put(
                type="astro.cutout.euclid",
                payload=payload,
                dataset_id=target.dataset_id,
                row_ids=[target.row_id] if target.row_id is not None else None,
                params=params,
                persist=False,
            )
        except Exception:
            traceback.print_exc()
            return None

    def _stretch_scale_value(self) -> Optional[float]:
        if self.stretch_input.value == "Linear":
            return None

        try:
            value = float(self.stretch_scale_input.value)
            return value if np.isfinite(value) else None
        except Exception:
            return None

    def _recompute_stretch(self) -> None:
        if self.euclid_object is None:
            return

        try:
            self.euclid_object.get_plot_data(
                stretch=self.stretch_input.value,
                stretch_scale=self._stretch_scale_value(),
            )
            self._refresh_display()
        except Exception as exc:
            self.status.object = f"**Could not update Euclid stretch:** {exc}"

    def _refresh_display(self) -> None:
        if self.euclid_object is None:
            return

        try:
            data = self._scaled_image()
            self._build_hv_figure(data)
            self._update_figure_object()
        except Exception as exc:
            self.status.object = f"**Could not display Euclid cutout:** {exc}"
            self.figure.object = self._empty_image()

    def _scaled_image(self) -> Any:
        filter_name = self.filter_input.value

        if filter_name == "Color":
            lows = [
                self.rgb_clip_r.value[0],
                self.rgb_clip_g.value[0],
                self.rgb_clip_b.value[0],
            ]
            highs = [
                self.rgb_clip_r.value[1],
                self.rgb_clip_g.value[1],
                self.rgb_clip_b.value[1],
            ]
            gammas = [
                self.gamma_r.value,
                self.gamma_g.value,
                self.gamma_b.value,
            ]

            return self.euclid_object.transform_image_range(
                "Color",
                lows,
                highs,
                gamma=gammas,
                scale_method="MinMax",
                scale_by_channel=True,
            )

        low, high = self.clip_slider.value
        return self.euclid_object.transform_image_range(
            filter_name,
            low,
            high,
            gamma=1,
            scale_method="MinMax",
        )

    def _build_hv_figure(self, data: Any) -> None:
        filter_name = self.filter_input.value
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
            try:
                base = max(float(self.contour_base.value), 1.01)
                exponent = max(float(self.contour_exponent.value), 0.01)
                temp_data = self.euclid_object.data[filter_name]

                if len(temp_data.shape) == 3:
                    temp_img = hv.RGB(temp_data[::-1, ...], bounds=bounds)
                else:
                    temp_img = hv.Image(temp_data[::-1, ...], bounds=bounds)

                levels = np.nanmax(temp_data) / (
                    base ** (np.arange(1, int(self.contour_levels.value) + 1) * exponent)
                )
                contours = hv.operation.contours(temp_img, levels=levels).opts(
                    cmap=["red"],
                    colorbar=False,
                    active_tools=[],
                    show_legend=False,
                )
                elements.append(contours)
            except Exception:
                pass

        if self.show_scale.value:
            elements.extend(self._scale_bar_elements())

        if self.show_source_coords.value and self._current_target is not None:
            point = self._source_coordinate_element(
                self._current_target.ra,
                self._current_target.dec,
            )
            if point is not None:
                elements.append(point)

        self.overplotted_coordinates = []
        if self.show_spectrum_coords.value:
            self.overplotted_coordinates = self._spectrum_coordinate_elements()
            elements.extend(self.overplotted_coordinates)

        self.euclid_fig = elements

    def _scale_bar_elements(self) -> List[Any]:
        filter_name = self.filter_input.value

        try:
            arcsec_per_pix = self.euclid_object.arcsec_per_pix[filter_name]
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
        try:
            x, y = self.euclid_object.world_2_pix(
                ra=ra,
                dec=dec,
                filtro=self.filter_input.value,
                zipped=False,
            )
            x_value = float(np.asarray(x).ravel()[0])
            y_value = float(np.asarray(y).ravel()[0])

            if 0 <= x_value < self.image_width and 0 <= y_value < self.image_height:
                label = f"{ra:.3f}, {dec:.3f}"
                return hv.Points([(x_value, y_value)], label=label).opts(
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
        event_payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, List[Any]]:
        """Normalise spectrum coordinate artifacts for cutout overplotting.

        Supports both the new structured form:

            {"points": [{"ra": ..., "dec": ..., "color": ...}, ...]}

        and the older array form:

            {"ra": [...], "dec": [...], "colors": [...]}
        """

        event_payload = event_payload or {}

        ra_values: List[Any] = []
        dec_values: List[Any] = []
        colours: List[str] = []
        labels: List[str] = []
        indices: List[int] = []

        if isinstance(coords, dict):
            points = coords.get("points") or event_payload.get("points") or []

            if isinstance(points, list) and points:
                for fallback_index, point in enumerate(points):
                    if not isinstance(point, dict):
                        continue

                    ra = point.get("ra")
                    dec = point.get("dec")

                    if ra is None or dec is None:
                        continue

                    index = point.get("index", fallback_index)
                    colour = (
                        point.get("color")
                        or point.get("colour")
                        or _fallback_spectrum_colour(fallback_index)
                    )
                    label = (
                        point.get("label")
                        or f"{fallback_source} spectrum {fallback_index + 1}"
                    )

                    ra_values.append(ra)
                    dec_values.append(dec)
                    colours.append(str(colour))
                    labels.append(str(label))
                    indices.append(int(index) if str(index).isdigit() else fallback_index)

                return {
                    "ra": ra_values,
                    "dec": dec_values,
                    "colors": colours,
                    "colours": colours,
                    "labels": labels,
                    "indices": indices,
                }

            raw_ra = coords.get("ra", [])
            raw_dec = coords.get("dec", [])
            raw_colours = (
                coords.get("colors")
                or coords.get("colours")
                or coords.get("color")
                or coords.get("colour")
                or event_payload.get("colors")
                or event_payload.get("colours")
                or []
            )
            raw_labels = coords.get("labels") or event_payload.get("labels") or []
            raw_indices = coords.get("indices") or []

        else:
            raw_ra = []
            raw_dec = []
            raw_colours = []
            raw_labels = []
            raw_indices = []

        def _as_list(value: Any) -> List[Any]:
            if value is None:
                return []
            if isinstance(value, list):
                return value
            if isinstance(value, tuple):
                return list(value)
            try:
                arr = np.asarray(value)
                if arr.ndim == 0:
                    return [arr.item()]
                return arr.tolist()
            except Exception:
                return [value]

        ra_values = _as_list(raw_ra)
        dec_values = _as_list(raw_dec)
        raw_colours = _as_list(raw_colours)
        raw_labels = _as_list(raw_labels)
        raw_indices = _as_list(raw_indices)

        n = min(len(ra_values), len(dec_values))

        colours = []
        labels = []
        indices = []

        for idx in range(n):
            colour = raw_colours[idx] if idx < len(raw_colours) and raw_colours[idx] else None
            label = raw_labels[idx] if idx < len(raw_labels) and raw_labels[idx] else None
            index = raw_indices[idx] if idx < len(raw_indices) else idx

            colours.append(str(colour or _fallback_spectrum_colour(idx)))
            labels.append(str(label or f"{fallback_source} spectrum {idx + 1}"))
            indices.append(int(index) if str(index).isdigit() else idx)

        return {
            "ra": ra_values[:n],
            "dec": dec_values[:n],
            "colors": colours,
            "colours": colours,
            "labels": labels,
            "indices": indices,
        }

    def _spectrum_coordinate_elements(self) -> List[Any]:
        elements: List[Any] = []

        if not self.stored_spectrum_coordinates:
            return elements

        for source, coords in self.stored_spectrum_coordinates.items():
            try:
                xs, ys = self.euclid_object.world_2_pix(
                    ra=coords["ra"],
                    dec=coords["dec"],
                    filtro=self.filter_input.value,
                    zipped=False,
                )

                xs = np.asarray(xs).ravel()
                ys = np.asarray(ys).ravel()

                colours = list(coords.get("colors") or coords.get("colours") or [])
                labels = list(coords.get("labels") or [])
                indices = list(coords.get("indices") or [])

                source_name = str(source).split(":", 1)[0]
                marker = "+" if source_name.upper() == "DESI" else "*"

                for idx, (x, y) in enumerate(zip(xs, ys)):
                    if not (0 <= x < self.image_width and 0 <= y < self.image_height):
                        continue

                    colour = colours[idx] if idx < len(colours) and colours[idx] else _fallback_spectrum_colour(idx)
                    label = labels[idx] if idx < len(labels) and labels[idx] else f"{source_name} spectrum {idx + 1}"
                    spectrum_index = indices[idx] if idx < len(indices) else idx

                    elements.append(
                        hv.Points(
                            [(float(x), float(y))],
                            label=f"{label}",
                        ).opts(
                            marker=marker,
                            size=18,
                            color=colour,
                            line_color=colour,
                            tools=["hover"],
                            legend_position="top_left",
                        )
                    )

                    # Add a tiny number label next to the marker. This helps when
                    # two spectra have similar colours or overlap closely.
                    elements.append(
                        hv.Text(
                            float(x) + 4,
                            float(y) + 4,
                            str(int(spectrum_index) + 1),
                        ).opts(
                            text_color=colour,
                            text_font_size="9pt",
                            text_align="left",
                            text_baseline="bottom",
                        )
                    )

            except Exception:
                continue

        return elements

    def _update_figure_object(self) -> None:
        overlay = hv.Overlay(self.euclid_fig).opts(
            responsive=True,
            aspect="equal",
            toolbar=None,
            shared_axes=False,
            axiswise=True,
        )
        self.figure.object = overlay
        self.status.object = ""

    @staticmethod
    def _empty_image() -> Any:
        return hv.Image(np.ones((10, 10))).opts(
            active_tools=[],
            clim=(0, 1),
            toolbar=None,
            padding=0,
            border=0,
            framewise=True,
            xaxis=None,
            yaxis=None,
            cmap="grey",
        )

    def _environment_changed(self, event: Any) -> None:
        self.login_column.visible = event.new in {"IDR", "OTF", "REG"}

    def _clean_async_jobs(self) -> None:
        try:
            self._runtime().clean_async_jobs()
            self.status.object = "Euclid async jobs cleaned where available."
        except Exception as exc:
            self.status.object = f"Could not clean Euclid async jobs: {exc}"


# ----------------------------------------------------------------------
# Plugin factories
# ----------------------------------------------------------------------


def create_euclid_cutout_panel(
    context: Any,
    data: Any = None,
    state: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Tuple[pn.viewable.Viewable, EuclidCutoutPanel]:
    panel = EuclidCutoutPanel(context=context, data=data, state=state, **kwargs)
    return panel.view(), panel


def create_euclid_cutout_artifact_viewer(
    context: Any,
    artifact_id: str,
    **_: Any,
) -> Tuple[pn.viewable.Viewable, Any]:
    artifacts = getattr(context, "artifacts", None)
    if artifacts is None:
        view = pn.pane.Markdown("Artifact store is not available.")
        return view, None

    try:
        payload = artifacts.get(artifact_id)
    except Exception as exc:
        view = pn.pane.Markdown(f"Could not load Euclid cutout artifact `{artifact_id}`: {exc}")
        return view, None

    image = payload.get("image") if isinstance(payload, dict) else None
    if image is None:
        view = pn.pane.Markdown(f"Artifact `{artifact_id}` does not contain displayable image data.")
        return view, None

    try:
        bounds = (0, 0, image.shape[1], image.shape[0])

        if len(image.shape) == 3:
            hv_image = hv.RGB(image[::-1, ...], bounds=bounds)
        else:
            hv_image = hv.Image(image[::-1, ...], bounds=bounds).opts(cmap="grey")

        pane = pn.pane.HoloViews(
            hv_image.opts(
                active_tools=[],
                toolbar=None,
                xaxis=None,
                yaxis=None,
                responsive=True,
            ),
            sizing_mode="stretch_both",
            min_height=420,
        )

        meta = pn.pane.Markdown(
            f"**Euclid cutout artifact:** `{artifact_id}`  \n"
            f"**RA/Dec:** `{payload.get('ra')}`, `{payload.get('dec')}`  \n"
            f"**Filter:** `{payload.get('filter_name')}`  \n"
            f"**Radius:** `{payload.get('radius_arcsec')}` arcsec",
            sizing_mode="stretch_width",
        )

        return pn.Column(meta, pane, sizing_mode="stretch_both"), None
    except Exception as exc:
        return pn.pane.Markdown(f"Could not render Euclid cutout artifact `{artifact_id}`: {exc}"), None